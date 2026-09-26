"""Core linear-production computations.

Conventions
-----------
* Industry-by-industry IO table in money units: one "unit" of good j = 1 currency unit of its
  gross output at market (basic) prices. Hence market prices are the vector of ones and the
  object of interest is the ratio z_j = (basis content per unit)_j / (market price)_j = content_j.
* Row vectors for prices/values, column vectors for quantities. A[i, j] = input of i per unit of j.
* All "bases" (labour and commodity bases) are computed by ONE function, `content`, applied to
  one augmented matrix. This is what makes the comparison symmetric (see `Economy.system`).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

EPS = 1e-12


# ----------------------------------------------------------------------------------------------
# Linear algebra helpers with explicit validity checks
# ----------------------------------------------------------------------------------------------
def spectral_radius(A: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(A)))) if A.size else 0.0


def leontief_inverse(A: np.ndarray, check: bool = True) -> np.ndarray:
    """(I - A)^{-1}. Checks: A >= 0, rho(A) < 1, result >= 0 and >= I on the diagonal."""
    n = A.shape[0]
    if check:
        if (A < -1e-9).any():
            raise ValueError(f"negative coefficients in A (min {A.min():.3g})")
        rho = spectral_radius(A)
        if rho >= 1:
            raise ValueError(f"spectral radius {rho:.4f} >= 1: system not productive")
    L = np.linalg.solve(np.eye(n) - A, np.eye(n))
    if check:
        if (L < -1e-8).any():
            raise ValueError("Leontief inverse has negative entries")
        if (np.diag(L) < 1 - 1e-8).any():
            raise ValueError("Leontief inverse diagonal < 1")
    return L


def content(M: np.ndarray, primary_row: np.ndarray, zero_row=None) -> np.ndarray:
    """Vertically integrated requirement of a primary input.

    M            : square input matrix of the (possibly augmented) system
    primary_row  : direct requirement of the primary input per unit of each good
    zero_row     : index (or list of indices) of goods of M that ARE the primary input (commodity
                   basis, or labour in the closed system). Their rows are removed from M, i.e. they
                   are treated as non-produced ("the k-row of A is turned into a primary resource").
    Returns lambda = primary_row (I - M_(k))^{-1}.
    """
    Mk = M.copy()
    if zero_row is not None:
        Mk[np.atleast_1d(zero_row), :] = 0.0
    return primary_row @ leontief_inverse(Mk)


def pf_left(M: np.ndarray) -> tuple[float, np.ndarray]:
    """Perron-Frobenius eigenvalue and (positive) left eigenvector of a non-negative matrix."""
    w, V = np.linalg.eig(M.T)
    i = int(np.argmax(w.real))
    lam = float(w[i].real)
    vec = np.abs(V[:, i].real)
    return lam, vec / vec.sum()


# ----------------------------------------------------------------------------------------------
# Economy (one country, one year)
# ----------------------------------------------------------------------------------------------
@dataclass
class Economy:
    labels: list[str]
    A: np.ndarray            # domestic intermediate coefficients (n x n)
    Am: np.ndarray           # imported intermediate coefficients (n x n), all origins
    x: np.ndarray            # gross output (money)
    hours: np.ndarray        # hours worked (all persons), per industry
    wages: np.ndarray        # compensation of employees D1
    cfc: np.ndarray          # consumption of fixed capital per industry (money)
    gfcf_dom: np.ndarray     # domestic GFCF by supplying industry (money)
    gfcf_imp: np.ndarray     # imported GFCF by supplying industry (money)
    hh_dom: np.ndarray       # household final consumption of domestic products (money)
    hh_imp: np.ndarray       # household final consumption of imported products (money)
    va: np.ndarray           # value added (all components incl. net taxes) per industry
    labour_income: np.ndarray | None = None  # D1 + imputed labour income of self-employed
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.labels)

    # --- capital -------------------------------------------------------------------------------
    def dep_coeffs(self) -> tuple[np.ndarray, np.ndarray]:
        """D = b d': depreciation per unit of output, allocated to supplying industries by the
        composition of the economy's GFCF (domestic part -> D, imported part -> Dm).
        Assumption A-CAP-1: every industry's capital stock has the economy-wide GFCF composition."""
        d = np.divide(self.cfc, self.x, out=np.zeros(self.n), where=self.x > 0)
        tot = self.gfcf_dom.sum() + self.gfcf_imp.sum()
        b_dom = self.gfcf_dom / tot
        b_imp = self.gfcf_imp / tot
        return np.outer(b_dom, d), np.outer(b_imp, d)

    def l(self, weights: np.ndarray | None = None) -> np.ndarray:
        h = self.hours if weights is None else self.hours * weights
        return np.divide(h, self.x, out=np.zeros(self.n), where=self.x > 0)

    def system(self, capital: bool, closed: bool, imports: str, l: np.ndarray):
        """Return (M, Mm, lab): input matrix of the chosen system, its imported-input matrix, and the
        indices of the labour goods (None in the open system).

        capital : add depreciation coefficients (A+ = A + D)
        closed  : append n labour goods ("hours worked in industry j"). Good j requires h_j/x_j hours
                  of labour j; one hour of labour j requires the household consumption bundle worth
                  its actual hourly labour income w_j (composition of household final consumption,
                  domestic part in M, imported part in Mm).  A-CLOSE: labour is reproduced at the
                  actual (industry-specific) wage; closed="uniform" (A-CLOSE-U) uses one bundle per
                  hour worth the average hourly labour income (scaled down only if unproductive).
                  In the closed system labour and every commodity are treated identically: the
                  basis' rows are made primary and everything else, labour included, is produced.
        imports : 'price'       imported inputs are a separate, non-produced input valued at price 1
                  'competitive' imported inputs are treated as if produced domestically (A + Am)
        """
        M, Mm = self.A.copy(), self.Am.copy()
        if capital:
            D, Dm = self.dep_coeffs()
            M, Mm = M + D, Mm + Dm
        lab = None
        if closed:
            n = self.n
            li = self.labour_income if self.labour_income is not None else self.wages
            hrs = self.hours
            cons = self.hh_dom.sum() + self.hh_imp.sum()
            b_dom, b_imp = self.hh_dom / cons, self.hh_imp / cons
            hcoef = np.divide(hrs, self.x, out=np.zeros(n), where=self.x > 0)  # hours per unit output
            if closed == "uniform":
                # A-CLOSE-U: one bundle per hour for all labour (no wage differentials), equal to the
                # average hourly labour income scaled by s <= 1, s the largest value keeping the closed
                # system productive (rho <= 0.995).
                w = np.full(n, self.uniform_wage_scale(M, Mm if imports == "competitive" else None)
                            * li.sum() / hrs.sum())
            else:
                w = np.divide(li, hrs, out=np.zeros(n), where=hrs > 0)      # money per hour, by industry
            M2 = np.zeros((2 * n, 2 * n))
            M2[:n, :n] = M
            M2[:n, n:] = np.outer(b_dom, w)
            M2[n:, :n] = np.diag(hcoef)
            Mm2 = np.zeros((2 * n, 2 * n))
            Mm2[:n, :n] = Mm
            Mm2[:n, n:] = np.outer(b_imp, w)
            M, Mm, lab = M2, Mm2, np.arange(n, 2 * n)
        if imports == "competitive":
            M, Mm = M + Mm, np.zeros_like(Mm)
        return M, Mm, lab

    def uniform_wage_scale(self, M: np.ndarray, Mm: np.ndarray | None, target: float = 0.995) -> float:
        key = (M.tobytes().__hash__(), None if Mm is None else Mm.tobytes().__hash__(), target)
        cache = self.meta.setdefault("_uws_cache", {})
        if key not in cache:
            cache[key] = self._uniform_wage_scale(M, Mm, target)
        return cache[key]

    def _uniform_wage_scale(self, M: np.ndarray, Mm: np.ndarray | None, target: float = 0.995) -> float:
        n = self.n
        li = self.labour_income if self.labour_income is not None else self.wages
        cons = self.hh_dom.sum() + self.hh_imp.sum()
        b = self.hh_dom / cons + (self.hh_imp / cons if Mm is not None else 0)
        Mt = M + (Mm if Mm is not None else 0)
        h = np.divide(self.hours, self.x, out=np.zeros(n), where=self.x > 0)
        wbar = li.sum() / self.hours.sum()
        f = lambda s: spectral_radius(Mt + s * wbar * np.outer(b, h))
        if f(1.0) <= target:
            return 1.0
        lo, hi = 0.0, 1.0
        for _ in range(40):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) <= target else (lo, mid)
        return lo

    # --- bases ---------------------------------------------------------------------------------
    def basis_ratios(self, basis: str | int, capital: bool = False, closed: bool = False,
                     imports: str = "price", l: np.ndarray | None = None) -> np.ndarray:
        """Value/price ratio z_j for the basis ('L' for labour, or an industry index k).

        The raw vertically integrated content lambda (hours or money-units of k per unit of j) is
        converted to a money-comparable magnitude by a single scalar e (the basis' "MELT"), then
        vertically integrated imported inputs (valued at their price) are added:
            z_j = e * lambda_j + mu_j ,  e chosen so that sum_j z_j x_j = sum_j x_j.
        With imports='competitive' mu = 0 and z = e * lambda.
        """
        l = self.l() if l is None else l
        n = self.n
        M, Mm, lab = self.system(capital, closed, imports, l)
        if basis == "L":
            if closed:
                # primary input = (weighted) hours; labour goods j carry weight l_j / (h_j/x_j)
                h = np.divide(self.hours, self.x, out=np.zeros(n), where=self.x > 0)
                wgt = np.divide(l, h, out=np.zeros(n), where=h > 0)
                row = np.zeros(2 * n)
                row[:n] = wgt * h                 # = l : direct (weighted) hours per unit of good j
                lam = content(M, row, zero_row=lab)
            else:
                lam = content(M, l)
            k = lab
        else:
            k = int(basis)
            lam = content(M, M[k, :], zero_row=k)
        # vertically integrated imported inputs in the same system (basis rows made primary)
        Mk = M.copy()
        if k is not None:
            Mk[np.atleast_1d(k), :] = 0.0
        mu = Mm.sum(0) @ leontief_inverse(Mk) if Mm.any() else np.zeros(M.shape[0])
        lam, mu = lam[:n], mu[:n]
        x = self.x
        e = (x.sum() - mu @ x) / (lam @ x)
        return e * lam + mu

    # --- prices of production ------------------------------------------------------------------
    def actual_profit_rate(self, capital: bool = True) -> float:
        """r = total (value added - labour income - CFC) / total non-labour inputs used (flow)."""
        M, Mm, _ = self.system(capital, False, "price", self.l())
        li = self.labour_income if self.labour_income is not None else self.wages
        inputs = (M.sum(0) + Mm.sum(0)) @ self.x
        cfc = self.cfc.sum() if capital else 0.0
        profits = self.va.sum() - li.sum() - cfc
        return float(profits / inputs)

    def prices_of_production(self, r: float, capital: bool = True, imports: str = "price",
                             wage_mode: str = "uniform", l: np.ndarray | None = None) -> np.ndarray:
        """p = (1+r)(p M + e Mm) + w l, normalised to sum p x = sum x.

        wage_mode 'uniform' : one wage per hour (homogeneous labour)
                  'actual'  : each industry pays its observed labour income per unit of output
                               (Sraffian cost-of-production with given wage differentials)
        Imports enter at their (exogenous) price 1. Solved as p = a + w b, w from normalisation.
        """
        l = self.l() if l is None else l
        M, Mm, _ = self.system(capital, False, imports, l)
        n = self.n
        Lr = leontief_inverse((1 + r) * M)
        a = (1 + r) * Mm.sum(0) @ Lr
        li = self.labour_income if self.labour_income is not None else self.wages
        if wage_mode == "uniform":
            b = l @ Lr
        else:
            b = np.divide(li, self.x, out=np.zeros(n), where=self.x > 0) @ Lr
        x = self.x
        w = (x.sum() - a @ x) / (b @ x)
        return a + w * b

    def max_profit_rate(self, capital: bool = True, imports: str = "price") -> float:
        M, _, _ = self.system(capital, False, imports, self.l())
        return 1.0 / spectral_radius(M) - 1.0

    def closed_eigen_prices(self, capital: bool = True) -> tuple[float, np.ndarray]:
        """Marx/Shaikh-type prices of production with wages advanced as the workers' bundle:
        p = (1+r) p (M + b w'), competitive imports, b = household consumption composition and
        w_j = actual labour income per unit of output of j. p is the left PF eigenvector and
        r = 1/lambda_PF - 1 is determined by the real wage. Normalised to sum p x = sum x."""
        l = self.l()
        M, _, lab = self.system(capital, True, "competitive", l)
        n = self.n
        Mc = M[:n, :n] + M[:n, lab] @ M[lab, :n]      # collapse labour goods into goods
        lam, p = pf_left(Mc)
        p = p * self.x.sum() / (p @ self.x)
        return 1 / lam - 1, p

    def k_exploitation_rates(self, capital: bool = False) -> dict:
        """Generalised commodity exploitation (Roemer; Bowles-Gintis): in the closed system with good
        k primary, one unit of k 'costs' lambda^k_k units of k; rate = 1/lambda^k_k - 1.
        For labour: hours worked / hours embodied in the wage bundles (Marx's rate of surplus value).
        Competitive imports. Returns {basis: rate}."""
        l = self.l()
        M, _, lab = self.system(capital, True, "competitive", l)
        n = self.n
        out = {}
        for k in range(n):
            if M[k, :].sum() <= 0:
                continue
            lam = content(M, M[k, :], zero_row=k)
            out[self.labels[k]] = 1.0 / lam[k] - 1.0
        # labour: hours embodied in the bundles bought by one hour (hours-weighted average)
        row = np.zeros(2 * n)
        row[:n] = l
        lamL = content(M, row, zero_row=lab)
        emb = lamL[lab]                       # hours embodied per hour of labour of type j
        out["L"] = float(self.hours.sum() / (emb @ self.hours) - 1.0)
        return out
