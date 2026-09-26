"""Iteration 3, stage P: placebos with the same unevenness across industries as labour.

Three placebo families (primary inputs of the open system, capital included, imports at price,
evaluated on the v2 industries, i.e. without T and U), 1000 each, fixed per country over time:
  (a) 'perm'  permutations of the direct hours coefficients l_j = h_j/x_j among evaluated industries
  (b) 'lnorm' l_j = exp(sigma * eps_j), sigma = SD ln l of labour in that year, eps ~ N(0,1) fixed
  (c) 'mix'   v2 cost-share mixtures of goods (same code and seeds as lts.v2.placebo_cost)
Share of placebos better than labour by MAWD and d; industry contributions to labour's MAWD;
re-evaluation without pre-registered industries (B, K64-66, L, J61, J62_63) and without the three
largest contributors (exploratory).  The helpers are reused for stage 1 (reduced labour vectors).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..core import leontief_inverse
from ..economy import build
from ..levels import mask_for
from ..v2.placebo_cost import N_MAIN, N_POOL, draw_placebos, metrics_mat

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "v3"
COUNTRIES = ("USA", "DEU", "MEX", "FRA", "ITA", "ESP", "NLD", "AUT", "POL", "CZE", "KOR", "JPN", "GBR")
YEARS = range(2010, 2023)
N_P = 1000
PREREG_EXCL = {"B", "K64", "K65", "K66", "L", "J61", "J62_63"}
METS = ("mawd", "d")


class Setup:
    """Open system of one country-year (capital included, imports at price)."""

    def __init__(self, e):
        M, Mm, _ = e.system(True, False, "price", e.l())
        self.e, self.M = e, M
        self.L = leontief_inverse(M)
        self.mu = Mm.sum(0) @ self.L
        self.x = e.x
        self.base_mask = mask_for(e.labels, set())

    def z(self, Lrows: np.ndarray) -> np.ndarray:
        """Ratios z for primary-input rows (P, n) (A-Z: z = e*lambda + mu, sum z x = sum x)."""
        lam = np.atleast_2d(Lrows) @ self.L
        s = (self.x.sum() - self.mu @ self.x) / (lam @ self.x)
        return s[:, None] * lam + self.mu[None]

    def mask(self, excl: set) -> np.ndarray:
        return mask_for(self.e.labels, set(excl))


class Draws:
    """Placebo draws fixed for a country: permutation indices, normal shocks, mixture weights."""

    def __init__(self, country: str, n: int, base_mask: np.ndarray, seed: int = 2026):
        rng = np.random.default_rng(seed + sum(map(ord, country)))
        idx = np.where(base_mask)[0]
        self.perm = np.tile(np.arange(n), (N_P, 1))
        for p in range(N_P):
            self.perm[p, idx] = idx[rng.permutation(len(idx))]
        self.eps = rng.normal(0, 1, (N_P, n))
        self.Om = None                                   # v2 mixtures, built in the first year
        self.country = country

    def rows(self, kind: str, l: np.ndarray, S: Setup | None = None) -> np.ndarray:
        if kind == "perm":
            return l[self.perm]
        if kind == "lnorm":
            pos = l > 0
            sd = np.log(l[pos]).std()
            return np.exp(self.eps * sd)
        if kind == "mix":
            if self.Om is None:                          # identical to lts.v2.placebo_cost
                rng = np.random.default_rng(77 + sum(map(ord, self.country)))
                lab_share = S.e.labour_income.sum() / S.x.sum()
                self.Om = draw_placebos(S.M, S.x, lab_share * S.x.sum(), rng, N_POOL)[:N_MAIN]
            return self.Om @ S.M
        raise ValueError(kind)


def shares(S: Setup, D: Draws, l: np.ndarray, masks: dict, kinds=("perm", "lnorm", "mix"),
           placebo_l: np.ndarray | None = None) -> list[dict]:
    """Share of placebos better than labour vector l for each kind and evaluation mask.
    placebo_l: vector the perm/lnorm placebos are built from (default: l itself)."""
    zl = S.z(l[None])
    pl = l if placebo_l is None else placebo_l
    out = []
    for kind in kinds:
        Z = S.z(D.rows(kind, pl, S))
        for mname, m in masks.items():
            mL = metrics_mat(zl[:, m], S.x[m])
            mP = metrics_mat(Z[:, m], S.x[m])
            for met in METS:
                v = mP[met]
                out.append(dict(kind=kind, mask=mname, metric=met, labour=float(mL[met][0]),
                                share_better=float((v < mL[met][0]).mean()),
                                p05=float(np.quantile(v, 0.05)), median=float(np.median(v)),
                                sdist=float((mL[met][0] - v.mean()) / v.std())))
    return out


def contributions(S: Setup, l: np.ndarray) -> pd.DataFrame:
    m = S.base_mask
    z = S.z(l[None])[0]
    x = S.x
    zz = z[m] * x[m].sum() / (z[m] @ x[m])
    e = S.e
    return pd.DataFrame(dict(industry=np.array(e.labels)[m], x_share=x[m] / x[m].sum(),
                             contrib=x[m] / x[m].sum() * np.abs(1 / zz - 1), z=zz,
                             hours_share=e.hours[m] / e.hours[m].sum(), va_share=e.va[m] / e.va[m].sum(),
                             li_share=e.labour_income[m] / e.labour_income[m].sum()))


def run(countries=COUNTRIES, years=YEARS):
    OUT.mkdir(parents=True, exist_ok=True)
    rows, contr = [], []
    for c in countries:
        D, top3 = None, None
        cons = []
        for y in years:
            e, _ = build(c, y)
            S = Setup(e)
            if D is None:
                D = Draws(c, e.n, S.base_mask)
                labels0 = list(e.labels)
            assert list(e.labels) == labels0
            cc = contributions(S, e.l())
            cc["country"], cc["year"] = c, y
            cons.append(cc)
            masks = {"all": S.base_mask, "no_prereg": S.mask(PREREG_EXCL)}
            for r in shares(S, D, e.l(), masks):
                rows.append(dict(country=c, year=y, **r))
            print("placebo_disp", c, y, flush=True)
        cc = pd.concat(cons)
        contr.append(cc)
        top3 = list(cc.groupby("industry").contrib.mean().sort_values(ascending=False).index[:3])
        for y in years:                                   # exploratory: drop the 3 largest contributors
            e, _ = build(c, y)
            S = Setup(e)
            for r in shares(S, D, e.l(), {"no_top3": S.mask(set(top3))}):
                rows.append(dict(country=c, year=y, top3="+".join(top3), **r))
    pd.DataFrame(rows).to_csv(OUT / "placebo_disp.csv", index=False)
    pd.concat(contr).to_csv(OUT / "placebo_disp_contrib.csv", index=False)


if __name__ == "__main__":
    run()
