"""Unit tests on small synthetic economies (python -m pytest tests)."""
import numpy as np
import pytest

from lts.core import Economy, content, leontief_inverse, pf_left
from lts.metrics import ratio_metrics


def toy(n=4, seed=0, imports=True):
    rng = np.random.default_rng(seed)
    A = rng.uniform(0, 0.15, (n, n))
    Am = rng.uniform(0, 0.05, (n, n)) if imports else np.zeros((n, n))
    x = rng.uniform(50, 150, n)
    va = x * (1 - A.sum(0) - Am.sum(0))
    assert (va > 0).all()
    hours = rng.uniform(1, 3, n) * x
    wages = 0.5 * va
    return Economy(labels=[f"s{i}" for i in range(n)], A=A, Am=Am, x=x, hours=hours, wages=wages,
                   cfc=0.1 * va, gfcf_dom=rng.uniform(1, 5, n), gfcf_imp=rng.uniform(0, 1, n) if imports else np.zeros(n),
                   hh_dom=rng.uniform(5, 20, n), hh_imp=rng.uniform(0, 3, n) if imports else np.zeros(n), va=va)


def test_leontief_series():
    e = toy()
    L = leontief_inverse(e.A)
    S = sum(np.linalg.matrix_power(e.A, k) for k in range(200))
    assert np.allclose(L, S)
    assert (L >= 0).all() and (np.diag(L) >= 1).all()


def test_unproductive_raises():
    with pytest.raises(ValueError):
        leontief_inverse(np.full((2, 2), 0.6))


def test_labour_values_identity():
    # v y = l x : total labour embodied in net product equals total hours
    e = toy(imports=False)
    l = e.l()
    v = l @ leontief_inverse(e.A)
    y = e.x - e.A @ e.x
    assert np.isclose(v @ y, e.hours.sum())


@pytest.mark.parametrize("imports", ["price", "competitive"])
@pytest.mark.parametrize("capital", [False, True])
def test_closed_system_labour_basis_equals_open(imports, capital):
    # appending labour as a produced good must not change labour values (its row is made primary)
    e = toy()
    z_open = e.basis_ratios("L", capital=capital, closed=False, imports=imports)
    z_closed = e.basis_ratios("L", capital=capital, closed=True, imports=imports)
    assert np.allclose(z_open, z_closed)


def test_normalisation():
    e = toy()
    for b in ["L", 0, 2]:
        for closed in (False, True):
            z = e.basis_ratios(b, closed=closed)
            assert np.isclose(z @ e.x, e.x.sum())


def test_commodity_basis_matches_formula():
    e = toy(imports=False)
    k = 1
    Ak = e.A.copy()
    Ak[k] = 0
    lam = e.A[k] @ np.linalg.inv(np.eye(e.n) - Ak)
    lam2 = content(e.A, e.A[k], zero_row=k)
    assert np.allclose(lam, lam2)
    # k-content of k itself < 1 for a productive system (commodity "exploitation")
    assert lam[k] < 1


def test_prices_r0_uniform_equal_values():
    # at r = 0 with uniform wage and imports at price, prices of production = labour value ratios
    e = toy()
    p0 = e.prices_of_production(0.0, capital=False, imports="price", wage_mode="uniform")
    zL = e.basis_ratios("L", capital=False, imports="price")
    assert np.allclose(p0, zL)


def test_prices_actual_wage_r_actual_reproduce_market_prices_when_profits_proportional():
    # if every industry's profit = r * (its non-labour inputs), actual-wage prices of production = 1
    e = toy()
    r = 0.2
    M, Mm, _ = e.system(False, False, "price", e.l())
    inputs = M.sum(0) + Mm.sum(0)
    li = 1 - inputs * (1 + r)
    assert (li > 0).all()
    e.labour_income = li * e.x
    e.va = e.x * (1 - inputs)
    e.cfc = np.zeros(e.n)
    p = e.prices_of_production(r, capital=False, wage_mode="actual")
    assert np.allclose(p, 1.0)
    assert np.isclose(e.actual_profit_rate(capital=False), r)


def test_eigen_prices():
    e = toy(imports=False)
    r, p = e.closed_eigen_prices(capital=False)
    M, _, lab = e.system(False, True, "competitive", e.l())
    Mc = M[:e.n, :e.n] + M[:e.n, lab] @ M[lab, :e.n]
    assert np.allclose(p, (1 + r) * p @ Mc)
    assert (p > 0).all()


def test_pf_left():
    M = np.array([[0.2, 0.3], [0.1, 0.4]])
    lam, v = pf_left(M)
    assert np.allclose(v @ M, lam * v)


def test_metrics_scale_invariance():
    rng = np.random.default_rng(1)
    z = rng.lognormal(0, 0.2, 30)
    x = rng.uniform(1, 10, 30)
    m1 = ratio_metrics(z, x)
    m2 = ratio_metrics(7.3 * z, x)
    for k in ["cv", "cv_w", "mad", "mawd", "d", "log_sd_w"]:
        assert np.isclose(m1[k], m2[k])
    # perfect proportionality -> zero deviation
    m0 = ratio_metrics(np.full(30, 3.0), x)
    assert m0["mawd"] < 1e-12 and m0["d"] < 1e-7


def test_k_exploitation_positive():
    e = toy(imports=False)
    rates = e.k_exploitation_rates()
    assert all(v > 0 for v in rates.values())


def test_closed_system_productive_with_heterogeneous_wages():
    # low-wage, labour-intensive industry: average-wage closure would be unproductive, ours is not
    e = toy(imports=False)
    e.hours[0] *= 50
    e.labour_income = 0.9 * e.va
    M, _, lab = e.system(True, True, "competitive", e.l())
    from lts.core import spectral_radius
    assert spectral_radius(M) < 1
    for k in range(e.n):
        z = e.basis_ratios(k, capital=True, closed=True, imports="competitive")
        assert np.isfinite(z).all() and (z > 0).all()


def test_closed_commodity_basis_counts_wage_goods():
    # in the closed system the k-content is at least the open-system k-content
    e = toy(imports=False)
    e.labour_income = 0.6 * e.va
    for k in range(e.n):
        M_open, _, _ = e.system(False, False, "competitive", e.l())
        M_cl, _, lab = e.system(False, True, "competitive", e.l())
        lo = content(M_open, M_open[k], zero_row=k)
        lc = content(M_cl, M_cl[k], zero_row=k)[:e.n]
        assert (lc >= lo - 1e-12).all()


def test_closed_uniform_productive_and_labour_unchanged():
    e = toy(imports=False)
    e.hours[0] *= 50
    e.labour_income = 0.9 * e.va
    from lts.core import spectral_radius
    M, _, lab = e.system(True, "uniform", "competitive", e.l())
    assert spectral_radius(M) < 1
    z_u = e.basis_ratios("L", capital=True, closed="uniform", imports="competitive")
    z_o = e.basis_ratios("L", capital=True, closed=False, imports="competitive")
    assert np.allclose(z_u, z_o)
