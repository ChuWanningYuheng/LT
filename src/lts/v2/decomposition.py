"""Iteration 2, stage 1b: decomposition of symmetric X-values into own content and labour part.

v_X^sym = O_X + theta_X * v_(k)   (derivation: report/methods_decomposition.md)

Outputs (results/v2/):
  decomp_basis.csv   one row per country-year-basis: identity errors, theta, labour share of v_X^sym,
                     comparison v_(k) vs v, metrics of z_sym, z_O (asymmetric X alone), z_L(k) (labour without X)
  decomp_theta.csv   labour percentile among bases as a function of basket scale s (0..1.5)
  decomp_z.npz       per country: arrays of z_labour[y, j], z_O[b, y, j], x[y, j], eval mask, labels
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..core import leontief_inverse
from ..economy import build
from ..levels import mask_for
from ..metrics import ratio_metrics

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "v2"
COUNTRIES = ("USA", "DEU", "MEX", "FRA", "ITA", "ESP", "NLD", "AUT", "POL", "CZE", "KOR", "JPN", "GBR")
S_GRID = np.round(np.arange(0, 1.51, 0.1), 2)
MET = ("mawd", "d", "cv_w")


def ztrans(lam, mu, x):
    return (x.sum() - mu @ x) / (lam @ x) * lam + mu


def btype(label: str) -> str:
    c = label.split("+")[0][0]
    return "goods" if c in "ABCDE" else ("construction" if c == "F" else "services")


def one_economy(e, country, year, rows, trows, store):
    M, Mm, _ = e.system(True, False, "price", e.l())
    n, x, h = e.n, e.x, e.l()
    wbar = e.labour_income.sum() / e.hours.sum()
    C = e.hh_dom.sum() + e.hh_imp.sum()
    b, bm = wbar * e.hh_dom / C, wbar * e.hh_imp / C
    mask = mask_for(e.labels, set())
    Lfull = leontief_inverse(M)
    v = h @ Lfull
    muL = Mm.sum(0) @ Lfull
    zL = ztrans(v, muL, x)
    mL = ratio_metrics(zL, x, mask)
    colsMm = Mm.sum(0)
    Mc, Mcm = M + np.outer(b, h), Mm + np.outer(bm, h)
    zO_store, parts = [], {}
    for k in range(n):
        if M[k].sum() <= 0:
            continue
        Mk = M.copy()
        Mk[k] = 0
        Lk = leontief_inverse(Mk)
        O = M[k] @ Lk
        vk = h @ Lk
        bk = b.copy()
        bk[k] = 0
        phi = vk @ bk
        theta = (b[k] + O @ bk) / (1 - phi)
        lam = O + theta * vk
        muO = colsMm @ Lk
        thm = (bm.sum() + muO @ bk) / (1 - phi)
        mu = muO + thm * vk
        # identity check against direct inversion of the closed system
        Sk = Mc.copy()
        Sk[k] = 0
        Lc = leontief_inverse(Sk)
        lam_d, mu_d = Mc[k] @ Lc, Mcm.sum(0) @ Lc
        err_l = float(np.max(np.abs(lam - lam_d) / np.maximum(np.abs(lam_d), 1e-300)))
        err_m = float(np.max(np.abs(mu - mu_d) / np.maximum(np.abs(mu_d), 1e-300)))
        e_sym = (x.sum() - mu @ x) / (lam @ x)
        zsym = e_sym * lam + mu
        zO = ztrans(O, muO, x)             # X alone (asymmetric)
        zLk = ztrans(vk, muO, x)           # labour without X as input
        share_w = float((x * theta * vk)[mask].sum() / (x * lam)[mask].sum())
        share_med = float(np.median((theta * vk / lam)[mask]))
        ms, mo, ml = (ratio_metrics(z, x, mask) for z in (zsym, zO, zLk))
        r = dict(country=country, year=year, basis=e.labels[k], type=btype(e.labels[k]),
                 err_lambda=err_l, err_mu=err_m, phi=float(phi), theta=float(theta),
                 share_L_w=share_w, share_L_med=share_med,
                 corr_vk_v=float(np.corrcoef(vk[mask], v[mask])[0, 1]),
                 maxdev_vk_v=float(np.max(np.abs(vk[mask] / v[mask] - 1))),
                 h_direct=float(h[k]), v_total=float(v[k]), cost_share=float(M[k] @ x / x.sum()))
        for m in MET:
            r[f"{m}_labour"], r[f"{m}_sym"], r[f"{m}_O"], r[f"{m}_Lk"] = mL[m], ms[m], mo[m], ml[m]
        rows.append(r)
        zO_store.append((e.labels[k], zO, zsym))
        parts[k] = (O, vk, theta / 1.0, muO, thm, phi, b[k] + O @ bk, bm.sum() + muO @ bk)
    # theta curve: basket scale s -> theta(s) = s*(b_k + O b_(k)) / (1 - s*phi)
    # feasibility: closed system incl. imports treated as produced must be productive (rho <= 0.995), the
    # same bound as in baskets.py; beyond it imported content exceeds the price (e < 0) in some bases
    from .baskets import max_scale
    s_max = max_scale(M, Mm, b, bm, h)
    for s in S_GRID:
        if s >= s_max:
            for m in MET:
                trows.append(dict(country=country, year=year, s=float(s), metric=m, labour=mL[m],
                                  share_better=np.nan, n_bases=0, feasible=False, s_max=s_max))
            continue
        vals = []
        for k, (O, vk, _, muO, _, phi, num, numm) in parts.items():
            if s * phi >= 0.999:
                continue
            th, thm = s * num / (1 - s * phi), s * numm / (1 - s * phi)
            z = ztrans(O + th * vk, muO + thm * vk, x)
            vals.append(ratio_metrics(z, x, mask))
        for m in MET:
            arr = np.array([q[m] for q in vals])
            trows.append(dict(country=country, year=year, s=float(s), metric=m, labour=mL[m],
                              share_better=float((arr < mL[m]).mean()), n_bases=len(arr),
                              feasible=True, s_max=s_max))
    store.setdefault("zL", []).append(zL)
    store.setdefault("zO", []).append({lab: zo for lab, zo, _ in zO_store})
    store.setdefault("zS", []).append({lab: zs for lab, _, zs in zO_store})
    store.setdefault("x", []).append(x)
    store["mask"], store["labels"] = mask, np.array(e.labels)
    store.setdefault("years", []).append(year)


def _stack(dicts, labels, n):
    """(basis, year, industry) array aligned on the union of bases; NaN where a basis is absent."""
    out = np.full((len(labels), len(dicts), n), np.nan)
    for t, d in enumerate(dicts):
        for i, lab in enumerate(labels):
            if lab in d:
                out[i, t] = d[lab]
    return out


def run_curve(countries=COUNTRIES, years=range(2010, 2023)):
    """Recompute only the theta curve (decomp_theta.csv)."""
    trows = []
    for c in countries:
        for y in years:
            e, _ = build(c, y)
            one_economy(e, c, y, [], trows, {})
        print("curve", c, flush=True)
    pd.DataFrame(trows).to_csv(OUT / "decomp_theta.csv", index=False)


def run(countries=COUNTRIES, years=range(2010, 2023)):
    OUT.mkdir(parents=True, exist_ok=True)
    rows, trows = [], []
    for c in countries:
        store = {}
        for y in years:
            e, _ = build(c, y)
            one_economy(e, c, y, rows, trows, store)
            print("decomp", c, y, flush=True)
        bases = [l for l in store["labels"] if all(l in d for d in store["zO"])]   # bases present in all years
        n = len(store["labels"])
        np.savez_compressed(OUT / f"decomp_z_{c}.npz", zL=np.array(store["zL"]),
                            zO=_stack(store["zO"], bases, n), zS=_stack(store["zS"], bases, n),
                            x=np.array(store["x"]), mask=store["mask"], labels=store["labels"],
                            bases=np.array(bases), years=np.array(store["years"]))
    pd.DataFrame(rows).to_csv(OUT / "decomp_basis.csv", index=False)
    pd.DataFrame(trows).to_csv(OUT / "decomp_theta.csv", index=False)


if __name__ == "__main__":
    import sys
    run_curve() if len(sys.argv) > 1 and sys.argv[1] == "curve" else run()
