"""Iteration 2, stage 1.3: workers' basket variants for the symmetric (closed) test.

The closed system is written in collapsed form: Mc = M + b_dom h' (uniform bundle b per hour, h = hours per
unit of output) or Mc = M + beta_dom omega' (actual wages: omega_j = labour income per unit of output).
Every commodity k is a basis: lambda^k = Mc[k,:](I - Mc_(k))^{-1}; imported content via Mc^m.
Labour is the open-system labour value (it is unchanged by the closure).

Basket variants (A-BASKET-*):
  a_uniform   b = wbar * beta_t                 (iteration 1, closed="uniform")
  a_actual    actual wages (iteration 1, closed=True)
  b_physical  base-year basket per hour held fixed in real terms: b_t = s_t * wbar_t * beta_t,
              s_t = (wbar_base,nat * Pc_t / Pc_base) / wbar_t,nat  (s_base = 1)
  scale_s     b = s * wbar * beta_t, s in a grid (and s_max: largest productive scale)
  c_necess_full   necessities only (food, clothing, housing, utilities, fuel, transport, retail),
                  full average wage spent on them
  c_necess_actual necessities only, at their actual share of household consumption
  d_social    household + government + NPISH final consumption ("social wage"): value per hour
              wbar * (C_hh + C_gov) / C_hh, composition of hh + gov
If a bundle makes the closed system unproductive (rho >= 0.995) it is scaled down to rho = 0.995 and
the scale actually used is recorded.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..core import Economy, leontief_inverse, spectral_radius
from ..levels import SUBSETS, mask_for
from ..metrics import ratio_metrics

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "v2"

NECESS_FIG = {"A01", "C10-12", "C13-15", "L", "D35", "E36", "H49", "C19", "G47"}
NECESS_BEA = {"111CA", "311FT", "313TT", "315AL", "HS+ORE", "22", "485", "324", "445", "452", "4A0"}
SCALES = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
METRICS = ["mawd", "d", "cv_w", "log_sd_w"]


def zvec(lam: np.ndarray, mu: np.ndarray, x: np.ndarray) -> np.ndarray:
    e = (x.sum() - mu @ x) / (lam @ x)
    return e * lam + mu


def closed_bases(M, Mm, add_dom, add_imp, x):
    """All commodity bases in the closed system Mc = M + add_dom, Mcm = Mm + add_imp."""
    Mc, Mcm = M + add_dom, Mm + add_imp
    rho = spectral_radius(Mc + Mcm)            # conservative: productive even if imports were produced
    out = {}
    for k in range(M.shape[0]):
        if M[k].sum() <= 0:
            continue
        S = Mc.copy()
        S[k, :] = 0.0
        L = leontief_inverse(S)
        out[k] = zvec(Mc[k] @ L, Mcm.sum(0) @ L, x)
    return out, rho


def max_scale(M, Mm, bdom, bimp, h, target=0.995):
    f = lambda s: spectral_radius(M + Mm + s * np.outer(bdom + bimp, h))
    if f(10.0) <= target:
        return 10.0
    lo, hi = 0.0, 10.0
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) <= target else (lo, mid)
    return lo


def percentile_rows(zL: np.ndarray, zk: dict, e: Economy, role_map, meta: dict) -> list[dict]:
    rows = []
    for sub, excl in SUBSETS.items():
        if sub not in ("all", "core"):
            continue
        mask = mask_for(e.labels, excl, role_map)
        mL = ratio_metrics(zL, e.x, mask)
        mk = [ratio_metrics(z, e.x, mask) for z in zk.values()]
        for met in METRICS:
            vals = np.array([m[met] for m in mk])
            rows.append(dict(**meta, subset=sub, metric=met, labour=mL[met],
                             share_better=float((vals < mL[met]).mean()), comm_median=float(np.median(vals)),
                             n_bases=len(vals)))
    return rows


def run_economy(e: Economy, country: str, year: int, capital: bool, s_phys: float, necess: set,
                role_map=None) -> list[dict]:
    M, Mm, _ = e.system(capital, False, "price", e.l())
    x = e.x
    h = e.l()
    lamL = h @ leontief_inverse(M)
    muL = Mm.sum(0) @ leontief_inverse(M)
    zL = zvec(lamL, muL, x)
    li = e.labour_income
    wbar = li.sum() / e.hours.sum()
    C = e.hh_dom.sum() + e.hh_imp.sum()
    bd, bi = e.hh_dom / C, e.hh_imp / C
    rows = []
    base = dict(country=country, year=year, capital=capital)

    def do(variant, add_dom, add_imp, scale_used, extra=None):
        zk, rho = closed_bases(M, Mm, add_dom, add_imp, x)
        rows.extend(percentile_rows(zL, zk, e, role_map,
                                    dict(**base, variant=variant, scale_used=scale_used, rho=rho, **(extra or {}))))

    smax = max_scale(M, Mm, wbar * bd, wbar * bi, h)
    # (a) iteration-1 variants
    omega = li / x
    do("a_actual", np.outer(bd, omega), np.outer(bi, omega), 1.0)
    do("a_uniform", wbar * np.outer(bd, h), wbar * np.outer(bi, h), 1.0)
    # (b) physically fixed basket
    s = min(s_phys, smax)
    do("b_physical", s * wbar * np.outer(bd, h), s * wbar * np.outer(bi, h), s, dict(s_phys=s_phys))
    # scale grid
    for sc in SCALES + [smax]:
        sc_u = min(sc, smax)
        do(f"scale_{sc:.2f}" if sc != smax else "scale_max", sc_u * wbar * np.outer(bd, h),
           sc_u * wbar * np.outer(bi, h), sc_u, dict(s_max=smax))
    # (c) necessities
    nm = np.array([any(p in necess for p in [lab] + lab.split("+")) for lab in e.labels])
    shareN = (bd[nm].sum() + bi[nm].sum())
    bdN, biN = np.where(nm, bd, 0) / shareN, np.where(nm, bi, 0) / shareN
    sN = min(1.0, max_scale(M, Mm, wbar * bdN, wbar * biN, h))
    do("c_necess_full", sN * wbar * np.outer(bdN, h), sN * wbar * np.outer(biN, h), sN, dict(necess_share=shareN))
    do("c_necess_actual", wbar * np.outer(np.where(nm, bd, 0), h), wbar * np.outer(np.where(nm, bi, 0), h),
       1.0, dict(necess_share=shareN))
    # (d) social wage
    gd, gi = e.meta.get("gov_dom"), e.meta.get("gov_imp")
    if gd is not None:
        G = gd.sum() + gi.sum()
        bdG, biG = (e.hh_dom + gd) / C, (e.hh_imp + gi) / C       # value per hour = wbar*(C+G)/C
        sG = min(1.0, max_scale(M, Mm, wbar * bdG, wbar * biG, h))
        do("d_social", sG * wbar * np.outer(bdG, h), sG * wbar * np.outer(biG, h), sG, dict(gov_to_hh=G / C))
    return rows


# ------------------------------------------------------------------------------------------------
def consumer_price_index(e: Economy, base_weights: np.ndarray) -> float:
    p = e.meta["price_index"]
    ok = np.isfinite(p) & (base_weights > 0)
    return float((base_weights[ok] * p[ok]).sum() / base_weights[ok].sum())


def run(source="figaro", countries=("USA", "DEU", "MEX"), years=None):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    if source == "bea":
        from .. import bea
        countries, years = ("USA_BEA",), years or range(1998, 2024)
        role_map, necess = bea.BEA_ROLE, NECESS_BEA
    else:
        from .. import oecd
        from ..economy import build
        years = years or range(2010, 2024)
        role_map, necess = None, NECESS_FIG
    for c in countries:
        base = None
        for y in years:
            e = bea.build(y)[0] if source == "bea" else build(c, y)[0]
            # nominal wage in national currency: FIGARO in EUR -> convert with EUR/national ratio of output
            if source == "bea":
                fx = 1.0
            else:
                p1 = oecd.series(c, "table6", "P1", "XDC", "V")
                fx = e.x.sum() / p1.loc[y, "_T"] if y in p1.index else np.nan
            wbar_nat = e.labour_income.sum() / e.hours.sum() / fx
            if base is None:
                base = dict(w=wbar_nat, weights=e.hh_dom.copy(), labels=list(e.labels), pc=None)
                base["pc"] = consumer_price_index(e, base["weights"])
            wts = pd.Series(base["weights"], index=base["labels"]).reindex(e.labels).fillna(0).to_numpy()
            pc = consumer_price_index(e, wts)
            s_phys = base["w"] * (pc / base["pc"]) / wbar_nat
            for cap in (True, False):
                rows += run_economy(e, c, y, cap, s_phys, necess, role_map)
            print(source, c, y, f"s_phys={s_phys:.3f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"baskets_{source}.csv", index=False)
    return df


if __name__ == "__main__":
    import sys
    run(sys.argv[1] if len(sys.argv) > 1 else "figaro")
