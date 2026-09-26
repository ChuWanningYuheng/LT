"""Stage 6: validation.

(1) Replication of a published result on the closest data we have:
    Zachariah (2006, Table 2, OECD data) for Germany: labour-cost proxy for labour, finance / real
    estate / public administration treated as unproductive (excluded), depreciation part of inputs,
    OECD 'total' tables (imports competitive), prices of production on flow profit rate.
    Published: DEU 1995 (33 sectors) MAWD(LTV) = 0.102, MAWD(TPP) = 0.143; DEU 1990: 0.110 / 0.156.
(2) Order-of-magnitude comparison with Cockshott & Cottrell (1997) as reported by Cockshott,
    Cottrell & Valle Baeza (2014): UK 1984, CV of x-content/price: labour 0.198, electricity 3.69,
    oil 11.41, iron & steel 7.81 (open system: other inputs of the basis not traced to labour).
(3) Shaikh (1998, Table 15.1): US 1947-72 MAWD(labour values vs market prices) 0.071-0.105.
(4) Technical checks of the code on every country-year: identities, spectral radii, non-negativity.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .core import Economy, leontief_inverse, spectral_radius
from .economy import build
from .levels import mask_for

ROOT = Path(__file__).resolve().parents[2]
TAB = ROOT / "results" / "tables"

ZACH_EXCL = {"K64", "K65", "K66", "L", "O84"}


def mawd_z(z, x):
    """Zachariah/Shaikh MAWD: sum w|y-x| / sum w x with y = prices (=1), x = values (rescaled)."""
    z = z * x.sum() / (z @ x)
    return float((x * np.abs(1 - z)).sum() / (x * z).sum())


def zachariah_like(e: Economy) -> dict:
    labels = e.labels
    mask = mask_for(labels, ZACH_EXCL)
    comp = e.wages / e.x                     # labour costs per unit of output (D1 only)
    M, _, _ = e.system(True, False, "competitive", comp)
    v = comp @ leontief_inverse(M)
    # flow profit rate r = S/(C+V), S = VA - D1 - CFC (economy aggregate over included sectors)
    C = M.sum(0) * e.x
    S = e.va - e.wages - e.cfc
    r = S[mask].sum() / (C[mask].sum() + e.wages[mask].sum())
    R = 1 / spectral_radius(M) - 1
    r_used = min(r, 0.99 * R)
    p = comp @ leontief_inverse((1 + r_used) * M)
    out = dict(n=int(mask.sum()), r=r, r_used=r_used,
               mawd_ltv=mawd_z(v[mask], e.x[mask]), mawd_tpp=mawd_z(p[mask], e.x[mask]),
               rho_totals_ltv=float(np.corrcoef(v[mask] * e.x[mask] / (v[mask] @ e.x[mask]), e.x[mask])[0, 1]))
    # same, but with hours instead of labour costs (to isolate the effect of the wage proxy)
    lh = e.l()
    vh = lh @ leontief_inverse(M)
    out["mawd_ltv_hours"] = mawd_z(vh[mask], e.x[mask])
    return out


def technical_checks(e: Economy) -> dict:
    D, Dm = e.dep_coeffs()
    L = leontief_inverse(e.A)
    colsum = e.A.sum(0) + e.Am.sum(0) + e.va / e.x
    r = dict(rho_A=spectral_radius(e.A), rho_A_D=spectral_radius(e.A + D),
             rho_A_Am_D=spectral_radius(e.A + e.Am + D + Dm),
             min_L=float(L.min()), min_diag_L=float(np.diag(L).min()),
             max_colsum_gap=float(np.abs(colsum - 1).max()),
             neg_A=int((e.A < 0).sum()), neg_Am=int((e.Am < 0).sum()),
             hours_total=float(e.hours.sum()), zero_hours_with_output=int(((e.hours <= 0) & (e.x > 0)).sum()),
             cfc_share_of_gos=float(e.cfc.sum() / max(e.va.sum() - e.wages.sum(), 1)),
             labour_income_share_va=float(e.labour_income.sum() / e.va.sum()))
    # closed systems productive?
    for mode in (True, "uniform"):
        M, _, _ = e.system(True, mode, "competitive", e.l())
        r[f"rho_closed_{mode}"] = spectral_radius(M)
    # labour values identity v y = l x (domestic, competitive-free open system)
    l = e.l()
    v = l @ L
    y = e.x - e.A @ e.x
    r["vy_minus_lx_rel"] = float((v @ y - l @ e.x) / (l @ e.x))
    return r


def run(countries=("USA", "DEU", "MEX"), years=range(2010, 2024)):
    zrows, trows = [], []
    for c in countries:
        for y in years:
            try:
                e, info = build(c, y)
            except Exception as ex:
                trows.append(dict(country=c, year=y, error=repr(ex)))
                continue
            zrows.append(dict(country=c, year=y, **zachariah_like(e)))
            trows.append(dict(country=c, year=y, **technical_checks(e)))
    from . import bea
    for y in range(1998, 2024):
        e, info = bea.build(y)
        trows.append(dict(country="USA_BEA", year=y, **technical_checks(e)))
    Z, T = pd.DataFrame(zrows), pd.DataFrame(trows)
    Z.to_csv(TAB / "validation_zachariah_like.csv", index=False)
    T.to_csv(TAB / "validation_technical.csv", index=False)
    return Z, T


if __name__ == "__main__":
    Z, T = run()
    pd.set_option("display.width", 200)
    print(Z.groupby("country").mean(numeric_only=True).round(3))
    print(T.describe().T.round(4))
