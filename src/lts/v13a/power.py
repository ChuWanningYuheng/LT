"""Iteration 13a: expected precision before the calculations (journal item 3, answer to review P17).

Uses only the series of iteration 12b (already published and disclosed in 'Известно при написании'): the same
CI procedure as stage 1 (wider of HAC lag 4 and the pairs block bootstrap) is applied to the 12b analogues on the
13a phases. Only half-widths, residual sigma and AR(1) are written - slope estimates are NOT computed into the
output and not printed.

Usage: PYTHONPATH=src python -P -m lts.v13a.power  ->  results/v13a/p0_power.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..v6.series import ROOT
from . import est

OUT = ROOT / "results" / "v13a"
PHASES = {"A": (1948, 1973), "B": (1983, 2007), "C": (2010, 2019)}


def resid_stats(y, phases):
    b, yy, ph = est.within_slope(y, phases)
    t = yy.index.to_numpy(float)
    r = yy.to_numpy(float).copy()
    for g in ph.unique():
        m = (ph == g).to_numpy()
        r[m] = r[m] - r[m].mean() - b * (t[m] - t[m].mean())
    rho = float(np.corrcoef(r[:-1], r[1:])[0, 1])
    return float(r.std(ddof=len(ph.unique()) + 1)), rho


def row(name, analogue, y, delta, kind="within", phases=PHASES, pa=None, pb=None):
    if kind == "within":
        o = est.within_trend(y, phases, delta, "<")
        sig, rho = resid_stats(y, phases)
        hac = est.Z90 * o["se_hac"]
    elif kind == "diff":
        o = est.two_phase_diff(y, pa, pb, delta)
        sig, rho = np.nan, np.nan
        hac = est.Z90 * o["se_hac"]
    else:
        o = est.single_trend(y, delta, "<")
        sig, rho = np.nan, np.nan
        hac = est.Z90 * o["se_hac"]
    h = o["half_width"]
    return dict(outcome=name, analogue_12b=analogue, delta=delta, half_width_hac=hac, half_width=h,
                sigma_resid=sig, rho1_resid=rho, equivalence_attainable=bool(h < delta),
                min_abs_slope_for_confirmed=h)


def run():
    d = pd.read_csv(ROOT / "results" / "v12b" / "s1_us_series.csv")
    d = d[d.variant == "main"].set_index("year").sort_index()
    lr, lk, le = np.log(d.r_h), np.log(d.k_h), np.log(d.e_h)
    r_all = d.S_h / (d.C_h_all + d.V_h)
    rows = [row("110 (k, внутри фаз)", "ln k_h 12б", lk, 0.002),
            row("111, 114, 115 (r, внутри фаз)", "ln r_h 12б", lr, 0.002),
            row("117 (r(В)/r(П), внутри фаз)", "ln(r_все / r_h) 12б", np.log(r_all / d.r_h), 0.001),
            row("125 (C/L, внутри фаз)", "ln(C_h / L) 12б", np.log(d.C_h / d.tot_h), 0.002),
            row("116 (e, B − A′)", "ln e_h 12б", le, 0.002, "diff", pa=(1964, 1973), pb=(1983, 2007)),
            row("126 (r, B − A)", "ln r_h 12б", lr, 0.002, "diff", pa=(1948, 1973), pb=(1983, 2007)),
            row("127 (k, B − A)", "ln k_h 12б", lk, 0.002, "diff", pa=(1948, 1973), pb=(1983, 2007)),
            row("131 (вековой тренд r, 1948–2023)", "ln r_h 12б", lr, 0.002, "single")]
    # world: 12b world hours (WIOD 2000-2014); residual sigma and AR(1) within 2000-2007 and 2010-2014, then the
    # analytic half-width for the EXIOBASE phases 1995-2007 and 2010-2019
    w = pd.read_csv(ROOT / "results" / "v12b" / "s2_world_hours.csv").set_index("year").sort_index()
    wph = {"A": (2000, 2007), "B": (2010, 2014)}
    sxx = sum(((np.arange(a, b + 1) - (a + b) / 2) ** 2).sum() for a, b in ((1995, 2007), (2010, 2019)))
    for nm, col in (("120 (мир, r)", "r_h"), ("121 (мир, k)", "k_h")):
        sig, rho = resid_stats(np.log(w[col]), wph)
        rho_c = min(max(rho, 0.0), 0.9)
        h = est.Z90 * sig / np.sqrt(sxx) * np.sqrt((1 + rho_c) / (1 - rho_c))
        rows.append(dict(outcome=nm, analogue_12b=f"ln {col} мир 12б (WIOD 2000–2014), аналитически для фаз 1995–2007, "
                                                  "2010–2019", delta=0.002, half_width_hac=np.nan, half_width=h,
                         sigma_resid=sig, rho1_resid=rho, equivalence_attainable=bool(h < 0.002),
                         min_abs_slope_for_confirmed=h))
    rows.append(dict(outcome="113 (R)", analogue_12b="аналога в 12б нет (моральный износ не считался)", delta=0.25))
    P = pd.DataFrame(rows)
    P.to_csv(OUT / "p0_power.csv", index=False)
    print(P.round(4).to_string())


if __name__ == "__main__":
    run()
