"""Iteration 12, descriptive additions after the main results (pre_registration_v12.md, journal 8).

AMECO countries of stage 1.2(B): log changes of r_full and its factors (sigma, n, rho, q, F) by period
(first year-1973, 1973-1997, 1997-2024 and the whole series), from results/v12/s1_ameco_*.csv.

Usage: PYTHONPATH=src python -P -m lts.v12.extra
Outputs: results/v12/x_ameco_contributions.csv
"""
from __future__ import annotations

import pandas as pd

from ..v6.series import ROOT
from .stage1 import contributions

OUT = ROOT / "results" / "v12"
GEOS = ["AUS", "CAN", "DEU", "FRA", "GBR", "ITA", "JPN", "NLD", "SWE", "USA"]


def run():
    rows = []
    for g in GEOS:
        a = pd.read_csv(OUT / f"s1_ameco_{g}.csv").set_index("year")
        parts = {k: a[k] for k in ("r_full", "r", "sigma", "n", "rho", "q", "F")}
        first = int(a.r_full.dropna().index.min())
        C = contributions(parts, ((min(first, 1973), 1973), (1973, 1997), (1997, 2024), (first, 2024)))
        rows.append(C.assign(geo=g))
    D = pd.concat(rows, ignore_index=True)
    D.to_csv(OUT / "x_ameco_contributions.csv", index=False)
    pd.set_option("display.width", 250)
    print(D[["geo", "period", "r_full", "sigma", "n", "rho", "q", "F"]].round(3).to_string())


if __name__ == "__main__":
    run()
