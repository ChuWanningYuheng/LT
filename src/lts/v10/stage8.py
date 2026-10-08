"""Iteration 10, stage 8: robots (HS 847950 imports, Comtrade) and profitability (pre_registration_v10.md, stage 8;
journal 12). Descriptive (no pre-registered threshold).

Robot stock = perpetual inventory of real imports (US$ / US PPI machinery, FRED WPU11), linear retirement over 12
years, from 1996; per manufacturing worker (EU KLEMS 2025, EMP in C); panel from 2007 (full 12-year window).
Producers without IFR density (JP, DE) excluded in the main set. Regressions (iteration 5 style): d ln r, d ln e,
d ln (real profit per hour) on d ln robots per worker, country and year effects, clusters = countries.

Usage: PYTHONPATH=src python -P -m lts.v10.stage8
Outputs: results/v10/s8_*.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ..v6.series import ROOT
from ..v9b.stage45 import wild_t

OUT = ROOT / "results" / "v10"
R10 = ROOT / "data" / "raw" / "v10"
R9B = ROOT / "data" / "raw" / "v9b"
M49 = {"AT": 40, "BE": 56, "BG": 100, "CY": 196, "CZ": 203, "DE": 276, "DK": 208, "EE": 233, "EL": 300, "ES": 724,
       "FI": 246, "FR": 251, "HR": 191, "HU": 348, "IE": 372, "IT": 380, "JP": 392, "LT": 440, "LU": 442, "LV": 428,
       "MT": 470, "NL": 528, "PL": 616, "PT": 620, "RO": 642, "SE": 752, "SI": 705, "SK": 703, "UK": 826, "US": 842}
PRODUCERS = {"JP", "DE"}
LIFE = 12


def robots():
    c = pd.read_csv(R10 / "comtrade_847950.csv")
    c = c[(c.flowCode == "M") & (c.partnerCode == 0)]
    c = c.groupby(["reporterCode", "refYear"]).primaryValue.sum().reset_index()
    p = pd.read_csv(R10 / "fred_WPU11.csv")
    p["year"] = pd.to_datetime(p.iloc[:, 0]).dt.year
    ppi = p.groupby("year").WPU11.mean()
    inv = {v: k for k, v in M49.items()}
    c = c[c.reporterCode.isin(inv)]
    c["geo"] = c.reporterCode.map(inv)
    c["real"] = c.primaryValue / c.refYear.map(ppi) * 100
    rows = []
    for g, x in c.groupby("geo"):
        s = x.set_index("refYear").real.reindex(range(1996, 2025))
        for t in range(1996, 2025):
            w = [(s.get(u, np.nan), 1 - (t - u) / LIFE) for u in range(max(1996, t - LIFE + 1), t + 1)]
            vals = [a * b for a, b in w if np.isfinite(a)]
            n_missing = sum(1 for a, _ in w if not np.isfinite(a))
            rows.append(dict(geo=g, year=t, stock=float(np.sum(vals)) if vals else np.nan, missing_years=n_missing))
    return pd.DataFrame(rows), c


def klems():
    na = pd.read_csv(R9B / "klems2025_national_accounts.csv",
                     usecols=["nace_r2_code", "geo_code", "year", "VA_CP", "VA_PI", "COMP", "EMP", "H_EMP"], low_memory=False)
    ca = pd.read_csv(R9B / "klems2025_capital_accounts.csv", usecols=["nace_r2_code", "geo_code", "year", "K_GFCF"],
                     low_memory=False)
    m = na.merge(ca, on=["nace_r2_code", "geo_code", "year"], how="left")
    return m[m.geo_code.isin(M49)]


def fe_reg(d, y, x):
    d = d.dropna(subset=[y, x]).copy()
    for v in (y, x):
        d[v + "_dm"] = d[v] - d.groupby("geo")[v].transform("mean")
        d[v + "_dm"] -= d.groupby("year")[v + "_dm"].transform("mean")
    for _ in range(50):                                             # alternating projections, two-way FE
        for v in (y, x):
            d[v + "_dm"] -= d.groupby("geo")[v + "_dm"].transform("mean")
            d[v + "_dm"] -= d.groupby("year")[v + "_dm"].transform("mean")
    X = d[[x + "_dm"]].to_numpy()
    Y = d[y + "_dm"].to_numpy()
    b, se, lo_b, hi_b, p, G = wild_t(X, Y, d.geo.to_numpy(), 0, B=9999, seed=8)
    tq = stats.t.ppf(0.95, G - 1)
    return dict(beta=b, se=se, ci90_lo=min(b - tq * se, lo_b), ci90_hi=max(b + tq * se, hi_b), p_wild=p, G=G, n=len(d))


def run():
    S, C = robots()
    S.to_csv(OUT / "s8_robot_stock.csv", index=False)
    K = klems()
    man = K[K.nace_r2_code == "C"][["geo_code", "year", "EMP"]].rename(columns={"geo_code": "geo", "EMP": "emp_c"})
    rows = []
    for sector in ("TOT", "C"):
        t = K[K.nace_r2_code == sector].rename(columns={"geo_code": "geo"}).copy()
        t["r"] = (t.VA_CP - t.COMP) / t.K_GFCF
        t["e"] = (t.VA_CP - t.COMP) / t.COMP
        t["pi_h"] = (t.VA_CP - t.COMP) / (t.VA_PI / 100) / t.H_EMP
        P = t.merge(man, on=["geo", "year"]).merge(S, on=["geo", "year"])
        P = P[(P.year >= 2007) & (P.missing_years == 0)]
        P["rob_w"] = P.stock / P.emp_c
        P = P.sort_values(["geo", "year"])
        g = P.groupby("geo")
        for v in ("r", "e", "pi_h", "rob_w"):
            P["dln_" + v] = g[v].transform(lambda s: np.log(s.where(s > 0)).diff())
        for sample, D in (("main (without JP, DE)", P[~P.geo.isin(PRODUCERS)]), ("all KLEMS countries", P)):
            for v in ("r", "e", "pi_h"):
                r = fe_reg(D, "dln_" + v, "dln_rob_w")
                rows.append(dict(sector=sector, sample=sample, dep="d ln " + v, **r,
                                 years=f"{int(D.year.min())}-{int(D.year.max())}", countries=D.geo.nunique()))
        if sector == "TOT":
            P.to_csv(OUT / "s8_panel.csv", index=False)
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "s8_regressions.csv", index=False)
    pd.set_option("display.width", 250)
    print(R.round(4).to_string(), flush=True)


if __name__ == "__main__":
    run()
