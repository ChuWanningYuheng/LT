"""Iteration 9, stage 6: real exchange rate and relative real unit labour costs of tradables (Shaikh).

q_ct = ln(xr_ct * P_US / P_c) (xr: national currency per US$, P: GDP deflator);
v_ct = ln[(ULC_c / P_c) / (ULC_US / P_US)], ULC = compensation / real value added in manufacturing.
Engle-Granger per country; out-of-sample ECM vs AR(1) (PPP reversion) vs random walk (pre_registration_v9.md, journal 7).
Outputs: results/v9/s6_*.csv
"""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

from ..v8.rule import decide
from .stage1 import OUT
from .stage34 import R9

ROOT = OUT.parents[1]


def stan():
    rows = {}
    for f in glob.glob(str(ROOT / "data" / "raw" / "oecd" / "stan2025_*.csv")):
        d = pd.read_csv(f, low_memory=False)
        c = d.REF_AREA.iloc[0]
        g = lambda act, m, pb: d[(d.ACTIVITY == act) & (d.MEASURE == m) & (d.PRICE_BASE == pb)].set_index("TIME_PERIOD").OBS_VALUE
        P = g("_T", "B1G", "V") / g("_T", "B1G", "L")
        ulc = g("C", "D1", "V") / g("C", "B1G", "L")
        rows[c] = pd.DataFrame({"P": P, "ulc": ulc}).sort_index()
    return rows


def panel(src="stan"):
    p = pd.read_excel(R9 / "pwt110.xlsx", sheet_name="Data")
    xr = {(r.countrycode, int(r.year)): r.xr for r in p.itertuples() if np.isfinite(r.xr)}
    if src == "stan":
        S = stan()
    else:
        S = sea_manuf()
    us = S["USA"]
    out = []
    for c, d in S.items():
        if c == "USA":
            continue
        d = d.join(us, rsuffix="_us", how="inner")
        d["xr"] = [xr.get((c, int(y)), np.nan) for y in d.index]
        d["q"] = np.log(d.xr * d.P_us / d.P)
        d["v"] = np.log((d.ulc / d.P) / (d.ulc_us / d.P_us))
        d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=["q", "v"])
        out.append(d.assign(country=c))
    return pd.concat(out)


def sea_manuf():
    from ..revisit.wiod import sea
    s = sea()
    s["year"] = s.year.astype(int)
    s = s[s.code.str.startswith("C") & s.variable.isin(["LAB", "VA", "VA_QI"])]
    w = s.pivot_table(index=["country", "code", "year"], columns="variable", values="v").reset_index()
    w["VA_real"] = w.VA.groupby([w.country, w.code]).transform(lambda v: v) * 0 + w.VA_QI     # index only
    # real VA in 2010 prices: VA_2010 * VA_QI / 100
    base = w[w.year == 2010].set_index(["country", "code"]).VA
    w["VAr"] = [base.get((c, k), np.nan) * qi / 100 for c, k, qi in zip(w.country, w.code, w.VA_QI)]
    a = w.groupby(["country", "year"]).agg(LAB=("LAB", "sum"), VA=("VA", "sum"), VAr=("VAr", "sum")).reset_index()
    out = {}
    for c, d in a.groupby("country"):
        d = d.set_index("year")
        out[c] = pd.DataFrame({"P": d.VA / d.VAr, "ulc": d.LAB / d.VAr})
    return out


def eg(d, maxlag=None):
    if maxlag is None:
        t, p, _ = coint(d.q, d.v, trend="c")
    else:
        t, p, _ = coint(d.q, d.v, trend="c", maxlag=maxlag, autolag=None)
    beta = sm.OLS(d.q, sm.add_constant(d.v)).fit().params.v
    return p, beta


def oos(d, h, train=10):
    q, v = d.q.to_numpy(), d.v.to_numpy()
    e_ecm, e_ar, e_rw = [], [], []
    for t in range(train - 1, len(q) - h):
        qs, vs = q[:t + 1], v[:t + 1]
        lr = sm.OLS(qs, sm.add_constant(vs)).fit().params
        ect = qs - lr[0] - lr[1] * vs
        y = qs[h:] - qs[:-h]
        f1 = sm.OLS(y, sm.add_constant(ect[:-h])).fit().params
        f2 = sm.OLS(y, sm.add_constant(qs[:-h])).fit().params
        actual = q[t + h] - q[t]
        e_ecm.append(actual - (f1[0] + f1[1] * ect[-1]))
        e_ar.append(actual - (f2[0] + f2[1] * qs[-1]))
        e_rw.append(actual)
    rm = lambda e: np.sqrt(np.mean(np.square(e)))
    return dict(h=h, n=len(e_ecm), rmse_ecm=rm(e_ecm), rmse_ar=rm(e_ar), rmse_rw=rm(e_rw),
                ratio_ecm_ar=rm(e_ecm) / rm(e_ar), ratio_ecm_rw=rm(e_ecm) / rm(e_rw))


def run():
    rows, fc = [], []
    for src in ("stan", "wiod"):
        P = panel(src)
        for c, d in P.groupby("country"):
            if len(d) < 12:
                continue
            p, beta = eg(d)
            p0, _ = eg(d, 0)
            p1, _ = eg(d, 1)
            rows.append(dict(source=src, country=c, years=f"{d.index.min()}-{d.index.max()}", n=len(d), eg_p=p, beta=beta,
                             coint=p < 0.05, eg_p_lag0=p0, coint_lag0=p0 < 0.05, eg_p_lag1=p1, coint_lag1=p1 < 0.05))
            if src == "stan":
                for h in (1, 3):
                    fc.append(oos(d, h) | dict(country=c))
    R = pd.DataFrame(rows)
    F = pd.DataFrame(fc)
    out = []
    from ..v8.rule import share_row
    for src, g in R.groupby("source"):
        out.append(share_row(f"outcome 35 ({src}): share of countries cointegrated", g.coint, 0.05, 0.15, ">"))
        for lag in (0, 1):
            out.append(share_row(f"variant (journal 14), ADF lag {lag} ({src}): share cointegrated", g[f"coint_lag{lag}"],
                                 0.05, 0.15, ">"))
    f3 = F[F.h == 3].ratio_ecm_ar.to_numpy()
    rng = np.random.default_rng(66)
    m = f3[rng.integers(0, len(f3), (2000, len(f3)))].mean(1)
    lo, hi = np.quantile(m, [0.05, 0.95])
    out.append(dict(outcome="outcome 36: mean RMSE ratio ECM / AR(1), h = 3", est=f3.mean(), ci90_lo=lo, ci90_hi=hi,
                    n=len(f3), label=decide(f3.mean(), lo, hi, 1, 0.05, "<")))
    R.to_csv(OUT / "s6_coint.csv", index=False)
    F.to_csv(OUT / "s6_oos.csv", index=False)
    pd.DataFrame(out).to_csv(OUT / "s6_outcomes.csv", index=False)
    return R, F, pd.DataFrame(out)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    R, F, O = run()
    print(R.round(3).to_string(), "\n", F.round(3).to_string(), "\n", O.round(3).to_string())
