"""Iteration 8, stage 5: Marx vs Kalecki - what leads, profitability or investment (US quarterly NIPA).

Series (quarterly, SAAR, deflated by the GDP deflator A191RD): corporate profits with IVA and CCAdj of domestic
industries (A445RC; variant: NFC net operating surplus W326RC), private nonresidential fixed investment (A008RC),
government net lending (AD01RC, as share of GDP A191RC), NFC profit rate W326RC / K (BEA FA 4.1 line 37,
log-linearly interpolated within the year).
Local projections h = 1..12 and out-of-sample Clark-West tests (expanding window from 1985Q1, h = 1, 4, 8; 4 lags).
Outputs: results/v8/s5_*.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from ..v6.series import fa
from .stage2 import OUT, R8


def nipa_q(codes):
    d = pd.read_csv(R8 / "NipaDataQ.txt", dtype=str)
    d.columns = ["code", "period", "v"]
    d = d[d.code.isin(codes)]
    d["v"] = pd.to_numeric(d.v.str.replace(",", ""), errors="coerce")
    d["q"] = pd.PeriodIndex(d.period.str.replace("Q", "-Q"), freq="Q")
    return d.pivot(index="q", columns="code", values="v").sort_index()


def data():
    n = nipa_q(["A445RC", "W326RC", "A008RC", "AD01RC", "A191RC", "A191RD"])
    K = fa(4, "FAAt401-A", 37)                                       # end-of-year, millions
    kq = pd.Series(np.nan, index=n.index)
    for y in K.index:
        if pd.Period(f"{y}Q4", "Q") in kq.index:
            kq[pd.Period(f"{y}Q4", "Q")] = K[y]
    kq = np.exp(np.log(kq).interpolate(limit_area="inside"))
    d = pd.DataFrame(index=n.index)
    defl = n.A191RD / 100
    d["lprof"] = np.log(n.A445RC / defl)
    d["lnos"] = np.log(n.W326RC / defl)
    d["linv"] = np.log(n.A008RC / defl)
    d["def"] = -n.AD01RC / n.A191RC                                  # deficit share of GDP
    d["r"] = n.W326RC / kq
    return d.dropna(subset=["linv", "lprof", "def"])


def lp(d, y, shock, ctrl_vars, H=12, lags=4):
    rows = []
    for h in range(1, H + 1):
        e = pd.DataFrame({"dy": d[y].shift(-h) - d[y].shift(1), "s": d[shock]})
        for v in ctrl_vars:
            for L in range(1, lags + 1):
                e[f"{v}_{L}"] = d[v].shift(L)
        e = e.dropna()
        f = sm.OLS(e.dy, sm.add_constant(e.drop(columns="dy"))).fit(cov_type="HAC", cov_kwds={"maxlags": h + 1})
        rows.append(dict(y=y, shock=shock, h=h, coef=f.params.s, se=f.bse.s, p=f.pvalues.s, n=len(e)))
    return rows


def oos(d, y, extra, h, lags=4, start="1985Q1"):
    """Direct h-step forecast of y_{t+h} - y_t. Restricted: lags of dy. Unrestricted: + lags of extra. Clark-West."""
    e = pd.DataFrame({"target": d[y].shift(-h) - d[y]})
    dy = d[y].diff()
    for L in range(0, lags):
        e[f"dy_{L}"] = dy.shift(L)
        for v in extra:
            e[f"{v}_{L}"] = d[v].shift(L) if v not in ("linv", "lprof", "lnos") else d[v].diff().shift(L)
    e = e.dropna()
    rcols = [c for c in e.columns if c.startswith("dy_")]
    ucols = [c for c in e.columns if c != "target"]
    f_r, f_u, act = [], [], []
    for t in e.index[e.index >= pd.Period(start, "Q")]:
        train = e[e.index <= t - h]
        if len(train) < 40:
            continue
        xr = sm.add_constant(train[rcols])
        xu = sm.add_constant(train[ucols])
        br = np.linalg.lstsq(xr, train.target, rcond=None)[0]
        bu = np.linalg.lstsq(xu, train.target, rcond=None)[0]
        f_r.append(float(np.r_[1, e.loc[t, rcols]] @ br))
        f_u.append(float(np.r_[1, e.loc[t, ucols]] @ bu))
        act.append(float(e.loc[t, "target"]))
    f_r, f_u, act = map(np.array, (f_r, f_u, act))
    er, eu = act - f_r, act - f_u
    fadj = er ** 2 - (eu ** 2 - (f_r - f_u) ** 2)
    se = np.sqrt(sm.OLS(fadj, np.ones_like(fadj)).fit(cov_type="HAC", cov_kwds={"maxlags": h}).cov_params()[0, 0])
    cw = fadj.mean() / se
    return dict(target=y, extra="+".join(extra), h=h, n_fc=len(act), rmse_ratio=float(np.sqrt((eu ** 2).mean() / (er ** 2).mean())),
                cw_stat=float(cw), cw_p=float(stats.norm.sf(cw)))


def episode_2020(d):
    h = pd.read_csv(R8 / "fred_HOANBS.csv")
    h.columns = ["date", "hours"]
    h["q"] = pd.PeriodIndex(pd.to_datetime(h.date), freq="Q")
    d = d.join(h.set_index("q").hours)
    s = d.loc[pd.Period("2019Q1", "Q"):pd.Period("2022Q4", "Q")].copy()
    return s.assign(prof_index=np.exp(s.lprof - s.lprof.iloc[0]) * 100, inv_index=np.exp(s.linv - s.linv.iloc[0]) * 100)


if __name__ == "__main__":
    d = data()
    print(d.index.min(), d.index.max(), len(d), flush=True)
    rows = lp(d, "linv", "r", ["r", "linv"]) + lp(d, "lprof", "linv", ["lprof", "linv", "def"]) + \
        lp(d, "lprof", "def", ["lprof", "linv", "def"])
    L = pd.DataFrame(rows)
    L.to_csv(OUT / "s5_lp.csv", index=False)
    print(L[L.h.isin([1, 4, 8, 12])].round(4).to_string(), flush=True)
    res = []
    for h in (1, 4, 8):
        res.append(oos(d, "linv", ["r"], h))                       # Marx: profitability -> investment
        res.append(oos(d, "lprof", ["linv", "def"], h))            # Kalecki: investment, deficit -> profits
        res.append(oos(d, "lnos", ["linv", "def"], h))             # variant: NFC NOS
    O = pd.DataFrame(res)
    O.to_csv(OUT / "s5_oos.csv", index=False)
    print(O.round(4).to_string(), flush=True)
    m4 = O[(O.h == 4) & (O.target == "linv")].cw_p.iloc[0] < 0.05
    k4 = O[(O.h == 4) & (O.target == "lprof")].cw_p.iloc[0] < 0.05
    verdict = {(True, False): "прибыль ведёт (Маркс)", (False, True): "инвестиции ведут (Калецки)",
               (True, True): "двусторонне", (False, False): "неинформативно"}[(m4, k4)]
    pd.Series({"verdict_h4": verdict}).to_csv(OUT / "s5_verdict.csv")
    print("verdict h=4:", verdict)
    episode_2020(d).to_csv(OUT / "s5_2020.csv")
