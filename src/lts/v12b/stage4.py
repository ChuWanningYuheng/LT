"""Iteration 12b, stage 4: the mass of profit before crises (Grossman; Carchedi & Roberts)
(pre_registration_v12b.md, stage 4; outcomes 104-109).

US: NFC profit (iteration 6, main), NBER annual peaks (iteration 8). JST: 18 countries, AMECO profit
(lts.v6.series derive / finish, main), peaks of iteration 11 (growth > 0 in t and < 0 in t + 1; peak = t).
g_t = d ln real profit mass (t - 1 -> t); 104/107: g at the peak year t minus other years;
105/108: [ln r(t-1) - ln r(t-4)] minus its mean in other years; 106/109: [ln r(t+3) - ln r(t)] minus mean.
Decomposition (descriptive): d ln Pi_real = d ln(Pi/Y) + d ln u + d ln Y*_real; Y* = Y / u.

Usage: PYTHONPATH=src python -P -m lts.v12b.stage4
Outputs: results/v12b/s4_*.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from ..v6.series import ROOT, ameco_country, ameco_raw, derive, finish, nipa
from ..v8.rule import decide
from ..v11.stage1 import reg
from .stage1 import US_PEAKS

OUT = ROOT / "results" / "v12b"
JST_ISO = ['AUS', 'BEL', 'CAN', 'CHE', 'DEU', 'DNK', 'ESP', 'FIN', 'FRA', 'GBR', 'IRL', 'ITA', 'JPN', 'NLD', 'NOR', 'PRT',
           'SWE', 'USA']


def event_reg(y, ev):
    """y_t on an event dummy: coefficient = mean(y | event) - mean(y | other); CI = wider of HAC(4) and OLS t."""
    d = pd.DataFrame({"y": y, "ev": ev}).dropna()
    X = sm.add_constant(d.ev.to_numpy(float))
    h = sm.OLS(d.y.to_numpy(), X).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
    o = sm.OLS(d.y.to_numpy(), X).fit()
    b = float(o.params[1])
    w = max(stats.norm.ppf(0.95) * h.bse[1], stats.t.ppf(0.95, len(d) - 2) * o.bse[1])
    return dict(est=b, ci90_lo=b - w, ci90_hi=b + w, n=len(d), events=int(d.ev.sum()))


def windows(df, grp=None):
    """adds g (d ln real mass), pre_r = ln r(t-1) - ln r(t-4), post_r = ln r(t+3) - ln r(t)"""
    def f(g):
        g = g.sort_values("year").set_index("year")
        lr = np.log(g.r.where(g.r > 0))
        lp = np.log(g.Pi_real.where(g.Pi_real > 0))
        g["g"] = lp.diff()
        g["pre_r"] = lr.shift(1) - lr.shift(4)
        g["post_r"] = lr.shift(-3) - lr
        g["dshare"] = np.log(g.Pi / g.Y).diff()
        g["dY_real"] = np.log(g.Y_real).diff()
        g["du"] = np.log(g.u).diff() if "u" in g else np.nan
        g["dYstar"] = g.dY_real - g.du
        g["dws"] = (g.W / g.Y).diff()
        g["dI"] = np.log(g.I_real).diff() if "I_real" in g else np.nan
        return g.reset_index()
    if grp is None:
        return f(df)
    return pd.concat([f(g) for _, g in df.groupby(grp)], ignore_index=True)


def us_data(deflator="gdp"):
    s = pd.read_csv(ROOT / "results" / "v6" / "series.csv")
    d = s[(s.geo == "USA_NFC") & (s.variant == "main")].copy()
    n = nipa(["A191RD"])["A191RD"]
    if deflator == "gdp":
        p = d.year.map(n)
    elif deflator == "cpi":
        c = pd.read_csv(ROOT / "data" / "raw" / "v11" / "fred_CPIAUCNS.csv")
        c.columns = ["date", "v"]
        c["year"] = pd.to_datetime(c.date).dt.year
        p = d.year.map(c.groupby("year").v.mean())
    else:
        p = d.P_Y
    d["Pi"] = d.PI
    d["Pi_real"] = d.PI / p
    d["Y_real"] = d.Y / p
    d["u"] = 1 + d.gap / 100
    return d[d.year.between(1947, 2024)]


def jst_data():
    J = pd.read_stata(ROOT / "data" / "raw" / "v11" / "JSTdatasetR6.dta")
    raw = ameco_raw()
    rows = []
    for iso in JST_ISO:
        src = "D_W" if iso == "DEU" else iso
        base = ameco_country(raw, src)
        if iso == "DEU":
            de = ameco_country(raw, "DEU")
            base = pd.concat([base.loc[:1990], de.loc[1991:]])
        try:
            d = finish(derive(base, 1.0))
        except Exception as exc:                                     # noqa: BLE001
            print("ameco fail", iso, exc)
            continue
        d = d.reset_index().rename(columns={"index": "year"})
        p = d.UVGD / d.OVGD
        d["iso"] = iso
        d["Pi"] = d.PI
        d["Pi_real"] = d.PI / p
        d["Y_real"] = d.Y / p
        d["u"] = 1 + d.AVGDGP / 100
        d["I_real"] = d.OIGT
        cpi = J[J.iso == iso].set_index("year").cpi
        d["Pi_real_cpi"] = d.PI / d.year.map(cpi)
        rows.append(d[["iso", "year", "Pi", "Pi_real", "Pi_real_cpi", "Y", "Y_real", "W", "r", "u", "I_real"]])
    A = pd.concat(rows, ignore_index=True)
    pk = []
    for iso, g in J.groupby("iso"):
        y = g.set_index("year").rgdpbarro.dropna()
        gr = np.log(y).diff()
        nxt = gr.shift(-1)
        pk += [(iso, int(t)) for t in gr.index[(gr > 0) & (nxt < 0)]]
    fc = J[J.crisisJST == 1][["iso", "year"]]
    A["peak"] = [float((i, int(t)) in set(pk)) for i, t in zip(A.iso, A.year)]
    A["fincrisis"] = [float((i, int(t)) in set(zip(fc.iso, fc.year.astype(int)))) for i, t in zip(A.iso, A.year)]
    A = A[A.year <= J.year.max()]
    return A


def profile(d, evcol, cols, k_range=range(-4, 4), grp=None, B=2000, seed=104):
    """mean of each col at event time k, CI by bootstrap over events (each event = one cluster)."""
    ev = d[d[evcol] == 1][[grp, "year"] if grp else ["year"]].to_numpy()
    key = d.set_index([grp, "year"]) if grp else d.set_index("year")
    mats = {c: [] for c in cols}
    for e_ in ev:
        for c in cols:
            vals = []
            for k in k_range:
                idx = (e_[0], int(e_[1]) + k) if grp else int(e_[0]) + k
                vals.append(key[c].get(idx, np.nan))
            mats[c].append(vals)
    rng = np.random.default_rng(seed)
    rows = []
    for c in cols:
        M = np.array(mats[c], float)
        bs = np.array([np.nanmean(M[rng.integers(0, len(M), len(M))], 0) for _ in range(B)])
        for j, k in enumerate(k_range):
            rows.append(dict(var=c, k=k, mean=float(np.nanmean(M[:, j])), lo=float(np.nanquantile(bs[:, j], 0.05)),
                             hi=float(np.nanquantile(bs[:, j], 0.95)), n=int(np.isfinite(M[:, j]).sum())))
    return pd.DataFrame(rows)


def run():
    out, desc = [], []
    # ---- US
    U = windows(us_data("gdp"))
    U["peak"] = U.year.isin(US_PEAKS).astype(float)
    spec = [("104: US NFC, d ln real profit mass in the peak year minus other years", "g", 0.02, "<"),
            ("105: US NFC, ln r(t-1) - ln r(t-4) at peaks minus other years", "pre_r", 0.02, "<"),
            ("106: US NFC, ln r(t+3) - ln r(t) at peaks minus other years", "post_r", 0.02, ">")]
    for nm, col, de, dr in spec:
        r = event_reg(U[col], U.peak)
        out.append(dict(outcome=nm, **r, theta0=0.0, delta=de, direction=dr,
                        label=decide(r["est"], r["ci90_lo"], r["ci90_hi"], 0, de, dr)))
    for defl in ("cpi", "own"):
        V = windows(us_data(defl))
        V["peak"] = V.year.isin(US_PEAKS).astype(float)
        r = event_reg(V.g, V.peak)
        desc.append(dict(item=f"104 variant: deflator {defl}", **r, label=decide(r["est"], r["ci90_lo"], r["ci90_hi"], 0, 0.02, "<")))
    U["peak_m1"] = U.year.isin([p - 1 for p in US_PEAKS]).astype(float)
    r = event_reg(U.g, U.peak_m1)
    desc.append(dict(item="US: d ln real mass in the year before the peak year (t-2 -> t-1) minus other years", **r))
    for col in ("dshare", "dY_real", "du", "dYstar", "dws"):
        r = event_reg(U[col], U.peak)
        desc.append(dict(item=f"US peak year: {col} minus other years", **r))
    # hours (stage 1)
    S1 = pd.read_csv(OUT / "s1_us_series.csv")
    m1 = S1[S1.variant == "main"].set_index("year")
    gh = np.log(m1.S_h).diff()
    r = event_reg(gh, pd.Series(gh.index.isin(US_PEAKS).astype(float), index=gh.index))
    desc.append(dict(item="US: d ln S_h (surplus labour, stage 1) in the peak year minus other years", **r))
    PU = profile(U, "peak", ["g", "dshare", "dY_real", "du", "dYstar", "dws"])
    PU.to_csv(OUT / "s4_us_profile.csv", index=False)
    U.to_csv(OUT / "s4_us_panel.csv", index=False)
    # ---- JST
    A = jst_data()
    A = windows(A, "iso")
    A["yr"] = A.year.astype(str)
    A.to_csv(OUT / "s4_jst_panel.csv", index=False)
    for nm, col, de, dr in (("107: JST (AMECO profit), d ln real profit mass in the peak year minus other years", "g", 0.02, "<"),
                            ("108: JST, ln r(t-1) - ln r(t-4) at peaks minus other years", "pre_r", 0.02, "<"),
                            ("109: JST, ln r(t+3) - ln r(t) at peaks minus other years", "post_r", 0.02, ">")):
        x = A.dropna(subset=[col, "peak"])
        r = reg(x, col, ["peak"], 0, fe=("iso",), cl="iso")
        out.append(dict(outcome=nm, **r, events=int(x.peak.sum()), theta0=0.0, delta=de, direction=dr,
                        label=decide(r["est"], r["ci90_lo"], r["ci90_hi"], 0, de, dr)))
    for ev in ("fincrisis",):
        x = A.dropna(subset=["g", ev])
        r = reg(x, "g", [ev], 0, fe=("iso",), cl="iso")
        desc.append(dict(item=f"107 variant: JST financial crisis years", **r, events=int(x[ev].sum())))
    A2 = A.copy()
    A2["Pi_real"] = A2.Pi_real_cpi
    A2 = windows(A2.drop(columns=["g", "pre_r", "post_r", "dshare", "dY_real", "du", "dYstar", "dws", "dI"]), "iso")
    x = A2.dropna(subset=["g", "peak"])
    r = reg(x, "g", ["peak"], 0, fe=("iso",), cl="iso")
    desc.append(dict(item="107 variant: CPI deflator", **r))
    x = A.dropna(subset=["g", "peak"])
    r = reg(x, "g", ["peak"], 0, fe=("iso", "yr"), cl="iso")
    desc.append(dict(item="107 variant: with year effects", **r))
    for col in ("dshare", "dY_real", "du", "dYstar", "dws", "dI"):
        x = A.dropna(subset=[col, "peak"])
        r = reg(x, col, ["peak"], 0, fe=("iso",), cl="iso")
        desc.append(dict(item=f"JST peak year: {col} minus other years", **r))
    PJ = profile(A, "peak", ["g", "dshare", "dY_real", "du", "dYstar", "dws", "dI"], grp="iso")
    PJ.to_csv(OUT / "s4_jst_profile.csv", index=False)
    O = pd.DataFrame(out)
    O.to_csv(OUT / "s4_outcomes.csv", index=False)
    D = pd.DataFrame(desc)
    D.to_csv(OUT / "s4_descriptive.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 90)
    print(O.round(4).to_string())
    print(D.round(4).to_string())
    print(PU.round(4).to_string())
    print(PJ.round(4).to_string())


if __name__ == "__main__":
    run()
