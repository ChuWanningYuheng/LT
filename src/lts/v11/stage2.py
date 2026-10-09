"""Iteration 11, stage 2: interest and fictitious capital (pre_registration_v11.md, stage 2).

2.1 i vs r: JST countries (r from PWT 11, i from JST), US NFC (r of iteration 6, Moody's Aaa): outcomes 68-70.
2.2 fictitious capital: US NFC q and market value / capitalised profit around NBER peaks (71, 72); JST valuation
    (P/D) and credit/GDP before crises (73, 74).
2.3 descriptive: growth of NFC financial assets (transactions vs revaluation) against profits.

Usage: PYTHONPATH=src python -P -m lts.v11.stage2
Outputs: results/v11/s2_*.csv
"""
from __future__ import annotations

import zipfile

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint

from ..v6.series import ROOT, fa
from ..v8.rule import decide, mean_row, share_row
from ..v8.stage4 import US_PEAKS
from .stage1 import reg, wild_fast, widest

OUT = ROOT / "results" / "v11"
R11 = ROOT / "data" / "raw" / "v11"
Y0, Y1 = 1950, 2020


# ================================================================ loaders
def jst():
    d = pd.read_stata(R11 / "JSTdatasetR6.dta")
    d["year"] = d.year.astype(int)
    d = d.sort_values(["iso", "year"]).reset_index(drop=True)
    d["pi"] = d.groupby("iso").cpi.transform(lambda s: s / s.shift(1) - 1)
    return d


def pwt_r():
    p = pd.read_excel(ROOT / "data" / "raw" / "v9" / "pwt110.xlsx", sheet_name="Data",
                      usecols=["countrycode", "year", "labsh", "cgdpo", "cn", "delta"])
    p["r_pwt"] = (1 - p.labsh) * p.cgdpo / p.cn - p.delta
    return p.rename(columns={"countrycode": "iso"})[["iso", "year", "r_pwt"]]


def ameco_r():
    s = pd.read_csv(ROOT / "results" / "v6" / "series.csv", usecols=["geo", "variant", "year", "r"])
    return s[s.variant == "main"].rename(columns={"geo": "iso", "r": "r_ameco"})[["iso", "year", "r_ameco"]]


def fred(name):
    d = pd.read_csv(R11 / f"fred_{name}.csv")
    d.columns = ["date", "v"]
    d["v"] = pd.to_numeric(d.v, errors="coerce")
    d["year"] = pd.to_datetime(d.date).dt.year
    g = d.groupby("year").v
    a = g.mean()
    return a[g.count() == 12]                                       # complete years only


def z1():
    z = zipfile.ZipFile(ROOT / "data" / "raw" / "v6" / "z1" / "z1_csv_files.zip")
    num = lambda s: pd.to_numeric(s.replace("ND", np.nan), errors="coerce")   # noqa: E731
    b = pd.read_csv(z.open("csv/S11_1_b.csv")).set_index("date")
    r = pd.read_csv(z.open("csv/S11_1_r.csv")).set_index("date")
    ia = pd.read_csv(z.open("csv/S11_1_i_a.csv")).set_index("date")
    ia.index = ia.index.astype(int)
    q4 = b[b.index.str.endswith("Q4")]
    q4.index = q4.index.str[:4].astype(int)
    d = pd.DataFrame({"EQ": num(q4["LM103164105.Q"]), "DS": num(q4["FL104122005.Q"]),
                      "LN": num(q4["FL104135005.Q"]), "FA": num(q4["FL104090005.Q"]),
                      "NFA": num(q4["LM102010005.Q"])})
    yr = r.index.str[:4].astype(int)
    full = pd.Series(1, index=r.index).groupby(yr).count() == 4
    for col, code in (("FA_tr", "FU104090005.Q"), ("FA_rev", "FR104090005.Q")):
        s = num(r[code]).groupby(yr).sum(min_count=4)
        d[col] = s.where(full)
    d["NOS"] = num(ia["FU106402101.A"])
    d["K"] = fa(4, "FAAt401-A", 37)
    d["MV"] = d.EQ + d.DS + d.LN
    return d.sort_index()


def us_nfc_r():
    s = pd.read_csv(ROOT / "results" / "v6" / "series.csv", usecols=["geo", "variant", "year", "r"])
    return s[(s.geo == "USA_NFC") & (s.variant == "main")].set_index("year").r


# ================================================================ helpers
def mean_ci(values, groups, delta, direction, B=9999):
    """mean with the wider of t(G-1) and the wild cluster bootstrap-t CI (groups = clusters)"""
    v = np.asarray(values, float)
    ok = np.isfinite(v)
    v, g = v[ok], np.asarray(groups)[ok]
    b, se, lo_b, hi_b, p, G = wild_fast(np.ones((len(v), 1)), v, g, 0, B=B)
    lo, hi = widest(b, se, G, lo_b, hi_b)
    return dict(est=b, se=se, ci90_lo=lo, ci90_hi=hi, p_wild=p, G=G, n=len(v), mde=2.8 * se, theta0=0.0,
                delta=delta, direction=direction, label=decide(b, lo, hi, 0.0, delta, direction))


def abnormal(X, t, gbar, log=True):
    """[X(t-1) - X(t-4)] - 3 gbar (in logs if log) and the crash [X(t+1) - X(t-1)] - 2 gbar"""
    f = np.log if log else (lambda v: v)
    get = lambda y: X.get(y, np.nan)                            # noqa: E731
    pre = f(get(t - 1)) - f(get(t - 4)) - 3 * gbar
    crash = f(get(t + 1)) - f(get(t - 1)) - 2 * gbar
    return pre, crash


def peaks(y):
    """annual Bry-Boschan-type peak: growth > 0 in t and < 0 in t+1; returns the first recession years t+1"""
    g = np.log(y).diff()
    nxt = g.shift(-1)
    return set((g.index[(g > 0) & (nxt < 0)] + 1).tolist())


# ================================================================ 2.1 i vs r
def stage21():
    J = jst()
    P = pwt_r()
    A = ameco_r()
    d = J.merge(P, on=["iso", "year"], how="left").merge(A, on=["iso", "year"], how="left")
    d["i_lt_real"] = d.ltrate / 100 - d.pi
    d["i_st_real"] = d.stir / 100 - d.pi
    d["i_lt"] = d.ltrate / 100
    d["i_st"] = d.stir / 100
    w = d[d.year.between(Y0, Y1)].copy()
    rows, desc, tests = [], [], []
    # ---- 68: share of years with i_real < r
    sh = {}
    for spec, ic, rc in (("main: ltrate real vs r PWT", "i_lt_real", "r_pwt"), ("nominal ltrate vs r PWT", "i_lt", "r_pwt"),
                         ("stir real vs r PWT", "i_st_real", "r_pwt"), ("ltrate real vs r AMECO", "i_lt_real", "r_ameco")):
        s = w.dropna(subset=[ic, rc]).groupby("iso").apply(lambda g: (g[ic] < g[rc]).mean(), include_groups=False)
        n = w.dropna(subset=[ic, rc]).groupby("iso").size()
        sh[spec] = pd.DataFrame({"share": s, "years": n})
        r = mean_row(f"68 [{spec}]: mean over countries of the share of years with i < r", s.to_numpy(), 0.5, 0.15, ">")
        (rows if spec.startswith("main") else desc).append(r)
    pd.concat(sh, names=["spec", "iso"]).to_csv(OUT / "s2_68_by_country.csv")
    # US NFC (descriptive)
    rn = us_nfc_r()
    cpi = fred("CPIAUCNS")
    pi_us = cpi / cpi.shift(1) - 1
    aaa = fred("AAA") / 100
    us = pd.DataFrame({"r": rn, "i_aaa_real": aaa - pi_us, "i_aaa": aaa, "i_tb3_real": fred("TB3MS") / 100 - pi_us,
                       "i_tb3": fred("TB3MS") / 100}).loc[1947:2024]
    for ic in ("i_aaa_real", "i_aaa", "i_tb3_real"):
        u = us.dropna(subset=["r", ic])
        desc.append(share_row(f"68 [US NFC, {ic} vs r of iteration 6, {u.index.min()}-{u.index.max()}]: share of years "
                              f"with i < r (Wilson; years are not independent)", (u[ic] < u.r).to_numpy(), 0.5, 0.15, ">"))
    # ---- 69: Engle-Granger
    for iso, g in w.groupby("iso"):
        g = g.dropna(subset=["i_lt_real", "r_pwt"])
        if len(g) < 30:
            continue
        t, p, _ = coint(g.i_lt_real, g.r_pwt, trend="c", maxlag=4, autolag="aic")
        gn = w[w.iso == iso].dropna(subset=["i_lt", "r_pwt"])
        tn, pn, _ = coint(gn.i_lt, gn.r_pwt, trend="c", maxlag=4, autolag="aic")
        sp = adfuller(g.r_pwt - g.i_lt_real, maxlag=4, regression="c", autolag="AIC")[1]
        ar = adfuller(g.r_pwt, maxlag=4, regression="c", autolag="AIC")[1]
        ai = adfuller(g.i_lt_real, maxlag=4, regression="c", autolag="AIC")[1]
        tests.append(dict(iso=iso, n=len(g), eg_p=p, eg_coint=p < 0.10, eg_p_nominal=pn, adf_spread_p=sp,
                          adf_r_p=ar, adf_i_p=ai, years=f"{g.year.min()}-{g.year.max()}"))
    u = us.dropna(subset=["r", "i_aaa_real"])
    tu, pu, _ = coint(u.i_aaa_real, u.r, trend="c", maxlag=4, autolag="aic")
    tests.append(dict(iso="US NFC (Aaa real)", n=len(u), eg_p=pu, eg_coint=pu < 0.10,
                      adf_spread_p=adfuller(u.r - u.i_aaa_real, maxlag=4, regression="c", autolag="AIC")[1],
                      adf_r_p=adfuller(u.r, maxlag=4, regression="c", autolag="AIC")[1],
                      adf_i_p=adfuller(u.i_aaa_real, maxlag=4, regression="c", autolag="AIC")[1],
                      years=f"{u.index.min()}-{u.index.max()}"))
    T = pd.DataFrame(tests)
    T.to_csv(OUT / "s2_69_tests.csv", index=False)
    Tj = T[~T.iso.str.startswith("US NFC")]
    rows.append(share_row("69: share of JST countries where i_real (ltrate) and r (PWT) are cointegrated "
                          "(Engle-Granger, p < 0.10)", Tj.eg_coint.to_numpy(), 0.5, 0.15, "<"))
    desc.append(share_row("69 [nominal ltrate]", (Tj.eg_p_nominal < 0.10).to_numpy(), 0.5, 0.15, "<"))
    desc.append(share_row("69 [ADF on r - i_real: stationary spread, p < 0.10]", (Tj.adf_spread_p < 0.10).to_numpy(),
                          0.5, 0.15, "<"))
    desc.append(dict(outcome="69 [ADF: share stationary r / share stationary i_real, p < 0.10]",
                     est=float((Tj.adf_r_p < 0.10).mean()), est2=float((Tj.adf_i_p < 0.10).mean()), n=len(Tj)))
    # ---- 70: spread at recession onset
    rec = []
    for iso, g in d.groupby("iso"):
        g = g.set_index("year")
        rs = peaks(g.rgdpbarro.dropna())
        rec += [(iso, y) for y in rs]
    R = set(rec)
    d["rec"] = [float((i, y) in R) for i, y in zip(d.iso, d.year)]
    d["crisis"] = d.crisisJST.astype(float)
    d = d.sort_values(["iso", "year"])
    for ic in ("i_st", "i_lt", "i_st_real", "i_lt_real"):
        d[f"sp_{ic}"] = d.r_pwt - d[ic]
        d[f"dsp_{ic}"] = d.groupby("iso")[f"sp_{ic}"].diff()            # years are consecutive in JST
    w = d[d.year.between(Y0 + 1, Y1)]
    for spec, ic, ev in (("main: stir nominal, recession onset", "i_st", "rec"), ("ltrate nominal", "i_lt", "rec"),
                         ("stir real", "i_st_real", "rec"), ("ltrate real", "i_lt_real", "rec"),
                         ("stir nominal, JST crisis years", "i_st", "crisis")):
        x = w.dropna(subset=[f"dsp_{ic}", ev])
        r = reg(x, f"dsp_{ic}", [ev], 0, fe=("iso",), cl="iso")
        r = dict(outcome=f"70 [{spec}]: change of r - i in the event year minus other years", **r, events=int(x[ev].sum()),
                 theta0=0.0, delta=0.005, direction="<")
        r["label"] = decide(r["est"], r["ci90_lo"], r["ci90_hi"], 0.0, 0.005, "<")
        (rows if spec.startswith("main") else desc).append(r)
    # US NFC (descriptive, HAC lag 2)
    us["rec"] = [float(y in {p + 1 for p in US_PEAKS}) for y in us.index]
    for ic in ("i_tb3", "i_aaa"):
        x = us.assign(dsp=(us.r - us[ic]).diff()).dropna(subset=["dsp"])
        m = sm.OLS(x.dsp, sm.add_constant(x.rec)).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
        lo, hi = m.conf_int(0.10).loc["rec"]
        desc.append(dict(outcome=f"70 [US NFC, {ic}, recession years = NBER peak + 1, HAC]", est=m.params.rec,
                         ci90_lo=lo, ci90_hi=hi, n=len(x), events=int(x.rec.sum()), theta0=0.0, delta=0.005,
                         direction="<", label=decide(m.params.rec, lo, hi, 0, 0.005, "<")))
    pd.DataFrame(sorted(rec), columns=["iso", "year"]).to_csv(OUT / "s2_70_recession_years.csv", index=False)
    us.to_csv(OUT / "s2_us_rates.csv")
    # long-run descriptive (1870-2020): returns
    lr = []
    for per, (a, b) in (("1870-2020", (1870, 2020)), ("1950-2020", (1950, 2020))):
        x = J[J.year.between(a, b)].dropna(subset=["risky_tr", "safe_tr", "pi"])
        g = x.groupby("iso")
        real = lambda c: (1 + x[c]) / (1 + x.pi) - 1                  # noqa: E731
        x = x.assign(risky_real=real("risky_tr"), safe_real=real("safe_tr"))
        g = x.groupby("iso")
        lr.append(pd.DataFrame({"period": per, "risky_real": g.risky_real.mean(), "safe_real": g.safe_real.mean(),
                                "share_years_risky_gt_safe": g.apply(lambda v: (v.risky_tr > v.safe_tr).mean(),
                                                                     include_groups=False), "n": g.size()}))
    pd.concat(lr).to_csv(OUT / "s2_longrun_returns.csv")
    return rows, desc


# ================================================================ 2.2 fictitious capital
def stage22():
    rows, desc, ev = [], [], []
    Z = z1()
    aaa, gs10 = fred("AAA") / 100, fred("GS10") / 100
    X = {"q (MV / K, BEA fixed capital)": Z.MV / Z.K, "q2 (MV / nonfinancial assets, Z.1)": Z.MV / Z.NFA,
         "q equity only (EQ / K)": Z.EQ / Z.K,
         "F (MV / (NOS / Aaa))": Z.MV * aaa.reindex(Z.index) / Z.NOS,
         "F (MV / (NOS / GS10))": Z.MV * gs10.reindex(Z.index) / Z.NOS}
    S = pd.DataFrame(X)
    S.to_csv(OUT / "s2_us_fictitious_series.csv")
    for name, x in X.items():
        x = x.loc[1945:2024].dropna()
        gbar = float(np.log(x).diff().loc[1946:2024].mean())
        for k, P in enumerate(US_PEAKS):
            pre, crash = abnormal(x, P, gbar)
            ev.append(dict(series=name, peak=P, pre=pre, crash=crash, gbar=gbar))
    E = pd.DataFrame(ev)
    E.to_csv(OUT / "s2_us_events.csv", index=False)
    for name in X:
        e = E[E.series == name]
        r = mean_ci(e.pre, e.peak, 0.05, ">")
        tag = "71" if name.startswith("q (") else "72" if name.startswith("F (MV / (NOS / Aaa") else None
        out = dict(outcome=f"{tag or '71/72 variant'} [{name}]: mean abnormal 3-year rise before NBER peaks", **r)
        (rows if tag else desc).append(out)
        c = mean_ci(e.crash, e.peak, 0.05, "<")
        desc.append(dict(outcome=f"[{name}]: mean abnormal change peak-1 -> peak+1 (descriptive)", **c))
    # ---- JST
    J = jst()
    J["lnPD"] = np.log(1 / J.eq_dp.where(J.eq_dp > 0))
    J["dlnP_real"] = np.log1p(J.eq_capgain) - np.log1p(J.pi)
    J["LY"] = J.tloans / J.gdp
    jev = []
    for iso, g in J.groupby("iso"):
        g = g.set_index("year")
        gpd = float(g.lnPD.diff().mean())
        gly = float(g.LY.diff().mean())
        gp = float(g.dlnP_real.mean())
        for t in g.index[g.crisisJST == 1]:
            pd_pre, pd_crash = abnormal(g.lnPD, t, gpd, log=False)
            ly_pre, ly_crash = abnormal(g.LY, t, gly, log=False)
            dp = g.dlnP_real
            p_pre = dp.reindex([t - 3, t - 2, t - 1]).sum(min_count=3) - 3 * gp
            p_crash = dp.reindex([t, t + 1]).sum(min_count=2) - 2 * gp
            jev.append(dict(iso=iso, year=int(t), pd_pre=pd_pre, pd_crash=pd_crash, ly_pre=ly_pre, ly_crash=ly_crash,
                            p_pre=p_pre, p_crash=p_crash))
    JE = pd.DataFrame(jev)
    JE.to_csv(OUT / "s2_jst_events.csv", index=False)
    war = JE.year.between(1914, 1918) | JE.year.between(1939, 1945)
    rows.append(dict(outcome="73: abnormal 3-year rise of ln(P/D) before JST crises (clusters: countries)",
                     **mean_ci(JE.pd_pre, JE.iso, 0.05, ">")))
    rows.append(dict(outcome="74: abnormal 3-year rise of credit/GDP before JST crises (clusters: countries)",
                     **mean_ci(JE.ly_pre, JE.iso, 0.02, ">")))
    for nm, m in (("after 1945", JE.year > 1945), ("without war years", ~war)):
        desc.append(dict(outcome=f"73 [{nm}]", **mean_ci(JE.pd_pre[m], JE.iso[m], 0.05, ">")))
        desc.append(dict(outcome=f"74 [{nm}]", **mean_ci(JE.ly_pre[m], JE.iso[m], 0.02, ">")))
    desc.append(dict(outcome="73 [real equity price instead of P/D]", **mean_ci(JE.p_pre, JE.iso, 0.05, ">")))
    desc.append(dict(outcome="73 [crash: ln(P/D) t-1 -> t+1, descriptive]", **mean_ci(JE.pd_crash, JE.iso, 0.05, "<")))
    desc.append(dict(outcome="73 [crash: real equity price t-1 -> t+1, descriptive]",
                     **mean_ci(JE.p_crash, JE.iso, 0.05, "<")))
    desc.append(dict(outcome="74 [credit/GDP t-1 -> t+1, descriptive]", **mean_ci(JE.ly_crash, JE.iso, 0.02, "<")))
    return rows, desc


# ================================================================ 2.3 financial assets vs profits (descriptive)
def stage23():
    Z = z1().loc[1945:2025]
    d = Z[["FA", "FA_tr", "FA_rev", "NOS", "K", "MV"]].copy()
    d["dFA"] = d.FA.diff()
    d["other"] = d.dFA - d.FA_tr - d.FA_rev
    d["FA_K"] = d.FA / d.K
    out = []
    for a, b in ((1946, 2025), (1946, 1959), (1960, 1969), (1970, 1979), (1980, 1989), (1990, 1999), (2000, 2009),
                 (2010, 2019), (2020, 2025)):
        x = d.loc[a:b]
        tot = x.dFA.sum()
        nos = Z.NOS.loc[a - 1:b].dropna()
        out.append(dict(period=f"{a}-{b}", dFA=tot, share_transactions=x.FA_tr.sum() / tot,
                        share_revaluation=x.FA_rev.sum() / tot, share_other=x.other.sum() / tot,
                        dlnFA_per_year=(np.log(Z.FA.loc[b]) - np.log(Z.FA.loc[a - 1])) / (b - a + 1),
                        dlnNOS_per_year=(np.log(nos.iloc[-1]) - np.log(nos.iloc[0])) / (nos.index[-1] - nos.index[0])
                        if len(nos) > 1 else np.nan,
                        FA_K_start=Z.FA.loc[a - 1] / Z.K.loc[a - 1] if np.isfinite(Z.K.get(a - 1, np.nan)) else np.nan,
                        FA_K_end=Z.FA.loc[b] / Z.K.loc[b] if np.isfinite(Z.K.get(b, np.nan)) else np.nan))
    D = pd.DataFrame(out)
    D.to_csv(OUT / "s2_23_decomposition.csv", index=False)
    d.to_csv(OUT / "s2_23_series.csv")
    # gap before peaks and revaluation in recessions
    gap = np.log(Z.FA).diff(3) - np.log(Z.NOS).diff(3)
    lnFA_NOS = np.log(Z.FA / Z.NOS)
    gbar = float(lnFA_NOS.diff().loc[1946:2024].mean())
    ev = [dict(peak=P, pre=abnormal(lnFA_NOS.loc[:2024], P, gbar)[0], crash=abnormal(lnFA_NOS.loc[:2024], P, gbar)[1],
               gap_at_peak_minus_1=gap.get(P - 1, np.nan)) for P in US_PEAKS]
    E = pd.DataFrame(ev)
    E.to_csv(OUT / "s2_23_events.csv", index=False)
    rv = (Z.FA_rev / Z.FA.shift(1)).loc[1946:2025]
    rec = {p + 1 for p in US_PEAKS}
    rn = us_nfc_r()
    t = pd.DataFrame({"rev_share": rv, "rec": [y in rec for y in rv.index], "dr": rn.diff().reindex(rv.index)})
    summ = [dict(item="abnormal rise of ln(FA/NOS) before peaks (descriptive)", **mean_ci(E.pre, E.peak, 0.05, ">")),
            dict(item="abnormal change of ln(FA/NOS) peak-1 -> peak+1", **mean_ci(E.crash, E.peak, 0.05, "<")),
            dict(item="revaluation / FA(t-1): recession years (mean)", est=float(t[t.rec].rev_share.mean()),
                 n=int(t.rec.sum())),
            dict(item="revaluation / FA(t-1): other years (mean)", est=float(t[~t.rec].rev_share.mean()),
                 n=int((~t.rec).sum())),
            dict(item="years with revaluation < 0 and r NFC falling / revaluation < 0",
                 est=int(((t.rev_share < 0) & (t.dr < 0)).sum()), n=int((t.rev_share < 0).sum())),
            dict(item="years with revaluation >= 0 and r NFC falling / revaluation >= 0",
                 est=int(((t.rev_share >= 0) & (t.dr < 0)).sum()), n=int((t.rev_share >= 0).sum()))]
    pd.DataFrame(summ).to_csv(OUT / "s2_23_summary.csv", index=False)
    return D, pd.DataFrame(summ)


def run():
    r1, d1 = stage21()
    r2, d2 = stage22()
    O = pd.DataFrame(r1 + r2)
    O.to_csv(OUT / "s2_outcomes.csv", index=False)
    Dd = pd.DataFrame(d1 + d2)
    Dd.to_csv(OUT / "s2_descriptive.csv", index=False)
    D, S = stage23()
    pd.set_option("display.width", 250)
    cols = ["outcome", "est", "ci90_lo", "ci90_hi", "n", "label"]
    print(O[[c for c in cols if c in O]].round(4).to_string(), "\n")
    print(Dd[[c for c in cols if c in Dd]].round(4).to_string(), "\n")
    print(D.round(3).to_string(), "\n", S.round(4).to_string())


if __name__ == "__main__":
    run()
