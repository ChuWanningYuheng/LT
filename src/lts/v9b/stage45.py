"""Iteration 9b, stages 4-5: labour power as a commodity (broad slack, wild bootstrap for outcome 27, LCI D11,
terms-of-trade control) and minor fixes (outcomes 20, 34, 35; b estimators) (pre_registration_v9b.md, stages 4-5).

Outputs: results/v9b/s4_*.csv, s5_*.csv
"""
from __future__ import annotations

import itertools
import json
import sys
import zipfile

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import optimize, stats

from ..v8.rule import decide
from ..v9 import stage2 as v9s2
from .stage1 import OUT, R, nipa

R9 = OUT.parents[1] / "results" / "v9"
v9s2.AM = v9s2.AM[v9s2.AM.year <= 2024].copy()                  # journal 2: no AMECO forecast years
WEBB = np.array([-np.sqrt(1.5), -1, -np.sqrt(0.5), np.sqrt(0.5), 1, np.sqrt(1.5)])


# ---------------------------------------------------------------- wild cluster bootstrap-t (unrestricted) CI and p
def wild_t(X, y, groups, j, B=9999, seed=7):
    """X: demeaned regressors (n x k), y: demeaned outcome; returns b, se (CR1), 90% bootstrap-t CI, symmetric p (null)."""
    gi = pd.factorize(groups)[0]
    G = gi.max() + 1
    XtX = np.linalg.inv(X.T @ X)

    def fit(yy):
        b = XtX @ X.T @ yy
        e = yy - X @ b
        meat = np.zeros((X.shape[1],) * 2)
        for g in range(G):
            s = X[gi == g].T @ e[gi == g]
            meat += np.outer(s, s)
        V = XtX @ meat @ XtX * G / (G - 1)
        return b, np.sqrt(np.diag(V))
    b, se = fit(y)
    yh, e = X @ b, y - X @ b
    rng = np.random.default_rng(seed)
    ts = []
    for _ in range(B):
        w = WEBB[rng.integers(0, 6, G)][gi]
        bb, ss = fit(yh + e * w)
        ts.append((bb[j] - b[j]) / ss[j])
    ts = np.array(ts)
    lo, hi = b[j] - np.quantile(ts, 0.95) * se[j], b[j] - np.quantile(ts, 0.05) * se[j]
    p = float(np.mean(np.abs(ts) >= abs(b[j] / se[j])))
    return float(b[j]), float(se[j]), float(lo), float(hi), p, G


def widest(b, se, G, lo_b, hi_b, delta, direction):
    tq = stats.t.ppf(0.95, G - 1)
    lo, hi = min(b - tq * se, lo_b), max(b + tq * se, hi_b)
    return lo, hi, decide(b, lo, hi, 0, delta, direction)


# ---------------------------------------------------------------- 4.1 broad slack
def slack_annual():
    s = v9s2.es_json(OUT.parents[1] / "data" / "raw" / "v9" / "es_slack_q.json")
    s["year"] = s.time.str[:4].astype(int)
    a = s.groupby(["geo", "year"]).value.mean().reset_index()
    return a


def ecm_panel(shock="slack"):
    a = v9s2.AM.copy().sort_values(["geo", "year"])
    a["lw"] = np.log(a.HWCDW / a.NLHA / a.PCPH)
    a["ly"] = np.log(a.RVGDE / a.NLHA)
    sl = slack_annual()
    sl = sl[sl.year <= 2024]
    inv = {v: k for k, v in v9s2.AM_TO2.items() if v}
    sl["geo"] = sl.geo.map(lambda g: inv.get(g, inv.get({"EL": "GR", "UK": "GB"}.get(g, g))))
    sl = sl.dropna(subset=["geo"]).rename(columns={"value": "slack"})
    a = a.merge(sl, on=["geo", "year"], how="inner")
    a["u"] = a.ZUTN
    a = a.replace([np.inf, -np.inf], np.nan).dropna(subset=["lw", "ly", "slack", "u"])
    g = a.groupby("geo")
    a["dlw"], a["dly"] = g.lw.diff(), g.ly.diff()
    a["gap1"] = g.lw.shift() - g.ly.shift()
    a["s1"] = g.slack.shift()
    a["u1"] = g.u.shift()
    for L in (1, 2):
        a[f"dlw{L}"], a[f"dly{L}"] = g.dlw.shift(L), g.dly.shift(L)
    rows = []
    for key in ("s1", "u1"):
        cols = ["dlw", "gap1", key, "dlw1", "dlw2", "dly1", "dly2"]
        e = a.dropna(subset=["dlw", "gap1", "s1", "u1", "dlw1", "dlw2", "dly1", "dly2"]).copy()
        e["gap1"] = e.gap1 - e.groupby("geo").gap1.transform("mean")
        X = e[cols] - e.groupby("geo")[cols].transform("mean")
        b, se, lo_b, hi_b, p, G = wild_t(X[cols[1:]].to_numpy(), X.dlw.to_numpy(), e.geo, 1)
        lo, hi, lab = widest(b, se, G, lo_b, hi_b, 0.002, "<")
        rows.append(dict(item=("outcome 26b: gamma on slack (t-1)" if key == "s1" else "same sample, official unemployment"),
                         est=b, se=se, ci90_lo=lo, ci90_hi=hi, p_wild=p, clusters=G, n=len(e),
                         years=f"{int(e.year.min())}-{int(e.year.max())}", label=lab))
    return pd.DataFrame(rows)


def ecm_us_u6():
    a = v9s2.AM[v9s2.AM.geo == "USA"].sort_values("year").set_index("year")
    lw = np.log(a.HWCDW / a.NLHA / a.PCPH)
    ly = np.log(a.RVGDE / a.NLHA)
    u6 = pd.read_csv(OUT.parents[1] / "data" / "raw" / "v9" / "fred_U6RATE.csv")
    u6.columns = ["date", "v"]
    u6 = u6.assign(year=pd.to_datetime(u6.date).dt.year).groupby("year").v.mean()
    u6 = u6[u6.index <= 2024]
    d = pd.DataFrame({"dlw": lw.diff(), "gap1": (lw - ly).shift(), "s1": u6.shift().reindex(lw.index),
                      "u1": a.ZUTN.shift(), "dlw1": lw.diff().shift(), "dlw2": lw.diff().shift(2),
                      "dly1": ly.diff().shift(), "dly2": ly.diff().shift(2)}).dropna()
    rows = []
    for key in ("s1", "u1"):
        f = sm.OLS(d.dlw, sm.add_constant(d[["gap1", key, "dlw1", "dlw2", "dly1", "dly2"]])).fit(cov_type="HAC",
                                                                                               cov_kwds={"maxlags": 2})
        rows.append(dict(item=f"US ECM, {'U-6' if key == 's1' else 'official unemployment'} (t-1)", est=f.params[key],
                         se=f.bse[key], ci90_lo=f.conf_int(0.1).loc[key, 0], ci90_hi=f.conf_int(0.1).loc[key, 1], n=len(d),
                         years=f"{d.index.min()}-{d.index.max()}"))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 4.2 outcome 27 with wild bootstrap-t
def outcome27():
    a = v9s2.AM.copy().sort_values(["geo", "year"])
    a["gap"] = np.log(a.HWCDW / a.NLHA / a.PCPH) - np.log(a.RVGDE / a.NLHA)
    a = a.replace([np.inf, -np.inf], np.nan).dropna(subset=["gap", "ZUTN"])
    a = a[a.groupby("geo").year.transform("count") >= 40]
    cbc, tud = v9s2.inst("oecd_cbc.csv"), v9s2.inst("oecd_tud.csv")
    e2 = a.merge(cbc, on=["geo", "year"], how="left").merge(tud, on=["geo", "year"], how="left").dropna(subset=["gap", "cbc", "tud"])
    X2 = e2[["gap", "cbc", "tud"]].copy()
    for _ in range(50):
        X2 = X2 - X2.groupby(e2.geo).transform("mean")
        X2 = X2 - X2.groupby(e2.year).transform("mean")
    b, se, lo_b, hi_b, p, G = wild_t(X2[["cbc", "tud"]].to_numpy(), X2.gap.to_numpy(), e2.geo, 0)
    b, se, lo_b, hi_b = 10 * b, 10 * se, 10 * lo_b, 10 * hi_b
    lo, hi, lab = widest(b, se, G, lo_b, hi_b, 0.01, ">")
    return dict(item="outcome 27 (wild bootstrap-t): ln(w/y) on coverage, per 10 pp", est=b, se=se, ci90_lo=lo, ci90_hi=hi,
                ci_t=(b - stats.t.ppf(0.95, G - 1) * se, b + stats.t.ppf(0.95, G - 1) * se), ci_wild=(lo_b, hi_b), p_wild=p,
                clusters=G, n=len(e2), label=lab)


# ---------------------------------------------------------------- 4.3 LCI D11 and outcome 22 without duplicate US
def lci_pt():
    lci = v9s2.es_json(OUT.parents[1] / "data" / "raw" / "v9" / "es_lci_q_wag.json")
    hicp = v9s2.es_json(OUT.parents[1] / "data" / "raw" / "v9" / "es_hicp_midx.json")
    hicp = hicp[hicp.coicop == "CP00"]
    curves = []
    for c2 in sorted(v9s2.EU27):
        l = lci[lci.geo == c2].set_index("time").value.sort_index()
        p = hicp[hicp.geo == c2].set_index("time").value.sort_index()
        if l.empty or p.empty:
            continue
        lq, pq = v9s2.to_q(l), v9s2.to_q(p)
        lw, dp = np.log(lq), np.log(pq).diff().reindex(lq.index)
        u = v9s2.unemp_q(c2)
        L = v9s2.lp(lw, dp, {"dw": lw.diff(), "dp": dp}, lags=4,
                    one_lag={"u": u.reindex(lw.index)} if u is not None else None, qdum=False)
        curves.append(L.assign(unit=c2))
    C = pd.concat(curves)
    rows = []
    for name, wb in (("outcome 22b (LCI D11, EU), h=8", C[C.h == 8].b.dropna()),):
        m = v9s2.boot_mean(wb)
        lo, hi = np.quantile(m, [0.05, 0.95])
        lab = ("подтверждено H_tech" if lo >= 0.9 and hi <= 1.1 else "опровергнуто H_tech" if hi < 0.9 else "неинформативно")
        rows.append(dict(item=name, est=wb.mean(), ci90_lo=lo, ci90_hi=hi, n=len(wb), label=lab))
    W = pd.read_csv(R9 / "s2_wage_lp.csv")
    for h in (4, 8):
        wb = W[(W.h == h) & (W.unit != "US (AHETPI)")].b.dropna()
        m = v9s2.boot_mean(wb)
        lo, hi = np.quantile(m, [0.05, 0.95])
        lab = ("подтверждено H_tech" if lo >= 0.9 and hi <= 1.1 else "опровергнуто H_tech" if hi < 0.9 else "неинформативно")
        rows.append(dict(item=f"outcome 22 without duplicate US (R11), h={h}", est=wb.mean(), ci90_lo=lo, ci90_hi=hi,
                         n=len(wb), label=lab))
    return C, pd.DataFrame(rows)


# ---------------------------------------------------------------- 4.4 terms of trade
def tot_2021():
    j = json.load(open(R / "es_nrg_ind_id.json"))
    geo = list(j["dimension"]["geo"]["category"]["index"])
    tim = list(j["dimension"]["time"]["category"]["index"])
    vals = {int(k): v for k, v in j["value"].items()}
    nt = len(tim)
    dep = {g: vals.get(i * nt + tim.index("2021")) for i, g in enumerate(geo)}
    D = pd.read_csv(R9 / "s2_2021.csv")
    D["dep2021"] = D.geo.map(lambda g: dep.get({"GR": "EL"}.get(g, g)))
    D["importer"] = D.dep2021 > 50
    imp = D[D.importer]
    c, t = imp[~imp.indexation].d2023.dropna().to_numpy(), imp[imp.indexation].d2023.dropna().to_numpy()
    rng = np.random.default_rng(425)
    bd = [t[rng.integers(0, len(t), len(t))].mean() - c[rng.integers(0, len(c), len(c))].mean() for _ in range(2000)]
    lo, hi = np.quantile(bd, [0.05, 0.95])
    est = t.mean() - c.mean()
    allv = np.r_[t, c]
    perm = [allv[list(k)].mean() - np.delete(allv, list(k)).mean() for k in itertools.combinations(range(len(allv)), len(t))]
    p = float(np.mean(np.array(perm) >= est))
    out = [dict(item="outcome 25b: indexation - control within energy importers (d2023)", est=est, ci90_lo=lo, ci90_hi=hi,
                n=len(allv), n_index=len(t), p_perm=p, label=decide(est, lo, hi, 0, 0.02, ">"))]
    ctl = D[~D.indexation]
    a, b = ctl[ctl.importer].d2023.dropna(), ctl[~ctl.importer].d2023.dropna()
    out.append(dict(item="descr: control countries, importers - non-importers (d2023)", est=a.mean() - b.mean(),
                    n=len(a) + len(b), n_importers=len(a)))
    return D, pd.DataFrame(out)


# ---------------------------------------------------------------- 5: outcome 20 market measures (Z.1)
def market_devaluation():
    z = zipfile.ZipFile(R / "z1_csv_files.zip")
    rr = pd.read_csv(z.open("csv/S11_1_r.csv")).set_index("date")
    bs = pd.read_csv(z.open("csv/S11_1_b.csv")).set_index("date")
    num = lambda c: pd.to_numeric(c.replace("ND", np.nan), errors="coerce")  # noqa: E731
    yr = lambda idx: idx.str[:4].astype(int)                                # noqa: E731
    flows = pd.DataFrame({k: num(rr[k + ".Q"]) for k in ("FR102010005", "FV102090005")})
    flows = flows.groupby(yr(flows.index)).sum(min_count=4)
    q4 = bs[bs.index.str.endswith("Q4")]
    q4.index = yr(q4.index)
    st = pd.DataFrame({"nfa_mv": num(q4["LM102010005.Q"]), "equity_mv": num(q4["LM103164105.Q"])})
    nw = pd.read_csv(z.open("csv/S11_1_b.csv")).set_index("date")
    nwq = nw[nw.index.str.endswith("Q4")]
    nwq.index = yr(nwq.index)
    netw = num(nwq["LM102090005.Q"]) if "LM102090005.Q" in nwq else num(nwq["FL102090005.Q"])
    py = nipa("A191RD")
    infl = py / py.shift() - 1
    d = pd.DataFrame({"reval_real": flows.FR102010005 / st.nfa_mv.shift() - infl,
                      "other_volume": flows.FV102090005 / netw.shift(),
                      "equity_real_growth": st.equity_mv / st.equity_mv.shift() / (1 + infl) - 1})
    rec = pd.read_csv(OUT.parents[1] / "data" / "raw" / "v8" / "fred_USREC.csv")
    rec.columns = ["date", "rec"]
    rec = rec.assign(year=pd.to_datetime(rec.date).dt.year).groupby("year").rec.max()
    d["crisis"] = rec.reindex(d.index)
    d = d.dropna(subset=["crisis"])
    rows = []
    for c in ("reval_real", "other_volume", "equity_real_growth"):
        x = d[[c, "crisis"]].dropna()
        a, b = x[x.crisis == 1][c], x[x.crisis == 0][c]
        t = stats.ttest_ind(a, b, equal_var=False)
        se = np.sqrt(a.var() / len(a) + b.var() / len(b))
        dfw = se ** 4 / ((a.var() / len(a)) ** 2 / (len(a) - 1) + (b.var() / len(b)) ** 2 / (len(b) - 1))
        tq = stats.t.ppf(0.95, dfw)
        est = a.mean() - b.mean()
        rows.append(dict(item=f"Z.1 NFC {c}: crisis - other years", est=est, ci90_lo=est - tq * se, ci90_hi=est + tq * se,
                         p=float(t.pvalue), n_crisis=len(a), n_other=len(b), years=f"{x.index.min()}-{x.index.max()}"))
    return d, pd.DataFrame(rows)


# ---------------------------------------------------------------- 5: outcome 34 paired percentile CI
def outcome34_paired(B=499, seed=34):
    from ..revisit.wiod import sea
    from ..v9.b_est import grouped_b
    s = sea()
    s["year"] = s.year.astype(int)
    w = s.pivot_table(index=["country", "code", "year"], columns="variable", values="v").reset_index()
    w = w[~w.code.isin(["O84", "P85", "Q", "T", "U", "L68", "B", "D35", "K64", "K65", "K66"])]
    w = w[(w.CAP > 0) & (w.LAB > 0) & (w.K > 0)].copy()
    w["PI"], w["W"] = w.CAP, w.LAB

    def both(d):
        d1 = d.assign(cy=d.country + d.year.astype(str), ind=d.code)
        d2 = d.assign(cy=d.code_id + d.year.astype(str), ind=d.country)
        return grouped_b(d2)[0] - grouped_b(d1)[0]
    w["code_id"] = w.code
    est = both(w)
    rng = np.random.default_rng(seed)
    codes = w.code.unique()
    groups = {c: g for c, g in w.groupby("code")}
    bs = []
    for i in range(B):
        pick = rng.choice(codes, len(codes), replace=True)
        smp = pd.concat([groups[c].assign(code=f"{c}#{k}", code_id=f"{c}#{k}") for k, c in enumerate(pick)], ignore_index=True)
        try:
            bs.append(both(smp))
        except Exception:                                          # noqa: BLE001
            bs.append(np.nan)
    lo, hi = np.nanquantile(bs, [0.05, 0.95])
    return dict(item="outcome 34, paired cluster bootstrap (industries), percentile CI", est=est, ci90_lo=lo, ci90_hi=hi,
                B=B, label=decide(est, lo, hi, 0, 0.1, "<"))


# ---------------------------------------------------------------- 5: outcome 35 (lag 1; WIOD with economy-wide deflator)
def outcome35():
    from ..v8.rule import share_row
    from ..v9 import stage6 as s6
    from ..revisit.wiod import sea
    C = pd.read_csv(R9 / "s6_coint.csv")
    rows = []
    st = C[C.source == "stan"]
    rows.append(share_row("outcome 35, main from 9b: STAN, ADF lag 1", st.coint_lag1, 0.05, 0.15, ">"))
    s = sea()
    s["year"] = s.year.astype(int)
    t = s[s.variable.isin(["VA", "VA_QI"])].pivot_table(index=["country", "code", "year"], columns="variable",
                                                         values="v").reset_index()
    base = t[t.year == 2010].set_index(["country", "code"]).VA
    t["VAr"] = [base.get((c, k), np.nan) * qi / 100 for c, k, qi in zip(t.country, t.code, t.VA_QI)]
    P = t.groupby(["country", "year"]).agg(VA=("VA", "sum"), VAr=("VAr", "sum"))
    P = (P.VA / P.VAr).rename("P_econ")
    man = s6.sea_manuf()
    S = {c: d.join(P.loc[c].rename("P2"), how="left").assign(P=lambda q: q.P2).drop(columns="P2") for c, d in man.items()
         if c in P.index.get_level_values(0)}
    orig = s6.sea_manuf
    s6.sea_manuf = lambda: S
    try:
        pan = s6.panel("wiod")
    finally:
        s6.sea_manuf = orig
    res = []
    for c, d in pan.groupby("country"):
        if len(d) < 12:
            continue
        p1, beta = s6.eg(d, 1)
        res.append(dict(country=c, n=len(d), eg_p_lag1=p1, beta=beta, coint=p1 < 0.05))
    Rw = pd.DataFrame(res)
    rows.append(share_row("outcome 35 variant: WIOD, economy-wide VA deflator, ADF lag 1", Rw.coint, 0.05, 0.15, ">"))
    return Rw, pd.DataFrame(rows)


# ---------------------------------------------------------------- 5: b (R16): wider NLS grid, boundary share, clustered F
def nls_b4(y, x, cy, lo=-1.0, hi=4.0):
    from ..v9.b_est import _a_cy
    lx = np.log(x)

    def ssr(b):
        xb = np.exp(b * lx)
        return float(((y - _a_cy(y, xb, cy, "nls") * xb) ** 2).sum())
    return optimize.minimize_scalar(ssr, bounds=(lo, hi), method="bounded", options={"xatol": 1e-5}).x


def b_r16(B=499, seed=16):
    from ..v9.b_est import eurostat_all, klems_all, wiod_all
    rows = []
    for name, loader in (("KLEMS", klems_all), ("WIOD", wiod_all), ("Eurostat", eurostat_all)):
        d = loader().copy()
        d["kw"], d["y"] = d.K / d.W, d.PI / d.W
        b = nls_b4(d.y.to_numpy(), d.kw.to_numpy(), d.cy.to_numpy())
        rng = np.random.default_rng(seed)
        inds = d.ind.unique()
        groups = {i: g for i, g in d.groupby("ind")}
        bs = []
        for _ in range(B):
            s = pd.concat([groups[i] for i in rng.choice(inds, len(inds), replace=True)], ignore_index=True)
            bs.append(nls_b4(s.y.to_numpy(), s.kw.to_numpy(), s.cy.to_numpy()))
        bs = np.array(bs)
        lo, hi = np.quantile(bs, [0.05, 0.95])
        # clustered first-stage F of the IV (same construction as v9 iv_b)
        e = d[d.PI > 0].copy()
        e["x"] = np.log(e.K / e.W)
        ssum = e.groupby(["ind", "year"]).x.transform("sum")
        cnt = e.groupby(["ind", "year"]).x.transform("count")
        e["z"] = (ssum - e.x) / (cnt - 1)
        e = e[cnt > 1].dropna(subset=["z"])
        Xd = e[["x", "z"]] - e.groupby("cy")[["x", "z"]].transform("mean")
        f = sm.OLS(Xd.x, Xd[["z"]]).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(e.ind)[0]})
        rows.append(dict(panel=name, b_nls=b, ci90_lo=lo, ci90_hi=hi, share_at_upper=float(np.mean(bs > 3.999)),
                         share_at_lower=float(np.mean(bs < -0.999)), share_above_2=float(np.mean(bs > 2)),
                         iv_first_stage_F_clustered=float(f.tvalues["z"] ** 2), n_clusters=int(e.ind.nunique())))
        print(rows[-1], flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    part = sys.argv[1] if len(sys.argv) > 1 else "all"
    if part in ("labour", "all"):
        E = ecm_panel()
        U = ecm_us_u6()
        o27 = outcome27()
        Cl, P22 = lci_pt()
        Dt, T = tot_2021()
        EG9, A9 = v9s2.anchor()                                   # iteration 9 outcomes 26-27 without forecast years
        A9.to_csv(OUT / "s4_anchor_v9_corrected.csv", index=False)
        EG9.to_csv(OUT / "s4_anchor_eg_v9_corrected.csv", index=False)
        print(A9.round(4).to_string(), flush=True)
        out = pd.concat([E, U, pd.DataFrame([o27]), P22, T], ignore_index=True)
        out.to_csv(OUT / "s4_outcomes.csv", index=False)
        Cl.to_csv(OUT / "s4_lci_lp.csv", index=False)
        Dt.to_csv(OUT / "s4_2021_tot.csv", index=False)
        print(out.round(4).to_string())
    if part in ("minor", "all"):
        d20, o20 = market_devaluation()
        d20.to_csv(OUT / "s5_market_devaluation_series.csv")
        o20.to_csv(OUT / "s5_outcome20_market.csv", index=False)
        print(o20.round(4).to_string(), flush=True)
        Rw, o35 = outcome35()
        Rw.to_csv(OUT / "s5_outcome35_wiod_econ_deflator.csv", index=False)
        o35.to_csv(OUT / "s5_outcome35.csv", index=False)
        print(o35.round(3).to_string(), flush=True)
    if part in ("o34", "all"):
        o34 = outcome34_paired()
        pd.DataFrame([o34]).to_csv(OUT / "s5_outcome34_paired.csv", index=False)
        print(o34, flush=True)
    if part in ("b", "all"):
        b = b_r16()
        b.to_csv(OUT / "s5_b_r16.csv", index=False)
        print(b.round(3).to_string())
