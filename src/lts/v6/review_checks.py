"""Iteration 6: exploratory checks requested by the independent review (pre_registration_v6.md, journal 8).

(a) historical-cost PIM validated on US NFC against official BEA 4.3;   -> rc_pim_usnfc.csv
(c)/(b) variant table: robust -/+ counts and outcome by the section-3 rule; -> rc_variants.csv
(d) fixed-b (Kiefer-Vogelsang 2005, Bartlett) significance for T1;       -> rc_fixedb.csv
(e) 5.2 simulated placebo: independent AR(1) profits per industry;       -> rc_s52_sim.csv
(f) 5.3 trends of the NFC main series from the common start 1958;        -> rc_s53_1958.csv
(g) T3 mechanics: d ln rM on d ln(K/Y) controlling d ln(PI/Y).            -> rc_t3_mech.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..v5.stage1 import nw_se, ols_fwl
from .series import MAIN8, OUT, fa, finish, hist_cost, nipa
from .stage5 import klems_panel_1995, pooled_ar1
from .tests import DECADES, period_panel, trend


# ---------------------------------------------------------------- (a)
def pim_usnfc():
    n = nipa(["W326RC", "A460RC"])
    d = pd.DataFrame({"PI": n.W326RC, "W": n.A460RC})
    d["K"] = fa(4, "FAAt401-A", 37)
    d["Khc_bea"] = fa(4, "FAAt403-A", 37)
    d["UIGT"] = fa(4, "FAAt407-A", 37)
    d["UKCT"] = fa(4, "FAAt404-A", 37)
    d = d.loc[1960:2024]
    out, tr = {}, []
    for name, kw in {"pim_const_delta": dict(), "pim_delta_t": dict(time_varying=True),
                     "pim_delta_t_i068": dict(time_varying=True, init_scale=0.68),
                     "pim_const_delta_i068": dict(init_scale=0.68)}.items():
        k, _ = hist_cost(d, **kw)
        out[name] = k
    out["bea_hc"] = d.Khc_bea
    rows = []
    for y in (1960, 1975, 1995, 2024):
        rows.append(dict(kind="ratio_to_bea", year=y, **{k: v.loc[y] / d.Khc_bea.loc[y] for k, v in out.items()}))
    for name, k in out.items():
        r = (d.PI / k).loc[1975:2024].to_numpy()
        for lag in (2, 4, 8):
            sl, se, p = trend(r, lag)
            tr.append(dict(kind="trend_r_1975_2024", series=name, lag=lag, slope=sl, p=p))
    return pd.concat([pd.DataFrame(rows), pd.DataFrame(tr)], ignore_index=True)


# ---------------------------------------------------------------- variant table
RULE_VARIANTS = ["main", "no_mi", "mi_half", "hc", "hc_d07", "hc_d13", "hc_i07", "hc_i13", "hc_hcdep", "hc_dt",
                 "hc_dt_i068", "kdrift"]


def variant_table(rob, pan):
    rows = []
    for v in RULE_VARIANTS:
        x = rob[(rob.variant == v) & rob.geo.isin(MAIN8) & (rob.series == "rM")]
        neg, pos = sorted(x[x.robust_neg].geo), sorted(x[x.robust_pos].geo)
        rows.append(dict(variant=v, n_geo=len(x), robust_neg=len(neg), neg=",".join(neg), robust_pos=len(pos),
                         pos=",".join(pos)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- (d) fixed-b
def kv_cv(b):
    """Kiefer-Vogelsang (2005) Bartlett, two-sided 5% (0.975 quantile) critical value."""
    return 1.9600 + 2.9694 * b + 0.4160 * b ** 2 - 0.5324 * b ** 3


def fixedb(s):
    rows = []
    for (geo, v), g in s[s.variant.isin(["main"] + RULE_VARIANTS)].groupby(["geo", "variant"]):
        g = g.set_index("year").sort_index()
        starts = [1951, 1956, 1946] if geo == "USA_NFC" else [g.index.min(), g.index.min() + 5]
        for a in starts:
            for b_ in (min(g.index.max(), 2024), 2019):
                y = g.loc[a:b_, "rM"].dropna().to_numpy()
                n = len(y)
                if n < 20:
                    continue
                X = np.column_stack([np.ones(n), np.arange(n)])
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                for lag in (2, 4, 8):
                    se = nw_se(X, y - X @ beta, lag)[1]
                    t = beta[1] / se
                    cv = kv_cv((lag + 1) / n)
                    rows.append(dict(geo=geo, variant=v, y0=a, y1=b_, lag=lag, t=t, cv_fixedb=cv,
                                     sig_fixedb=abs(t) > cv, sig_std=abs(t) > 1.96))
    f = pd.DataFrame(rows)
    agg = f.groupby(["geo", "variant"]).apply(lambda x: pd.Series(dict(
        robust_neg_fixedb=bool(((x.t < 0) & x.sig_fixedb).all()), robust_pos_fixedb=bool(((x.t > 0) & x.sig_fixedb).all()),
        share_sig_fixedb=x.sig_fixedb.mean(), share_sig_std=x.sig_std.mean())), include_groups=False).reset_index()
    return f, agg


# ---------------------------------------------------------------- (e) 5.2 simulated placebo
def s52_sim(reps=50, seed=17):
    p = klems_panel_1995()
    p = p[p.cfc_sh.notna() & (p.K > 0) & ~p.gov.astype(bool) & ~p.rent.astype(bool)].copy()
    p = p.sort_values(["geo", "ind", "year"])
    g = p.groupby(["geo", "ind"])
    cons = g.year.diff() == 1
    p["K_lag"], p["I_lag"], p["PI_lag"] = g.K.shift(), g.I_GFCF.shift(), g.PI.shift()
    ok = cons & (p.I_lag >= 0.01 * p.K_lag)
    # per-industry AR(1) in levels
    par = {}
    for key, x in p.groupby(["geo", "ind"]):
        y, yl = x.PI.to_numpy()[1:], x.PI.to_numpy()[:-1]
        m = np.isfinite(y) & np.isfinite(yl)
        if m.sum() < 10:
            continue
        X = np.column_stack([np.ones(m.sum()), yl[m]])
        b = np.linalg.lstsq(X, y[m], rcond=None)[0]
        b[1] = min(b[1], 0.99)
        sd = np.std(y[m] - X @ b, ddof=2)
        par[key] = (b[0], b[1], sd, x.PI.iloc[0])
    rng = np.random.default_rng(seed)
    cy = [p.geo, p.year]
    out = []
    for rep in range(reps + 1):
        q = p.copy()
        if rep > 0:                                   # rep 0 = actual data, same pipeline
            sim = pd.Series(np.nan, index=q.index)
            for key, idx in q.groupby(["geo", "ind"]).groups.items():
                if key not in par:
                    continue
                c, rho, sd, x0 = par[key]
                v = np.empty(len(idx))
                v[0] = x0
                for t in range(1, len(idx)):
                    v[t] = c + rho * v[t - 1] + rng.normal(0, sd)
                sim.loc[idx] = v
            q["PI"] = sim
            q["PI_lag"] = q.groupby(["geo", "ind"]).PI.shift()
        q["r_inc"] = np.where(ok, (q.PI - q.PI_lag) / q.I_lag, np.nan)
        lo = q.groupby(cy).r_inc.transform(lambda s: s.quantile(0.01))
        hi = q.groupby(cy).r_inc.transform(lambda s: s.quantile(0.99))
        q["r_inc"] = q.r_inc.clip(lo, hi)
        q["dev"] = q.r_inc - q.groupby(cy).r_inc.transform("mean")
        ar = pooled_ar1(q, "dev")
        out.append(dict(rep=rep, kind="actual" if rep == 0 else "sim", lam_fe=ar["lam_fe"], lam_nofe=ar["lam_nofe"]))
    return pd.DataFrame(out)


# ---------------------------------------------------------------- (f) 5.3 from 1958
def s53_1958():
    n = nipa(["W326RC", "A460RC"])
    d = pd.DataFrame({"PI": n.W326RC, "W": n.A460RC})
    d["K"] = fa(4, "FAAt401-A", 37)
    d = d.loc[1951:2024]
    d["r"], d["rM"] = d.PI / d.K, d.PI / (d.K + d.W)
    rows = []
    for a in (1951, 1958, 1963):
        for series in ("r", "rM"):
            for lag in (2, 4, 8):
                sl, se, p = trend(d.loc[a:2024, series].to_numpy(), lag)
                rows.append(dict(y0=a, series=series, lag=lag, slope=sl, p=p))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- (g) T3 mechanics
def t3_mech(s):
    rows = []
    for name, (geos, var) in {"main8": (MAIN8, "main"), "kdrift7": ([g for g in MAIN8 if g != "AUS"], "kdrift")}.items():
        f = period_panel(s, geos, var, DECADES)
        Y = s[(s.variant == var) & s.geo.isin(geos)].copy()
        Y["period"] = 0
        for per, (a, b) in DECADES.items():
            Y.loc[Y.year.between(a, b), "period"] = per
        S = Y[Y.period > 0].groupby(["geo", "period"])[["PI", "K", "W", "Y"]].sum().reset_index()
        S["lnky"], S["lnpy"] = np.log(S.K / S.Y), np.log(S.PI / S.Y)
        S = S.sort_values(["geo", "period"])
        gg = S.groupby("geo")
        S["d_lnky"], S["d_lnpy"] = gg.lnky.diff(), gg.lnpy.diff()
        f = f.merge(S[["geo", "period", "d_lnky", "d_lnpy"]], on=["geo", "period"])
        for fe, fe_name in ((["geo", "period"], "country+period"), (["geo"], "country")):
            for x, ctrl in (("d_ln1k", []), ("d_lnky", ["d_lnpy"]), ("d_lnky", [])):
                r = ols_fwl(f.dropna(subset=[x] + ctrl), "d_lnrM", x, ctrl, fe, B=9999)
                rows.append(dict(sample=name, fe=fe_name, x=x, controls="+".join(ctrl) or "none", **r))
    return pd.DataFrame(rows)


def e_k_corr(s):
    f = period_panel(s, MAIN8, "main", DECADES)
    full = []
    for geo in MAIN8:
        g = s[(s.geo == geo) & (s.variant == "main")].set_index("year").sort_index()
        for per, (a, b) in DECADES.items():
            x = g.loc[a:b]
            if len(x) == b - a + 1:
                S = x[["PI", "K", "W"]].sum()
                full.append(dict(geo=geo, period=per, lne=np.log(S.PI / S.W), ln1k=np.log1p(S.K / S.W)))
    q = pd.DataFrame(full).sort_values(["geo", "period"])
    q["d_lne"], q["d_ln1k"] = q.groupby("geo").lne.diff(), q.groupby("geo").ln1k.diff()
    q = q.dropna()
    return dict(corr=q.d_lne.corr(q.d_ln1k), slope_e_on_k=np.polyfit(q.d_ln1k, q.d_lne, 1)[0], n=len(q))


if __name__ == "__main__":
    s = pd.read_csv(OUT / "series.csv")
    a = pim_usnfc()
    a.to_csv(OUT / "rc_pim_usnfc.csv", index=False)
    print(a.round(4).to_string(), flush=True)
    rob = pd.read_csv(OUT / "trends_robust.csv")
    vt = variant_table(rob, None)
    vt.to_csv(OUT / "rc_variants.csv", index=False)
    print(vt.to_string(), flush=True)
    f, agg = fixedb(s)
    agg.to_csv(OUT / "rc_fixedb.csv", index=False)
    print(agg[agg.geo.isin(MAIN8 + ["USA_NFC"])].to_string(), flush=True)
    f53 = s53_1958()
    f53.to_csv(OUT / "rc_s53_1958.csv", index=False)
    print(f53.round(5).to_string(), flush=True)
    m = t3_mech(s)
    m.to_csv(OUT / "rc_t3_mech.csv", index=False)
    print(m.round(3).to_string(), flush=True)
    print("e-k corr", e_k_corr(s), flush=True)
    sim = s52_sim()
    sim.to_csv(OUT / "rc_s52_sim.csv", index=False)
    print(sim.describe().round(3).to_string(), sim.iloc[0].to_dict(), flush=True)
