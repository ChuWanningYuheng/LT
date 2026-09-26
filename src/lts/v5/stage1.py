"""Iteration 5, stage 1: aggregate profit rate and organic composition (pre_registration_v5.md, section 1).

Inputs: results/v5/agg_country_year.csv (lts.v5.agg), OECD EO output gap, Eurostat/FRED capacity utilisation,
NIPA T11400 + FRED Z.1 for the US nonfinancial-corporate long series, EPWT 7.0 (reference only).

Outputs (results/v5/):
  s1_decomp.csv     Weisskopf and Marx decompositions per country x variant (cumulative and covariance shares)
  s1_trend.csv      T1: OLS trend (Newey-West lag 2) and sup-F break (AR(1) parametric bootstrap, 999), Holm
  s1_t2.csv         T2 / T2-mechanical / T3 panels (5-year means and annual), wild cluster bootstrap by country
  s1_us_long.csv    US NFC 1951-2024 series; s1_us_decomp.csv decade decompositions
  s1_epwt.csv       EPWT 7.0 profit-rate correlation with the main aggregate (reference)
  s1_criteria.md    pre-registered criteria evaluated
"""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd

from ..v3.crosscountry import holm, wild_cluster
from ..v3.summarize import md

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw" / "v5"
OUT = ROOT / "results" / "v5"
ISO3 = dict(AT="AUT", BE="BEL", BG="BGR", CZ="CZE", DE="DEU", DK="DNK", EE="EST", EL="GRC", ES="ESP", FI="FIN",
            FR="FRA", HR="HRV", HU="HUN", IE="IRL", IT="ITA", JP="JPN", LT="LTU", LU="LUX", LV="LVA", MT="MLT",
            NL="NLD", PL="POL", PT="PRT", RO="ROU", SE="SWE", SI="SVN", SK="SVK", UK="GBR", US="USA")
PERIODS = {1: range(1995, 2000), 2: range(2000, 2005), 3: range(2005, 2010), 4: range(2010, 2015),
           5: range(2015, 2020)}
TRAPS = ["main", "no_mi", "tangible", "intangibles", "all", "no_fin", "no_mining"]
UA = "LT-research-script/1.0 (noreply@anthropic.com)"


# ---------------------------------------------------------------- data
def gap():
    g = pd.read_csv(RAW / "oecd_eo_gap.csv", usecols=["REF_AREA", "TIME_PERIOD", "OBS_VALUE"])
    inv = {v: k for k, v in ISO3.items()}
    g = g[g.REF_AREA.isin(inv)].assign(geo=lambda x: x.REF_AREA.map(inv))
    return g.rename(columns={"TIME_PERIOD": "year", "OBS_VALUE": "gap"})[["geo", "year", "gap"]]


def cu():
    with gzip.open(RAW / "ei_bsin_q_r2.tsv.gz", "rt") as f:
        t = pd.read_csv(f, sep="\t")
    k = t.columns[0]
    t[["freq", "indic", "s_adj", "geo"]] = t[k].str.split(",", expand=True)
    t = t[(t.indic == "BS-ICU-PC") & (t.s_adj == "SA")].drop(columns=[k, "freq", "indic", "s_adj"])
    t = t.melt(id_vars="geo", var_name="q", value_name="v")
    t["v"] = pd.to_numeric(t.v.astype(str).str.strip().str.split(" ").str[0], errors="coerce")
    t["year"] = t.q.str.strip().str[:4].astype(int)
    t["geo"] = t.geo.replace({"GR": "EL"})
    e = t.groupby(["geo", "year"]).v.mean().rename("cu").reset_index()
    u = pd.read_csv(RAW / "fred_TCU.csv")
    u["year"] = u.observation_date.str[:4].astype(int)
    u = u.groupby("year").TCU.mean().rename("cu").reset_index().assign(geo="US")
    return pd.concat([e, u])


def load():
    d = pd.read_csv(OUT / "agg_country_year.csv")
    d = d.merge(gap(), on=["geo", "year"], how="left").merge(cu(), on=["geo", "year"], how="left")
    d["tech"] = d.Q_K / d.H
    d["pk_w"] = d.P_K / d.w_hour
    d["pk_py"] = d.P_K / d.P_Y
    return d


def logs(d):
    pos = d.PI > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        d = d.assign(lnr=np.where(pos, np.log(d.r), np.nan), lnrM=np.where(pos, np.log(d.rM), np.nan),
                     lne=np.where(pos, np.log(d.e), np.nan), ln1k=np.log1p(d.k),
                     lns=np.where(pos, np.log(d.share_PI), np.nan), lnqyk=np.log(d.Q_Y / d.Q_K),
                     lnpyk=np.log(d.P_Y / d.P_K), lntech=np.log(d.tech), lnpkw=np.log(d.pk_w),
                     lnpkpy=np.log(d.pk_py))
    return d


# ---------------------------------------------------------------- decompositions
def decomp(d):
    rows = []
    for (v, geo), g in d.groupby(["variant", "geo"]):
        g = logs(g.sort_values("year")).set_index("year")
        ok = g.lnr.dropna()
        if len(ok) < 5:
            continue
        y0, y1 = ok.index.min(), ok.index.max()
        a, b = g.loc[y0], g.loc[y1]
        row = dict(variant=v, geo=geo, y0=y0, y1=y1, n_pos=len(ok), n=len(g),
                   dlnr=b.lnr - a.lnr, c_share=b.lns - a.lns, c_qyk=b.lnqyk - a.lnqyk, c_pyk=b.lnpyk - a.lnpyk,
                   dlnrM=b.lnrM - a.lnrM, c_e=b.lne - a.lne, c_1k=-(b.ln1k - a.ln1k), dk=b.k - a.k)
        dd = g[["lnr", "lns", "lnqyk", "lnpyk", "lnrM", "lne", "ln1k"]].diff().dropna()
        if len(dd) >= 5:
            vr, vm = dd.lnr.var(), dd.lnrM.var()
            row.update(v_share=dd.lnr.cov(dd.lns) / vr, v_qyk=dd.lnr.cov(dd.lnqyk) / vr,
                       v_pyk=dd.lnr.cov(dd.lnpyk) / vr, v_e=dd.lnrM.cov(dd.lne) / vm,
                       v_1k=-dd.lnrM.cov(dd.ln1k) / vm, n_diff=len(dd))
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- T1 trend and break
def nw_se(X, e, lag=2):
    n = len(e)
    XtXi = np.linalg.inv(X.T @ X)
    u = X * e[:, None]
    S = u.T @ u
    for l in range(1, lag + 1):
        w = 1 - l / (lag + 1)
        G = u[l:].T @ u[:-l]
        S += w * (G + G.T)
    V = XtXi @ S @ XtXi * n / (n - X.shape[1])
    return np.sqrt(np.diag(V))


def sup_f(y, trim=0.15):
    n = len(y)
    t = np.arange(n, dtype=float)
    X0 = np.column_stack([np.ones(n), t])
    e0 = y - X0 @ np.linalg.lstsq(X0, y, rcond=None)[0]
    ssr0 = e0 @ e0
    lo, hi = int(np.floor(trim * n)), int(np.ceil((1 - trim) * n))
    best, arg = -np.inf, None
    for tau in range(max(lo, 2), min(hi, n - 2)):
        D = (t >= tau).astype(float)
        X1 = np.column_stack([X0, D, D * (t - tau)])
        e1 = y - X1 @ np.linalg.lstsq(X1, y, rcond=None)[0]
        ssr1 = e1 @ e1
        F = ((ssr0 - ssr1) / 2) / (ssr1 / (n - 4))
        if F > best:
            best, arg = F, tau
    return best, arg


def t1(y, years, B=999, seed=3):
    n = len(y)
    t = np.arange(n, dtype=float)
    X = np.column_stack([np.ones(n), t])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    se = nw_se(X, e)
    from scipy import stats
    tt = b[1] / se[1]
    p_tr = 2 * stats.t.sf(abs(tt), n - 2)
    F, tau = sup_f(y)
    rho = float(np.clip((e[1:] @ e[:-1]) / (e[:-1] @ e[:-1]), -0.99, 0.99))
    u = e[1:] - rho * e[:-1]
    u = u - u.mean()
    rng = np.random.default_rng(seed)
    Fs = np.empty(B)
    for i in range(B):
        us = rng.choice(u, n)
        es = np.empty(n)
        es[0] = us[0] / np.sqrt(1 - rho ** 2)
        for j in range(1, n):
            es[j] = rho * es[j - 1] + us[j]
        Fs[i] = sup_f(X @ b + es)[0]
    return dict(slope=b[1], se_nw=se[1], t_nw=tt, p_trend=p_tr, supF=F, break_year=int(years[tau]),
                p_break=(1 + (Fs >= F).sum()) / (B + 1), rho=rho, n=n)


def trends(d, variants=("main", "all", "no_mi")):
    rows = []
    for v in variants:
        for geo, g in d[d.variant == v].groupby("geo"):
            g = g.sort_values("year")
            for s in ("r", "rM"):
                y = g[s].to_numpy()
                if len(y) < 15 or not np.isfinite(y).all():
                    continue
                rows.append(dict(variant=v, geo=geo, series=s, y0=g.year.min(), y1=g.year.max(),
                                 **t1(y, g.year.to_numpy())))
    r = pd.DataFrame(rows)
    for (v, s), g in r.groupby(["variant", "series"]):
        r.loc[g.index, "p_trend_holm"] = g.index.map(holm(dict(zip(g.index, g.p_trend))))
        r.loc[g.index, "p_break_holm"] = g.index.map(holm(dict(zip(g.index, g.p_break))))
    return r


# ---------------------------------------------------------------- panels
def five_year(d):
    rows = []
    for (v, geo), g in d.groupby(["variant", "geo"]):
        for per, yrs in PERIODS.items():
            s = g[g.year.isin(yrs)]
            if len(s) < len(yrs):                   # A-V5-5Y: only complete five-year periods
                continue
            S = s[["PI", "K", "W", "VA", "GO", "Q_K", "H"]].sum()
            rows.append(dict(variant=v, geo=geo, period=per, rM=S.PI / (S.K + S.W), e=S.PI / S.W, k=S.K / S.W,
                             pcm=S.PI / S.GO, tech=S.Q_K / S.H, pk_py=s.pk_py.mean(), pk_w=s.pk_w.mean(),
                             fin=s.fin_share.mean(), gap=s.gap.mean(), cu=s.cu.mean(), r=S.PI / S.K))
    f = pd.DataFrame(rows).sort_values(["variant", "geo", "period"])
    with np.errstate(divide="ignore", invalid="ignore"):
        f["lnrM"] = np.where(f.rM > 0, np.log(f.rM), np.nan)
        f["lne"] = np.where(f.e > 0, np.log(f.e), np.nan)
    f["ln1k"], f["lntech"] = np.log1p(f.k), np.log(f.tech)
    f["lnpkpy"], f["lnpkw"] = np.log(f.pk_py), np.log(f.pk_w)
    g = f.groupby(["variant", "geo"])
    for c in ("lnrM", "lne", "ln1k", "lntech", "lnpkpy", "lnpkw", "pcm", "fin", "gap", "rM", "cu"):
        f["d_" + c] = g[c].diff()
    f["gap_prev_period"] = g.period.diff()
    return f[f.gap_prev_period == 1]


def fe_resid(df, cols, fe):
    """Residualise cols on dummies for each FE column (exact LS, unbalanced panels)."""
    D = [np.ones((len(df), 1))]
    for c in fe:
        D.append(pd.get_dummies(df[c], drop_first=True).to_numpy(float))
    D = np.column_stack(D)
    M = df[cols].to_numpy(float)
    return M - D @ np.linalg.lstsq(D, M, rcond=None)[0]


def ols_fwl(df, y, x, controls, fe, cluster="geo", B=9999):
    """beta on x with controls and FE; CR1 t and restricted wild cluster bootstrap p via FWL."""
    s = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[y, x] + controls).copy()
    R = fe_resid(s, [y, x] + controls, fe)
    yv, xv, C = R[:, 0], R[:, 1], R[:, 2:]
    if C.shape[1]:
        P = C @ np.linalg.lstsq(C, np.column_stack([yv, xv]), rcond=None)[0]
        yv, xv = yv - P[:, 0], xv - P[:, 1]
    b, t, p = wild_cluster(yv, xv, s[cluster].to_numpy(), B=B)
    from scipy import stats
    G = s[cluster].nunique()
    return dict(beta=b, t_cr1=t, p_cr1=2 * stats.t.sf(abs(t), G - 1), p_wild=p, n=len(s), clusters=G)


CTRL5 = ["d_gap", "d_lnpkpy", "d_pcm", "d_fin"]


def panels(d):
    f = five_year(d)
    rows = []
    for v in TRAPS:
        s = f[f.variant == v]
        for spec, ctrl in (("base", []), ("controls", CTRL5)):
            rows.append(dict(test="T2", variant=v, spec=spec, y="d_lnrM", x="d_ln1k",
                             **ols_fwl(s, "d_lnrM", "d_ln1k", ctrl, ["geo", "period"])))
            rows.append(dict(test="T2_mech", variant=v, spec=spec, y="d_lnrM", x="d_ln1k",
                             **ols_fwl(s, "d_lnrM", "d_ln1k", ctrl + ["d_lne"], ["geo", "period"])))
            rows.append(dict(test="T2_levels", variant=v, spec=spec, y="d_rM", x="d_ln1k",
                             **ols_fwl(s, "d_rM", "d_ln1k", ctrl, ["geo", "period"])))
        rows.append(dict(test="T3", variant=v, spec="e+gap", y="d_lnrM", x="d_lntech",
                         **ols_fwl(s, "d_lnrM", "d_lntech", ["d_lne", "d_gap"], ["geo", "period"])))
        rows.append(dict(test="T3_check", variant=v, spec="gap", y="d_lnpkw", x="d_lntech",
                         **ols_fwl(s, "d_lnpkw", "d_lntech", ["d_gap"], ["geo", "period"])))
        rows.append(dict(test="T2_no_pcm_fin", variant=v, spec="gap+pk", y="d_lnrM", x="d_ln1k",
                         **ols_fwl(s, "d_lnrM", "d_ln1k", ["d_gap", "d_lnpkpy"], ["geo", "period"])))
    # annual version with the cycle
    a = logs(d).sort_values(["variant", "geo", "year"])
    g = a.groupby(["variant", "geo"])
    for c in ("lnrM", "ln1k", "lne", "rM"):
        a["d_" + c] = g[c].diff()
    a["dy"] = g.year.diff()
    a = a[a.dy == 1]
    for v in TRAPS:
        s = a[a.variant == v]
        rows.append(dict(test="T2_annual", variant=v, spec="gap", y="d_lnrM", x="d_ln1k",
                         **ols_fwl(s, "d_lnrM", "d_ln1k", ["gap"], ["geo", "year"])))
        rows.append(dict(test="T2_annual", variant=v, spec="cu", y="d_lnrM", x="d_ln1k",
                         **ols_fwl(s, "d_lnrM", "d_ln1k", ["cu"], ["geo", "year"])))
        rows.append(dict(test="T2_annual_levels", variant=v, spec="gap", y="d_rM", x="d_ln1k",
                         **ols_fwl(s, "d_rM", "d_ln1k", ["gap"], ["geo", "year"])))
    return pd.DataFrame(rows), f


# ---------------------------------------------------------------- US long series
def nipa(codes):
    d = pd.read_csv(RAW / "NipaDataA.txt", dtype=str)
    d.columns = ["code", "year", "v"]
    d = d[d.code.isin(codes)]
    d["v"] = pd.to_numeric(d.v.str.replace(",", ""), errors="coerce")
    d["year"] = d.year.astype(int)
    return d.pivot(index="year", columns="code", values="v")


def fred(series):
    s = pd.read_csv(RAW / f"fred_{series}.csv")
    s["year"] = s.observation_date.str[:4].astype(int)
    return s.set_index("year")[series].astype(float)


def us_long():
    n = nipa(["A455RC", "B455RX", "B456RC", "A460RC", "W325RC", "W326RC", "B008RG"])
    d = pd.DataFrame({"VA": n.A455RC, "Q_Y": n.B455RX, "CFC": n.B456RC, "W": n.A460RC, "TAX": n.W325RC,
                      "PI": n.W326RC, "P_Kidx": n.B008RG})
    K = fred("BOGZ1FL105013865A")          # millions of dollars, end of year; NIPA in millions as well
    IPP = fred("BOGZ1FL105013765A")
    d["K"], d["K_tang"] = K, K - IPP
    d = d.dropna(subset=["VA", "PI", "W", "K", "P_Kidx"])
    d = d[d.index >= 1951]
    d["Q_K"] = d.K / d.P_Kidx
    d["P_Y"], d["P_K"] = d.VA / d.Q_Y, d.P_Kidx
    for suf, K in (("", d.K), ("_tang", d.K_tang)):
        d["r" + suf], d["k" + suf] = d.PI / K, K / d.W
        d["rM" + suf] = d.PI / (K + d.W)
    d["e"], d["share_PI"] = d.PI / d.W, d.PI / d.VA
    d["check_nva"] = (d.VA - d.CFC - d.W - d.TAX - d.PI) / d.VA     # ~0 (net business transfers only)
    return d.reset_index().rename(columns={"index": "year"})


def us_decomp(u):
    rows = []
    u = u.set_index("year")
    dec = [(1951, 1966), (1966, 1982), (1982, 1997), (1997, 2007), (2007, 2019), (1951, int(u.index.max()))]
    for a, b in dec:
        if a not in u.index or b not in u.index:
            continue
        A, Bv = u.loc[a], u.loc[b]
        rows.append(dict(y0=a, y1=b, r0=A.r, r1=Bv.r, dlnr=np.log(Bv.r / A.r),
                         c_share=np.log(Bv.share_PI / A.share_PI),
                         c_qyk=np.log((Bv.Q_Y / Bv.Q_K) / (A.Q_Y / A.Q_K)),
                         c_pyk=np.log((Bv.P_Y / Bv.P_K) / (A.P_Y / A.P_K)),
                         dlnrM=np.log(Bv.rM / A.rM), c_e=np.log(Bv.e / A.e), c_1k=-np.log((1 + Bv.k) / (1 + A.k)),
                         k0=A.k, k1=Bv.k, e0=A.e, e1=Bv.e))
    return pd.DataFrame(rows)


def epwt(d):
    e = pd.read_excel(RAW / "epwt70.xlsx", sheet_name="EPWT7.0", usecols=["Countrycode", "Year", "rnatcur"])
    inv = {v: k for k, v in ISO3.items()}
    e = e[e.Countrycode.isin(inv)].assign(geo=lambda x: x.Countrycode.map(inv)).rename(columns={"Year": "year"})
    m = d[d.variant == "main"].merge(e, on=["geo", "year"])
    rows = []
    for geo, g in m.groupby("geo"):
        g = g.dropna(subset=["r", "rnatcur"])
        if len(g) >= 10:
            rows.append(dict(geo=geo, n=len(g), corr_level=g.r.corr(g.rnatcur),
                             corr_diff=g.r.diff().corr(g.rnatcur.diff()), r_mean=g.r.mean(),
                             epwt_mean=g.rnatcur.mean()))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- criteria
def criteria(dec, tr, t2):
    L = ["# Этап 1: предрегистрированные критерии (автоматически)\n"]
    m = t2[(t2.test == "T2")]
    L.append("## T2 (Δ ln r_M на Δ ln(1+k), пятилетки, без e)\n")
    L.append(md(m.set_index("variant")[["spec", "beta", "t_cr1", "p_cr1", "p_wild", "n", "clusters"]], "{:.3f}"))
    an = t2[(t2.test == "T2_annual") & (t2.spec == "gap")]
    trap = m[(m.spec == "base") & m.variant.isin(["main", "no_mi", "tangible", "intangibles", "all"])]
    pro_m = (trap.beta < 0).all() and (trap.p_wild < 0.05).all() and \
        (an[an.variant == "main"].beta < 0).all() and (an[an.variant == "main"].p_wild < 0.05).all()
    dm = dec[dec.variant == "main"]
    rising = dm[dm.dk > 0]
    comp_fail = (rising.dlnrM < 0).sum()
    L.append(f"\n* T2 β<0, p_wild<0,05 во всех вариантах ловушек и в годовой версии: **{pro_m}**")
    L.append(f"* Маркс (main): стран с растущим k: {len(rising)} из {len(dm)}; из них r_M упала: {comp_fail}")
    wv = dm.dropna(subset=["v_share"])
    other = ((wv.v_share + wv.v_pyk) > 0.5).sum()
    L.append(f"* Вайскопф (main): стран, где доля прибыли + отн. цена капитала > 50% дисперсии Δ ln r: "
             f"{other} из {len(wv)}; где Q_Y/Q_K > 50%: {(wv.v_qyk > 0.5).sum()}")
    tm = tr[(tr.variant == "main")]
    for s in ("r", "rM"):
        x = tm[tm.series == s]
        L.append(f"* T1 {s}: отрицательный тренд p_Holm<0,05: {((x.slope < 0) & (x.p_trend_holm < 0.05)).sum()}, "
                 f"положительный: {((x.slope > 0) & (x.p_trend_holm < 0.05)).sum()}, из {len(x)}; "
                 f"разрыв p_Holm<0,05: {(x.p_break_holm < 0.05).sum()}")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    d = load()
    dec = decomp(d)
    dec.to_csv(OUT / "s1_decomp.csv", index=False)
    print("decomp done", flush=True)
    t2, f = panels(d)
    t2.to_csv(OUT / "s1_t2.csv", index=False)
    f.to_csv(OUT / "s1_five_year.csv", index=False)
    print(t2.round(3).to_string(), flush=True)
    u = us_long()
    u.to_csv(OUT / "s1_us_long.csv", index=False)
    ud = us_decomp(u)
    ud.to_csv(OUT / "s1_us_decomp.csv", index=False)
    print(ud.round(3).to_string())
    ut = []
    for s in ("r", "rM", "r_tang"):
        ut.append(dict(series=s, y0=int(u.year.min()), y1=int(u.year.max()), **t1(u[s].to_numpy(), u.year.to_numpy())))
    pd.DataFrame(ut).to_csv(OUT / "s1_us_trend.csv", index=False)
    print(pd.DataFrame(ut).round(4).to_string())
    ep = epwt(d)
    ep.to_csv(OUT / "s1_epwt.csv", index=False)
    tr = trends(d)
    tr.to_csv(OUT / "s1_trend.csv", index=False)
    c = criteria(dec, tr, t2)
    (OUT / "s1_criteria.md").write_text(c)
    print(c)
