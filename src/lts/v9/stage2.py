"""Iteration 9, stage 2: is labour power produced like a commodity? (pre_registration_v9.md, stage 2; journal 2)

2.1 pass-through: input prices -> output prices (goods) vs consumer prices -> hourly wages, local projections h = 0..12.
2.2 2021-2023 inflation: automatic indexation (BE, LU, CY, MT) vs other EU countries, real hourly wages.
2.3 anchor and reversion: AMECO 1960-2024, real hourly wage vs productivity, unemployment, institutions.
2.4 cross-section (descriptive).
Outputs: results/v9/s2_*.csv
"""
from __future__ import annotations

import glob
import itertools
import json
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from ..v6.series import ROOT
from ..v8.rule import decide
from .stage1 import OUT

R9 = ROOT / "data" / "raw" / "v9"
H = range(0, 13)
INDEX = {"BE", "LU", "CY", "MT"}
EU27 = {"AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "EL", "ES", "FI", "FR", "HR", "HU", "IE", "IT", "LT", "LU", "LV",
        "MT", "NL", "PL", "PT", "RO", "SE", "SI", "SK"}
ISO3 = {"AUT": "AT", "BEL": "BE", "BGR": "BG", "CYP": "CY", "CZE": "CZ", "DEU": "DE", "DNK": "DK", "EST": "EE",
        "GRC": "EL", "ESP": "ES", "FIN": "FI", "FRA": "FR", "HRV": "HR", "HUN": "HU", "IRL": "IE", "ITA": "IT",
        "LTU": "LT", "LUX": "LU", "LVA": "LV", "MLT": "MT", "NLD": "NL", "POL": "PL", "PRT": "PT", "ROU": "RO",
        "SWE": "SE", "SVN": "SI", "SVK": "SK", "GBR": "UK", "USA": "US", "JPN": "JP", "CAN": "CA", "AUS": "AU",
        "NOR": "NO", "CHE": "CH", "NZL": "NZ", "ISL": "IS", "ISR": "IL", "KOR": "KR", "MEX": "MX", "TUR": "TR",
        "CHL": "CL", "COL": "CO", "CRI": "CR"}


# ---------------------------------------------------------------- data
def es_json(path):
    d = json.load(open(path))
    ids, size = d["id"], d["size"]
    cats = [sorted(d["dimension"][k]["category"]["index"].items(), key=lambda x: x[1]) for k in ids]
    mult = [int(np.prod(size[i + 1:])) for i in range(len(size))]
    rows = []
    for k, v in d["value"].items():
        k = int(k)
        rows.append({idn: cats[i][(k // mult[i]) % size[i]][0] for i, idn in enumerate(ids)} | {"value": v})
    return pd.DataFrame(rows)


def to_q(s):
    """monthly 'YYYY-MM' or 'YYYY-MMM' / quarterly 'YYYY-Qn' -> quarterly PeriodIndex (mean)."""
    idx = pd.Index(s.index.astype(str))
    if idx.str.contains("Q").all():
        p = pd.PeriodIndex(idx.str.replace("-Q", "Q"), freq="Q")
        return pd.Series(s.to_numpy(), index=p)
    p = pd.PeriodIndex(pd.to_datetime(idx.str.replace("M", "-")), freq="Q")
    return pd.Series(s.to_numpy(), index=p).groupby(level=0).mean()


def fred_q(code):
    d = pd.read_csv(R9 / f"fred_{code}.csv")
    d.columns = ["date", "v"]
    d["v"] = pd.to_numeric(d.v, errors="coerce")
    d["q"] = pd.PeriodIndex(pd.to_datetime(d.date), freq="Q")
    return d.groupby("q").v.mean().dropna()


def oecd_wages(activity="C"):
    d = pd.read_csv(R9 / "oecd_hou_ear.csv")
    d = d[(d.FREQ == "Q") & (d.ADJUSTMENT == "Y") & (d.ACTIVITY == activity) & (d.SECTOR == "S1")]
    out = {}
    for c, g in d.groupby("REF_AREA"):
        if c in ISO3:
            s = to_q(g.set_index("TIME_PERIOD").OBS_VALUE.sort_index())
            if s.notna().sum() >= 40:
                out[ISO3[c]] = s
    return out


def oecd_cpi(exp="_T"):
    d = pd.read_csv(R9 / "oecd_prices_all.csv", low_memory=False)
    d = d[(d.FREQ == "Q") & (d.EXPENDITURE == exp)]
    return {ISO3[c]: to_q(g.set_index("TIME_PERIOD").OBS_VALUE.sort_index()) for c, g in d.groupby("REF_AREA")
            if c in ISO3}


def ameco(codes):
    rows = []
    for f in sorted(glob.glob(str(ROOT / "data" / "raw" / "v6" / "ameco" / "AMECO*.TXT"))):
        d = pd.read_csv(f, sep=";", dtype=str, encoding="latin1")
        parts = d.CODE.str.split(".", expand=True)
        keep = parts[5].isin(codes) & (parts[2] != "99") & (parts[3] == "0") & (parts[4] == "0")
        for _, r in d[keep].iterrows():
            p = r.CODE.split(".")
            for y in [c for c in d.columns if c.strip().isdigit()]:
                v = pd.to_numeric(r[y], errors="coerce")
                if np.isfinite(v):
                    rows.append((p[0], p[5], int(y), v))
    a = pd.DataFrame(rows, columns=["geo", "var", "year", "v"]).drop_duplicates(["geo", "var", "year"])
    return a.pivot_table(index=["geo", "year"], columns="var", values="v").reset_index()


def unemp_q(c2):
    if c2 == "US":
        return fred_q("UNRATE")
    a = UNEMP_A.get(c2)
    if a is None or a.dropna().empty:
        return None
    a = a.dropna()
    q = pd.period_range(f"{a.index.min()}Q1", f"{a.index.max()}Q4", freq="Q")
    return pd.Series([a.get(p.year, np.nan) for p in q], index=q)


# ---------------------------------------------------------------- local projections
def lp(y, shock, controls, H=H, lags=4, qdum=True, one_lag=None):
    """y: log level; shock: series; controls: dict name -> series (lags 1..lags of each are added)."""
    out = []
    base = pd.DataFrame({"y": y, "s": shock}).dropna(subset=["y"])
    for h in H:
        e = pd.DataFrame({"dy": base.y.shift(-h) - base.y.shift(1), "s": base.s})
        for nm, c in controls.items():
            for L in range(1, lags + 1):
                e[f"{nm}_{L}"] = c.reindex(base.index).shift(L)
        for nm, c in (one_lag or {}).items():
            e[f"{nm}_1"] = c.reindex(base.index).shift(1)
        if qdum:
            for k in (1, 2, 3):
                e[f"q{k}"] = (e.index.quarter == k).astype(float)
        e = e.replace([np.inf, -np.inf], np.nan).dropna()
        if len(e) < 30:
            out.append(dict(h=h, b=np.nan, se=np.nan, n=len(e)))
            continue
        f = sm.OLS(e.dy, sm.add_constant(e.drop(columns="dy"))).fit(cov_type="HAC", cov_kwds={"maxlags": h + 1})
        out.append(dict(h=h, b=f.params.s, se=f.bse.s, n=len(e)))
    return pd.DataFrame(out)


def boot_mean(v, B=2000, seed=21):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    rng = np.random.default_rng(seed)
    m = v[rng.integers(0, len(v), (B, len(v)))].mean(1)
    return m


# ---------------------------------------------------------------- 2.1 goods
def cost_shares():
    from ..economy import national_blocks
    from ..figaro import INDUSTRIES
    from ..v8.stage3 import _table
    U = _table("IOUse_Before_Redefinitions_PRO_2017_Detail.xlsx")
    out_ = U.loc["T008"]
    us = {}
    us["steel"] = (U.loc[["2122A0", "S00401"], "331110"].sum() / out_["331110"])
    us["steel_ore"] = U.loc["2122A0", "331110"] / out_["331110"]
    us["steel_scrap"] = U.loc["S00401", "331110"] / out_["331110"]
    us["refined"] = U.loc["211000", "324110"] / out_["324110"]
    us["petrochem"] = U.loc["211000", "325110"] / out_["325110"]
    food = [c for c in U.columns if str(c).startswith("311")]
    farm = [r for r in U.index if str(r).startswith(("111", "112"))]
    us["food"] = U.loc[farm, food].to_numpy().sum() / out_[food].sum()
    eu = {}
    for c2 in ("AT", "BE", "BG", "CZ", "DE", "DK", "EE", "EL", "ES", "FI", "FR", "HR", "HU", "IE", "IT", "LT", "LU", "LV",
               "NL", "PL", "PT", "RO", "SE", "SI", "SK"):
        try:
            b = national_blocks(c2 if c2 != "EL" else "GR", 2017)
        except Exception:                                       # noqa: BLE001
            continue
        Z = b["Zdom"] + b["Zimp"]
        iB = INDUSTRIES.index("B")
        for prod in ("C24", "C19", "C20"):
            j = INDUSTRIES.index(prod)
            if b["x"][j] > 0:
                eu[(c2, prod)] = Z[iB, j] / b["x"][j]
    return us, eu


def goods_pt():
    us_s, eu_s = cost_shares()
    rows, curves = [], []
    # US pairs
    pairs = [("steel", "PCU3311103311105", ["WPU1011", "WPU1012"], [us_s["steel_ore"], us_s["steel_scrap"]]),
             ("refined", "PCU324110324110", ["WPU0561"], [us_s["refined"]]),
             ("petrochem", "PCU325110325110", ["WPU0561"], [us_s["petrochem"]]),
             ("food", "PCU311311", ["WPU01"], [us_s["food"]])]
    for name, out_code, ins, sh in pairs:
        po = np.log(fred_q(out_code))
        dins = [np.log(fred_q(c)).diff() for c in ins]
        w = np.array(sh) / sum(sh)
        din = sum(wi * d for wi, d in zip(w, dins))
        L = lp(po, din, {"dpo": po.diff(), "dpi": din})
        s = float(sum(sh))
        curves.append(L.assign(unit=f"US {name}", s=s, rpt=L.b / s))
    # EU pairs (domestic PPI, quarterly means)
    e = es_json(R9 / "es_ppi_m.json")
    e["q"] = pd.PeriodIndex(pd.to_datetime(e.time.str.replace("M", "-")), freq="Q")
    pq = e.groupby(["geo", "nace_r2", "q"]).value.mean()
    for prod, inp in (("C24", "B07"), ("C19", "B06"), ("C20", "B06")):
        for c2 in sorted(set(e.geo)):
            if (c2, prod) not in eu_s or eu_s[(c2, prod)] <= 0:
                continue
            try:
                po, pi = np.log(pq.loc[(c2, prod)]), np.log(pq.loc[(c2, inp)])
            except KeyError:
                continue
            if po.notna().sum() < 40 or pi.notna().sum() < 40:
                continue
            din = pi.diff()
            L = lp(po, din, {"dpo": po.diff(), "dpi": din})
            s = eu_s[(c2, prod)]
            curves.append(L.assign(unit=f"{c2} {prod}<-{inp}", s=s, rpt=L.b / s))
    C = pd.concat(curves)
    return C


# ---------------------------------------------------------------- 2.1 wages
def wage_pt(activity="C", iv=False):
    W = oecd_wages(activity)
    P = oecd_cpi("_T")
    Pf, Pe = oecd_cpi("CP01"), oecd_cpi("CP045_0722")
    curves = []
    for c2, w in W.items():
        if c2 not in P:
            continue
        lw = np.log(w)
        dp = np.log(P[c2]).diff()
        u = unemp_q(c2)
        ctr = {"dw": lw.diff(), "dp": dp}
        L = lp(lw, dp, ctr, lags=4, one_lag={"u": u.reindex(lw.index)} if u is not None else None)
        curves.append(L.assign(unit=c2))
    US = np.log(fred_q("AHETPI"))
    dpu = np.log(fred_q("CPIAUCSL")).diff()
    curves.append(lp(US, dpu, {"dw": US.diff(), "dp": dpu}, one_lag={"u": fred_q("UNRATE")}).assign(unit="US (AHETPI)"))
    return pd.concat(curves)


def wage_pt_iv(activity="C", h_list=(4, 8)):
    """IV-LP point estimates: headline CPI inflation instrumented by food and energy CPI inflation."""
    W = oecd_wages(activity)
    P, Pf, Pe = oecd_cpi("_T"), oecd_cpi("CP01"), oecd_cpi("CP045_0722")
    rows = []
    for c2, w in W.items():
        if c2 not in P or c2 not in Pf or c2 not in Pe:
            continue
        lw = np.log(w)
        d = pd.DataFrame({"lw": lw, "dp": np.log(P[c2]).diff(), "zf": np.log(Pf[c2]).diff(), "ze": np.log(Pe[c2]).diff()})
        for h in h_list:
            e = pd.DataFrame({"dy": d.lw.shift(-h) - d.lw.shift(1), "dp": d.dp, "zf": d.zf, "ze": d.ze})
            for L in range(1, 5):
                e[f"dw{L}"], e[f"dp{L}"] = d.lw.diff().shift(L), d.dp.shift(L)
            e = e.dropna()
            if len(e) < 30:
                continue
            ex = sm.add_constant(e[[c for c in e.columns if c.startswith(("dw", "dp")) and c != "dp"]])
            fs = sm.OLS(e.dp, pd.concat([ex, e[["zf", "ze"]]], axis=1)).fit()
            e["dph"] = fs.fittedvalues
            ss = sm.OLS(e.dy, pd.concat([ex, e[["dph"]]], axis=1)).fit()
            rows.append(dict(unit=c2, h=h, b_iv=ss.params.dph, fs_F=float(fs.f_test("zf = 0, ze = 0").fvalue), n=len(e)))
    return pd.DataFrame(rows)


def outcomes_pt(G, Wc):
    rows = []
    for h in (4, 8):
        wb = Wc[Wc.h == h].b.dropna()
        m = boot_mean(wb)
        lo, hi = np.quantile(m, [0.05, 0.95])
        est = wb.mean()
        lab = ("подтверждено H_tech" if lo >= 0.9 and hi <= 1.1 else "опровергнуто H_tech" if hi < 0.9
               else "неинформативно")
        rows.append(dict(item=f"outcome 22 (h={h}): mean wage pass-through", est=est, ci90_lo=lo, ci90_hi=hi,
                         n=len(wb), label=lab if h == 8 else f"(h=4) {lab}"))
    g4 = G[G.h == 4].rpt.dropna()
    w4 = Wc[Wc.h == 4].b.dropna()
    mg, mw = boot_mean(g4, seed=31), boot_mean(w4, seed=32)
    d = mg - mw
    lo, hi = np.quantile(d, [0.05, 0.95])
    est = g4.mean() - w4.mean()
    rows.append(dict(item="outcome 23 (h=4): mean RPT goods - mean wage pass-through", est=est, ci90_lo=lo,
                     ci90_hi=hi, n=len(g4), label=decide(est, lo, hi, 0, 0.1, ">")))
    for h in (0, 1, 2, 4, 8, 12):
        rows.append(dict(item=f"descr: median RPT goods h={h}", est=G[G.h == h].rpt.median(), n=G[G.h == h].rpt.notna().sum()))
        rows.append(dict(item=f"descr: median wage beta h={h}", est=Wc[Wc.h == h].b.median(), n=Wc[Wc.h == h].b.notna().sum()))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 2.2 2021-2023
def inflation_2021():
    lci = es_json(R9 / "es_lci_q_wag.json")
    hicp = es_json(R9 / "es_hicp_midx.json")
    hicp = hicp[hicp.coicop == "CP00"]
    rows = []
    for c2 in sorted(EU27):
        l = lci[lci.geo == c2].set_index("time").value.sort_index()
        p = hicp[hicp.geo == c2].set_index("time").value.sort_index()
        if l.empty or p.empty:
            continue
        lq, pq = to_q(l), to_q(p)
        rw = np.log(lq) - np.log(pq.reindex(lq.index))
        yr = rw.groupby(rw.index.year).mean()
        if 2021 in yr and 2023 in yr:
            rows.append(dict(geo=c2, indexation=c2 in INDEX, d2023=yr[2023] - yr[2021],
                             d2024=yr.get(2024, np.nan) - yr[2021]))
    D = pd.DataFrame(rows)
    ctrl, trt = D[~D.indexation], D[D.indexation]
    out = []
    for col in ("d2023", "d2024"):
        c, t = ctrl[col].dropna().to_numpy(), trt[col].dropna().to_numpy()
        m = boot_mean(c, seed=41)
        lo, hi = np.quantile(m, [0.05, 0.95])
        lab = ("подтверждено H_tech" if lo >= -0.02 and hi <= 0.02 else "опровергнуто H_tech" if hi < -0.02
               else "неинформативно")
        out.append(dict(item=f"outcome 24 ({col}): mean change of real hourly wage, control", est=c.mean(), ci90_lo=lo,
                        ci90_hi=hi, n=len(c), label=lab))
        rng = np.random.default_rng(42)
        bd = [t[rng.integers(0, len(t), len(t))].mean() - c[rng.integers(0, len(c), len(c))].mean() for _ in range(2000)]
        lo, hi = np.quantile(bd, [0.05, 0.95])
        est = t.mean() - c.mean()
        allv = np.r_[t, c]
        perm = [allv[list(k)].mean() - np.delete(allv, list(k)).mean() for k in itertools.combinations(range(len(allv)), len(t))]
        p = float(np.mean(np.array(perm) >= est))
        out.append(dict(item=f"outcome 25 ({col}): indexation - control", est=est, ci90_lo=lo, ci90_hi=hi, n=len(allv),
                        p_perm=p, label=decide(est, lo, hi, 0, 0.02, ">")))
    # exploratory: coverage and unemployment before the shock
    cbc = pd.read_csv(R9 / "oecd_cbc.csv")
    cbc = cbc[cbc.TIME_PERIOD <= 2020].sort_values("TIME_PERIOD").groupby("REF_AREA").OBS_VALUE.last()
    cbc.index = [ISO3.get(i, i) for i in cbc.index]
    D["cbc"] = D.geo.map(cbc)
    D["u2019"] = D.geo.map({k: v.get(2019, np.nan) for k, v in UNEMP_A.items()})
    e = D.dropna(subset=["cbc", "u2019"])
    if len(e) > 8:
        f = sm.OLS(e.d2023, sm.add_constant(e[["cbc", "u2019"]])).fit(cov_type="HC1")
        for k in ("cbc", "u2019"):
            out.append(dict(item=f"explor: d2023 on {k}", est=f.params[k], ci90_lo=f.conf_int(0.1).loc[k, 0],
                            ci90_hi=f.conf_int(0.1).loc[k, 1], n=len(e)))
    return D, pd.DataFrame(out)


# ---------------------------------------------------------------- 2.3 anchor
def anchor():
    a = AM.copy()
    a = a.sort_values(["geo", "year"])
    a["lw"] = np.log(a.HWCDW / a.NLHA / a.PCPH)
    a["ly"] = np.log(a.RVGDE / a.NLHA)
    a["u"] = a.ZUTN
    a = a.replace([np.inf, -np.inf], np.nan).dropna(subset=["lw", "ly", "u"])
    a = a[a.groupby("geo").year.transform("count") >= 40]
    rows = []
    for g, d in a.groupby("geo"):                                   # Engle-Granger per country
        try:
            from statsmodels.tsa.stattools import coint
            t, p, _ = coint(d.lw, d.ly, trend="c")
        except Exception:                                           # noqa: BLE001
            p = np.nan
        rows.append(dict(geo=g, years=len(d), first=int(d.year.min()), eg_p=p))
    EG = pd.DataFrame(rows)
    g = a.groupby("geo")
    a["dlw"] = g.lw.diff()
    a["dly"] = g.ly.diff()
    a["gap"] = (a.lw - a.ly)
    a["gap1"] = g.gap.shift()
    a["u1"] = g.u.shift()
    a["int1"] = a.gap1 * a.u1
    for L in (1, 2):
        a[f"dlw{L}"] = g.dlw.shift(L)
        a[f"dly{L}"] = g.dly.shift(L)
    cols = ["dlw", "gap1", "u1", "int1", "dlw1", "dlw2", "dly1", "dly2"]
    e = a.dropna(subset=cols)
    X = e[cols] - e.groupby("geo")[cols].transform("mean")
    fit = sm.OLS(X.dlw, X[cols[1:]]).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(e.geo)[0]})
    G = e.geo.nunique()
    tq = stats.t.ppf(0.95, G - 1)
    wild_p = wild_boot(X, cols, e.geo, "u1")
    out = []
    for k in ("gap1", "u1", "int1"):
        b, se = fit.params[k], fit.bse[k]
        lab = decide(b, b - tq * se, b + tq * se, 0, 0.002, "<") if k == "u1" else ""
        out.append(dict(item=f"ECM {k}" + (" (outcome 26)" if k == "u1" else ""), est=b, se=se, ci90_lo=b - tq * se,
                        ci90_hi=b + tq * se, n=len(e), clusters=G, p_wild=wild_p if k == "u1" else np.nan, label=lab))
    for var, cl in (("centred gap", ["dlw", "gap1c", "u1", "int1c", "dlw1", "dlw2", "dly1", "dly2"]),
                    ("no interaction", ["dlw", "gap1", "u1", "dlw1", "dlw2", "dly1", "dly2"])):
        e = e.copy()
        e["gap1c"] = e.gap1 - e.groupby("geo").gap1.transform("mean")
        e["int1c"] = e.gap1c * e.u1
        Xv = e[cl] - e.groupby("geo")[cl].transform("mean")
        fv = sm.OLS(Xv.dlw, Xv[cl[1:]]).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(e.geo)[0]})
        b, se = fv.params["u1"], fv.bse["u1"]
        out.append(dict(item=f"ECM u1, variant ({var}; journal 3)", est=b, se=se, ci90_lo=b - tq * se,
                        ci90_hi=b + tq * se, n=len(e), clusters=G, p_wild=wild_boot(Xv, cl, e.geo, "u1", B=1999),
                        label=decide(b, b - tq * se, b + tq * se, 0, 0.002, "<")))
        out.append(dict(item=f"ECM gap1, variant ({var})", est=fv.params[cl[1]], se=fv.bse[cl[1]], n=len(e)))
    # institutions: level of ln(w/y) on coverage and density, country and year effects
    cbc = inst("oecd_cbc.csv")
    tud = inst("oecd_tud.csv")
    a2 = a.merge(cbc, on=["geo", "year"], how="left").merge(tud, on=["geo", "year"], how="left")
    e2 = a2.dropna(subset=["gap", "cbc", "tud"])
    cols2 = ["gap", "cbc", "tud"]
    X2 = e2[cols2].copy()
    for _ in range(50):
        X2 = X2 - X2.groupby(e2.geo).transform("mean")
        X2 = X2 - X2.groupby(e2.year).transform("mean")
    f2 = sm.OLS(X2.gap, X2[["cbc", "tud"]]).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(e2.geo)[0]})
    G2 = e2.geo.nunique()
    tq2 = stats.t.ppf(0.95, G2 - 1)
    for k in ("cbc", "tud"):
        b, se = 10 * f2.params[k], 10 * f2.bse[k]
        out.append(dict(item=f"institutions: ln(w/y) on {k}, per 10 pp" + (" (outcome 27)" if k == "cbc" else ""),
                        est=b, se=se, ci90_lo=b - tq2 * se, ci90_hi=b + tq2 * se, n=len(e2), clusters=G2,
                        label=decide(b, b - tq2 * se, b + tq2 * se, 0, 0.01, ">") if k == "cbc" else ""))
    return EG, pd.DataFrame(out)


def wild_boot(X, cols, geo, k, B=9999, seed=5):
    """Wild cluster bootstrap (Webb weights), null imposed, p for coefficient k."""
    rng = np.random.default_rng(seed)
    xr = [c for c in cols[1:] if c != k]
    Xf, Xr = X[cols[1:]].to_numpy(), X[xr].to_numpy()
    y = X.dlw.to_numpy()
    br = np.linalg.lstsq(Xr, y, rcond=None)[0]
    yh, er = Xr @ br, y - Xr @ br
    gi = pd.factorize(geo)[0]
    G = gi.max() + 1
    j = cols[1:].index(k)
    XtX = np.linalg.inv(Xf.T @ Xf)

    def tstat(yy):
        b = XtX @ Xf.T @ yy
        e = yy - Xf @ b
        meat = np.zeros((Xf.shape[1],) * 2)
        for g in range(G):
            s = Xf[gi == g].T @ e[gi == g]
            meat += np.outer(s, s)
        V = XtX @ meat @ XtX * G / (G - 1)
        return b[j] / np.sqrt(V[j, j])
    t0 = tstat(y)
    webb = np.array([-np.sqrt(1.5), -1, -np.sqrt(0.5), np.sqrt(0.5), 1, np.sqrt(1.5)])
    ts = []
    for _ in range(B if len(y) < 3000 else 999):
        w = webb[rng.integers(0, 6, G)][gi]
        ts.append(tstat(yh + er * w))
    return float(np.mean(np.abs(ts) >= abs(t0)))


def inst(f):
    d = pd.read_csv(R9 / f)
    name = "cbc" if "cbc" in f else "tud"
    out = []
    for c, g in d.groupby("REF_AREA"):
        if c not in ISO3:
            continue
        s = g.set_index("TIME_PERIOD").OBS_VALUE.sort_index()
        s = s.reindex(range(int(s.index.min()), int(s.index.max()) + 1)).interpolate(limit_area="inside")
        out.append(pd.DataFrame({"geo": AM_ISO.get(c, c), "year": s.index, name: s.to_numpy()}))
    return pd.concat(out)


# ---------------------------------------------------------------- 2.4
def cross_section():
    k = pd.read_csv(ROOT / "data" / "raw" / "euklems" / "national_accounts.csv", low_memory=False,
                    usecols=lambda c: c in ("nace_r2_code", "geo_code", "year", "GO_PI", "II_PI", "GO_CP", "II_CP"))
    k = k.rename(columns={"nace_r2_code": "ind", "geo_code": "geo"})
    k = k[k.ind.str.match(r"^[A-U]\d") | k.ind.isin(list("ABCDEFGHIJKLMNOPQRSTU"))].copy()
    k = k.sort_values(["geo", "ind", "year"])
    g = k.groupby(["geo", "ind"])
    k["dp"] = np.log(k.GO_PI) - np.log(g.GO_PI.shift(5))
    k["dc"] = (np.log(k.II_PI) - np.log(g.II_PI.shift(5))) * (k.II_CP / k.GO_CP)
    e = k.replace([np.inf, -np.inf], np.nan).dropna(subset=["dp", "dc"])
    e = e[e.year.isin([2005, 2010, 2015, 2020])]
    r2 = []
    for (geo, y), d in e.groupby(["geo", "year"]):
        if len(d) >= 10:
            r2.append(np.corrcoef(d.dp - d.dp.mean(), d.dc - d.dc.mean())[0, 1] ** 2)
    return dict(item="goods: median R2 of 5-year output price change on unit intermediate-cost change",
                est=float(np.median(r2)), n=len(r2))


AM = ameco(["HWCDW", "NLHA", "PCPH", "RVGDE", "ZUTN"])
AM_ISO = {k: k for k in AM.geo.unique()}
AM_TO2 = {g: ISO3.get(g) for g in AM.geo.unique()}
UNEMP_A = {AM_TO2[g]: d.set_index("year").ZUTN.dropna() for g, d in AM.groupby("geo") if AM_TO2.get(g)}

if __name__ == "__main__":
    pd.set_option("display.width", 250)
    part = sys.argv[1] if len(sys.argv) > 1 else "all"
    if part in ("pt", "all"):
        G = goods_pt()
        G.to_csv(OUT / "s2_goods_lp.csv", index=False)
        Wc = wage_pt("C")
        Wc.to_csv(OUT / "s2_wage_lp.csv", index=False)
        Wz = wage_pt("_Z")
        Wz.to_csv(OUT / "s2_wage_lp_total.csv", index=False)
        IV = wage_pt_iv()
        IV.to_csv(OUT / "s2_wage_iv.csv", index=False)
        O = outcomes_pt(G, Wc)
        O.to_csv(OUT / "s2_pt_outcomes.csv", index=False)
        print(O.round(3).to_string(), flush=True)
    if part in ("infl", "all"):
        D, O = inflation_2021()
        D.to_csv(OUT / "s2_2021.csv", index=False)
        O.to_csv(OUT / "s2_2021_outcomes.csv", index=False)
        print(D.round(3).to_string(), "\n", O.round(4).to_string(), flush=True)
    if part in ("anchor", "all"):
        EG, O = anchor()
        EG.to_csv(OUT / "s2_anchor_eg.csv", index=False)
        O.to_csv(OUT / "s2_anchor.csv", index=False)
        print(EG.round(3).to_string(), "\n", O.round(4).to_string(), flush=True)
        cs = cross_section()
        pd.DataFrame([cs]).to_csv(OUT / "s2_cross_section.csv", index=False)
        print(cs, flush=True)
