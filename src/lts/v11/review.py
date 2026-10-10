"""Iteration 11: checks requested by the independent review (results/v11/review_independent.md; journal 5).
All after the results; pre-registered labels do not change.

R1   outcome 69: both Engle-Granger normalisations, Johansen (k_ar_diff 1, 2), OLS slopes of i on r
R3   leave-one-country-out DD (WIOD; FIGARO with recomputed signals)
R4   FIGARO signals recomputed from the current economies (lts.economy.build) for all 13 countries; table vs current
     comparison; FIGARO DD, components, placebo (50 permutations fixed per country, seed 11 + sum(ord))
     + earlier outcomes 1, 2, 5, 60, 61 with JPN rows from tables rebuilt with current data (tag robust_jpnfix)
R5   placebo pre-trend: ln H(s) - ln H(s-3) on signals at s, no controls (both panels)
R6   placebo centres (WIOD from s1_placebo_perm.csv; FIGARO from the new permutations)
R7   group coefficients inside the DD model; fully interacted fixed effects; clusters by country; without IDN, TWN
R8   pooled b_v, b_pp at h = 1..5 (WIOD)
R9   outcome 70 split: d r, d i_st, d i_lt, d pi at recession onset and in JST crisis years
R10  real Aaa around peaks; F capitalised at Aaa minus trailing 5-year CPI inflation
R11  2.3 with annual Z.1 flows (S11_1_i_a) where the quarterly ones are missing; years with data
R13  continuous sigma rescaled to the actual mean contrast of the two groups
R16  JST events with complete windows of both series
R18  F with MV - FA in the numerator
R20  credit/GDP in crises split into d ln loans and d ln GDP

Usage: PYTHONPATH=src python -P -m lts.v11.review [stage1|figaro|stage2|jpn|all]
Outputs: results/v11/rv_*.csv
"""
from __future__ import annotations

import sys
import zipfile

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from ..v8.rule import decide, share_row
from ..v8.stage4 import US_PEAKS
from . import stage1 as s1
from . import stage2 as s2

OUT = s1.OUT
TAB = s1.ROOT / "results" / "tables"
B_RV = 1999


# ================================================================ stage 1
def wiod_panel(h=3, s_years=range(2001, 2011)):
    sig = pd.read_parquet(OUT / "s1_signals.parquet")
    d, _ = s1.prepare(sig)
    return d, s1.build_panel(d, h, s_years=s_years)


def group_coefs(p, y="R_hours", sv="Sv", spp="Spp", ctrl=s1.CTRL, fe=("ind", "cy"), cl="ind"):
    """b_v and b_pp of each group as single coefficients of one regression (the DD model)"""
    p = p.assign(grp=s1.groups(p))
    p = p[p.grp.isin(["se", "corp"])].dropna(subset=[y, sv, spp] + ctrl)
    p = s1.std(p, [sv, spp])
    D = (p.grp == "se").astype(float)
    cols = {"bv_corp": p[sv] * (1 - D), "bpp_corp": p[spp] * (1 - D), "bv_se": p[sv] * D, "bpp_se": p[spp] * D, "D": D}
    cols.update({f"{c}_c": p[c] * (1 - D) for c in ctrl})
    cols.update({f"{c}_s": p[c] * D for c in ctrl})
    p = p.assign(**cols)
    xs = list(cols)
    return {k: s1.reg(p, y, xs, i, fe=fe, cl=cl, B=B_RV) for i, k in enumerate(["bv_corp", "bpp_corp", "bv_se", "bpp_se"])}


def dd_custom(p, y="R_hours", sv="Sv", spp="Spp", ctrl=s1.CTRL, fe=("ind", "cy"), cl="ind", interact_fe=False):
    p = p.assign(grp=s1.groups(p))
    p = p[p.grp.isin(["se", "corp"])].dropna(subset=[y, sv, spp] + ctrl)
    p = s1.std(p, [sv, spp])
    D = (p.grp == "se").astype(float)
    p = p.assign(U=p[sv] + p[spp], W=p[sv] - p[spp], D=D)
    p = p.assign(WD=p.W * D, UD=p.U * D, **{f"{c}D": p[c] * D for c in ctrl})
    if interact_fe:
        p = p.assign(**{f"{f}_g": p[f].astype(str) + "_" + p.grp for f in fe})
        fe = tuple(f"{f}_g" for f in fe)
        xs = ["WD", "W", "UD", "U"] + ctrl + [f"{c}D" for c in ctrl]
    else:
        xs = ["WD", "W", "UD", "U", "D"] + ctrl + [f"{c}D" for c in ctrl]
    return s1.labelled(s1.reg(p, y, xs, 0, fe=fe, cl=cl, B=B_RV, scale=2.0), 0.02, ">")


def pretrend_panel(d, s_years, h=3):
    """response ln H(s) - ln H(s-h) with signals at s"""
    p = s1.build_panel(d, h, s_years=s_years)
    k = d[~d.ind.isin(s1.EXCL)].set_index(["country", "ind", "year"])
    a = k.hours.reindex(pd.MultiIndex.from_arrays([p.country, p.ind, p.year - h])).to_numpy()
    b = k.hours.reindex(pd.MultiIndex.from_arrays([p.country, p.ind, p.year])).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        p["R_past"] = np.where((a > 0) & (b > 0), np.log(b) - np.log(np.where(a > 0, a, np.nan)), np.nan)
    return p


def stage1_checks():
    rows = []
    d, P3 = wiod_panel()
    # R7 group coefficients inside the DD model
    for k, v in group_coefs(P3).items():
        rows.append(dict(item=f"R7 WIOD DD model: {k}", **v))
    # R7 variants of the DD
    rows.append(dict(item="R7 WIOD DD: fully interacted FE (group x industry, group x country-year)",
                     **dd_custom(P3, interact_fe=True)))
    rows.append(dict(item="R7 WIOD DD: clusters by country", **dd_custom(P3, cl="country")))
    rows.append(dict(item="R7 WIOD DD: without IDN, TWN", **dd_custom(P3[~P3.country.isin(["IDN", "TWN"])])))
    # R13 continuous sigma on the actual contrast
    g = s1.groups(P3)
    q = P3[g.isin(["se", "corp"])]
    contrast = float(q[g[q.index] == "se"].sigma.mean() - q[g[q.index] == "corp"].sigma.mean())
    r = s1.dd_cont(P3, "R_hours", B=B_RV)
    sc = contrast / (s1.SE_HI - s1.CORP_LO)
    r = dict(r, est=r["est"] * sc, ci90_lo=r["ci90_lo"] * sc, ci90_hi=r["ci90_hi"] * sc, se=r["se"] * sc)
    rows.append(dict(item=f"R13 continuous sigma, slope x actual contrast {contrast:.3f}", **s1.labelled(r, 0.02, ">")))
    # R8 pooled coefficients by horizon
    for h in (1, 2, 3, 4, 5):
        Ph = s1.build_panel(d, h, s_years=range(2001, 2014 - h))
        for k, v in s1.coefs(Ph, "R_hours", ["Sv", "Spp"], B=B_RV).items():
            rows.append(dict(item=f"R8 WIOD pooled h = {h}: b_{k}", **v))
    # R5 pre-trend (no controls), same specification of the DD
    pp = pretrend_panel(d, range(2003, 2011))
    rows.append(dict(item="R5 WIOD DD pre-trend: ln H(s) - ln H(s-3), signal at s, no controls, s = 2003-2010",
                     **dd_custom(pp, y="R_past", ctrl=[])))
    rows.append(dict(item="R5 WIOD DD future response on the same signal years, no controls",
                     **dd_custom(pp, y="R_hours", ctrl=[])))
    # R3 leave one country out (WIOD)
    for c in sorted(P3.country.unique()):
        r = s1.dd(P3[P3.country != c], "R_hours", B=B_RV)
        rows.append(dict(item=f"R3 WIOD DD without {c}", **s1.labelled(r, 0.02, ">")))
        print("loo", c, round(r["est"], 4), flush=True)
    # R6 placebo centre (WIOD)
    pl = pd.read_csv(OUT / "s1_placebo_perm.csv")
    dd0 = float(pd.read_csv(OUT / "s1_outcomes.csv").est.iloc[0])
    rows.append(dict(item="R6 WIOD placebo DD: mean, SD; DD minus placebo mean", est=float(pl.dd.mean()),
                     se=float(pl.dd.std()), est2=dd0 - float(pl.dd.mean()), n=len(pl)))
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "rv_stage1.csv", index=False)
    return R


# ================================================================ FIGARO with recomputed signals (R4)
def figaro_signals():
    from ..economy import build
    from ..levels import labour_vectors
    rows, prow, chk = [], [], []
    old = {}
    for f in ("main", "robust"):
        r = pd.read_parquet(TAB / f"ratios_long_{f}.parquet", columns=["basis", "industry", "z", "capital", "closed",
                                                                        "imports", "labour", "country", "year"])
        r = r[(r.capital.astype(str) == "True") & (r.closed.astype(str) == "False") & (r.imports == "price") &
              (r.labour == "hours") & r.basis.isin(["labour", "pp_actual_wage"])]
        old[f] = r
    old = pd.concat(old.values()).set_index(["country", "year", "basis", "industry"]).z
    for c in s1.FIG_COUNTRIES:
        perms = None
        for y in range(2010, 2023):
            try:
                e, _ = build(c, y)
            except Exception as exc:                             # noqa: BLE001
                print("skip", c, y, exc, flush=True)
                continue
            if perms is None:
                rng = np.random.default_rng(11 + sum(map(ord, c)))
                perms = np.vstack([rng.permutation(e.n) for _ in range(50)])
            lv = labour_vectors(e, c, y)
            S = s1.Setup(e)
            r = e.actual_profit_rate(True)
            pp = e.prices_of_production(r, True, "price", "actual")
            z = {k: S.z(lv[k][None])[0] for k in ("hours", "edu_years", "edu_train")}
            zP = S.z(lv["hours"][perms])
            rows.append(pd.DataFrame(dict(country=c, year=y, ind=e.labels, x=e.x, hours=e.hours,
                                          persons=e.meta["persons"], z_hours=z["hours"], z_edu_years=z["edu_years"],
                                          z_edu_train=z["edu_train"], pp_actual_wage=pp)))
            prow.append(pd.DataFrame(zP.T, columns=[f"placebo_perm_{k:03d}" for k in range(50)]))
            for basis, new in (("labour", z["hours"]), ("pp_actual_wage", pp)):
                o = old.reindex(pd.MultiIndex.from_arrays([[c] * e.n, [y] * e.n, [basis] * e.n, e.labels])).to_numpy()
                ok = np.isfinite(o) & (o > 0) & (new > 0)
                chk.append(dict(country=c, year=y, basis=basis, max_abs_diff=float(np.nanmax(np.abs(o - new))) if ok.any() else np.nan,
                                corr_ln=float(np.corrcoef(np.log(o[ok]), np.log(new[ok]))[0, 1]) if ok.sum() > 2 else np.nan,
                                n=int(ok.sum())))
            print("fig", c, y, flush=True)
    W = pd.concat(rows, ignore_index=True)
    Pm = pd.concat(prow, ignore_index=True)
    W = pd.concat([W, Pm], axis=1)
    W.to_parquet(OUT / "rv_figaro_signals.parquet")
    pd.DataFrame(chk).to_csv(OUT / "rv_figaro_table_check.csv", index=False)
    return W


def figaro_panel_new(W, sig_wiod, h=3, s_years=range(2011, 2019), past=False):
    w = W.copy()
    sg = s1.sigma(sig_wiod)
    emp = sig_wiod[sig_wiod.year.between(2000, 2002)].groupby(["country", "ind"]).EMP.sum()

    def sig_for(c, lab):
        codes = s1.fig_wiod_code(lab)
        s = sg.reindex([(c, k) for k in codes])
        e = emp.reindex([(c, k) for k in codes])
        ok = s.notna().to_numpy() & e.notna().to_numpy()
        return float(np.average(s[ok], weights=e[ok])) if ok.any() and e[ok].sum() > 0 else np.nan
    lab = w[["country", "ind"]].drop_duplicates()
    lab["sigma"] = [sig_for(c, l_) for c, l_ in zip(lab.country, lab.ind)]
    w = w.merge(lab, on=["country", "ind"], how="left")
    excl = w.ind.map(lambda l_: any(x in s1.FIG_EXCL for x in l_.split("+")))
    w = w[~excl].copy()
    zc = [c for c in w.columns if c.startswith("z_") or c == "pp_actual_wage" or c.startswith("placebo_perm_")]
    with np.errstate(divide="ignore", invalid="ignore"):
        S = {"S" + c: -np.log(w[c].where(w[c] > 0)) for c in zc}
    w = pd.concat([w.reset_index(drop=True), pd.DataFrame(S).reset_index(drop=True)], axis=1)
    w["lint"] = np.log((w.hours / w.x).where((w.hours > 0) & (w.x > 0)))
    k = w.set_index(["country", "ind", "year"])

    def lag(col, j):
        return k[col].reindex(pd.MultiIndex.from_arrays([w.country, w.ind, w.year + j])).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        for col in ("hours", "persons"):
            a, b = lag(col, 1), lag(col, 1 + h)
            w[f"R_{col}"] = np.where((a > 0) & (b > 0), np.log(b) - np.log(np.where(a > 0, a, np.nan)), np.nan)
        a, b = lag("hours", -h), lag("hours", 0)
        w["R_past"] = np.where((a > 0) & (b > 0), np.log(b) - np.log(np.where(a > 0, a, np.nan)), np.nan)
    w["lint_lag"] = w.lint
    w["dlint"] = w.lint - lag("lint", -1)
    w["cy"] = w.country + "_" + w.year.astype(str)
    return w[w.year.isin(list(s_years))]


def figaro_checks():
    sig = pd.read_parquet(OUT / "s1_signals.parquet")
    W = figaro_signals()
    F = figaro_panel_new(W, sig)
    rows = []
    sv, spp = "Sz_hours", "Spp_actual_wage"
    rows.append(dict(item="R4 FIGARO (recomputed signals) DD, hours",
                     **s1.labelled(s1.dd(F, "R_hours", sv=sv, spp=spp, B=B_RV), 0.02, ">")))
    rows.append(dict(item="R4 FIGARO (recomputed) corporate b_pp - b_v",
                     **s1.labelled(s1.dd(F, "R_hours", sv=sv, spp=spp, which="corp", B=B_RV), 0.02, ">")))
    for k, v in s1.coefs(F, "R_hours", [sv, spp], B=B_RV).items():
        rows.append(dict(item=f"R4/R8 FIGARO (recomputed) pooled b_{k}", **v))
    for k, v in group_coefs(F, sv=sv, spp=spp).items():
        rows.append(dict(item=f"R4/R7 FIGARO (recomputed) DD model: {k}", **v))
    for zc in ("Sz_edu_years", "Sz_edu_train"):
        rows.append(dict(item=f"R4 FIGARO (recomputed) DD, reduction {zc[3:]}",
                         **s1.labelled(s1.dd(F, "R_hours", sv=zc, spp=spp, B=B_RV), 0.02, ">")))
    rows.append(dict(item="R4 FIGARO (recomputed) DD, persons",
                     **s1.labelled(s1.dd(F, "R_persons", sv=sv, spp=spp, B=B_RV), 0.02, ">")))
    # placebo (50 permutations fixed per country)
    pc = [c for c in F.columns if c.startswith("Splacebo_perm_")]
    fp = pd.Series([s1.dd(F, "R_hours", sv=c, spp=spp, boot=False)["est"] for c in pc])
    dd0 = rows[0]["est"]
    rows.append(dict(item="R4/R6 FIGARO (recomputed) placebo DD: mean, SD; share >= DD; DD minus mean",
                     est=float(fp.mean()), se=float(fp.std()), est2=float(dd0 - fp.mean()),
                     share_ge=float((fp >= dd0).mean()), p95=float(fp.quantile(0.95)), n=len(fp)))
    # R5 pre-trend (no controls), signal years 2013-2018
    Fp = figaro_panel_new(W, sig, s_years=range(2013, 2019))
    rows.append(dict(item="R5 FIGARO DD pre-trend: ln H(s) - ln H(s-3), no controls, s = 2013-2018",
                     **dd_custom(Fp, y="R_past", sv=sv, spp=spp, ctrl=[])))
    rows.append(dict(item="R5 FIGARO DD future response, same years, no controls",
                     **dd_custom(Fp, y="R_hours", sv=sv, spp=spp, ctrl=[])))
    # R3 leave one country out
    for c in s1.FIG_COUNTRIES:
        r = s1.dd(F[F.country != c], "R_hours", sv=sv, spp=spp, B=B_RV)
        rows.append(dict(item=f"R3 FIGARO (recomputed) DD without {c}", **s1.labelled(r, 0.02, ">")))
        print("fig loo", c, round(r["est"], 4), flush=True)
    g = s1.groups(F)
    rows.append(dict(item="R3 FIGARO cells by group and country (se)",
                     **{f"se_{c}": int(((g == "se") & (F.country == c)).sum()) for c in s1.FIG_COUNTRIES}))
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "rv_figaro.csv", index=False)
    return R


# ================================================================ earlier outcomes with rebuilt JPN tables (R4)
def jpn_outcomes():
    from ..report_tables import kind, table_levels
    from ..v8.rule import mean_row
    from ..v10 import stage10 as s10
    ms = []
    for tag in ("main", "robust", "robust_jpnfix"):
        m = pd.read_csv(TAB / f"metrics_long_{tag}.csv", low_memory=False)
        m["closed"] = m["closed"].astype(str)
        m["tag"] = tag
        ms.append(m)
    m = pd.concat(ms, ignore_index=True)
    m["kind"] = kind(m.basis)
    old = m[m.tag.isin(["main", "robust"])]
    new = m[(m.tag == "main") | ((m.tag == "robust") & (m.country != "JPN")) | (m.tag == "robust_jpnfix")]
    rows = []
    for nm, mm in (("tables as used", old), ("JPN rebuilt", new)):
        lv = table_levels(mm)
        op, cl = lv[lv["система"] == "открытая"], lv[lv["система"] == "замкнутая (единая з/п)"]
        r1 = mean_row("1", op["доля товаров лучше труда"].to_numpy(), 0.5, 0.10, "<")
        r2 = mean_row("2", cl["доля товаров лучше труда"].to_numpy(), 0.5, 0.10, "<")
        r5 = share_row("5", (op["PP факт. з/п"] < op["труд (часы)"]).to_numpy(), 0.5, 0.15, "<")
        j = op[op["страна"] == "JPN"].iloc[0]
        jc = cl[cl["страна"] == "JPN"].iloc[0]
        for o, r in (("1", r1), ("2", r2), ("5", r5)):
            rows.append(dict(item=f"outcome {o} [{nm}]", est=r["est"], ci90_lo=r["ci90_lo"], ci90_hi=r["ci90_hi"],
                             n=r["n"], label=r["label"]))
        rows.append(dict(item=f"JPN [{nm}]: share commodity better (open / closed uniform); MAWD labour; MAWD PP actual",
                         est=float(j["доля товаров лучше труда"]), est2=float(jc["доля товаров лучше труда"]),
                         mawd_labour=float(j["труд (часы)"]), mawd_pp=float(j["PP факт. з/п"])))
    # outcomes 60, 61 with JPN from the rebuilt table
    orig = s10.d_frame

    def d_frame_fix(tag, basis):
        d = orig(tag, basis)
        if tag == "robust":
            d = pd.concat([d[d.country != "JPN"], orig("robust_jpnfix", basis)], ignore_index=True)
        return d
    for nm, fn in (("tables as used", orig), ("JPN rebuilt", d_frame_fix)):
        s10.d_frame = fn
        D = s10.figaro_d("pp_actual_wage")
        s = s10.stat(D, s10.RENT)
        lo, hi = s10.boot_countries(s.to_numpy())
        rows.append(dict(item=f"outcome 60 [{nm}]", est=float(s.mean()), ci90_lo=lo, ci90_hi=hi, n=len(s),
                         label=decide(float(s.mean()), lo, hi, 0, 0.05, ">"), jpn=float(s.get("JPN", np.nan))))
        ci = s10.commodity_index()
        Bd = D[D.industry == "B"][["country", "year", "d"]].merge(ci, left_on="year", right_index=True)
        r61 = s10.fe_wild(Bd, "d", "ln_em", 61)
        rows.append(dict(item=f"outcome 61 [{nm}]", **r61))
    s10.d_frame = orig
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "rv_jpn_earlier_outcomes.csv", index=False)
    return R


# ================================================================ stage 2
def coint_table():
    J = s2.jst()
    d = J.merge(s2.pwt_r(), on=["iso", "year"], how="left")
    d["i_lt_real"] = d.ltrate / 100 - d.pi
    w = d[d.year.between(s2.Y0, s2.Y1)]
    rows = []
    for iso, g in w.groupby("iso"):
        g = g.dropna(subset=["i_lt_real", "r_pwt"])
        _, p_ir, _ = coint(g.i_lt_real, g.r_pwt, trend="c", maxlag=4, autolag="aic")
        _, p_ri, _ = coint(g.r_pwt, g.i_lt_real, trend="c", maxlag=4, autolag="aic")
        X = np.c_[g.r_pwt, g.i_lt_real]
        ranks = {}
        for k in (1, 2):
            jo = coint_johansen(X, 0, k)
            rank = 0
            for j in range(2):
                if jo.lr1[j] > jo.cvt[j, 0]:                   # 90% critical value of the trace statistic
                    rank = j + 1
                else:
                    break
            ranks[k] = rank
        A = np.c_[np.ones(len(g)), g.r_pwt]
        b = np.linalg.lstsq(A, g.i_lt_real, rcond=None)[0]
        e = g.i_lt_real - A @ b
        se = np.sqrt((e @ e) / (len(g) - 2) * np.linalg.inv(A.T @ A)[1, 1])
        rows.append(dict(iso=iso, eg_i_on_r_p=p_ir, eg_r_on_i_p=p_ri, johansen_rank_k1=ranks[1],
                         johansen_rank_k2=ranks[2], slope_i_on_r=b[1], t_slope=b[1] / se))
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "rv_69_coint.csv", index=False)
    out = [share_row("69 [EG, i on r: as used]", (T.eg_i_on_r_p < 0.10).to_numpy(), 0.5, 0.15, "<"),
           share_row("69 [EG, r on i: reverse normalisation]", (T.eg_r_on_i_p < 0.10).to_numpy(), 0.5, 0.15, "<"),
           share_row("69 [Johansen trace 90%, k_ar_diff = 1: rank exactly 1]", (T.johansen_rank_k1 == 1).to_numpy(),
                     0.5, 0.15, "<"),
           share_row("69 [Johansen trace 90%, k_ar_diff = 2: rank exactly 1]", (T.johansen_rank_k2 == 1).to_numpy(),
                     0.5, 0.15, "<"),
           share_row("69 [EG i on r significant AND slope > 0]", ((T.eg_i_on_r_p < 0.10) & (T.slope_i_on_r > 0)).to_numpy(),
                     0.5, 0.15, "<")]
    return pd.DataFrame(out)


def stage2_checks():
    rows = []
    C = coint_table()
    rows += C.assign(item=C.outcome).drop(columns=["outcome"]).to_dict("records")
    # R9 decomposition of 70
    J = s2.jst()
    d = J.merge(s2.pwt_r(), on=["iso", "year"], how="left").sort_values(["iso", "year"])
    d["i_st"], d["i_lt"] = d.stir / 100, d.ltrate / 100
    rec = set(map(tuple, pd.read_csv(OUT / "s2_70_recession_years.csv")[["iso", "year"]].to_numpy()))
    d["rec"] = [float((i, y) in rec) for i, y in zip(d.iso, d.year)]
    d["crisis"] = d.crisisJST.astype(float)
    for c in ("r_pwt", "i_st", "i_lt", "pi"):
        d[f"d_{c}"] = d.groupby("iso")[c].diff()
    w = d[d.year.between(s2.Y0 + 1, s2.Y1)]
    for ev in ("rec", "crisis"):
        for c in ("r_pwt", "i_st", "i_lt", "pi"):
            x = w.dropna(subset=[f"d_{c}", ev])
            r = s1.reg(x, f"d_{c}", [ev], 0, fe=("iso",), cl="iso", B=B_RV)
            rows.append(dict(item=f"R9 70 split: d {c} in {ev} years minus other years", **r))
    # R10 Aaa, inflation and real Aaa before peaks; F with expected-inflation-adjusted rate; R18 MV - FA
    Z = s2.z1()
    aaa = s2.fred("AAA") / 100
    cpi = s2.fred("CPIAUCNS")
    pi = cpi / cpi.shift(1) - 1
    pie = pi.rolling(5).mean()
    for name, x in (("Aaa", aaa), ("CPI inflation", pi), ("real Aaa (ex post)", aaa - pi)):
        ch = [x.get(P - 1, np.nan) - x.get(P - 4, np.nan) for P in US_PEAKS]
        rows.append(dict(item=f"R10 change of {name} from P-4 to P-1 (mean over 11 peaks)",
                         **s2.mean_ci(ch, US_PEAKS, 0.005, ">", B=B_RV)))
    rate = (aaa - pie).reindex(Z.index)
    rows.append(dict(item="R10 years 1946-2024 with Aaa minus trailing 5-year inflation <= 0",
                     est=float((rate.loc[1946:2024] <= 0).sum()), n=int(rate.loc[1946:2024].notna().sum())))
    for name, x in (("F real (MV x (Aaa - trailing 5y CPI inflation) / NOS)", Z.MV * rate.where(rate > 0) / Z.NOS),
                    ("F with MV - FA (Aaa)", (Z.MV - Z.FA) * aaa.reindex(Z.index) / Z.NOS)):
        x = x.loc[1945:2024].dropna()
        gbar = float(np.log(x).diff().loc[1946:2024].mean())
        pre = [s2.abnormal(x, P, gbar)[0] for P in US_PEAKS]
        rows.append(dict(item=f"R10/R18 72 variant: {name}", **s2.mean_ci(pre, US_PEAKS, 0.05, ">", B=B_RV)))
    # R16 JST events with complete windows of both series; R20 credit split
    JE = pd.read_csv(OUT / "s2_jst_events.csv")
    ok = JE[["pd_pre", "ly_pre", "pd_crash", "ly_crash"]].notna().all(1)
    rows.append(dict(item="R16 73 on events with complete windows of both series",
                     **s2.mean_ci(JE.pd_pre[ok], JE.iso[ok], 0.05, ">", B=B_RV)))
    rows.append(dict(item="R16 74 on events with complete windows of both series",
                     **s2.mean_ci(JE.ly_pre[ok], JE.iso[ok], 0.02, ">", B=B_RV)))
    J["lnL"], J["lnY"] = np.log(J.tloans), np.log(J.gdp)
    ev = []
    for iso, g in J.groupby("iso"):
        g = g.set_index("year")
        gl, gy = float(g.lnL.diff().mean()), float(g.lnY.diff().mean())
        for t in g.index[g.crisisJST == 1]:
            ev.append(dict(iso=iso, year=t, dlnL=g.lnL.get(t + 1, np.nan) - g.lnL.get(t - 1, np.nan) - 2 * gl,
                           dlnY=g.lnY.get(t + 1, np.nan) - g.lnY.get(t - 1, np.nan) - 2 * gy))
    E = pd.DataFrame(ev)
    for c in ("dlnL", "dlnY"):
        rows.append(dict(item=f"R20 crisis t-1 -> t+1 abnormal {c}", **s2.mean_ci(E[c], E.iso, 0.02, "<", B=B_RV)))
    # R11 2.3 with annual flows where quarterly ones are missing
    z = zipfile.ZipFile(s2.ROOT / "data" / "raw" / "v6" / "z1" / "z1_csv_files.zip")
    ia = pd.read_csv(z.open("csv/S11_1_i_a.csv")).set_index("date")
    ia.index = ia.index.astype(int)
    num = lambda s: pd.to_numeric(s.replace("ND", np.nan), errors="coerce")   # noqa: E731
    tr_a, rv_a = num(ia["FU104090005.A"]), num(ia["FR104090005.A"])
    tr = Z.FA_tr.combine_first(tr_a)
    rv = Z.FA_rev.combine_first(rv_a)
    dFA = Z.FA.diff()
    for a, b in ((1946, 2025), (1946, 1959)):
        tot = dFA.loc[a:b].sum()
        rows.append(dict(item=f"R11 2.3 [{a}-{b}] with annual flows for missing years: transactions / revaluation / other",
                         est=float(tr.loc[a:b].sum() / tot), est2=float(rv.loc[a:b].sum() / tot),
                         est3=float((dFA.loc[a:b] - tr.loc[a:b] - rv.loc[a:b]).sum() / tot)))
    rows.append(dict(item="R11 max |quarterly sum - annual| for transactions / revaluation, 1952-2025",
                     est=float((Z.FA_tr - tr_a).loc[1952:2025].abs().max()),
                     est2=float((Z.FA_rev - rv_a).loc[1952:2025].abs().max())))
    sh = (Z.FA_rev / Z.FA.shift(1)).loc[1946:2025]
    recy = {p + 1 for p in US_PEAKS}
    other = sh[[y not in recy for y in sh.index]]
    rows.append(dict(item="R11 revaluation / FA(t-1), other years: mean and years with data",
                     est=float(other.mean()), n=int(other.notna().sum())))
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "rv_stage2.csv", index=False)
    return R


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 110)
    cols = ["item", "est", "ci90_lo", "ci90_hi", "n", "G", "label"]
    for key, fn in (("stage2", stage2_checks), ("jpn", jpn_outcomes), ("stage1", stage1_checks), ("figaro", figaro_checks)):
        if what in (key, "all"):
            R = fn()
            print(R[[c for c in cols if c in R]].round(4).to_string(), flush=True)
