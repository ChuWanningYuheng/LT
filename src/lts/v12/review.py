"""Iteration 12: checks requested by the independent review (results/v12/review_independent.md; journal 10).

All descriptive and after the results; pre-registered labels 75-89 are not changed.

Usage: PYTHONPATH=src python -P -m lts.v12.review
Outputs: results/v12/rv_*.csv
"""
from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd

from ..v6.series import ROOT, fa
from ..v8.rule import decide
from . import stage1 as s1

OUT = ROOT / "results" / "v12"
R12 = ROOT / "data" / "raw" / "v12"


def tr(series, base, name, **kw):
    return dict(item=name, base=base, **s1.trend(np.log(series), **kw))


# ================================================================ R1, R2, R4, R12 (stage 1)
def inv_price_nfc():
    c, q = fa(4, "FAAt407-A", 37), fa(4, "FAAt408-A", 37)
    return c / (q * c.loc[2017] / q.loc[2017])


def inv_price_private():
    c, q = fa(1, "FAAt105-A", 3), fa(1, "FAAt106-A", 3)
    return c / (q * c.loc[2017] / q.loc[2017])


def stage1_checks():
    d = s1.us_nfc()
    p = s1.us_private()
    rows, contrib = [], []
    # R1: only sigma and F held (rho counted as part of the tendency)
    for b in s1.BASES:
        rows.append(tr(s1.counterfactual(d.r_full, [d.sigma, d.F], b), b, "R1 NFC: r_cf with sigma and F fixed"))
    # R2: net and gross value added over K
    for nm, s in (("net VA / K", d.Y / d.K), ("gross VA / K (P_Y Q_Y / K)", d.P_Y * d.Q_Y / d.K)):
        rows.append(dict(item=f"R2 NFC: ln change 1950-2024 of {nm}", est=float(np.log(s.loc[2024] / s.loc[1950]))))
    # R4: price of new capital goods instead of the stock price
    nipa_inv = s1.nipa1("A008RD").reindex(d.index)
    for pk_name, PI in (("NIPA A008RD (private nonresidential investment)", nipa_inv),
                        ("BEA FAAt407/408 l.37 (NFC investment)", inv_price_nfc().reindex(d.index))):
        rho = d.P_Y / PI
        q = (d.P_Y * d.Q_Y / d.K) / rho
        assert (d.sigma * d.n * rho * q / d.r - 1).abs().max() < 1e-9
        for b in s1.BASES:
            cf = s1.counterfactual(d.r_full, [d.sigma, rho, d.F], b)
            rows.append(tr(cf, b, f"R4 NFC [{pk_name}]: 75 ln r_cf"))
            rows.append(tr(cf / d.r_full.loc[b:], b, f"R4 NFC [{pk_name}]: 76 ln(r_cf / r_full)"))
            rows.append(tr(q.loc[b:], b, f"R4 NFC [{pk_name}]: 78 ln q"))
        c = s1.contributions({"rho": rho, "q": q})
        contrib.append(c.assign(variant=pk_name))
    for pk_name, PI in (("NIPA A007RD (private fixed investment)", s1.nipa1("A007RD").reindex(p.index)),
                        ("BEA FAAt105/106 l.3 (private investment)", inv_price_private().reindex(p.index))):
        relp = (p.P * PI.loc[2017] / PI) / p.P.loc[2017]
        for b in s1.BASES:
            cf = s1.counterfactual(p.rB_full, [p.s_share, p.unprod, relp, p.F], b)
            rows.append(tr(cf, b, f"R4 private [{pk_name}]: 77 ln r_B,cf"))
        c = s1.contributions({"P/P_K": relp})
        contrib.append(c.assign(variant=pk_name))
    # R12: F after 1997
    rows.append(tr(d.F.loc[1997:], 1997, "R12 NFC: trend of ln F, 1997-2024", delta=0.002, direction=">"))
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "rv_stage1.csv", index=False)
    pd.concat(contrib, ignore_index=True).to_csv(OUT / "rv_stage1_contributions.csv", index=False)
    return R


# ================================================================ R5 (stage 2.2)
def cluster_F(d, x, insts, ctrl):
    from ..v11.stage1 import demean
    d = d.dropna(subset=[x, "dRP"] + ctrl + insts).copy()
    d[x] = d[x] / d[x].std()
    M = demean(d[[x] + ctrl + insts].to_numpy(float), [d.yr.to_numpy()])
    xe, C, Z = M[:, 0], M[:, 1:1 + len(ctrl)], M[:, 1 + len(ctrl):]
    Zf = np.c_[Z, C]
    b = np.linalg.lstsq(Zf, xe, rcond=None)[0]
    e = xe - Zf @ b
    gi = pd.factorize(d.country)[0]
    G = gi.max() + 1
    XtX = np.linalg.inv(Zf.T @ Zf)
    meat = sum(np.outer(Zf[gi == g].T @ e[gi == g], Zf[gi == g].T @ e[gi == g]) for g in range(G))
    V = XtX @ meat @ XtX * G / (G - 1) * (len(e) - 1) / (len(e) - Zf.shape[1])
    k = Z.shape[1]
    bz = b[:k]
    F = float(bz @ np.linalg.solve(V[:k, :k], bz) / k)
    return F, int(G), len(d)


def stage2_checks():
    from . import stage2 as s2
    d = s2.panel(5)
    geo = pd.read_excel(R12 / "cepii_geo_cepii.xls")
    ll = geo.drop_duplicates("iso3").set_index("iso3").landlocked
    c00 = d[d.year == 2000].set_index("country").commodity_share_X
    d["commodity_share_2000"] = d.country.map(c00)
    rows = []
    for fix in (False, True):
        m = ll.rename(index={"ROM": "ROU"}) if fix else ll
        d["landlocked"] = d.country.map(m).astype(float)
        tag = "ROM -> ROU" if fix else "as in stage 2 (ROU missing)"
        for x in ("O_raw", "O_adj"):
            F, G, n = cluster_F(d, x, ["commodity_share_2000", "landlocked"], s2.CTRL)
            r = s2.iv(d, x, ["commodity_share_2000", "landlocked"])
            rows.append(dict(item=f"R5 {x} 2SLS [{tag}]", **r, cluster_first_stage_F=F))
    # R6: Taiwan per-worker years in PWT
    p = s2.pwt()
    tw = p[p.country == "TWN"].set_index("year").loc[2010:2016, ["lnP", "per_worker"]]
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "rv_stage2.csv", index=False)
    tw.to_csv(OUT / "rv_stage2_twn_pwt.csv")
    return R


# ================================================================ R8, R9 (stage 4)
def content_model(v):
    from . import stage4 as s4
    f = s4.VERS[v][0]
    z = zipfile.ZipFile(R12 / f)
    name = [n for n in z.namelist() if n.endswith("Content Model Reference.txt")][0]
    d = pd.read_csv(io.StringIO(z.open(name).read().decode("latin-1")), sep="\t")
    return d[d["Element ID"].isin(["4.C.3.b.8", "4.C.3.b.7", "4.C.3.a.4", "4.C.3.d.3"])].assign(version=v)


def stage4_checks():
    from . import stage4 as s4
    W = pd.concat([content_model(v) for v in ("15.0", "30.0")], ignore_index=True)
    W.to_csv(OUT / "rv_stage4_wording.csv", index=False)
    a, da, sa = s4.work_context("15.0")
    b, db, sb = s4.work_context("30.0")
    C = pd.read_csv(OUT / "s4_within_15.0_30.0.csv", index_col=0)
    inc = (sa.reindex(C.index)[["freedom", "structured"]].eq("Incumbent").all(1)
           & sb.reindex(C.index)[["freedom", "structured"]].eq("Incumbent").all(1))
    rows = []
    for nm, mask in (("all re-rated (as outcome 87)", pd.Series(True, index=C.index)), ("Incumbent in both versions", inc)):
        for item in ("autonomy", "freedom", "structured", "pace", "repeat"):
            est, lo, hi, n = s4.boot_mean(C.loc[mask, item])
            delta, direction = 0.1, ("<" if item in ("autonomy", "freedom", "structured") else ">")
            rows.append(dict(item=f"R8 {item}, {nm}", est=est, ci90_lo=lo, ci90_hi=hi, n=n,
                             label=decide(est, lo, hi, 0, delta, direction)))
    # R9: Job Zone 30.0 applied to 2004 occupations (exact 6-digit match)
    jz30 = s4.job_zones("30.0")
    soc30 = jz30.groupby(jz30.index.str[:7]).mean().round()
    o4 = pd.read_csv(OUT / "s4_oews_2004.csv")
    o23 = pd.read_csv(OUT / "s4_oews_2023.csv")
    tot4 = pd.read_csv(OUT / "s4_oews_2004.csv").emp.sum()
    o4["jz30"] = o4.OCC_CODE.map(soc30)
    m = o4.dropna(subset=["jz30"])
    def shares(e, z):
        return {f"share_jz{k}": float(e[z == k].sum() / e.sum()) for k in range(1, 6)}
    rows9 = [dict(item="2004, JZ 10.0, all matched to 10.0", n_occ=len(o4), **shares(o4.emp, o4.jz)),
             dict(item="2004, JZ 10.0, subset matched to 30.0", n_occ=len(m), coverage_of_2004_sample=float(m.emp.sum() / tot4),
                  **shares(m.emp, m.jz)),
             dict(item="2004, JZ 30.0, subset matched to 30.0", n_occ=len(m), **shares(m.emp, m.jz30)),
             dict(item="2023, JZ 30.0", n_occ=len(o23), **shares(o23.emp, o23.jz))]
    S = pd.DataFrame(rows9)
    S.loc[len(S)] = dict(item="share of 2004 matched employment that changed Job Zone (10.0 -> 30.0)",
                         n_occ=len(m), share_jz1=float(m.emp[m.jz != m.jz30].sum() / m.emp.sum()))
    rng = np.random.default_rng(89)
    def sh3(e, z, idx):
        return e[idx][z[idx] == 3].sum() / e[idx].sum()
    e4, z4, e23, z23 = m.emp.to_numpy(), m.jz30.to_numpy(), o23.emp.to_numpy(), o23.jz.to_numpy()
    bs = [sh3(e23, z23, rng.integers(0, len(e23), len(e23))) - sh3(e4, z4, rng.integers(0, len(e4), len(e4)))
          for _ in range(2000)]
    est = float(S.share_jz3.iloc[3] - S.share_jz3.iloc[2])
    lo, hi = np.quantile(bs, [0.05, 0.95])
    rows.append(dict(item="R9 89 with JZ 30.0 for both years (2004 exact-code subset)", est=est, ci90_lo=float(lo),
                     ci90_hi=float(hi), n=len(e4) + len(e23), label=decide(est, lo, hi, 0, 0.02, "<")))
    S.to_csv(OUT / "rv_stage4_jobzones.csv", index=False)
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "rv_stage4.csv", index=False)
    return R


# ================================================================ R10 (stage 3, real output)
def stage3_checks():
    from ..revisit import wiod
    from ..v11.stage1 import reg
    s = wiod.sea()
    s = s[s.variable.isin(["GO", "GO_PI"])].pivot_table(index=["country", "code", "year"], columns="variable", values="v")
    s = s.reset_index()
    s["year"] = s.year.astype(int)
    cls = pd.read_csv(OUT / "s3_classification_2000.csv")
    s = s.merge(cls, on=["country", "code"], how="inner")
    s = s[(s.GO > 0) & (s.GO_PI > 0)]
    s["GOr"] = s.GO / s.GO_PI * 100
    g = s.groupby(["country", "year"])
    agg = pd.DataFrame({"XIr": g.apply(lambda x: x.GOr[x.dept_I].sum(), include_groups=False),
                        "XIIr": g.apply(lambda x: x.GOr[~x.dept_I].sum(), include_groups=False),
                        "XIn": g.apply(lambda x: x.GO[x.dept_I].sum(), include_groups=False),
                        "XIIn": g.apply(lambda x: x.GO[~x.dept_I].sum(), include_groups=False)}).reset_index()
    agg = agg.sort_values(["country", "year"])
    agg["R_real"] = np.log(agg.XIr / agg.XIIr)
    agg["R_sea_nom"] = np.log(agg.XIn / agg.XIIn)
    for v in ("R_real", "R_sea_nom"):
        agg["d3_" + v] = agg.groupby("country")[v].diff(3)
    D = pd.read_csv(OUT / "s3_panel.csv").merge(agg[["country", "year", "d3_R_real", "d3_R_sea_nom"]],
                                                 on=["country", "year"], how="left")
    sm = D[D.year.between(2003, 2014)].dropna(subset=["Y", "d3_R", "d3_inv", "d3_R_real", "d3_R_sea_nom"]).copy()
    rows = []
    for v in ("d3_R", "d3_R_sea_nom", "d3_R_real"):
        sm[v + "_z"] = sm[v] / sm[v].std()
    sm["d3_inv_z"] = sm.d3_inv / sm.d3_inv.std()
    for v, nm in (("d3_R", "WIOT nominal (as 83), common sample"), ("d3_R_sea_nom", "SEA nominal"),
                  ("d3_R_real", "SEA real (GO / GO_PI)")):
        for fe in (("country",), ("country", "yr")):
            r = reg(sm, "Y", [v + "_z", "d3_inv_z"], 0, fe=fe, cl="country")
            rows.append(dict(item=f"R10 83: {nm}, FE {'+'.join(fe)}", **r,
                             label=decide(r["est"], r["ci90_lo"], r["ci90_hi"], 0, 0.02, ">")))
    corr = float(sm[["d3_R", "d3_R_real"]].corr().iloc[0, 1])
    rows.append(dict(item="R10 correlation of D (WIOT nominal) and D (SEA real)", est=corr, n=len(sm)))
    yr = sm.groupby("year")[["d3_R", "d3_R_real"]].mean()
    yr.to_csv(OUT / "rv_stage3_by_year.csv")
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "rv_stage3.csv", index=False)
    return R


# ================================================================ R11 (stage 5)
def stage5_checks():
    O = pd.read_csv(OUT / "s5_exploratory.csv")
    P = pd.read_csv(OUT / "s5_panel.csv")
    rows = []
    for _, r in O.iterrows():
        sd_level = float(P[r.policy].std())
        delta = 0.1 * sd_level
        rows.append(dict(item=f"R11 {r.policy}: delta = 0.1 SD of the level", est=r.est, ci90_lo=r.ci90_lo,
                         ci90_hi=r.ci90_hi, sd_level=sd_level, delta=delta, delta_used=r.delta,
                         label=decide(r.est, r.ci90_lo, r.ci90_hi, 0, delta, ">"), label_used=r.label))
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "rv_stage5.csv", index=False)
    return R


# ================================================================ R12, R13 (stage 6, stage 2 units)
def misc_checks():
    from .stage6 import COUNTRIES, klems
    k = klems().reset_index()
    tot = k[(k.nace_r2_code == "TOT") & k.year.between(2010, 2021)].groupby("geo_code").kva.median().rename("K_GFCF/VA_CP, TOT, median 2010-2021")
    tot.to_csv(OUT / "rv_klems_kva.csv")
    v10 = pd.read_csv(ROOT / "results" / "v10" / "s10_rent_d_pp_actual_wage.csv").set_index("country")
    rows = [dict(item="R13 iteration 10 d(L) (flow-based pp, country means over its years), same 11 countries: mean",
                 est=float(v10.loc[list(COUNTRIES), "L"].mean()), n=len(COUNTRIES))]
    # stage 2 units: H / EMP after imputation, 2010
    from ..v9.stage34 import year_system
    from .stage2 import patch_china
    _, cells = year_system(2010)
    cells, info = patch_china(cells, 2010)
    for c in ("CHN", "USA", "IND"):
        x = cells[(cells.country == c) & (cells.EMP > 0)]
        rows.append(dict(item=f"R12 H / EMP after imputation, {c}, 2010 (median over industries)", est=float((x.H / x.EMP).median())))
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "rv_misc.csv", index=False)
    return R


def run():
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 90)
    for f in (stage1_checks, stage2_checks, stage4_checks, stage3_checks, stage5_checks, misc_checks):
        R = f()
        cols = [c for c in ("item", "base", "est", "ci90_lo", "ci90_hi", "n", "G", "label", "cluster_first_stage_F",
                            "first_stage_F", "delta", "label_used") if c in R]
        print(R[cols].round(4).to_string(), flush=True)


if __name__ == "__main__":
    run()
