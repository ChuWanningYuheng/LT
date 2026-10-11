"""Iteration 12b: checks requested by the independent review (pre_registration_v12b.md, journal 14). Descriptive.

R1   persons within bridge groups: (a) one group, compensation allocator; (b) within groups by output x
R1/R12  hours vs money where industry persons exist (NIPA detail variant), 1998-2023
R5   US 2000-2014: BEA (stage 1) vs WIOD-USA
R6   Maito (2014) statistical annex RoPW vs our GDP-weighted world mean
R7   stage 4: g on the peak dummy controlling for d ln Y_real; peak-year decomposition on a common sample
R8   insurance multiplier 1 + SELF/SAL (K65, OECD Table 7 USA) by year
R11  omega seam 1986/87: omega before 1987 x f, f = omega(NAICS 1987) / omega(SIC 1987)
R17  country contributions to the cumulative weights index (outcome 99)

Usage: PYTHONPATH=src python -P -m lts.v12b.review [r1|other|all]
Outputs: results/v12b/rv_*.csv
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from ..v6.series import ROOT
from ..v8.rule import decide
from ..v11.stage1 import reg
from . import stage1 as s1
from . import us_io as u
from .stage1 import trend

OUT = ROOT / "results" / "v12b"


def _trends(d, name, cols=("r_h", "k_h", "e_h"), ratio=True):
    rows = []
    for c, dr in zip(cols, ("<", ">", ">")):
        rows.append(dict(check=name, item=f"trend ln {c}", **trend(np.log(d[c]), 4, 0.002, dr)))
    if ratio:
        rows.append(dict(check=name, item="trend ln(r_h / r_$)", **trend(np.log(d.r_h / d["r_$"]), 4, 0.002, "<")))
    return rows


# ---------------------------------------------------------------- R1
def r1():
    orig = u.persons

    def one_group(year, method="group"):
        base = orig(year, "group")
        alloc = u.comp_allocator(year).clip(lower=0)
        return float(base.sum()) * alloc / alloc.sum()

    def by_output(year, method="group"):
        base = orig(year, "group")
        x = pd.Series(u.year_tables(year)["x"], index=u.CODES).clip(lower=0)
        out = pd.Series(0.0, index=u.CODES)
        for cs in u.GROUPS.values():
            out[cs] = float(base[cs].sum()) * x[cs] / x[cs].sum()
        return out

    rows = []
    for name, f in (("(a) one group, compensation allocator", one_group), ("(b) within groups by output x", by_output)):
        u.persons = f
        try:
            for y in u.YEARS:
                r = s1.year_values(y)
                r["variant"] = name
                rows.append(r)
        finally:
            u.persons = orig
        print("r1", name, flush=True)
    V = pd.DataFrame(rows)
    S = pd.read_csv(OUT / "s1_us_series.csv")
    dec = S[S.variant == "main"][["year", "QK", "QC"]]
    V = V.merge(dec, on="year", how="left")
    V.to_csv(OUT / "rv_r1_series.csv", index=False)
    out = []
    for name, g in V.groupby("variant"):
        g = g.set_index("year").sort_index()
        out += _trends(g, f"R1 {name}")
        for nm, s, dr in (("(a) Q_K/L", g.QK / g.L, ">"), ("(b) h_K/h_C", (g.C_h / g.QK) / (g.V_h / g.QC), ">"),
                          ("(c) Q_C/L", g.QC / g.L, ">")):
            out.append(dict(check=f"R1 {name}", item=f"trend ln {nm}", **trend(np.log(s), 4, 0.002, dr)))
        out.append(dict(check=f"R1 {name}", item="corr(r_h, r_$)", est=float(g.r_h.corr(g["r_$"]))))
        m = S[S.variant == "main"].set_index("year").r_h
        out.append(dict(check=f"R1 {name}", item="corr(r_h, main r_h)", est=float(g.r_h.corr(m))))
    return out


# ---------------------------------------------------------------- other checks
def r1_detail_r5():
    S = pd.read_csv(OUT / "s1_us_series.csv")
    out = []
    for v in ("main", "labour: NIPA detail from 1998"):
        g = S[S.variant == v].set_index("year").sort_index().loc[1998:2023]
        out += _trends(g, f"R1/R12 {v}, 1998-2023")
    m = S[S.variant == "main"].set_index("year").sort_index().loc[2000:2014]
    out += _trends(m, "R5 BEA (stage 1 main), 2000-2014", cols=("r_h",))
    out.append(dict(check="R5 BEA (stage 1 main), 2000-2014", item="trend ln r_$", **trend(np.log(m["r_$"]), 4, 0.002, "<")))
    w = pd.read_csv(OUT / "s1w_wiod.csv")
    w = w[w.country == "USA"].set_index("year").sort_index()
    out += _trends(w, "R5 WIOD-USA, 2000-2014", cols=("r_h",))
    out.append(dict(check="R5 WIOD-USA, 2000-2014", item="trend ln r_$", **trend(np.log(w["r_$"]), 4, 0.002, "<")))
    return out


def maito_annex():
    pdf = ROOT / "data" / "raw" / "v12b" / "maito2014_mpra55894.pdf"
    with tempfile.TemporaryDirectory() as td:
        txt = f"{td}/m.txt"
        subprocess.run(["pdftotext", "-layout", str(pdf), txt], check=True)
        lines = open(txt, encoding="utf-8", errors="replace").read().splitlines()
    pat = re.compile(r"^\s*(\d{4})\s+([\d,]+)%\s+([\d,]+)%\s+([\d,]+)%\s+(\d{4})\s+([\d,]+)%\s+([\d,]+)%\s+([\d,]+)%\s*$")
    rows = []
    for ln in lines:
        m = pat.match(ln)
        if m:
            g = m.groups()
            for k in (0, 4):
                rows.append(dict(year=int(g[k]), RoPM=float(g[k + 1].replace(",", ".")) / 100,
                                 RoPW=float(g[k + 2].replace(",", ".")) / 100, RoP_CH=float(g[k + 3].replace(",", ".")) / 100))
    return pd.DataFrame(rows).drop_duplicates("year").set_index("year").sort_index()


def r6():
    M = maito_annex()
    M.to_csv(OUT / "rv_maito_annex.csv")
    W = pd.read_csv(OUT / "s2_world_series.csv", index_col=0)
    ours = W["PWT xr balanced, GDP-weighted mean (Maito-style)"]
    rw = W["PWT xr balanced"]
    out = [dict(check="R6 Maito annex", item="years parsed", est=float(len(M)), y0=int(M.index.min()), y1=int(M.index.max()))]
    for a, b in ((1960, 2010), (1960, 1980), (1982, 2007), (1980, 2010)):
        for nm, s in (("Maito RoPW", M.RoPW), ("ours, GDP-weighted mean", ours), ("ours, sum P / sum K", rw)):
            out.append(dict(check="R6 Maito annex", item=f"trend ln {nm}, {a}-{b}", **trend(np.log(s.loc[a:b]), 4, 0.002, "<")))
    for y in (1960, 1980, 1982, 2007, 2010):
        out.append(dict(check="R6 Maito annex", item=f"level {y}: Maito RoPW / ours GDP-weighted", est=float(M.RoPW.loc[y]),
                        ci90_lo=float(ours.loc[y]) if y in ours.index else np.nan))
    return out


def _event_reg_ctrl(d, y, xs):
    d = d.dropna(subset=[y] + xs)
    X = sm.add_constant(d[xs].to_numpy(float))
    h = sm.OLS(d[y].to_numpy(), X).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
    o = sm.OLS(d[y].to_numpy(), X).fit()
    b = float(o.params[1])
    w = max(stats.norm.ppf(0.95) * h.bse[1], stats.t.ppf(0.95, len(d) - len(xs) - 1) * o.bse[1])
    return dict(est=b, ci90_lo=b - w, ci90_hi=b + w, n=len(d), events=int(d[xs[0]].sum()))


def r7():
    out = []
    U = pd.read_csv(OUT / "s4_us_panel.csv")
    J = pd.read_csv(OUT / "s4_jst_panel.csv")
    r = _event_reg_ctrl(U, "g", ["peak", "dY_real"])
    out.append(dict(check="R7 US", item="g on peak, controlling d ln Y_real: peak coefficient", **r,
                    label=decide(r["est"], r["ci90_lo"], r["ci90_hi"], 0, 0.02, "<")))
    o = sm.OLS(U.dropna(subset=["g", "dY_real"]).g, sm.add_constant(U.dropna(subset=["g", "dY_real"])[["peak", "dY_real"]])).fit()
    out.append(dict(check="R7 US", item="elasticity of g to d ln Y_real (same regression)", est=float(o.params["dY_real"])))
    x = J.dropna(subset=["g", "peak", "dY_real"])
    r = reg(x, "g", ["peak", "dY_real"], 0, fe=("iso",), cl="iso")
    out.append(dict(check="R7 JST", item="g on peak, controlling d ln Y_real: peak coefficient", **r, events=int(x.peak.sum()),
                    label=decide(r["est"], r["ci90_lo"], r["ci90_hi"], 0, 0.02, "<")))
    r2 = reg(x, "g", ["peak", "dY_real"], 1, fe=("iso",), cl="iso")
    out.append(dict(check="R7 JST", item="elasticity of g to d ln Y_real (same regression)", **r2))
    cols = ["g", "dshare", "dYstar", "du", "dws"]
    Uc = U.dropna(subset=cols)
    for c in cols:
        out.append(dict(check="R7 US common sample", item=f"peak year: {c} minus other years",
                        **_event_reg_ctrl(Uc, c, ["peak"])))
    Jc = J.dropna(subset=cols)
    for c in cols:
        r = reg(Jc, c, ["peak"], 0, fe=("iso",), cl="iso")
        out.append(dict(check="R7 JST common sample", item=f"peak year: {c} minus other years", **r, events=int(Jc.peak.sum())))
    return out


def r8():
    from .. import oecd
    t7 = oecd.series("USA", "table7", "SELF", "JB")
    s7 = oecd.series("USA", "table7", "SAL", "JB")
    col = "K65"
    m = (1 + t7[col] / s7[col]).loc[1998:2023]
    out = [dict(check="R8 insurance", item=f"1 + SELF/SAL K65, {y}", est=float(v), ci90_lo=float(t7.loc[y, col]),
                ci90_hi=float(s7.loc[y, col])) for y, v in m.items()]
    out.append(dict(check="R8 insurance", item="range 1998-2023 (outcome 7)", est=float(m.min()), ci90_lo=float(m.min()),
                    ci90_hi=float(m.max())))
    out.append(dict(check="R8 insurance", item="2022 (outcome 18)", est=float(m.loc[2022])))
    return out


def r11():
    S = pd.read_csv(OUT / "s1_us_series.csv")
    m = S[S.variant == "main"].set_index("year").sort_index().copy()
    prod = [c for c in u.CODES if c not in u.UNPRODUCTIVE]
    C = u.sic_blocks()["comp"]
    alloc = u.comp_allocator(1987).clip(lower=0)
    sic = pd.Series(0.0, index=u.CODES)
    for g, cs in u.GROUPS.items():
        sic[cs] = float(C.loc[1987, g]) * alloc[cs] / alloc[cs].sum()
    w_sic, w_naics = float(sic[prod].sum()), float(u.compensation(1987)[prod].sum())
    f = w_naics / w_sic
    pre = m.index < 1987
    for X, tot in (("h", m.L), ("$", m["tot_$"])):
        V = m[f"V_{X}"].where(~pre, m[f"V_{X}"] * f)
        Sx = tot - V
        m[f"e_{X}"], m[f"k_{X}"], m[f"r_{X}"] = Sx / V, m[f"C_{X}"] / V, Sx / (m[f"C_{X}"] + V)
    out = [dict(check="R11 omega seam", item="W_prod 1987: SIC / NAICS; f = NAICS / SIC", est=f, ci90_lo=w_sic, ci90_hi=w_naics)]
    out += _trends(m, "R11 omega spliced at 1987")
    return out


def r17():
    from .stage2 import balanced, pwt
    b = balanced(pwt(), "xr")
    b = b.assign(r_c=b.P_xr / b.K_xr, wk=b.K_xr / b.groupby("year").K_xr.transform("sum"))
    P = b.pivot(index="year", columns="c", values="r_c")
    Wk = b.pivot(index="year", columns="c", values="wk")
    base = float((Wk * P).sum(1).iloc[0])
    between = (Wk.diff() * P).sum(0) / base
    within = (Wk.shift(1) * P.diff()).sum(0) / base
    D = pd.DataFrame({"between_cum_2019": between, "within_cum_2019": within}).sort_values("between_cum_2019")
    D.to_csv(OUT / "rv_r17_country_contributions.csv")
    out = [dict(check="R17 weights index", item=f"{c}: contribution to cumulative weights index", est=float(v))
           for c, v in pd.concat([D.between_cum_2019.head(5), D.between_cum_2019.tail(5)]).items()]
    out.append(dict(check="R17 weights index", item="total cumulative weights index 2019", est=float(between.sum())))
    return out


def run(part="all"):
    rows = []
    if part in ("r1", "all"):
        rows += r1()
    if part in ("other", "all"):
        rows += r1_detail_r5() + r6() + r7() + r8() + r11() + r17()
    D = pd.DataFrame(rows)
    f = OUT / f"rv_checks_{part}.csv"
    D.to_csv(f, index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 80)
    pd.set_option("display.max_rows", 300)
    cols = [c for c in ("check", "item", "est", "ci90_lo", "ci90_hi", "n", "label") if c in D]
    print(D[cols].round(5).to_string())


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "all")
