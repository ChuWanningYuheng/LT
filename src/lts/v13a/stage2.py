"""Iteration 13a, stage 2: bridge to the literature where the Marxian rate of profit falls (pre_registration_v13a.md,
'Этап 2'; definitions fixed before the analogues in results/v13a/s2_definitions.md). Outcome 119.

Works: Kliman (2009), Shaikh (2016), Paitaridis & Tsoulfidis (2012), Dumenil & Levy (uslt4x), Basu et al. (2025);
Mohun (2014) - text values only (no TOST, not in the share).
Bridge (steps i-vi): authors' analogue on our data -> r_m-type money rate of our productive industries (all workers in
V, current cost, (P)) -> V of rank-and-file workers -> historical valuation -> hours -> denominator (B) = r*_h.
For the world (Basu et al.) the comparison is with the 12b world series 'PWT ppp balanced' (PPP, as the authors'
main series); 'PWT xr balanced' - descriptive.

Usage: PYTHONPATH=src python -P -m lts.v13a.stage2  ->  results/v13a/s2_*.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..v6.series import ROOT, RAW
from ..v8.rule import wilson
from . import est
from . import stage0 as s0
from . import stage1 as s1

OUT = ROOT / "results" / "v13a"
V13 = RAW / "v13"
D = 0.002

# Paitaridis & Tsoulfidis (2012), RRPE 44(2), table 1, rows 'PT' (annual growth rates, percent)
PT_TABLE = {
    "R": {(1965, 1982): -2.20, (1982, 1989): 3.37, (1982, 2007): 1.08, (1964, 1989): -0.51, (1964, 2007): -0.22},
    "C/v": {(1965, 1982): 2.69, (1982, 1989): -0.33, (1982, 2007): 0.56, (1964, 1989): 1.65, (1964, 2007): 1.34},
    "RSV": {(1965, 1982): 0.49, (1982, 1989): 3.03, (1982, 2007): 1.64, (1964, 1989): 1.13, (1964, 2007): 1.12},
    "r": {(1948, 1965): -0.23, (1965, 1982): -6.29, (1982, 1989): 6.60, (1982, 2007): 1.67, (1948, 1989): -3.80,
          (1964, 1989): -2.03, (1964, 2007): -1.29},
}
# Mohun (2014), RRPE 46(3): values named in the text (section 4-5)
MOHUN_TEXT = [
    ("supervisory wages / money value added", 1964, 0.221),
    ("supervisory wages / money value added", 2000, 0.355),
    ("unproductive working-class wages / money value added", 1964, 0.129),
    ("unproductive working-class wages / money value added", 2010, 0.129),
    ("productive working class / total employment", 1964, 0.58),
    ("productive wages / total wages", 1966, 0.506),
    ("productive wages / total wages", 1970, 0.476),
    ("productive wages / total wages", 1979, 0.474),
    ("productive wages / total wages", 2000, 0.352),
]


# ================================================================ readers
def nipa_flat(codes):
    d = pd.read_csv(RAW / "v9b" / "bea_NipaDataA.txt", dtype=str)
    d.columns = ["code", "year", "v"]
    d = d[d.code.isin(codes)]
    d["v"] = pd.to_numeric(d.v.str.replace(",", ""), errors="coerce")
    d["year"] = d.year.astype(int)
    return d.pivot(index="year", columns="code", values="v")


def fa_line(path, sheet, line):
    x = pd.read_excel(path, sheet_name=sheet, header=None)
    hdr = x.index[x.iloc[:, 0].astype(str).str.strip() == "Line"][0]
    h = pd.to_numeric(x.iloc[hdr], errors="coerce")
    cols = [i for i in range(2, x.shape[1]) if np.isfinite(h.iloc[i])]
    row = x[pd.to_numeric(x.iloc[:, 0], errors="coerce") == line].iloc[0]
    return pd.Series(pd.to_numeric(row.iloc[cols], errors="coerce").values, index=h.iloc[cols].astype(int).values)


def arch_line(tables, label):
    """NIPA archive (section 6, vintage 28.06.2024): first line with this label, spliced B (to 1987), C (1988-1997),
    D (1998-)."""
    out = {}
    for tab, (a, b) in zip(tables, ((1948, 1987), (1988, 1997), (1998, 2022))):
        row = s0._first(s0.arch(tab), label)
        for y in range(a, b + 1):
            if y in row.index:
                out[y] = float(row[y])
    return pd.Series(out)


# ================================================================ authors' series
def kliman_author():
    x = pd.read_excel(V13 / "literature" / "kliman_2009_persistent_fall_data_graphs.xls", sheet_name="ROP data",
                      header=None)
    hdr = 3
    assert str(x.iloc[hdr, 7]).strip() == "PIH / HCFA" and str(x.iloc[hdr, 5]).strip() == "PBT / HCFA"
    body = x.iloc[hdr + 1:]
    yr = pd.to_numeric(body.iloc[:, 0], errors="coerce")
    ok = yr.notna()
    return pd.DataFrame({"PIH_HCFA": pd.to_numeric(body.iloc[:, 7], errors="coerce")[ok].values,
                         "PBT_HCFA": pd.to_numeric(body.iloc[:, 5], errors="coerce")[ok].values},
                        index=yr[ok].astype(int).values)


def shaikh_author():
    x = pd.read_excel(V13 / "shaikh_capitalism" / "Appendix_6.8_Data_Tables_Corrected.xlsx",
                      sheet_name="Appndx 6.8.II.7", header=None)
    hdr = 18
    cols = {i: int(v) for i, v in x.iloc[hdr].items() if isinstance(v, float) and np.isfinite(v) and v >= 1947}
    def row(var):
        r = x[x.iloc[:, 3].astype(str).str.strip() == var].iloc[0]
        return pd.Series({y: pd.to_numeric(r[i], errors="coerce") for i, y in cols.items()})
    return pd.DataFrame({"rcorpnipa": row("rcorpnipa"), "rcorp": row("rcorp")})


def dl_author():
    return s0.uslt4x()


def basu_author(name="cl_global__EPWT_PPP_LimitCountries.csv"):
    d = pd.read_csv(V13 / "world_profitability_dashboard" / name)
    return d.set_index("year").ROP / 100.0


# ================================================================ our analogues
def kliman_ours():
    n = nipa_flat(["A451RC", "A442RC"])
    f6 = V13 / "bea_fa_tables" / "Section6All_xls.xlsx"
    hcfa = fa_line(f6, "FAAt603-A", 2)
    dhc = fa_line(f6, "FAAt606-A", 2)
    pih = n.A451RC - dhc.reindex(n.index) - n.A442RC
    return (pih / hcfa.shift(1).reindex(n.index)).dropna()


def shaikh_ours():
    n = nipa_flat(["A445RC"])
    k = fa_line(V13 / "bea_fa_tables" / "Section6All_xls.xlsx", "FAAt601-A", 2)
    return (n.A445RC / k.shift(1).reindex(n.index)).dropna()


def rd_ae_stock():
    """Current-cost stock of R&D and entertainment originals, all FA industries (FA Detailed stk1), $ million."""
    v = s1.vint()
    m = np.array([str(a).startswith(("RD", "AE")) for a in v["asset"]])
    return pd.Series(np.nansum(v["K1"][m], axis=0), index=v["years"])


def dl_ours(no_rd=False):
    n = nipa_flat(["A191RC", "A765RC", "W276RC", "W279RC", "B952RC", "B1033C"])
    comp = arch_line(("T60200B", "T60200C", "T60200D"), "private industries")
    fte = arch_line(("T60500B", "T60500C", "T60500D"), "private industries")
    se = arch_line(("T60700B", "T60700C", "T60700D"), "self-employed persons")
    yrs = [y for y in range(1948, 2023) if y in comp.index and y in fte.index and y in se.index]
    ndp = (n.A191RC - n.A765RC - n.W276RC - n.W279RC - n.B952RC).reindex(yrs)
    wl = ((comp - n.B1033C.reindex(comp.index)) * (fte + se) / fte).reindex(yrs)
    kn = fa_line(RAW / "v6" / "bea_fa" / "Section1All_xls.xlsx", "FAAt101-A", 4).reindex(yrs)
    if no_rd:
        kn = kn - rd_ae_stock().reindex(yrs)
    return ((ndp - wl) / kn).dropna()


def pt_ours(no_rd=False):
    """ST-type general rate R = S / (C + V), C/v, RSV (definitions file)."""
    n = nipa_flat(["A191RC", "A765RC", "B701RC", "B702RC"])
    yrs = list(range(1948, 2024))
    mva = (n.A191RC - n.A765RC - n.B701RC - n.B702RC).reindex(yrs)
    V = pd.Series({y: float(np.nansum(s1.wages().loc[y].to_numpy(float) * s1.ces()["sW"].loc[y].fillna(1.0).to_numpy(float)
                                      * s1.PROD)) for y in yrs})
    C = (fa_line(RAW / "v6" / "bea_fa" / "Section1All_xls.xlsx", "FAAt101-A", 4)
         + fa_line(RAW / "v6" / "bea_fa" / "Section7All_xls.xlsx", "FAAt701-A", 79)).reindex(yrs)
    if no_rd:
        C = C - rd_ae_stock().reindex(yrs)
    S = mva - V
    return pd.DataFrame({"R": S / (C + V), "C/v": C / V, "RSV": S / V})


def mohun_ours():
    """Our analogues of the shares named by Mohun (2014). Y = NDP - general government compensation (general
    government GVA less its consumption of fixed capital) - imputed rental of owner-occupied housing."""
    n = nipa_flat(["A191RC", "A262RC", "A765RC", "A265RC", "DOWNRC"])
    sNt = pd.read_csv(OUT / "s0_ces_shares.csv", dtype={"code": str})
    sNt = sNt[sNt.variant == "main"].pivot_table(index="year", columns="code", values="sN").reindex(columns=s1.CODES)
    priv = ~s1.GOV
    rows = []
    for y in range(1964, 2023):
        ws = s1.wages().loc[y].to_numpy(float)
        sw = s1.ces()["sW"].loc[y].fillna(1.0).to_numpy(float)
        Y = float(n.A191RC[y] - n.A262RC[y] - (n.A765RC[y] - n.A265RC[y]) - n.DOWNRC[y])
        pns_w = ws * sw
        persons = s1.u.persons(y).to_numpy(float)
        sn = sNt.loc[y].fillna(1.0).to_numpy(float)
        rows.append({"year": y,
                     "supervisory wages / money value added": float(np.nansum((ws - pns_w)[priv])) / Y,
                     "unproductive working-class wages / money value added": float(np.nansum(pns_w[priv & ~s1.PROD])) / Y,
                     "productive working class / total employment": float(np.nansum((persons * sn)[s1.PROD]))
                     / float(np.nansum(persons[priv])),
                     "productive wages / total wages": float(np.nansum(pns_w[s1.PROD])) / float(np.nansum(ws[priv]))})
    return pd.DataFrame(rows).set_index("year")


# ================================================================ comparisons
def compare_series(name, ours, author, window=None):
    """TOST on the trend of ln(ours / author) over the common window; trends of each; level correlation."""
    idx = sorted(set(ours.dropna().index) & set(author.dropna().index))
    if window:
        idx = [y for y in idx if window[0] <= y <= window[1]]
    o, a = ours.reindex(idx), author.reindex(idx)
    d_ = est.single_trend(np.log(o / a), D, "<")
    to = est.single_trend(np.log(o), D, "<")
    ta = est.single_trend(np.log(a), D, "<")
    rep = bool(d_["ci90_lo"] >= -D and d_["ci90_hi"] <= D)
    return dict(work=name, y0=idx[0], y1=idx[-1], n=len(idx), corr_levels=float(np.corrcoef(o, a)[0, 1]),
                trend_ours=to["est"], trend_ours_lo=to["ci90_lo"], trend_ours_hi=to["ci90_hi"], label_ours=to["label_full"],
                trend_author=ta["est"], trend_author_lo=ta["ci90_lo"], trend_author_hi=ta["ci90_hi"],
                label_author=ta["label_full"], diff=d_["est"], diff_lo=d_["ci90_lo"], diff_hi=d_["ci90_hi"],
                reproduced=rep)


def compare_number(name, ours, period, author_pct, var):
    a, b = period
    y = np.log(ours.loc[a:b])
    t = est.single_trend(y, D, "<")
    g = author_pct / 100.0
    lo, hi = t["ci90_lo"] - g, t["ci90_hi"] - g
    endpoint = float((ours.loc[b] / ours.loc[a]) ** (1 / (b - a)) - 1)
    return dict(work=name, variable=var, y0=a, y1=b, n=len(y), trend_ours=t["est"], trend_ours_lo=t["ci90_lo"],
                trend_ours_hi=t["ci90_hi"], endpoint_growth_ours=endpoint, author=g, diff=t["est"] - g, diff_lo=lo,
                diff_hi=hi, reproduced=bool(lo >= -D and hi <= D))


def bridge(window):
    """Steps after (i): r_m-type money rate -> PNS V -> historical -> hours -> (B); trend and label on the window."""
    M = s1.money_series()
    ms = pd.read_csv(OUT / "s1_main_series.csv", index_col=0)
    steps = [("ii: r_m-тип (деньги, все наёмные в V, текущая, (П))", (M.NDP - M.V_all) / (M.C_curr_P + M.V_all)),
             ("iii: V рядовых", (M.NDP - M.V) / (M.C_curr_P + M.V)),
             ("iv: историческая оценка", (M.NDP - M.V) / (M.C_hist_P + M.V)),
             ("v: часы", ms.r_P),
             ("vi: знаменатель (В) = r*_h", ms.r)]
    rows = []
    for nm, ser in steps:
        s_ = ser.loc[window[0]:window[1]]
        t = est.single_trend(np.log(s_), D, "<")
        rows.append(dict(step=nm, y0=window[0], y1=window[1], est=t["est"], ci90_lo=t["ci90_lo"], ci90_hi=t["ci90_hi"],
                         label=t["label_full"]))
    return rows


def run():
    comp, nums, br, desc = [], [], [], []
    # Kliman
    ka, ko = kliman_author(), kliman_ours()
    comp.append(compare_series("Kliman (2009): PIH / HCFA", ko, ka.PIH_HCFA))
    # Shaikh
    sa, so = shaikh_author(), shaikh_ours()
    comp.append(compare_series("Shaikh (2016): rcorpnipa", so, sa.rcorpnipa))
    desc.append(dict(item="Shaikh (2016): rcorp (adjusted; not reproduced) vs our rcorpnipa analogue",
                     **{k: v for k, v in compare_series("x", so, sa.rcorp).items() if k != "work"}))
    # Dumenil-Levy
    da, do = dl_author(), dl_ours()
    comp.append(compare_series("Duménil–Lévy (uslt4x): r", do, da["R"], (1948, 2014)))
    desc.append(dict(item="Duménil–Lévy: our analogue without R&D and originals in KN",
                     **{k: v for k, v in compare_series("x", dl_ours(True), da["R"], (1948, 2014)).items() if k != "work"}))
    # Basu et al.
    w12 = pd.read_csv(ROOT / "results" / "v12b" / "s2_world_series.csv").set_index("year")
    comp.append(compare_series("Basu et al. (2025): Global EPWT PPP LimitCountries", w12["PWT ppp balanced"],
                               basu_author()))
    desc.append(dict(item="Basu et al.: XR LimitCountries vs 12b PWT xr balanced",
                     **{k: v for k, v in compare_series("x", w12["PWT xr balanced"],
                                                        basu_author("cl_global__EPWT_XR_LimitCountries.csv")).items()
                        if k != "work"}))
    # Paitaridis & Tsoulfidis: table numbers
    P = pt_ours()
    Pn = pt_ours(True)
    for var in ("R", "C/v", "RSV"):
        for per, g in PT_TABLE[var].items():
            nums.append(compare_number("Paitaridis & Tsoulfidis (2012)", P[var], per, g, var))
            nums.append(dict(**compare_number("Paitaridis & Tsoulfidis (2012), без НИОКР и оригиналов", Pn[var], per, g,
                                              var)))
    pt_main = [r for r in nums if r["work"] == "Paitaridis & Tsoulfidis (2012)" and r["variable"] == "R"
               and (r["y0"], r["y1"]) == (1964, 2007)][0]
    # outcome 119
    flags = [c["reproduced"] for c in comp] + [pt_main["reproduced"]]
    k, n = int(np.sum(flags)), len(flags)
    lo, hi = wilson(k, n)
    share = k / n
    o119 = dict(outcome="119: share of works whose trend our analogue reproduces within ±0.002 a year (TOST)",
                est=share, ci90_lo=lo, ci90_hi=hi, k=k, n=n,
                works=";".join([c["work"] for c in comp] + ["Paitaridis & Tsoulfidis (2012): R 1964–2007"]),
                reproduced=";".join(str(f) for f in flags),
                rule="ниже 0,5 — выводы этапа 1 предварительные (правило задания)",
                stage1_preliminary=bool(share < 0.5))
    # bridges on the common windows (within 1948-2023)
    for c in comp[:3]:
        for r in bridge((max(c["y0"], 1948), min(c["y1"], 2023))):
            br.append(dict(work=c["work"], **r))
    for r in bridge((1964, 2007)):
        br.append(dict(work="Paitaridis & Tsoulfidis (2012)", **r))
    # Mohun: text values
    Mo = mohun_ours()
    mrows = [dict(variable=v, year=y, mohun_text=x, ours=float(Mo.loc[y, v]) if y in Mo.index else np.nan)
             for v, y, x in MOHUN_TEXT]
    pd.DataFrame(comp).to_csv(OUT / "s2_compare.csv", index=False)
    pd.DataFrame(nums).to_csv(OUT / "s2_compare_pt.csv", index=False)
    pd.DataFrame(br).to_csv(OUT / "s2_bridge.csv", index=False)
    pd.DataFrame(mrows).to_csv(OUT / "s2_mohun_text.csv", index=False)
    pd.DataFrame(desc).to_csv(OUT / "s2_descriptive.csv", index=False)
    pd.DataFrame([o119]).to_csv(OUT / "s2_outcomes.csv", index=False)
    pd.DataFrame({"kliman_ours": ko, "shaikh_ours": so, "dl_ours": do}).join(P, how="outer").to_csv(OUT / "s2_series.csv")
    pd.set_option("display.width", 250)
    print(pd.DataFrame(comp)[["work", "y0", "y1", "trend_ours", "trend_author", "diff", "diff_lo", "diff_hi",
                              "reproduced"]].round(4).to_string())
    print(pd.DataFrame([o119]).T.to_string())


if __name__ == "__main__":
    run()
