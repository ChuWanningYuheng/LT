"""Consistency check for report/REPORT_FINAL.md and report/SUMMARY.md.

Every number quoted in the final report carries an id [N..]. This registry says, for each id, the value
as printed, the tolerance, the results file and how the value is obtained from it. Running the script
recomputes every value and writes report/FINAL_numbers_check.md (id, printed, recomputed, file, rule,
status). It also checks that every [N..] id used in the report exists in the registry.
Usage:  PYTHONPATH=src python -m lts.final.check_numbers
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]


@lru_cache(None)
def csv(path):
    return pd.read_csv(ROOT / path, low_memory=False)


@lru_cache(None)
def md_section(path, header):
    """Parse the first markdown table after a '## header' line into a DataFrame (strings)."""
    lines = (ROOT / path).read_text().splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith("## ") and header in l)
    rows = []
    for l in lines[i + 1:]:
        if l.startswith("## "):
            break
        if l.startswith("|") and not set(l.replace("|", "").strip()) <= set("-: "):
            rows.append([c.strip() for c in l.strip("|").split("|")])
    df = pd.DataFrame(rows[1:], columns=rows[0])
    return df


def num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return np.nan


RT = "results/report_tables.md"


def rt_levels(country, system, col):
    d = md_section(RT, "Уровни: MAWD, подвыборка `all`, капитал=True")
    r = d[(d["страна"] == country) & (d["система"] == system)]
    return num(r[col].iloc[0])


def rt_t1(country, subset, h, col, spec="open_cap", tag="main"):
    d = md_section(RT, f"Динамика ({tag}): T1")
    r = d[(d.country == country) & (d.spec == spec) & (d.subset == subset) & (d.h == str(h))]
    return num(r[col].iloc[0])


def rt_t4(country, test, col, tag="bea", subset="core"):
    d = md_section(RT, f"Вне выборки ({tag}): T4")
    r = d[(d.country == country) & (d.spec == "open_cap") & (d.subset == subset) & (d.test == test)]
    return num(r[col].iloc[0])


# ------------------------------------------------------------------ v2 helpers
def decomp_pairs():
    d = csv("results/v2/decomp_basis.csv")
    a = d.groupby(["country", "basis"])[["mawd_sym", "mawd_labour", "share_L_w", "mawd_O"]].mean()
    a["better"] = a.mawd_sym < a.mawd_labour
    return a


def mixture():
    return csv("results/v2/decomp_mixture.csv")


def v2_pp(country, basis, subset="all"):
    d = csv("results/v2/pp_levels.csv")
    return float(d[(d.country == country) & (d.capital) & (d.subset == subset) & (d.basis == basis)].mawd.iloc[0])


def v2_basket(country, variant, metric="mawd"):
    d = csv("results/v2/baskets_figaro.csv")
    g = d[(d.country == country) & (d.capital) & (d.subset == "all") & (d.variant == variant) & (d.metric == metric)]
    return float(g.share_better.mean())


def sstar(theta, country):
    t = theta[(theta.country == country) & (theta.metric == "mawd") & theta.feasible.astype(bool)]
    c = t.groupby("s").share_better.mean()
    return float(c[c > 0.2].index.min())


def long_run(country, basis, col):
    d = csv("results/v2/long_run.csv")
    return float(d[(d.country == country) & (d.basis == basis) & (d.subset == "core")][col].iloc[0])


# ------------------------------------------------------------------ v3 helpers
def stageP(file, kinds=("perm", "lnorm")):
    p = csv(file)
    p = p[p["mask"] == "all"]
    a = p.groupby(["country", "kind", "metric"]).share_better.mean().unstack(["kind", "metric"])
    cols = [(k, m) for k in kinds for m in ("mawd", "d")]
    return int((a[cols] <= 0.05).all(1).sum())


@lru_cache(None)
def gap():
    from ..v3.summarize import gap_table
    return gap_table()


def gap_median(variant, metric="mawd"):
    g = gap()
    return float(g[(g.variant == variant) & (g.metric == metric)].G.median())


# ------------------------------------------------------------------ v4 helpers
def a1(country, variant, kind, metric):
    d = csv("results/v4/a1_placebo.csv")
    return float(d[(d.country == country) & (d.variant == variant) & (d.kind == kind) & (d.metric == metric)].share_better.mean())


def a2_count(scale):
    d = csv("results/v4/a2_mrio_placebo.csv")
    a = d[d.rest_scale == scale].groupby(["country", "kind", "metric"]).share_better.mean().unstack(["kind", "metric"])
    return int((a <= 0.05).all(1).sum())


def bench(mask, metric):
    d = csv("results/final/benchmarks_flat_power.csv")
    a = d[d["mask"] == mask].groupby(["country", "vector"])[metric].mean().unstack()
    return int((a.flat < a.hours).sum())


def bench_G(vector, mask="all"):
    d = csv("results/final/benchmarks_flat_power.csv")
    a = d[d["mask"] == mask].groupby(["country", "vector"]).mawd.mean().unstack()
    gapw = a.hours - a.wagebill
    G = ((a.hours - a[vector]) / gapw)[gapw > 0.01 * a.hours]
    return float(G.median())


def b0_count(col, h):
    r = csv("results/v4/b0_summary.csv")
    return int(r[(r.subset == "core") & (r.h == h)][col].sum())


def b0b(cand, col, mask="all", method="a_quantile"):
    c = csv("results/v4/b0b_counts.csv")
    return float(c[(c.cand == cand) & (c["mask"] == mask) & (c.method == method)][col].iloc[0])


def b0b_perm(cand, metric="mawd", mask="all"):
    b = csv("results/v4/b0b_boot.csv")
    g = b[(b.method == "a_quantile") & (b["mask"] == mask) & (b.cand == cand) & (b.metric == metric)]
    return float(g.perm_better.median())


def b31(variant, x="kw", controls=False, iv="", rent=False, col="beta"):
    d = csv("results/v4/b31.csv").fillna({"iv": ""})
    r = d[(d.variant == variant) & (d.x == x) & (d.controls == controls) & (d.iv == iv) & (d.with_rent == rent)]
    return float(r[col].iloc[0])


def b32(source, col="lam", variant="primary", rent=False, window="2000-2021"):
    d = csv("results/v4/b32.csv")
    r = d[(d.source == source) & (d.variant == variant) & (d.with_rent == rent) & (d.window == window)]
    return float(r[col].iloc[0])


# ------------------------------------------------------------------ registry
# id: (printed value, tolerance, file, rule description, function)
R = {
    # ---- validation and levels (iteration 1)
    "N1": (0.066, 5e-4, RT, "Zachariah-like MAWD LTV, DEU", lambda: num(md_section(RT, "Валидация").query("country=='DEU'").mawd_ltv.iloc[0])),
    "N2": (0.109, 5e-4, RT, "Zachariah-like MAWD LTV, USA", lambda: num(md_section(RT, "Валидация").query("country=='USA'").mawd_ltv.iloc[0])),
    "N3": (0.158, 5e-4, RT, "MAWD hours DEU, open, all, capital", lambda: rt_levels("DEU", "открытая", "труд (часы)")),
    "N4": (0.315, 5e-4, RT, "MAWD hours USA (FIGARO)", lambda: rt_levels("USA", "открытая", "труд (часы)")),
    "N5": (0.002, 5e-4, RT, "share of commodity bases better than labour, DEU open", lambda: rt_levels("DEU", "открытая", "доля товаров лучше труда")),
    "N6": (0.18, 0.005, RT, "same, USA open", lambda: rt_levels("USA", "открытая", "доля товаров лучше труда")),
    "N7": (0.45, 0.005, RT, "share better, DEU closed uniform", lambda: rt_levels("DEU", "замкнутая (единая з/п)", "доля товаров лучше труда")),
    "N8": (0.65, 0.005, RT, "share better, USA closed uniform", lambda: rt_levels("USA", "замкнутая (единая з/п)", "доля товаров лучше труда")),
    "N9": (0.094, 5e-4, RT, "MAWD wage-weighted labour DEU", lambda: rt_levels("DEU", "открытая", "труд (з/п)")),
    "N10": (0.200, 5e-4, RT, "MAWD wage-weighted labour USA", lambda: rt_levels("USA", "открытая", "труд (з/п)")),
    "N11": (0.082, 5e-4, RT, "MAWD PP actual wage DEU", lambda: rt_levels("DEU", "открытая", "PP факт. з/п")),
    "N12": (13, 0, RT, "countries where PP actual < hours (open)", lambda: int(sum(num(r["PP факт. з/п"]) < num(r["труд (часы)"]) for _, r in md_section(RT, "Уровни: MAWD, подвыборка `all`, капитал=True").query("система=='открытая' and страна!='USA_BEA'").iterrows()))),
    "N13": (0.271, 5e-4, RT, "random placebo 5th percentile DEU", lambda: rt_levels("DEU", "открытая", "плацебо случ.: 5-й перц.")),
    # ---- dynamics (iteration 1)
    "N14": (0.34, 0.005, RT, "T1 labour DEU core h=1", lambda: rt_t1("DEU", "core", 1, "labour")),
    "N15": (0.54, 0.005, RT, "T1 labour DEU core h=5", lambda: rt_t1("DEU", "core", 5, "labour")),
    "N16": (0.24, 0.005, RT, "T1 labour USA core h=1", lambda: rt_t1("USA", "core", 1, "labour")),
    "N17": (0.73, 0.005, RT, "T1 labour USA_BEA core h=5", lambda: rt_t1("USA_BEA", "core", 5, "labour", tag="bea")),
    "N18": (0.066, 5e-4, RT, "T4 BEA 2009-2014 core labour RMSE", lambda: rt_t4("USA_BEA", "2009-2014", "labour")),
    "N19": (0.064, 5e-4, RT, "T4 BEA 2009-2014 naive", lambda: rt_t4("USA_BEA", "2009-2014", "rmse_naive_frozen_prices")),
    "N20": (0.048, 5e-4, RT, "T4 BEA 2009-2014 PP actual", lambda: rt_t4("USA_BEA", "2009-2014", "pp_actual_wage")),
    # ---- iteration 2
    "N21": (414, 0, "results/v2/decomp_basis.csv", "country x basis pairs with symmetric X better than labour (MAWD)", lambda: int(decomp_pairs().better.sum())),
    "N22": (796, 0, "results/v2/decomp_basis.csv", "country x basis pairs", lambda: len(decomp_pairs())),
    "N23": (0.61, 0.005, "results/v2/decomp_basis.csv", "median labour share L_X among bases better than labour", lambda: float(decomp_pairs().query("better").share_L_w.median())),
    "N24": (0, 0, "results/v2/decomp_mixture.csv", "asym pairs where labour+O_X significantly better than labour (BH q<0.05)", lambda: int(((mixture().setting == "asym") & (mixture().q_mix_vs_L < 0.05)).sum())),
    "N25": (0.04, 0.005, "results/v2/baskets_figaro.csv", "DEU necessities-only actual basket: share of bases better", lambda: v2_basket("DEU", "c_necess_actual")),
    "N26": (0.7, 1e-9, "results/v2/decomp_theta.csv", "s* DEU", lambda: sstar(csv("results/v2/decomp_theta.csv"), "DEU")),
    "N27": (0.1, 1e-9, "results/v2/decomp_theta.csv", "s* USA", lambda: sstar(csv("results/v2/decomp_theta.csv"), "USA")),
    "N28": (0.4, 1e-9, "results/v3/jpn_fixed_theta.csv", "s* JPN after data fix", lambda: sstar(csv("results/v3/jpn_fixed_theta.csv"), "JPN")),
    "N29": (0.126, 5e-4, "results/v2/pp_levels.csv", "MAWD PP(b) uniform wage DEU", lambda: v2_pp("DEU", "PPb")),
    "N30": (0.226, 5e-4, "results/v2/pp_levels.csv", "MAWD PP(b) USA", lambda: v2_pp("USA", "PPb")),
    "N31": (0.244, 5e-4, "results/v2/pp_levels.csv", "MAWD PP(b) USA BEA", lambda: v2_pp("USA_BEA", "PPb")),
    "N32": (5.1, 0.05, "results/v2/long_run.csv", "half-life labour DEU (model B, core)", lambda: long_run("DEU", "labour", "half_life_B")),
    "N33": (16, 0.6, "results/v2/long_run.csv", "half-life labour USA BEA", lambda: long_run("USA_BEA", "labour", "half_life_B")),
    # ---- iteration 3 (after JPN fix)
    "N34": (10, 0, "results/v3/placebo_disp.csv", "countries in top 5% vs (a) and (b), both metrics", lambda: stageP("results/v3/placebo_disp.csv")),
    "N35": (9, 0, "results/v3/placebo_disp_prefix_jpn.csv", "same before JPN fix", lambda: stageP("results/v3/placebo_disp_prefix_jpn.csv")),
    "N36": (8, 0, "results/v3/placebo_disp.csv", "countries in top 5% vs (c) mixtures", lambda: stageP("results/v3/placebo_disp.csv", ("mix",))),
    "N37": (0.04, 0.005, "results/v3/reduction_levels.csv", "median G 1.1a (MAWD)", lambda: gap_median("1.1a_edu_price")),
    "N38": (0.13, 0.005, "results/v3/reduction_levels.csv", "median G 1.1b", lambda: gap_median("1.1b_occ_price")),
    "N39": (0.01, 0.005, "results/v3/reduction_levels.csv", "median G 1.2a", lambda: gap_median("1.2a_jz_time")),
    "N40": (0.92, 0.005, "results/v3/reduction_levels.csv", "median G 1.5a frozen 2010 own wages", lambda: gap_median("1.5a_frozen2010")),
    "N41": (0.72, 0.005, "results/v3/reduction_levels.csv", "median G 1.5b foreign median wages", lambda: gap_median("1.5b_foreign_median")),
    "N42": (0.57, 0.005, "results/v3/section_bound.csv", "median G own wages averaged to sections", lambda: gap_median("диагн.: зарплата, средняя по секции")),
    "N43": (-0.279, 5e-4, "results/v3/premium_2a_pooled.csv", "beta premium on PCM", lambda: float(csv("results/v3/premium_2a_pooled.csv").query("x=='pcm'").beta.iloc[0])),
    "N44": (0.029, 5e-4, "results/v3/premium_2a_pooled.csv", "p premium on PCM", lambda: float(csv("results/v3/premium_2a_pooled.csv").query("x=='pcm'").p.iloc[0])),
    "N45": (0.336, 5e-4, "results/v3/cc_panel_tests.csv", "wild bootstrap p K1 coverage (DV mix)", lambda: float(csv("results/v3/cc_panel_tests.csv").query("candidate=='K1' and dv=='dv' and placebo=='mix'").p_wild.iloc[0])),
    "N46": (-0.60, 0.005, "results/v3/cc_cross_country.csv", "Spearman K1 cross-country (mix)", lambda: float(csv("results/v3/cc_cross_country.csv").query("candidate=='K1' and dv=='dv' and placebo=='mix'").rho.iloc[0])),
    "N47": (0.031, 5e-4, "results/v3/cc_cross_country.csv", "permutation p K1", lambda: float(csv("results/v3/cc_cross_country.csv").query("candidate=='K1' and dv=='dv' and placebo=='mix'").p_perm.iloc[0])),
    "N48": (1.0, 1e-9, "results/v3/placebo_disp.csv", "USA share of mixtures better than labour (MAWD)", lambda: float(csv("results/v3/placebo_disp.csv").query("country=='USA' and mask=='all' and kind=='mix' and metric=='mawd'").share_better.mean())),
    "N49": (0.204, 5e-4, "results/v3/reduction_levels.csv", "MAWD hours JPN after fix", lambda: float(csv("results/v3/reduction_levels.csv").query("country=='JPN' and variant=='hours' and kind=='err' and metric=='mawd'").value.mean())),
    # ---- iteration 4, part A
    "N50": (0.073, 5e-4, "results/v4/a1_placebo.csv", "JPN A1(a) share of permutations better (MAWD)", lambda: a1("JPN", "A1a_nosplit", "perm", "mawd")),
    "N51": (0.140, 5e-4, "results/v4/a1_placebo.csv", "JPN A1(a) permutations (d)", lambda: a1("JPN", "A1a_nosplit", "perm", "d")),
    "N52": (0.066, 5e-4, "results/v4/a1_placebo.csv", "KOR A1(b) permutations (d)", lambda: a1("KOR", "A1b_donor", "perm", "d")),
    "N53": (54, 0, "results/v4/a1_split_lists.csv", "KOR split industries from 2013", lambda: int(csv("results/v4/a1_split_lists.csv").query("country=='KOR' and year==2015").n_split.iloc[0])),
    "N54": (45, 0, "results/v4/a1_split_lists.csv", "JPN split industries", lambda: int(csv("results/v4/a1_split_lists.csv").query("country=='JPN' and year==2015").n_split.iloc[0])),
    "N55": (10, 0, "results/v4/a2_mrio_placebo.csv", "MRIO: countries top 5% vs (a),(b), rest of world x1", lambda: a2_count(1.0)),
    "N56": (9, 0, "results/v4/a2_mrio_placebo.csv", "same, x0.5", lambda: a2_count(0.5)),
    "N57": (9, 0, "results/v4/a2_mrio_placebo.csv", "same, x2", lambda: a2_count(2.0)),
    "N58": (12, 0, "results/final/benchmarks_flat_power.csv", "countries where flat vector beats hours (MAWD, all industries)", lambda: bench("all", "mawd")),
    "N59": (10, 0, "results/final/benchmarks_flat_power.csv", "same, d", lambda: bench("all", "d")),
    "N60": (4, 0, "results/final/benchmarks_flat_power.csv", "flat beats hours without B,K,L,J61-63 (MAWD)", lambda: bench("no_prereg", "mawd")),
    "N61": (0.91, 0.01, "results/final/benchmarks_flat_power.csv", "median G of hours^0.5 (MAWD)", lambda: bench_G("hours^0.5")),
    "N62": (0.60, 0.01, "results/final/benchmarks_flat_power.csv", "median G of hours^0.75 (MAWD)", lambda: bench_G("hours^0.75")),
    # ---- iteration 4, part B
    "N63": (8, 0, "results/v4/b0_summary.csv", "B0: data sets where money-flat beats labour, core h=1", lambda: 8 - b0_count("beats_flat_money", 1)),
    "N64": (6, 0, "results/v4/b0_summary.csv", "B0: labour beats physical flat and 95% of permutations, h=5", lambda: b0_count("informative_vs_phys", 5)),
    "N65": (2, 0, "results/v4/b0_summary.csv", "same, h=1", lambda: b0_count("informative_vs_phys", 1)),
    "N66": (9, 0, "results/v4/b0b_counts.csv", "B0b: non-labour costs not worse than hours, (a) all", lambda: b0b("nonlabour", "not_worse")),
    "N67": (10, 0, "results/v4/b0b_counts.csv", "B0b: hours point-better than non-labour on both metrics, (a) all", lambda: b0b("nonlabour", "hours_point_better_both")),
    "N68": (6, 0, "results/v4/b0b_counts.csv", "B0b: hours significantly better than CFC, (a) all", lambda: b0b("cfc", "hours_sig_better_both")),
    "N69": (11, 0, "results/v4/b0b_counts.csv", "same, without rent industries", lambda: b0b("cfc", "hours_sig_better_both", mask="no_rent")),
    "N70": (7, 0, "results/v4/b0b_counts.csv", "B0b: hours significantly better than capital (of 10)", lambda: b0b("capital", "hours_sig_better_both")),
    "N71": (4, 0, "results/v4/b0b_counts.csv", "B0b: wage bill significantly better than hours, (a)", lambda: b0b("wagebill", "cand_sig_better_any")),
    "N72": (8, 0, "results/v4/b0b_counts.csv", "same, (c) common dispersion", lambda: b0b("wagebill", "cand_sig_better_any", method="c_common")),
    "N73": (0.77, 0.01, "results/v4/b0b_boot.csv", "median share of own permutations better than capital (MAWD)", lambda: b0b_perm("capital")),
    "N74": (0.94, 0.01, "results/v4/b0b_boot.csv", "same, d", lambda: b0b_perm("capital", "d")),
    "N75": (0.003, 5e-4, "results/v4/b0b_boot.csv", "same for hours, MAWD", lambda: b0b_perm("hours")),
    "N76": (-0.028, 5e-4, "results/v4/b31.csv", "B3.1 beta ln(K/W), primary", lambda: b31("primary")),
    "N77": (0.043, 5e-4, "results/v4/b31.csv", "Holm p, primary", lambda: b31("primary", col="p_holm")),
    "N78": (-0.003, 5e-4, "results/v4/b31.csv", "beta with controls", lambda: b31("primary", controls=True)),
    "N79": (0.79, 0.005, "results/v4/b31.csv", "p with controls", lambda: b31("primary", controls=True, col="p")),
    "N80": (-0.111, 5e-4, "results/v4/b31.csv", "beta without mixed-income correction", lambda: b31("no_mi")),
    "N81": (0.23, 0.005, "results/v4/b31.csv", "p, IV median of other countries", lambda: b31("primary", iv="other_countries", col="p")),
    "N82": (0.044, 5e-4, "results/v4/b31_no_pcm_exploratory.csv", "exploratory beta, controls without PCM", lambda: float(csv("results/v4/b31_no_pcm_exploratory.csv").query("variant=='primary' and x=='kw'").beta.iloc[0])),
    "N83": (-1.15, 0.005, "results/v4/b32.csv", "B3.2 lambda labour cost", lambda: b32("labour_W")),
    "N84": (-3.93, 0.005, "results/v4/b32.csv", "lower bootstrap bound", lambda: b32("labour_W", "ci_lo")),
    "N85": (1.38, 0.005, "results/v4/b32.csv", "upper bootstrap bound", lambda: b32("labour_W", "ci_hi")),
    "N86": (0.25, 0.005, "results/v4/b32.csv", "median R2 labour", lambda: b32("labour_W", "r2")),
    "N87": (0.45, 0.005, "results/v4/b32.csv", "median R2 gross output", lambda: b32("GO", "r2")),
    "N88": (0.01, 0.005, "results/v4/b32.csv", "lambda gross output", lambda: b32("GO")),
    "N89": (0.0, 1e-9, "results/v4/b32.csv", "share of perm placebos with lambda <= labour", lambda: b32("placebo a_perm", "share_lam_below_labour")),
    "N90": (0.074, 5e-4, "results/v4/b33.csv", "B3.3 beta h=1", lambda: float(csv("results/v4/b33.csv").query("h==1").beta.iloc[0])),
    "N91": (12595, 0, "results/v4/b32.csv", "observations B3.2 primary", lambda: int(csv("results/v4/b32.csv").query("variant=='primary' and not with_rent").n_obs.iloc[0])),
    "N92": (28, 0, "results/v4/b_panel.csv.gz", "countries in the KLEMS panel", lambda: int(csv("results/v4/b_panel.csv.gz").geo.nunique())),
    "N93": (-0.33, 0.005, "results/v4/b31.csv", "beta with rent industries (x100 check: -0.033)", lambda: 10 * b31("primary", rent=True)),
    "N94": (0.61, 0.005, "results/v4/b32.csv", "R2 labour with rent industries", lambda: b32("labour_W", "r2", rent=True)),
    "N95": (0.67, 0.005, "results/v4/b32.csv", "R2 GO with rent industries", lambda: b32("GO", "r2", rent=True)),
}

def a1_count(variant):
    """Countries in top 5% vs (a),(b) on both metrics when JPN, KOR use the A1 variant (others: v3,
    identical by construction).  Countries where the variant is undefined are dropped."""
    p = csv("results/v3/placebo_disp.csv")
    p = p[(p["mask"] == "all") & p.kind.isin(["perm", "lnorm"])]
    base = p.groupby(["country", "kind", "metric"]).share_better.mean().unstack(["kind", "metric"])
    top = (base <= 0.05).all(1)
    a = csv("results/v4/a1_placebo.csv")
    for c in ("JPN", "KOR"):
        g = a[(a.country == c) & (a.variant == variant) & a.kind.isin(["perm", "lnorm"])]
        if g.empty:
            top = top.drop(c)
            continue
        top[c] = bool((g.groupby(["kind", "metric"]).share_better.mean() <= 0.05).all())
    return int(top.sum()), len(top)


def v2_gap_max(subset, metric):
    g = csv("results/v2/reduction_gap.csv")
    g = g[g.capital & (g.variant.eq("edu_years") | g.variant.str.startswith("ck_")) & (g.country != "MEX")]
    return float(g[(g.subset == subset) & (g.metric == metric)].G.max())


R2 = {
    "N96": (0.007, 5e-4, RT, "share better, MEX open", lambda: rt_levels("MEX", "открытая", "доля товаров лучше труда")),
    "N97": (0.26, 0.005, RT, "share better, USA BEA open", lambda: rt_levels("USA_BEA", "открытая", "доля товаров лучше труда")),
    "N98": (0.45, 0.005, RT, "share better, MEX closed uniform", lambda: rt_levels("MEX", "замкнутая (единая з/п)", "доля товаров лучше труда")),
    "N99": (0.68, 0.005, RT, "share better, USA BEA closed uniform", lambda: rt_levels("USA_BEA", "замкнутая (единая з/п)", "доля товаров лучше труда")),
    "N100": (0.127, 5e-4, "results/v2/pp_levels.csv", "MAWD PP(a) USA", lambda: v2_pp("USA", "PPa")),
    "N101": (0.157, 5e-4, "results/v2/pp_levels.csv", "MAWD PP(a) USA BEA", lambda: v2_pp("USA_BEA", "PPa")),
    "N102": (0.339, 5e-4, "results/v2/pp_levels.csv", "MAWD hours USA BEA", lambda: v2_pp("USA_BEA", "LV_hours")),
    "N103": (0.13, 0.005, "results/v2/reduction_gap.csv", "max G of education / Cockshott reductions (DEU, USA, BEA; core, MAWD)", lambda: v2_gap_max("core", "mawd")),
    "N104": (6, 0, "results/v4/b0b_counts.csv", "non-labour not worse than hours, (a) without rent", lambda: b0b("nonlabour", "not_worse", mask="no_rent")),
    "N105": (7, 0, "results/final/benchmarks_flat_power.csv", "flat beats hours without rent industries (d)", lambda: bench("no_prereg", "d")),
    "N106": (0.51, 0.005, "results/v4/b32.csv", "median R2 value added (reference)", lambda: b32("VA (reference)", "r2")),
    "N107": (8, 0, "results/v4/a1_placebo.csv + results/v3/placebo_disp.csv", "A1(a): countries top 5% (of defined)", lambda: a1_count("A1a_nosplit")[0]),
    "N108": (12, 0, "results/v4/a1_placebo.csv + results/v3/placebo_disp.csv", "A1(a): defined countries", lambda: a1_count("A1a_nosplit")[1]),
    "N109": (9, 0, "results/v4/a1_placebo.csv + results/v3/placebo_disp.csv", "A1(b): countries top 5%", lambda: a1_count("A1b_donor")[0]),
    "N110": (0.75, 0.005, "results/v2/decomp_mixture.csv", "asym pairs where labour dominates (X adds <5%, labour adds >20%)", lambda: float(((mixture().setting == "asym") & (mixture().red_L_by_X < 0.05) & (mixture().red_X_by_L > 0.2)).sum() / (mixture().setting == "asym").sum())),
    "N112": (0.02, 0.005, "results/v3/reduction_levels.csv", "median G 1.3a Hilferding", lambda: gap_median("1.3a_hilf")),
    "N111": (0.0, 1e-9, "results/v4/a1_placebo.csv", "JPN A1(b) share of permutations better (MAWD)", lambda: a1("JPN", "A1b_donor", "perm", "mawd")),
}
R.update(R2)


def main():
    rows, bad = [], 0
    for k, (val, tol, f, rule, fn) in R.items():
        try:
            got = fn()
            ok = abs(got - val) <= tol + 1e-12
        except Exception as ex:                      # noqa: BLE001
            got, ok = f"ERROR {ex!r}", False
        bad += not ok
        rows.append(f"| {k} | {val} | {got if isinstance(got, str) else round(got, 4)} | `{f}` | {rule} | {'OK' if ok else 'РАСХОЖДЕНИЕ'} |")
    used = set()
    for rep in ("report/REPORT_FINAL.md", "report/SUMMARY.md"):
        p = ROOT / rep
        if p.exists():
            used |= set(re.findall(r"\[(N\d+)\]", p.read_text()))
    missing = sorted(used - set(R), key=lambda s: int(s[1:]))
    unused = sorted(set(R) - used, key=lambda s: int(s[1:]))
    out = ["# Сверка чисел итогового отчёта", "",
           "Генерируется `PYTHONPATH=src python -m lts.final.check_numbers`. Для каждого идентификатора [N…] из REPORT_FINAL.md и SUMMARY.md: значение в тексте, пересчитанное значение, файл и правило.", "",
           f"Итого: {len(R)} чисел, расхождений: {bad}. Идентификаторы в тексте без записи в реестре: {missing or 'нет'}. Записи реестра, не использованные в тексте: {unused or 'нет'}.", "",
           "| id | в тексте | пересчёт | файл | правило | статус |", "|---|---|---|---|---|---|"] + rows
    (ROOT / "report" / "FINAL_numbers_check.md").write_text("\n".join(out) + "\n")
    print(f"{len(R)} numbers, {bad} mismatches, missing ids {missing}, unused {len(unused)}")
    for r in rows:
        if "РАСХОЖДЕНИЕ" in r:
            print(r)


if __name__ == "__main__":
    main()
