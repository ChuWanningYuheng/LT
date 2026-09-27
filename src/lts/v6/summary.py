"""Iteration 6: tables for REPORT_v6.md (results/v6/tables_v6.md)."""
from __future__ import annotations

import pandas as pd

from ..v3.summarize import md
from .series import MAIN8, OUT

ORDER = MAIN8 + ["USA_NFC", "JPN", "CAN"]


def section(L, title, df, fmt="{:.3f}"):
    L.append(f"\n## {title}\n")
    L.append(md(df, fmt))


def main():
    L = ["# Итерация 6: таблицы (генерируются `python -m lts.v6.summary`)\n"]
    s = pd.read_csv(OUT / "series.csv")
    cov = s.groupby(["geo", "variant"]).year.agg(["min", "max", "size"]).reset_index()
    cov = cov[cov.variant.isin(["main", "no_mi", "hc", "kdrift", "bea_K", "no_ip"])]
    cov.index = cov.geo + " " + cov.variant
    section(L, "Покрытие рядов", cov[["min", "max", "size"]], "{:.0f}")
    ki = pd.read_csv(OUT / "kdrift_info.csv").set_index("geo")
    section(L, "Дрейф капитала AMECO относительно официальных запасов (журнал п. 4)", ki.drop(columns="note"), "{:.4f}")
    full = pd.read_csv(OUT / "decomp_full.csv")
    f = full[full.variant.isin(["main", "no_mi", "kdrift", "bea_K", "hc"])].copy()
    f.index = f.geo + " " + f.variant
    section(L, "Разложения за весь период (Маркс, Вайскопф, Басу; ковариационные доли)",
            f[["y0", "y1", "r0", "r1", "dlnrM", "c_e", "c_1k", "e0", "e1", "k0", "k1", "dlnr", "c_share", "c_qyk", "c_pyk",
               "v_share", "v_qyk", "v_pyk", "v_e", "v_1k"]])
    dd = pd.read_csv(OUT / "decomp_decades.csv")
    d = dd[dd.variant.isin(["main"])].copy()
    d.index = d.geo + " " + d.decade
    section(L, "Разложения по десятилетиям, основной вариант",
            d[["rM0", "rM1", "dlnrM", "c_e", "c_1k", "dlnk", "c_tech", "c_pkw", "dlnr", "c_share", "c_qyk", "c_pyk"]])
    tr = pd.read_csv(OUT / "trends.csv")
    b = tr[tr.base & tr.variant.isin(["main", "no_mi", "hc", "kdrift", "bea_K", "no_ip", "hc_no_ip", "gni", "hc_hcdep"])]
    b = b.pivot_table(index=["geo", "variant", "series"], columns="lag", values=["slope", "p"]).reset_index()
    b.columns = [f"{a}_{c}" if c != "" else a for a, c in b.columns]
    b.index = b.geo + " " + b.variant + " " + b.series
    section(L, "T1: тренды (полный период; наклон в год; p при лагах NW 2/4/8)",
            b[["slope_4", "p_2", "p_4", "p_8"]], "{:.4f}")
    rob = pd.read_csv(OUT / "trends_robust.csv")
    r = rob[rob.variant.isin(["main", "no_mi", "hc", "kdrift", "bea_K", "no_ip", "hc_no_ip", "hc_hcdep", "gni"])].copy()
    r.index = r.geo + " " + r.variant + " " + r.series
    section(L, "T1: устойчивость (все лаги и концы)", r[["robust_neg", "robust_pos", "share_neg_sig", "n_specs"]])
    br = pd.read_csv(OUT / "breaks.csv")
    br.index = br.geo + " " + br.variant + " " + br.series
    section(L, "T2: разрывы (Bai–Perron по LWZ; окно 1970–1982)",
            br[["n", "bp_m", "bp_dates", "bp3_dates", "win_F", "win_year", "win_p"]])
    p = pd.read_csv(OUT / "panel.csv")
    p.index = p["sample"] + " " + p.periods + " " + p.fe + " " + p.cycle
    section(L, "T3: панель Δ ln r_M на Δ ln(1+k)", p[["beta", "t_cr1", "p_wild", "n", "clusters"]])
    cv = pd.read_csv(OUT / "compare_v5.csv").set_index("geo")
    section(L, "T4: сравнение с итерацией 5 (1995–2021)", cv, "{:.4f}")
    g = pd.read_csv(OUT / "s4_gni_trends.csv")
    g = g.pivot_table(index=["geo", "view"], columns="lag", values=["slope", "p"]).reset_index()
    g.columns = [f"{a}_{c}" if c != "" else a for a, c in g.columns]
    g.index = g.geo + " " + g.view
    section(L, "Этап 4.1: тренды r в ВВП- и ВНД-представлении", g[["slope_4", "p_2", "p_4", "p_8"]], "{:.4f}")
    om = pd.read_csv(OUT / "s4_omega.csv")
    o = om.pivot_table(index="geo", columns="source", values="omega", aggfunc=["first", "last"])
    o.columns = [f"{a}_{b}" for a, b in o.columns]
    section(L, "Этап 4.2: ω = иностранная оплата труда в промежуточном импорте / своя (первый и последний год)", o)
    kp = pd.read_csv(OUT / "s4_panel.csv")
    kp.index = kp.source + " " + kp.x
    section(L, "Этап 4.2: панель Δ ln r_M на Δ ln(1+k) и Δ ln(1+k*)", kp[["beta", "p_wild", "r2_within", "n", "clusters"]])
    for fn, title in (("s5_1.csv", "Этап 5.1: пропорциональность долей прибыли"),
                      ("s5_1_placebo.csv", "Этап 5.1: плацебо для K_adv (τ = 0,25)"),
                      ("s5_2_summary.csv", "Этап 5.2: средние и инкрементальные нормы"),
                      ("s5_2_persistence.csv", "Этап 5.2: устойчивость различий средних норм (1996–2005 против 2011–2021)")):
        f_ = OUT / fn
        if f_.exists():
            x = pd.read_csv(f_)
            x = x.set_index(x.columns[0])
            section(L, title, x)
    t53 = OUT / "s5_3_trends.csv"
    if t53.exists():
        t = pd.read_csv(t53)
        t = t[(t.y0 == t.groupby("spec").y0.transform("min")) & (t.y1 == 2024)]
        t = t.pivot_table(index=["spec", "series"], columns="lag", values=["slope", "p"]).reset_index()
        t.columns = [f"{a}_{c}" if c != "" else a for a, c in t.columns]
        t.index = t.spec + " " + t.series
        section(L, "Этап 5.3: США — тренды (начало ряда–2024)", t[["slope_4", "p_2", "p_4", "p_8"]], "{:.4f}")
        dc = pd.read_csv(OUT / "s5_3_decomp.csv")
        dc.index = dc.spec + " " + dc.y0.astype(str) + "–" + dc.y1.astype(str)
        section(L, "Этап 5.3: США — марксово разложение", dc[["r0", "r1", "rM0", "rM1", "dlnrM", "c_e", "c_1k"]])
    (OUT / "tables_v6.md").write_text("\n".join(L) + "\n")
    print("ok")


if __name__ == "__main__":
    main()
