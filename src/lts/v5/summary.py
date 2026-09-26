"""Iteration 5: tables for REPORT_v5.md (results/v5/tables_v5.md)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..v3.summarize import md
from .stage1 import OUT

BM = ["labour", "pp", "equal_margin", "energy", "flat"]


def s3_tables():
    d = pd.read_csv(OUT / "s3_abm.csv")
    ok = d[d.ok == True]  # noqa: E712
    rows = []
    for (ck, rule, s), g in d.groupby(["cK", "rule", "share"]):
        o = g[g.ok == True]  # noqa: E712
        r = dict(cK=ck, rule=rule, share=s, runs=len(g), alive=len(o), stationary=int(o.stationary.sum()),
                 r_med=o.r_real.median(), recap_med=o.recap_share.median())
        for b in BM:
            c = f"mawd_{b}"
            r[f"MAWD_{b}"] = o[c].median() if c in o and o[c].notna().any() else np.nan
        cols = [f"mawd_{b}" for b in BM if f"mawd_{b}" in o and o[f"mawd_{b}"].notna().any() and b != "equal_margin"]
        if len(o):
            best = o[cols].idxmin(axis=1).str.replace("mawd_", "")
            for b in ("labour", "pp", "energy", "flat"):
                r[f"best_{b}"] = int((best == b).sum())
            r["labour_beats_energy"] = int((o.mawd_labour < o.mawd_energy).sum())
            r["labour_beats_flat"] = int((o.mawd_labour < o.mawd_flat).sum())
            r["perm_better_labour_med"] = o.perm_better_labour.median()
            r["ratio_profit_SL_med"] = o.ratio_profit_SL.median() if "ratio_profit_SL" in o else np.nan
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "s3_cells.csv", index=False)
    st = ok[ok.stationary == True]  # noqa: E712
    t2 = st.groupby(["cK", "rule", "share"])[[f"mawd_{b}" for b in ("labour", "pp", "energy", "flat")]].median()
    t2["n"] = st.groupby(["cK", "rule", "share"]).size()
    t2.to_csv(OUT / "s3_cells_stationary.csv")
    return t, t2


def s1_s2_tables():
    L = []
    t2 = pd.read_csv(OUT / "s1_t2.csv")
    for test, title in (("T2", "T2: Δ ln r_M на Δ ln(1+k), пятилетки, без e"),
                        ("T2_levels", "T2 в уровнях: Δ r_M на Δ ln(1+k) (все страны, вкл. Π ≤ 0)"),
                        ("T2_mech", "T2-механическая (с Δ ln e): тождество"),
                        ("T2_annual", "T2 годовая: Δ ln r_M на Δ ln(1+k) + цикл, эффекты страны и года"),
                        ("T2_no_pcm_fin", "T2 с контролем только цикла и Δ ln(P_K/P_Y)"),
                        ("T3", "T3: Δ ln r_M на Δ ln(Q_K/H) при Δ ln e и цикле"),
                        ("T3_check", "T3-проверка: Δ ln(P_K/w) на Δ ln(Q_K/H)")):
        x = t2[t2.test == test].copy()
        x.index = x.variant + " " + x.spec
        L.append(f"\n## Этап 1. {title}\n")
        L.append(md(x[["beta", "t_cr1", "p_cr1", "p_wild", "n", "clusters"]], "{:.3f}"))
    dec = pd.read_csv(OUT / "s1_decomp.csv")
    rows = []
    for v, g in dec.groupby("variant"):
        w = g.dropna(subset=["v_share"])
        rows.append(dict(variant=v, countries=len(g), med_v_share=w.v_share.median(), med_v_qyk=w.v_qyk.median(),
                         med_v_pyk=w.v_pyk.median(), share_plus_price_gt50=int(((w.v_share + w.v_pyk) > 0.5).sum()),
                         qyk_gt50=int((w.v_qyk > 0.5).sum()), med_v_e=w.v_e.median(), med_v_1k=w.v_1k.median(),
                         k_rising=int((g.dk > 0).sum()), k_rising_rM_fell=int(((g.dk > 0) & (g.dlnrM < 0)).sum()),
                         r_fell=int((g.dlnr < 0).sum()),
                         cum_share_dominant=int((g[["c_share", "c_qyk", "c_pyk"]].abs().idxmax(axis=1) == "c_share").sum()),
                         cum_qyk_dominant=int((g[["c_share", "c_qyk", "c_pyk"]].abs().idxmax(axis=1) == "c_qyk").sum()),
                         cum_pyk_dominant=int((g[["c_share", "c_qyk", "c_pyk"]].abs().idxmax(axis=1) == "c_pyk").sum())))
    x = pd.DataFrame(rows).set_index("variant")
    L.append("\n## Этап 1. Разложения по странам (сводка)\n")
    L.append(md(x, "{:.3f}"))
    m = dec[dec.variant == "main"].set_index("geo")
    L.append("\n## Этап 1. Разложения, основной агрегат, по странам\n")
    L.append(md(m[["y0", "y1", "dlnr", "c_share", "c_qyk", "c_pyk", "dlnrM", "c_e", "c_1k", "dk", "v_share", "v_qyk",
                   "v_pyk"]], "{:.3f}"))
    tr = pd.read_csv(OUT / "s1_trend.csv")
    rows = []
    for (v, s_), g in tr.groupby(["variant", "series"]):
        rows.append(dict(variant=v, series=s_, n=len(g), neg_sig=int(((g.slope < 0) & (g.p_trend_holm < 0.05)).sum()),
                         pos_sig=int(((g.slope > 0) & (g.p_trend_holm < 0.05)).sum()),
                         break_raw=int((g.p_break < 0.05).sum()), break_holm=int((g.p_break_holm < 0.05).sum()),
                         break_2008_10=int(g.break_year.between(2008, 2010).sum())))
    x = pd.DataFrame(rows)
    x.index = x.variant + " " + x.series
    L.append("\n## Этап 1. T1: тренды и разрывы (число стран)\n")
    L.append(md(x.drop(columns=["variant", "series"]), "{:.0f}"))
    us = pd.read_csv(OUT / "s1_us_decomp.csv")
    us.index = us.y0.astype(str) + "–" + us.y1.astype(str)
    L.append("\n## Этап 1. США, нефинансовые корпорации: разложения по подпериодам\n")
    L.append(md(us.drop(columns=["y0", "y1"]), "{:.3f}"))
    ut = pd.read_csv(OUT / "s1_us_trend.csv").set_index("series")
    L.append("\n## Этап 1. США: тренд и разрыв\n")
    L.append(md(ut, "{:.4f}"))
    ep = pd.read_csv(OUT / "s1_epwt.csv")
    L.append(f"\n## Этап 1. EPWT 7.0 (справочно)\n\nСтран: {len(ep)}; медианная корреляция уровней "
             f"{ep.corr_level.median():.2f}, годовых изменений {ep.corr_diff.median():.2f}; стран с отрицательной "
             f"корреляцией уровней: {(ep.corr_level < 0).sum()}.\n")
    s2 = pd.read_csv(OUT / "s2_tests.csv")
    s2 = s2[s2.variant.isin(["main", "no_mi", "tangible", "intangibles", "all"])]
    s2.index = s2.test + " " + s2.variant + " " + s2.A.fillna("")
    L.append("\n## Этап 2. Тесты автоматизации (основные варианты ловушек)\n")
    L.append(md(s2[["beta", "t_cr1", "p_wild", "p_holm", "n", "clusters"]], "{:.3f}"))
    ll = pd.read_csv(OUT / "s2_lowlab.csv")
    w = ll.pivot(index="geo", columns="group", values=["r", "pcm", "intang", "e"])
    rows = {c: dict(bottom_decile_med=w[(c, "bottom_decile")].median(), rest_med=w[(c, "rest")].median(),
                    countries_bd_higher=int((w[(c, "bottom_decile")] > w[(c, "rest")]).sum()), countries=len(w))
            for c in ("r", "pcm", "intang", "e")}
    L.append("\n## Этап 2. Почти безлюдные отрасли (нижний дециль H/Q_VA, 2015–2019)\n")
    L.append(md(pd.DataFrame(rows).T, "{:.3f}"))
    ind = ll[ll.group == "bottom_decile"].inds.str.split(",").explode().value_counts().head(8)
    L.append("\nЧаще всего в дециле: " + ", ".join(f"{k} ({v})" for k, v in ind.items()) + ".\n")
    trr = pd.read_csv(OUT / "s2_transfer_reg.csv")
    trr.index = trr.x + " " + trr.spec
    L.append("\n## Этап 2. Карта перетоков: T_j/GO_j на характеристики (эффекты страна×год, SE по отраслям)\n")
    L.append(md(trr[["beta", "t_cr1", "p_wild", "n", "clusters"]], "{:.3f}"))
    return L


def main():
    L = ["# Итерация 5: таблицы (генерируются `python -m lts.v5.summary`)\n"]
    L += s1_s2_tables()
    t, t2 = s3_tables()
    for ck in (1.0, 0.8):
        x = t[t.cK == ck].set_index(["rule", "share"]).drop(columns="cK")
        x.index = [f"{a} {b}" for a, b in x.index]
        L.append(f"\n## Этап 3.2, c_K = {ck}: медианы по живым прогонам\n")
        L.append(md(x[["runs", "alive", "stationary", "r_med", "recap_med", "MAWD_labour", "MAWD_pp", "MAWD_equal_margin",
                       "MAWD_energy", "MAWD_flat"]], "{:.3f}"))
        L.append("\n")
        L.append(md(x[["best_labour", "best_pp", "best_energy", "best_flat", "labour_beats_energy", "labour_beats_flat",
                       "perm_better_labour_med", "ratio_profit_SL_med"]], "{:.3f}"))
    t2.index = [f"{a} {b} {c}" for a, b, c in t2.index]
    L.append("\n## Этап 3.2: только стационарные прогоны (относительные цены, отклонение < 2%)\n")
    L.append(md(t2, "{:.3f}"))
    w = pd.read_csv(OUT / "s3_wright.csv")
    wg = w.groupby("L").agg(runs=("rep", "size"), converged=("converged", "sum"),
                            first_no_trade=("first_window_no_trade", "median"),
                            windows_all_traded=("n_windows_all_traded", "median"),
                            last_corr=("last_all_traded_corr", "mean"))
    L.append("\n## Этап 3.1: воспроизведение Wright (2008), правила как в статье\n")
    L.append(md(wg, "{:.3f}"))
    dg = pd.read_csv(OUT / "s3_wright_diag.csv")
    dgg = dg.groupby(["variant", "L"]).agg(runs=("rep", "size"), converged=("converged", "sum"),
                                           corr=("corr", "mean"), last_corr=("last_all_traded_corr", "mean"))
    dgg.index = [f"{a} L={b}" for a, b in dgg.index]
    L.append("\n## Этап 3.1: диагностические варианты (не модель Райта)\n")
    L.append(md(dgg, "{:.3f}"))
    (OUT / "tables_v5.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
