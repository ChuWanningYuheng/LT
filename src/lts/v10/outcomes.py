"""Iteration 10: summary table of primary outcomes of iterations 1-10 (report/outcomes_table.md).

Rows 1-46 (with 22b, 25b, 26b) from iteration 9b (lts.v9b.outcomes.build), with iteration-10 notes on outcomes 1, 3,
15, 28, 44 (stage 4 data checks, stage 3.1 placebo); new rows 47-63 from results/v10 (outcome 52 was not assigned).
"""
from __future__ import annotations

import pandas as pd

from ..v6.series import ROOT
from ..v8.stage1 import fmt
from ..v9 import outcomes as v9o
from ..v9b import outcomes as v9bo

OUT = ROOT / "results" / "v10"


def r(p):
    return pd.read_csv(OUT / p)


def build():
    D = v9bo.build()

    def add_note(n, txt):
        i = D.index[D.n == n][0]
        cur = D.loc[i, "note"]
        D.loc[i, "note"] = (str(cur) + " " if isinstance(cur, str) and cur else "") + txt

    L = r("s4_outcomes_recomputed.csv")
    L["n_outcome"] = L.n_outcome.astype(str)
    for n in ("1", "3", "15", "28", "44"):
        a = L[(L.n_outcome == n) & (L.variant == "clean")].iloc[0]
        c = L[(L.n_outcome == n) & (L.variant == "corr")].iloc[0]
        add_note(n, f"10 (этап 4, проверка данных R1–R4): без помеченных отраслей-лет — {fmt(a.est, 3)} «{a.label}»; "
                    f"с исправлениями — {fmt(c.est, 3)} «{c.label}»; метка не изменилась.")
    p = r("s3_placebo44.csv").set_index("item")
    o54 = r("s3_outcomes.csv").iloc[0]
    add_note("44", f"10 (этап 3.1): случайный вектор с межстрановым разбросом часов даёт долю "
                   f"{fmt(p.loc['random vector, hours dispersion (b)', 'share'], 2)} (часы — "
                   f"{fmt(p.loc['hours (actual)', 'share'], 2)}), с разбросом фонда оплаты — "
                   f"{fmt(p.loc['random vector, LAB dispersion (b)', 'share'], 2)}; исход 54 «{o54.label}». "
                   f"Описательно: разность плацебо с разбросом часов и фонда оплаты (0,16) почти равна исходу 44 — перевес "
                   f"часов может объясняться их разбросом; по правилу не установлено (рецензия итерации 10, Р3).")
    rows = []

    def row(n, name, est, lo, hi, label, src, theta0, delta, direction, note=""):
        rows.append(dict(n=n, outcome=name, it="10", theta0=theta0, delta=delta, direction=direction, est=est, lo=lo,
                         hi=hi, old="— (новый исход)", label=label, src=src, note=note))
    o1 = r("s1_outcomes.csv").set_index(r("s1_outcomes.csv").outcome.str[:2])
    q = r("s1_us_q.csv")
    lp = r("s1_us_lp.csv")
    b = lp[(lp.variant == "main (BH)") & (lp.h == 2)].set_index(["group", "shock"])
    x = o1.loc["47"]
    row("47", "Q_нефть: отклик доли прибыли всех частных отраслей / потребителей нефти, нормированный на долю затрат "
        "(шок BH, h = 2)", x.est, x.ci90_lo, x.ci90_hi, x.label, "`results/v10/s1_outcomes.csv`", 1, 0.25, "<",
        note=f"знаменатель около нуля: отклик потребителей нефти {fmt(b.loc[('oil_con', 'bh'), 'beta'], 3)} "
             f"(SE {fmt(b.loc[('oil_con', 'bh'), 'se_hac'], 3)}); точка вне бутстреп-ДИ — отношение не определено.")
    x = o1.loc["48"]
    row("48", "Q_труд (−Δ безработицы, h = 2), эквивалентность ≈ 1", x.est, x.ci90_lo, x.ci90_hi, x.label,
        "`results/v10/s1_outcomes.csv`", 1, 0.25, "≈")
    x = o1.loc["49"]
    vk = q[(q.h == 2)].set_index("variant")
    row("49", "Q_нефть − Q_труд (главный исход этапа 1)", x.est, x.ci90_lo, x.ci90_hi, x.label,
        "`results/v10/s1_outcomes.csv`", 0, 0.25, "<",
        note=f"плацебо-перестановки: p (одност.) = {fmt(x.p_placebo_one, 3)}; Känzig {fmt(vk.loc['Kaenzig', 'D'], 2)}, "
             f"Kilian {fmt(vk.loc['Kilian', 'D'], 2)} — тоже неинформативно.")
    x = o1.loc["50"]
    t = r("s1_rtw_trend_adjusted.csv")
    tm = t[(t.variant == "first full year (main)") & (t.outcome == "gos_share")].iloc[0]
    row("50", "Законы «права на работу»: эффект на долю ВОИ частных отраслей штата, годы 0…+5 (п. п.)", x.est, x.ci90_lo,
        x.ci90_hi, x.label, "`results/v10/s1_outcomes.csv`", 0, 0.5, ">",
        note=f"**оговорка (журнал 4, после результата):** доля ВОИ росла у штатов лечения до закона; с линейным "
             f"предтрендом эффект {fmt(tm.est, 2)} ({fmt(tm.ci90_lo, 2)}; {fmt(tm.ci90_hi, 2)}), «{tm.label}».")
    sc = r("s1_sc.csv").set_index("case")
    x = o1.loc["51"]
    pg = r("rv_sc51_summary.csv").set_index("item").value
    row("51", "Хартц IV: разрыв доли прибыли Германии с синтетическим контролем, 2005–2012 (п. п.)", x.est, float("nan"),
        float("nan"), x.label, "`results/v10/s1_sc.csv`", 0, 0.5, ">",
        note=f"особое правило (|разрыв| ≤ 0,5) точности не требовало: SD разрывов доноров-плацебо "
             f"{fmt(pg['donor post-gap SD (all donors)'], 1)} п. п. ({fmt(pg['donor post-gap SD (pre-RMSPE <= 2x Germany)'], 1)} "
             f"при хорошей подгонке); в 2005–2007 гг. разрыв +0,7…+1,4, после кризиса отрицателен (рецензия Р4). "
             f"Ранговый p = {fmt(x.p_rank, 2)}; минимальная зарплата Германии 2015: {fmt(sc.loc['Germany minimum wage 2015', 'post_gap_pp'], 2)} "
             f"п. п. (p = {fmt(sc.loc['Germany minimum wage 2015', 'p_rank'], 2)}); NMW Великобритании 1999: "
             f"{fmt(sc.loc['UK NMW 1999', 'post_gap_pp'], 2)} (p = {fmt(sc.loc['UK NMW 1999', 'p_rank'], 2)}) — описательно.")
    s2 = r("s2_outcomes.csv").iloc[0]
    rb = r("s2_rd_robust.csv")
    rd = r("s2_rd.csv")
    rr = rb[rb.outcome == "d_roa_2"].set_index("sample")
    d3 = rd[(rd.outcome == "d_roa_3") & (rd.bw == 0.15)].iloc[0]
    row("53", "Выборы профсоюзов (NLRB × SEC): скачок Δ ROA (t − 1 → t + 2) на пороге 50%", s2.est, s2.ci90_lo,
        s2.ci90_hi, s2.label, "`results/v10/s2_outcomes.csv`", 0, 0.005, "<",
        note=f"{int(s2.in_window_with_droa2)} выборов в окне, из них 122 — Starbucks (43%); **не устойчиво** (журналы 13, 16б): "
             f"t + 3 — {fmt(d3.est, 3)} "
             f"({fmt(d3.ci90_lo, 3)}; {fmt(d3.ci90_hi, 3)}); один выбор на фирма-год — "
             f"{fmt(rr.loc['one election per firm-year', 'est'], 3)} ({fmt(rr.loc['one election per firm-year', 'ci90_lo'], 3)}; "
             f"{fmt(rr.loc['one election per firm-year', 'ci90_hi'], 3)}); без Starbucks — "
             f"{fmt(rr.loc['without Starbucks (CIK 829224)', 'est'], 3)} ({fmt(rr.loc['without Starbucks (CIK 829224)', 'ci90_lo'], 3)}; "
             f"{fmt(rr.loc['without Starbucks (CIK 829224)', 'ci90_hi'], 3)}); окна 0,10 и 0,25 — ДИ включает 0.")
    row("54", "Плацебо исхода 44: доля(часы) − доля(случайный вектор с разбросом часов)", o54.est, o54.ci90_lo, o54.ci90_hi,
        o54.label, "`results/v10/s3_outcomes.csv`", 0, 0.1, ">")
    o55 = r("s4_outcomes.csv").iloc[0]
    fb = r("s4_flags_by_country.csv").groupby("source")[["cells", "flagged", "corrected"]].sum()
    row("55", "Число меток исходов 1, 3, 15, 28, 44, изменившихся при проверке данных (R1–R4)", o55.est, float("nan"),
        float("nan"), "описательно", "`results/v10/s4_outcomes.csv`", "", "", "",
        note=f"0 из 5; помечено {int(fb.loc['FIGARO', 'flagged'])} из {int(fb.loc['FIGARO', 'cells'])} ячеек FIGARO и "
             f"{int(fb.loc['WIOD', 'flagged'])} из {int(fb.loc['WIOD', 'cells'])} WIOD (в основном R2 — малые отрасли); "
             f"исправлено {int(fb.loc['FIGARO', 'corrected'])} и {int(fb.loc['WIOD', 'corrected'])}.")
    s5 = r("s5_outcomes.csv")
    x = s5.iloc[0]
    row("56", "Доля лет, где Лаплас/Субботин лучше гаммы со сдвигом по BIC (норма прибыли фирм на активы, SEC)", x.est,
        x.ci90_lo, x.ci90_hi, x.label, "`results/v10/s5_outcomes.csv`", 0.5, 0.15, ">",
        note="15 из 15 лет; лучшая форма — асимметричный Субботин; на исправленной выписке SEC (журнал 7).")
    x = s5.iloc[1]
    row("57", "QCD(операционная прибыль / трудовые затраты) − QCD(операционная прибыль / активы), медиана по годам",
        x.est, x.ci90_lo, x.ci90_hi, x.label, "`results/v10/s5_outcomes.csv`", 0, 0.05, "<",
        note=f"{int(x.n_firm_years)} фирма-лет с тегом LaborAndRelatedExpense (выборка смещена); прибыль на трудовые затраты "
             f"разбросана **сильнее**, чем на активы.")
    s9 = r("s9_outcomes.csv")
    x = s9.iloc[0]
    v = s9.iloc[1]
    row("58", "Доля стран с «подтверждённым» падением нормы прибыли бизнес-сектора (BSDB → KLEMS 2009 → 2025, ≥ 30 лет)",
        x.est, x.ci90_lo, x.ci90_hi, x.label, "`results/v10/s9_outcomes.csv`", 0.5, 0.15, ">",
        note=f"{int(x.k)} из {int(x.n)}; устойчивы к концам {int(x.robust_ends)}; стык по среднему 1995–2000: "
             f"{int(v.k)} из {int(v.n)} «{v.label}». С поправкой на смешанный доход самозанятых (описательно, рецензия Р8) — "
             f"{int(r('rv_s9_selfemp_outcome.csv').iloc[0].k)} из 17 (CHE, ITA). Ряды CHE и CAN кончаются в 1993 и 1996 гг.")
    x = s9.iloc[3]
    row("59", "Доля стран с «подтверждённым» падением марксовой нормы r_m (KLEMS 2009 + 2025, ≥ 30 лет)", x.est, x.ci90_lo,
        x.ci90_hi, x.label, "`results/v10/s9_outcomes.csv`", 0.5, 0.15, ">",
        note=f"{int(x.k)} из {int(x.n)}; r_m растёт в 9 странах из 10. Обобщение закона тенденции на длинный период — "
             f"«{s9.iloc[-1].label}».")
    s10 = r("s10_outcomes.csv")
    x, xv = s10.iloc[0], s10.iloc[1]
    row("60", "Среднее отклонение цены от цены производства рентных отраслей (A01, B, L), FIGARO", x.est, x.ci90_lo,
        x.ci90_hi, x.label, "`results/v10/s10_outcomes.csv`", 0, 0.05, ">",
        note=f"**ренту не измеряет** (рецензия Р1): цены производства в коде — наценка на поток затрат, а не на запас; "
             f"держится на L: A01 {fmt(x.d_A01, 3)}, B {fmt(x.d_B, 3)}, L {fmt(x.d_L, 3)}; превышение того же порядка у финансов "
             f"(K64 +0,25) и опта (G46 +0,13); перцентиль плацебо {fmt(x.placebo_percentile, 3)}; при единой зарплате — "
             f"{fmt(xv.est, 3)} «{xv.label}».")
    x = s10.iloc[2]
    dd = r("s10_descriptive.csv").set_index("item")
    bv = dd.loc["61 (BEA variant): mean d(211, 212) on ln real energy+metals index, HAC"]
    row("61", "Эластичность отклонения добычи (B) по реальному индексу цен энергии и металлов, FIGARO", x.est, x.ci90_lo,
        x.ci90_hi, x.label, "`results/v10/s10_outcomes.csv`", 0, 0.05, ">",
        note=f"BEA (211, 212): {fmt(bv.est, 3)} ({fmt(bv.ci90_lo, 3)}; {fmt(bv.ci90_hi, 3)}) «{bv.label}».")
    s11 = r("s11_outcomes.csv")
    x = s11.iloc[0]
    row("62", "Доля окон (страна × 5 лет, PWT, ОЭСР) с марксовым техническим прогрессом", x.est, x.ci90_lo, x.ci90_hi,
        x.label, "`results/v10/s11_outcomes.csv`", 0.5, 0.1, ">",
        note=f"{int(x.k)} из {int(x.n)}; на пороге (θ₀ + Δ = 0,60); без окна 2015–2020 (COVID) — 0,583 «неинформативно»; "
             f"кластерный бутстреп по странам (0,552; 0,648) метку не меняет (рецензия Р6).")
    x = s11.iloc[1]
    row("63", "Окна с падающей r: доля, где Y/K даёт больше половины падения и ω не растёт", x.est, x.ci90_lo, x.ci90_hi,
        x.label, "`results/v10/s11_outcomes.csv`", 0.5, 0.15, ">",
        note=f"{int(x.k)} из {int(x.n)}; **не тест Окисио** (рецензия Р7, журнал 16д): теорема исходит из постоянной реальной "
             f"зарплаты, а при Δω ≤ 0 Y/K даёт всё падение по тождеству; реальная зарплата не росла лишь в 16 из 187 окон с "
             f"падающей r.")
    return pd.concat([D, pd.DataFrame(rows)], ignore_index=True)


def write_md(D):
    t = v9bo.write_md(D)
    t = t.replace("# Сводная таблица первичных исходов итераций 1–9б", "# Сводная таблица первичных исходов итераций 1–10")
    t = t.replace("исходы 39–46, 22б, 25б, 26б и пересмотры 9б — [`pre_registration_v9b.md`](../pre_registration_v9b.md).",
                  "исходы 39–46, 22б, 25б, 26б и пересмотры 9б — [`pre_registration_v9b.md`](../pre_registration_v9b.md); "
                  "исходы 47–63 (52 не присваивался) — [`pre_registration_v10.md`](../pre_registration_v10.md).")
    t = t.replace("`src/lts/v9/outcomes.py`, `src/lts/v9b/outcomes.py`; машиночитаемая версия — `results/v9b/outcomes_table.csv`.",
                  "`src/lts/v9/outcomes.py`, `src/lts/v9b/outcomes.py`, `src/lts/v10/outcomes.py`; машиночитаемая версия — "
                  "`results/v10/outcomes_table.csv`.")
    return t


if __name__ == "__main__":
    D = build()
    D.to_csv(OUT / "outcomes_table.csv", index=False)
    (ROOT / "report" / "outcomes_table.md").write_text(write_md(D))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 70)
    print(D[["n", "outcome", "est", "lo", "hi", "label"]].tail(20).round(4).to_string())
