"""Iteration 9: summary table of primary outcomes of iterations 1-9 (report/outcomes_table.md).

Rows 1-19 from iteration 8 (results/v8/s1_outcomes.csv); outcomes 6, 16 and 19 shown as one fact (review of
iteration 8, R6) with the recomputation on the B3 panel without overlaps; rows 20-38 from iteration 9 results.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..v6.series import ROOT
from ..v8.stage1 import fmt
from .stage1 import OUT


def r(path):
    return pd.read_csv(OUT / path)


def row(n, name, est, lo, hi, label, src, theta0="", delta="", direction="", it="9", old="— (новый исход)", note=""):
    return dict(n=n, outcome=name, it=it, theta0=theta0, delta=delta, direction=direction, est=est, lo=lo, hi=hi,
                old=old, label=label, src=src, note=note)


def build():
    v8 = pd.read_csv(ROOT / "results" / "v8" / "s1_outcomes.csv")
    rows = []
    no = r("s1_noovl.csv").set_index(["panel", "item"])
    for x in v8.itertuples():
        if x.n_outcome in (16, 19):
            continue
        if x.n_outcome == 6:
            b6 = no.loc[("no overlaps", "outcome 6: B3.1 beta")]
            b19 = no.loc[("no overlaps", "outcome 19: b KLEMS (log, PI > 0)")]
            w = v8[v8.n_outcome == 16].iloc[0]
            rows.append(row("6 = 16 = 19", "Межотраслевой наклон: β r на ln(K/W) (KLEMS, WIOD) и b < 1 (KLEMS) — один факт",
                            x.est, x.ci90_lo, x.ci90_hi, x.label, x.src, x.theta0, x.delta, x.direction, x.it, x.old,
                            note=(f"WIOD: {fmt(w.est, 4)} ({fmt(w.ci90_lo, 4)}; {fmt(w.ci90_hi, 4)}); "
                                  f"b KLEMS (лог, Π > 0): 0,54. Панель Б3 без перекрытий: β = {fmt(b6.est, 4)} "
                                  f"(SE {fmt(b6.se, 4)}), b = {fmt(b19.est, 3)}. Исход 19 не говорит «за» ни одну "
                                  "версию теории; его порог задан при известной оценке (рецензия итерации 8, Р6). "
                                  "Какое b правильное — исход 21")))
            continue
        rows.append(row(str(x.n_outcome), x.outcome, x.est, x.ci90_lo, x.ci90_hi, x.label, x.src, x.theta0, x.delta,
                        x.direction, x.it, x.old))
    # ---- iteration 9
    rv = r("s1_real_reval_summary.csv").iloc[0]
    rows.append(row("20", "Реальная переоценка капитала: кризис − прочие годы (США НФК; журнал 1)", rv.est, rv.ci90_lo,
                    rv.ci90_hi, rv.label, "`results/v9/s1_real_reval*.csv`", 0, 0.01, "<"))
    ch = r("s1_b_choice.csv").set_index("panel").loc["KLEMS"]
    be = r("s1_b_estimators.csv").set_index(["panel", "estimator"])
    e = be.loc[("KLEMS", "(e) grouped (K/W quintiles)")]
    c = be.loc[("KLEMS", "(c) NLS in levels")]
    rows.append(row("21", "Степень трансформации b: основная оценка (сгруппированная) и правило «интервал при расхождении > 0,15»",
                    e.b, e.ci90_lo, c.ci90_hi, "неинформативно",
                    "`results/v9/s1_b_estimators.csv`, `s1_b_choice.csv`", 1, 0.1, "<",
                    note=(f"(д) = {fmt(e.b, 2)} ({fmt(e.ci90_lo, 2)}; {fmt(e.ci90_hi, 2)}) — «подтверждено» отдельно; "
                          f"(в) НМНК = {fmt(c.b, 2)}; |в − д| = {fmt(ch.gap, 2)} > 0,15 → вывод — интервал "
                          f"{fmt(min(c.b, e.b), 2)}–{fmt(max(c.b, e.b), 2)}, он содержит 1. ДИ в строке — от нижней границы (д) "
                          "до верхней (в)")))
    pt = r("s2_pt_outcomes.csv")
    o22 = pt[pt["item"].str.startswith("outcome 22 (h=8)")].iloc[0]
    rows.append(row("22", "Перенос ИПЦ в почасовую оплату, h = 8 кварталов (средний по 33 странам)", o22.est, o22.ci90_lo,
                    o22.ci90_hi, o22.label, "`results/v9/s2_pt_outcomes.csv`", 1, 0.1, "[0,9; 1,1] = H_tech"))
    o23 = pt[pt["item"].str.startswith("outcome 23")].iloc[0]
    rows.append(row("23", "Относительный перенос: товары − рабочая сила, h = 4", o23.est, o23.ci90_lo, o23.ci90_hi,
                    o23.label, "`results/v9/s2_pt_outcomes.csv`", 0, 0.1, ">",
                    note="доли затрат в парах «нефтехимия ← нефть» занижены (журнал 4); с парами s ≥ 0,1 — `s2_pt_s01.csv`"))
    inf = r("s2_2021_outcomes.csv").set_index("item")
    o24 = inf.loc["outcome 24 (d2023): mean change of real hourly wage, control"]
    rows.append(row("24", "Реальная почасовая оплата 2021 → 2023, страны без индексации", o24.est, o24.ci90_lo, o24.ci90_hi,
                    o24.label, "`results/v9/s2_2021_outcomes.csv`", 0, 0.02, "[−0,02; 0,02] = H_tech"))
    o25 = inf.loc["outcome 25 (d2023): indexation - control"]
    rows.append(row("25", "Индексация (BE, LU, CY, MT) − контроль, 2021 → 2023", o25.est, o25.ci90_lo, o25.ci90_hi,
                    o25.label, "`results/v9/s2_2021_outcomes.csv`", 0, 0.02, ">",
                    note=f"перестановочный p = {fmt(o25.p_perm, 3)}; к 2024 г. разность исчезает"))
    an = r("s2_anchor.csv").set_index("item")
    o26 = an.loc["ECM u1 (outcome 26)"]
    v26 = an.loc["ECM u1, variant (centred gap; journal 3)"]
    rows.append(row("26", "Δ ln реальной оплаты часа на безработицу t−1 (AMECO, 18 стран)", o26.est, o26.ci90_lo,
                    o26.ci90_hi, o26.label, "`results/v9/s2_anchor.csv`", 0, 0.002, "<",
                    note=(f"дикий бутстреп p = {fmt(o26.p_wild, 3)}; вариант с центрированным разрывом (журнал 3): "
                          f"{fmt(v26.est, 4)} ({fmt(v26.ci90_lo, 4)}; {fmt(v26.ci90_hi, 4)}) — «{v26.label}»")))
    o27 = an.loc["institutions: ln(w/y) on cbc, per 10 pp (outcome 27)"]
    rows.append(row("27", "Уровень ln(w/y) на охват договорами, на 10 п. п.", o27.est, o27.ci90_lo, o27.ci90_hi,
                    o27.label, "`results/v9/s2_anchor.csv`", 0, 0.01, ">"))
    s34 = r("s34_outcomes.csv").set_index("outcome")
    o28 = s34.loc["28: countries where world-average labour beats national (majority of years)"]
    rows.append(row("28", "Среднемировой труд лучше национального (доля стран, большинство лет)", o28.est, o28.ci90_lo,
                    o28.ci90_hi, o28.label, "`results/v9/s3_national_vs_world.csv`", 0.5, 0.15, ">"))
    o29 = s34.loc["29 (2014)"]
    rows.append(row("29", "MAWD мирового труда − min(капитал, энергия), равный разброс, 2014", o29.est, o29.ci90_lo,
                    o29.ci90_hi, o29.label, "`results/v9/s3_outcome29_by_year.csv`", 0, 0.01, "<"))
    for k, name in (("30 b-c (2005)", "MAWD(часы с поправкой на производительность) − MAWD(фонд оплаты), ячейки, 2005"),
                    ("31 b-min(cap,energy) (2005)", "MAWD(поправленные часы) − min(капитал, энергия), 2005"),
                    ("32 a-b (2005)", "MAWD(сырые часы) − MAWD(поправленные часы), 2005")):
        o = s34.loc[k]
        rows.append(row(k[:2], name, o.est, o.ci90_lo, o.ci90_hi, o.label, "`results/v9/s4_outcomes_by_year.csv`", 0,
                        0.01, ">" if k.startswith("32") else "<"))
    o33 = s34.loc["33: R2(b) - R2(c), 5-year changes, mean of windows"]
    rows.append(row("33", "R²(поправленные часы) − R²(фонд оплаты), изменения за 5 лет", o33.est, o33.ci90_lo, o33.ci90_hi,
                    o33.label, "`results/v9/s34_outcomes.csv`", 0, 0.05, ">"))
    o34 = r("s5_b_between_outcome.csv").iloc[0]
    rows.append(row("34", "b между странами − b между отраслями (сгруппированная, WIOD)", o34.est, o34.ci90_lo,
                    o34.ci90_hi, o34.label, "`results/v9/s5_b_between*.csv`", 0, 0.1, "<"))
    s6 = r("s6_outcomes.csv")
    o35 = s6[s6.outcome.str.startswith("outcome 35 (stan)")].iloc[0]
    o35w = s6[s6.outcome.str.startswith("outcome 35 (wiod)")].iloc[0]
    rows.append(row("35", "Доля стран, где реальный курс коинтегрирован с относительной реальной стоимостью труда (STAN)",
                    o35.est, o35.ci90_lo, o35.ci90_hi, o35.label, "`results/v9/s6_coint.csv`", 0.05, 0.15, ">",
                    note=f"WIOD (15 лет): {int(o35w.k)} из {int(o35w.n)}, «{o35w.label}»"))
    o36 = s6[s6.outcome.str.startswith("outcome 36")].iloc[0]
    rows.append(row("36", "RMSE ECM / AR(1), прогноз реального курса на 3 года", o36.est, o36.ci90_lo, o36.ci90_hi,
                    o36.label, "`results/v9/s6_oos.csv`", 1, 0.05, "<"))
    s7 = r("s7_outcomes.csv")
    m = s7[s7.variant == "main"].set_index("item")
    for k, n, name in (("outcome 37: trend of r_m per year", "37", "Тренд марксовой нормы прибыли r_m (США, 1997–2024)"),
                       ("outcome 38: trend of r_m - trend of r", "38", "Тренд r_m − тренд обычной r")):
        o = m.loc[k]
        vv = s7[s7["item"] == k]
        rows.append(row(n, name, o.est, o.ci90_lo, o.ci90_hi, o.label, "`results/v9/s7_outcomes.csv`", 0, 0.0002, "<",
                        note="варианты классификации: " + "; ".join(f"{v.variant.split(' ')[0]} {fmt(v.est, 5)} «{v.label}»"
                                                                    for v in vv.itertuples() if v.variant != "main")))
    return pd.DataFrame(rows)


def write_md(D):
    L = ["# Сводная таблица первичных исходов итераций 1–9",
         "",
         "Правило вывода — [`pre_registration_v8.md`](../pre_registration_v8.md) (этап 1); исходы 20–38 и их пороги — "
         "[`pre_registration_v9.md`](../pre_registration_v9.md). Числа — из сохранённых файлов результатов (колонка "
         "«Источник»). Код — `src/lts/v8/stage1.py` (исходы 1–19) и `src/lts/v9/outcomes.py`; машиночитаемая версия — "
         "`results/v9/outcomes_table.csv`.",
         "",
         "* **подтверждено** — 90% ДИ не содержит θ₀, и оценка дальше Δ от θ₀ в ожидаемую сторону;",
         "* **опровергнуто** — 90% ДИ целиком в [θ₀ − Δ; θ₀ + Δ] или целиком по другую сторону от θ₀;",
         "* **неинформативно** — иначе.",
         "",
         "«За» — направление, в котором исход говорит в пользу трудовой теории / Маркса (для исходов этапа 2 — в "
         "пользу H_tech: рабочая сила производится как товар). Исходы 6, 16 и 19 — один факт (рецензия итерации 8, Р6).",
         "",
         "| # | Исход | Итерация | θ₀ | Δ | «За» | Оценка | 90% ДИ | Старая метка | Метка | Примечание | Источник |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in D.itertuples():
        ci = "—" if not (isinstance(x.lo, float) and np.isfinite(x.lo)) else f"{fmt(x.lo, 4)}; {fmt(x.hi, 4)}"
        th = fmt(x.theta0, 2) if isinstance(x.theta0, (int, float)) and x.theta0 != "" else x.theta0
        de = fmt(x.delta, 4) if isinstance(x.delta, (int, float)) and x.delta != "" else x.delta
        dr = f"{x.direction} θ₀" if x.direction in ("<", ">") else x.direction
        L.append(f"| {x.n} | {x.outcome} | {x.it} | {th} | {de} | {dr} | {fmt(x.est, 4)} | {ci} | {x.old} | "
                 f"**{x.label}** | {x.note} | {x.src} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    D = build()
    D.to_csv(OUT / "outcomes_table.csv", index=False)
    (ROOT / "report" / "outcomes_table.md").write_text(write_md(D))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    print(D[["n", "outcome", "est", "lo", "hi", "label"]].round(4).to_string())
