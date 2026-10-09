"""Iteration 11: summary table of primary outcomes of iterations 1-11 (report/outcomes_table.md).

Rows 1-63 from iteration 10 (lts.v10.outcomes.build); new rows 64-74 from results/v11.
"""
from __future__ import annotations

import pandas as pd

from ..v6.series import ROOT
from ..v8.stage1 import fmt
from ..v10 import outcomes as v10o

OUT = ROOT / "results" / "v11"


def r(p):
    return pd.read_csv(OUT / p)


def build():
    D = v10o.build()
    rows = []

    def row(n, name, x, src, theta0, delta, direction, note=""):
        rows.append(dict(n=n, outcome=name, it="11", theta0=theta0, delta=delta, direction=direction, est=x.est,
                         lo=x.ci90_lo, hi=x.ci90_hi, old="— (новый исход)", label=x.label, src=src, note=note))
    s1 = r("s1_outcomes.csv")
    ds = r("s1_descriptive.csv").set_index("outcome")
    ck = r("ck_s1_subsets.csv").set_index("item")
    pl = r("s1_placebo_summary.csv").set_index(r("s1_placebo_summary.csv").columns[0]).iloc[:, 0]
    fdd = ds.loc["FIGARO DD (hours)"]
    w13 = ck.loc["WIOD 13 FIGARO countries: DD"]
    row("64", "Закон стоимости: DD = (b_v − b_pp) в отраслях самозанятых − то же в корпоративных, Δ ln часов за 3 года "
        "(WIOD, 42 страны, 2001–2010)", s1.iloc[0], "`results/v11/s1_outcomes.csv`", 0, 0.02, ">",
        note=f"ниже всех 200 плацебо-перестановок (95-й перцентиль {fmt(float(pl['dd_perm_p95']), 3)}); FIGARO 2011–2018: "
             f"{fmt(fdd.est, 3)} ({fmt(fdd.ci90_lo, 3)}; {fmt(fdd.ci90_hi, 3)}), выше всех 50 плацебо; WIOD на тех же 13 "
             f"странах — {fmt(w13.est, 3)} (журнал 3, после результата).")
    row("65", "Корпоративные отрасли: b_pp − b_v, часы", s1.iloc[1], "`results/v11/s1_outcomes.csv`", 0, 0.02, ">")
    row("66", "Число самозанятых (σ ≥ 0,10): b_v − b_pp", s1.iloc[2], "`results/v11/s1_outcomes.csv`", 0, 0.05, ">",
        note="опровергнуто через эквивалентность; при σ ≥ 0,25 — −0,060 (−0,101; −0,018).")
    row("67", "Инвестиции (EU KLEMS), корпоративные отрасли: b_pp − b_v", s1.iloc[3], "`results/v11/s1_outcomes.csv`", 0,
        0.05, ">", note="b_pp = 0,024 (0,002; 0,045), b_v = 0,013 (−0,012; 0,037).")
    s2 = r("s2_outcomes.csv")
    t69 = r("ck_69_i1.csv")
    row("68", "Доля лет с i_real (долгосрочная) < r (PWT), среднее по 18 странам JST, 1950–2020", s2.iloc[0],
        "`results/v11/s2_outcomes.csv`", 0.5, 0.15, ">", note="условие М; его предсказывает и теория с премией за риск.")
    row("69", "Доля стран, где i_real и r коинтегрированы (Энгл–Грейнджер, p < 0,10)", s2.iloc[1],
        "`results/v11/s2_outcomes.csv`", 0.5, 0.15, "<",
        note=f"оговорка из предрегистрации: i_real стационарна (ADF) в {int(t69.n.iloc[1])} из 18 странах; где оба ряда "
             f"нестационарны ({int(t69.n.iloc[0])}), коинтеграция в {int(t69.cointegrated.iloc[0])}; номинальная ставка — 0 из 18.")
    row("70", "Изменение r − i (краткосрочная номинальная) в первый год рецессии против прочих лет", s2.iloc[2],
        "`results/v11/s2_outcomes.csv`", 0, 0.005, "<",
        note="долгосрочная номинальная: −0,0052 (−0,0069; −0,0036); годы кризисов JST: −0,0096 (−0,0148; −0,0044).")
    row("71", "США НФК: аномальный рост q = (акции + долг) / основной капитал за 3 года до пиков NBER", s2.iloc[3],
        "`results/v11/s2_outcomes.csv`", 0, 0.05, ">")
    dec = r("ck_72_decomposition.csv").set_index("component")
    row("72", "США НФК: аномальный рост рыночной стоимости прав / капитализированной прибыли (NOS / Aaa) до пиков",
        s2.iloc[4], "`results/v11/s2_outcomes.csv`", 0, 0.05, ">",
        note=f"разложение (после результата): ln MV {fmt(dec.loc['ln MV', 'est'], 3)}, ln Aaa {fmt(dec.loc['ln Aaa', 'est'], 3)}, "
             f"−ln NOS {fmt(dec.loc['ln NOS (sign flipped)', 'est'], 3)} — больше половины от роста ставки.")
    row("73", "JST: аномальный рост ln(P/D) за 3 года до финансовых кризисов", s2.iloc[5], "`results/v11/s2_outcomes.csv`",
        0, 0.05, ">")
    row("74", "JST: аномальный рост кредит/ВВП за 3 года до финансовых кризисов", s2.iloc[6],
        "`results/v11/s2_outcomes.csv`", 0, 0.02, ">", note="повтор Schularick & Taylor (2012); после 1945 г. — 0,121.")
    return pd.concat([D, pd.DataFrame(rows)], ignore_index=True)


def write_md(D):
    t = v10o.write_md(D)
    t = t.replace("# Сводная таблица первичных исходов итераций 1–10", "# Сводная таблица первичных исходов итераций 1–11")
    t = t.replace("исходы 47–63 (52 не присваивался) — [`pre_registration_v10.md`](../pre_registration_v10.md).",
                  "исходы 47–63 (52 не присваивался) — [`pre_registration_v10.md`](../pre_registration_v10.md); "
                  "исходы 64–74 — [`pre_registration_v11.md`](../pre_registration_v11.md).")
    t = t.replace("`src/lts/v10/outcomes.py`; машиночитаемая версия — `results/v10/outcomes_table.csv`.",
                  "`src/lts/v10/outcomes.py`, `src/lts/v11/outcomes.py`; машиночитаемая версия — "
                  "`results/v11/outcomes_table.csv`.")
    return t


if __name__ == "__main__":
    D = build()
    D.to_csv(OUT / "outcomes_table.csv", index=False)
    (ROOT / "report" / "outcomes_table.md").write_text(write_md(D))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 70)
    print(D[["n", "outcome", "est", "lo", "hi", "label"]].tail(14).round(4).to_string())
