"""Iteration 12: summary table of primary outcomes of iterations 1-12 (report/outcomes_table.md).

Rows 1-74 from iteration 11 (lts.v11.outcomes.build); new rows 75-89 from results/v12; notes on rows 28-33, 44-46
(China hours in the WIOD world system) and 64 (Korea check).
"""
from __future__ import annotations

import pandas as pd

from ..v6.series import ROOT
from ..v8.stage1 import fmt
from ..v11 import outcomes as v11o

OUT = ROOT / "results" / "v12"


def r(p):
    return pd.read_csv(OUT / p)


def ci(x):
    return f"{fmt(x.est, 3)} ({fmt(x.ci90_lo, 3)}; {fmt(x.ci90_hi, 3)})"


def build():
    D = v11o.build()

    def add_note(n, txt):
        i = D.index[D.n == n][0]
        cur = D.loc[i, "note"]
        D.loc[i, "note"] = (str(cur) + " " if isinstance(cur, str) and cur else "") + txt
    china = ("12 (журнал v12 п. 3): в мировой системе WIOD (`lts.v9.stage34.year_system`) часы Китая нулевые — в SEA нет "
             "часов наёмных Китая, вменение делалось только для остального мира; исход не пересчитан (открытый вопрос).")
    for n in ("28", "29", "30", "31", "32", "33", "44", "45", "46"):
        add_note(n, china)
    k = r("k_results.csv").set_index("item")
    ka, kb = k.loc["(a) Korea with donor-median split: DD"], k.loc["(b) Korea without split industries: DD"]
    add_note("64", f"12 (проверка Кореи): FIGARO с донорской разбивкой занятости Кореи — {ci(ka)} «{ka.label}»; без разбитых "
                   f"отраслей Кореи (2013–2022 гг. разбиты почти все) — {ci(kb)} «{kb.label}». Способ разбивки эффект не "
                   f"снимает; эффект сосредоточен в разбитых отраслях Кореи — открытый вопрос.")
    rows = []

    def row(n, name, x, src, theta0, delta, direction, note=""):
        rows.append(dict(n=n, outcome=name, it="12", theta0=theta0, delta=delta, direction=direction, est=x.est,
                         lo=x.ci90_lo, hi=x.ci90_hi, old="— (новый исход)", label=x.label, src=src, note=note))
    s1 = r("s1_outcomes.csv")
    o1 = {int(s.split(":")[0]): s1.iloc[i] for i, s in enumerate(s1.outcome)}
    cf = r("s1_counterfactual_trends.csv")
    a = cf[cf.series == "A: US NFC"].set_index("base")
    src1 = "`results/v12/s1_outcomes.csv`"
    row("75", "США НФК: тренд ln r_cf (σ, ρ, F на уровне 1950 г.), 1950–2024", o1[75], src1, 0, 0.002, "<",
        note=f"по базовым годам: 1965 — {ci(a.loc[1965])} «{a.loc[1965].label}», 1980 — {ci(a.loc[1980])} «{a.loc[1980].label}», "
             f"1997 — {ci(a.loc[1997])} «{a.loc[1997].label}». Фактическая ln r_full: −0,0001 (неинформативно). Держит r_cf "
             f"рост q (+0,48 лог. п. за 1950–2024); ρ = P_Y/P_K упала на 0,55 — относительная цена капитала росла "
             f"(разделение ρ и q зависит от дефляторов).")
    row("76", "США НФК: тренд ln(r_cf / r_full), 1950–2024 — противодействующие факторы поддерживали r", o1[76], src1, 0,
        0.002, "<", note="знак обратный: вместе σ, ρ и F снижали r; F (зарубежная прибыль) действовал в пользу r "
                         "(+0,11 лог. п.), ρ — против (−0,55). Итог этапа 1 по правилу — «не поддержан».")
    row("77", "США, частная экономика (марксовы категории 9б): тренд ln r_B,cf (S/ЧВП, 1 − U/S, P/P_K, F на 1950 г.)", o1[77],
        src1, 0, 0.002, "<", note="непроизводительная оплата (1 − U/S) снизила r_B на 0,16 лог. п., P/P_K — на 0,34; "
                                  "q_B выросло на 0,19. При всех базовых годах «опровергнуто».")
    row("78", "США НФК: тренд ln q = ln(Q_Y / Q_K), 1950–2024", o1[78], src1, 0, 0.002, "<",
        note="производительность часа (Q_Y/H) росла быстрее реального капитала на час (Q_K/H): 1,09 против 0,81 лог. п. за "
             "1970–2024. В текущих ценах Y/K (ρ·q) за 1950–2024 изменилось на −0,07 лог. п.")
    row("79", "AMECO: доля стран (из 10), где тренд ln r_cf (база 1965 г. или первый год) «подтверждён» (< 0)", o1[79], src1,
        0.5, 0.15, ">", note="5 из 10 (CAN, FRA, ITA, JPN, SWE); база CAN — 1991, JPN — 1980 (ряды начинаются позже).")
    row("80", "США: тренд доли прибыли корпораций от остального мира (B394RC / A051RC), 1948–2024", o1[80], src1, 0, 0.001,
        ">", note="доля 0,040 (1948) → 0,112 (1973) → 0,121 (1997) → 0,290 (2008) → 0,120 (2024); вклад в Δ ln r корпораций: "
                  "+0,077 (1948–1973), +0,010 (1973–1997), −0,001 (1997–2024).")
    s2 = r("s2_outcomes.csv")
    fr = r("s2_outcomes_firstrun_china_missing.csv")
    src2 = "`results/v12/s2_outcomes.csv`"
    row("81", "Круг: β оттока часов O_c (сбалансированный, сырые часы) на рост относительной производительности за 5 лет "
              "(WIOD 2000–2014, 41 страна)", s2.iloc[0], src2, 0, 0.01, "<",
        note=f"первый прогон с нулевыми часами Китая (журнал 3): {ci(fr.iloc[0])}; все варианты неинформативны, кроме "
             f"эффектов стран: +0,026 (0,008; 0,043) «опровергнуто».")
    row("82", "Круг: β строгого оттока S_c (часы с поправкой на производительность)", s2.iloc[1], src2, 0, 0.01, "<",
        note=f"первый прогон: {ci(fr.iloc[1])}. Итог 2.2 — «неразличимо».")
    s3 = r("s3_outcomes.csv").iloc[0]
    d3 = r("s3_descriptive.csv").set_index("item")
    yfe = d3.loc["83 variant: with year effects"]
    cont = d3.loc["83 variant: continuous allocation of output to departments"]
    row("83", "Схемы воспроизводства: LPM начала рецессии в t+1/t+2 на D_t = Δ₃ ln(X_I/X_II) (станд.), с Δ₃ ln(ВНОК/ВВП), "
              "эффекты стран, WIOD 2003–2014", s3, "`results/v12/s3_outcomes.csv`", 0, 0.02, ">",
        note=f"с эффектами лет — {ci(yfe)}; непрерывное разнесение — {ci(cont)}; оба неинформативны. Держится на общем "
             f"времени: из 55 начал рецессий 2003–2016 гг. 37 приходятся на 2008–2009 гг., 14 — на 2011–2013 гг.")
    s6 = r("s6_outcomes.csv")
    fb = r("s6_outcomes_fallback.csv")
    src6 = "`results/v12/s6_outcomes.csv`"
    x84 = s6.iloc[0]
    fbn = "с крупными группами KLEMS для непокрытых отраслей (журнал 9, после результата)"
    row("84", "Рента: среднее d = ln(p/p_пп) по A01 и B, цены производства на запас (FIGARO × KLEMS, 11 стран, 2010–2021)",
        x84, src6, 0, 0.05, ">", note=f"A01 {fmt(x84.d_A01, 3)}, B {fmt(x84.d_B, 3)}; перцентиль плацебо "
                                      f"{fmt(x84.placebo_percentile, 3)} (медиана плацебо {fmt(x84.placebo_median, 3)}, "
                                      f"пул — {int(x84.pool)} отраслей). Вариант {fbn}: {ci(fb.iloc[0])} «{fb.iloc[0].label}».")
    row("85", "Рента: d недвижимости (L), цены производства на запас", s6.iloc[1], src6, 0, 0.05, ">",
        note=f"превышение L итерации 10 (+0,42 на потоке) — артефакт цен производства на поток; d(L) на реальные цены жилья "
             f"BIS — неинформативно. Земля в K не входит. Вариант {fbn}: {ci(fb.iloc[1])} «{fb.iloc[1].label}».")
    row("86", "Рента: эластичность d(B) по ln реальному индексу энергии и металлов (цены на запас, эффекты стран)", s6.iloc[2],
        src6, 0, 0.05, ">", note=f"повтор исхода 61 на ценах производства на запас. Вариант {fbn}: {ci(fb.iloc[2])} "
                                 f"«{fb.iloc[2].label}».")
    s4 = r("s4_outcomes.csv")
    src4 = "`results/v12/s4_outcomes.csv`"
    row("87", "Браверман: средний сдвиг самостоятельности внутри профессий, O*NET 15.0 → 30.0 (SD)", s4.iloc[0], src4, 0, 0.1,
        "<", note="605 переоценённых профессий; 20.1 → 30.0: −0,094 (−0,143; −0,046). Доля оценок профессиональных экспертов выросла "
                  "с 16% до 24% — сопоставимость версий ограничена.")
    row("88", "Браверман: сдвиг контроля темпа оборудованием внутри профессий (SD)", s4.iloc[1], src4, 0, 0.1, ">",
        note="рутинность (повторение задач) выросла на 0,12 SD (0,06; 0,18), описательно.")
    row("89", "Поляризация: изменение доли занятых Job Zone 3 (середина), OEWS 2004 → 2023", s4.iloc[2], src4, 0, 0.02, "<",
        note="Job Zone 1: 0,177 → 0,064; 4: 0,148 → 0,273 (версии O*NET 10.0 и 30.0, классификации SOC 2000 и 2018).")
    return pd.concat([D, pd.DataFrame(rows)], ignore_index=True)


def write_md(D):
    t = v11o.write_md(D)
    t = t.replace("# Сводная таблица первичных исходов итераций 1–11", "# Сводная таблица первичных исходов итераций 1–12")
    t = t.replace("исходы 64–74 — [`pre_registration_v11.md`](../pre_registration_v11.md).",
                  "исходы 64–74 — [`pre_registration_v11.md`](../pre_registration_v11.md); "
                  "исходы 75–89 — [`pre_registration_v12.md`](../pre_registration_v12.md).")
    t = t.replace("`src/lts/v11/outcomes.py`; машиночитаемая версия — `results/v11/outcomes_table.csv`.",
                  "`src/lts/v11/outcomes.py`, `src/lts/v12/outcomes.py`; машиночитаемая версия — "
                  "`results/v12/outcomes_table.csv`.")
    return t


if __name__ == "__main__":
    D = build()
    D.to_csv(OUT / "outcomes_table.csv", index=False)
    (ROOT / "report" / "outcomes_table.md").write_text(write_md(D))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 70)
    print(D[["n", "outcome", "est", "lo", "hi", "label"]].tail(16).round(4).to_string())
