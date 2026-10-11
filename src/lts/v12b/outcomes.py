"""Iteration 12b: summary table of primary outcomes of iterations 1-12b (report/outcomes_table.md).

Rows 1-89 from iteration 12 (lts.v12.outcomes.build); the iteration-12 note on rows 28-33, 44-46, 54, 55 (China hours
zero in the WIOD world system) is replaced by the stage-0 recomputation (results/v12b/s0_compare.csv); new rows 90-109
from results/v12b.

Usage: PYTHONPATH=src python -P -m lts.v12b.outcomes
"""
from __future__ import annotations

import pandas as pd

from ..v6.series import ROOT
from ..v8.stage1 import fmt
from ..v12 import outcomes as v12o

OUT = ROOT / "results" / "v12b"
CHINA_V12 = ("12 (журнал v12 п. 3): в мировой системе WIOD (`lts.v9.stage34.year_system`) часы Китая нулевые — в SEA нет "
             "часов наёмных Китая, вменение делалось только для остального мира; исход не пересчитан (открытый вопрос).")
BEA_NOTE = ("12б (журнал v12b п. 1): в экономиках BEA (`lts.bea`) труд страхования (524) завышен примерно в 11–25 раз "
            "(отношение самозанятые / наёмные OECD Table 7 для K65 — 10–24); экономика BEA — один из наборов этого исхода; "
            "не пересчитано (открытый вопрос).")


def r(p):
    return pd.read_csv(OUT / p)


def f5(v):
    return f"{v:.5f}".replace(".", ",").replace("-", "−")


def ci5(x):
    return f"{f5(x.est)} ({f5(x.ci90_lo)}; {f5(x.ci90_hi)})"


def ci3(est, lo, hi):
    return f"{fmt(est, 3)} ({fmt(lo, 3)}; {fmt(hi, 3)})"


def build():
    D = v12o.build()

    def note_of(n):
        return D.index[D.n == n][0]

    s0 = r("s0_compare.csv")
    for n in ("28", "29", "30", "31", "32", "33", "44", "45", "46", "54", "55"):
        i = note_of(n)
        a = s0[(s0.variant == "a") & (s0.n == int(n))].iloc[0]
        b = s0[(s0.variant == "b") & (s0.n == int(n))].iloc[0]
        if n == "55":
            new = ("12 (журнал v12 п. 3): часы Китая в мировой системе WIOD были нулевыми. 12б (этап 0): с часами Китая — "
                   f"(а) {fmt(a.est_new, 0)} из 5, (б) {fmt(b.est_new, 0)} из 5; не изменилось.")
        else:
            ch = [v for v, x in (("(а)", a), ("(б)", b)) if bool(x.label_changed)]
            tail = (f"**метка меняется при {', '.join(ch)}**" if ch else "метка не изменилась")
            new = ("12 (журнал v12 п. 3): часы Китая в мировой системе WIOD были нулевыми. 12б (этап 0): с часами Китая — "
                   f"(а) EMP × avh PWT: {ci3(a.est_new, a.lo_new, a.hi_new)} «{a.label_new}»; (б) донорская медиана: "
                   f"{ci3(b.est_new, b.lo_new, b.hi_new)} «{b.label_new}»; {tail}.")
        D.loc[i, "note"] = str(D.loc[i, "note"]).replace(CHINA_V12, new)
    for n in ("7", "18"):                                   # rows 1, 2, 5 exclude USA_BEA (lts.v8.stage1)
        i = note_of(n)
        cur = D.loc[i, "note"]
        D.loc[i, "note"] = (str(cur) + " " if isinstance(cur, str) and cur else "") + BEA_NOTE
    rows = []

    def row(n, name, x, src, theta0, delta, direction, note="", est=None):
        rows.append(dict(n=n, outcome=name, it="12б", theta0=theta0, delta=delta, direction=direction,
                         est=x.est if est is None else est, lo=x.get("ci90_lo"), hi=x.get("ci90_hi"),
                         old="— (новый исход)", label=x.label, src=src, note=note))
    # ---- stage 1, US
    s1 = r("s1_outcomes.csv")
    d1 = r("s1_descriptive.csv").set_index("item")
    src1 = "`results/v12b/s1_outcomes.csv`"
    v = lambda k: ci5(d1.loc[k]) + f" «{d1.loc[k].label}»"               # noqa: E731
    row("90", "США: тренд ln r_h (норма прибыли в часах труда), 1948–2023", s1.iloc[0], src1, 0, 0.002, "<",
        note="устойчиво во всех 9 сочетаниях лагов 2/4/8 и концов. Уровни r_h: 0,424 (1948), 0,435 (1973), 0,483 (1997), "
             f"0,460 (2023). Варианты: все отрасли {v('variant [all industries (L, K, basket)]: trend ln r_h')}; λ с "
             f"амортизацией {v('variant [lambda with depreciation]: trend ln r_h')}; с запасами "
             f"{v('variant [advanced capital incl. inventories]: trend ln r_h')}; корзины (i)–(iii) от +0,0014 до +0,0019 "
             "«опровергнуто». Труд — лица, не часы (журнал v12b п. 1).")
    row("91", "США: тренд ln k_h (стоимостное строение в часах), 1948–2023", s1.iloc[1], src1, 0, 0.002, ">",
        note="неустойчиво (3 из 9 сочетаний — «опровергнуто»). Разложение: (а) ln(Q_K/L) +0,0106, (б) ln(h_K/h_C) "
             "+0,0066, (в) ln(Q_C/L) +0,0160 — все «подтверждено»; рост (а) и (б) почти погашен ростом (в).")
    row("92", "США: тренд ln(r_h / r_$) — специфичность падения для часов против денег", s1.iloc[2], src1, 0, 0.002, "<",
        note=f"тренд ln r_$ {v('trend ln r_$')}; корреляция уровней r_h и r_$ 0,944. Динамика в часах и в деньгах "
             "совпадает в пределах ±0,2% в год.")
    row("93", "США: тренд ln(r_h / r_E) — против энергии, 1949–2023", s1.iloc[3], src1, 0, 0.002, "<",
        note=f"r_E колеблется сильно (0,24–0,71); тренд ln r_E {v('trend ln r_E (1949-)')}.")
    x94 = s1.iloc[4].copy()
    rows.append(dict(n="94", outcome="США: доля (а) технического строения в тренде ln k_h", it="12б", theta0=0.5, delta=0.1,
                     direction=">", est=x94.share_point, lo=None, hi=None, old="— (новый исход)",
                     label="не применимо (91 не подтверждён)", src=src1,
                     note="точечная доля 9,1: тренд k_h мал, потому что (а) и (б) погашены (в)."))
    # ---- stage 1.6 and world hours
    w = r("s1w_outcomes.csv")
    dw = r("s1w_descriptive.csv").set_index("item")
    srcw = "`results/v12b/s1w_outcomes.csv`"
    vw = lambda k: ci5(dw.loc[k]) + f" «{dw.loc[k].label}»"               # noqa: E731
    lab = dw[dw.index.str.match(r"WIOD [A-Z]{3}: trend ln r_h")].label.value_counts()
    row("95", "WIOD 2000–2014: общий внутристрановой тренд ln r_h (эффекты стран, кластеры — страны)", w.iloc[0], srcw, 0,
        0.002, "<", note=f"43 страны, Китай — вменение этапа 0 (а). По странам: «подтверждено» {lab.get('подтверждено', 0)}, "
                         f"«опровергнуто» {lab.get('опровергнуто', 0)}, «неинформативно» {lab.get('неинформативно', 0)}; США "
                         f"{vw('WIOD USA: trend ln r_h')}. Денежная норма: {vw('WIOD within-country trend of ln r_$')}; k_h "
                         f"{vw('WIOD within-country trend of ln k_h')}.")
    row("96", "WIOD: то же для ln(r_h / r_$)", w.iloc[1], srcw, 0, 0.002, "<",
        note="wild-кластерный p = 0,20; падение в часах не отличимо от падения в деньгах.")
    # ---- stage 2
    s2 = r("s2_outcomes.csv")
    d2 = r("s2_descriptive.csv").set_index("item")
    src2 = "`results/v12b/s2_outcomes.csv`"
    v2 = lambda k: ci5(d2.loc[k]) + f" «{d2.loc[k].label}»"               # noqa: E731
    row("97", "Мировая r (PWT 11, по курсу, сбалансированная панель 1960–2019, 79 стран): тренд ln", s2.iloc[0], src2, 0,
        0.002, "<", note=f"r_w 0,116 (1960) → 0,079 (2019). 1960–1980: {v2('trend ln world r, PWT xr balanced, 1960-1980')}; "
                         f"1980–2019: {v2('trend ln world r, PWT xr balanced, 1980-2019')}. По ППС "
                         f"{v2('trend ln PWT ppp balanced')}; веса Мaито {v2('trend ln PWT xr balanced, GDP-weighted mean (Maito-style)')}; "
                         "EPWT 7.0 в текущих ценах (журнал 4, после результата) −0,00531 «подтверждено», в постоянных ППС 2017 г. "
                         "+0,00411 «опровергнуто» (несопоставимо).")
    row("98", "Мировая r: тренд накопленного индекса «внутри стран» (/ r_w 1960)", s2.iloc[1], src2, 0, 0.002, "<",
        note=f"по ППС {v2('PWT ppp: trend cumulative within (/ r_w 1960)')}; EPWT в текущих ценах "
             f"{v2('EPWT 7.0 current: trend cumulative within (/ r_w 1960)')}.")
    row("99", "Мировая r: тренд накопленного индекса «веса стран» (маскировка)", s2.iloc[2], src2, 0, 0.002, ">",
        note="доля Китая в K сбалансированной панели 0,034 (1960) → 0,164 (2019). Сдвиг весов к странам с более высокой r "
             "частично маскирует падение внутри стран.")
    row("100", "WIOD, мировая система (часы Китая вменены): тренд ln мировой r_h, 2000–2014", w.iloc[2], srcw, 0, 0.002, "<",
        note=f"мировая r_$: {vw('WIOD world system: trend ln r_$')}; ln(r_h / r_$): "
             f"{vw('WIOD world system: trend ln(r_h / r_$)')}. 15 лет.")
    # ---- stage 3
    s3 = r("s3_outcomes.csv")
    d3 = r("s3_descriptive.csv").set_index("item")
    src3 = "`results/v12b/s3_outcomes.csv`"
    v3 = lambda k: ci5(d3.loc[k]) + f" «{d3.loc[k].label}»"               # noqa: E731
    row("101", "США, обработка: тренд ln r (чистая, текущая стоимость), 1948–2023", s3.iloc[0], src3, 0, 0.002, "<",
        note="устойчиво (9 из 9). r 0,353 (1948) → 0,083 (1980) → 0,190 (1997) → 0,160 (2023). Шов 1986/87 у оплаты "
             f"(журнал 7, после результата): со сшивкой {v3('101 variant (journal 7, after result): SIC share spliced to NAICS 1987')}. "
             f"Подпериоды: 1948–1973 {v3('US trend ln r manufacturing 1948-1973')}, 1973–1997 "
             f"{v3('US trend ln r manufacturing 1973-1997')}, 1997–2023 {v3('US trend ln r manufacturing 1997-2023')}. "
             f"r_h обработки {v3('US trend ln r_h manufacturing')}.")
    row("102", "США: тренд ln(r_обработки / r_частного сектора)", s3.iloc[1], src3, 0, 0.002, "<",
        note=f"r частного сектора {v3('US trend ln r private sector')}; со сшивкой 1987 г. "
             f"{v3('102 variant (journal 7, after result): ln(r_mfg / r_priv), both spliced')}.")
    row("103", "Европа (EU KLEMS 2009 + 2025, ряды ≥ 30 лет): доля стран с подтверждённым падением r обработки", s3.iloc[2],
        src3, 0.5, 0.15, ">", note="2 из 8 (Испания, Италия); 6 — «опровергнуто» (рост). Валовая норма (ДС − LAB) / K — в "
                                  "KLEMS 2025 нет амортизации (журнал 6). Чистая норма KLEMS 2009 (1970–2007): 2 из 7.")
    # ---- stage 4
    s4 = r("s4_outcomes.csv")
    d4 = r("s4_descriptive.csv").set_index("item")
    src4 = "`results/v12b/s4_outcomes.csv`"
    v4 = lambda k: ci5(d4.loc[k])                                          # noqa: E731
    row("104", "США НФК: g реальной массы прибыли в год пика минус прочие годы", s4.iloc[0], src4, 0, 0.02, "<",
        note=f"11 пиков NBER. Год до пика (t − 2 → t − 1): {v4('US: d ln real mass in the year before the peak year (t-2 -> t-1) minus other years')}. "
             f"Разложение года пика: Δ ln(Π/Y) {v4('US peak year: dshare minus other years')}, Δ ln Y "
             f"{v4('US peak year: dY_real minus other years')}, Δ доли оплаты {v4('US peak year: dws minus other years')}. "
             "Дефляторы ИПЦ и собственный — «подтверждено».")
    row("105", "США НФК: Δ ln r за t − 4 → t − 1 у пиков минус прочие годы", s4.iloc[1], src4, 0, 0.02, "<")
    row("106", "США НФК: Δ ln r за t → t + 3 у пиков минус прочие годы (восстановление)", s4.iloc[2], src4, 0, 0.02, ">")
    row("107", "JST (18 стран, AMECO): g реальной массы прибыли в год пика минус прочие годы", s4.iloc[3], src4, 0, 0.02,
        "<", note=f"100 пиков. С эффектами лет {v4('107 variant: with year effects')}; финансовые кризисы JST "
                  f"{v4('107 variant: JST financial crisis years')}. Разложение: Δ ln(Π/Y) "
                  f"{v4('JST peak year: dshare minus other years')}, Δ ln Y {v4('JST peak year: dY_real minus other years')}, "
                  f"Δ доли оплаты {v4('JST peak year: dws minus other years')}, Δ ln I {v4('JST peak year: dI minus other years')}.")
    row("108", "JST: Δ ln r за t − 4 → t − 1 у пиков минус прочие годы", s4.iloc[4], src4, 0, 0.02, "<")
    row("109", "JST: Δ ln r за t → t + 3 у пиков минус прочие годы", s4.iloc[5], src4, 0, 0.02, ">")
    return pd.concat([D, pd.DataFrame(rows)], ignore_index=True)


def write_md(D):
    t = v12o.write_md(D)
    t = t.replace("# Сводная таблица первичных исходов итераций 1–12", "# Сводная таблица первичных исходов итераций 1–12б")
    t = t.replace("исходы 75–89 — [`pre_registration_v12.md`](../pre_registration_v12.md).",
                  "исходы 75–89 — [`pre_registration_v12.md`](../pre_registration_v12.md); "
                  "исходы 90–109 — [`pre_registration_v12b.md`](../pre_registration_v12b.md).")
    t = t.replace("`src/lts/v11/outcomes.py`, `src/lts/v12/outcomes.py`; машиночитаемая версия — "
                  "`results/v12/outcomes_table.csv`.",
                  "`src/lts/v11/outcomes.py`, `src/lts/v12/outcomes.py`, `src/lts/v12b/outcomes.py`; машиночитаемая "
                  "версия — `results/v12b/outcomes_table.csv`.")
    return t


if __name__ == "__main__":
    D = build()
    D.to_csv(OUT / "outcomes_table.csv", index=False)
    (ROOT / "report" / "outcomes_table.md").write_text(write_md(D))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 70)
    print(D[["n", "outcome", "est", "lo", "hi", "label"]].tail(21).round(5).to_string())
