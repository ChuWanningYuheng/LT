"""Iteration 13a: report/REPORT_v13a.md generated from the CSV files of results/v13a (all numbers come from files).

Usage: PYTHONPATH=src python -P -m lts.v13a.report
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..v6.series import ROOT
from . import est as est_
from . import matrix as mx
from .stage1 import CRISES as CR1, NBER_TROUGH, PHASES as PH1

R = ROOT / "results" / "v13a"


def rd(name):
    p = R / name
    return pd.read_csv(p) if p.exists() else None


def num(x, d=4):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(x):
        return "—"
    s = f"{x:.{d}f}".replace(".", ",")
    return s.replace("-", "−")


def _f(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def pct(x, d=2):
    """slope of a log per year -> percent a year."""
    x = _f(x)
    return "—" if x is None else num(100 * x, d) + "%"


def ci_pct(e, lo, hi, d=2):
    lo, hi = _f(lo), _f(hi)
    if lo is None or hi is None:
        return pct(e, d)
    return f"{pct(e, d)} ({num(100 * lo, d)}; {num(100 * hi, d)})"


def lab(x):
    return f"«{x}»" if isinstance(x, str) and x else "—"


def md_table(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


NAMES = {110: "тренд ln k*_h внутри фаз (историческая оценка, (В))", 111: "тренд ln r*_h внутри фаз",
         112: "восстановление между фазами, средняя Δ ln r*_h", 113: "R: моральный износ в кризисных окнах / внутри фаз",
         114: "контрфакт Окисио, текущая оценка", 115: "контрфакт Окисио, историческая оценка",
         116: "ln e*_h: наклон B минус A′", 117: "ln(r^(В) / r^(П)) внутри фаз", 118: "доля разрыва r НФК и r*_h (Мозли)",
         123: "ln r_h,curr внутри фаз (текущая оценка, (В))", 124: "ln k_h,curr внутри фаз (текущая оценка, (В))",
         125: "ln(C_h,hist^(П) / L) внутри фаз", 126: "ln r*_h: наклон B минус A", 127: "ln(C/L)^(П): наклон B минус A",
         128: "ln e с управляющими в V: наклон B минус A′", 129: "ln r_TSSI (MELT) внутри фаз",
         130: "ln(r*_h / r*_$) внутри фаз", 131: "вековой тренд ln r*_h, 1948–2023 (описательно)",
         120: "мировая ln r_h внутри фаз", 121: "мировой ln k_h внутри фаз", 122: "мир: ln(r_h / r_$) внутри фаз",
         119: "доля работ, где наш аналог воспроизводит тренд автора (TOST ± 0,002)"}


def outcome_rows(O, nos):
    rows = []
    for no in nos:
        g = O[O.no == no]
        if not len(g):
            continue
        r = g.iloc[0]
        th0 = r.get("theta0", 0.0)
        if no in (113,):
            est = f"{num(r.est, 2)} ({num(r.ci90_lo, 2)}; {num(r.ci90_hi, 2)})"
        elif no in (112,):
            est = num(r.est, 3)
        else:
            est = ci_pct(r.est, r.ci90_lo, r.ci90_hi)
        rob = r.get("robust")
        robs = "—" if not isinstance(rob, (bool, np.bool_)) else ("да" if rob else f"нет: {r.get('robust_why', '')}")
        rows.append([no, NAMES.get(no, r.outcome), est, lab(r.get("label_full", r.get("label"))), robs])
    return rows


def build():
    O1 = rd("s1_outcomes.csv")
    rule = rd("s1_rule.csv").set_index("item").value
    ident = rd("s1_identity.csv").set_index("series")
    sh = rd("s1_shapley.csv")
    desc = rd("s1_descriptive.csv").set_index("item")
    mon = rd("s1_money_outcomes.csv").set_index("no")
    s2c, s2p, s2b, s2m, s2o = rd("s2_compare.csv"), rd("s2_compare_pt.csv"), rd("s2_bridge.csv"), rd("s2_mohun_text.csv"), \
        rd("s2_outcomes.csv")
    O3, s3d, s3m = rd("s3_outcomes.csv"), rd("s3_descriptive.csv"), rd("s3_matrix.csv")
    s4d, s4f = rd("s4_dumenil_levy.csv"), rd("s4_insurance_fix.csv")

    def o(no):
        return O1[O1.no == no].iloc[0]
    o110, o111, o113, o114, o115, o116, o117 = (o(n) for n in (110, 111, 113, 114, 115, 116, 117))
    o123, o124, o125, o126, o127, o128, o129, o130, o131 = (o(n) for n in (123, 124, 125, 126, 127, 128, 129, 130, 131))
    o112 = o(112)
    o118 = O1[O1.no == 118].iloc[0]
    o118w = O1[O1.no == 118].iloc[1]
    verdict = rule["stage 1 verdict (task rule 1.5)"]
    shw = sh[sh.statistic == "within"].set_index("switch")
    shl = sh[sh.statistic == "linear"].set_index("switch")
    L = []
    A = L.append
    A("# Итерация 13а: закон тенденции нормы прибыли — второй заход, в форме Маркса")
    A("")
    A("* **Предрегистрация** — [`pre_registration_v13a.md`](../pre_registration_v13a.md), закоммичена до расчётов. "
      "Код основного расчёта каждого этапа закоммичен до запуска.")
    A("* **Журнал отступлений** (в конце предрегистрации), каждая запись — отдельным коммитом:")
    A("  * п. 1 — данные, скачанные до предрегистрации (до данных);")
    A("  * п. 2 — реализация этапа 0 (до результата);")
    A("  * п. 3 — ответы на рецензию до расчётов, 24 замечания, правки предрегистрации (до результата);")
    A("  * п. 4 — ошибка кода при первом запуске этапа 1: целочисленный массив весов (до результата; значения исходов "
      "не вычислялись);")
    A("  * п. 5 — охват часов EXIOBASE: самозанятые и крестьяне входят (до расчёта этапа 3);")
    A("  * п. 6 — описательные дополнения: оценки в вариантах λ до 1947 г., окно Kliman 1947–2007 (после результата).")
    A("* **Рецензия до расчётов** — [`results/v13a/review_pre.md`](../results/v13a/review_pre.md). "
      "**После расчётов** — [`results/v13a/review_independent.md`](../results/v13a/review_independent.md); ответы — §11.")
    A("* **Числа** — `results/v13a/`; отчёт генерируется кодом `lts.v13a.report` из этих файлов. "
      "Код — `src/lts/v13a/` (`stage0.py`, `est.py`, `power.py`, `matrix.py`, `stage1.py`, `stage2.py`, `stage3.py`, "
      "`stage4.py`, `report.py`).")
    A("* **Сводная таблица исходов** — [`outcomes_table.md`](outcomes_table.md) (исходы 110–131).")
    A("")
    A("Эмпирика (§1–§7) отделена от интерпретации (§8). Метка «неинформативно» ничего не говорит ни в чью пользу. "
      "Тренды — годовые наклоны логарифма в процентах в год; в скобках — 90% ДИ. «Внутри фаз» — общий наклон по фазам "
      "A 1948–1973, B 1983–2007, C 2010–2019 со своими уровнями фаз.")
    A("")
    # ---------------------------------------------------------------- 0
    A("## 0. Резюме")
    A("")
    rows = [["0", "Данные: часы 1947–2023, рядовые работники (CES), поколения капитала, λ до 1947 г.",
             "сделано; проверки — журнал п. 2", "—"],
            ["1", "Падает ли норма прибыли США внутри фаз в основной форме (историческая оценка, весь деловой капитал, "
                  "V без управляющих, часы)?",
             f"Строение растёт: k*_h {ci_pct(o110.est, o110.ci90_lo, o110.ci90_hi)}, устойчиво. Норма не падает и не "
             f"растёт: r*_h {ci_pct(o111.est, o111.ci90_lo, o111.ci90_hi)}. Моральный износ в кризисных окнах не выше, "
             f"чем в фазах (R = {num(o113.est, 2)}), но это неустойчиво. **Итог по правилу задания — «{verdict}»**",
             "110 «подтверждено»; 111 «неинформативно»; 113 — для правила «неинформативно»"],
            ["2", "Какой выбор построения решает знак у авторов, где норма падает?",
             f"Наш аналог воспроизводит тренд в {int(s2o.k.iloc[0])} из {int(s2o.n.iloc[0])} работ "
             f"(доля {num(s2o.est.iloc[0], 2)}; ниже 0,5 нет — выводы этапа 1 не предварительные). Знак меняет переход "
             "к марксовой мере производительных отраслей (шаг ii моста)", "119"]]
    if O3 is not None:
        r120 = O3[O3.no == 120].iloc[0]
        r121 = O3[O3.no == 121].iloc[0]
        rows.append(["3", "Мировая система в часах (EXIOBASE 1995–2024)",
                     f"r_h {ci_pct(r120.est, r120.ci90_lo, r120.ci90_hi)} {lab(r120.label_full)}; k_h "
                     f"{ci_pct(r121.est, r121.ci90_lo, r121.ci90_hi)} {lab(r121.label_full)}. Часы включают "
                     "самозанятых и крестьян — подтверждение закона из этого не следует", "120–122"])
    else:
        rows.append(["3", "Мировая система в часах (EXIOBASE 1995–2024)", "расчёт не завершён", "120–122"])
    if s4f is not None:
        rows.append(["4", "Длинный ряд Д–Л; ошибка BEA со страхованием",
                     "; ".join(f"исход {int(r.outcome)}: {lab(r.label_before)} → {lab(r.label_after)}"
                               for _, r in s4f.iterrows()), "7, 18"])
    A("")
    L += md_table(rows, ["Этап", "Вопрос", "Результат", "Исходы"])
    A("")
    A("**Главное (эмпирика).**")
    A(f"1. **Внутри фаз стоимостное строение капитала в часах растёт, а норма прибыли — нет.** k*_h растёт на "
      f"{pct(o110.est)} в год, устойчиво во всех слоях проверки. r*_h: {ci_pct(o111.est, o111.ci90_lo, o111.ci90_hi)}. "
      f"Почти весь рост строения перекрыт ростом нормы эксплуатации: e*_h {pct(ident.loc['ln e*_h', 'est'])} в год "
      f"против ln(1 + k*_h) {pct(ident.loc['ln(1 + k*_h)', 'est'])} (тождество ln r = ln e − ln(1 + k)).")
    ex0 = rd("s1_extra_after_results.csv").set_index("item")
    c1, c2 = ex0.loc["111: lambda before 1947 = const"], ex0.loc["111: lambda before 1947 = current"]
    A(f"   * **Оговорка:** результат 111 зависит от оценки поколений до 1947 г. При λ_1947 без поправки r*_h падает на "
      f"{pct(-c1.est)} в год {lab(c1.label_full)}, при текущей оценке старых поколений — на {pct(-c2.est)} "
      f"{lab(c2.label_full)}. Почти вся разница — в фазе A. По правилу задания такая зависимость даёт "
      "«неинформативно».")
    A(f"2. **Фазы разные.** Наклон ln r*_h: A {pct(o111.slope_A)}, B {pct(o111.slope_B)}, C {pct(o111.slope_C)} в год; "
      f"разность B − A {ci_pct(o126.est, o126.ci90_lo, o126.ci90_hi)} «подтверждено». Однородность наклонов "
      f"отвергается (p {'< 0,001' if o111.p_equal_slopes < 0.001 else num(o111.p_equal_slopes, 3)}). Падение нормы "
      "внутри фазы есть только в A (1948–1973).")
    A(f"3. **При постоянной реальной корзине на час норма растёт при обеих оценках:** текущая "
      f"{ci_pct(o114.est, o114.ci90_lo, o114.ci90_hi)}, историческая {ci_pct(o115.est, o115.ci90_lo, o115.ci90_hi)}. "
      "Граница метода задания (114 не падает, 115 падает) не сработала.")
    A(f"4. **Непроизводительный капитал снижает тренд нормы** на {pct(-o117.est)} в год "
      f"(ln(r^(В)/r^(П)) {ci_pct(o117.est, o117.ci90_lo, o117.ci90_hi)}, устойчиво) — посылка Мозли выполняется.")
    A(f"5. **Моральный износ** сосредоточен в годах дна спадов NBER (R = {num(o113.nber_est, 2)}), но в "
      f"структурных кризисных окнах 1974–1982, 2008–2009, 2020–2021 гг. он ниже, чем в фазах "
      f"(R = {num(o113.est, 2)} ({num(o113.ci90_lo, 2)}; {num(o113.ci90_hi, 2)})).")
    A("6. **Мост к литературе:** падающие нормы Шейха (корпорации) и Дюмениля–Леви (частная экономика) на наших "
      "данных воспроизводятся; при переходе к марксовой норме производительных отраслей (вычитается только оплата "
      "производительных работников, знаменатель — их капитал) падение исчезает; знаменатель (В) возвращает тренд к нулю.")
    A("")
    # ---------------------------------------------------------------- 1
    A("## 1. Что изменилось до расчётов")
    A("")
    A("* **Рецензия до расчётов** (журнал п. 3): 24 замечания. Главные правки:")
    A("  * историческая оценка — прочтение задания (близкое к TSSI), а не бесспорная «форма Маркса»; в матрице два "
      "прочтения Маркса — М-ист и М-воспр;")
    A("  * поправлены ячейки матрицы (TSSI, Окисио, Шейх, неоклассика); добавлен столбец «сжатие прибыли»; "
      "неоклассика, Калецки, Д–Л и сжатие прибыли засчитываются по денежному аналогу той же меры;")
    A("  * добавлены исходы 123–131 (текущая оценка той же конструкции, материализованное строение, разности фаз, "
      "управляющие в V, мера TSSI с MELT, r_h / r_$ по США, вековой тренд);")
    A("  * устойчивость — по слоям, а не единогласием по 2 750 вариантам; бутстреп — пар (t, y).")
    A("* **Ожидаемая точность** (`p0_power.csv`, ряды 12б): полуширина ДИ внутри фаз больше Δ = 0,002 для всех "
      "исходов. «Опровергнуто (экв.)» практически недостижимо, и теории с «≈ 0» по этим исходам «за» получить не могут.")
    A("* **Ошибка кода** (журнал п. 4): веса λ в исторической оценке обрезались до нуля из-за целочисленного массива. "
      "Найдена до вычисления исходов; ряды пересчитаны.")
    A("")
    # ---------------------------------------------------------------- 2
    A("## 2. Этап 0: данные")
    A("")
    A("* Часы 1947–2023: исследовательский счёт BEA–BLS (часы всех занятых: медиана отношения к H_EMP JHS 0,981), "
      "2017–2023 гг. — по индексу счёта 1997–2024; шов 1997–2016 — 0,01 п. п. в год.")
    A("* Рядовые работники (CES): доли в часах и в фонде оплаты по 47 отраслям; сцепка рядов по первым трём общим "
      "годам; s_W не обрезается на 1 (журнал п. 2; где s_W > 1 — журнал п. 7).")
    chk0 = rd("s0_checks.csv")
    agg = chk0[chk0.check == "a: aggregate weighted by compensation"].copy()
    agg["year"] = agg.year.astype(int)
    agg = agg.set_index("year").sort_index()
    rat = agg.value / agg.ref
    def ols(y):
        t = np.asarray(y.index, float)
        return float(np.polyfit(t, np.log(y.to_numpy(float)), 1)[0])
    ph_c = agg.loc[2010:2019]
    A(f"* **Сверка s_W с CES 82/57** (предрегистрация, 0.2; сделана после рецензии — журнал п. 7, `s0_checks.csv`), "
      f"{int(agg.index.min())}–{int(agg.index.max())} гг., взвешенно по оплате: наша s_W {num(agg.value.mean(), 3)} "
      f"против отношения CES {num(agg.ref.mean(), 3)} (отношение {num(rat.min(), 3)}–{num(rat.max(), 3)}). Уровень "
      "ниже, потому что зарплата NIPA шире фонда оплаты CES (премии, опционы и т. п.). Наклон за 2010–2019 гг.: "
      f"наша {pct(ols(ph_c.value))}, CES {pct(ols(ph_c.ref))} в год — наша мера V падает быстрее.")
    seam = chk0[chk0.check.str.startswith("c:")].copy()
    seam["year"] = seam.year.astype(int)
    sm_ = seam[seam.check.str.contains(", main")].set_index("year").value
    ss_ = seam[seam.check.str.contains("splice")].set_index("year").value
    b0 = chk0[(chk0.check.str.startswith("b:")) & (chk0.code == "private total")].value
    A(f"* **Шов 1998 г. в s_W** (рецензия Р2): доля рядовых в оплате производительных отраслей 1997 → 1998 гг. "
      f"{num(sm_.loc[1997], 4)} → {num(sm_.loc[1998], 4)} (Δ ln {num(100 * np.log(sm_.loc[1998] / sm_.loc[1997]), 2)} "
      "п.); до 1998 г. зарплата — SIC (6.3B/C), распределённая по кодам, с 1998 г. — NAICS (6.3D). Итог частных "
      f"отраслей совпадает (отношение 1998–2000 гг. {num(b0.min(), 3)}–{num(b0.max(), 3)}); расходится распределение по "
      "кодам. Вариант «сшивка 1998» (журнал п. 7): зарплата 1948–1997 гг. × отношение NAICS / SIC кода за 1998–2000 гг.; "
      f"скачок 1997 → 1998 гг. — {num(100 * np.log(ss_.loc[1998] / ss_.loc[1997]), 2)} п.")
    vc = rd("s0_vintage_check.csv").set_index("year")
    late = vc.loc[1973:].hist_dev.abs()
    A(f"* Поколения капитала FA Detailed 1901–2023: историческая стоимость против табл. 4.3 — отклонение не больше "
      f"{num(100 * vc.hist_dev.abs().max(), 1)}% ({int(vc.hist_dev.abs().idxmax())} г.); с 1973 г. — не больше "
      f"{num(100 * late.max(), 2)}% ({int(late.idxmax())} г.; в журнале п. 2 было «меньше 1%» — неточно).")
    A("* Доля самозанятых по кодам: NIPA 6.7 / 6.8. Ошибка сопоставления подписей 6.7D (с 1998 г. доля обнулялась "
      "у 12 кодов) исправлена после рецензии (Р1, журнал п. 7).")
    A("* λ до 1947 г. — по индексу «часы на доллар ЧВП» Duménil–Lévy (`uslt4x`); варианты (а) λ_1947 и (б) текущая "
      "оценка поколений до 1947 г.")
    A("")
    # ---------------------------------------------------------------- 3
    A("## 3. Этап 1: США, исходы 110–131")
    A("")
    A("### 3.1 Исходы")
    A("")
    L += md_table(outcome_rows(O1, [110, 111, 112, 113, 114, 115, 116, 117, 123, 124, 125, 126, 127, 128, 129, 130, 131]),
                  ["№", "Исход", "Оценка (90% ДИ)", "Метка", "Устойчив"])
    A("")
    A(f"* **112:** Δ ln r*_h 1973→1983 и 2007→2010 — {o112.events.replace(';', ' и ').replace('.', ',').replace('-', '−')}: "
      "между фазами норма не восстанавливается, а ниже, чем в конце предыдущей фазы (описательно, 2 события).")
    A(f"* **118:** за 1948–2023 гг. компонента Мозли ln(1 − U / S_$) даёт {num(o118.est, 2)} "
      f"({num(o118.ci90_lo, 2)}; {num(o118.ci90_hi, 2)}) разрыва трендов r НФК и r*_h (наклон разрыва "
      f"{ci_pct(o118.gap_slope, o118.gap_ci90_lo, o118.gap_ci90_hi)}). Внутри фаз ДИ наклона разрыва включает 0 — "
      "доля не приводится (правило П23.11).")
    A(f"* **131** (описательно): вековой тренд r*_h за 1948–2023 гг. — {ci_pct(o131.est, o131.ci90_lo, o131.ci90_hi)}, "
      f"{lab(o131.label_full)}.")
    A("")
    A("### 3.2 Итоговое правило этапа 1 и описательные чтения")
    A("")
    A(f"* Метки для правила (устойчивые или «неинформативно»): {rule['labels for the rule (robust or ' + chr(39) + 'неинформативно' + chr(39) + ')'] if False else rule.iloc[0]}.")
    A(f"* **Итог по правилу задания 1.5: «{verdict}».** 110 — «подтверждено»; 111 — «неинформативно»; 113 — основная "
      f"метка {lab(o113.label_full)}, но в варианте NBER (годы дна спадов) — {lab(o113.nber_label)}, поэтому для правила "
      "«неинформативно».")
    A(f"* Чтение 110 × 111: «{rule['reading 110 x 111 (descriptive)']}». Чтение 125 × 111: "
      f"«{rule['reading 125 x 111 (descriptive)']}».")
    A(f"* Чтение 114 × 115: «{rule['reading 114 x 115 (descriptive, main labels)']}» (основные метки; 115 неустойчив по "
      "λ до 1947 г.).")
    A("")
    A("### 3.3 Устойчивость")
    A("")
    rows = []
    for no in (110, 111, 114, 115, 117, 123, 124, 125, 129, 130, 116, 126, 127, 128, 113):
        r = o(no)
        rows.append([no, lab(r.label_full), (r.nine if isinstance(r.nine, str) else "—"),
                     (r.lambda_labels if isinstance(r.lambda_labels, str) else "—"),
                     num(r.grid_share_main, 3) if np.isfinite(r.grid_share_main) else "—",
                     (r.nber_label if isinstance(r.nber_label, str) else "—"),
                     (r.spec_labels if isinstance(r.spec_labels, str) else "—"),
                     "да" if bool(r.robust) else "нет"])
    L += md_table(rows, ["№", "Основная метка", "9 сочетаний", "λ до 1947 г. (const; current)", "Доля основной метки в "
                         "сетке границ", "NBER", "Спецификационный слой (без самозанятых; ω* без трансфертов; без ИС)",
                         "Устойчив"])
    A("")
    ex = rd("s1_extra_after_results.csv").set_index("item")
    LV = (("index (main)", "основной (индекс `uslt4x`)"), ("const", "λ_1947 без поправки"),
          ("current", "поколения до 1947 г. по текущей стоимости"))
    A("**Чувствительность к λ до 1947 г.** (журнал п. 6 и п. 7; правило задания применяет этот слой к 110, 111, 115; "
      "для 117, 125–127 — описательно, после рецензии Р3):")
    A("")
    rows = []
    for no in (110, 111, 115, 117, 125, 126, 127):
        row = [no]
        for v, _ in LV:
            r = ex.loc[f"{no}: lambda before 1947 = {v}"]
            sl = "" if no in (126, 127) else f"; A {pct(r.slope_A)}, B {pct(r.slope_B)}"
            row.append(f"{ci_pct(r.est, r.ci90_lo, r.ci90_hi)} {lab(r.label_full)}{sl}")
        rows.append(row)
    for rd_ in ("110 x 111", "125 x 111"):
        rows.append([f"чтение {rd_.replace(' x ', ' × ')}"] + [lab(ex.loc[f"reading {rd_}: lambda before 1947 = {v} "
                                                                       "(main labels of the variant)", "label_full"])
                                                           for v, _ in LV])
    L += md_table(rows, ["№"] + [n for _, n in LV])
    A("")
    A("* Чтения по вариантам — по основным меткам каждого варианта (слои устойчивости для вариантов не считались). "
      "Чтение основного варианта в §3.2 — по меткам для правила.")
    alt111 = [ex.loc[f"111: lambda before 1947 = {v}", "label_full"] for v in ("const", "current")]
    both_fall = all(x == "подтверждено" for x in alt111)
    A("* Основной вариант даёт поколениям 1901–1946 гг. больший вес в часах (до 1947 г. часов на доллар было больше). "
      "Варианты меняют наклоны и фазы A, и фазы B (таблица). "
      + (f"При обоих альтернативных вариантах историческая норма внутри фаз падает (111 «подтверждено»), а текущая "
         f"(123) — {lab(o123.label_full)}: знак исторической меры зависит от допущения о λ до 1947 г. " if both_fall
         else f"Метки 111 в альтернативных вариантах: {', '.join(lab(x) for x in alt111)}. ")
      + "По правилу задания 111 и 115 при зависимости от варианта получают для итога «неинформативно».")
    A("")
    A("### 3.4 Наклоны по фазам и тождество")
    A("")
    rows = []
    for nm in ("ln r*_h", "ln e*_h", "ln(1 + k*_h)", "ln k*_h", "ln(C^(В) / L)", "ln(1 + e*_h)", "ln k*_h at (P)"):
        r = ident.loc[nm]
        rows.append([nm.replace(" at (P)", " при (П)"), ci_pct(r.est, r.ci90_lo, r.ci90_hi), pct(r.slope_A),
                     pct(r.slope_B), pct(r.slope_C)])
    L += md_table(rows, ["Ряд", "Общий наклон внутри фаз", "A 1948–1973", "B 1983–2007", "C 2010–2019"])
    A("")
    A("* Разложение точное: ln r*_h = ln e*_h − ln(1 + k*_h); ln k*_h = ln(C^(В)/L) + ln(1 + e*_h).")
    A(f"* В фазе C строение падает ({pct(ident.loc['ln k*_h', 'slope_C'])} в год), норма растёт.")
    A(f"* 116: β_A′ = {pct(desc.loc['116 slopes: beta_A' + chr(39) + ' and beta_B', 'slope_A'])}, "
      f"β_B = {pct(desc.loc['116 slopes: beta_A' + chr(39) + ' and beta_B', 'slope_B'])} в год. Ожидавшегося по "
      "литературе о сжатии прибыли падения e*_h в 1964–1973 гг. в этой мере нет. Почему (рецензия Р5; журнал п. 7, "
      "описательно):")
    def xv(item):
        return ex.loc[item]
    mg_e = xv("e: V variant = managers in V and L (who=all)")
    A(f"  * **управляющие (допущение С7).** Если управляющие входят в V и L, наклон ln e внутри фаз "
      f"{ci_pct(mg_e.est, mg_e.ci90_lo, mg_e.ci90_hi)} (A {pct(mg_e.slope_A)}, B {pct(mg_e.slope_B)}, "
      f"C {pct(mg_e.slope_C)}) против {pct(ident.loc['ln e*_h', 'est'])} (A {pct(ident.loc['ln e*_h', 'slope_A'])}, "
      f"B {pct(ident.loc['ln e*_h', 'slope_B'])}) в основной мере; 128: β_A′ {pct(o128.slope_A)}, β_B {pct(o128.slope_B)}. "
      "Рост e*_h в фазе B в основном — рост доли оплаты управляющих;")
    om = xv("A' 1964-1973: OLS slope of ln omega*")
    shp = xv("A' 1964-1973: OLS slope of ln PNS share of productive compensation")
    oa = desc.loc["116 [omega_alt]"]
    A(f"  * **построение V через личный доход (С3).** В 1964–1973 гг. доля рядовых в оплате производительных отраслей "
      f"меняется на {pct(shp.est)} в год, а ω* — на {pct(om.est)}: V падает относительно потребления за счёт отношения "
      f"оплаты производительных отраслей к личному доходу. Вариант ω* без трансфертов: β_A′ {pct(oa.slope_A)}. "
      "Такое V не совпадает с долей заработной платы в стоимости, о которой литература сжатия прибыли;")
    cap = xv("116: V variant = sW capped at 1")
    sp_ = xv("116: V variant = splice 1998")
    A(f"  * **s_W > 1.** С обрезкой s_W на 1: β_A′ {pct(cap.slope_A)}, 116 = {ci_pct(cap.est, cap.ci90_lo, cap.ci90_hi)} "
      f"{lab(cap.label_full)}; 110 и 111 — {pct(xv('110: V variant = sW capped at 1').est)} и "
      f"{pct(xv('111: V variant = sW capped at 1').est)};")
    A(f"  * **шов 1998 г.** Со сшивкой зарплаты (§2): наклон ln e*_h в фазе B {pct(xv('e: V variant = splice 1998').slope_B)} "
      f"(основной {pct(ident.loc['ln e*_h', 'slope_B'])}); 116 = {ci_pct(sp_.est, sp_.ci90_lo, sp_.ci90_hi)} "
      f"{lab(sp_.label_full)}; 111 = {ci_pct(xv('111: V variant = splice 1998').est, xv('111: V variant = splice 1998').ci90_lo, xv('111: V variant = splice 1998').ci90_hi)} "
      f"{lab(xv('111: V variant = splice 1998').label_full)}.")
    A("")
    A("### 3.5 Разложение отличий от 12б (Шепли)")
    A("")
    rows = []
    for swt, nm in (("cap", "данные капитала: FA 2.1 → FA Detailed"), ("lab", "лица → часы"),
                    ("who", "все наёмные → рядовые"), ("dep", "валовая → чистая λ"),
                    ("val", "текущая → историческая оценка"), ("den", "(П) → (В)")):
        r = shw.loc[swt]
        rows.append([nm, ci_pct(r.shapley, r.ci90_lo, r.ci90_hi), pct(r.fixed_order_step), pct(shl.loc[swt, "shapley"])])
    rows.append(["**итого 13а − 12б**", pct(shw.loc["total (13a - 12b)", "shapley"]), "",
                 pct(shl.loc["total (13a - 12b)", "shapley"])])
    L += md_table(rows, ["Переключатель", "Вклад Шепли в наклон ln r внутри фаз (90% ДИ)",
                         "Шаг в фиксированном порядке", "Вклад Шепли в линейный тренд 1948–2023"])
    A("")
    chk = rd("s1_check_12b.csv")
    A(f"* Проверки на концах: все переключатели в положении 12б воспроизводят r_h 12б (max |Δ| < 10⁻¹⁶: "
      f"{num(chk.absdiff_12b_end.max() * 1e17, 1)}·10⁻¹⁷) — это независимая проверка. В положении 13а ряд "
      f"конфигурации 111111 совпадает с r*_h (max |Δ| = {num(chk.absdiff_13a_end.max(), 1)}), но оба считаются одной "
      "функцией (`year_values`), так что это проверка сборки, а не независимая (рецензия Р14).")
    A("* Больше всего наклон внутри фаз снижают переход к часам и знаменатель (В); переход к рядовым повышает его. "
      "Историческая оценка внутри фаз почти не меняет наклон (ДИ включает 0), а в линейном тренде за 75 лет "
      "повышает его.")
    A("")
    A("### 3.6 Описательно")
    A("")
    rows = []
    for it, nm in (("within phases: money, historical: ln r*_$", "деньги, историческая оценка: ln r*_$"),
                   ("within phases: money, current: ln r_$", "деньги, текущая оценка: ln r_$"),
                   ("within phases: money, historical: ln k*_$", "деньги, историческая: ln k*_$"),
                   ("within phases: energy (current): ln r_E", "энергия (текущая): ln r_E"),
                   ("within phases: manufacturing: ln r*", "обработка, основная форма: ln r*"),
                   ("within phases: manufacturing: ln k*", "обработка: ln k*"),
                   ("within phases: 110 at (P): ln k*_h(P)", "110 при (П): ln k*_h^(П)")):
        r = desc.loc[it]
        rows.append([nm, ci_pct(r.est, r.ci90_lo, r.ci90_hi), lab(r.label_full)])
    L += md_table(rows, ["Ряд (внутри фаз)", "Наклон", "Метка (описательно)"])
    A("")
    A("* Варианты (все часы; ρ_h = 1; доли до 1964 г. с поправкой; сельское хозяйство по частному сектору; CES без "
      "сцепки; валовая λ): метки 110, 111, 115, 117 те же, что в основном расчёте (`s1_descriptive.csv`).")
    r_h = desc.loc["113 R without hedonic assets"]
    rg = desc.loc["113 regression m_t = a_group + gamma 1[NBER trough year]"]
    A(f"* 113: без активов с гедоническими ценами R = {num(r_h.est, 2)} ({num(r_h.ci90_lo, 2)}; {num(r_h.ci90_hi, 2)}); "
      f"в годы дна спадов NBER темп морального износа выше на {pct(rg.est)} (γ, {num(100 * rg.ci90_lo, 2)}; "
      f"{num(100 * rg.ci90_hi, 2)}).")
    MD = rd("s1_moral_depreciation.csv").set_index("year")
    def grp_mean(col, wins):
        msk = np.any([(MD.index >= a) & (MD.index <= b) for a, b in wins], axis=0)
        return float(MD.loc[msk, col].mean())
    nb_y = [y for y in NBER_TROUGH if y in MD.index]
    rows = []
    for nm, wins in (("внутри фаз A, B, C", list(PH1.values())), ("структурные кризисные окна", CR1),
                     ("годы дна спадов NBER", [(y, y) for y in nb_y])):
        rows.append([nm, pct(grp_mean("m", wins)), pct(grp_mean("m_productivity", wins)),
                     pct(grp_mean("m_relative_price", wins))])
    for y in nb_y:
        rows.append([f"{y} (дно спада)", pct(MD.m[y]), pct(MD.m_productivity[y]), pct(MD.m_relative_price[y])])
    A("")
    A("**Разложение темпа морального износа m** (предрегистрация, 1.6; средние за годы, процент капитала в год; "
      "компоненты — середина отрезка, остаток — взаимодействие):")
    A("")
    L += md_table(rows, ["Годы", "m", "компонента производительности", "компонента цены активов"])
    A("")
    kp = sum(MD.m_productivity[y] > MD.m_relative_price[y] for y in nb_y)
    A(f"* В {kp} из {len(nb_y)} годов дна спадов компонента производительности больше компоненты цены активов: часы "
      "падают быстрее выпуска, и измеренная λ падает. Значит, мера ловит и циклические колебания измеренной "
      "производительности, а не только обесценивание по гл. 15 (рецензия Р9).")
    A("")
    # ---------------------------------------------------------------- 4
    A("## 4. Этап 2: сверка с работами, где марксова норма падает")
    A("")
    rows = []
    for _, r in s2c.iterrows():
        rows.append([r.work, f"{int(r.y0)}–{int(r.y1)}", num(r.corr_levels, 2), ci_pct(r.trend_ours, r.trend_ours_lo,
                     r.trend_ours_hi), ci_pct(r.trend_author, r.trend_author_lo, r.trend_author_hi),
                     ci_pct(r["diff"], r.diff_lo, r.diff_hi), "да" if r.reproduced else "нет"])
    pm = s2p[(s2p.work == "Paitaridis & Tsoulfidis (2012)") & (s2p.variable == "R") & (s2p.y0 == 1964) & (s2p.y1 == 2007)].iloc[0]
    rows.append(["Paitaridis & Tsoulfidis (2012): R (табл. 1)", "1964–2007", "—",
                 ci_pct(pm.trend_ours, pm.trend_ours_lo, pm.trend_ours_hi), pct(pm.author) + " (число автора)",
                 ci_pct(pm["diff"], pm.diff_lo, pm.diff_hi), "да" if pm.reproduced else "нет"])
    L += md_table(rows, ["Работа", "Окно", "Корреляция уровней", "Наш аналог", "Автор", "Разность (наш − автор)",
                         "TOST ± 0,2 п. п."])
    A("")
    A(f"* **Исход 119:** {int(s2o.k.iloc[0])} из {int(s2o.n.iloc[0])}, доля {num(s2o.est.iloc[0], 2)} "
      f"(Уилсон 90%: {num(s2o.ci90_lo.iloc[0], 2)}; {num(s2o.ci90_hi.iloc[0], 2)}). Порог задания 0,5 пройден: выводы "
      "этапа 1 предварительными не считаются.")
    kl = rd("s1_extra_after_results.csv").set_index("item").loc["Kliman (2009): PIH / HCFA, 1947-2007"]
    A(f"* Kliman (2009) на послевоенном окне 1947–2007 гг. (журнал п. 6, описательно): ряд автора "
      f"{ci_pct(kl.est, kl.ci90_lo, kl.ci90_hi)} {lab(kl.label_full)}, наш аналог {ci_pct(kl.ours, kl.ours_lo, kl.ours_hi)}; "
      f"разность {ci_pct(kl['diff'], kl.diff_lo, kl.diff_hi)} — {'в пределах' if kl.reproduced else 'вне'} ± 0,2 п. п. "
      "Окно выбрано после результата. На предрегистрированном окне (с 1929 г., начало — Великая депрессия) тренд "
      "автора не падает; 119 считается по нему.")
    bx = rd("s2_descriptive.csv").set_index("item")
    fl = dict(zip(str(s2o.works.iloc[0]).split(";"), str(s2o.reproduced.iloc[0]).split(";")))
    nb_ = {w: f for w, f in fl.items() if not w.startswith("Basu")}
    k_nb, n_nb = sum(f == "True" for f in nb_.values()), len(nb_)
    bxr = bx.loc["Basu et al.: XR LimitCountries vs 12b PWT xr balanced"]
    A(f"* Basu et al.: сравнение с мировой нормой 12б (PWT), а не с аналогом по EPWT — источники разные. По ППС TOST не "
      f"выполнен; по рыночным курсам (XR) — выполнен: разность {ci_pct(bxr['diff'], bxr.diff_lo, bxr.diff_hi)}, "
      f"корреляция уровней {num(bxr.corr_levels, 2)} (рецензия Р12). Расхождение — в агрегировании по ППС. Без Basu "
      f"доля 119 была бы {k_nb} из {n_nb}.")
    pe = s2p[(s2p.work == "Paitaridis & Tsoulfidis (2012)") & (s2p.y0 == 1964) & (s2p.y1 == 2007)].set_index("variable")
    A(f"* Paitaridis & Tsoulfidis: по периодам табл. 1 TOST не выполнен ни для одной величины (`s2_compare_pt.csv`); "
      "наш аналог MVA — приближение (без детальной классификации ST). Способ «годовых темпов» у авторов неизвестен. "
      f"Наш темп R по концам 1964–2007 гг. — {pct(pe.loc['R', 'endpoint_growth_ours'])} против {pct(pe.loc['R', 'author'])} у "
      f"авторов; для C/v точечная разность {num(100 * pe.loc['C/v', 'diff'], 2)} п. п., TOST не выполнен из-за ширины ДИ.")
    A("")
    A("**Мост** (шаги в фиксированном порядке, каждый — одна перемена; тренд ln, правило одного ряда; все шаги, "
      "включая i, — на одном окне; подшаги ii-а…ii-г — после рецензии Р6, журнал п. 7):")
    A("")
    rows = []
    signs = {}
    for w, g in s2b.groupby("work", sort=False):
        g = g.reset_index(drop=True)
        rows.append([w, f"{int(g.y0.iloc[0])}–{int(g.y1.iloc[0])}", "", ""])
        for _, r in g.iterrows():
            rows.append(["", r.step, ci_pct(r.est, r.ci90_lo, r.ci90_hi), lab(r.label)])
        e0 = np.sign(g.est.iloc[0])
        flip = g[np.sign(g.est) != e0]
        signs[w] = (flip.step.iloc[0].split(":")[0] if len(flip) else None, g)
    L += md_table(rows, ["Работа", "Шаг", "Тренд", "Метка"])
    A("")
    for w, (st, g) in signs.items():
        A(f"* {w}: шаг i {ci_pct(g.est.iloc[0], g.ci90_lo.iloc[0], g.ci90_hi.iloc[0])} {lab(g.label.iloc[0])}; "
          + (f"знак точечной оценки впервые меняется на шаге {st}." if st else "знак точечной оценки на шагах не меняется."))
    A("* Подшаги ii-а…ii-г разделяют то, что шаг ii меняет сразу: охват добавленной стоимости (частный сектор без "
      "государства и недвижимости против NDP с ними), оплату непроизводительных отраслей (издержки или S), меру V "
      "(выплаченная оплата или корзина ω·Σc) и охват капитала ((В) или (П)). Порядок фиксированный: вклад шага зависит "
      "от порядка. Шаг v («деньги → часы») тоже не только смена единиц: из S выпадает добавленная стоимость "
      "непроизводительных отраслей и государства.")
    rm = bx.loc["bridge step ii vs r_m of iteration 9b (main), 1948-2023"]
    A(f"* **Шаг ii против r_m итерации 9б** (предрегистрация «к r_m 9б»; 1948–2023 гг.): наш "
      f"{ci_pct(rm.trend_ours, rm.trend_ours_lo, rm.trend_ours_hi)}, 9б {ci_pct(rm.trend_author, rm.trend_author_lo, rm.trend_author_hi)}, "
      f"разность {ci_pct(rm['diff'], rm.diff_lo, rm.diff_hi)}; {'в пределах' if rm.reproduced else 'вне'} ± 0,2 п. п.; "
      f"корреляция уровней {num(rm.corr_levels, 2)}.")
    A("")
    A("**Mohun (2014)** — только величины, названные в тексте (без TOST):")
    A("")
    MOH = {"supervisory wages / money value added": "оплата надзора / денежная добавленная стоимость",
           "unproductive working-class wages / money value added": "оплата непроизводительного рабочего класса / "
                                                                   "денежная добавленная стоимость",
           "productive working class / total employment": "производительный рабочий класс / вся занятость",
           "productive wages / total wages": "оплата производительного труда / вся оплата"}
    L += md_table([[MOH.get(r.variable, r.variable), int(r.year), num(r.mohun_text, 3), num(r.ours, 3)]
                   for _, r in s2m.iterrows()],
                  ["Величина", "Год", "Mohun (текст)", "Наш аналог"])
    A("")
    mt = s2m.set_index(["variable", "year"])
    un = "unproductive working-class wages / money value added"
    sv = "supervisory wages / money value added"
    yrs = sorted(int(y) for v, y in mt.index if v == un)
    A(f"* Доля оплаты непроизводительного рабочего класса у Mohun {num(mt.loc[(un, yrs[0]), 'mohun_text'], 3)} "
      f"({yrs[0]}) и {num(mt.loc[(un, yrs[-1]), 'mohun_text'], 3)} ({yrs[-1]}), у нас — {num(mt.loc[(un, yrs[0]), 'ours'], 3)} "
      f"и {num(mt.loc[(un, yrs[-1]), 'ours'], 3)}: у нас она растёт. Это та оплата, которая на шаге ii моста уходит в S. "
      f"Доля оплаты надзора у нас ниже ({num(mt.loc[(sv, yrs[0]), 'ours'], 3)} против {num(mt.loc[(sv, yrs[0]), 'mohun_text'], 3)} "
      f"в {yrs[0]} г.). Граница «рядовые / надзор» (CES s_W по 47 отраслям) независимо не проверена; s_W > 1 и шов "
      "1998 г. (§2, §3.4) — её известные слабости.")
    A("")
    # ---------------------------------------------------------------- 5
    A("## 5. Этап 3: мировая система в часах (EXIOBASE 3.11.2, 1995–2024)")
    A("")
    if O3 is not None:
        L += md_table(outcome_rows(O3, [120, 121, 122]), ["№", "Исход", "Оценка (90% ДИ)", "Метка", "Устойчив (9 сочетаний)"])
        A("")
        A("* **Оговорка (журнал п. 5):** часы EXIOBASE включают самозанятых и крестьян, а V — только оплату наёмных. "
          "Сокращение крестьянской занятости механически снижает мировые e и r_h, поэтому 120 и 121 не толкуются как "
          "подтверждение закона.")
        if s3d is not None:
            d3 = s3d.set_index("item")
            VAR = {"main": "основной", "lowmed": "L — низкая и средняя квалификация", "PQ_out": "образование и "
                   "здравоохранение вне производительных"}
            COL = {"r_h": "ln r_h", "k_h": "ln k_h", "r_h_nore": "ln r_h без капитала недвижимости",
                   "k_h_nore": "ln k_h без капитала недвижимости", "r_h_gfcf": "ln r_h, капитал ∝ структуре ВНОК",
                   "k_h_gfcf": "ln k_h, капитал ∝ структуре ВНОК", "r_h_natp": "ln r_h, масштаб PWT в национальных ценах",
                   "k_h_natp": "ln k_h, масштаб PWT в национальных ценах", "r_d": "ln r_$ (деньги)",
                   "r_d_natp": "ln r_$, масштаб PWT в национальных ценах"}
            rows = []
            for lamk, lamn in (("net", "чистая λ"), ("gross", "валовая λ")):
                for var, vn in VAR.items():
                    for col, cn in COL.items():
                        it = f"lambda={lamk}, variant={var}: within phases ln {col}"
                        if it in d3.index and (var == "main" or col in ("r_h", "k_h")):
                            r = d3.loc[it]
                            rows.append([f"{lamn}; {vn}", cn, ci_pct(r.est, r.ci90_lo, r.ci90_hi), lab(r.label_full)])
            L += [""] + md_table(rows, ["Вариант", "Ряд (внутри фаз)", "Наклон", "Метка (описательно)"])
            A("")
            A("* Варианты со столбцами «без недвижимости», «ВНОК» и «национальные цены» — описательно; варианты "
              "«национальные цены» добавлены после рецензии (Р8, журнал п. 7). Капитал региона = (cn / cgdpo) × ДС "
              "EXIOBASE; cn и cgdpo в PWT — в текущих ППС, а ДС EXIOBASE — в рыночных ценах. Вариант (cn·pl_n) / "
              "(cgdpo·pl_gdpo) переводит отношение K/Y в национальные цены.")
            A("")
            A("**Наклоны по фазам** (свои наклоны фаз, основной вариант; после рецензии Р8):")
            A("")
            rows = []
            for nm in ("ln e_h", "ln k_h", "ln r_h", "ln r_$", "ln(r_h / r_$)", "ln k_h, national prices",
                       "ln r_h, national prices", "agricultural share of L"):
                it = f"phase slopes: {nm}"
                if it in d3.index:
                    r = d3.loc[it]
                    nm_ru = (nm.replace("national prices", "национальные цены")
                             .replace("agricultural share of L", "доля часов сельского хозяйства в L (п. в год)"))
                    if nm.startswith("agricultural"):
                        rows.append([nm_ru, num(100 * r.slope_A, 3), num(100 * r.slope_B, 3)])
                    else:
                        rows.append([nm_ru, pct(r.slope_A), pct(r.slope_B)])
            L += md_table(rows, ["Ряд", "1995–2007", "2010–2019"])
            A("")
            ag = [d3.loc[f"mean agricultural share of L {a}-{b}", "est"] for a, b in ((1995, 2007), (2010, 2019))
                  if f"mean agricultural share of L {a}-{b}" in d3.index]
            if len(ag) == 2:
                A(f"* Доля часов сельского хозяйства (EXIOBASE, отрасли 0–18) в L: в среднем {num(ag[0], 3)} в 1995–2007 гг. "
                  f"и {num(ag[1], 3)} в 2010–2019 гг.")
            r = d3.loc["lambda=net, variant=main: within phases ln r_d"]
            A(f"* **Часы против денег.** Денежная норма на тех же физических величинах внутри фаз "
              f"{ci_pct(r.est, r.ci90_lo, r.ci90_hi)} {lab(r.label_full)}; падение есть только в часах (122). Толкование — "
              "§8.")
            jump = [f"{c} {num(100 * d3.loc[f'd ln {c} 2009->2010', 'est'], 1)}" for c in ("e_h", "r_h", "V_h", "S_d")
                    if f"d ln {c} 2009->2010" in d3.index]
            if jump:
                top = [c for c in ("e_h", "r_h", "V_h", "S_d") if f"d ln {c} 2009->2010" in d3.index
                       and int(d3.loc[f"d ln {c} 2009->2010", "rank_abs"]) == 1]
                A(f"* **Разрыв 2009→2010** (Δ ln × 100): {'; '.join(jump)}. Крупнейшее годовое изменение ряда — у "
                  f"{', '.join(top) if top else 'ни одного из них'}. В Changelog EXIOBASE 3.10.1/3.11.2 сказано об отдельной обработке лет до 2010 г. и об "
                  "использовании FIGARO; связь со скачком не проверена. Разрыв приходится на границу фаз, и уровни фаз "
                  "его поглощают; один тренд 1995–2019 гг. и сверка с WIOD 2000–2014 гг. его пересекают.")
            r = d3.loc["single trend ln r_h 1995-2019 without 2008-2009 (gaps closed)"]
            A(f"* Один тренд 1995–2019 гг. без 2008–2009 гг.: {ci_pct(r.est, r.ci90_lo, r.ci90_hi)} {lab(r.label_full)}.")
            a = d3.loc["WIOD check 2000-2014: EXIOBASE ln r_h trend"]
            b = d3.loc["WIOD check 2000-2014: WIOD (12b) ln r_h trend"]
            A(f"* Сверка с WIOD (12б, исход 100), 2000–2014 гг.: EXIOBASE {ci_pct(a.est, a.ci90_lo, a.ci90_hi)} "
              f"{lab(a.label_full)}, WIOD {ci_pct(b.est, b.ci90_lo, b.ci90_hi)} {lab(b.label_full)} — знак точечной оценки тот же.")
    else:
        A("Расчёт не завершён.")
    A("")
    # ---------------------------------------------------------------- 6
    A("## 6. Этап 4: длинный ряд и ошибка BEA")
    A("")
    if s4d is not None:
        d4 = s4d.set_index("item")
        r = d4.iloc[0]
        A(f"* **Duménil–Lévy 1869–2014** (описательно): общий наклон ln r внутри фаз литературы "
          f"{ci_pct(r.est, r.ci90_lo, r.ci90_hi)}; без 1869–1889 гг. (до 1889 г. часть рядов — средние за десятилетия) "
          f"{ci_pct(d4.iloc[1].est, d4.iloc[1].ci90_lo, d4.iloc[1].ci90_hi)}.")
        for it in d4.index[2:5]:
            A(f"  * {it.replace('recovery d ln r', 'восстановление Δ ln r')}: {num(d4.loc[it, 'est'], 3)}.")
    if s4f is not None:
        A("* **Ошибка BEA (страхование)** — `lts.bea` с отношением самозанятых к наёмным по секции K для отраслей "
          "64–66:")
        A("")
        L += md_table([[int(r.outcome), r["name"], f"{num(r.est_before, 4)}", lab(r.label_before), f"{num(r.est_after, 4)}",
                        lab(r.label_after), "да" if r.label_changes else "нет"] for _, r in s4f.iterrows()],
                      ["№", "Исход", "Было", "Метка", "Стало", "Метка", "Метка меняется"])
        ub = rd("s4_b0_usa_bea.csv")
        if ub is not None:
            def g_(sub, h, ver):
                r = ub[(ub.subset == sub) & (ub.h == h) & (ub.version == ver)].iloc[0]
                return r
            parts = []
            for sub, h in (("all", 5), ("all", 1), ("core", 1)):
                a, b = g_(sub, h, "before"), g_(sub, h, "after")
                parts.append(f"{sub}, h = {h}: R² труда {num(a.labour, 3)} → {num(b.labour, 3)}, информативно "
                             f"{'да' if a.informative_vs_phys else 'нет'} → {'да' if b.informative_vs_phys else 'нет'}")
            A("")
            A(f"* Метки исходов не меняются, но сама экономика USA_BEA меняется сильно (`s4_b0_usa_bea.csv`; рецензия "
              f"Р10): {'; '.join(parts)}.")
            A("* «Закрыто» относится только к первичным исходам 7 и 18. Вторичные результаты итераций 1–3, 8, 10 на "
              "экономиках `lts.bea` не пересчитывались.")
    else:
        A("* Пересчёт исходов 7 и 18 не завершён.")
    A("")
    # ---------------------------------------------------------------- 7
    A("## 7. Матрица «теория × исход»: попарные выводы")
    A("")
    A("Зачёт — по правилу предрегистрации (журнал п. 3). Теория с пометкой [$] засчитана по денежному аналогу той же "
      "меры (`s1_money_outcomes.csv`), остальные — по мере в часах. Метки — основного расчёта; неустойчивые помечены.")
    A("")
    M1 = rd("s1_matrix.csv")
    def tag(g, cond):
        sub = g[cond]
        return ", ".join(t + (" [$]" if b == "деньги [$]" else "") for t, b in zip(sub.theory, sub.basis)) or "—"
    rows = []
    allm = M1.copy()
    if s3m is not None:
        allm = pd.concat([allm, s3m], ignore_index=True)
    for no, g in allm.groupby("no"):
        no = int(no)
        if no in set(O1.no):
            rr = o(no)
            lb = lab(rr.label_full) + ("" if bool(rr.get("robust", True)) else " (неустойчиво)")
        else:
            rr = O3[O3.no == no].iloc[0]
            lb = lab(rr.label_full) + ("" if bool(rr.get("robust", True)) else " (неустойчиво)")
        za = tag(g, g.verdict == "за")
        if no in (120, 121) and za != "—":
            za += " (оговорка п. 5: не толкуется как подтверждение)"
        rows.append([no, NAMES.get(no, ""), lb, za, tag(g, g.verdict == "против"),
                     tag(g, (g.verdict != "за") & (g.verdict != "против") & (g.prediction != "молчит")),
                     tag(g, g.prediction == "молчит")])
    L += md_table(rows, ["№", "Исход", "Метка (часы)", "За", "Против", "Предсказывали, но нейтрально", "Молчат"])
    A("")
    A("**Денежные аналоги** (по ним засчитаны теории с [$]; `s1_money_outcomes.csv`):")
    A("")
    rows = []
    for no, r in mon.iterrows():
        rows.append([int(no), NAMES.get(int(no), ""), ci_pct(r.est, r.ci90_lo, r.ci90_hi), lab(r.label_full)])
    L += md_table(rows, ["№", "Исход", "Денежный аналог", "Метка"])
    A("")
    A("* Теории с «≈ 0» (Неокл.-К; М для 130; Шейх для 130) получить «за» не могли: полуширина ДИ больше Δ — для "
      "исходов из `p0_power.csv` по ожидаемой точности, для 130 — по наблюдаемой ширине ДИ.")
    A("* 130 — не чистый тест «цена против стоимости»: S_$ = NDP_$ − V_$ включает добавленную стоимость "
      "непроизводительных отраслей и государства, а в часах их труд стоимости не создаёт. 130 смешивает отклонение цен "
      "от стоимостей с изменением доли непроизводительного сектора (рецензия Р14).")
    A("")
    # ---------------------------------------------------------------- 8
    A("## 8. Интерпретация")
    A("")
    A("Это толкование, а не данные.")
    A("")
    A("1. **Закон в форме задания не подтверждён и не опровергнут.** Тенденция строения есть, но внутри фаз её "
      "перекрывает рост нормы эксплуатации — то, что Маркс называет противодействующей причиной I. "
      "Правило задания даёт «неразличимо».")
    A("2. **«Пила» в данных другая, чем ожидалось.** Норма падает внутри фазы A и растёт внутри B и C, а между фазами "
      "не восстанавливается, а проседает (112). Это ближе к прочтениям Д–Л и Шейха (разные знаки по фазам), чем к "
      "пилообразной схеме «падение внутри, восстановление через кризис».")
    A("3. **Способ оценки не решает дело.** При текущей и исторической оценке внутри фаз норма одинаково без тренда "
      "(111, 123), строение одинаково растёт (110, 124); при постоянной реальной корзине норма растёт при обеих "
      "(114, 115). Граница метода задания не сработала.")
    A("4. **Что решает знак у авторов.** Падающие ряды корпораций и частной экономики (Шейх; Д–Л; Kliman на "
      "послевоенном окне) на наших данных воспроизводятся. Знак меняет шаг ii моста. Он сразу переводит оплату "
      "непроизводительного труда из издержек в прибавочную стоимость и сужает капитал до производительных "
      "отраслей. Какая из двух частей важнее, мост не разделяет. Но по 118 доля оплаты непроизводительного труда "
      "в S_$ растёт, и компонента Мозли объясняет около двух третей разрыва трендов r НФК и r*_h. Знаменатель (В) "
      "возвращает норму лишь к нулю. Значит, знак зависит от того, где проводится граница производительного "
      "труда и куда относится оплата непроизводительного.")
    A("5. **Моральный износ** рывками есть, но в годах спадов промышленного цикла, а не в структурных кризисных окнах. "
      "Это совместимо с гл. 15 в прочтении «кризисы промышленного цикла» и плохо — с прочтением через структурные фазы.")
    if O3 is not None:
        A("6. **Мир.** Мировая норма в часах падает внутри фаз 1995–2007 и 2010–2019 гг. во всех вариантах, а в "
          "деньгах на тех же величинах — нет. В этой конструкции падение в часах нельзя отделить от сокращения "
          "крестьянской и самозанятой занятости (журнал п. 5), поэтому как подтверждение закона оно не читается.")
    A("")
    # ---------------------------------------------------------------- 9
    A("## 9. Что это не доказывает")
    A("")
    A(f"* Отсутствие тренда внутри фаз — не «закон опровергнут»: полуширина 90% ДИ 111 — {pct(o111.half_width)}, "
      f"MDE = 2,8·SE = {pct(o111.mde)} в год; падение меньше этого не различимо.")
    A("* Часы — только производственные и рядовые работники BLS; это не марксова граница производительного труда.")
    A("* Мера 113 не видит ценового обесценивания гл. 15 (банкротства, списания).")
    A("* Мировой уровень: часы разных стран сложены как однородные; самозанятые и крестьяне — в часах.")
    A("* Все выводы — для США и мира 1948–2023 / 1995–2024 гг. и для выбранной классификации производительного труда.")
    A("")
    # ---------------------------------------------------------------- 10
    A("## 10. Невыполненное и недоступное")
    A("")
    A("* Не сделано по рецензии до расчётов (журнал п. 3): 114ʹ — проверка теоремы Окисио по равновесной норме и "
      "жизнеспособности изменений техники; часы мира, взвешенные производительностью; запасы в C; отраслевые "
      "дефляторы выпуска до 1997 г.")
    A("* Вариант часов JHS (1948–2010; предрегистрация 0.1) не построен: перевод 40 кодов ISIC Rev. 3 в 47 кодов не "
      "сделан (журнал п. 7).")
    A("* Аддитивный мост r НФК → r*_h по шагам (П21) не сделан: данных по корпорациям в разрезе 47 кодов нет; вместо "
      "него — мост от определений авторов (§4) (журнал п. 7).")
    A("* Разрыв EXIOBASE 2009→2010 не исследован: нужна документация обработки лет до 2010 г. (FIGARO) по отраслям.")
    A("* Недоступно (список для пользователя): таблицы книг Shaikh & Tonak (1994), Moseley (1991), Kliman (2012); "
      "матрицы капитала Södersten et al. для EXIOBASE 3.10.2 (Zenodo, доступ по запросу); полный EXIOBASE 3.8.2.")
    A("")
    A("## 11. Рецензия после расчётов и ответы")
    A("")
    A("Рецензия — [`results/v13a/review_independent.md`](../results/v13a/review_independent.md): 14 замечаний, "
      "критичных (меняющих итог этапа 1) нет. Исправления и дополнения — журнал п. 7 (после результата), отдельными "
      "коммитами: код — до перезапуска. Все числа этого отчёта — уже после исправлений.")
    A("")
    sp1 = rd("s1_spec_variants.csv")
    nose = sp1[sp1.spec == "no_se"].set_index("year")
    nse = est_.within_trend(np.log(nose.r), PH1, 0.002, "<")
    spl = ex.loc["116: V variant = splice 1998"]
    s2b_k = s2b[s2b.work.str.startswith("Kliman")]
    k_i = s2b_k[s2b_k.step.str.startswith("i:")].iloc[0]
    rows = [
        ["Р1", "принято", f"Сопоставление подписей NIPA 6.7D исправлено (6.7D даёт самозанятых по укрупнённым секторам), "
         f"при ненайденной строке — ошибка. Этап 1 пересчитан: 111 = {ci_pct(o111.est, o111.ci90_lo, o111.ci90_hi)}; "
         f"вариант «без самозанятых» — {ci_pct(nse['est'], nse['ci90_lo'], nse['ci90_hi'])} {lab(nse['label_full'])}"],
        ["Р2", "принято", f"Сверка s_W с CES 82/57 сделана (`s0_checks.csv`, §2). Шов 1998 г. подтверждён: итог частных "
         "отраслей SIC и NAICS совпадает, расходится распределение по кодам. Вариант «сшивка 1998»: 116 = "
         f"{ci_pct(spl.est, spl.ci90_lo, spl.ci90_hi)} {lab(spl.label_full)} (основной {lab(o116.label_full)}). Основная "
         "оценка оставлена предрегистрированной; зависимость 116 от шва вынесена в §0, §7, §8"],
        ["Р3", "принято", "λ-варианты 117, 125–127 и чтения по вариантам — §3.3; §0, §8, REPORT_FINAL и SUMMARY переписаны "
         "условно («при основном индексе `uslt4x`»); чтение 114 × 115 — и по меткам для правила (§3.2)"],
        ["Р4", "принято", f"Оценки в фазах NBER записаны (`s1_outcomes.csv`); r*_h в них "
         f"{ci_pct(o111.nber_est, o111.nber_ci90_lo, o111.nber_ci90_hi)} {lab(o111.nber_label)}, k*_h "
         f"{ci_pct(o110.nber_est, o110.nber_ci90_lo, o110.nber_ci90_hi)} {lab(o110.nber_label)}. Формулировки §0 и §8 "
         "исправлены"],
        ["Р5", "принято", "Варианты «управляющие в V», «s_W ≤ 1», «сшивка 1998» и ω* в A′ — §3.4; описание s_W > 1 в "
         "журнале п. 2 исправлено в п. 7"],
        ["Р6", "принято", f"Шаг i — на окне моста (Kliman: {ci_pct(k_i.est, k_i.ci90_lo, k_i.ci90_hi)} {lab(k_i.label)} на "
         f"{int(k_i.y0)}–{int(k_i.y1)} гг.); шаг ii разбит на ii-а…ii-г; сверка с r_m 9б; вывод §8 смягчён"],
        ["Р7", "принято", "§7: пометки [$], столбец «молчат», денежные оценки, неустойчивость 120–122, оговорка п. 5 к «за» "
         "по 120/121; строки в формате предрегистрации — §12.4"],
        ["Р8", "принято", "Вариант масштаба PWT в национальных ценах, наклоны по фазам, все варианты k_h, доля часов "
         "сельского хозяйства, разрыв 2009→2010 — §5; толкование про крестьян перенесено в §8 и сужено"],
        ["Р9", "принято", f"Окно 2020–2021 в сетке без лет фазы C; доля основной метки 113 в сетке — "
         f"{num(o113.grid_share_main, 3)}; слой подвыборок применён ({o113.nine if isinstance(o113.nine, str) else '—'}); "
         "разложение m и оговорка о циклической производительности — §3.6, §8"],
        ["Р10", "принято", "Изменение USA_BEA — §6; «закрыто» сужено до исходов 7 и 18 (§6, REPORT_FINAL, SUMMARY)"],
        ["Р11", "принято", "Отступления записаны в журнал п. 7; недостающие таблицы — §12; аддитивный мост и JHS — в "
         "невыполненное (§10)"],
        ["Р12", "принято", "Темп P&T по концам, сверка Basu по XR, доля без Basu, доля непроизводительной оплаты у Mohun — §4"],
        ["Р13", "принято", "Kliman 1947–2007 помечен «после результата» в отчёте и сводных документах; статусы вопросов "
         "12б о часах и о Basu обновлены"],
        ["Р14", "принято", "Исправлены: «< 1% с 1973 г.» (§2), проверка Шепли на конце 13а (§3.5), формулировка U (§8), "
         "оговорка к 130 (§7), лицензия EXIOBASE (журнал п. 7), оформление, числа в прозе — из CSV"],
    ]
    L += md_table(rows, ["№", "Решение", "Что сделано"])
    A("")
    # ---------------------------------------------------------------- 12 appendix
    A("## 12. Приложение: предрегистрированные таблицы (рецензия Р11, п. 7)")
    A("")
    A("### 12.1 Наклоны по фазам и однородность (HAC, лаг 4)")
    A("")
    rows = []
    for no in (110, 111, 114, 115, 117, 123, 124, 125, 129, 130):
        r = o(no)
        rows.append([no] + [f"{pct(r[f'slope_{g}'])} ± {num(100 * 1.645 * r[f'se_{g}'], 2)}" for g in "ABC"]
                    + [num(r.p_equal_slopes, 3)])
    L += md_table(rows, ["№", "A 1948–1973", "B 1983–2007", "C 2010–2019", "p (Вальд, β_A = β_B = β_C)"])
    A("")
    A("Полуширина — 1,645·SE_HAC. Однородность «≈ 0» засчитывается только при p ≥ 0,10.")
    A("")
    A("### 12.2 Шепли: средний уровень ln r и парные дивиденды Харсаньи")
    A("")
    shv = sh[sh.statistic == "level"].set_index("switch")
    rows = [[swt, num(shv.loc[swt, "shapley"], 4), num(shv.loc[swt, "fixed_order_step"], 4)]
            for swt in ("cap", "lab", "who", "dep", "val", "den", "total (13a - 12b)")]
    L += md_table(rows, ["Переключатель", "Вклад Шепли в средний уровень ln r", "Шаг в фиксированном порядке"])
    A("")
    pr = rd("s1_shapley_pairs.csv")
    pw = pr[pr.statistic == "within"].set_index("pair").harsanyi_dividend
    pl = pr[pr.statistic == "linear"].set_index("pair").harsanyi_dividend
    pv = pr[pr.statistic == "level"].set_index("pair").harsanyi_dividend
    rows = [[k, pct(pw[k]), pct(pl[k]), num(pv[k], 4)] for k in pw.abs().sort_values(ascending=False).index]
    L += md_table(rows, ["Пара", "Дивиденд: наклон внутри фаз", "Дивиденд: линейный тренд", "Дивиденд: уровень ln r"])
    A("")
    A("### 12.3 Ряды по годам (основная форма) и профиль фаз")
    A("")
    ms_ = pd.read_csv(R / "s1_main_series.csv", index_col=0)
    def mark(y):
        for k_, (a, b) in PH1.items():
            if a <= y <= b:
                return k_
        for a, b in CR1:
            if a <= y <= b:
                return f"окно {a}–{b}"
        return "—"
    rows = [[int(y), mark(int(y)), num(ms_.r[y], 4), num(ms_.k[y], 3), num(ms_.e[y], 3),
             pct(MD.m[y]) if y in MD.index else "—", "дно спада NBER" if int(y) in NBER_TROUGH else ""]
            for y in ms_.index]
    L += md_table(rows, ["Год", "Фаза / окно", "r*_h", "k*_h", "e*_h", "m", ""])
    A("")
    A("### 12.4 Попарные выводы строками (формат предрегистрации)")
    A("")
    for no, g in allm.groupby("no"):
        za, pr_ = tag(g, g.verdict == "за"), tag(g, g.verdict == "против")
        ne = tag(g, (g.verdict != "за") & (g.verdict != "против") & (g.prediction != "молчит"))
        si = tag(g, g.prediction == "молчит")
        A(f"* **{int(no)}:** за {za}; против {pr_}; предсказывали, но нейтрально: {ne}; молчат: {si}.")
    A("")
    (ROOT / "report" / "REPORT_v13a.md").write_text("\n".join(L) + "\n")
    print("written", len(L), "lines")


if __name__ == "__main__":
    build()
