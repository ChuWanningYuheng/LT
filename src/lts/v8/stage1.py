"""Iteration 8, stage 1: the 19 pre-registered primary outcomes of iterations 1-8 under the single inference rule.

Every number is read from saved results files; nothing is re-estimated except bootstrap / Wilson intervals of
cross-country means and shares (pre_registration_v8.md, 'Методы ДИ').  Outputs results/v8/s1_outcomes.csv and
report/outcomes_table.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..final.check_numbers import md_section
from ..v6.series import ROOT
from .rule import decide, mean_row, share_row
from .stage2 import OUT

RT = "results/report_tables.md"
Z90 = 1.645


def num(s):
    try:
        return float(str(s).replace(",", "."))
    except ValueError:
        return np.nan


def csv(p):
    return pd.read_csv(ROOT / p, low_memory=False)


def coef_row(name, est, se, theta0, delta, direction, **extra):
    lo, hi = est - Z90 * se, est + Z90 * se
    return dict(outcome=name, est=est, ci90_lo=lo, ci90_hi=hi, mde=2.8 * se, theta0=theta0, delta=delta,
                direction=direction, label=decide(est, lo, hi, theta0, delta, direction), **extra)


def no_ci_row(name, est, theta0, delta, direction, **extra):
    return dict(outcome=name, est=est, ci90_lo=np.nan, ci90_hi=np.nan, theta0=theta0, delta=delta, direction=direction,
                label=decide(est, np.nan, np.nan, theta0, delta, direction), **extra)


def outcomes():
    R = []
    lv = md_section(RT, "Уровни: MAWD, подвыборка `all`, капитал=True")
    lv = lv[lv["страна"] != "USA_BEA"]
    op = lv[lv["система"] == "открытая"]
    cl = lv[lv["система"] == "замкнутая (единая з/п)"]
    src_rt = f"`{RT}` (уровни, all, капитал)"
    R.append(mean_row("Итерации 1–2, асимметричная постановка: доля товарных базисов лучше труда (MAWD)",
                      op["доля товаров лучше труда"].map(num), 0.5, 0.10, "<", it="1–2", src=src_rt,
                      old="установлено (как вычислительный факт)"))
    R.append(mean_row("Итерация 2, симметричная постановка: то же (корзина по единой ставке)",
                      cl["доля товаров лучше труда"].map(num), 0.5, 0.10, "<", it="2", src=src_rt,
                      old="неразличимо (предрег. «смешанный»)"))
    p = csv("results/v3/placebo_disp.csv")
    a = p[p["mask"] == "all"].groupby(["country", "kind", "metric"]).share_better.mean().unstack(["kind", "metric"])
    top = (a[[("perm", "mawd"), ("perm", "d"), ("lnorm", "mawd"), ("lnorm", "d")]] <= 0.05).all(1)
    R.append(share_row("Итерация 3, стадия П: доля стран, где часы в верхних 5% плацебо (а) и (б)", top, 0.05, 0.15, ">",
                       it="3", src="`results/v3/placebo_disp.csv`",
                       old="«провал v2 объясняется равномерностью плацебо» (10 из 13); в итоговом отчёте — частично"))
    b = csv("results/final/benchmarks_flat_power.csv")
    b = b[b["mask"] == "all"].groupby(["country", "vector"]).mawd.mean().unstack()
    R.append(share_row("Итерации 3–4: плоский вектор лучше часов по MAWD", b.flat < b.hours, 0.5, 0.15, "<", it="3–4",
                       src="`results/final/benchmarks_flat_power.csv`",
                       old="не предрегистрирован (проверка рецензента v3: 12 из 13)"))
    R.append(share_row("Итерация 3: цены производства (факт. зарплаты) ближе к ценам, чем стоимости",
                       op["PP факт. з/п"].map(num) < op["труд (часы)"].map(num), 0.5, 0.15, "<", it="1–3", src=src_rt,
                       old="частично (не отделено от сжатия разброса)"))
    b31 = csv("results/v4/b31.csv").query("variant=='primary' and not with_rent and not controls and x=='kw'").iloc[0]
    R.append(coef_row("Итерация 4, Б3.1: β r на ln(K/W)", b31.beta, b31.se, 0, 0.01, "<", it="4",
                      src="`results/v4/b31.csv`", old="знак есть (p_Holm = 0,043); итог — не подтверждено (требовалась устойчивость)"))
    s = csv("results/v4/b0_summary.csv").query("subset=='core' and h==5")
    R.append(share_row("Итерация 4, Б0: труд информативен против физически плоского эталона, h = 5",
                       s.informative_vs_phys.astype(bool), 0.5, 0.15, ">", it="4", src="`results/v4/b0_summary.csv`",
                       old="информативен (6 из 8); в итоговом отчёте — частично"))
    b32 = csv("results/v4/b32.csv").query("variant=='primary' and not with_rent and window=='2000-2021'").set_index("source")
    R.append(no_ci_row("Итерация 4, Б3.2: R² выпуска − R² затрат на труд (медианы по странам-годам)",
                       b32.r2["GO"] - b32.r2["labour_W"], 0, 0.05, "<", it="4", src="`results/v4/b32.csv`",
                       old="И (масса прибыли ближе к выпуску); итог части Б — неразличимо"))
    t2 = csv("results/v5/s1_t2.csv").query("test=='T2_annual' and variant=='main' and spec=='gap'").iloc[0]
    R.append(coef_row("Итерация 5: Δ ln r_M на Δ ln(1 + k), годовая панель", t2.beta, t2.beta / t2.t_cr1, 0, 0.25, "<",
                      it="5", src="`results/v5/s1_t2.csv` (T2_annual, main, gap)", old="не подтверждено"))
    s2 = csv("results/v5/s2_tests.csv").query("test=='T2.1a_levels' and variant=='main' and A=='A1'").iloc[0]
    ag = csv("results/v5/agg_country_year.csv").query("variant=='main'")
    me = csv("results/v5/s2_measures.csv")
    dl = 0.5 * ag.rM.std() / me.A1.std()
    R.append(coef_row("Итерация 5: автоматизация (A1) → r_M, уровни", s2.beta, s2.beta / s2.t_cr1, 0, dl, "<", it="5",
                      src="`results/v5/s2_tests.csv` (T2.1a_levels, A1)", old="не подтверждено (и для М, и для И)"))
    R.append(coef_row("Итерация 6, T3: Δ ln r_M на Δ ln(1 + k), десятилетия", 0.982, 0.982 / 5.225, 0, 0.25, "<", it="6",
                      src="`results/v6/tests.log` (main8, decades, country+period, без цикла)", old="против М"))
    tr = csv("results/v6/trends.csv")
    g8 = tr[(tr.variant == "main") & (tr.series == "rM") & (tr.lag == 4) & tr.base &
            tr.geo.isin(["AUS", "DEU", "FRA", "GBR", "ITA", "NLD", "SWE", "USA"])]
    R.append(mean_row("Итерация 6, T1: средний по 8 странам тренд r_M", g8.slope.to_numpy(), 0, 0.0002, "<", it="6",
                      src="`results/v6/trends.csv` (main, лаг 4, полный период)",
                      old="не подтверждено (0 из 8 устойчиво отрицательных)"))
    us = tr[(tr.geo == "USA_NFC") & (tr.variant == "main") & (tr.series == "r") & (tr.lag == 4) & tr.base].iloc[0]
    R.append(coef_row("Итерация 6: тренд r США НФК 1951–2024", us.slope, us.se, 0, 0.0002, "<", it="6",
                      src="`results/v6/trends.csv` (USA_NFC, main, лаг 4)",
                      old="отрицателен при лагах 2–4, неустойчив к ИС и концам; итог — не подтверждено"))
    s51 = csv("results/v6/s5_1.csv").set_index("source").r2_prop
    R.append(no_ci_row("Итерация 6, 5.1: R²(K_adv) − R²(GO)", s51["K_adv τ=0.127 (US)"] - s51["GO"], 0, 0.05, ">",
                       it="6", src="`results/v6/s5_1.csv`", old="не подтверждено"))
    r1 = csv("results/revisit/r1_summary.csv").query("variant=='main' and mask=='all'").iloc[0]
    R.append(share_row("Пересмотр R1: доля стран WIOD, где часы в верхних 5%",
                       np.r_[np.ones(int(r1.top5)), np.zeros(int(r1.n - r1.top5))], 0.05, 0.15, ">", it="пересмотр",
                       src="`results/revisit/r1_summary.csv`", old="промежуточно (24 из 41)"))
    r2 = csv("results/revisit/r2_b31_wiod.csv").query("sample=='all' and not with_rent and variant=='primary' and controls=='none'").iloc[0]
    R.append(coef_row("Пересмотр R2: β r на ln(K/W), WIOD", r2.beta, r2.beta / r2.t_cr1, 0, 0.01, "<", it="пересмотр",
                      src="`results/revisit/r2_b31_wiod.csv`", old="М (знак устойчив); итог — частично"))
    k2 = csv("results/v7/p1_extra.csv").query("check=='K2 beta, profile (a), m = 1'").iloc[0]
    R.append(coef_row("Итерация 7, ч. 1: β при наблюдаемом нематериальном капитале (m = 1)", k2.est, k2.se, 0, 0.01, "<",
                      it="7", src="`results/v7/p1_extra.csv`", old="устойчив (по знаку, m*₀ = 3,5)"))
    pw = csv("results/v7/p2_power.csv").query("vector=='psi Shaikh'")
    R.append(no_ci_row("Итерация 7, ч. 2, тест 3: выигрыш MAWD ψ Шейха против c2 (среднее по BEA и FIGARO)",
                       pw.gain_vs_c2_mawd.mean(), 0, 0.005, ">", it="7", src="`results/v7/p2_power.csv`",
                       old="смешанно"))
    b = csv("results/v8/s2_b.csv").query("panel=='KLEMS' and variant=='main'").iloc[0]
    R.append(coef_row("Итерация 8, этап 2: степень трансформации b (KLEMS)", b.b, b.se, 1, 0.1, "<", it="8",
                      src="`results/v8/s2_b.csv`", old="— (новый исход)"))
    D = pd.DataFrame(R)
    D.insert(0, "n_outcome", range(1, len(D) + 1))
    return D


def fmt(v, nd=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    a = abs(v)
    s = f"{v:.2e}" if 0 < a < 0.001 else f"{v:.{nd}f}"
    return s.replace(".", ",").replace("-", "−")


def write_md(D):
    L = ["# Сводная таблица первичных исходов итераций 1–8",
         "",
         "Правило вывода и пороги Δ заданы до расчёта в [`pre_registration_v8.md`](../pre_registration_v8.md) (этап 1). "
         "Числа — из сохранённых файлов результатов (колонка «Источник»); пересчитываются только интервалы средних и долей "
         "по странам. Код — `src/lts/v8/stage1.py`; машиночитаемая версия — `results/v8/s1_outcomes.csv`.",
         "",
         "* **подтверждено** — 90% ДИ не содержит θ₀, и оценка дальше Δ от θ₀ в ожидаемую сторону;",
         "* **опровергнуто** — 90% ДИ целиком в [θ₀ − Δ; θ₀ + Δ] или целиком по другую сторону от θ₀;",
         "* **неинформативно** — иначе. «Нет оценки неопределённости» — для исходов, у которых в сохранённых файлах нет "
         "SE или единиц для бутстрепа.",
         "",
         "ДИ: регрессии — кластерные SE исходной итерации (90% = ±1,645 SE); доли стран — Уилсон; средние по странам — "
         "бутстреп по странам (2000). MDE = 2,8 · SE (80% мощности, 5%). «За» — направление, в котором исход говорит "
         "в пользу трудовой теории / Маркса.",
         "",
         "**Оговорки (независимая рецензия итерации 8, Р6, Р8).** Исходы 6, 16 и 19 — одна межотраслевая закономерность "
         "(ln(Π/K) = ln(Π/W) − ln(K/W)); в балансе их стоит считать одним фактом. Исход 19 (b < 1) не говорит «за» ни одну "
         "из версий теории: b ≈ 0,5 противоречит и b → 1, и b ≈ 0; его порог задан при известной оценке итерации 7. "
         "ДИ ±1,645·SE при 8–26 кластерах узковаты; с t(G − 1) или диким бутстрепом метки исходов 9 и 11 не меняются.",
         "",
         "| # | Исход | Итерация | θ₀ | Δ | «За» | Оценка | 90% ДИ | MDE | Старая метка | Новая метка | Источник |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in D.itertuples():
        mde = fmt(r.mde, 4) if "mde" in D and np.isfinite(getattr(r, "mde", np.nan)) else "—"
        ci = "—" if not np.isfinite(r.ci90_lo) else f"{fmt(r.ci90_lo, 4)}; {fmt(r.ci90_hi, 4)}"
        est = fmt(r.est, 4)
        if np.isfinite(getattr(r, "k", np.nan)):
            est += f" ({int(r.k)} из {int(r.n)})"
        L.append(f"| {r.n_outcome} | {r.outcome} | {r.it} | {fmt(r.theta0, 2)} | {fmt(r.delta, 4)} | "
                 f"{'<' if r.direction == '<' else '>'} θ₀ | {est} | {ci} | {mde} | {r.old} | **{r.label}** | {r.src} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    D = outcomes()
    D.to_csv(OUT / "s1_outcomes.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 50)
    print(D[["n_outcome", "outcome", "est", "ci90_lo", "ci90_hi", "theta0", "delta", "label"]].round(5).to_string())
    (ROOT / "report" / "outcomes_table.md").write_text(write_md(D))
