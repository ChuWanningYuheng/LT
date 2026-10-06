"""Iteration 9b: summary table of primary outcomes of iterations 1-9b (report/outcomes_table.md).

Rows 1-38 from iteration 9 (lts.v9.outcomes.build), with the 9b revisions (outcomes 20, 22, 26, 27, 30-35);
new rows 39-46 and 22b, 25b, 26b from results/v9b.
"""
from __future__ import annotations

import pandas as pd

from ..v6.series import ROOT
from ..v8.stage1 import fmt
from ..v9 import outcomes as v9o
from .stage1 import OUT


def r(p):
    return pd.read_csv(OUT / p)


def build():
    D = v9o.build()
    D["n"] = D.n.astype(str)

    def upd(n, **kw):
        i = D.index[D.n == n]
        assert len(i) == 1, n
        for k, v in kw.items():
            D.loc[i[0], k] = v

    def add_note(n, txt):
        i = D.index[D.n == n][0]
        cur = D.loc[i, "note"]
        D.loc[i, "note"] = (str(cur) + " " if isinstance(cur, str) and cur else "") + txt

    m20 = r("s5_outcome20_market.csv").set_index("item")
    rv = m20.loc["Z.1 NFC reval_real: crisis - other years"]
    upd("20", outcome="Относительная цена капитала НФК в кризис, кризис − прочие годы (США НФК; прежнее название — "
        "«реальная переоценка», рецензия итерации 9, Р9)")
    add_note("20", f"9б, разведочно: рыночная реальная переоценка нефинансовых активов НФК (Z.1) в кризис ниже на "
             f"{fmt(-rv.est, 3)} ({fmt(rv.ci90_lo, 3)}; {fmt(rv.ci90_hi, 3)}).")
    o4 = r("s4_outcomes.csv").set_index("item")
    o22 = o4.loc["outcome 22 without duplicate US (R11), h=8"]
    add_note("22", f"9б (Р11): без дубля США (32 страны) — {fmt(o22.est, 2)} ({fmt(o22.ci90_lo, 2)}; {fmt(o22.ci90_hi, 2)}), "
             f"«{o22.label}».")
    a9 = r("s4_anchor_v9_corrected.csv").set_index("item")
    c26 = a9.loc["ECM u1 (outcome 26)"]
    old26 = D.loc[D.n == "26"].iloc[0]
    upd("26", est=c26.est, lo=c26.ci90_lo, hi=c26.ci90_hi, label=c26.label,
        old=f"итерация 9 (с прогнозными годами AMECO): {fmt(old26.est, 4)}, «{old26.label}»",
        src="`results/v9b/s4_anchor_v9_corrected.csv`")
    v26 = a9.loc["ECM u1, variant (centred gap; journal 3)"]
    upd("26", note=f"9б, журнал 2: годы AMECO ≤ 2024; p дикого бутстрепа {fmt(c26.p_wild, 3)}; вариант с центрированным "
                   f"разрывом (журнал итерации 9, п. 3): {fmt(v26.est, 4)} ({fmt(v26.ci90_lo, 4)}; {fmt(v26.ci90_hi, 4)}) — «{v26.label}».")
    w27 = o4.loc["outcome 27 (wild bootstrap-t): ln(w/y) on coverage, per 10 pp"]
    old27 = D.loc[D.n == "27"].iloc[0]
    upd("27", lo=w27.ci90_lo, hi=w27.ci90_hi, label=w27.label, old=f"итерация 9 (t(G − 1)): «{old27.label}»",
        src="`results/v9b/s4_outcomes.csv`")
    add_note("27", f"9б: ДИ — более широкий из t(G − 1) и дикого бутстрепа-t; p дикого бутстрепа {fmt(w27.p_wild, 2)}.")
    for n in ("30", "31", "32", "33"):
        add_note(n, "**Снят как тест трудовой теории в 9б** (этап 2.1): межстрановое сравнение внутри отрасли проверяет "
                    "закон единой цены.")
    p34 = r("s5_outcome34_paired.csv").iloc[0]
    add_note("34", f"9б: перцентильный ДИ парного кластерного бутстрепа по отраслям (499) — {fmt(p34.ci90_lo, 3)}; "
                   f"{fmt(p34.ci90_hi, 3)}, «{p34.label}». Вывод о мобильности капитала снят.")
    o35 = r("s5_outcome35.csv").set_index("outcome")
    m35 = o35.loc["outcome 35, main from 9b: STAN, ADF lag 1"]
    w35 = o35.loc["outcome 35 variant: WIOD, economy-wide VA deflator, ADF lag 1"]
    old35 = D.loc[D.n == "35"].iloc[0]
    upd("35", outcome="Доля стран, где реальный курс коинтегрирован с относительной реальной стоимостью труда (STAN, "
        "лаг ADF = 1)", est=m35.est, lo=m35.ci90_lo, hi=m35.ci90_hi, label=m35.label,
        old=f"итерация 9 (подбор лагов): {fmt(old35.est, 2)}, «{old35.label}»", src="`results/v9b/s5_outcome35.csv`",
        note=f"9б (Р3, Р4): основной — лаг 1. WIOD с дефлятором ДС всей экономики: {int(w35.k)} из {int(w35.n)}, «{w35.label}».")
    rows = []

    def row(n, name, est, lo, hi, label, src, theta0, delta, direction, note=""):
        rows.append(dict(n=n, outcome=name, it="9б", theta0=theta0, delta=delta, direction=direction, est=est, lo=lo, hi=hi,
                         old="— (новый исход)", label=label, src=src, note=note))
    ln = r("s1_long_outcomes.csv")
    sel = lambda it, var="main", sp="ratio splice", per="1947-2024": ln[(ln["item"] == it) & (ln.variant == var) &  # noqa: E731
                                                                        (ln.splice == sp) & (ln.period == per)].iloc[0]
    for n, it, name in (("39", "39: trend r_m", "Тренд марксовой нормы прибыли r_m, США 1947–2024 (стык по отношению)"),
                        ("40", "40: trend r_m - r", "Тренд (r_m − r), США 1947–2024")):
        x = sel(it)
        vv = "; ".join(f"{v} {fmt(sel(it, v).est, 5)} «{sel(it, v).label}»" for v in ("V2", "V3", "V4", "V5"))
        raw = sel(it, sp="raw splice")
        row(n, name, x.est, x.ci90_lo, x.ci90_hi, x.label, "`results/v9b/s1_long_outcomes.csv`", 0, 0.0002, "<",
            note=f"варианты: {vv}; сырой стык {fmt(raw.est, 5)} «{raw.label}». Обобщение падения 1997–2024 гг. на длинный ряд опровергнуто.")
    eu = r("s1_eu_outcomes.csv")
    eux = r("s1_eu_outcomes_extended.csv")
    for n, key, name in (("41", "41:", "Доля стран (Eurostat, ≥ 20 лет), где r_m падает («подтверждено»)"),
                         ("42", "42:", "Доля стран, где r_m падает быстрее обычной r")):
        x = eu[(eu.variant == "main") & eu.outcome.str.startswith(key)].iloc[0]
        xx = eux[(eux.variant == "main") & eux.outcome.str.startswith(key)].iloc[0]
        row(n, name, x.est, x.ci90_lo, x.ci90_hi, x.label, "`results/v9b/s1_eu_outcomes.csv`", 0.5, 0.15, ">",
            note=f"{int(x.k)} из {int(x.n)}; расширенная выборка (журнал 1): {int(xx.k)} из {int(xx.n)}, «{xx.label}».")
    o43 = r("s1_moseley_outcome.csv").iloc[0]
    row("43", "Тренд доли непроизводительной оплаты, США 1947–2024 (Мозли)", o43.est, o43.ci90_lo, o43.ci90_hi, o43.label,
        "`results/v9b/s1_moseley_outcome.csv`", 0, 0.0005, ">")
    s2 = r("s2_outcomes.csv")
    a = s2[s2.outcome.str.startswith("44: share(hours) - share(lab)")].iloc[0]
    b = s2[s2.outcome.str.startswith("44: share(hours) - share(cap)")].iloc[0]
    row("44", "«Мировое среднее лучше национального»: доля стран для часов − для фонда оплаты (эталон Р18)", a.est,
        a.ci90_lo, a.ci90_hi, a.label, "`results/v9b/s2_outcomes.csv`", 0, 0.1, ">",
        note=f"часы {fmt(a.share_hours, 2)}, фонд оплаты {fmt(a.share_other, 2)}, капитал {fmt(b.share_other, 2)}; "
             f"часы − капитал {fmt(b.est, 2)} ({fmt(b.ci90_lo, 2)}; {fmt(b.ci90_hi, 2)}). Хрупко: расходятся "
             f"{int(a.discordant_hours_only)} против {int(a.discordant_other_only)} стран, Мак-Немар p = "
             f"{fmt(a.mcnemar_p_one_sided, 3)}; плацебо не посчитаны.")
    for n, key, name, th, de, dr in (("45", "45:", "R²(мировые часы) − max(R² капитала, энергии), изменения за 5 лет", 0, 0.05, ">"),
                                     ("46", "46:", "MAWD(мировые часы) − min(капитал, энергия), веса по ППС, 2005", 0, 0.01, "<")):
        x = s2[s2.outcome.str.startswith(key)].iloc[0]
        row(n, name, x.est, x.ci90_lo, x.ci90_hi, x.label, "`results/v9b/s2_outcomes.csv`", th, de, dr)
    for n, key, name, th, de, dr in (
            ("22б", "outcome 22b (LCI D11, EU), h=8", "Перенос ИПЦ в LCI D11 (ЕС), h = 8", "", "[0,9; 1,1]", "H_tech"),
            ("25б", "outcome 25b: indexation - control within energy importers (d2023)",
             "Индексация − контроль внутри импортёров энергии, 2021 → 2023", 0, 0.02, ">"),
            ("26б", "outcome 26b: gamma on slack (t-1)", "Δ ln реальной оплаты на slack t−1 (ЕС, 2008–2024)", 0, 0.002, "<")):
        x = o4.loc[key]
        note = ""
        if n == "25б":
            note = f"перестановочный p = {fmt(x.p_perm, 3)}; импортёры и прочие в контроле не различаются."
        if n == "26б":
            u = o4.loc["same sample, official unemployment"]
            note = (f"p дикого бутстрепа {fmt(x.p_wild, 3)}; та же выборка с официальной безработицей: {fmt(u.est, 4)} "
                    f"({fmt(u.ci90_lo, 4)}; {fmt(u.ci90_hi, 4)}), «{u.label}».")
        row(n, name, x.est, x.ci90_lo, x.ci90_hi, x.label, "`results/v9b/s4_outcomes.csv`", th, de, dr, note=note)
    return pd.concat([D, pd.DataFrame(rows)], ignore_index=True)


def write_md(D):
    t = v9o.write_md(D)
    t = t.replace("# Сводная таблица первичных исходов итераций 1–9", "# Сводная таблица первичных исходов итераций 1–9б")
    t = t.replace("исходы 20–38 и их пороги — [`pre_registration_v9.md`](../pre_registration_v9.md).",
                  "исходы 20–38 и их пороги — [`pre_registration_v9.md`](../pre_registration_v9.md); исходы 39–46, 22б, 25б, 26б и "
                  "пересмотры 9б — [`pre_registration_v9b.md`](../pre_registration_v9b.md).")
    t = t.replace("`src/lts/v9/outcomes.py`; машиночитаемая версия — `results/v9/outcomes_table.csv`.",
                  "`src/lts/v9/outcomes.py`, `src/lts/v9b/outcomes.py`; машиночитаемая версия — `results/v9b/outcomes_table.csv`.")
    return t


if __name__ == "__main__":
    D = build()
    D.to_csv(OUT / "outcomes_table.csv", index=False)
    (ROOT / "report" / "outcomes_table.md").write_text(write_md(D))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 70)
    print(D[["n", "outcome", "est", "lo", "hi", "label"]].round(4).to_string())
