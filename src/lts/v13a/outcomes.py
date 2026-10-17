"""Iteration 13a: summary table of primary outcomes of iterations 1-13a (report/outcomes_table.md).

Rows 1-109 from iteration 12b (lts.v12b.outcomes.build); the 12b note on rows 7 and 18 (BEA insurance error, not
recomputed) is completed by the stage-4 recomputation (results/v13a/s4_insurance_fix.csv); new rows 110-131.

Usage: PYTHONPATH=src python -P -m lts.v13a.outcomes
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..v6.series import ROOT
from ..v12b import outcomes as v12bo

OUT = ROOT / "results" / "v13a"


def r(p):
    f = OUT / p
    return pd.read_csv(f) if f.exists() else None


def f4(v):
    return "—" if v is None or not np.isfinite(v) else f"{v:.4f}".replace(".", ",").replace("-", "−")


def build():
    D = v12bo.build()
    fix = r("s4_insurance_fix.csv")
    if fix is not None:
        for _, x in fix.iterrows():
            i = D.index[D.n == str(int(x.outcome))][0]
            add = (f" 13а (этап 4, `results/v13a/s4_insurance_fix.csv`): пересчитано с отношением самозанятых к наёмным по "
                   f"секции K для отраслей 64–66 — {f4(x.est_before)} → {f4(x.est_after)}; метка "
                   f"{'меняется' if x.label_changes else 'не меняется'} («{x.label_after}»). Вторичные результаты "
                   "итераций на экономиках `lts.bea` не пересчитаны.")
            D.loc[i, "note"] = str(D.loc[i, "note"]) + add
    rows = []
    O1 = r("s1_outcomes.csv")
    s2 = r("s2_outcomes.csv")
    O3 = r("s3_outcomes.csv")
    ex = r("s1_extra_after_results.csv").set_index("item")

    def row(n, name, x, src, theta0, delta, direction, note=""):
        rob = x.get("robust")
        if isinstance(rob, (bool, np.bool_)):
            note = ("устойчиво. " if rob else f"неустойчиво ({x.get('robust_why', '')}). ") + note
        rows.append(dict(n=str(n), outcome=name, it="13а", theta0=theta0, delta=delta, direction=direction, est=x.est,
                         lo=x.get("ci90_lo"), hi=x.get("ci90_hi"), old="— (новый исход)",
                         label=x.get("label_full") if isinstance(x.get("label_full"), str) else x.get("label"), src=src,
                         note=note.strip()))
    src1 = "`results/v13a/s1_outcomes.csv`"
    o = lambda no: O1[O1.no == no].iloc[0]                                  # noqa: E731
    lam = lambda no: "; ".join(f"{v}: {f4(ex.loc[f'{no}: lambda before 1947 = {v}', 'est'])} «{ex.loc[f'{no}: lambda before 1947 = {v}', 'label_full']}»"  # noqa: E501
                               for v in ("const", "current"))
    row(110, "США: тренд ln k*_h внутри фаз A, B, C (историческая оценка, (В), V без управляющих, часы)", o(110), src1, 0,
        0.002, ">", note=f"по фазам: A {f4(o(110).slope_A)}, B {f4(o(110).slope_B)}, C {f4(o(110).slope_C)}. λ до 1947 г.: "
                         f"{lam(110)}.")
    row(111, "США: тренд ln r*_h внутри фаз", o(111), src1, 0, 0.002, "<",
        note=f"по фазам: A {f4(o(111).slope_A)}, B {f4(o(111).slope_B)}, C {f4(o(111).slope_C)}. λ до 1947 г.: {lam(111)}. "
             f"Фазы NBER (форма 12б): {f4(o(111).nber_est)} «{o(111).nber_label}». "
             "Правило этапа 1 (задание): «неразличимо».")
    x112 = o(112)
    rows.append(dict(n="112", outcome="США: восстановление r*_h между фазами, средняя Δ ln (1973→1983, 2007→2010)",
                     it="13а", theta0=0, delta=0.05, direction=">", est=x112.est, lo=None, hi=None,
                     old="— (новый исход)", label="описательно (2 события)", src=src1,
                     note="события: " + " и ".join(f4(float(v)) for v in str(x112.events).split(";")) + "."))
    x113 = o(113)
    row(113, "США: R = моральный износ в кризисных окнах / внутри фаз", x113, src1, 1, 0.25, ">",
        note=f"годы дна спадов NBER: R = {f4(x113.nber_est)} «{x113.nber_label}»; для правила этапа «неинформативно».")
    row(114, "США: контрфакт Окисио (реальная корзина на час постоянна внутри фазы), текущая оценка", o(114), src1, 0,
        0.002, "<")
    row(115, "США: контрфакт Окисио, историческая оценка", o(115), src1, 0, 0.002, "<", note=f"λ до 1947 г.: {lam(115)}.")
    spl = ex.loc["116: V variant = splice 1998"]
    row(116, "США: наклон ln e*_h в фазе B минус в A′ = 1964–1973", o(116), src1, 0, 0.002, ">",
        note=f"β_A′ {f4(o(116).slope_A)}, β_B {f4(o(116).slope_B)}. После результата (рецензия Р2, журнал п. 7): со "
             f"сшивкой зарплаты NIPA 1998 г. — {f4(spl.est)} ({f4(spl.ci90_lo)}; {f4(spl.ci90_hi)}) «{spl.label_full}».")
    row(117, "США: тренд ln(r^(В) / r^(П)) внутри фаз", o(117), src1, 0, 0.001, "<")
    x118 = O1[O1.no == 118].iloc[0]
    rows.append(dict(n="118", outcome="США: доля разрыва трендов r НФК и r*_h от ln(1 − U/S_$) (Мозли), 1948–2023",
                     it="13а", theta0=None, delta=None, direction="—", est=x118.est, lo=x118.ci90_lo, hi=x118.ci90_hi,
                     old="— (новый исход)", label="описательно", src=src1,
                     note="внутри фаз ДИ наклона разрыва включает 0 — доля не приводится."))
    if s2 is not None:
        x = s2.iloc[0]
        rows.append(dict(n="119", outcome="Доля работ, где наш аналог воспроизводит тренд автора в пределах ± 0,002 (TOST)",
                         it="13а", theta0=0.5, delta=None, direction="—", est=x.est, lo=x.ci90_lo, hi=x.ci90_hi,
                         old="— (новый исход)", label=f"{int(x.k)} из {int(x.n)}; порог 0,5 пройден" if x.est >= 0.5
                         else f"{int(x.k)} из {int(x.n)}; ниже 0,5 — выводы этапа 1 предварительные",
                         src="`results/v13a/s2_outcomes.csv`",
                         note="Kliman, Шейх, Д–Л — да; Basu et al., Paitaridis & Tsoulfidis — нет."))
    if O3 is not None:
        src3 = "`results/v13a/s3_outcomes.csv`"
        d3 = r("s3_descriptive.csv").set_index("item")
        def natp(col):
            it = f"lambda=net, variant=main: within phases ln {col}"
            if it not in d3.index:
                return ""
            z = d3.loc[it]
            return (f" Масштаб PWT в национальных ценах (после результата, рецензия Р8): {f4(z.est)} ({f4(z.ci90_lo)}; "
                    f"{f4(z.ci90_hi)}) «{z.label_full}».")
        cav = ("часы включают самозанятых и крестьян — не толкуется как подтверждение закона (журнал п. 5); капитал "
               "масштабирован по PWT cn / cgdpo (ППС) при ДС EXIOBASE в рыночных ценах.")
        x = O3[O3.no == 120].iloc[0]
        row(120, "Мир (EXIOBASE 3.11.2): тренд ln r_h внутри фаз 1995–2007 и 2010–2019", x, src3, 0, 0.002, "<",
            note=cav + natp("r_h_natp"))
        x = O3[O3.no == 121].iloc[0]
        row(121, "Мир: тренд ln k_h внутри фаз", x, src3, 0, 0.002, ">", note=cav + natp("k_h_natp"))
        x = O3[O3.no == 122].iloc[0]
        row(122, "Мир: тренд ln(r_h / r_$) внутри фаз", x, src3, 0, 0.002, "<")
    row(123, "США: тренд ln r_h,curr внутри фаз (та же конструкция, текущая оценка, (В))", o(123), src1, 0, 0.002, "<")
    row(124, "США: тренд ln k_h,curr внутри фаз (текущая оценка, (В))", o(124), src1, 0, 0.002, ">")
    row(125, "США: тренд ln(C_h,hist^(П) / L) внутри фаз (материализованное строение)", o(125), src1, 0, 0.002, ">",
        note=f"λ до 1947 г. (после результата, рецензия Р3): {lam(125)}.")
    row(126, "США: ln r*_h, наклон фазы B минус наклон фазы A", o(126), src1, 0, 0.002, ">")
    row(127, "США: ln(C_h,hist^(П) / L), наклон B минус A", o(127), src1, 0, 0.002, "<")
    row(128, "США: ln e с управляющими в V и L, наклон B минус A′", o(128), src1, 0, 0.002, ">")
    row(129, "США: тренд ln r_TSSI (MELT) внутри фаз", o(129), src1, 0, 0.002, "<")
    row(130, "США: тренд ln(r*_h / r*_$) внутри фаз", o(130), src1, 0, 0.002, "<")
    row(131, "США: вековой тренд ln r*_h, 1948–2023 (описательно)", o(131), src1, 0, 0.002, "<",
        note="описательно; в зачёт не идёт.")
    new = pd.DataFrame(rows)
    new = new.iloc[np.argsort(new.n.astype(int).to_numpy(), kind="stable")]
    return pd.concat([D, new], ignore_index=True)        # rows 1-109 keep their order


def write_md(D):
    t = v12bo.write_md(D)
    t = t.replace("# Сводная таблица первичных исходов итераций 1–12б", "# Сводная таблица первичных исходов итераций 1–13а")
    t = t.replace("исходы 90–109 — [`pre_registration_v12b.md`](../pre_registration_v12b.md).",
                  "исходы 90–109 — [`pre_registration_v12b.md`](../pre_registration_v12b.md); исходы 110–131 — "
                  "[`pre_registration_v13a.md`](../pre_registration_v13a.md) (метки — с формой «экв./обр.»).")
    t = t.replace("`src/lts/v12b/outcomes.py`; машиночитаемая версия — `results/v12b/outcomes_table.csv`.",
                  "`src/lts/v12b/outcomes.py`, `src/lts/v13a/outcomes.py`; машиночитаемая версия — "
                  "`results/v13a/outcomes_table.csv`.")
    return t


if __name__ == "__main__":
    D = build()
    D.to_csv(OUT / "outcomes_table.csv", index=False)
    (ROOT / "report" / "outcomes_table.md").write_text(write_md(D))
    print(len(D), "rows")
