"""Iteration 13b: expected precision before the calculations (pre_registration_v13b.md, section 'Мощность').

Only data already in the project and disclosed in 'Известно при написании' are used: the 13a stage-0 tables
(CES shares, NIPA wages, hours, self-employment), the 13a main series, the 13a world variants and the World Bank
CMO oil price (iteration 10). The same CI procedures as in 13a are applied to the analogues of the 13b outcomes.
Only half-widths (and spreads) are written - slopes and elasticities are NOT written and not printed.

Usage: PYTHONPATH=src python3 -P -m lts.v13b.power  ->  results/v13b/p0_power.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from ..v6.series import ROOT, nipa
from ..v12b import us_io as u
from ..v13a import est

OUT = ROOT / "results" / "v13b"
V13A = ROOT / "results" / "v13a"
CODES = u.CODES
PROD = [c for c in CODES if c not in u.UNPRODUCTIVE]
A_PRIME, B = (1964, 1973), (1983, 2007)


def split_parts(codes):
    """Wage bills and hours of non-PNS ('supervisors' in CES terms) and PNS employees, summed over codes (13a tables)."""
    sh = pd.read_csv(V13A / "s0_ces_shares.csv", dtype={"code": str})
    sh = sh[sh.variant == "main"]
    sW = sh.pivot_table(index="year", columns="code", values="sW").reindex(columns=codes)
    sH = sh.pivot_table(index="year", columns="code", values="sH").reindex(columns=codes)
    W = pd.read_csv(V13A / "s0_wages_47.csv", index_col=0).reindex(columns=codes)
    H = pd.read_csv(V13A / "s0_hours_47.csv", index_col=0).reindex(columns=codes)
    se = pd.read_csv(V13A / "s0_selfemployed_share.csv", index_col=0).reindex(columns=codes).fillna(0.0)
    yrs = sW.index.intersection(W.index).intersection(H.index)
    sW, sH, W, Hemp = sW.loc[yrs].fillna(1.0), sH.loc[yrs].fillna(1.0), W.loc[yrs], (H * (1 - se)).loc[yrs]
    return dict(W_s=((1 - sW) * W).sum(axis=1), W_r=(sW * W).sum(axis=1), H_s=((1 - sH) * Hemp).sum(axis=1),
                H_r=(sH * Hemp).sum(axis=1))


def rel_pay(codes):
    p = split_parts(codes)
    return (p["W_s"] / p["H_s"]) / (p["W_r"] / p["H_r"])


def elasticity_halfwidth(y, x, lag=4):
    """Half-width of the 90% CI of b in y = a + b x + e: wider of HAC (lag 4, normal) and t(n-2)."""
    d = pd.concat([y, x], axis=1).dropna()
    X = sm.add_constant(d.iloc[:, 1].to_numpy(float))
    yy = d.iloc[:, 0].to_numpy(float)
    ols = sm.OLS(yy, X).fit()
    hac = sm.OLS(yy, X).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
    return max(est.Z90 * float(hac.bse[1]), stats.t.ppf(0.95, len(d) - 2) * float(ols.bse[1])), len(d)


def real_oil():
    x = pd.read_excel(ROOT / "data" / "raw" / "v10" / "wb_CMO_Historical_Data_Annual.xlsx",
                      sheet_name="Annual Prices (Nominal)", header=None)
    col = [j for j in range(x.shape[1]) if str(x.iloc[6, j]).strip() == "Crude oil, average"][0]
    s = x.iloc[8:, [0, col]].dropna()
    s = s[pd.to_numeric(s.iloc[:, 1], errors="coerce").notna()]
    p = pd.Series(s.iloc[:, 1].astype(float).to_numpy(), index=s.iloc[:, 0].astype(int).to_numpy())
    defl = nipa(["A191RD"]).A191RD
    return (p / defl.reindex(p.index)).dropna()


def row(outcome, analogue, half_width, boundary, n=np.nan, note=""):
    return dict(outcome=outcome, analogue=analogue, half_width=half_width, zone_boundary=boundary,
                zones_attainable=(bool(half_width < boundary) if np.isfinite(boundary) and np.isfinite(half_width)
                                  else np.nan), n=n, note=note)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    # ---- 132: trend of ln pi, B - A'. Analogue: ln(rel/q - 1), rel = relative hourly pay of non-PNS to PNS employees
    # of productive industries (13a CES/NIPA tables); q = equivalent-wage ratio w_eq / w_rf, unknown before the CPS
    # data -> three fixed values. Zones: M+доп > 0.003, ПП-1 (-0.003; 0.003); both attainable iff h < 0.003.
    rel = rel_pay(PROD)
    for q in (1.0, 1.2, 1.4):
        y = np.log(rel / q - 1).where(rel / q > 1)
        o = est.two_phase_diff(y, A_PRIME, B, 0.002)
        rows.append(row("132 (ln π, B − A′)", f"ln(rel/q − 1), q = {q}, производительные отрасли, 13а", o["half_width"],
                        0.003, o["n"], "rel — отношение часовой оплаты не-PNS к PNS (CES/NIPA 13а)"))
    # ---- 133: ln r_split within phase B only (single trend 1983-2007). Analogue: ln r*_h of 13a.
    ms = pd.read_csv(V13A / "s1_main_series.csv", index_col=0)
    lr = np.log(ms.r)
    o = est.single_trend(lr.loc[B[0]:B[1]], 0.002, ">")
    rows.append(row("133 (ln r_split, внутри B)", "ln r*_h 13а, 1983–2007, тренд одного ряда", o["half_width"], 0.002,
                    o["n"], "метка по правилу: «опровергнуто (экв.)» достижимо, если h < Δ"))
    # ---- 135: common slope within A, B, C. Main phase A of 13b = 1961-1973 (CPS years); variant 1948-1973.
    for a0 in (1961, 1948):
        o = est.within_trend(lr, {"A": (a0, 1973), "B": B, "C": (2010, 2019)}, 0.002, "<")
        rows.append(row("135 (ln r_split, внутри A, B, C)", f"ln r*_h 13а, A = {a0}–1973", o["half_width"], 0.002,
                        o["n"], "метка по правилу: «опровергнуто (экв.)» достижимо, если h < Δ"))
    # ---- 137: elasticity of pi_211 on the 5-year moving average of the real oil price (1972-2023: own CES series
    # of 211 from 1972). Analogue: ln(rel_211 - 1). Zones: > 0.3 / < 0.3; both attainable iff h < 0.3.
    po = real_oil()
    x5 = np.log(po.rolling(5).mean())
    for codes, a0, nm in ((["211"], 1972, "211"), (["211", "213"], 1990, "211 + 213")):
        r = rel_pay(codes)
        for q in (1.0, 1.2, 1.4):
            y = np.log(r / q - 1).where(r / q > 1).loc[a0:2023]
            h, n = elasticity_halfwidth(y, x5.reindex(y.index))
            rows.append(row("137 (Т3, h = 5)", f"ln(rel/q − 1), q = {q}, кода {nm} на ln скользящего среднего (5 лет) "
                                                f"реальной цены нефти (WB CMO / дефлятор ВВП), {a0}–2023", h, 0.3, n,
                            "решение о достижимости — по q = 1,2 (раздел «Мощность»)"))
    # ---- 141-143: the 'interval over variants' is not a sampling CI. Analogue of its width: the relative half-range of
    # world V_h / L across the 13a world variants (lambda gross/net x classification), by year.
    w = pd.read_csv(V13A / "s3_world_years.csv")
    w["v"] = w.V_h / w.L
    g = w.groupby("year").v
    d = ((g.max() - g.min()) / (2 * g.median()))
    for nm in ("141 (e_мир (а), сырые часы)", "142 (e_мир (б))", "143 (e_мир, Хильфердинг)"):
        rows.append(row(nm, "относительный полуразмах V_h/L мира по вариантам 13а (λ валовая/чистая × классификация), "
                            "среднее по годам 1995–2024", float(d.mean()), np.nan, int(g.ngroups),
                        f"интервал по вариантам ожидаемо не накрывает 0, если |e_мир| > ≈ {d.mean():.3f} "
                        f"(максимум по годам {d.max():.3f})"))
    # ---- 139: literature context only (PSS 2014, fig. 3: 0.47, SE 0.11, 18 countries); the blind step decides.
    rows.append(row("139 (Т4а)", "PSS (2014), рис. 3: SE = 0,11, 18 стран; h = t(16) · SE", stats.t.ppf(0.95, 16) * 0.11,
                    0.25, 18, "только ориентир; решает слепой шаг (данные WID, свои долгие разности)"))
    for nm in ("138 (Т2)", "139 (Т4а)", "140 (Т4б)", "145 (штаты, β)", "146 (штаты, γ)"):
        rows.append(row(nm, "аналога в 13а нет — слепой шаг (раздел «Мощность»)", np.nan, np.nan))
    P = pd.DataFrame(rows)
    P.to_csv(OUT / "p0_power.csv", index=False)
    print(P.drop(columns=["analogue", "note"]).round(4).to_string())


if __name__ == "__main__":
    run()
