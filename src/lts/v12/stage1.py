"""Iteration 12, stage 1: counteracting factors of the law of the tendency, measured explicitly (pre_registration_v12.md).

1.1 profits from the rest of the world: US corporations (NIPA B394RC, B933RC, A445RC, A051RC; BEA FAAt401 corporate
    capital; Z.1 US direct investment abroad), EU non-financial corporations (Eurostat B2A3N, D43).
1.2 decomposition and counterfactual profit rate:
    (A) US NFC: ln r = ln sigma + ln n + ln rho + ln q;  r_full = r * F;  counteracting: sigma, rho, F
    (B) US private economy (9b): ln r_B = ln(S/NDP) + ln(1 - U/S) + ln(P/P_K) + ln q_B; counteracting: S/NDP, 1-U/S, P/P_K, F
        r_m counterfactual with nu and the relative price of capital held at the base year
    (C) AMECO 10 countries: as (A), F = PI_gni / PI
Outcomes 75-80.

Usage: PYTHONPATH=src python -P -m lts.v12.stage1
Outputs: results/v12/s1_*.csv
"""
from __future__ import annotations

import zipfile

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from ..v6.national import EUROSTAT, es, eurostat
from ..v6.series import ROOT, fa, nipa
from ..v8.rule import decide, share_row

OUT = ROOT / "results" / "v12"
OUT.mkdir(parents=True, exist_ok=True)
BASES = (1950, 1965, 1980, 1997)
END = 2024
PERIODS = ((1950, 1973), (1973, 1997), (1997, 2024), (1950, 2024))
AMECO = ("AUS", "CAN", "DEU", "FRA", "GBR", "ITA", "JPN", "NLD", "SWE", "USA")


# ================================================================ trends (rule of 9b, any delta / direction)
def trend(y, delta=0.002, direction="<"):
    y = pd.Series(y).dropna().astype(float)
    t = np.arange(len(y))
    X = sm.add_constant(t)
    h = sm.OLS(y.to_numpy(), X).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
    o = sm.OLS(y.to_numpy(), X).fit()
    b = float(o.params[1])
    w = max(stats.norm.ppf(0.95) * h.bse[1], stats.t.ppf(0.95, len(y) - 2) * o.bse[1])
    return dict(est=b, ci90_lo=b - w, ci90_hi=b + w, n=len(y), y0=int(y.index.min()), y1=int(y.index.max()),
                theta0=0.0, delta=delta, direction=direction, label=decide(b, b - w, b + w, 0, delta, direction))


def counterfactual(actual, factors, b):
    """actual(t) * prod_f f(b)/f(t), t >= b"""
    a = actual.loc[b:]
    cf = a.copy()
    for f in factors:
        cf = cf * f.loc[b] / f.loc[b:]
    return cf


def contributions(parts, periods=PERIODS):
    rows = []
    for a, b in periods:
        r = {"period": f"{a}-{b}"}
        for k, s in parts.items():
            s = s.dropna()
            a2, b2 = max(a, s.index.min()), min(b, s.index.max())
            r[k] = float(np.log(s.loc[b2] / s.loc[a2])) if a2 < b2 else np.nan
            if (a2, b2) != (a, b):
                r[k + "_years"] = f"{a2}-{b2}"
        rows.append(r)
    return pd.DataFrame(rows)


# ================================================================ data
def nipa1(code):
    return nipa([code])[code]


def us_nfc():
    s = pd.read_csv(ROOT / "results" / "v6" / "series.csv")
    d = s[(s.geo == "USA_NFC") & (s.variant == "main")].set_index("year").sort_index()
    row = nipa1("B394RC")
    d["ROW"] = row.reindex(d.index)
    d["F"] = 1 + d.ROW / d.PI
    d["r_full"] = d.r * d.F
    d["sigma"] = d.PI / d.Y
    d["n"] = d.Y / (d.P_Y * d.Q_Y)
    d["rho"] = d.P_Y / d.P_K
    d["q"] = d.Q_Y / d.Q_K
    d["prod"] = d.Q_Y / d.H
    d["tcc"] = d.Q_K / d.H
    d["u"] = 1 + d.gap / 100
    tcu = pd.read_csv(ROOT / "data" / "raw" / "v5" / "fred_TCU.csv")
    tcu.columns = ["date", "v"]
    tcu["year"] = pd.to_datetime(tcu.date).dt.year
    g = tcu.groupby("year").v
    d["u_tcu"] = (g.mean()[g.count() == 12] / 100).reindex(d.index)
    chk = (d.sigma * d.n * d.rho * d.q / d.r - 1).abs().max()
    assert chk < 1e-9, chk
    return d.loc[1948:END]


def us_private():
    L = pd.read_csv(ROOT / "results" / "v9b" / "s1_long_series.csv")
    d = L[L.variant == "main"].set_index("year").sort_index()
    P = nipa1("A191RD").reindex(d.index)
    kc = fa(1, "FAAt101-A", 3)
    kq = fa(1, "FAAt102-A", 3)
    PK = kc / (kq * kc.loc[2017] / kq.loc[2017])
    d["P"] = P / P.loc[2017]
    d["PK"] = PK.reindex(d.index)
    d["S"] = d.ndp - d.v
    d["U"] = d.comp_all - d.v
    d["s_share"] = d.S / d.ndp
    d["unprod"] = 1 - d.U / d.S
    d["relp"] = d.P / d.PK
    d["rB"] = (d.ndp - d.comp_all) / d.K_private
    d["qB"] = (d.ndp / d.K_private) / d.relp
    d["ROW"] = nipa1("B394RC").reindex(d.index)
    d["F"] = 1 + d.ROW / (d.ndp - d.comp_all)
    d["rB_full"] = d.rB * d.F
    d["RP"] = 1 / d.relp
    assert (d.s_share * d.unprod * d.relp * d.qB / d.rB - 1).abs().max() < 1e-9
    assert (d.rB - d.r).abs().max() < 1e-12
    return d.loc[1948:END]


def ameco_country(geo):
    s = pd.read_csv(ROOT / "results" / "v6" / "series.csv")
    d = s[(s.geo == geo) & (s.variant == "main")].set_index("year").sort_index()
    d = d[(d.index <= END)]
    d["F"] = d.PI_gni / d.PI
    d["r_full"] = d.r * d.F
    d["sigma"] = d.PI / d.Y
    d["n"] = d.Y / (d.P_Y * d.Q_Y)
    d["rho"] = d.P_Y / d.P_K
    d["q"] = d.Q_Y / d.Q_K
    return d


# ================================================================ 1.1 profits from the rest of the world
def stage11():
    rows, desc = [], []
    A051, A445 = nipa1("A051RC"), nipa1("A445RC")
    B394, B933, B934 = nipa1("B394RC"), nipa1("B933RC"), nipa1("B934RC")
    Kc = fa(4, "FAAt401-A", 17)
    yrs = range(1948, END + 1)
    d = pd.DataFrame({"total": A051, "domestic": A445, "row_net": B394, "row_rec": B933, "row_pay": B934, "K_corp": Kc})
    d = d.reindex(yrs)
    d["share_row"] = d.row_net / d.total
    d["share_rec"] = d.row_rec / d.total
    d["r_dom"] = d.domestic / d.K_corp
    d["r_full"] = d.total / d.K_corp
    # foreign return: receipts on the US direct investment position abroad (Z.1, market value, Q4)
    z = zipfile.ZipFile(ROOT / "data" / "raw" / "v6" / "z1" / "z1_csv_files.zip")
    f = pd.read_csv(z.open("csv/F89_2_s.csv")).set_index("date")
    q4 = f[f.index.str.endswith("Q4")]
    pos = pd.to_numeric(q4["LM263192005.Q"].replace("ND", np.nan), errors="coerce")
    pos.index = q4.index.str[:4].astype(int)
    d["usdia_mv"] = pos.reindex(d.index)
    d["r_foreign"] = d.row_rec / d.usdia_mv.shift(1)
    d.to_csv(OUT / "s1_us_row_series.csv")
    rows.append(dict(outcome="80: trend of the rest-of-the-world share of US corporate profits (B394RC/A051RC), 1948-2024",
                     **trend(d.share_row, 0.001, ">")))
    for k in ("r_dom", "r_full"):
        desc.append(dict(item=f"1.1 trend of US corporate {k} (A445RC or A051RC / FAAt401 corporate), 1948-2024",
                         **trend(d[k], 0.0002, "<")))
    desc.append(dict(item="1.1 trend of (r_full - r_dom), 1948-2024", **trend(d.r_full - d.r_dom, 0.0002, ">")))
    C = contributions({"r_full": d.r_full, "r_dom": d.r_dom}, ((1948, 1973), (1973, 1997), (1997, 2024), (1948, 2024)))
    C["foreign_contribution"] = C.r_full - C.r_dom
    C.to_csv(OUT / "s1_us_row_periods.csv", index=False)
    desc.append(dict(item="1.1 receipts share B933RC/A051RC: 1998 / 2024", est=float(d.share_rec.loc[1998]),
                     est2=float(d.share_rec.loc[END])))
    desc.append(dict(item="1.1 foreign return B933RC / US direct investment abroad (Z.1, market value, previous Q4): "
                          "mean 1999-2024 and trend", est=float(d.r_foreign.loc[1999:END].mean()),
                     **{k: v for k, v in trend(d.r_foreign.loc[1999:END], 0.0002, "<").items() if k != "est"}))
    # EU non-financial corporations
    t = es(ROOT / "data" / "raw" / "v12" / "es_nasa_10_nf_tr_S11_property.json")
    eu = []
    for frame in eurostat():
        g3 = frame.geo.iloc[0]
        g2 = EUROSTAT[g3]
        rec = t[(t.geo == g2) & (t.na_item == "D43") & (t.direct == "RECV")].set_index("time").value
        pay = t[(t.geo == g2) & (t.na_item == "D43") & (t.direct == "PAID")].set_index("time").value
        x = frame.copy()
        x["D43_rec"], x["D43_pay"] = rec.reindex(x.index), pay.reindex(x.index)
        x = x.loc[1995:END].dropna(subset=["PI", "K", "D43_rec"])
        if len(x) < 10:
            continue
        x["r_dom"] = x.PI / x.K
        x["r_full"] = (x.PI + x.D43_rec) / x.K
        x["r_full_net"] = (x.PI + x.D43_rec - x.D43_pay) / x.K
        x["share_D43"] = x.D43_rec / (x.PI + x.D43_rec)
        for k in ("r_dom", "r_full", "share_D43"):
            tr = trend(x[k], 0.0002, "<" if k != "share_D43" else ">")
            eu.append(dict(geo=g3, series=k, first=float(x[k].iloc[0]), last=float(x[k].iloc[-1]), **tr))
        x.assign(geo=g3).to_csv(OUT / f"s1_eu_{g3}.csv")
    pd.DataFrame(eu).to_csv(OUT / "s1_eu_trends.csv", index=False)
    return rows, desc


# ================================================================ 1.2 decomposition and counterfactuals
def stage12():
    rows, desc, cf_rows = [], [], []
    # (A) US NFC
    d = us_nfc()
    d.to_csv(OUT / "s1_us_nfc_factors.csv")
    parts = {"r_full": d.r_full, "r": d.r, "sigma": d.sigma, "n": d.n, "rho": d.rho, "q": d.q, "F": d.F,
             "prod (Q_Y/H)": d["prod"], "tcc (Q_K/H)": d.tcc, "u (AMECO gap)": d.u, "u (TCU)": d.u_tcu}
    contributions(parts).to_csv(OUT / "s1_us_nfc_contributions.csv", index=False)
    S = {}
    for b in BASES:
        cf = counterfactual(d.r_full, [d.sigma, d.rho, d.F], b)
        S[b] = cf
        tr = trend(np.log(cf))
        cf_rows.append(dict(series="A: US NFC", base=b, **tr))
        tr2 = trend(np.log(cf / d.r_full.loc[b:]))
        cf_rows.append(dict(series="A: US NFC, ln(r_cf / r_full)", base=b, **tr2))
        cf_rows.append(dict(series="A: US NFC, actual ln r_full", base=b, **trend(np.log(d.r_full.loc[b:]))))
        cf_nof = counterfactual(d.r, [d.sigma, d.rho], b)
        cf_rows.append(dict(series="A: US NFC without F (r, sigma and rho fixed)", base=b, **trend(np.log(cf_nof))))
        cf_n = counterfactual(d.r_full, [d.sigma, d.rho, d.F, d.n], b)
        cf_rows.append(dict(series="A: US NFC, also n (net/gross) fixed", base=b, **trend(np.log(cf_n))))
        for nm, f in (("sigma only", [d.sigma]), ("rho only", [d.rho]), ("F only", [d.F])):
            cf_rows.append(dict(series=f"A: US NFC, {nm} fixed", base=b, **trend(np.log(counterfactual(d.r_full, f, b)))))
        if b == 1950:
            rows.append(dict(outcome="75: US NFC trend of ln r_cf (sigma, rho, F at 1950), 1950-2024", **tr))
            rows.append(dict(outcome="76: US NFC trend of ln(r_cf / r_full), 1950-2024", **tr2))
            rows.append(dict(outcome="78: US NFC trend of ln q = ln(Q_Y / Q_K), 1950-2024", **trend(np.log(d.q.loc[1950:]))))
    pd.DataFrame(S).to_csv(OUT / "s1_us_nfc_counterfactuals.csv")
    for nm, s in (("ln(Q_Y/H)", np.log(d["prod"])), ("ln(Q_K/H)", np.log(d.tcc)), ("ln sigma", np.log(d.sigma)),
                  ("ln rho", np.log(d.rho)), ("ln F", np.log(d.F)), ("ln n", np.log(d.n)), ("ln u (AMECO gap)", np.log(d.u))):
        desc.append(dict(item=f"1.2A trend of {nm} (from first year with data or 1950)", **trend(s.loc[1950:], 0.002, "<")))
    # (B) US private economy, Marxian categories
    p = us_private()
    p.to_csv(OUT / "s1_us_private_factors.csv")
    contributions({"rB_full": p.rB_full, "rB": p.rB, "S/NDP": p.s_share, "1-U/S": p.unprod, "P/P_K": p.relp,
                   "q_B": p.qB, "F_B": p.F, "r_m": p.r_m}).to_csv(OUT / "s1_us_private_contributions.csv", index=False)
    for b in BASES:
        cf = counterfactual(p.rB_full, [p.s_share, p.unprod, p.relp, p.F], b)
        tr = trend(np.log(cf))
        cf_rows.append(dict(series="B: US private (S/NDP, 1-U/S, P/P_K, F fixed)", base=b, **tr))
        cf_rows.append(dict(series="B: US private, ln(r_cf / r_full)", base=b, **trend(np.log(cf / p.rB_full.loc[b:]))))
        rm_cf = (1 - p.nu.loc[b]) / (p.kappa.loc[b:] * p.RP.loc[b] / p.RP.loc[b:] + p.nu.loc[b])
        cf_rows.append(dict(series="B: r_m counterfactual (nu and P_K/P fixed)", base=b, **trend(np.log(rm_cf))))
        cf_rows.append(dict(series="B: actual ln r_m", base=b, **trend(np.log(p.r_m.loc[b:]))))
        if b == 1950:
            rows.append(dict(outcome="77: US private economy trend of ln r_cf (S/NDP, 1-U/S, P/P_K, F at 1950), 1950-2024",
                             **tr))
    # (C) AMECO countries
    labs = []
    for geo in AMECO:
        a = ameco_country(geo).dropna(subset=["r_full", "sigma", "rho", "F"])
        if a.empty:
            continue
        a.to_csv(OUT / f"s1_ameco_{geo}.csv")
        for b0 in (1965, 1980, 1997):
            b = max(b0, int(a.index.min()))
            cf = counterfactual(a.r_full, [a.sigma, a.rho, a.F], b)
            tr = trend(np.log(cf))
            cf_rows.append(dict(series=f"C: AMECO {geo}", base=b, base_planned=b0, **tr))
            cf_rows.append(dict(series=f"C: AMECO {geo}, actual ln r_full", base=b, base_planned=b0,
                                **trend(np.log(a.r_full.loc[b:]))))
            if b0 == 1965:
                labs.append(dict(geo=geo, base=b, label=tr["label"]))
    L = pd.DataFrame(labs)
    r79 = share_row("79", (L.label == "подтверждено").to_numpy(), 0.5, 0.15, ">")
    r79["outcome"] = "79: AMECO share of countries with trend ln r_cf (base 1965 or first year) 'подтверждено' (< 0)"
    rows.append(r79)
    CF = pd.DataFrame(cf_rows)
    CF.to_csv(OUT / "s1_counterfactual_trends.csv", index=False)
    L.to_csv(OUT / "s1_ameco_labels.csv", index=False)
    return rows, desc


def run():
    r1, d1 = stage11()
    r2, d2 = stage12()
    O = pd.DataFrame(r2 + r1)
    O.to_csv(OUT / "s1_outcomes.csv", index=False)
    D = pd.DataFrame(d1 + d2)
    D.to_csv(OUT / "s1_descriptive.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 100)
    print(O[["outcome", "est", "ci90_lo", "ci90_hi", "n", "label"]].round(5).to_string())
    print(D.round(5).to_string())
    CF = pd.read_csv(OUT / "s1_counterfactual_trends.csv")
    print(CF[["series", "base", "est", "ci90_lo", "ci90_hi", "n", "label"]].round(5).to_string())


if __name__ == "__main__":
    run()
