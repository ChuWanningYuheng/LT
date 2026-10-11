"""Iteration 12b, stage 1 (US): the rate of profit in labour hours with unit-of-measurement placebos
(pre_registration_v12b.md 1.1-1.5; outcomes 90-94; implementation: journal 1).

For each year 1948-2023 and each unit X (hours = persons engaged in productive industries; primary energy, EIA;
dollar = market price):
  lambda_X = x_X (I - A)^-1   (total-use coefficients: imports competitive; variant: + depreciation D)
  X_total = x_X . x  (= lambda_X . n, n = net product incl. net exports)
  V_X = lambda_X . b   (b = worker basket),  C_X = lambda_X . K  (K = productive capital by producing industry)
  S_X = X_total - V_X;  e_X = S_X / V_X;  k_X = C_X / V_X;  r_X = S_X / (C_X + V_X).

Usage: PYTHONPATH=src python -P -m lts.v12b.stage1 [series|tests|all]
Outputs: results/v12b/s1_*.csv
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from ..v6.series import ROOT
from ..v8.rule import decide
from . import us_io as u

OUT = ROOT / "results" / "v12b"
OUT.mkdir(parents=True, exist_ok=True)
US_PEAKS = [1953, 1957, 1960, 1969, 1973, 1980, 1981, 1990, 2001, 2007, 2019]
VARIANTS = {
    "main": dict(),
    "all industries (L, K, basket)": dict(scope="all"),
    "lambda with depreciation": dict(dep=True),
    "capital composition by investment column": dict(kcomp="inv"),
    "basket (i) classical: all productive wages consumed": dict(basket="classical"),
    "basket (ii) without real estate (531)": dict(basket="no531"),
    "basket (iii) necessities only": dict(basket="necessities"),
    "labour: NIPA detail from 1998": dict(lab="detail"),
    "advanced capital incl. inventories": dict(circ=True),
}


# ================================================================ one year
def year_values(year, scope="productive", dep=False, kcomp="asset", basket="main", lab="group", circ=False):
    s = u.system(year)
    A, n, c, x = s["A"], s["n"], s["c"], s["x"]
    P = u.persons(year, "detail" if (lab == "detail" and year >= 1998) else "group").to_numpy()
    W = u.compensation(year).to_numpy()
    mask = np.array([cd not in u.UNPRODUCTIVE for cd in u.CODES]) if scope == "productive" else np.ones(len(u.CODES), bool)
    M = A.copy()
    if dep:
        d = np.divide(u.depreciation(year), x, out=np.zeros(len(x)), where=x > 0)
        g = s["inv"] / s["inv"].sum()
        M = M + np.outer(g, d)
    Linv = np.linalg.inv(np.eye(len(x)) - M)
    l = np.divide(P * mask, x, out=np.zeros(len(x)), where=x > 0)
    rows = {"h": l}
    if year >= 1949:
        e_row, _ = u.energy_row(year, x)
        rows["E"] = e_row
    lam = {k: v @ Linv for k, v in rows.items()}
    lam["$"] = np.ones(len(x))
    tot = {"h": float((P * mask).sum()), "$": float(n.sum())}
    if "E" in rows:
        tot["E"] = float(rows["E"] @ x)
    # basket
    nip = u.nipa_series()
    Wp = float((W * mask).sum())
    omega = Wp / float(nip.loc[year, "A065RC"])
    if basket == "classical":
        b = Wp * c / c.sum()
    elif basket == "no531":
        c2 = c.copy()
        c2[u.CODES.index("531")] = 0.0
        b = omega * c2
    elif basket == "necessities":
        c2 = np.where(np.isin(u.CODES, u.NECESSITIES), c, 0.0)
        b = c2 * (omega * c.sum() / c2.sum())
    else:
        b = omega * c
    # capital
    K = u.capital(year, "all" if scope == "all" else "productive")
    if kcomp == "inv":
        tot_k = sum(v.sum() for v in K.values())
        K = {"all": tot_k * s["inv"] / s["inv"].sum()}
    Kvec = sum(K.values())
    circ_vec = np.zeros(len(x))
    if circ:
        inv_stock = u.inventories()
        if year in inv_stock.index:
            ic = A @ x
            circ_vec = float(inv_stock.loc[year]) * ic / ic.sum()
    out = dict(year=year, omega=omega, W_prod=Wp, L=tot["h"], basket_usd=float(b.sum()), K_usd=float(Kvec.sum()),
               gdp=float(n.sum()))
    for X, lx in lam.items():
        V = float(lx @ b)
        C = float(lx @ Kvec) + float(lx @ circ_vec)
        S = tot[X] - V
        out.update({f"V_{X}": V, f"C_{X}": C, f"S_{X}": S, f"tot_{X}": tot[X], f"e_{X}": S / V, f"k_{X}": C / V,
                    f"r_{X}": S / (C + V), f"ident_{X}": float(lx @ n) / tot[X] - 1 if not dep else np.nan})
        for typ, kv in K.items():
            out[f"C_{X}_{typ}"] = float(lx @ kv)
    return out


def decomposition_inputs(year):
    """Real productive capital by type (2017 $), real basket (omega x real PCE)."""
    rk = u.real_capital(year)
    return dict(year=year, QK=sum(rk.values()), **{f"QK_{t}": v for t, v in rk.items()})


def series():
    rows = []
    for vname, kw in VARIANTS.items():
        for y in u.YEARS:
            try:
                r = year_values(y, **kw)
            except Exception as exc:                                   # noqa: BLE001
                print("fail", vname, y, exc, flush=True)
                continue
            r["variant"] = vname
            rows.append(r)
        print("series", vname, flush=True)
    S = pd.DataFrame(rows)
    D = pd.DataFrame([decomposition_inputs(y) for y in u.YEARS])
    nip = u.nipa_series()
    D["real_pce"] = D.year.map(nip.DPCERX)
    S = S.merge(D, on="year", how="left")
    S["QC"] = S.omega * S.real_pce
    S.to_csv(OUT / "s1_us_series.csv", index=False)
    return S


# ================================================================ tests
def trend(y, lag=4, delta=0.002, direction="<"):
    y = pd.Series(y).dropna().astype(float)
    t = np.arange(len(y))
    X = sm.add_constant(t)
    h = sm.OLS(y.to_numpy(), X).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
    o = sm.OLS(y.to_numpy(), X).fit()
    b = float(o.params[1])
    w = max(stats.norm.ppf(0.95) * h.bse[1], stats.t.ppf(0.95, len(y) - 2) * o.bse[1])
    return dict(est=b, ci90_lo=b - w, ci90_hi=b + w, n=len(y), y0=int(y.index.min()), y1=int(y.index.max()),
                theta0=0.0, delta=delta, direction=direction, label=decide(b, b - w, b + w, 0, delta, direction))


def robust(y, delta=0.002, direction="<"):
    """Labels for HAC lags 2, 4, 8 x samples (full, without first 5, without last 5)."""
    y = pd.Series(y).dropna()
    labs = []
    for lag in (2, 4, 8):
        for s in (y, y.iloc[5:], y.iloc[:-5]):
            labs.append(trend(s, lag, delta, direction)["label"])
    return dict(robust_labels=";".join(labs), robust=len(set(labs)) == 1)


def share_a(lnQKL, lnk, B=2000, block=5, seed=94):
    """Share of (a) in the trend of ln k_h: slope(ln Q_K/L) / slope(ln k_h); moving-block bootstrap over years."""
    d = pd.DataFrame({"a": lnQKL, "k": lnk}).dropna()
    t = np.arange(len(d), dtype=float)
    sl = lambda tt, v: np.polyfit(tt, v, 1)[0]
    est = sl(t, d.a.to_numpy()) / sl(t, d.k.to_numpy())
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(len(d) / block))
    starts = np.arange(len(d) - block + 1)
    bs = []
    for _ in range(B):
        idx = np.concatenate([np.arange(s0, s0 + block) for s0 in rng.choice(starts, nb)])[:len(d)]
        tt = t[idx]
        den = sl(tt, d.k.to_numpy()[idx])
        if abs(den) > 1e-12:
            bs.append(sl(tt, d.a.to_numpy()[idx]) / den)
    lo, hi = np.quantile(bs, [0.05, 0.95])
    return dict(est=float(est), ci90_lo=float(lo), ci90_hi=float(hi), n=len(d), theta0=0.5, delta=0.1, direction=">",
                label=decide(est, lo, hi, 0.5, 0.1, ">"))


def periods(s, cols, per=((1948, 1973), (1973, 1997), (1997, 2023), (1948, 2023))):
    rows = []
    for a, b in per:
        r = {"period": f"{a}-{b}"}
        for c in cols:
            v = s[c]
            r[c] = float(np.log(v.loc[b] / v.loc[a])) if a in v.index and b in v.index else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def tests():
    S = pd.read_csv(OUT / "s1_us_series.csv")
    m = S[S.variant == "main"].set_index("year").sort_index()
    out, desc = [], []
    lr = np.log(m.r_h)
    o90 = trend(lr, 4, 0.002, "<")
    o90.update(robust(lr, 0.002, "<"))
    out.append(dict(outcome="90: trend ln r_h, 1948-2023", **o90))
    lk = np.log(m.k_h)
    o91 = trend(lk, 4, 0.002, ">")
    o91.update(robust(lk, 0.002, ">"))
    out.append(dict(outcome="91: trend ln k_h, 1948-2023", **o91))
    o92 = trend(np.log(m.r_h / m["r_$"]), 4, 0.002, "<")
    o92.update(robust(np.log(m.r_h / m["r_$"]), 0.002, "<"))
    out.append(dict(outcome="92: trend ln(r_h / r_$), 1948-2023", **o92))
    me = m[m.r_E.notna()]
    o93 = trend(np.log(me.r_h / me.r_E), 4, 0.002, "<")
    o93.update(robust(np.log(me.r_h / me.r_E), 0.002, "<"))
    out.append(dict(outcome="93: trend ln(r_h / r_E), 1949-2023", **o93))
    # decomposition of k_h
    m["hK"] = m.C_h / m.QK
    m["hC"] = m.V_h / m.QC
    m["a_tech"] = m.QK / m.L
    m["b_rel"] = m.hK / m.hC
    m["c_basket_per_L"] = m.QC / m.L
    if o91["label"] == "подтверждено":
        o94 = share_a(np.log(m.a_tech), lk)
    else:
        o94 = dict(est=np.nan, ci90_lo=np.nan, ci90_hi=np.nan, label="не применимо (91 не подтверждён)",
                   theta0=0.5, delta=0.1, direction=">")
    o94["share_point"] = float(np.polyfit(np.arange(len(m)), np.log(m.a_tech), 1)[0] / np.polyfit(np.arange(len(m)), lk, 1)[0])
    out.append(dict(outcome="94: share of (a) technical composition in the trend of ln k_h", **o94))
    O = pd.DataFrame(out)
    O.to_csv(OUT / "s1_outcomes.csv", index=False)
    # descriptive trends of each unit and component
    for nm, s, d_ in (("ln r_$", np.log(m["r_$"]), "<"), ("ln r_E (1949-)", np.log(me.r_E), "<"), ("ln e_h", np.log(m.e_h), ">"),
                      ("ln k_$", np.log(m["k_$"]), ">"), ("ln k_E (1949-)", np.log(me.k_E), ">"),
                      ("ln e_$", np.log(m["e_$"]), ">"), ("ln e_E (1949-)", np.log(me.e_E), ">"),
                      ("ln (a) Q_K/L", np.log(m.a_tech), ">"), ("ln (b) h_K/h_C", np.log(m.b_rel), ">"),
                      ("ln (c) Q_C/L", np.log(m.c_basket_per_L), ">"), ("ln omega", np.log(m.omega), ">")):
        desc.append(dict(item=f"trend {nm}", **trend(s, 4, 0.002, d_)))
    for typ in ("E", "S", "I"):
        hk = m[f"C_h_{typ}"] / m[f"QK_{typ}"]
        desc.append(dict(item=f"trend ln h_K/h_C, {typ}", **trend(np.log(hk / m.hC), 4, 0.002, ">")))
        desc.append(dict(item=f"trend ln Q_K/L, {typ}", **trend(np.log(m[f"QK_{typ}"] / m.L), 4, 0.002, ">")))
    # variants
    for vname in VARIANTS:
        v = S[S.variant == vname].set_index("year").sort_index()
        if v.empty:
            continue
        for k_, d_ in (("r_h", "<"), ("k_h", ">"), ("e_h", ">")):
            desc.append(dict(item=f"variant [{vname}]: trend ln {k_}", **trend(np.log(v[k_]), 4, 0.002, d_)))
        desc.append(dict(item=f"variant [{vname}]: trend ln(r_h / r_$)", **trend(np.log(v.r_h / v["r_$"]), 4, 0.002, "<")))
    D = pd.DataFrame(desc)
    D.to_csv(OUT / "s1_descriptive.csv", index=False)
    # contributions by period
    cols = ["r_h", "r_$", "r_E", "k_h", "k_$", "k_E", "e_h", "e_$", "e_E", "a_tech", "b_rel", "c_basket_per_L", "omega"]
    P = periods(m, cols)
    P.to_csv(OUT / "s1_periods.csv", index=False)
    # phases between NBER peaks: within-phase slopes
    ph = []
    bounds = [1948] + US_PEAKS + [2023]
    for a, b in zip(bounds[:-1], bounds[1:]):
        w = m.loc[a:b]
        if len(w) < 3:
            continue
        t = np.arange(len(w))
        ph.append(dict(phase=f"{a}-{b}", n=len(w), **{f"slope_ln_{k_}": float(np.polyfit(t, np.log(w[k_]), 1)[0])
                                                        for k_ in ("r_h", "k_h", "e_h", "r_$")}))
    PH = pd.DataFrame(ph)
    PH.to_csv(OUT / "s1_phases.csv", index=False)
    # comparison with money rates of earlier iterations
    v6 = pd.read_csv(ROOT / "results" / "v6" / "series.csv")
    nfc = v6[(v6.geo == "USA_NFC") & (v6.variant == "main")].set_index("year").r
    L9 = pd.read_csv(ROOT / "results" / "v9b" / "s1_long_series.csv")
    rm = L9[L9.variant == "main"].set_index("year").r_m
    cmp = pd.DataFrame({"r_h": m.r_h, "r_$": m["r_$"], "r_E": m.r_E, "r_NFC (iter 6)": nfc.reindex(m.index),
                        "r_m (9b)": rm.reindex(m.index)})
    cmp.to_csv(OUT / "s1_compare_money.csv")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 70)
    print(O.round(5).to_string())
    print(D.round(5).to_string())
    print(P.round(3).to_string())
    print(PH.round(4).to_string())
    print(cmp.corr().round(3).to_string())


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else "all"
    if part in ("series", "all"):
        series()
    if part in ("tests", "all"):
        tests()
