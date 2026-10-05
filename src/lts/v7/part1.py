"""Iteration 7, part 1: unmeasured capital and the Marx sign (pre_registration_v7.md, part 1).

Threshold m* at which the slope of r on ln(K/W) (country x year effects, industry clusters) reaches zero /
insignificance, when U = m * profile is added to capital and m * (I_U - D_U) to profit.
Profiles: (a) observed KLEMS non-national-accounts intangibles; (b) worst case, U proportional to the wage bill
in labour-intensive industries.  Panels: KLEMS (iteration 4, B3 primary) and WIOD (revisit R2).
Outputs: results/v7/p1_*.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from ..v4.b_data import divs
from ..v4.b_tests import load as load_klems
from ..v4.b_tests import sample as sample_klems
from ..revisit.wiod import r2_panel
from ..v6.series import ROOT

OUT = ROOT / "results" / "v7"
OUT.mkdir(parents=True, exist_ok=True)
KL = ROOT / "data" / "raw" / "euklems"
GRID = np.round(np.arange(0, 20.0001, 0.05), 2)
LOW_INTANG = ["C10-C12", "C13-C15", "C16-C18", "C22-C23", "C24-C25", "F", "H49", "H50", "H51", "H52", "H53", "I"]
TYPES = ["OrgCap", "Brand", "Train", "Design", "NFP", "RD"]
ISO3 = {"AUT": "AT", "BEL": "BE", "BGR": "BG", "CYP": "CY", "CZE": "CZ", "DEU": "DE", "DNK": "DK", "EST": "EE",
        "GRC": "EL", "ESP": "ES", "FIN": "FI", "FRA": "FR", "HRV": "HR", "HUN": "HU", "IRL": "IE", "ITA": "IT",
        "LTU": "LT", "LUX": "LU", "LVA": "LV", "MLT": "MT", "NLD": "NL", "POL": "PL", "PRT": "PT", "ROU": "RO",
        "SWE": "SE", "SVN": "SI", "SVK": "SK", "GBR": "UK", "USA": "US", "JPN": "JP"}


# ---------------------------------------------------------------- estimation
def slope(d, r, x, ctrl=(), fe="cy"):
    """beta of r on x after demeaning by fe (one key or a list of keys, alternating projections), CR1 by industry."""
    cols = [r, x] + list(ctrl)
    keys = [fe] if isinstance(fe, str) else list(fe)
    X = d[cols].astype(float).copy()
    for _ in range(1 if len(keys) == 1 else 50):
        for k in keys:
            X = X - X.groupby(d[k]).transform("mean")
    fit = sm.OLS(X[r], X[[x] + list(ctrl)]).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.ind)[0]})
    return float(fit.params[x]), float(fit.pvalues[x]), fit


def threshold(d, make):
    """make(d, m) -> (r, x) arrays; returns per-m beta/p and the thresholds."""
    rows = []
    for m in GRID:
        r, x = make(d, m)
        e = d.assign(_r=r, _x=x)
        e = e[np.isfinite(e._r) & np.isfinite(e._x)]
        b, p, _ = slope(e, "_r", "_x")
        rows.append(dict(m=m, beta=b, p=p, n=len(e)))
        if b >= 0 and m > 0 and p >= 0.05 and len(rows) > 3 and rows[-2]["beta"] >= 0:
            break
    t = pd.DataFrame(rows)
    m0 = t.loc[t.beta >= 0, "m"].min() if (t.beta >= 0).any() else np.nan
    ms = t.loc[t.p >= 0.05, "m"].min() if (t.p >= 0.05).any() else np.nan
    return t, m0, ms


# ---------------------------------------------------------------- KLEMS panel
def klems_types():
    cols = ["nace_r2_code", "geo_code", "year"] + [f"{a}_{t}" for t in TYPES for a in ("K", "I")]
    ia = pd.read_csv(KL / "intangibles_analytical.csv", low_memory=False, usecols=cols)
    return ia.rename(columns={"nace_r2_code": "ind", "geo_code": "geo"})


def klems_panel():
    p = load_klems("primary")
    p = sample_klems(p, rent=False)
    p = p.merge(klems_types(), on=["geo", "ind", "year"], how="left")
    p = p.sort_values(["geo", "ind", "year"])
    g = p.groupby(["geo", "ind"])
    p["U_lag"] = g.K_nonNA.shift()
    p.loc[g.year.diff() != 1, "U_lag"] = np.nan
    p["W5"] = g.W.shift(5)
    p["gW"] = (p.W / p.W5) ** (1 / 5) - 1
    p["gW"] = p.gW.fillna(p.groupby(["geo", "year"]).gW.transform("median"))
    p["cy"] = p.geo + p.year.astype(str)
    p["x0"] = np.log(p.K / p.W)
    p["low"] = p.x0 < p.groupby("cy").x0.transform("median")
    return p


def make_a(D="consistent", types=None, delta=None):
    def f(d, m):
        if types is None:
            U, I, Ul = d.K_nonNA, d.I_nonNA, d.U_lag
        else:
            U = sum(d[f"K_{t}"] for t in types)
            I = sum(d[f"I_{t}"] for t in types)
            Ul = U.groupby([d.geo, d.ind]).shift()
        dep = (I - (U - Ul)) if D == "consistent" else delta * Ul
        K2 = d.K + m * U
        return (d.PI + m * (I - dep)) / K2, np.log(K2 / d.W)
    return f


def make_b(g="observed"):
    def f(d, m):
        U = m * d.W * d.low
        gg = d.gW if g == "observed" else g
        K2 = d.K + U
        return (d.PI + gg * U) / K2, np.log(K2 / d.W)
    return f


def part1_klems():
    p = klems_panel()
    base = p[p.K_nonNA.notna() & p.I_nonNA.notna() & p.U_lag.notna()].copy()     # common sample for (a)
    res, curves = [], []
    b0, p0, _ = slope(base.assign(_r=base.r, _x=base.x0), "_r", "_x")
    specs = {"a: non-NA, D = I - dU": make_a(),
             "a: non-NA, delta 0.05": make_a("delta", delta=0.05),
             "a: non-NA, delta 0.15": make_a("delta", delta=0.15),
             "a: non-NA, delta 0.20": make_a("delta", delta=0.20),
             "a: OrgCap + Brand": make_a(types=["OrgCap", "Brand"]),
             "a: Train": make_a(types=["Train"])}
    for name, fn in specs.items():
        t, m0, ms = threshold(base, fn)
        curves.append(t.assign(spec=name, panel="KLEMS"))
        res.append(dict(panel="KLEMS", spec=name, beta0=b0, p0=p0, m_zero=m0, m_insig=ms, n=len(base)))
    allp = p[p.K.notna()].copy()
    b0b, p0b, _ = slope(allp.assign(_r=allp.r, _x=allp.x0), "_r", "_x")
    lows = allp[allp.low & allp.K_nonNA.notna()]
    m_obs_b = (lows.groupby("cy").K_nonNA.sum() / lows.groupby("cy").W.sum()).median()
    for g in ("observed", 0.0, 0.02, 0.05):
        t, m0, ms = threshold(allp, make_b(g))
        name = f"b: U = m W in low-K/W, g = {g}"
        curves.append(t.assign(spec=name, panel="KLEMS"))
        res.append(dict(panel="KLEMS", spec=name, beta0=b0b, p0=p0b, m_zero=m0, m_insig=ms, n=len(allp),
                        m_obs_b=m_obs_b))
    return pd.DataFrame(res), pd.concat(curves), p


# ---------------------------------------------------------------- WIOD panel
FINE_ORDER = ["A", "B", "C10-C12", "C13-C15", "C16-C18", "C19", "C20", "C21", "C22-C23", "C24-C25", "C26", "C27",
              "C28", "C29-C30", "C31-C33", "D", "E", "F", "G45", "G46", "G47", "H49", "H50", "H51", "H52", "H53", "I",
              "J58-J60", "J61", "J62-J63", "K", "L", "M", "N", "O", "P", "Q", "R", "S"]


def wiod_to_klems(code):
    dv = divs(code)
    for k in FINE_ORDER:
        if dv and dv <= divs(k):
            return k
    return None


def part1_wiod(klems):
    w = r2_panel()
    w = w[~w.rent & (w.r.abs() <= 2)].copy()
    w["ind"] = w.code
    w["kind"] = w.code.map(wiod_to_klems)
    w["geo2"] = w.country.map(ISO3)
    ratio = klems.assign(q=klems.K_nonNA / klems.K)[["geo", "ind", "year", "q"]].dropna()
    w = w.merge(ratio.rename(columns={"geo": "geo2", "ind": "kind"}), on=["geo2", "kind", "year"], how="left")
    med = ratio.groupby(["ind", "year"]).q.median().rename("qmed").reset_index().rename(columns={"ind": "kind"})
    w = w.merge(med, on=["kind", "year"], how="left")
    w["q"] = w.q.fillna(w.qmed)
    w = w[w.q.notna()].copy()
    w["W"], w["x0"] = w.LAB, np.log(w.K / w.LAB)
    w["low"] = w.x0 < w.groupby("cy").x0.transform("median")
    w = w.sort_values(["country", "ind", "year"])
    w["gW"] = w.groupby(["country", "ind"]).W.pct_change().rolling(5, min_periods=1).mean()
    w["gW"] = w.gW.fillna(w.groupby("cy").gW.transform("median")).fillna(0.04)
    res, curves = [], []
    b0, p0, _ = slope(w.assign(_r=w.r, _x=w.x0), "_r", "_x")
    for name, fn in {
        # (a) observed intensity, steady-state profit correction with the panel's own growth of U ~ g of W
        "a: KLEMS intensity, profit + g U": lambda d, m: ((d.PI + d.gW * m * d.q * d.K) / (d.K * (1 + m * d.q)),
                                                          np.log(d.K * (1 + m * d.q) / d.W)),
        "a: KLEMS intensity, no profit correction": lambda d, m: (d.PI / (d.K * (1 + m * d.q)),
                                                                  np.log(d.K * (1 + m * d.q) / d.W)),
        "b: U = m W in low-K/W, g observed": make_b("observed"),
    }.items():
        t, m0, ms = threshold(w, fn)
        curves.append(t.assign(spec=name, panel="WIOD"))
        res.append(dict(panel="WIOD", spec=name, beta0=b0, p0=p0, m_zero=m0, m_insig=ms, n=len(w)))
    return pd.DataFrame(res), pd.concat(curves), w


# ---------------------------------------------------------------- other checks (1.3)
def other_checks(p, w):
    rows = []
    base = p[p.K_nonNA.notna()].copy()
    # 1. low-intangible subsample
    for name, d in (("KLEMS", p), ("WIOD", w)):
        key = d.ind if name == "KLEMS" else d.kind
        s = d[key.isin(LOW_INTANG)]
        b, pv, _ = slope(s.assign(_r=s.r, _x=s.x0), "_r", "_x")
        rows.append(dict(check="low-intangible subsample", panel=name, beta=b, p=pv, n=len(s)))
        # 2. within country-industry over time
        d2 = d.assign(ci=(d.geo if name == "KLEMS" else d.country) + "|" + d.ind)
        b, pv, _ = slope(d2.assign(_r=d2.r, _x=d2.x0), "_r", "_x", fe=["cy", "ci"])
        rows.append(dict(check="within country-industry (+ country-year)", panel=name, beta=b, p=pv, n=len(d2)))
    # 3. SG&A-like: (i) capital (profile a at m=1); (ii) unproductive cost: PI + I_nonNA, K unchanged
    b, pv, _ = slope(base.assign(_r=(base.PI + base.I_nonNA) / base.K, _x=base.x0), "_r", "_x")
    rows.append(dict(check="non-NA spending as unproductive cost (PI + I, K)", panel="KLEMS", beta=b, p=pv, n=len(base)))
    r1, x1 = make_a()(base[base.U_lag.notna()], 1.0)
    bb = base[base.U_lag.notna()]
    b, pv, _ = slope(bb.assign(_r=r1, _x=x1), "_r", "_x")
    rows.append(dict(check="non-NA as capital (m = 1, D = I - dU)", panel="KLEMS", beta=b, p=pv, n=len(bb)))
    # 4. R&D: correlation within country-year, and slope with K net of K_RD
    q = base[base.K_RD.notna()].copy()
    q["rd"] = q.K_RD / q.K
    dm = q[["rd", "x0"]] - q.groupby("cy")[["rd", "x0"]].transform("mean")
    rows.append(dict(check="corr(K_RD/K, ln K/W) within country-year", panel="KLEMS", beta=dm.rd.corr(dm.x0), p=np.nan, n=len(q)))
    Knr = q.K - q.K_RD
    ok = Knr > 0
    b, pv, _ = slope(q[ok].assign(_r=q.PI[ok] / Knr[ok], _x=np.log(Knr[ok] / q.W[ok])), "_r", "_x")
    rows.append(dict(check="K without R&D capital", panel="KLEMS", beta=b, p=pv, n=int(ok.sum())))
    b, pv, _ = slope(q.assign(_r=q.r, _x=q.x0), "_r", "_x")
    rows.append(dict(check="same sample, K with R&D (reference)", panel="KLEMS", beta=b, p=pv, n=len(q)))
    return pd.DataFrame(rows)


def oster_sensemakr(p):
    d = p[p.K_nonNA.notna() & p.K_RD.notna()].copy()
    d["intang"] = d.K_nonNA / (d.K + d.K_nonNA)
    d["rd"] = d.K_RD / d.K
    cols = ["r", "x0", "intang", "rd"]
    X = d[cols] - d.groupby("cy")[cols].transform("mean")
    short = sm.OLS(X.r, X[["x0"]]).fit()
    longf = sm.OLS(X.r, X[["x0", "intang"]]).fit()
    b_dot, r_dot = short.params.x0, short.rsquared
    b_t, r_t = longf.params.x0, longf.rsquared
    rmax = min(1.3 * r_t, 1.0)
    beta_star = b_t - 1.0 * (b_dot - b_t) * (rmax - r_t) / (r_t - r_dot) if r_t != r_dot else np.nan
    # Cinelli-Hazlett on the model r ~ x0 + rd (rd = benchmark covariate), classical SE
    m = sm.OLS(X.r, X[["x0", "rd"]]).fit()
    dfree = m.df_resid
    t = m.tvalues.x0
    f = t / np.sqrt(dfree)
    rv = 0.5 * (np.sqrt(f ** 4 + 4 * f ** 2) - f ** 2)
    r2yd = t ** 2 / (t ** 2 + dfree)                              # partial R2 of x0 with r
    dz = sm.OLS(X.x0, X[["rd"]]).fit()
    r2dz = dz.rsquared                                            # R2 of D on benchmark Z (given FE)
    r2yz = m.tvalues.rd ** 2 / (m.tvalues.rd ** 2 + dfree)        # partial R2 of Y on Z given D
    out = [dict(method="Oster", beta_short=b_dot, beta_long=b_t, R2_short=r_dot, R2_long=r_t, Rmax=rmax,
                beta_star=beta_star),
           dict(method="Cinelli-Hazlett", beta=m.params.x0, t=t, partial_R2_x=r2yd, RV_q1=rv, R2_D_on_RD=r2dz,
                partialR2_Y_on_RD=r2yz)]
    for k in (1, 2, 3):
        r2dz_b = k * r2dz / (1 - r2dz)
        if r2dz_b >= 1:
            out.append(dict(method=f"CH bound k={k}", note="infeasible (R2_DZ bound >= 1)"))
            continue
        r2zxj = k * r2dz ** 2 / ((1 - k * r2dz) * (1 - r2dz))
        r2yz_b = ((np.sqrt(k) + np.sqrt(r2zxj)) / np.sqrt(1 - r2zxj)) ** 2 * (r2yz / (1 - r2yz))
        r2yz_b = min(r2yz_b, 1.0)
        bias = m.bse.x0 * np.sqrt(dfree) * np.sqrt(r2yz_b * r2dz_b / (1 - r2dz_b))
        out.append(dict(method=f"CH bound k={k}", beta=m.params.x0, bias_max=bias,
                        beta_adj_towards_zero=m.params.x0 + np.sign(-m.params.x0) * bias,
                        R2_DU_bound=r2dz_b, R2_YU_bound=r2yz_b))
    return pd.DataFrame(out)


if __name__ == "__main__":
    rk, ck, p = part1_klems()
    print(rk.round(4).to_string(), flush=True)
    rw, cw, w = part1_wiod(p)
    print(rw.round(4).to_string(), flush=True)
    pd.concat([rk, rw]).to_csv(OUT / "p1_threshold.csv", index=False)
    pd.concat([ck, cw]).to_csv(OUT / "p1_curves.csv", index=False)
    oc = other_checks(p, w)
    oc.to_csv(OUT / "p1_checks.csv", index=False)
    print(oc.round(4).to_string(), flush=True)
    os_ = oster_sensemakr(p)
    os_.to_csv(OUT / "p1_oster_ch.csv", index=False)
    print(os_.round(4).to_string(), flush=True)
