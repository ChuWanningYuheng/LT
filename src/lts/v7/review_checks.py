"""Iteration 7: exploratory checks requested by the independent review (pre_registration_v7.md, journal 6).

R4   set of m compatible with beta = 0 (profile (a)).
R2   threshold (a) in log form: b(m) under three treatments of losses; smallest m with 90% CI of b above 0.9.
R3   SEC profile (gamma EPW, delta 0.20) carried over to all countries of the KLEMS panel.
R6   Cinelli-Hazlett robustness value from the clustered t.
R15  corr(K_NonNatAcc / K, ln K/W) within country-year.
R12  R2 of the strict Shaikh form (no constant).
Outputs: results/v7/p1_review.csv, p1_review_curves.csv, p2_review.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .part1 import GRID, OUT, klems_panel, make_a, slope, threshold
from .sec import RAW, ROOT, profile_v


def ci_set(t):
    ok = t[t.p >= 0.05]
    return dict(m_compat_min=ok.m.min(), m_compat_max=ok.m.max(), grid_end_reached=bool(ok.m.max() >= GRID.max()))


def full_curve(d, make, grid=GRID):
    """beta(m) on the whole grid, without the stopping rule of part1.threshold."""
    rows = []
    for m in grid:
        r, x = make(d, m)
        e = d.assign(_r=r, _x=x)
        e = e[np.isfinite(e._r) & np.isfinite(e._x)]
        b, pv, fit = slope(e, "_r", "_x")
        rows.append(dict(m=m, beta=b, p=pv, lo95=b - 1.96 * fit.bse["_x"], hi95=b + 1.96 * fit.bse["_x"], n=len(e)))
    return pd.DataFrame(rows)


def log_b(a, m, how):
    r, x = make_a()(a, m)
    K2 = np.exp(x) * a.W
    pw = r * K2 / a.W                                       # PI_m / W
    d = a.assign(_x=x, _pw=pw)
    if how == "positive":
        d = d[d._pw > 0]
        y = np.log(d._pw)
    elif how == "p1 replace":
        floor = d._pw.where(d._pw > 0).groupby(d.cy).transform(lambda s: s.quantile(0.01))
        y = np.log(d._pw.where(d._pw > 0, floor))
    else:
        y = np.arcsinh(d._pw)
    d = d.assign(_y=y)
    d = d[np.isfinite(d._y) & np.isfinite(d._x)]
    b, p, fit = slope(d, "_y", "_x")
    return b, float(fit.bse["_x"]), len(d)


def log_threshold(a):
    rows = []
    for how in ("positive", "p1 replace", "asinh"):
        for m in np.round(np.arange(0, 20.0001, 0.25), 2):
            b, se, n = log_b(a, m, how)
            rows.append(dict(spec=f"log form, {how}", m=m, b=b, se=se, lo90=b - 1.645 * se, n=n))
    return pd.DataFrame(rows)


def sec_full(p):
    firms = pd.read_csv(RAW / "sec_firm_fy.csv")
    epw = pd.read_csv(ROOT / "data" / "raw" / "v7" / "epw" / "capital_accum_parameters_2023.csv")
    prof = profile_v(firms, epw).groupby("ind")[["PPE", "gSGA"]].sum()
    q = prof.gSGA / (0.04 + 0.20) / prof.PPE
    d = p[p.ind.isin(q.index)].copy()
    d["q"] = d.ind.map(q)

    def fn(e, m):
        K2 = e.K * (1 + m * e.q)
        return (e.PI + 0.04 * m * e.q * e.K) / K2, np.log(K2 / e.W)

    _, m0, ms = threshold(d, fn)
    t = full_curve(d, fn)
    t1 = t.set_index("m")
    return t.assign(spec="SEC EPW d0.20, all countries"), dict(
        check="R3 SEC profile, all countries", beta0=t1.beta.iloc[0], p0=t1.p.iloc[0],
        beta_m1=t1.beta.get(1.0), p_m1=t1.p.get(1.0), m_zero=m0, m_insig=ms, n=len(d),
        median_U_over_K=float(q.median()), **ci_set(t))


def rv_clustered(p):
    d = p[p.K_nonNA.notna() & p.K_RD.notna()].copy()
    d["rd"] = d.K_RD / d.K
    b, pv, fit = slope(d, "r", "x0", ctrl=("rd",))
    t = fit.tvalues["x0"]
    f = t / np.sqrt(fit.df_resid)
    return dict(check="R6 CH robustness value, clustered t", est=b, t=float(t), RV=float(0.5 * (np.sqrt(f ** 4 + 4 * f ** 2) - f ** 2)),
                n=len(d))


def corr_share(p):
    d = p[p.K_nonNA.notna()].copy()
    d["sh"] = d.K_nonNA / d.K
    X = d[["sh", "x0"]] - d.groupby("cy")[["sh", "x0"]].transform("mean")
    return dict(check="R15 corr(K_nonNA/K, ln K/W) within country-year", est=float(X.corr().iloc[0, 1]), n=len(d))


def strict_shaikh():
    o = pd.read_csv(OUT / "p2_occupations.csv", index_col=0)
    rows = []
    for k in [c[2:-1] for c in o.columns if c.startswith("h[")]:
        d = o[o[f"h[{k}]"].notna()]
        h, y, w = d[f"h[{k}]"], d.ln_obs, d.emp
        lam = sm.WLS(y, h, weights=w).fit().params.iloc[0]
        ybar = (w * y).sum() / w.sum()
        r2 = 1 - (w * (y - lam * h) ** 2).sum() / (w * (y - ybar) ** 2).sum()
        rows.append(dict(variant=k, lambda_strict=lam, R2_strict=r2))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    p = klems_panel()
    a = p[p.U_lag.notna()].copy()
    rows, curves = [], []
    _, m0, ms = threshold(a, make_a())
    t = full_curve(a, make_a())
    curves.append(t.assign(spec="profile (a), full grid"))
    rows.append(dict(check="R4 profile (a): m compatible with beta = 0", m_zero=m0, m_insig=ms, n=len(a),
                     beta_max=t.beta.max(), lo95_at_m20=t.lo95.iloc[-1], **ci_set(t)))
    lt = log_threshold(a)
    curves.append(lt)
    for spec, g in lt.groupby("spec"):
        g = g.set_index("m")
        above = g[g.lo90 >= 0.9]
        rows.append(dict(check=f"R2 {spec}", b_m0=g.b[0.0], b_m1=g.b[1.0], b_m3_5=g.b[3.5], se_m1=g.se[1.0],
                         m_lo90_ge_0_9=above.index.min() if len(above) else np.nan, n=g.n[0.0]))
    sc, sr = sec_full(p)
    curves.append(sc)
    rows += [sr, rv_clustered(p), corr_share(p)]
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "p1_review.csv", index=False)
    pd.concat(curves).to_csv(OUT / "p1_review_curves.csv", index=False)
    S = strict_shaikh()
    S.to_csv(OUT / "p2_review.csv", index=False)
    pd.set_option("display.width", 250)
    print(R.round(4).to_string())
    print(S.round(4).to_string())
