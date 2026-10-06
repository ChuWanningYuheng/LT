"""Iteration 7: exploratory checks after the external project review (pre_registration_v7.md, journal 4).

K1  degree of transformation: ln(PI/W) on ln(K/W) (b = 0 values, b = 1 full equalisation), 90% CI, TOST.
K2  estimate, 90% CI and MDE at the observed volume (m = 1); monotone worst case (b').
K5  without 2020-2021 and without industries with owner labour.
Part 2 power: positive control (actual industry wages compressed to the spread of psi) against c2.
Outputs: results/v7/p1_extra.csv, p1_extra_curves.csv, p2_power.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .part1 import OUT, klems_panel, make_a, part1_wiod, slope, threshold
from .part2 import (bea, build_figaro, compress, industry_psi, label_psi, metrics_mat, oews_industry, psi_rowthorn,
                    Setup)

OWNER_K = {"M"}
OWNER_W = {"M69_M70", "M71", "M72", "M73", "M74_M75", "Q86"}


def est(d, r, x):
    b, p, fit = slope(d.assign(_r=r, _x=x), "_r", "_x")
    se = float(fit.bse["_x"])
    return b, se, p


def tost(b, se, lo, hi):
    """Equivalence p (max of the two one-sided tests); equivalence at 5% iff the 90% CI lies in [lo, hi]."""
    return float(max(stats.norm.sf((b - lo) / se), stats.norm.cdf((b - hi) / se)))


def k1(p, w):
    rows = []
    for name, d, PI, K in (("KLEMS", p, p.PI, p.K), ("WIOD", w, w.PI, w.K)):
        ok = (PI > 0) & (K > 0) & (d.W > 0)
        e = d[ok]
        b, se, pv = est(e, np.log(PI[ok] / e.W), np.log(K[ok] / e.W))
        rows.append(dict(check="K1 b: ln(PI/W) on ln(K/W)", sample=name, est=b, se=se, ci90_lo=b - 1.645 * se,
                         ci90_hi=b + 1.645 * se, p=pv, tost_full_eq=tost(b, se, 0.9, 1.1),
                         tost_values=tost(b, se, -0.1, 0.1), n=len(e)))
    a = p[p.U_lag.notna()].copy()
    r1, x1 = make_a()(a, 1.0)
    PI1 = r1 * (a.K + a.K_nonNA)
    ok = (PI1 > 0)
    e = a[ok]
    b, se, pv = est(e, np.log(PI1[ok] / e.W), x1[ok])
    rows.append(dict(check="K1 b: ln(PI/W) on ln(K/W)", sample="KLEMS, K + non-NA (m = 1)", est=b, se=se,
                     ci90_lo=b - 1.645 * se, ci90_hi=b + 1.645 * se, p=pv, tost_full_eq=tost(b, se, 0.9, 1.1),
                     tost_values=tost(b, se, -0.1, 0.1), n=len(e)))
    return rows


def k2(p):
    rows = []
    a = p[p.U_lag.notna()].copy()
    for m in (0.0, 1.0):
        r, x = make_a()(a, m)
        b, se, pv = est(a, r, x)
        rows.append(dict(check=f"K2 beta, profile (a), m = {m:g}", sample="KLEMS", est=b, se=se,
                         ci90_lo=b - 1.645 * se, ci90_hi=b + 1.645 * se, p=pv, mde80=2.8 * se, n=len(a)))
    return rows


def make_b2(d, m):
    U = m * d.W * d.wgt
    K2 = d.K + U
    return (d.PI + d.gW * U) / K2, np.log(K2 / d.W)


def b_prime(p):
    d = p.copy()
    d["wgt"] = 1 - d.groupby("cy").x0.rank(pct=True)
    t, m0, ms = threshold(d, make_b2)
    obs = d[d.K_nonNA.notna()]
    m_obs = (obs.groupby("cy").K_nonNA.sum() / (obs.W * obs.wgt).groupby(obs.cy).sum()).median()
    return t.assign(spec="b': U = m W (1 - rank K/W)"), dict(check="K2 monotone worst case (b')", sample="KLEMS",
                                                              m_zero=m0, m_insig=ms, m_obs=m_obs, n=len(d))


def k5(p, w):
    rows, curves = [], []
    subs = {"no 2020-2021": ~p.year.isin([2020, 2021]), "no owner-labour industries (M)": ~p.ind.isin(OWNER_K)}
    subs["both"] = subs["no 2020-2021"] & subs["no owner-labour industries (M)"]
    for name, mask in subs.items():
        d = p[mask]
        b, se, pv = est(d, d.r, d.x0)
        a = d[d.U_lag.notna()]
        t, m0, ms = threshold(a, make_a())
        curves.append(t.assign(spec=f"K5 {name}"))
        r1, x1 = make_a()(a, 1.0)
        b1, se1, p1 = est(a, r1, x1)
        rows.append(dict(check=f"K5 {name}", sample="KLEMS", est=b, se=se, p=pv, beta_m1=b1, p_m1=p1, m_zero=m0,
                         m_insig=ms, n=len(d)))
    d = w[~w.code.isin(OWNER_W)]
    b, se, pv = est(d, d.r, d.x0)
    rows.append(dict(check="K5 no owner-labour industries", sample="WIOD", est=b, se=se, p=pv, n=len(d)))
    return rows, curves


def power(o_path=OUT / "p2_occupations.csv"):
    o = pd.read_csv(o_path, index_col=0)
    ind = oews_industry()
    lam = pd.read_csv(OUT / "p2_test1.csv").set_index("variant").loc["main", "lambda_hat"]
    E = float(pd.read_csv(OUT / "p2_info.csv", index_col=0).iloc[:, 0]["E"])
    o["psi[Rowthorn]"] = psi_rowthorn(o["h[main]"], E)
    o["psi[Shaikh]"] = np.exp(lam * o["h[main]"])
    ef, _ = build_figaro("USA", 2022)
    eb, _ = bea.build(2022)
    rows = []
    for tname, e, kind in (("BEA 2022", eb, "bea"), ("FIGARO USA 2022", ef, "figaro")):
        S = Setup(e)
        m = S.base_mask
        l = e.l()
        wage = np.divide(e.labour_income / e.hours, e.labour_income.sum() / e.hours.sum(), out=np.ones(e.n),
                         where=e.hours > 0)
        for k in ("Rowthorn", "Shaikh"):
            pb, eb_ = industry_psi(o, ind, f"psi[{k}]")
            psi = label_psi(e.labels, pb, eb_, kind)
            pos = m & (l > 0) & (wage > 0)
            a = np.log(psi[pos]).std() / np.log(wage[pos]).std()
            wc = np.ones(e.n)
            wc[pos] = wage[pos] ** a
            for vname, wv in ((f"psi {k}", psi), (f"positive control: wages compressed to SD ln psi {k}", wc)):
                lr = l * wv
                _, c2, _ = compress(l, lr, m)
                mr = metrics_mat(S.z(lr[None])[:, m], S.x[m])
                mc = metrics_mat(S.z(c2[None])[:, m], S.x[m])
                mh = metrics_mat(S.z(l[None])[:, m], S.x[m])
                rows.append(dict(table=tname, vector=vname, sd_ln=float(np.log(wv[pos]).std()),
                                 mawd=mr["mawd"][0], d=mr["d"][0], c2_mawd=mc["mawd"][0], c2_d=mc["d"][0],
                                 hours_mawd=mh["mawd"][0], gain_vs_c2_mawd=mc["mawd"][0] - mr["mawd"][0],
                                 gain_vs_c2_d=mc["d"][0] - mr["d"][0],
                                 beats_c2_both=bool(mr["mawd"][0] < mc["mawd"][0] and mr["d"][0] < mc["d"][0])))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    p = klems_panel()
    _, _, w = part1_wiod(p)
    rows = k1(p, w) + k2(p)
    tb, rb = b_prime(p)
    rows.append(rb)
    r5, c5 = k5(p, w)
    rows += r5
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "p1_extra.csv", index=False)
    pd.concat([tb] + c5).to_csv(OUT / "p1_extra_curves.csv", index=False)
    print(res.round(4).to_string(), flush=True)
    pw = power()
    pw.to_csv(OUT / "p2_power.csv", index=False)
    print(pw.round(4).to_string())
