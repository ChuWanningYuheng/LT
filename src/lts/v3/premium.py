"""Iteration 3, stage 2: wage bill = hours x skill wage x (1 + industry premium).

W_j = H_j q_j omega (1 + pi_j), q_j from variant 1.1a (education priced at national relative earnings;
1.2b as a fallback for JPN, which has no education shares, A-V3-PREM-JPN), omega such that
sum H q omega = sum W.  W = labour income (D1 + imputed self-employed), as for the wage-bill vector.
Profitability: PCM_j = (VA_j - D1_j)/x_j (pre-registered primary; also with labour income),
NOS_j/CFC_j = (VA_j - D1_j - CFC_j)/CFC_j.
2a: weighted correlation premium ~ PCM per country-year; pooled WLS with country-year FE, SE clustered
    by industry.
2b: per country, pooled over years with year FE, WLS (output weights), SE clustered by industry:
    y = ln(1/z^q) on x1 = ln(z^W/z^q) (vertically integrated premium) and x2 = ln(Gamma/z^q), Gamma the
    vertically integrated gross operating surplus (VA - D1) per unit of output.
Outputs results/v3/premium_industry.csv, premium_2a.csv, premium_2b.csv, premium_country_year.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from ..economy import build
from . import weights as W
from .placebo_disp import COUNTRIES, OUT, YEARS, Setup


def skill_weights(e, c, y):
    w, src = W.w_edu_price(e, c, y)
    if w is None:
        w, src = W.w_jobzone_price(e, c, y)
        src = "1.2b fallback: " + src
    return w, src


def build_rows(countries=COUNTRIES, years=YEARS):
    rows, cy = [], []
    for c in countries:
        for y in years:
            e, _ = build(c, y)
            S = Setup(e)
            m = S.base_mask & (e.hours > 0) & (e.labour_income > 0)
            q, src = skill_weights(e, c, y)
            H, Wb = e.hours, e.labour_income
            omega = Wb.sum() / (H * q).sum()
            prem = np.log(np.divide(Wb, H * q * omega, out=np.ones(e.n), where=(H > 0) & (Wb > 0)))
            pcm = (e.va - e.wages) / e.x
            pcm_li = (e.va - e.labour_income) / e.x
            nos_cfc = np.divide(e.va - e.wages - e.cfc, e.cfc, out=np.full(e.n, np.nan), where=e.cfc > 0)
            zq = S.z((e.l(q))[None])[0]
            zW = S.z((e.l(W.wage_rel(e)))[None])[0]
            gos = np.clip(e.va - e.wages, 1e-9, None) / e.x
            Gam = gos @ S.L                                       # vertically integrated GOS per unit
            # normalise z on evaluated industries (as in the metrics)
            nq = zq * e.x[S.base_mask].sum() / (zq[S.base_mask] @ e.x[S.base_mask])
            nW = zW * e.x[S.base_mask].sum() / (zW[S.base_mask] @ e.x[S.base_mask])
            for j in np.where(m)[0]:
                rows.append(dict(country=c, year=y, industry=e.labels[j], x=e.x[j], hours=H[j], q=q[j],
                                 premium=prem[j], pcm=pcm[j], pcm_li=pcm_li[j], nos_cfc=nos_cfc[j],
                                 y=np.log(1 / nq[j]), x1=np.log(nW[j] / nq[j]), x2=np.log(Gam[j] / nq[j])))
            hw = H[m] / H[m].sum()
            lw = np.log(Wb[m] / H[m])
            cy.append(dict(country=c, year=y, skill_source=src,
                           sd_premium=float(np.sqrt(hw @ (prem[m] - hw @ prem[m]) ** 2)),
                           sd_logwage=float(np.sqrt(hw @ (lw - hw @ lw) ** 2)),
                           sd_logwage_d1=float(np.sqrt(hw @ (np.log(e.wages[m] / H[m]) - hw @ np.log(e.wages[m] / H[m])) ** 2)),
                           pcm_mean=float((e.va[m] - e.wages[m]).sum() / e.x[m].sum()),
                           import_share=float((e.Am.sum(0) * e.x).sum() / e.x.sum()),
                           share_BKL=float(sum(e.x[i] for i, lab in enumerate(e.labels)
                                               if any(p in {"B", "K64", "K65", "K66", "L"} for p in lab.split("+")))
                                           / e.x.sum())))
            print("premium", c, y, flush=True)
    return pd.DataFrame(rows), pd.DataFrame(cy)


def wcorr(a, b, w):
    w = w / w.sum()
    a, b = a - w @ a, b - w @ b
    return float((w * a * b).sum() / np.sqrt((w * a * a).sum() * (w * b * b).sum()))


def demean(df, cols, by, w):
    out = df.copy()
    for col in cols:
        m = df.groupby(by).apply(lambda g: np.average(g[col], weights=g[w]), include_groups=False)
        out[col] = df[col] - df.set_index(by).index.map(m).values
    return out


def fe_wls(df, ycol, xcols, by, w="x", cluster="industry"):
    d = df.dropna(subset=[ycol] + xcols).copy()
    d = demean(d, [ycol] + xcols, by, w)
    X = d[xcols].values
    fit = sm.WLS(d[ycol].values, X, weights=d[w].values).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(d[cluster])[0]})
    # within R^2 (weighted)
    r2 = 1 - (d[w] * fit.resid ** 2).sum() / (d[w] * d[ycol] ** 2).sum()
    return fit, float(r2), len(d)


def analyse(ind: pd.DataFrame):
    ind = ind.copy()
    ind["cy"] = ind.country + "_" + ind.year.astype(str)
    ind["xw"] = ind.groupby("cy").x.transform(lambda v: v / v.sum())
    r2a = []
    for (c, y), g in ind.groupby(["country", "year"]):
        r2a.append(dict(country=c, year=y, corr_pcm=wcorr(g.premium.values, g.pcm.values, g.x.values),
                        corr_pcm_li=wcorr(g.premium.values, g.pcm_li.values, g.x.values),
                        corr_nos_cfc=wcorr(g.premium.values, g.nos_cfc.fillna(g.nos_cfc.median()).values, g.x.values)))
    per_cy = pd.DataFrame(r2a)
    pooled = []
    for xv in ("pcm", "pcm_li", "nos_cfc"):
        d = ind[np.isfinite(ind[xv])]
        if xv == "nos_cfc":
            d = d[d.nos_cfc.between(d.nos_cfc.quantile(0.01), d.nos_cfc.quantile(0.99))]
        fit, r2, n = fe_wls(d, "premium", [xv], "cy", w="xw")
        pooled.append(dict(x=xv, beta=fit.params[0], se=fit.bse[0], p=fit.pvalues[0], r2_within=r2, n=n))
    # 2b per country
    res = []
    for c, g in ind.groupby("country"):
        g = g.assign(yr=g.year.astype(str))
        f12, r12, n = fe_wls(g, "y", ["x1", "x2"], "yr")
        f1, r1, _ = fe_wls(g, "y", ["x1"], "yr")
        f2, r2, _ = fe_wls(g, "y", ["x2"], "yr")
        res.append(dict(country=c, n=n, b1_alone=f1.params[0], b1=f12.params[0], p1=f12.pvalues[0],
                        b2=f12.params[1], p2=f12.pvalues[1], r2_x1=r1, r2_x2=r2, r2_both=r12,
                        share_x1_unique=(r12 - r2) / r12 if r12 > 0 else np.nan,
                        b1_drop=1 - f12.params[0] / f1.params[0]))
    return per_cy, pd.DataFrame(pooled), pd.DataFrame(res)


def run():
    ind, cy = build_rows()
    ind.to_csv(OUT / "premium_industry.csv", index=False)
    cy.to_csv(OUT / "premium_country_year.csv", index=False)
    per_cy, pooled, res = analyse(ind)
    per_cy.to_csv(OUT / "premium_2a_corr.csv", index=False)
    pooled.to_csv(OUT / "premium_2a_pooled.csv", index=False)
    res.to_csv(OUT / "premium_2b.csv", index=False)


if __name__ == "__main__":
    run()
