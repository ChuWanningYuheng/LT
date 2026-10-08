"""Iteration 10, stage 3: additions to 9b (pre_registration_v10.md, stage 3).

3.1 placebos for outcome 44 (WIOD 2000-2014): (a) inter-industry permutation of world hours, (b) random national
    vectors z_cj = lambda_bar_j * exp(eps_cj) with the cross-country dispersion of hours and of LAB; outcome 54
3.2 market devaluation of NFC nonfinancial assets by phase (Z.1) and recovery of the NFC profit rate (descriptive)
3.3 Moseley 1947-1979: lead of the unproductive share over the profit rate (F-tests, both directions)

Usage: PYTHONPATH=src python -P -m lts.v10.stage3 [placebo|phases|moseley]
Outputs: results/v10/s3_*.csv
"""
from __future__ import annotations

import sys
import zipfile

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from ..v6.series import ROOT
from ..v8.rule import decide

OUT = ROOT / "results" / "v10"
N_DRAW = 200


# ================================================================ 3.1
def run_placebo():
    from ..v9.stage34 import GOVL, YEARS, contents, mawd, qmap, world_industries, year_system
    rng = np.random.default_rng(54)
    real, perm, rh, rl = [], [], [], []
    for y in YEARS:
        A, cells = year_system(y)
        cont = contents(A, cells)
        W, _ = world_industries(cont, cells)
        keep = ~cells.code.isin(GOVL) & (cells.x > 0) & (cells.country != "ROW")
        C = cells[keep].copy()
        C["lam"] = cont["hours"][C.index.to_numpy()]
        C["lab"] = cont["lab"][C.index.to_numpy()]
        C["wh"] = C.code.map(W.hours)
        C["wl"] = C.code.map(W.lab)
        ok = (C.lam > 0) & (C.wh > 0)
        sig_h = np.log(C.lam / C.wh).where(ok).groupby(C.code).std()
        okl = (C.lab > 0) & (C.wl > 0)
        sig_l = np.log(C.lab / C.wl).where(okl).groupby(C.code).std()
        wvals = W.hours.to_numpy()
        wcodes = W.index.to_numpy()
        perms = [dict(zip(wcodes, wvals[rng.permutation(len(wvals))])) for _ in range(N_DRAW)]
        for c, g in C.groupby("country"):
            lam, wld, x = g.lam.to_numpy(), g.wh.to_numpy(), g.x.to_numpy()
            if (lam <= 0).any() or len(g) < 20 or not np.isfinite(wld).all():
                continue
            m_nat = mawd(lam, x)[0]
            real.append(dict(year=y, country=c, world_better=mawd(qmap(wld, lam), x)[0] < m_nat))
            codes = g.code.to_numpy()
            pb = [mawd(qmap(np.array([p[k] for k in codes]), lam), x)[0] < m_nat for p in perms]
            perm.append(dict(year=y, country=c, **{f"d{i}": v for i, v in enumerate(pb)}))
            sh = sig_h.reindex(codes).fillna(sig_h.median()).to_numpy()
            sl = sig_l.reindex(codes).fillna(sig_l.median()).to_numpy()
            eps = rng.normal(0, 1, (N_DRAW, len(codes)))
            for store, sg in ((rh, sh), (rl, sl)):
                fl = []
                for i in range(N_DRAW):
                    z = wld * np.exp(eps[i] * sg)
                    fl.append(mawd(qmap(wld, z), x)[0] < mawd(z, x)[0])
                store.append(dict(year=y, country=c, **{f"d{i}": v for i, v in enumerate(fl)}))
        print("placebo", y, flush=True)
    R = pd.DataFrame(real)
    maj = R.groupby("country").world_better.mean() > 0.5
    out = {"hours (actual)": maj}
    mats = {}
    for name, rows in (("permutation of world hours (a)", perm), ("random vector, hours dispersion (b)", rh),
                       ("random vector, LAB dispersion (b)", rl)):
        P = pd.DataFrame(rows)
        M = P.groupby("country")[[f"d{i}" for i in range(N_DRAW)]].mean() > 0.5
        M = M.reindex(maj.index)
        mats[name] = M.to_numpy(float)
    rows = [dict(item="hours (actual)", share=float(maj.mean()), n=len(maj))]
    for name, M in mats.items():
        rows.append(dict(item=name, share=float(M.mean()), share_draw_p05=float(np.quantile(M.mean(0), 0.05)),
                         share_draw_p95=float(np.quantile(M.mean(0), 0.95)), n=len(maj)))
    S = pd.DataFrame(rows)
    h = maj.to_numpy(float)
    Mh = mats["random vector, hours dispersion (b)"]
    est = float(h.mean() - Mh.mean())
    rb = np.random.default_rng(5454)
    bs = []
    for _ in range(2000):
        ii = rb.integers(0, len(h), len(h))
        bs.append(h[ii].mean() - Mh[ii].mean())
    lo, hi = np.quantile(bs, [0.05, 0.95])
    o54 = dict(outcome="54: share(hours) - share(random vector with hours dispersion), world beats national", est=est,
               ci90_lo=lo, ci90_hi=hi, n=len(h), theta0=0, delta=0.1, direction=">", label=decide(est, lo, hi, 0, 0.1, ">"))
    Ml = mats["random vector, LAB dispersion (b)"]
    el = float(h.mean() - Ml.mean())
    bs = []
    for _ in range(2000):
        ii = rb.integers(0, len(h), len(h))
        bs.append(h[ii].mean() - Ml[ii].mean())
    lo2, hi2 = np.quantile(bs, [0.05, 0.95])
    S = pd.concat([S, pd.DataFrame([dict(item="share(hours) - share(random vector, LAB dispersion)", share=el,
                                         share_draw_p05=lo2, share_draw_p95=hi2, n=len(h))])], ignore_index=True)
    S.to_csv(OUT / "s3_placebo44.csv", index=False)
    R.to_csv(OUT / "s3_placebo44_actual.csv", index=False)
    pd.DataFrame([o54]).to_csv(OUT / "s3_outcomes.csv", index=False)
    print(S.round(4).to_string(), "\n", o54, flush=True)


# ================================================================ 3.2
def run_phases():
    from ..v8.stage4 import US_PEAKS, series_all
    from ..v9b.stage1 import nipa
    z = zipfile.ZipFile(ROOT / "data" / "raw" / "v6" / "z1" / "z1_csv_files.zip")
    num = lambda c: pd.to_numeric(c.replace("ND", np.nan), errors="coerce")  # noqa: E731
    r = pd.read_csv(z.open("csv/S11_1_r.csv"))
    b = pd.read_csv(z.open("csv/S11_1_b.csv"))
    r["year"] = r.date.str[:4].astype(int)
    rev = num(r["FR102010005.Q"]).groupby(r.year).agg(["sum", "count"])
    rev = rev.loc[rev["count"] == 4, "sum"]
    b = b[b.date.str.endswith("Q4")]
    A = pd.Series(num(b["LM102010005.Q"]).to_numpy(), index=b.date.str[:4].astype(int))
    p = nipa("A191RD")
    infl = (p / p.shift(1) - 1)
    real = (rev / A.shift(1) - infl).dropna()
    rec = pd.read_csv(ROOT / "data" / "raw" / "v8" / "fred_USREC.csv")
    rec.columns = ["date", "rec"]
    rec = rec.assign(year=pd.to_datetime(rec.date).dt.year).groupby("year").rec.max()
    s = series_all()
    rn = s[(s.geo == "USA_NFC") & (s.variant == "main")].set_index("year").r
    bounds = [1950] + US_PEAKS + [2025]
    rows = []
    for a, bb in zip(bounds[:-1], bounds[1:]):
        if bb - a < 3:
            continue
        yrs = [y for y in range(a + 1, bb + 1) if y in real.index]
        ry = [y for y in yrs if rec.get(y, 0) == 1] or yrs[:1]
        if not ry:
            continue
        ymin = min(ry, key=lambda y: real.loc[y])
        rows.append(dict(start=a + 1, end=bb, recession_years=",".join(map(str, ry)), year_min=ymin,
                         depth_real_reval=float(real.loc[ymin]),
                         recovery_r=float(rn.get(ymin + 3, np.nan) - rn.get(ymin, np.nan))))
    P = pd.DataFrame(rows).dropna()
    rho, pv = stats.spearmanr(P.depth_real_reval, P.recovery_r)
    P.to_csv(OUT / "s3_phases_devaluation.csv", index=False)
    S = pd.DataFrame([dict(item="Spearman rho(depth of real revaluation, recovery of r NFC over 3 years)", rho=rho, p=pv,
                           n=len(P), note="M: rho < 0 (deeper devaluation -> stronger recovery); descriptive")])
    S.to_csv(OUT / "s3_phases_summary.csv", index=False)
    real.rename("real_revaluation_rate").to_csv(OUT / "s3_real_revaluation.csv")
    print(P.round(4).to_string(), "\n", S.to_string(), flush=True)


# ================================================================ 3.3
def granger(y, x, lags_x=3, lags_y=2):
    d = pd.DataFrame({"y": y, "x": x}).dropna()
    D = pd.DataFrame({"y": d.y.diff()})
    for k in range(1, lags_y + 1):
        D[f"y{k}"] = d.y.diff().shift(k)
    for k in range(1, lags_x + 1):
        D[f"x{k}"] = d.x.diff().shift(k)
    D = D.dropna()
    X = sm.add_constant(D.drop(columns="y"))
    f = sm.OLS(D.y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
    names = [f"x{k}" for k in range(1, lags_x + 1)]
    R = np.zeros((lags_x, X.shape[1]))
    for i, n in enumerate(names):
        R[i, list(X.columns).index(n)] = 1
    w = f.wald_test(R, scalar=True, use_f=True)
    s = f.params[names].sum()
    se = np.sqrt(np.ones(lags_x) @ f.cov_params().loc[names, names].to_numpy() @ np.ones(lags_x))
    return dict(F=float(w.statistic), p=float(w.pvalue), sum_lags=float(s), se_sum=float(se), n=len(D),
                years=f"{D.index.min()}-{D.index.max()}")


def run_moseley():
    L = pd.read_csv(ROOT / "results" / "v9b" / "s1_long_series.csv")
    x = L[L.variant == "main"].set_index("year")
    ups = (x.comp_all - x.v) / x.ndp
    rows = []
    for a, b in ((1947, 1979), (1979, 2024), (1947, 2024)):
        w = slice(a, b)
        for rname in ("r", "r_m"):
            rows.append(dict(period=f"{a}-{b}", direction=f"d upsilon -> d {rname}", **granger(x[rname].loc[w], ups.loc[w])))
            rows.append(dict(period=f"{a}-{b}", direction=f"d {rname} -> d upsilon", **granger(ups.loc[w], x[rname].loc[w])))
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "s3_moseley_lead.csv", index=False)
    print(R.round(4).to_string(), flush=True)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    parts = sys.argv[1:] or ["moseley", "phases", "placebo"]
    if "moseley" in parts:
        run_moseley()
    if "phases" in parts:
        run_phases()
    if "placebo" in parts:
        run_placebo()
