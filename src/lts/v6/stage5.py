"""Iteration 6, stage 5 (pre_registration_v6.md §5 and journal 7): profit-rate equalisation.

5.1 advanced capital K_adv = K + tau (II + W): proportionality of industry profit shares to K, K_adv(tau), GO, W,
    costs (II + W + CFC); placebos for K_adv; lambda model of iteration 4 with competitor K.
5.2 average r = PI_t / K_{t-1} vs incremental r_inc = dPI_t / I_{t-1} (and lag-2 version): dispersion,
    pooled AR(1) half-life, Mueller long-run deviations, placebo (dPI permuted across industries).
5.3 US corporations: all corporations; NFC with dividend and interest income (trends, Marx decomposition).
Outputs: results/v6/s5_*.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ..v3.crosscountry import holm
from ..v4 import b_data
from ..v4.b_tests import lam_fit, load, sample, shares
from .series import OUT, RAW, fa, finish, nipa
from .tests import contrib, logs, trend

TAUS = {"0.1": 0.1, "0.127 (US)": 0.127, "0.25": 0.25, "0.5": 0.5}


# ---------------------------------------------------------------- 5.1
def r2_prop(pi, s, cy):
    df = pd.DataFrame(dict(cy=cy, pi=pi, s=s))
    r = df.groupby("cy").apply(lambda g: 1 - ((g.pi - g.s) ** 2).sum() / ((g.pi - g.pi.mean()) ** 2).sum(),
                               include_groups=False)
    return float(np.median(r))


def s51(n_p=1000, seed=11):
    p = load("primary")
    d = sample(p)
    d = d[d.II.notna()]
    d = d[d.groupby(["geo", "year"]).ind.transform("count") >= 8].copy()
    d["cy"] = d.geo + d.year.astype(str)
    d["costs"] = d.II + d.Wv + d.CFC
    for name, t in TAUS.items():
        d[f"Kadv_{name}"] = d.Kv + t * (d.II + d.Wv)
    pi, k, cy = shares(d, "PIv"), shares(d, "Kv"), d.cy.to_numpy()
    rows = []
    srcs = {"K": "Kv", **{f"K_adv τ={n}": f"Kadv_{n}" for n in TAUS}, "GO": "GO", "W": "Wv", "costs": "costs"}
    for name, col in srcs.items():
        s = shares(d, col)
        lam, r2l = lam_fit(pi, k, s, cy)
        rows.append(dict(source=name, r2_prop=r2_prop(pi, s, cy), lam_vs_K=lam, r2_lam=r2l, n=len(d),
                         country_years=d.cy.nunique()))
    res = pd.DataFrame(rows)
    # placebos for K_adv (main tau = 0.25)
    col = "Kadv_0.25"
    rng = np.random.default_rng(seed)
    coef = (d[col] / d.GO).to_numpy()
    lsd = np.log(d[col] / d.GO).groupby(d.cy).transform("std").to_numpy()
    GO = d.GO.to_numpy()
    geo_inds = {g: sorted(gg.ind.unique()) for g, gg in d.groupby("geo")}
    key = pd.Series(coef, index=pd.MultiIndex.from_arrays([d.cy, d.ind]))
    med = pd.Series(coef).groupby(cy).transform("median").to_numpy()
    pos = {g: {i: j for j, i in enumerate(li)} for g, li in geo_inds.items()}
    pa, pd_ = [], []
    for _ in range(n_p):
        pm = {g: dict(zip(li, np.array(li)[rng.permutation(len(li))])) for g, li in geo_inds.items()}
        tgt = [pm[g][i] for g, i in zip(d.geo, d.ind)]
        v = key.reindex(pd.MultiIndex.from_arrays([d.cy, tgt])).to_numpy()
        v = np.where(np.isfinite(v), v, med) * GO
        pa.append(r2_prop(pi, v / pd.Series(v).groupby(cy).transform("sum").to_numpy(), cy))
        eps = {g: rng.normal(0, 1, len(li)) for g, li in geo_inds.items()}
        e = np.array([eps[g][pos[g][i]] for g, i in zip(d.geo, d.ind)])
        vd = np.exp(lsd * e) * GO
        pd_.append(r2_prop(pi, vd / pd.Series(vd).groupby(cy).transform("sum").to_numpy(), cy))
    r2k = res.loc[res.source == "K_adv τ=0.25", "r2_prop"].iloc[0]
    plac = pd.DataFrame([dict(placebo="a_perm", median_r2=np.median(pa), share_ge=np.mean(np.array(pa) >= r2k)),
                         dict(placebo="d_lnorm", median_r2=np.median(pd_), share_ge=np.mean(np.array(pd_) >= r2k))])
    return res, plac


# ---------------------------------------------------------------- 5.2
def klems_panel_1995():
    f = OUT / "klems_panel_1995.csv.gz"
    if f.exists():
        return pd.read_csv(f)
    orig = b_data.na_items

    def fixed():
        n = orig()
        n["nace"] = n.nace.replace({"R_S_-": "R_S_T", "_-": "_T"})   # iteration-4 code bug (report v5, journal 7)
        return n
    b_data.na_items = fixed
    p = b_data.build_panel(years=range(1995, 2022))
    b_data.na_items = orig
    ca = pd.read_csv(RAW / "euklems" / "capital_accounts.csv", usecols=["nace_r2_code", "geo_code", "year", "I_GFCF"],
                     low_memory=False).rename(columns={"nace_r2_code": "ind", "geo_code": "geo"})
    p = p.merge(ca, on=["geo", "ind", "year"], how="left")
    p.to_csv(f, index=False)
    return p


def mueller(d, col):
    """per-industry AR(1) of deviations with intercept: long-run deviation LR = a/(1-l) and delta-method p."""
    rows = []
    for (g, i), x in d.groupby(["geo", "ind"]):
        x = x.sort_values("year")
        y, ylag = x[col].to_numpy()[1:], x[col].to_numpy()[:-1]
        ok = np.isfinite(y) & np.isfinite(ylag) & (np.diff(x.year.to_numpy()) == 1)
        y, ylag = y[ok], ylag[ok]
        if len(y) < 10:
            continue
        X = np.column_stack([np.ones(len(y)), ylag])
        b, res_, *_ = np.linalg.lstsq(X, y, rcond=None)
        e = y - X @ b
        V = np.linalg.inv(X.T @ X) * (e @ e) / (len(y) - 2)
        a, l = b
        if abs(1 - l) < 1e-6:
            continue
        lr = a / (1 - l)
        grad = np.array([1 / (1 - l), a / (1 - l) ** 2])
        se = float(np.sqrt(grad @ V @ grad))
        rows.append(dict(geo=g, ind=i, lam=l, lr=lr, se=se, p=2 * stats.norm.sf(abs(lr / se)), n=len(y)))
    r = pd.DataFrame(rows)
    r["p_holm"] = r.index.map(holm(dict(zip(r.index, r.p))))
    return r


def pooled_ar1(d, col):
    x = d.sort_values(["geo", "ind", "year"]).copy()
    g = x.groupby(["geo", "ind"])
    x["lag"] = g[col].shift()
    x = x[(g.year.diff() == 1) & x[col].notna() & x.lag.notna()]
    # industry fixed effects (within transformation)
    for c in (col, "lag"):
        x[c + "_w"] = x[c] - x.groupby(["geo", "ind"])[c].transform("mean")
    lam = float((x[col + "_w"] * x["lag_w"]).sum() / (x["lag_w"] ** 2).sum())
    lam_nofe = float((x[col] * x["lag"]).sum() / (x["lag"] ** 2).sum())
    hl = lambda l: np.log(0.5) / np.log(l) if 0 < l < 1 else np.nan   # noqa: E731
    return dict(lam_fe=lam, half_life_fe=hl(lam), lam_nofe=lam_nofe, half_life_nofe=hl(lam_nofe), n=len(x))


def s52(seed=13):
    p = klems_panel_1995()
    p = p[p.cfc_sh.notna() & (p.K > 0) & ~p.gov.astype(bool) & ~p.rent.astype(bool)].copy()
    p = p.sort_values(["geo", "ind", "year"])
    g = p.groupby(["geo", "ind"])
    cons = g.year.diff() == 1
    p["K_lag"], p["I_lag"], p["PI_lag"] = g.K.shift(), g.I_GFCF.shift(), g.PI.shift()
    p["I_lag2"], p["PI_lag2"] = g.I_GFCF.shift(2), g.PI.shift(2)
    p["r"] = np.where(cons, p.PI / p.K_lag, np.nan)
    ok = cons & (p.I_lag >= 0.01 * p.K_lag)
    p["r_inc"] = np.where(ok, (p.PI - p.PI_lag) / p.I_lag, np.nan)
    ok2 = ok & (g.year.diff(2) == 2)
    p["r_inc2"] = np.where(ok2, (p.PI - p.PI_lag2) / (p.I_lag + p.I_lag2), np.nan)
    # placebo: dPI permuted across industries within country-year
    rng = np.random.default_rng(seed)
    p["dPI"] = p.PI - p.PI_lag
    p["dPI_perm"] = p.groupby(["geo", "year"]).dPI.transform(lambda s: rng.permutation(s.to_numpy()))
    p["r_inc_placebo"] = np.where(ok, p.dPI_perm / p.I_lag, np.nan)
    cy = [p.geo, p.year]
    for c in ("r_inc", "r_inc2", "r_inc_placebo"):                  # winsorise 1/99% within country-year
        lo, hi = p.groupby(cy)[c].transform(lambda s: s.quantile(0.01)), p.groupby(cy)[c].transform(lambda s: s.quantile(0.99))
        p[c] = p[c].clip(lo, hi)
    rows, mrows = [], []
    for c in ("r", "r_inc", "r_inc2", "r_inc_placebo"):
        dev = p[c] - p.groupby(cy)[c].transform("mean")
        p["dev_" + c] = dev
        sd = p.groupby(cy)[c].std().median()
        iqr = p.groupby(cy)[c].apply(lambda s: s.quantile(0.75) - s.quantile(0.25)).median()
        ar = pooled_ar1(p, "dev_" + c)
        m = mueller(p, "dev_" + c)
        m["measure"] = c
        mrows.append(m)
        rows.append(dict(measure=c, n_obs=int(p[c].notna().sum()), sd_median=sd, iqr_median=iqr, **ar,
                         mueller_n=len(m), share_lr_sig=(m.p < 0.05).mean(), share_lr_sig_holm=(m.p_holm < 0.05).mean()))
    res = pd.DataFrame(rows)
    # persistence of average rates over decades: industry mean r 1995-2005 vs 2011-2021
    a = p[p.year.between(1996, 2005)].groupby(["geo", "ind"]).r.mean()
    b = p[p.year.between(2011, 2021)].groupby(["geo", "ind"]).r.mean()
    j = pd.concat([a, b], axis=1, keys=["early", "late"]).dropna()
    dem = j - j.groupby(level=0).transform("mean")
    pers = dict(n=len(j), corr_raw=j.early.corr(j.late), corr_within_country=dem.early.corr(dem.late),
                spearman_within=dem.early.corr(dem.late, method="spearman"))
    return res, pd.concat(mrows), pers


# ---------------------------------------------------------------- 5.3
def s53():
    n = nipa(["W326RC", "W322RC", "A460RC", "A442RC", "B1444C", "B1135C", "B1144C", "A457RC", "A455RC", "B455RX"])
    base = pd.DataFrame({"PI": n.W326RC, "W": n.A460RC, "Y": n.A457RC, "VA": n.A455RC, "Q_Y": n.B455RX})
    base["K"] = fa(4, "FAAt401-A", 37)
    qi = fa(4, "FAAt402-A", 37)
    base["Q_K"] = qi * base.K.loc[2017] / qi.loc[2017]
    base["P_K"], base["P_Y"] = base.K / base.Q_K, base.VA / base.Q_Y
    base["H"] = np.nan
    specs = {
        "NFC (main)": base.copy(),
        "NFC + dividends + interest received": base.assign(PI=base.PI + n.B1444C + n.B1135C + n.B1144C),
        "all corporations": base.assign(PI=n.W322RC, W=n.A442RC, K=fa(4, "FAAt401-A", 17)),
    }
    rows, dec = [], []
    for name, d in specs.items():
        d = finish(d.loc[1951:2024].copy())
        a0 = int(d.PI.first_valid_index())              # dividends received by NFC start in 1958
        for series in ("r", "rM"):
            y = d[series].dropna()
            for lag in (2, 4, 8):
                for a, b in ((a0, 2024), (a0 + 5, 2024), (a0, 2019), (a0 + 5, 2019)):
                    yy = y.loc[a:b]
                    sl, se, pv = trend(yy.to_numpy(), lag)
                    rows.append(dict(spec=name, series=series, lag=lag, y0=a, y1=b, slope=sl, p=pv, mean=yy.mean()))
        g = logs(d)
        g["lntech"] = np.nan
        g["lnpkw"] = np.nan
        dec.append(dict(spec=name, **contrib(g, a0, 2024)))
        for a, b in ((a0, 1966), (1966, 1982), (1982, 2007), (2007, 2024)):
            dec.append(dict(spec=name, **contrib(g, a, b)))
    return pd.DataFrame(rows), pd.DataFrame(dec)


if __name__ == "__main__":
    tr, dc = s53()
    tr.to_csv(OUT / "s5_3_trends.csv", index=False)
    dc.to_csv(OUT / "s5_3_decomp.csv", index=False)
    print(tr[(tr.y0 == tr.groupby("spec").y0.transform("min")) & (tr.y1 == 2024)].round(4).to_string())
    print(dc[["spec", "y0", "y1", "r0", "r1", "dlnrM", "c_e", "c_1k"]].round(3).to_string(), flush=True)
    res, m, pers = s52()
    res.to_csv(OUT / "s5_2_summary.csv", index=False)
    m.to_csv(OUT / "s5_2_mueller.csv", index=False)
    pd.Series(pers).to_csv(OUT / "s5_2_persistence.csv")
    print(res.round(3).to_string())
    print(pers, flush=True)
    r1, pl = s51()
    r1.to_csv(OUT / "s5_1.csv", index=False)
    pl.to_csv(OUT / "s5_1_placebo.csv", index=False)
    print(r1.round(3).to_string())
    print(pl.round(3).to_string())
