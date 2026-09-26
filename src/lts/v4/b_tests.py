"""Iteration 4, part B tests (pre-registration v4, B3 + journal 6).

B3.1  r = PI/K on ln(K/W) and ln(K/H); country x year FE; SE clustered by industry code; controls
      risk (SD r over t-4..t), PCM = PI/GO, non-NA intangible share; trap variants.
B3.2  pi_j = lambda k_j + (1 - lambda) s_j (shares within country-year); pooled OLS without constant;
      industry-cluster bootstrap; sources: labour cost W (primary), hours, GO, VA (reference), flat (1/n),
      energy (FIGARO purchases of B, C19, D35), materials (II), placebos (a) perm H/GO, (b) lnorm sd ln H/GO,
      (c) mixtures of purchased products with labour's cost share, (d) lnorm sd ln W/GO; 1000 each.
B3.3  exploratory dynamics: d ln(H/GO_Q) (demeaned by country-year) -> r(t+h) - r(t), h = 1..5.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .. import figaro
from ..codes import divisions
from ..figaro import INDUSTRIES
from .b_data import OUT, divs

N_P, N_BOOT = 1000, 1000


# ------------------------------------------------------------------ sample construction
def load(variant="primary"):
    p = pd.read_csv(OUT / "b_panel.csv.gz")
    p = p[p.cfc_sh.notna() & (p.K > 0) & (p.W > 0) & (p.GO > 0) & (p.H > 0)].copy()
    p["Kv"], p["PIv"], p["Wv"] = p.K, p.PI, p.W
    if variant == "no_mi":
        p["PIv"], p["Wv"] = p.PI_nomi, p.COMP
    elif variant == "tangible":
        p["Kv"] = p.K_tang
    elif variant == "intangibles":
        dep = p.I_nonNA - p.groupby(["geo", "ind"]).K_nonNA.diff()
        p["Kv"] = p.K + p.K_nonNA
        p["PIv"] = p.PI + (p.VAadj - p.VA_ia) - dep
    elif variant == "cfc_as_capital":
        p["Kv"] = p.CFC
    p = p[(p.Kv > 0) & np.isfinite(p.PIv)]
    p["r"] = p.PIv / p.Kv
    return p


def sample(p, rent=False):
    s = p[~p.gov & (rent | ~p.rent)]
    out = s[s.r.abs() <= 2].copy()
    out.attrs["n_outliers"] = int((s.r.abs() > 2).sum())
    return out


# ------------------------------------------------------------------ B3.1
def demean(d, cols, by):
    return d[cols] - d.groupby(by)[cols].transform("mean")


def b31(p, rent=False, controls=False, iv=None, xname="kw"):
    d = sample(p, rent).sort_values(["geo", "ind", "year"])
    d["cy"] = d.geo + d.year.astype(str)
    d["x"] = np.log(d.Kv / (d.Wv if xname == "kw" else d.H))
    cols = ["r", "x"]
    if controls:
        g = d.groupby(["geo", "ind"]).r
        d["risk"] = g.transform(lambda s: s.rolling(5, min_periods=4).std())
        d["pcm"] = d.PIv / d.GO
        d["intang"] = (d.K_nonNA / (d.K + d.K_nonNA)).fillna(0)
        cols += ["risk", "pcm", "intang"]
    if iv == "other_countries":
        def loo_median(s):
            v = s.to_numpy()
            return pd.Series([np.median(np.delete(v, i)) if len(v) > 1 else np.nan for i in range(len(v))],
                             index=s.index)
        d["z"] = d.groupby(["ind", "year"], group_keys=False).x.apply(loo_median)   # median of other countries
        cols += ["z"]
    elif iv == "lag2":
        d["z"] = d.groupby(["geo", "ind"]).x.shift(2)
        cols += ["z"]
    d = d.dropna(subset=cols)
    X = demean(d, cols, "cy")
    ctrl = [c for c in cols if c not in ("r", "x", "z")]
    grp = pd.factorize(d.ind)[0]
    if iv is None:
        fit = sm.OLS(X.r, X[["x"] + ctrl]).fit(cov_type="cluster", cov_kwds={"groups": grp})
        b, se, pv = fit.params["x"], fit.bse["x"], fit.pvalues["x"]
    else:
        # 2SLS by hand with cluster-robust SE
        Z = X[["z"] + ctrl].values
        Xm = X[["x"] + ctrl].values
        Pz = Z @ np.linalg.pinv(Z.T @ Z) @ Z.T
        Xh = Pz @ Xm
        beta = np.linalg.solve(Xh.T @ Xm, Xh.T @ X.r.values)
        e = X.r.values - Xm @ beta
        A = np.linalg.inv(Xh.T @ Xm)
        meat = sum(np.outer(Xh[grp == g].T @ e[grp == g], Xh[grp == g].T @ e[grp == g]) for g in np.unique(grp))
        G = len(np.unique(grp))
        V = A @ meat @ A.T * G / (G - 1)
        b, se = beta[0], np.sqrt(V[0, 0])
        from scipy import stats
        pv = 2 * stats.t.sf(abs(b / se), G - 1)
        fs = sm.OLS(X.x, X[["z"] + ctrl]).fit()
    return dict(beta=float(b), se=float(se), p=float(pv), n=len(d), clusters=int(len(np.unique(grp))),
                n_outliers=d.attrs.get("n_outliers", np.nan) if hasattr(d, "attrs") else np.nan,
                first_stage_t=float(fs.tvalues["z"]) if iv is not None else np.nan)


def run_b31():
    rows = []
    specs = [("primary", False, False, None), ("primary", False, True, None), ("primary", True, False, None),
             ("primary", True, True, None), ("no_mi", False, False, None), ("no_mi", False, True, None),
             ("tangible", False, False, None), ("tangible", False, True, None),
             ("intangibles", False, False, None), ("intangibles", False, True, None),
             ("cfc_as_capital", False, False, None), ("cfc_as_capital", False, True, None),
             ("primary", False, False, "other_countries"), ("primary", False, True, "other_countries"),
             ("primary", False, False, "lag2"), ("primary", False, True, "lag2")]
    cache = {}
    for var, rent, ctrl, iv in specs:
        p = cache.setdefault(var, load(var))
        for xname in ("kw", "kh"):
            r = b31(p, rent, ctrl, iv, xname)
            rows.append(dict(variant=var, with_rent=rent, controls=ctrl, iv=iv or "", x=xname, **r))
    t = pd.DataFrame(rows)
    # Holm over the two primary regressors (primary, no rent, no controls, OLS)
    m = (t.variant == "primary") & ~t.with_rent & ~t.controls & (t.iv == "")
    ps = t.loc[m, "p"].sort_values()
    holm, run = {}, 0.0
    for i, (ix, v) in enumerate(ps.items()):
        run = max(run, min(1, (len(ps) - i) * v))
        holm[ix] = run
    t["p_holm"] = pd.Series(holm)
    t.to_csv(OUT / "b31.csv", index=False)
    return t


# ------------------------------------------------------------------ B3.2
def lam_fit(pi, k, s, cy):
    """pooled lambda (unconstrained) and median per-country-year R^2 (constrained lambda)."""
    a, b = pi - s, k - s
    lam = float((a * b).sum() / (b * b).sum())
    lc = min(max(lam, 0.0), 1.0)
    pred = lc * k + (1 - lc) * s
    df = pd.DataFrame(dict(cy=cy, pi=pi, pred=pred))
    r2 = df.groupby("cy").apply(lambda g: 1 - ((g.pi - g.pred) ** 2).sum() / ((g.pi - g.pi.mean()) ** 2).sum(),
                                include_groups=False)
    return lam, float(np.median(r2))


def shares(d, col):
    return (d[col] / d.groupby("cy")[col].transform("sum")).to_numpy()


@__import__("functools").lru_cache(None)
def figaro_use(year):
    """(countries, products x industries purchases by country) from FIGARO."""
    d = figaro.load(year)
    C = [str(c) for c in d["countries"]]
    N = len(INDUSTRIES)
    Z = d["Z"]
    U = {c: Z[:, i * N:(i + 1) * N].reshape(len(C), N, N).sum(0) for i, c in enumerate(C)}
    return U


def to_klems(vec_fig: np.ndarray, ind: str) -> float:
    """Sum a FIGARO-industry vector over the FIGARO industries inside a KLEMS industry."""
    kd = divs(ind)
    return float(sum(v for j, v in zip(INDUSTRIES, vec_fig) if divisions(j) and divisions(j) <= kd))


def add_figaro_sources(d):
    ENERGY = [INDUSTRIES.index(k) for k in ("B", "C19", "D35")]
    en, uses = [], {}
    for r in d.itertuples():
        U = figaro_use(int(r.year)).get(r.geo)
        if U is None:
            en.append(np.nan)
            continue
        en.append(to_klems(U[ENERGY].sum(0), r.ind))
        uses[(r.geo, r.year, r.ind)] = np.array([to_klems(U[k], r.ind) for k in range(U.shape[0])])
    d = d.copy()
    d["energy"] = en
    return d, uses


def b32(variant="primary", rent=False, n_p=N_P, seed=7):
    p = load(variant)
    d = sample(p, rent)
    d = d[d.groupby(["geo", "year"]).ind.transform("count") >= 8].copy()
    d["cy"] = d.geo + d.year.astype(str)
    d["flat"] = 1.0
    d["hours"] = d.H
    res = []
    pi, k, cy = shares(d, "PIv"), shares(d, "Kv"), d.cy.to_numpy()
    srcs = {"labour_W": "Wv", "hours": "hours", "GO": "GO", "VA (reference)": "VA", "flat": "flat", "materials": "II"}
    for name, col in srcs.items():
        if d[col].notna().all():
            lam, r2 = lam_fit(pi, k, shares(d, col), cy)
        else:                                   # II missing for 0.2% of rows: fit on complete country-years
            dm = d[d.groupby("cy")[col].transform(lambda s: s.notna().all())]
            lam, r2 = lam_fit(shares(dm, "PIv"), shares(dm, "Kv"), shares(dm, col), dm.cy.to_numpy())
        res.append(dict(source=name, lam=lam, r2=r2, window="2000-2021"))
    # cluster bootstrap by industry code for labour lambda
    rng = np.random.default_rng(seed)
    inds = d.ind.unique()
    boots = []
    s_w = shares(d, "Wv")
    for _ in range(N_BOOT):
        pick = rng.choice(inds, len(inds))
        idx = np.concatenate([np.where(d.ind.to_numpy() == i)[0] for i in pick])
        a, b = (pi - s_w)[idx], (k - s_w)[idx]
        boots.append((a * b).sum() / (b * b).sum())
    lo, hi = np.quantile(boots, [0.025, 0.975])
    res[0].update(ci_lo=float(lo), ci_hi=float(hi))
    # FIGARO window 2010-2021: energy, mixtures, labour on the same window
    f = d[d.year >= 2010].copy()
    f, uses = add_figaro_sources(f)
    f = f[f.energy.notna() & (f.energy > 0)]
    pif, kf, cyf = shares(f, "PIv"), shares(f, "Kv"), f.cy.to_numpy()
    for name, col in (("labour_W", "Wv"), ("energy", "energy"), ("GO", "GO"), ("flat", "flat")):
        lam, r2 = lam_fit(pif, kf, shares(f, col), cyf)
        res.append(dict(source=name, lam=lam, r2=r2, window="2010-2021"))
    # placebos (fixed per country over time)
    plac = {t: [] for t in ("a_perm", "b_lnorm_H", "c_mix", "d_lnorm_W")}
    geo_inds = {g: sorted(gg.ind.unique()) for g, gg in d.groupby("geo")}
    lh = np.log(d.H / d.GO)
    lw = np.log(d.Wv / d.GO)
    sd_h = lh.groupby(d.cy).transform("std").to_numpy()
    sd_w = lw.groupby(d.cy).transform("std").to_numpy()
    hco = (d.H / d.GO).to_numpy()
    med_h = pd.Series(hco).groupby(d.cy.to_numpy()).transform("median").to_numpy()
    GO = d.GO.to_numpy()
    key = pd.Series(hco, index=pd.MultiIndex.from_arrays([d.cy, d.ind]))
    prng = np.random.default_rng(seed + 1)
    perm_maps = [{g: dict(zip(li, np.array(li)[prng.permutation(len(li))])) for g, li in geo_inds.items()}
                 for _ in range(n_p)]
    eps = {g: prng.normal(0, 1, (n_p, len(li))) for g, li in geo_inds.items()}
    pos = np.array([geo_inds[g].index(i) for g, i in zip(d.geo, d.ind)])
    E = np.vstack([eps[g][:, j] for g, j in zip(d.geo, pos)]).T          # (n_p, rows)
    for i in range(n_p):
        tgt = [perm_maps[i][g][ind] for g, ind in zip(d.geo, d.ind)]
        v = key.reindex(pd.MultiIndex.from_arrays([d.cy, tgt])).to_numpy()
        v = np.where(np.isfinite(v), v, med_h) * GO
        plac["a_perm"].append(lam_fit(pi, k, v / pd.Series(v).groupby(cy).transform("sum").to_numpy(), cy))
        vb = np.exp(sd_h * E[i]) * GO
        plac["b_lnorm_H"].append(lam_fit(pi, k, vb / pd.Series(vb).groupby(cy).transform("sum").to_numpy(), cy))
        vd = np.exp(sd_w * E[i]) * GO
        plac["d_lnorm_W"].append(lam_fit(pi, k, vd / pd.Series(vd).groupby(cy).transform("sum").to_numpy(), cy))
    # (c) mixtures on the FIGARO window: fractions of purchases of random products, total = labour cost share
    geos = sorted(f.geo.unique())
    first = {g: f[f.geo == g].year.min() for g in geos}
    om = {}
    mrng = np.random.default_rng(seed + 2)
    for g in geos:
        ff = f[(f.geo == g) & (f.year == first[g])]
        Ug = np.vstack([uses[(g, first[g], i)] for i in ff.ind])        # rows industries, cols products
        cost = Ug.sum(0)
        target = ff.Wv.sum() / ff.GO.sum() * ff.GO.sum()
        oms = []
        for _ in range(n_p):
            o, tot = np.zeros(len(cost)), 0.0
            for kk in mrng.permutation(np.where(cost > 0)[0]):
                w = mrng.uniform(0.2, 1.0)
                if tot + w * cost[kk] >= target:
                    o[kk] = (target - tot) / cost[kk]
                    break
                o[kk], tot = w, tot + w * cost[kk]
            oms.append(o)
        om[g] = np.array(oms)
    Uf = np.vstack([uses[(g, y, i)] for g, y, i in zip(f.geo, f.year, f.ind)])  # rows x products
    OMf = np.stack([om[g] for g in f.geo], 1)                                  # (n_p, rows, products)
    for i in range(n_p):
        v = (OMf[i] * Uf).sum(1)
        v = np.where(v > 0, v, 1e-9)
        plac["c_mix"].append(lam_fit(pif, kf, v / pd.Series(v).groupby(cyf).transform("sum").to_numpy(), cyf))
    lamL = res[0]["lam"]
    lamLf = next(r["lam"] for r in res if r["source"] == "labour_W" and r["window"] == "2010-2021")
    for t, vals in plac.items():
        a = np.array(vals)
        ref = lamLf if t == "c_mix" else lamL
        res.append(dict(source=f"placebo {t}", lam=float(np.median(a[:, 0])), r2=float(np.median(a[:, 1])),
                        lam_p05=float(np.quantile(a[:, 0], 0.05)), share_lam_below_labour=float((a[:, 0] <= ref).mean()),
                        window="2010-2021" if t == "c_mix" else "2000-2021"))
    out = pd.DataFrame(res)
    out["variant"], out["with_rent"], out["n_obs"], out["n_cy"] = variant, rent, len(d), d.cy.nunique()
    return out


def run_b32():
    frames = []
    for var, rent in (("primary", False), ("primary", True), ("no_mi", False), ("tangible", False),
                      ("intangibles", False), ("cfc_as_capital", False)):
        frames.append(b32(var, rent, n_p=N_P if (var == "primary" and not rent) else 200))
        pd.concat(frames).to_csv(OUT / "b32.csv", index=False)
        print("b32", var, rent, flush=True)
    # per-country lambda (primary)
    p = load("primary")
    d = sample(p)
    d["cy"] = d.geo + d.year.astype(str)
    rows = []
    for g, dd in d.groupby("geo"):
        dd = dd[dd.groupby("year").ind.transform("count") >= 8]
        if dd.empty:
            continue
        lam, r2 = lam_fit(shares(dd, "PIv"), shares(dd, "Kv"), shares(dd, "Wv"), dd.cy.to_numpy())
        rows.append(dict(geo=g, lam=lam, r2=r2, n=len(dd)))
    pd.DataFrame(rows).to_csv(OUT / "b32_country.csv", index=False)


# ------------------------------------------------------------------ B3.3 (exploratory)
def run_b33():
    p = load("primary")
    d = sample(p).sort_values(["geo", "ind", "year"])
    d["cy"] = d.geo + d.year.astype(str)
    d["lh"] = np.log(d.H / d.GO_Q)
    g = d.groupby(["geo", "ind"])
    d["dl"] = g.lh.diff()
    d.loc[g.year.diff() != 1, "dl"] = np.nan
    d["dl"] = d.dl - d.groupby("cy").dl.transform("mean")
    rows = []
    for h in range(1, 6):
        d["y"] = g.r.shift(-h) - d.r
        dd = d.dropna(subset=["y", "dl"])
        X = demean(dd, ["y", "dl"], "cy")
        fit = sm.OLS(X.y, X[["dl"]]).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(dd.ind)[0]})
        rows.append(dict(h=h, beta=fit.params["dl"], se=fit.bse["dl"], p=fit.pvalues["dl"], n=len(dd)))
    pd.DataFrame(rows).to_csv(OUT / "b33.csv", index=False)


if __name__ == "__main__":
    import sys
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("all", "b31"):
        print(run_b31().round(4).to_string())
    if what in ("all", "b33"):
        run_b33()
    if what in ("all", "b32"):
        run_b32()
