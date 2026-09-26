"""Iteration 3, stage 3: why labour loses to placebos in some countries.

3.1 artefact table (data construction features by country) and harmonised labour vectors
    (persons; hours with the national average hours per person) re-evaluated against placebos.
3.2/3.3 candidates K1-K5 (country-year), DV = share of placebos better than labour (mean of MAWD and d),
    panel with country and year FE, cluster-robust SE by country and wild cluster bootstrap p-values
    (Webb weights, restricted, 9999 draws), Holm correction over the 5 primary indicators;
    cross-country Spearman with permutation p-values (supplementary).
Inputs: results/v3/placebo_disp.csv, premium_country_year.csv, data/raw/v3/cbc.csv (OECD CBC).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ..economy import build
from .placebo_disp import COUNTRIES, OUT, YEARS, Draws, Setup, shares
from .weights import RAW

LOSERS = ["USA", "GBR", "JPN", "KOR", "NLD", "POL"]
WINNERS = ["AUT", "CZE", "DEU", "ESP", "FRA", "ITA", "MEX"]
PRIMARY = {"K1": "cbc", "K2": "sd_logwage", "K3": "import_share", "K4": "pcm_mean", "K5": "sd_premium"}
SECONDARY = {"K2": "sd_logwage_d1", "K4": "share_BKL"}


def coverage() -> pd.DataFrame:
    d = pd.read_csv(RAW / "cbc.csv")
    d = d[d.REF_AREA.isin(COUNTRIES)].groupby(["REF_AREA", "TIME_PERIOD"]).OBS_VALUE.mean()
    out = []
    for c in COUNTRIES:
        s = d.loc[c] if c in d.index.get_level_values(0) else pd.Series(dtype=float)
        s = s.reindex(range(min(YEARS), max(YEARS) + 1))
        s = s.interpolate(limit_area="inside")          # A-V3-CBC: interior gaps only, no extrapolation
        out.append(pd.DataFrame(dict(country=c, year=s.index, cbc=s.values)))
    return pd.concat(out)


def dv_table(kind="mix", mask="all") -> pd.DataFrame:
    p = pd.read_csv(OUT / "placebo_disp.csv")
    p = p[(p.kind == kind) & (p["mask"] == mask)]
    w = p.pivot_table(index=["country", "year"], columns="metric", values=["share_better", "sdist"])
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    w["dv"] = (w.share_better_mawd + w.share_better_d) / 2
    w["dv_sdist"] = (w.sdist_mawd + w.sdist_d) / 2
    return w.reset_index()


def panel_data(kind="mix"):
    cy = pd.read_csv(OUT / "premium_country_year.csv")
    d = dv_table(kind).merge(cy, on=["country", "year"]).merge(coverage(), on=["country", "year"], how="left")
    return d


def twoway_demean(d, cols):
    """Two-way (country, year) within transformation on an unbalanced panel by alternating projections."""
    X = d[cols].astype(float).copy()
    for _ in range(200):
        prev = X.copy()
        X = X - X.groupby(d.country.values).transform("mean")
        X = X - X.groupby(d.year.values).transform("mean")
        if np.nanmax(np.abs(X.values - prev.values)) < 1e-12:
            break
    return X


def wild_cluster(y, x, g, B=9999, seed=11):
    """Restricted wild cluster bootstrap (Webb 6-point weights) for H0: beta = 0 in y = b x + e
    (variables already demeaned).  Returns beta, cluster-robust t (CR1), bootstrap p (symmetric)."""
    cl, idx = np.unique(g, return_inverse=True)
    G = len(cl)
    xx = x @ x
    s_c = np.bincount(idx, weights=x * y, minlength=G)          # sum_i in c of x_i y_i
    xx_c = np.bincount(idx, weights=x * x, minlength=G)

    def t_of(V):                                                # V: (B, G) cluster multipliers
        b = (V @ s_c) / xx
        meat = ((V * s_c[None]) - b[:, None] * xx_c[None]) ** 2
        se = np.sqrt(meat.sum(1) * G / (G - 1)) / xx
        return b, b / se
    b0, t0 = t_of(np.ones((1, G)))
    rng = np.random.default_rng(seed)
    webb = np.array([-np.sqrt(1.5), -1, -np.sqrt(0.5), np.sqrt(0.5), 1, np.sqrt(1.5)])
    _, ts = t_of(webb[rng.integers(0, 6, (B, G))])             # restricted residuals under H0 = y
    return float(b0[0]), float(t0[0]), float((np.abs(ts) >= abs(t0[0])).mean())


def holm(p: dict) -> dict:
    items = sorted(p.items(), key=lambda kv: kv[1])
    m, out, run = len(items), {}, 0.0
    for i, (k, v) in enumerate(items):
        run = max(run, min(1.0, (m - i) * v))
        out[k] = run
    return out


def panel_tests(d: pd.DataFrame, dv="dv"):
    rows = []
    for k, col in {**PRIMARY, **{f"{k}s": v for k, v in SECONDARY.items()}}.items():
        s = d.dropna(subset=[dv, col])
        # countries contribute only if the candidate varies within them
        var = s.groupby("country")[col].transform(lambda v: v.std() > 1e-9)
        s = s[var]
        X = twoway_demean(s, [dv, col])
        xs = (X[col] / s[col].std()).values                # beta per 1 SD of the candidate
        b, t, pb = wild_cluster(X[dv].values, xs, s.country.values)
        rows.append(dict(candidate=k, indicator=col, dv=dv, beta_per_sd=b, t_cluster=t, p_wild=pb,
                         n=len(s), countries=s.country.nunique(), primary=k in PRIMARY))
    r = pd.DataFrame(rows)
    ph = holm(dict(zip(r[r.primary].candidate, r[r.primary].p_wild)))
    r["p_holm"] = r.candidate.map(ph)
    return r


def cross_country(d: pd.DataFrame, dv="dv", n_perm=20000, seed=5):
    m = d.groupby("country").mean(numeric_only=True)
    rows = []
    rng = np.random.default_rng(seed)
    for k, col in {**PRIMARY, **{f"{k}s": v for k, v in SECONDARY.items()}}.items():
        s = m[[dv, col]].dropna()
        rho = stats.spearmanr(s[dv], s[col]).statistic
        ra = stats.rankdata(s[dv].values)
        rb = stats.rankdata(s[col].values)
        ra, rb = (ra - ra.mean()) / ra.std(), (rb - rb.mean()) / rb.std()
        P = np.argsort(rng.random((n_perm, len(ra))), axis=1)
        perm = (ra[P] * rb[None]).mean(1)                    # Spearman under permutation (ranks)
        rows.append(dict(candidate=k, indicator=col, dv=dv, rho=rho, p_perm=float((np.abs(perm) >= abs(rho)).mean()),
                         n=len(s)))
    return pd.DataFrame(rows), m


def artefact_table():
    rows = []
    for c in COUNTRIES:
        e, info = build(c, 2015)
        lg = info["labour_log"]
        hpp = e.hours / np.where(e.meta["persons"] > 0, e.meta["persons"], np.nan)
        rows.append(dict(country=c, group="loser" if c in LOSERS else "winner", n_industries=e.n,
                         persons_source=lg.get("persons_source"), zero_persons=lg.get("persons_zero_with_output"),
                         hpp_national_fallback=lg.get("hpp_national_fallback"),
                         hours_rescale=lg.get("hours_rescale"), cfc_source=lg.get("cfc_source"),
                         import_share=float((e.Am.sum(0) * e.x).sum() / e.x.sum()),
                         hpp_cv=float(np.nanstd(hpp) / np.nanmean(hpp)),
                         min_hours_share=float((e.hours / e.hours.sum())[e.x > 0].min())))
    return pd.DataFrame(rows)


def harmonised(countries=COUNTRIES, years=YEARS):
    """Labour = persons (no hours per person) and = persons x national hours per person;
    placebos built from each vector (perm, lnorm) and v2 mixtures."""
    rows = []
    for c in countries:
        D = None
        for y in years:
            e, _ = build(c, y)
            S = Setup(e)
            if D is None:
                D = Draws(c, e.n, S.base_mask)
            pers = e.meta["persons"].astype(float)
            for name, vec in (("persons", pers), ("hours", e.hours)):
                l = np.divide(vec, e.x, out=np.zeros(e.n), where=e.x > 0)
                for r in shares(S, D, l, {"all": S.base_mask}):
                    rows.append(dict(country=c, year=y, vector=name, **r))
            print("harmonised", c, y, flush=True)
    return pd.DataFrame(rows)


def run():
    artefact_table().to_csv(OUT / "cc_artefacts.csv", index=False)
    res, cc = [], []
    for kind in ("mix", "perm", "lnorm"):
        d = panel_data(kind)
        if kind == "mix":
            d.to_csv(OUT / "cc_panel_data.csv", index=False)
        for dv in ("dv", "share_better_mawd", "share_better_d", "dv_sdist"):
            r = panel_tests(d, dv)
            r["placebo"] = kind
            res.append(r)
            x, _ = cross_country(d, dv)
            x["placebo"] = kind
            cc.append(x)
    pd.concat(res).to_csv(OUT / "cc_panel_tests.csv", index=False)
    pd.concat(cc).to_csv(OUT / "cc_cross_country.csv", index=False)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "harmonised":
        harmonised().to_csv(OUT / "cc_harmonised.csv", index=False)
    else:
        run()
