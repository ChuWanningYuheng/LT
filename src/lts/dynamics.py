"""Theory-discriminating tests on changes over time (Stage 5).

Notation. For basis B, industry j, year t: z^B_jt = (B-content per currency unit of output) after
normalisation (sum_j z x = sum_j x each year). With P_jt the output price index,
    log B^phys_jt = log z^B_jt + log P_jt + const_j
is the B-content per REAL unit of output (because z = content per nominal unit = content per real
unit / price).  "Relative" = deviation from the (output-weighted) cross-industry mean in year t.

Tests
  T1  constrained dynamic R^2 : 1 - Var(D log z^B_rel) / Var(D log P_rel)
      = share of the variance of relative price changes accounted for by changes of B-content per
      real unit with the theoretical elasticity of 1. Not circular: the deflator enters only as
      the benchmark variance (log z is computed from current-price tables and hours only).
  T2  panel regression  D log P_rel_jt = b * D log B^phys_rel_j,t-k + industry FE + year FE, k=0,1,2.
      For k=0 the deflator appears on both sides (D log B^phys = D log z + D log P), so b and R^2 are
      partly mechanical; the placebo bases measure how much.
  T3  encompassing / horse race: min Var(sum_B w_B D log z^B_rel) s.t. sum w = 1; the weight on labour
      shows whether other bases add information (w_L = 1 -> labour encompasses them).
  T4  out-of-sample: industry effects a_jB estimated on training years; in test years the prediction
      error of relative prices given B^phys is log z^B_rel,jt - a_jB. Benchmarks: frozen relative prices.
  T5  incremental power of values over prices of production (T3 with B in {labour, pp}).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .economy import BASES
from .levels import SUBSETS, mask_for

ROOT = Path(__file__).resolve().parents[2]
TAB = ROOT / "results" / "tables"


# ----------------------------------------------------------------------------------------------
def load_panel(tag: str, spec: dict, role_map=None) -> pd.DataFrame:
    df = pd.read_parquet(TAB / f"ratios_long_{tag}.parquet")
    q = np.ones(len(df), bool)
    for k, v in spec.items():
        q &= (df[k].astype(str) == str(v)).to_numpy()
    df = df[q | df.basis.str.startswith("pp_eigen") & (df.capital == spec.get("capital"))]
    ex = pd.read_json(TAB / f"economy_extra_{tag}.json")
    pr = []
    for _, r in ex.iterrows():
        pr.append(pd.DataFrame({"country": r.country, "year": r.year, "industry": r.labels, "P": r.price_index}))
    pr = pd.concat(pr)
    df = df.merge(pr, on=["country", "year", "industry"], how="left")
    return df


def prepare(df: pd.DataFrame, subset: str, role_map=None) -> pd.DataFrame:
    """Long panel: country, basis, industry, year, lz (log z), lP (log P), w (output share)."""
    excl = SUBSETS[subset]
    keep = []
    for (c, y), g in df.groupby(["country", "year"]):
        labs = g.industry.unique().tolist()
        ok = set(np.array(labs)[mask_for(labs, excl, role_map)])
        keep.append(g[g.industry.isin(ok)])
    d = pd.concat(keep)
    d = d[(d.z > 0) & d.P.notna() & (d.P > 0)].copy()
    d["lz"] = np.log(d.z)
    d["lP"] = np.log(d.P)
    d["w"] = d.x / d.groupby(["country", "basis", "year"]).x.transform("sum")
    for v in ["lz", "lP"]:
        m = (d[v] * d.w).groupby([d.country, d.basis, d.year]).transform("sum")
        d[v + "_rel"] = d[v] - m
    d = d.sort_values(["country", "basis", "industry", "year"])
    g = d.groupby(["country", "basis", "industry"])
    for v in ["lz_rel", "lP_rel"]:
        d["d_" + v] = g[v].diff()
    d["gap"] = g.year.diff()
    d.loc[d.gap != 1, ["d_lz_rel", "d_lP_rel"]] = np.nan
    d["lB_rel"] = d.lz_rel + d.lP_rel            # log B^phys, relative (up to industry constant)
    d["d_lB_rel"] = d.d_lz_rel + d.d_lP_rel
    for k in (1, 2):
        d[f"d_lB_rel_l{k}"] = g["d_lB_rel"].shift(k)
    return d


def wvar(x, w):
    w = w / w.sum()
    m = (w * x).sum()
    return float((w * (x - m) ** 2).sum())


# ----------------------------------------------------------------------------------------------
def t1_constrained_r2(d: pd.DataFrame, horizons=(1, 3, 5)) -> pd.DataFrame:
    out = []
    for (c, b), g in d.groupby(["country", "basis"]):
        for h in horizons:
            gg = g.sort_values(["industry", "year"])
            gi = gg.groupby("industry")
            dz = gi.lz_rel.diff(h)
            dp = gi.lP_rel.diff(h)
            ok = dz.notna() & dp.notna() & (gi.year.diff(h) == h)
            if ok.sum() < 10:
                continue
            w = gg.loc[ok, "w"]
            # remove year means of the h-differences (relative changes)
            yz = dz[ok] - (dz[ok] * w).groupby(gg.year[ok]).transform("sum") / w.groupby(gg.year[ok]).transform("sum")
            yp = dp[ok] - (dp[ok] * w).groupby(gg.year[ok]).transform("sum") / w.groupby(gg.year[ok]).transform("sum")
            out.append(dict(country=c, basis=b, h=h, n=int(ok.sum()),
                            r2_w=1 - wvar(yz, w) / wvar(yp, w),
                            r2_u=1 - yz.var() / yp.var()))
    return pd.DataFrame(out)


def _demean2(y: pd.Series, a: pd.Series, b: pd.Series, iters: int = 30, tol: float = 1e-10) -> pd.Series:
    """Two-way within transformation (alternating projections, stops at convergence)."""
    r = y.to_numpy(dtype=float).copy()
    ia, ib = pd.factorize(a)[0], pd.factorize(b)[0]
    na, nb = ia.max() + 1, ib.max() + 1
    ca, cb = np.bincount(ia, minlength=na), np.bincount(ib, minlength=nb)
    for _ in range(iters):
        ma = np.bincount(ia, r, na) / ca
        r = r - ma[ia]
        mb = np.bincount(ib, r, nb) / cb
        r = r - mb[ib]
        if np.abs(mb).max() < tol and np.abs(ma).max() < tol:
            break
    return pd.Series(r, index=y.index)


def t2_regressions(d: pd.DataFrame, lags=(0, 1, 2)) -> pd.DataFrame:
    out = []
    keep = d.basis.str.startswith(("k:", "labour", "pp_")) | d.basis.str.match(r"placebo_random_0[0-4]\d") \
        | d.basis.str.match(r"placebo_perm_0[01]\d")          # T2 on 50 random + 20 permutation placebos
    for (c, b), g in d[keep].groupby(["country", "basis"]):
        for k in lags:
            xcol = "d_lB_rel" if k == 0 else f"d_lB_rel_l{k}"
            s = g[["d_lP_rel", xcol, "industry", "year"]].dropna()
            if len(s) < 20:
                continue
            y = _demean2(s.d_lP_rel, s.industry, s.year)
            x = _demean2(s[xcol], s.industry, s.year)
            beta = float((x * y).sum() / (x * x).sum())
            e = y - beta * x
            # cluster-robust SE by industry
            num = float(((x * e).groupby(s.industry).sum() ** 2).sum())
            se = np.sqrt(num) / float((x * x).sum())
            r2 = 1 - float((e ** 2).sum() / (y ** 2).sum())
            out.append(dict(country=c, basis=b, lag=k, n=len(s), beta=beta, se=se, r2_within=r2))
    return pd.DataFrame(out)


def encompassing(g: pd.DataFrame, bases: list[str], col: str) -> dict:
    """min Var(sum_B w_B y_B) s.t. sum w = 1 (weighted by output shares). Returns weights & variances."""
    wide = g.pivot_table(index=["industry", "year"], columns="basis", values=col)
    wts = g.pivot_table(index=["industry", "year"], columns="basis", values="w").mean(axis=1)
    wide = wide[bases].dropna()
    wts = wts.reindex(wide.index)
    Y = wide.to_numpy()
    W = (wts / wts.sum()).to_numpy()
    Yc = Y - W @ Y
    S = (Yc * W[:, None]).T @ Yc
    one = np.ones(len(bases))
    try:
        Si = np.linalg.pinv(S)
        wopt = Si @ one / (one @ Si @ one)
    except np.linalg.LinAlgError:
        return {}
    res = {f"w_{b}": float(v) for b, v in zip(bases, wopt)}
    res.update({f"var_{b}": float(S[i, i]) for i, b in enumerate(bases)})
    res["var_opt"] = float(wopt @ S @ wopt)
    res["n"] = len(wide)
    return res


def t3_horse_race(d: pd.DataFrame, sets: dict) -> pd.DataFrame:
    out = []
    for c, g in d.groupby("country"):
        for name, bases in sets.items():
            bs = [b for b in bases if b in set(g.basis)]
            if len(bs) < 2:
                continue
            for col, lab in [("d_lz_rel", "changes"), ("lz_rel", "levels")]:
                r = encompassing(g[g.basis.isin(bs)], bs, col)
                if r:
                    out.append(dict(country=c, set=name, on=lab, **r))
    return pd.DataFrame(out)


def t4_out_of_sample(d: pd.DataFrame, splits: list[tuple[int, int, int]]) -> pd.DataFrame:
    out = []
    for c, gc in d.groupby("country"):
        for (t_end, s0, s1) in splits:
            for b, g in gc.groupby("basis"):
                tr = g[g.year <= t_end]
                te = g[(g.year >= s0) & (g.year <= s1)]
                if tr.year.nunique() < 3 or te.empty:
                    continue
                a_mean = tr.groupby("industry").lz_rel.mean()
                a_last = tr[tr.year == tr.year.max()].set_index("industry").lz_rel
                e_mean = te.lz_rel - te.industry.map(a_mean)
                e_last = te.lz_rel - te.industry.map(a_last)
                p_last = tr[tr.year == tr.year.max()].set_index("industry").lP_rel
                e_naive = te.lP_rel - te.industry.map(p_last)
                ok = e_mean.notna() & e_last.notna() & e_naive.notna()
                w = te.w[ok] / te.w[ok].sum()
                rm = lambda e: float(np.sqrt((w * e[ok] ** 2).sum()))
                out.append(dict(country=c, train_end=t_end, test=f"{s0}-{s1}", basis=b, n=int(ok.sum()),
                                rmse_anchor_mean=rm(e_mean), rmse_anchor_last=rm(e_last),
                                rmse_naive_frozen_prices=rm(e_naive)))
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------------------------
def summarise_distribution(tab: pd.DataFrame, value: str, keys: list[str]) -> pd.DataFrame:
    """Labour vs the distribution of commodity bases and placebos for statistic `value`."""
    t = tab.copy()
    t["kind"] = np.select([t.basis.str.startswith("k:"), t.basis.str.startswith("placebo_random"),
                           t.basis.str.startswith("placebo_perm")], ["commodity", "placebo_random", "placebo_perm"],
                          t.basis)
    rows = []
    for k, g in t.groupby(keys):
        k = k if isinstance(k, tuple) else (k,)
        r = dict(zip(keys, k))
        for kind in ["labour", "pp_uniform_wage", "pp_actual_wage", "pp_eigen"]:
            v = g.loc[g.kind == kind, value]
            r[kind] = float(v.iloc[0]) if len(v) else np.nan
        for kind in ["commodity", "placebo_random", "placebo_perm"]:
            v = g.loc[g.kind == kind, value]
            if len(v):
                r[f"{kind}_median"] = float(v.median())
                r[f"{kind}_p95"] = float(v.quantile(0.95))
                r[f"{kind}_max"] = float(v.max())
                if np.isfinite(r["labour"]):
                    r[f"labour_pct_vs_{kind}"] = float((v < r["labour"]).mean())
        rows.append(r)
    return pd.DataFrame(rows)


NAMED = {f"k:{v}": k for k, v in BASES.items()}


def run(tag: str = "main", role_map=None, splits=None):
    specs = {
        "open_cap": dict(capital=True, closed=False, imports="price", labour="hours"),
        "open_nocap": dict(capital=False, closed=False, imports="price", labour="hours"),
        "closed_cap": dict(capital=True, closed=True, imports="price", labour="hours"),
        "closedU_cap": dict(capital=True, closed="uniform", imports="price", labour="hours"),
        "open_cap_comp": dict(capital=True, closed=False, imports="competitive", labour="hours"),
        "open_cap_wage": dict(capital=True, closed=False, imports="price", labour="wage_weighted"),
        "open_cap_edu": dict(capital=True, closed=False, imports="price", labour="edu_years"),
    }
    if splits is None:
        splits = [(2016, 2017, 2023), (2018, 2019, 2023), (2016, 2017, 2019)]
    allt1, allt2, allt3, allt4 = [], [], [], []
    for sname, spec in specs.items():
        raw = load_panel(tag, spec, role_map)
        if raw.empty:
            continue
        for subset in ["all", "core"]:
            d = prepare(raw, subset, role_map)
            meta = dict(spec=sname, subset=subset)
            t1 = t1_constrained_r2(d).assign(**meta)
            t2 = t2_regressions(d).assign(**meta)
            key = [b for b in ["labour", "pp_uniform_wage", "pp_actual_wage"] if b in set(d.basis)]
            if role_map is None:
                named = [b for b in NAMED if b in set(d.basis)]
                elec, oil = "k:D35", "k:C19"
            else:
                named = [f"k:{lab}" for lab, r in role_map.items() if r in BASES.values() and f"k:{lab}" in set(d.basis)]
                elec, oil = "k:22", "k:324"
            sets = {"labour_vs_named": ["labour"] + named,
                    "labour_vs_pp": key,
                    "labour_vs_electricity": ["labour", elec] if elec in set(d.basis) else [],
                    "labour_vs_oil": ["labour", oil] if oil in set(d.basis) else []}
            t3 = t3_horse_race(d, {k: v for k, v in sets.items() if v}).assign(**meta)
            t4 = t4_out_of_sample(d, splits).assign(**meta)
            allt1.append(t1), allt2.append(t2), allt3.append(t3), allt4.append(t4)
            print(tag, sname, subset, "done", flush=True)
    T1, T2, T3, T4 = (pd.concat(x, ignore_index=True) for x in (allt1, allt2, allt3, allt4))
    T1.to_csv(TAB / f"dyn_T1_{tag}.csv", index=False)
    T2.to_csv(TAB / f"dyn_T2_{tag}.csv", index=False)
    T3.to_csv(TAB / f"dyn_T3_{tag}.csv", index=False)
    T4.to_csv(TAB / f"dyn_T4_{tag}.csv", index=False)
    return T1, T2, T3, T4


if __name__ == "__main__":
    import sys
    tag = sys.argv[1] if len(sys.argv) > 1 else "main"
    if tag == "bea":
        from .bea import BEA_ROLE
        run("bea", role_map=BEA_ROLE, splits=[(2008, 2009, 2014), (2008, 2009, 2023), (2014, 2015, 2023)])
    else:
        run(tag)
