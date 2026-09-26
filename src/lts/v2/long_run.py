"""Iteration 2, stage 4: prices' long-run attraction to anchors.

Deviation of industry j from anchor B: dev_jt = log P_rel,jt - log B^phys_rel,jt = -log z^B_rel,jt (+ const_j).
Tests per anchor:
  * ADF (constant, 1 lag) on each industry's dev series -> share rejecting a unit root at 10%; Maddala-Wu
    Fisher test (-2 sum ln p ~ chi2(2N)). (Restricted cointegration of log P and log B^phys with coefficient 1.)
  * Panel ECM, industry fixed effects:
      A (pre-registered):  D log P_rel,t = a_j + rho dev_{t-1} + g D log B^phys_rel,t + e
      B (no contemporaneous anchor term; logged deviation, see pre-registration journal):
                           D log P_rel,t = a_j + rho dev_{t-1} + e
    half-life = ln 0.5 / ln(1 + rho) for -1 < rho < 0. Industry-clustered SE.
Anchors: labour values, PP (a, b, c), commodity bases (open and symmetric-uniform), fixed nominal random
placebos (upper reference, stationary by construction) and permuted-hours placebos (null).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller

from ..dynamics import prepare

ROOT = Path(__file__).resolve().parents[3]
TAB = ROOT / "results" / "tables"
OUT = ROOT / "results" / "v2"


def panel(tag: str, src: str, role_map):
    d = pd.read_parquet(TAB / f"ratios_long_{tag}.parquet",
                        filters=[("capital", "==", True), ("imports", "==", "price"), ("labour", "==", "hours")])
    keep = ((d.closed == "False") & (d.basis.str.match(r"^(labour|k:.*|placebo_random_0[0-4]\d|placebo_perm_0[0-4]\d)$"))) \
        | ((d.closed == "uniform") & d.basis.str.startswith("k:"))
    d = d[keep].copy()
    d.loc[d.closed == "uniform", "basis"] = "sym" + d.loc[d.closed == "uniform", "basis"]
    ex = pd.read_json(TAB / f"economy_extra_{tag}.json")
    pr = pd.concat([pd.DataFrame({"country": r.country, "year": r.year, "industry": r.labels, "P": r.price_index})
                    for _, r in ex.iterrows()])
    d = d.merge(pr, on=["country", "year", "industry"], how="left")
    a = pd.read_parquet(OUT / f"anchors_{src}.parquet")
    a = a[(a.capital == True) & a.basis.isin(["PPa", "PPb", "PPc_edu_years"])]
    return pd.concat([d[["country", "year", "basis", "industry", "z", "x", "P"]],
                      a[["country", "year", "basis", "industry", "z", "x", "P"]]], ignore_index=True)


def kind(b: str) -> str:
    if b == "labour":
        return "labour"
    if b.startswith("PP"):
        return b
    if b.startswith("symk:"):
        return "commodity_sym"
    if b.startswith("k:"):
        return "commodity_open"
    return "placebo_fixed_nominal" if b.startswith("placebo_random") else "placebo_perm"


def ecm(g: pd.DataFrame, with_g: bool) -> dict:
    g = g.sort_values(["industry", "year"]).copy()
    g["dev"] = -g.lz_rel
    g["dev_l1"] = g.groupby("industry").dev.shift(1)
    s = g.dropna(subset=["d_lP_rel", "dev_l1"] + (["d_lB_rel"] if with_g else []))
    if len(s) < 30:
        return {}
    cols = ["dev_l1"] + (["d_lB_rel"] if with_g else [])
    Y = s.d_lP_rel - s.groupby("industry").d_lP_rel.transform("mean")
    X = s[cols] - s.groupby("industry")[cols].transform("mean")
    Xv, Yv = X.to_numpy(), Y.to_numpy()
    XtX = Xv.T @ Xv
    beta = np.linalg.solve(XtX, Xv.T @ Yv)
    e = Yv - Xv @ beta
    Si = np.linalg.inv(XtX)
    meat = sum(np.outer(Xv[idx].T @ e[idx], Xv[idx].T @ e[idx]) for idx in s.groupby("industry").indices.values())
    se = np.sqrt(np.diag(Si @ meat @ Si))
    rho = beta[0]
    hl = np.log(0.5) / np.log(1 + rho) if -1 < rho < 0 else np.nan
    return dict(rho=rho, se_rho=se[0], half_life=hl, n=len(s))


def adf_panel(g: pd.DataFrame) -> dict:
    ps = []
    for _, gi in g.groupby("industry"):
        y = (-gi.sort_values("year").lz_rel).to_numpy()
        if len(y) < 8 or np.std(y) < 1e-10:
            continue
        try:
            ps.append(adfuller(y, maxlag=1, regression="c", autolag=None)[1])
        except Exception:
            continue
    if not ps:
        return {}
    ps = np.clip(np.array(ps), 1e-12, 1)
    fisher = -2 * np.log(ps).sum()
    return dict(adf_share_reject10=float((ps < 0.10).mean()), fisher_p=float(stats.chi2.sf(fisher, 2 * len(ps))),
                n_ind=len(ps))


def run():
    from ..bea import BEA_ROLE
    rows = []
    for tag, src, rm in (("bea", "bea", BEA_ROLE), ("main", "figaro", None)):
        df = panel(tag, src, rm)
        for sub in ("all", "core"):
            d = prepare(df, sub, rm)
            for (c, b), g in d.groupby(["country", "basis"]):
                r = dict(country=c, basis=b, kind=kind(b), subset=sub, T=g.year.nunique())
                r.update(adf_panel(g))
                for lab, wg in (("A", True), ("B", False)):
                    res = ecm(g, wg)
                    r.update({f"{k}_{lab}": v for k, v in res.items()})
                rows.append(r)
            print("long_run", tag, sub, flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "long_run.csv", index=False)
    return out


if __name__ == "__main__":
    run()
