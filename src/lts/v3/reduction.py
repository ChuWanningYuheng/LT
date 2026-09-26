"""Iteration 3, stage 1: reduction of heterogeneous labour without the country's industry wages.

For every country-year (13 countries, 2010-2022; open system, capital included, imports at price,
industries without T/U) and every labour vector l = hours x weight:
  err (MAWD, d), percentile among commodity bases (share of bases with a lower metric),
  share of placebos better than labour for (a) permutations, (b) log-normal, (c) v2 mixtures,
  with placebos built from the reduced vector (primary) and from plain hours (secondary).
Weights: see weights.py.  1.4 (EWCS) is not available (A-V3-EWCS).  Sensitivity grids are evaluated
for err only.  Output: results/v3/reduction_levels.csv, results/v3/reduction_weights_info.csv.
"""
from __future__ import annotations

import itertools
import sys

import numpy as np
import pandas as pd

from ..economy import build
from ..v2.placebo_cost import metrics_mat
from . import weights as W
from .placebo_disp import COUNTRIES, OUT, YEARS, Draws, Setup, shares

CENTRAL = ["hours", "wagebill", "1.1a_edu_price", "1.1b_occ_price", "1.2a_jz_time", "1.2b_jz_price",
           "1.3a_hilf", "1.3b_hilf_subs", "1.5a_frozen2010", "1.5b_foreign_median"]


def all_wage_rel(years=YEARS):
    """fig_wage_rel for every country-year (donors for 1.5b)."""
    out = {}
    for c in COUNTRIES:
        for y in years:
            e, _ = build(c, y)
            out[(c, y)] = W.fig_wage_rel(e)
    return out


def foreign_median(e, c, y, WR):
    donors = pd.concat([WR[(d, y)] for d in COUNTRIES if d != c], axis=1)
    med = donors.median(1)
    return W.normalise(W._map_ind(med, e, None), e.hours)


def vectors(e, c, y, WR, frozen):
    """name -> (weight or None, info). Central variants + sensitivity grids."""
    V = {"hours": (np.ones(e.n), ""), "wagebill": (W.wage_rel(e), "")}
    V["1.1a_edu_price"] = W.w_edu_price(e, c, y)
    V["1.1a_edu_price_ses"] = W.w_edu_price(e, c, y, source="ses")
    V["1.1b_occ_price"] = W.w_occ_price(e, c, y)
    V["1.2a_jz_time"] = W.w_jobzone_time(e, c, y)
    V["1.2b_jz_price"] = W.w_jobzone_price(e, c, y)
    V["1.3a_hilf"] = W.w_hilferding(e, c, y)
    V["1.3b_hilf_subs"] = W.w_hilferding(e, c, y, subsistence=True)
    V["1.5a_frozen2010"] = (frozen, "own 2010 relative hourly labour income") if y > 2010 else (None, "base year")
    V["1.5b_foreign_median"] = (foreign_median(e, c, y, WR), "median of 12 other countries")
    for H, T, jz5 in itertools.product((1000, 1600, 2000), (35, 40, 45), (4.0, 5.0, 6.0)):
        V[f"grid_1.2a_H{H}_T{T}_JZ{jz5:g}"] = W.w_jobzone_time(e, c, y, H=H, T=T, jz5=jz5)
    for jz5 in (4.0, 6.0):
        V[f"grid_1.2b_JZ{jz5:g}"] = W.w_jobzone_price(e, c, y, jz5=jz5)
    for H, T in itertools.product((1000, 1600, 2000), (35, 40, 45)):
        V[f"grid_1.3a_H{H}_T{T}"] = W.w_hilferding(e, c, y, H=H, T=T)
        V[f"grid_1.3b_H{H}_T{T}"] = W.w_hilferding(e, c, y, H=H, T=T, subsistence=True)
    return V


def commodity_Z(e, S):
    D, _ = e.dep_coeffs()
    Z = []
    for k in range(e.n):
        if e.A[k].sum() + D[k].sum() <= 0:
            continue
        Z.append(e.basis_ratios(k, True, False, "price"))
    return np.array(Z)


def run(countries=COUNTRIES, years=YEARS, tag=""):
    OUT.mkdir(parents=True, exist_ok=True)
    WR = all_wage_rel(years)
    rows, info = [], []
    for c in countries:
        D, frozen = None, None
        for y in years:
            e, _ = build(c, y)
            S = Setup(e)
            if D is None:
                D = Draws(c, e.n, S.base_mask)
            if y == min(years):
                frozen = W.wage_rel(e)
            m = S.base_mask
            ZB = commodity_Z(e, S)
            mB = W_metrics(ZB, S, m)
            h = e.l()
            for name, (w, src) in vectors(e, c, y, WR, frozen).items():
                info.append(dict(country=c, year=y, variant=name, available=w is not None, source=src,
                                 logsd_w=float(np.std(np.log(w))) if w is not None else np.nan,
                                 corr_wage=float(np.corrcoef(np.log(w), np.log(W.wage_rel(e)))[0, 1])
                                 if w is not None and name != "hours" else np.nan))
                if w is None:
                    continue
                l = h * w
                zl = S.z(l[None])
                ml = metrics_mat(zl[:, m], S.x[m])
                base = dict(country=c, year=y, variant=name)
                for met in ("mawd", "d"):
                    rows.append(dict(base, kind="err", metric=met, value=float(ml[met][0])))
                    rows.append(dict(base, kind="pct_bases", metric=met,
                                     value=float((mB[met] < ml[met][0]).mean())))
                if name.startswith("grid_") or name == "1.1a_edu_price_ses":
                    continue
                for r in shares(S, D, l, {"all": m}):
                    rows.append(dict(base, kind=f"plac_{r['kind']}_own", metric=r["metric"], value=r["share_better"]))
                for r in shares(S, D, l, {"all": m}, kinds=("perm", "lnorm"), placebo_l=h):
                    rows.append(dict(base, kind=f"plac_{r['kind']}_hours", metric=r["metric"], value=r["share_better"]))
            print("reduction", c, y, flush=True)
    pd.DataFrame(rows).to_csv(OUT / f"reduction_levels{tag}.csv", index=False)
    pd.DataFrame(info).to_csv(OUT / f"reduction_weights_info{tag}.csv", index=False)


def W_metrics(Z, S, m):
    return metrics_mat(Z[:, m], S.x[m])


if __name__ == "__main__":
    cs = sys.argv[1].split(",") if len(sys.argv) > 1 else COUNTRIES
    run(cs, tag="" if len(sys.argv) <= 1 else "_" + "_".join(cs))
