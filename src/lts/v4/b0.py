"""Iteration 4, B0: dynamic test T1 (iteration 1) with a flat benchmark and 1000 permutation placebos.

T1 = 1 - Var(d lz_rel) / Var(d lP_rel): share of the variance of relative price (deflator) changes
explained by relative changes of vertically integrated content with beta = 1 (lts.dynamics).
Bases: labour (hours), flat (equal direct input per unit of output), 1000 permutations of the hours
coefficients (fixed per country over time, same seeds as stage P of v3).  Open system, capital,
imports at price.  13 FIGARO countries 2010-2022 (countries without deflators drop out) + USA BEA
1998-2023.  Output: results/v4/b0_t1.csv.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from .. import bea
from ..dynamics import prepare, t1_constrained_r2
from ..economy import build
from ..v3.placebo_disp import COUNTRIES, N_P, Draws, Setup
from .splits import OUT


def frames_for(e, S, D, c, y, P0):
    h = e.l()
    flat = np.where(e.x > 0, 1.0, 0.0)
    P = e.meta["price_index"]
    phys = np.where(np.isfinite(P0 / P) & (e.x > 0), P0 / P, 0.0)     # constant per unit of real output
    Z = np.vstack([S.z(h[None]), S.z(flat[None]), S.z(phys[None]), S.z(D.rows("perm", h))])
    names = ["labour", "flat", "flat_phys"] + [f"perm_{i:04d}" for i in range(N_P)]
    n = e.n
    return pd.DataFrame(dict(country=c, year=y, basis=np.repeat(names, n), industry=np.tile(e.labels, len(names)),
                             z=Z.ravel(), x=np.tile(e.x, len(names)), P=np.tile(e.meta["price_index"], len(names))))


def run_country(c, source="figaro"):
    frames, D, P0 = [], None, None
    years = range(1998, 2024) if source == "bea" else range(2010, 2023)
    for y in years:
        e, _ = bea.build(y) if source == "bea" else build(c, y)
        if not np.isfinite(e.meta["price_index"]).any():
            return None
        S = Setup(e)
        if D is None:
            D = Draws(c, e.n, S.base_mask)
            P0 = e.meta["price_index"]
        frames.append(frames_for(e, S, D, c, y, P0))
    df = pd.concat(frames, ignore_index=True)
    out = []
    for subset in ("core", "all"):
        d = prepare(df, subset, bea.BEA_ROLE if source == "bea" else None)
        t = t1_constrained_r2(d, horizons=(1, 5))
        t["subset"] = subset
        out.append(t)
    return pd.concat(out)


def run(countries=COUNTRIES):
    res = []
    for c in countries:
        t = run_country(c)
        if t is None:
            print("b0", c, "no deflators", flush=True)
            continue
        res.append(t)
        print("b0", c, flush=True)
    t = run_country("USA_BEA", "bea")
    res.append(t)
    pd.concat(res).to_csv(OUT / "b0_t1.csv", index=False)


if __name__ == "__main__":
    run(sys.argv[1].split(",") if len(sys.argv) > 1 else COUNTRIES)
