"""Iteration 4, A2: imports valued at the labour content of the supplying countries (world FIGARO
system, depreciation included in every country) and the v3 stage-P classification in that system.

Labour (hours) per unit of output, FIGARO level:
  13 focus countries: hours as in v3 (oecd.labour_block);
  other FIGARO countries: OECD TiMBC persons (split by output) x OECD average annual hours per worker
      (DSD_HW, all workers; 1800 h if missing, A-V4-MRIO-H);
  countries without TiMBC and the rest of the world: output-weighted hours coefficient of the non-OECD
      FIGARO countries with data (BR, CN, IN, ID, AR, RU, SA), times rest_scale (1; 0.5 and 2).
Depreciation: D_c = b_c d_c', b_c = world composition of country c's GFCF (FIGARO P51G column),
  d_c = CFC/x: own for the 13 countries (labour_block cfc_ratio), median of the 13 otherwise.
Placebos in the same world system (A-V4-MRIO-PLAC): (a) the focus country's permutation of its
  evaluated FIGARO industries applied to every country's coefficient vector; (b) exp(sigma_c eps_j),
  eps_j common to all countries, sigma_c = SD ln l of country c.  (c) mixtures: v3 values (imports at
  price), reference only.
"""
from __future__ import annotations

import sys
from functools import lru_cache

import numpy as np
import pandas as pd

from .. import figaro, oecd
from ..core import leontief_inverse
from ..economy import MERGES, _group_matrix
from ..figaro import INDUSTRIES
from ..v2.placebo_cost import metrics_mat
from ..v3.placebo_disp import COUNTRIES, N_P, YEARS
from .splits import OUT, ROOT

PROXY_POOL = ["BR", "CN", "IN", "ID", "AR", "RU", "SA"]
N = len(INDUSTRIES)
EVAL = np.array([j not in ("T", "U") for j in INDUSTRIES])


@lru_cache(None)
def _avg_hours():
    d = pd.read_csv(ROOT / "data" / "raw" / "v4" / "oecd_avg_hours.csv")
    d = d[d.WORKER_STATUS == "_T"]
    return d.groupby(["REF_AREA", "TIME_PERIOD"]).OBS_VALUE.mean()


def avg_hours(c3, year):
    s = _avg_hours()
    if c3 not in s.index.get_level_values(0):
        return 1800.0, False
    s = s.loc[c3]
    y = min(s.index, key=lambda t: abs(t - year))
    return float(s[y]), True


@lru_cache(None)
def focus_labour(c3, year, xs_key):
    xs = pd.Series(np.array(xs_key), index=INDUSTRIES)
    lab = oecd.labour_block(c3, year, xs)
    return lab["hours"].reindex(INDUSTRIES).fillna(0).to_numpy(), lab["cfc_ratio"].reindex(INDUSTRIES).to_numpy()


def world_system(year, rest_scale=1.0):
    d = figaro.load(year)
    C = [str(c) for c in d["countries"]]
    x = d["Z"].sum(1) + d["F"].sum(1)
    hours = np.full(len(x), np.nan)
    cfcr = np.full(len(x), np.nan)
    log = {}
    focus2 = {oecd.ISO3_TO_2[c]: c for c in COUNTRIES}
    for i, c2 in enumerate(C):
        sl = slice(i * N, (i + 1) * N)
        if c2 in focus2:
            h, r = focus_labour(focus2[c2], year, tuple(x[sl]))
            hours[sl], cfcr[sl] = h, r
            log[c2] = "own hours"
            continue
        c3 = oecd.ISO2_TO_3.get(c2)
        if c3 is None:
            continue
        p = oecd.employment_persons_timbc(c3, min(max(year, 2008), 2022), pd.Series(x[sl], index=INDUSTRIES))
        if p.notna().sum() > 0:
            ah, ok = avg_hours(c3, year)
            hours[sl] = p.fillna(0).to_numpy() * ah          # TiMBC EMPN is in persons (checked: BE 4.9m)
            log[c2] = f"TiMBC x {'OECD' if ok else '1800'} h"
    lcoef = np.divide(hours, x, out=np.full(len(x), np.nan), where=x > 0)
    pool = [C.index(c) for c in PROXY_POOL if c in C and not np.isnan(lcoef[C.index(c) * N:(C.index(c) + 1) * N]).all()]
    Lp = np.array([lcoef[p * N:(p + 1) * N] for p in pool])
    Xp = np.array([x[p * N:(p + 1) * N] for p in pool])
    avg = np.nansum(Lp * Xp, 0) / np.nansum(np.where(np.isnan(Lp), 0, Xp), 0)
    for i, c2 in enumerate(C):
        sl = slice(i * N, (i + 1) * N)
        if np.isnan(lcoef[sl]).all():
            lcoef[sl] = avg * rest_scale
            log[c2] = f"proxy pool x {rest_scale}"
    lcoef = np.nan_to_num(lcoef)
    # depreciation in every country
    med = np.nanmedian(np.array([cfcr[C.index(oecd.ISO3_TO_2[c]) * N:(C.index(oecd.ISO3_TO_2[c]) + 1) * N]
                                 for c in COUNTRIES]), 0)
    K = len(figaro.FD)
    P51 = list(figaro.FD).index("P51G")
    gos_row = list(d["varows"]).index("B2A3G")
    with np.errstate(divide="ignore", invalid="ignore"):
        A = np.where(x[None, :] > 0, d["Z"] / x[None, :], 0.0)
    for i, c2 in enumerate(C):
        sl = slice(i * N, (i + 1) * N)
        r = cfcr[sl] if np.isfinite(cfcr[sl]).any() else med
        r = np.where(np.isfinite(r), r, np.nanmedian(r))
        cfc = np.minimum(r * x[sl], np.clip(d["V"][gos_row, sl], 0, None))
        g = d["F"][:, i * K + P51]
        if g.sum() <= 0:
            continue
        dcoef = np.divide(cfc, x[sl], out=np.zeros(N), where=x[sl] > 0)
        A[:, sl] += np.outer(g / g.sum(), dcoef)
    Lw = leontief_inverse(A, check=False)
    return dict(C=C, x=x, lcoef=lcoef, Lw=Lw, log=log)


def focus_eval(W, c3, V):
    """Aggregate world contents V (P, n_world) of country c3 to its labels (output-weighted), return
    z (P, n_labels) normalised to sum z x = sum x, x, eval mask."""
    c2 = oecd.ISO3_TO_2[c3]
    ci = W["C"].index(c2)
    sl = slice(ci * N, (ci + 1) * N)
    names, G = _group_matrix(INDUSTRIES, MERGES.get(c2, []))
    xs = W["x"][sl]
    xg = G @ xs
    keep = xg > 0
    v = (np.atleast_2d(V)[:, sl] * xs[None]) @ G.T
    v = v[:, keep] / xg[keep][None]
    X = xg[keep]
    z = v * X.sum() / (v @ X)[:, None]
    labs = [n for n, k in zip(names, keep) if k]
    mask = np.array([not any(m in ("T", "U") for m in lab.split("+")) for lab in labs])
    return z, X, mask, labs


def run(years=YEARS, rest_scales=(1.0, 0.5, 2.0)):
    rows = []
    for y in years:
        for rs in rest_scales:
            W = world_system(y, rs)
            C = W["C"]
            lc = W["lcoef"]
            v_lab = lc @ W["Lw"]
            nC = len(C)
            Lmat = lc.reshape(nC, N)
            sig = np.array([np.log(r[r > 0]).std() if (r > 0).any() else 0 for r in Lmat])
            for c3 in COUNTRIES:
                rng = np.random.default_rng(4040 + sum(map(ord, c3)))
                ev = np.where(EVAL)[0]
                perms = np.tile(np.arange(N), (N_P, 1))
                for p in range(N_P):
                    perms[p, ev] = ev[rng.permutation(len(ev))]
                eps = rng.normal(0, 1, (N_P, N))
                zl, X, m, _ = focus_eval(W, c3, v_lab)
                ml = metrics_mat(zl[:, m], X[m])
                for kind in ("perm", "lnorm"):
                    if kind == "perm":
                        Pl = Lmat[:, perms].transpose(1, 0, 2).reshape(N_P, nC * N)
                    else:
                        Pl = (np.exp(eps[:, None, :] * sig[None, :, None]) * (Lmat > 0)[None]).reshape(N_P, nC * N)
                    Vp = Pl @ W["Lw"]
                    zp, _, _, _ = focus_eval(W, c3, Vp)
                    mp = metrics_mat(zp[:, m], X[m])
                    for met in ("mawd", "d"):
                        rows.append(dict(country=c3, year=y, rest_scale=rs, kind=kind, metric=met,
                                         labour=float(ml[met][0]), share_better=float((mp[met] < ml[met][0]).mean())))
            print("a2", y, rs, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "a2_mrio_placebo.csv", index=False)
    pd.Series(W["log"]).to_csv(OUT / "a2_labour_sources.csv")


if __name__ == "__main__":
    run()
