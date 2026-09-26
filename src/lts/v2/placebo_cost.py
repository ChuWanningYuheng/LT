"""Iteration 2, stage 1c step 2: placebo resources with the same cost share as labour.

A placebo resource consists of fractions omega_k in (0,1] of the actual use of a random set K of goods,
drawn in random order with omega ~ U(0.2, 1) until the resource's total cost sum_k omega_k (M[k,:] x)
equals labour's share of gross output (sum labour income / sum x); the last good is trimmed (A-PLAC-COST).
It is a primary input with direct requirement a = sum_k omega_k M[k,:] and content
lambda = a (I - M)^{-1} = omega (L - I); goods keep being produced. Imports at price as for labour.
Set and weights are fixed over time (construction in the first year, applied to every year).
'Universal' placebos: weighted CV of a_j across industries within +-20% of the CV of labour income per
unit of output.
Static metrics for 13 countries; T1/T4 for USA, DEU, MEX.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..core import leontief_inverse
from ..economy import build
from ..levels import mask_for

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "v2"
N_POOL, N_MAIN = 5000, 1000


def wcv(v, w):
    w = w / w.sum()
    m = (w * v).sum()
    return float(np.sqrt((w * (v - m) ** 2).sum()) / m)


def metrics_mat(Z: np.ndarray, x: np.ndarray) -> dict:
    """Z: (P, J) ratios on evaluated industries, x: (J,). Normalise each row, return mawd, d, cv_w."""
    Z = Z * x.sum() / (Z @ x)[:, None]
    w = x / x.sum()
    q = 1 / Z
    mawd = (np.abs(q - 1) * w).sum(1)
    cos = q.sum(1) / (np.linalg.norm(q, axis=1) * np.sqrt(Z.shape[1]))
    d = np.sqrt(np.clip(2 * (1 - cos), 0, None))
    cv = np.sqrt(((Z - 1) ** 2 * w).sum(1))
    return dict(mawd=mawd, d=d, cv_w=cv)


def draw_placebos(M, x, target, rng, n):
    goods = np.where(M.sum(1) > 0)[0]
    cost = M @ x                                   # cost of each good's intermediate (+capital) use
    Om = np.zeros((n, M.shape[0]))
    for p in range(n):
        tot = 0.0
        for k in rng.permutation(goods):
            w = rng.uniform(0.2, 1.0)
            if tot + w * cost[k] >= target:
                Om[p, k] = (target - tot) / cost[k]
                break
            Om[p, k] = w
            tot += w * cost[k]
    return Om


def run(countries=None, dyn_countries=("USA", "DEU", "MEX"), years=range(2010, 2023), seed=77):
    from .decomposition import COUNTRIES
    countries = countries or COUNTRIES
    OUT.mkdir(parents=True, exist_ok=True)
    stat_rows, dyn_frames = [], []
    for c in countries:
        rng = np.random.default_rng(seed + sum(map(ord, c)))
        Om, labels0, univ = None, None, None
        for y in years:
            e, _ = build(c, y)
            M, Mm, _ = e.system(True, False, "price", e.l())
            x, h = e.x, e.l()
            L = leontief_inverse(M)
            mu = Mm.sum(0) @ L
            lab_share = e.labour_income.sum() / x.sum()
            if Om is None:
                Om = draw_placebos(M, x, lab_share * x.sum(), rng, N_POOL)
                labels0 = list(e.labels)
                A = Om @ M                                             # direct requirement per unit of output
                cv_lab = wcv(e.labour_income / x, x)
                cvs = np.array([wcv(a, x) for a in A])
                univ = np.abs(cvs / cv_lab - 1) <= 0.2
                nK = (Om > 0).sum(1)
            assert list(e.labels) == labels0
            lam = Om @ (L - np.eye(e.n))                                # (P, n)
            e_s = (x.sum() - mu @ x) / (lam @ x)
            Z = e_s[:, None] * lam + mu[None]
            vL = h @ L
            zL = (x.sum() - mu @ x) / (vL @ x) * vL + mu
            mask = mask_for(e.labels, set())
            mP = metrics_mat(Z[:, mask], x[mask])
            mL = metrics_mat(zL[None, mask], x[mask])
            for met in ("mawd", "d", "cv_w"):
                for grp, sel in (("all_placebos", slice(0, N_MAIN)), ("universal", univ)):
                    vals = mP[met][sel]
                    if len(vals) == 0:
                        stat_rows.append(dict(country=c, year=y, metric=met, group=grp, n=0, labour=float(mL[met][0]),
                                              cv_labour_coef=cv_lab, cv_placebo_median=float(np.median(cvs))))
                        continue
                    stat_rows.append(dict(country=c, year=y, metric=met, group=grp, n=len(vals),
                                          labour=float(mL[met][0]), share_better=float((vals < mL[met][0]).mean()),
                                          p05=float(np.quantile(vals, 0.05)), median=float(np.median(vals)),
                                          cost_share_labour=lab_share,
                                          cost_share_placebo_mean=float((Om[sel] @ (M @ x)).mean() / x.sum()),
                                          n_goods_median=float(np.median(nK[sel])),
                                          cv_labour_coef=cv_lab, cv_placebo_median=float(np.median(cvs[sel]))))
            if c in dyn_countries:
                keep = np.r_[np.arange(N_MAIN)]
                P = e.meta["price_index"]
                frames = [pd.DataFrame(dict(country=c, year=y, basis="labour", industry=e.labels, z=zL, x=x, P=P))]
                for i in keep:
                    frames.append(pd.DataFrame(dict(country=c, year=y, basis=f"pc_{i:04d}{'u' if univ[i] else ''}",
                                                    industry=e.labels, z=Z[i], x=x, P=P)))
                dyn_frames.append(pd.concat(frames, ignore_index=True))
            print("placebo_cost", c, y, "universal:", int(univ.sum()), flush=True)
    pd.DataFrame(stat_rows).to_csv(OUT / "placebo_cost_static.csv", index=False)
    if dyn_frames:
        from ..dynamics import prepare, t1_constrained_r2, t4_out_of_sample
        df = pd.concat(dyn_frames, ignore_index=True)
        d = prepare(df, "all")
        T1 = t1_constrained_r2(d, horizons=(1, 5))
        T4 = t4_out_of_sample(d, [(2016, 2017, 2022)])
        T1.to_csv(OUT / "placebo_cost_T1.csv", index=False)
        T4.to_csv(OUT / "placebo_cost_T4.csv", index=False)


if __name__ == "__main__":
    run()
