"""Deviation metrics for value/price ratios.

z_j is the ratio (basis-derived "value") / (market price) of industry j, with market prices = 1.
All metrics are computed on a subset (mask) after re-normalising z so that sum_j z_j x_j = sum_j x_j
on that subset (total value = total price of gross output, the normalisation used by
Ochoa/Shaikh/Mariolis-Tsoulfidis).  Following Mariolis & Tsoulfidis (2010), MAD and MAWD are defined on
price/value ratios p_j/v_j = 1/z_j, and the Steedman-Tomkins d-distance is sqrt(2(1-cos theta)), theta the
angle between the vector of price/value ratios and the vector of ones.
"""
from __future__ import annotations

import numpy as np


def normalise(z: np.ndarray, x: np.ndarray) -> np.ndarray:
    return z * x.sum() / (z @ x)


def ratio_metrics(z: np.ndarray, x: np.ndarray, mask: np.ndarray | None = None,
                  size: np.ndarray | None = None) -> dict:
    if mask is None:
        mask = np.ones(len(z), bool)
    mask = mask & np.isfinite(z) & (z > 0) & (x > 0)
    z, x = z[mask], x[mask]
    z = normalise(z, x)
    w = x / x.sum()
    pv = 1.0 / z
    lz = np.log(z)
    out = dict(
        n=int(mask.sum()),
        cv=float(z.std() / z.mean()),
        cv_w=float(np.sqrt(w @ (z - 1) ** 2)),          # weighted mean of z is 1 after normalisation
        mad=float(np.mean(np.abs(pv - 1))),
        mawd=float(w @ np.abs(pv - 1)),
        d=float(np.sqrt(2 * (1 - pv.sum() / (np.linalg.norm(pv) * np.sqrt(len(pv)))))),
        log_sd_w=float(np.sqrt(w @ (lz - w @ lz) ** 2)),
        # literature-comparable, SIZE-BIASED: correlation of industry totals (value vs price)
        corr_totals=float(np.corrcoef(z * x, x)[0, 1]),
        corr_log_totals=float(np.corrcoef(np.log(z * x), np.log(x))[0, 1]),
    )
    if size is not None:
        s = size[mask]
        ok = s > 0
        # Kliman-type de-sized correlation: totals divided by an independent size measure
        out["corr_desized"] = float(np.corrcoef((z * x / s)[ok], (x / s)[ok])[0, 1])
    return out
