"""Iteration 8: the single inference rule (pre_registration_v8.md, 'Единое правило вывода').

confirmed  - the 90% CI excludes theta0 and the point estimate lies beyond the equivalence zone in the expected direction;
refuted    - the 90% CI lies inside [theta0 - delta, theta0 + delta] (TOST 5%) or entirely on the other side of theta0;
uninformative - otherwise.
"""
from __future__ import annotations

import numpy as np

CONF, REF, UNINF = "подтверждено", "опровергнуто", "неинформативно"


def decide(est, lo, hi, theta0, delta, direction):
    """direction '<' or '>': the side of theta0 that counts 'for' the theory."""
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return UNINF + " (нет оценки неопределённости)"
    if direction == "<":
        if hi < theta0 and est < theta0 - delta:
            return CONF
        if lo > theta0:
            return REF
    else:
        if lo > theta0 and est > theta0 + delta:
            return CONF
        if hi < theta0:
            return REF
    if lo >= theta0 - delta and hi <= theta0 + delta:
        return REF
    return UNINF


def wilson(k, n, z=1.645):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def boot_mean(v, B=2000, seed=8):
    """Percentile 90% CI of the mean over units (countries), 2000 resamples."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    m = v[rng.integers(0, len(v), (B, len(v)))].mean(1)
    return float(np.quantile(m, 0.05)), float(np.quantile(m, 0.95))


def share_row(name, flags, theta0, delta, direction, **extra):
    flags = np.asarray(flags, bool)
    k, n = int(flags.sum()), len(flags)
    lo, hi = wilson(k, n)
    est = k / n if n else np.nan
    return dict(outcome=name, est=est, ci90_lo=lo, ci90_hi=hi, k=k, n=n, theta0=theta0, delta=delta,
                direction=direction, label=decide(est, lo, hi, theta0, delta, direction), **extra)


def mean_row(name, values, theta0, delta, direction, **extra):
    v = np.asarray(values, float)
    lo, hi = boot_mean(v)
    est = float(np.nanmean(v))
    return dict(outcome=name, est=est, ci90_lo=lo, ci90_hi=hi, n=int(np.isfinite(v).sum()), theta0=theta0,
                delta=delta, direction=direction, label=decide(est, lo, hi, theta0, delta, direction), **extra)
