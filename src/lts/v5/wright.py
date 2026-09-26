"""Iteration 5, stage 3.1: replication of Wright (2008), 'The Law of Value in a Dynamic Simple Commodity
Economy', Review of Political Economy 20(3).  Rule set SCE = {R1, P1, C1, O1, {M1, E1}, S1}.

Implementation choices where the paper is silent (A-V5-W*, recorded in ASSUMPTIONS.md):
  W1  construction of l and c with h = sum l_j / c_j = 1: c_j ~ U{2..R} (integer periods), shares
      a ~ Dirichlet(1,...,1), l_j = a_j c_j (real).  Production and consumption deficits accrue through
      accumulators (1/l_j resp. 1/c_j per step), so non-integer periods are allowed.
  W2  money is integer coins, offers discrete U{0..m_i}, exchange price discrete U{p_b..p_s} (rules O1, E1
      say 'discrete interval').  M/N = 2.5 is not an integer: each actor gets floor(M/N) = 2 coins and the
      remaining 100 coins go one by one to random actors.  A continuous-money version was tried first:
      money concentrated, trade dried up and no run converged (journal, iteration 5).
  W3  offers are redrawn at every exchange (O1 does not say how long an offer lives).
  W4  transaction-attempt limit: 5 per actor per market (pre-registered; the paper has a limit, no number).
  W5  convergence: correlation of mean prices over consecutive 1000-step windows with labour values;
      stop when |change| < 0.001 (both windows with trades in every commodity), then sample 5000 steps;
      limit 200 000 steps.
Outputs: results/v5/s3_wright.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "v5"


@njit(cache=True)
def _market(e, d, m, L, N, price_sum, price_n, max_tx):
    tx = np.zeros(N, np.int64)
    unclear = np.ones(L, np.bool_)
    n_unclear = L
    sellers = np.empty(N, np.int64)
    buyers = np.empty(N, np.int64)
    while n_unclear > 0:
        k = np.random.randint(n_unclear)
        j = -1
        for jj in range(L):
            if unclear[jj]:
                if k == 0:
                    j = jj
                    break
                k -= 1
        ns = 0
        nb = 0
        for x in range(N):
            if tx[x] >= max_tx:
                continue
            if e[x, j] > d[x, j]:
                sellers[ns] = x
                ns += 1
            elif d[x, j] > e[x, j]:
                buyers[nb] = x
                nb += 1
        if ns == 0 or nb == 0:
            unclear[j] = False
            n_unclear -= 1
            continue
        s = sellers[np.random.randint(ns)]
        b = buyers[np.random.randint(nb)]
        pb = np.random.randint(m[b] + 1)                  # O1: discrete U{0..m_i}
        ps = np.random.randint(m[s] + 1)
        lo, hi = min(pb, ps), max(pb, ps)
        xp = lo + np.random.randint(hi - lo + 1)          # E1: discrete U{p_b..p_s}
        tx[b] += 1
        tx[s] += 1
        if m[b] >= xp:
            m[b] -= xp
            m[s] += xp
            e[b, j] += 1
            e[s, j] -= 1
            price_sum[j] += xp
            price_n[j] += 1


@njit(cache=True)
def run_sce(l, c, N, M, C, seed, max_steps=200000, window=1000, thresh=0.001, sample=5000, max_tx=5, variant=0):
    """variant 0: rules as written; diagnostic variants (not Wright's): 1 initial division of labour
    efficient; 2 S1 compares the deficit accrued in the period (flow) instead of the stock; 3 unmet
    consumption is forgotten at every S1 sampling (deficit reset to 0 after S1)."""
    np.random.seed(seed)
    L = l.shape[0]
    A = np.random.randint(0, L, N)
    if variant == 1:
        sh = l / c
        cum = np.cumsum(sh) / sh.sum()
        for i in range(N):
            u = (i + 0.5) / N
            k = 0
            while cum[k] < u:
                k += 1
            A[i] = k
    e = np.zeros((N, L), np.int64)
    d = np.zeros((N, L), np.int64)
    m = np.full(N, M // N, np.int64)                         # integer coins; remainder to random actors
    for k in range(M - (M // N) * N):
        m[np.random.randint(N)] += 1
    pacc = np.zeros(N)
    cacc = np.zeros(L)
    T = int(C * c.max())
    prev_err = np.full(N, np.inf)
    d_prev = np.zeros((N, L), np.int64)
    ps = np.zeros(L)
    pn = np.zeros(L)
    corr_prev = np.nan
    converged = -1
    sec = np.zeros(L)
    hist = np.full((max_steps // window + 1, 2), np.nan)     # window correlation, min trades per commodity
    t = 0
    while t < max_steps:
        t += 1
        for i in range(N):                                   # P1
            pacc[i] += 1.0 / l[A[i]]
            while pacc[i] >= 1.0:
                e[i, A[i]] += 1
                pacc[i] -= 1.0
        for j in range(L):                                   # C1 (identical deficits for all actors)
            cacc[j] += 1.0 / c[j]
            while cacc[j] >= 1.0:
                for i in range(N):
                    d[i, j] += 1
                cacc[j] -= 1.0
        for i in range(N):
            for j in range(L):
                o = min(e[i, j], d[i, j])
                e[i, j] -= o
                d[i, j] -= o
        _market(e, d, m, L, N, ps, pn, max_tx)               # M1 + E1 (O1 inside)
        if t % T == 0:                                       # S1
            for i in range(N):
                err = np.sqrt((d[i].astype(np.float64) ** 2).sum())
                if variant == 2:
                    flow = np.sqrt(((d[i] - d_prev[i]).astype(np.float64) ** 2).sum())
                    switch = flow > prev_err[i]
                    prev_err[i] = flow
                    d_prev[i] = d[i]
                else:
                    switch = err > prev_err[i]
                    prev_err[i] = err
                if switch:
                    A[i] = np.random.randint(L)
                    pacc[i] = 0.0
                if variant == 3:
                    d[i] = 0
                    prev_err[i] = err
        if converged > 0:
            for i in range(N):
                sec[A[i]] += 1
            if t - converged >= sample:
                break
        elif t % window == 0:
            hist[t // window, 1] = pn.min()
            if (pn > 0).all():
                p = ps / pn
                cr = np.corrcoef(p, l)[0, 1]
                hist[t // window, 0] = cr
                if np.isfinite(corr_prev) and abs(cr - corr_prev) < thresh:
                    converged = t
                    ps[:] = 0.0
                    pn[:] = 0.0
                corr_prev = cr
            else:
                corr_prev = np.nan
            if converged < 0:
                ps[:] = 0.0
                pn[:] = 0.0
    return ps, pn, converged, t, sec / max(1, t - converged if converged > 0 else 1), hist, (m, A, e, d)


def make_economy(L, R, rng):
    c = rng.integers(2, R + 1, L).astype(float)
    a = rng.dirichlet(np.ones(L))
    return a * c, c


def mawd(p, b, q):
    b = b * (p @ q) / (b @ q)
    return float((q * np.abs(p - b)).sum() / (q * p).sum())


def d_st(p, b):
    return float(np.linalg.norm(p / np.linalg.norm(p) - b / np.linalg.norm(b)))


def one(args):
    L, rep, variant, N, M, R, C, n_perm, seed = args
    rng = np.random.default_rng([seed, L, rep])
    l, c = make_economy(L, R, rng)
    ps, pn, conv, t, sec, hist, _ = run_sce(l, c, N, M, C, int(rng.integers(1 << 30)), variant=variant)
    return l, c, ps, pn, conv, t, sec, hist, rng


def run_all(Ls=range(3, 11), reps=10, variant=0, N=200, M=500, R=20, C=2, n_perm=1000, seed=2008, procs=3):
    from multiprocessing import Pool
    rows = []
    jobs = [(L, rep, variant, N, M, R, C, n_perm, seed) for L in Ls for rep in range(reps)]
    with Pool(procs) as pool:
        res = pool.map(one, jobs)
    for (L, rep, *_), (l, c, ps, pn, conv, t, sec, hist, rng) in zip(jobs, res):
        if True:
            p = ps / np.where(pn > 0, pn, np.nan)
            q = pn / max(1, t - conv) if conv > 0 else pn
            ok = np.isfinite(p).all() and conv > 0
            tr = hist[1:, 1]
            last = np.where(np.isfinite(tr))[0]
            row = dict(L=L, rep=rep, variant=variant, converged=conv > 0, t_conv=conv, t_end=t, h=float((l / c).sum()),
                       min_trades_first_window=float(tr[0]),
                       first_window_no_trade=int(np.argmax(tr == 0) + 1) * 1000 if (tr == 0).any() else -1,
                       min_trades_last_window=float(tr[last[-1]]) if len(last) else np.nan)
            hc = hist[1:, 0]
            fc = np.where(np.isfinite(hc))[0]
            row.update(n_windows_all_traded=len(fc), last_all_traded_corr=float(hc[fc[-1]]) if len(fc) else np.nan)
            if np.isfinite(p).all():
                flat = np.ones(L)
                perm = np.array([mawd(p, rng.permutation(l), q) for _ in range(n_perm)])
                row.update(corr=float(np.corrcoef(p, l)[0, 1]), mawd_v=mawd(p, l, q), mawd_flat=mawd(p, flat, q),
                           d_v=d_st(p, l), d_flat=d_st(p, flat), perm_share_better=float((perm < mawd(p, l, q)).mean()),
                           div_err=float(np.abs(sec / N - l / c).sum() / 2), melt=float((p @ q) / (l @ q)))
            row["ok"] = ok
            rows.append(row)
            print(L, rep, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    if len(sys.argv) > 2 and sys.argv[2] == "diag":
        diag = pd.concat([run_all(Ls=(3, 6, 10), reps=3, variant=v) for v in (1, 2, 3)])
        diag.to_csv(OUT / "s3_wright_diag.csv", index=False)
        print(diag.groupby(["variant", "L"])[["converged", "first_window_no_trade"]].mean())
        sys.exit()
    r = run_all(reps=reps)
    r.to_csv(OUT / "s3_wright.csv", index=False)
    diag = pd.concat([run_all(Ls=(3, 6, 10), reps=3, variant=v) for v in (1, 2, 3)])
    diag.to_csv(OUT / "s3_wright_diag.csv", index=False)
    print(diag.groupby(["variant", "L"])[["converged", "first_window_no_trade"]].mean())
    print(r.groupby("L")[["corr", "mawd_v", "mawd_flat", "perm_share_better", "converged", "t_conv"]].mean().round(3))
