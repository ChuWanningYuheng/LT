"""Iteration 5, stage 3.2: agent-based price model with several resources (pre_registration_v5.md 3.2,
journal item 9).

Goods: 0-2 consumer goods, 3 energy, 4 machines (circulating: used up in one period).  Technology A = kappa*A0
(rows 3 and 4 only), direct labour l; kappa sets the aggregate direct-labour share of costs at value prices
(0.8 ... 0.05).  Firms (8 per good) post prices, buy inputs, hire labour at the fixed money wage w = 1, produce;
goods produced in t are sold in t+1, unsold goods perish.  Buyers take the best of 3 random sellers, within
their money, at most 5 transactions per good and period.  Prices move by a random factor U(0, 0.05)
towards clearing inventories.  Rules for the allocation of capacity X_j (total employment fixed):
  K0 none;  K1 by profit rate on costs;  K2 by margin (profit/revenue);  K3 simple commodity economy:
  no capitalists, producers keep revenue minus input costs, labour moves by income per hour.
Benchmarks: labour values, prices of production at the realised profit rate, energy values, flat.

Outputs: results/v5/s3_abm.csv (one row per run), s3_abm_summary.csv
"""
from __future__ import annotations

import sys
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "v5"
NG, EN, MA = 5, 3, 4
SHARES = (0.8, 0.6, 0.4, 0.2, 0.05)
RULES = {"K0": 0, "K1": 1, "K2": 2, "K3": 3}


# ---------------------------------------------------------------- technology and benchmarks
def balanced(A, l, b):
    L = np.linalg.inv(np.eye(NG) - A)
    v = l @ L
    c = np.zeros(NG)
    c[:3] = b / v[:3]
    return L @ c, v


def labour_share(A, l, b):
    x, v = balanced(A, l, b)
    return (l @ x) / (v @ x)


def technology(rng, s):
    A0 = np.zeros((NG, NG))
    for j in range(NG):
        if j != EN:
            A0[EN, j] = rng.uniform(0.05, 0.3)
        A0[MA, j] = rng.uniform(0.05, 0.3)
    l = rng.uniform(0.5, 2.0, NG)
    b = rng.dirichlet(np.ones(3))
    rho = max(abs(np.linalg.eigvals(A0)))
    lo, hi = 0.0, 0.999999 / rho
    for _ in range(200):
        mid = (lo + hi) / 2
        if labour_share(mid * A0, l, b) > s:
            lo = mid
        else:
            hi = mid
    A = lo * A0
    assert max(abs(np.linalg.eigvals(A))) < 1
    return A, l, b


def values(A, l):
    return l @ np.linalg.inv(np.eye(NG) - A)


def energy_values(A):
    At = A.copy()
    At[EN, :] = 0.0
    return A[EN, :] @ np.linalg.inv(np.eye(NG) - At)


def prod_prices(A, l, r, w=1.0):
    return (1 + r) * w * l @ np.linalg.inv(np.eye(NG) - (1 + r) * A)


def mawd(p, bm, q):
    bm = bm * (p @ q) / (bm @ q)
    return float((q * np.abs(p - bm)).sum() / (q * p).sum())


def d_st(p, bm):
    return float(np.linalg.norm(p / np.linalg.norm(p) - bm / np.linalg.norm(bm)))


# ---------------------------------------------------------------- simulation
@njit(cache=True)
def _buy(buyer_cash, need, good, F, price, inv, cash_f, max_tx, qsold, vsold):
    """buyer with money buyer_cash wants `need` units (np.inf = spend the budget); returns (units, spent)"""
    got = 0.0
    spent = 0.0
    for _ in range(max_tx):
        if need - got <= 1e-12 or buyer_cash - spent <= 1e-12:
            break
        best = -1
        for _k in range(3):
            f = good * F + np.random.randint(F)
            if inv[f] > 1e-12 and (best < 0 or price[f] < price[best]):
                best = f
        if best < 0:
            continue
        q = min(need - got, inv[best], (buyer_cash - spent) / price[best])
        inv[best] -= q
        cash_f[best] += q * price[best]
        qsold[best] += q
        vsold[best] += q * price[best]
        got += q
        spent += q * price[best]
    return got, spent


@njit(cache=True)
def simulate(A, l, b, rule, X0, p0, cash0, wcash0, kcash0, T, seed, cK=1.0, phi=0.02, sigma=0.01, tau=10,
             F=8, NW=100, NK=10, s_adj=0.05, max_tx=5, w=1.0, delta=0.2, buf=0.1, util0=0.8, dbg=False):
    np.random.seed(seed)
    nf = NG * F
    X = X0.copy()
    H = (l * X).sum()
    price = p0.copy()
    cash = cash0.copy()
    wcash = np.full(NW, wcash0 / NW)
    kcash = np.full(NK, kcash0 / NK)
    inv = np.zeros(nf)
    ema = np.zeros(nf)
    istock = np.zeros((nf, 2))
    for f in range(nf):
        inv[f] = X[f // F] / F                       # initial stocks: one period of capacity output
        ema[f] = X[f // F] / F
        istock[f, 0] = A[EN, f // F] * X[f // F] / F
        istock[f, 1] = A[MA, f // F] * X[f // F] / F
    # window accumulators per good
    acc_q = np.zeros(NG)
    acc_v = np.zeros(NG)
    acc_out = np.zeros(NG)
    acc_hours = np.zeros(NG)
    acc_prof = np.zeros(NG)
    acc_cost = np.zeros(NG)
    acc_rev = np.zeros(NG)
    acc_inp = np.zeros(NG)          # input costs
    acc_cw = np.zeros(NG)           # workers' purchases (quantities)
    acc_ck = np.zeros(NG)           # capitalists' purchases
    q34 = np.zeros((2, NG, 2))      # quarters 3 and 4: quantity, value per good
    mv_prof = np.zeros(NG)
    mv_cost = np.zeros(NG)
    mv_rev = np.zeros(NG)
    mv_hours = np.zeros(NG)
    mv_net = np.zeros(NG)
    X_path = np.zeros((T // tau + 1, NG))
    trace = np.zeros((T, 3 * NG + 2))
    recap = 0.0           # output, mean price, closing stock per good; firm cash; household cash
    for t in range(T):
        qsold = np.zeros(nf)
        vsold = np.zeros(nf)
        target = np.zeros(nf)
        stock0 = inv.copy()
        # 1. input market
        # input-sector firms (energy, machines) buy first, then consumer-good firms; random order within each
        order = np.concatenate((EN * F + np.random.permutation(2 * F), np.random.permutation(3 * F)))
        bought = np.zeros((nf, 2))
        inpcost = np.zeros(nf)
        pm = np.zeros(NG)
        for g in range(NG):
            pm[g] = price[g * F:(g + 1) * F].mean()
        # recapitalisation: a firm short of working capital is topped up from household money
        # (capitalists; producer households under K3), so no sector can die out (money is conserved)
        for f in range(nf):
            g = f // F
            cap = X[g] / F
            res_f = 1.5 * ((A[EN, g] * pm[EN] + A[MA, g] * pm[MA]) + (0.0 if rule == 3 else w * l[g])) * cap
            if cash[f] < res_f / 3:
                gap = res_f - cash[f]
                avail = kcash.sum() + wcash.sum()
                take = min(gap, 0.5 * avail)
                if avail > 0:
                    kcash *= 1.0 - take / avail
                    wcash *= 1.0 - take / avail
                cash[f] += take
                recap += take
        for f in order:
            g = f // F
            cap = X[g] / F
            # firms produce at capacity when the price covers unit cost at current input prices (and the
            # wage), otherwise they cut output in proportion (not below 20% of capacity)
            uc = A[EN, g] * pm[EN] + A[MA, g] * pm[MA] + (0.0 if rule == 3 else w * l[g])
            target[f] = cap * min(1.0, max(0.2, price[f] / uc))
            wage_need = 0.0 if rule == 3 else w * l[g] * target[f]
            for k in range(2):
                i = EN if k == 0 else MA
                need = A[i, g] * target[f] - istock[f, k]          # input stocks carry over
                if i == g and need > 0:                             # in-house use of own output first
                    own = min(need, inv[f])
                    inv[f] -= own
                    istock[f, k] += own
                    need -= own
                budget = cash[f] - wage_need
                if need > 0 and budget > 0:
                    got, spent = _buy(budget, need, i, F, price, inv, cash, max_tx, qsold, vsold)
                    cash[f] -= spent
                    istock[f, k] += got
                    inpcost[f] += spent
        if dbg:
            for g in range(NG):
                print(t, g, target[g * F:(g + 1) * F].sum(), istock[g * F:(g + 1) * F, 0].sum(),
                      istock[g * F:(g + 1) * F, 1].sum(), cash[g * F:(g + 1) * F].sum(), inv[g * F:(g + 1) * F].sum(),
                      pm[g])
        # 2. consumer market
        nh = NW + NK
        horder = np.random.permutation(nh)
        for h in horder:
            m0 = wcash[h] if h < NW else kcash[h - NW]
            for j in range(3):
                got, spent = _buy(b[j] * m0, np.inf, j, F, price, inv, cash, max_tx, qsold, vsold)
                if h < NW:
                    wcash[h] -= spent
                    if t >= T // 2:
                        acc_cw[j] += got
                else:
                    kcash[h - NW] -= spent
                    if t >= T // 2:
                        acc_ck[j] += got
        # 3. production, wages, profits
        out = np.zeros(nf)
        for f in range(nf):
            g = f // F
            cap = X[g] / F
            y = target[f]
            if A[EN, g] > 0:
                y = min(y, istock[f, 0] / A[EN, g])
            if A[MA, g] > 0:
                y = min(y, istock[f, 1] / A[MA, g])
            th = y / cap if cap > 0 else 0.0
            wage = 0.0
            if rule != 3:
                wage = w * l[g] * y
                if wage > cash[f]:
                    y = max(0.0, cash[f]) / (w * l[g])
                    wage = w * l[g] * y
                cash[f] -= wage
                for h in range(NW):
                    wcash[h] += wage / NW
            istock[f, 0] -= A[EN, g] * y
            istock[f, 1] -= A[MA, g] * y
            out[f] = y
            rev = vsold[f]
            prof = rev - inpcost[f] - wage
            if rule == 3:
                reserve = 1.5 * (A[EN, g] * pm[EN] + A[MA, g] * pm[MA]) * cap
                pay = max(0.0, min(rev - inpcost[f], cash[f] - reserve))
                cash[f] -= pay
                for h in range(NW):
                    wcash[h] += pay / NW
            else:
                reserve = 1.5 * ((A[EN, g] * pm[EN] + A[MA, g] * pm[MA]) + w * l[g]) * cap
                if prof > 0 and cash[f] > reserve:
                    dv = cK * min(prof, cash[f] - reserve)
                    cash[f] -= dv
                    for h in range(NK):
                        kcash[h] += dv / NK
            mv_prof[g] += prof
            mv_cost[g] += inpcost[f] + wage
            mv_rev[g] += rev
            mv_hours[g] += l[g] * y
            mv_net[g] += rev - inpcost[f]
            if t >= T // 2:
                acc_out[g] += y
                acc_hours[g] += l[g] * y
                acc_prof[g] += prof
                acc_cost[g] += inpcost[f] + wage
                acc_rev[g] += rev
                acc_inp[g] += inpcost[f]
        # 4. prices: sold out -> up, unsold stock -> down
        for f in range(nf):
            if stock0[f] > 1e-12:
                if inv[f] <= 1e-9 * stock0[f]:
                    price[f] *= 1.0 + np.random.random() * s_adj
                else:
                    price[f] *= 1.0 - np.random.random() * s_adj
            g = f // F
            if t >= T // 2:
                acc_q[g] += qsold[f]
                acc_v[g] += vsold[f]
            if t >= T // 2 and t < 3 * T // 4:
                q34[0, g, 0] += qsold[f]
                q34[0, g, 1] += vsold[f]
            elif t >= 3 * T // 4:
                q34[1, g, 0] += qsold[f]
                q34[1, g, 1] += vsold[f]
        ema[:] = 0.8 * ema + 0.2 * qsold
        inv[:] = (1.0 - delta) * inv + out            # unsold stock carries over, decays at rate delta
        for g in range(NG):
            trace[t, g] = out[g * F:(g + 1) * F].sum()
            trace[t, NG + g] = price[g * F:(g + 1) * F].mean()
            trace[t, 2 * NG + g] = inv[g * F:(g + 1) * F].sum()
        trace[t, 3 * NG] = cash.sum()
        trace[t, 3 * NG + 1] = wcash.sum() + kcash.sum()
        # 5. reallocation of capacity
        if (t + 1) % tau == 0:
            X_path[(t + 1) // tau] = X
            if rule > 0:
                m = np.zeros(NG)
                for g in range(NG):
                    if rule == 1:
                        m[g] = mv_prof[g] / mv_cost[g] if mv_cost[g] > 0 else 0.0
                    elif rule == 2:
                        m[g] = mv_prof[g] / mv_rev[g] if mv_rev[g] > 0 else 0.0
                    else:
                        m[g] = mv_net[g] / mv_hours[g] if mv_hours[g] > 0 else 0.0
                wts = mv_hours / max(1e-12, mv_hours.sum())
                mbar = (m * wts).sum()
                for g in range(NG):
                    z = min(1.0, max(-1.0, (m[g] - mbar) / (abs(mbar) + 0.01)))
                    X[g] *= np.exp(phi * z + sigma * np.random.randn())
                X *= H / (l * X).sum()
            mv_prof[:] = 0.0
            mv_cost[:] = 0.0
            mv_rev[:] = 0.0
            mv_hours[:] = 0.0
            mv_net[:] = 0.0
    return (acc_q, acc_v, acc_out, acc_hours, acc_prof, acc_cost, acc_rev, acc_inp, acc_cw, acc_ck,
            q34, X_path, np.array([cash.sum(), wcash.sum(), kcash.sum(), recap]), trace)


# ---------------------------------------------------------------- one run
def run(rule, s, rep, T=4000, cK=1.0, money=2.0, seed=56, x_noise=0.3, p_noise=0.1, markup=1.3):
    rng = np.random.default_rng([seed, RULES[rule], int(s * 100), rep, int(cK * 100)])
    A, l, b = technology(rng, s)
    xb, v = balanced(A, l, b)
    X0 = xb * np.exp(rng.normal(0, x_noise, NG))
    X0 *= 1000.0 / (l @ X0)
    F = 8
    p0 = np.repeat(v * markup, F) * np.exp(rng.normal(0, p_noise, NG * F))
    unit_cost = A.T @ (v * markup) + (0.0 if rule == "K3" else 1.0) * l
    cost_f = np.repeat(unit_cost * X0 / F, F)
    cash0 = money * cost_f
    wage_bill = l @ X0
    M0 = cash0.sum() + wage_bill + (0.2 * wage_bill if rule != "K3" else 0.0)
    res = simulate(A, l, b, RULES[rule], X0, p0, cash0, wage_bill, 0.0 if rule == "K3" else 0.2 * wage_bill,
                   T, int(rng.integers(1 << 30)), cK=cK)
    (q, val, out, hours, prof, cost, rev, inp, cw, ck, q34, Xp, M1, trace) = res
    p = val / np.where(q > 0, q, np.nan)
    row = dict(rule=rule, share=s, rep=rep, cK=cK, money_conserved=abs(M1[:3].sum() / M0 - 1) < 1e-6, recap_per_period=M1[3] / T,
               labour_share_check=labour_share(A, l, b))
    if not np.isfinite(p).all():
        row["ok"] = False
        return row
    # stationarity of relative prices (each quarter's prices divided by their quantity-weighted mean)
    pq3, pq4 = q34[0, :, 1] / q34[0, :, 0], q34[1, :, 1] / q34[1, :, 0]
    rel3, rel4 = pq3 / (q34[0, :, 1].sum() / q34[0, :, 0].sum()), pq4 / (q34[1, :, 1].sum() / q34[1, :, 0].sum())
    stat = np.nanmax(np.abs(rel4 / rel3 - 1))
    r_real = prof.sum() / cost.sum() if rule != "K3" else np.nan
    bms = {"labour": v, "energy": energy_values(A), "flat": np.ones(NG)}
    R = 1 / max(abs(np.linalg.eigvals(A))) - 1
    if rule != "K3" and -0.99 < r_real < R:
        bms["pp"] = prod_prices(A, l, r_real)
    m_ = 0.0
    if rule == "K2":
        m_ = prof.sum() / rev.sum()
        bms["equal_margin"] = l @ np.linalg.inv((1 - m_) * np.eye(NG) - A)
    row.update(ok=True, stationary=stat < 0.02, max_q3q4_dev=stat, r_real=r_real, margin=m_,
               level_drift=float((q34[1, :, 1].sum() / q34[1, :, 0].sum()) / (q34[0, :, 1].sum() / q34[0, :, 0].sum()) - 1),
               recap_share=float(M1[3] / T / max(1e-12, (val.sum() - inp.sum()) / (T - T // 2))),
               sold_share=float(q.sum() / max(1e-12, out.sum())))
    for k, bm in bms.items():
        row[f"mawd_{k}"] = mawd(p, bm, q)
        row[f"d_{k}"] = d_st(p, bm)
        row[f"corr_{k}"] = float(np.corrcoef(p, bm)[0, 1])
        perm = [mawd(p, bm[list(pi)], q) for pi in permutations(range(NG))]
        row[f"perm_better_{k}"] = float(np.mean(np.array(perm) < row[f"mawd_{k}"] - 1e-12))
    # profit in commanded hours vs surplus labour
    Hs = hours.sum()
    net = out - A @ out
    melt = (p @ net) / Hs if Hs > 0 else np.nan
    row["melt"] = melt
    if rule != "K3":
        SL = Hs - v @ cw
        row["profit_hours"] = prof.sum() / melt
        row["surplus_labour"] = SL
        row["ratio_profit_SL"] = row["profit_hours"] / SL if SL != 0 else np.nan
        row["e_value"] = SL / (v @ cw)
    row["sector_hours_share_err"] = float(np.abs(hours / Hs - (l * xb) / (l @ xb)).sum() / 2)
    return row


def _job(a):
    return run(*a[:3], cK=a[3])


def run_grid(reps=10, procs=3, cks=(1.0,)):
    from multiprocessing import Pool
    jobs = [(r, s, k, ck) for ck in cks for r in RULES for s in SHARES for k in range(reps)]
    with Pool(procs) as pool:
        rows = pool.map(_job, jobs)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    d = run_grid(reps, cks=(1.0, 0.8))
    d.to_csv(OUT / "s3_abm.csv", index=False)
    cols = [c for c in d.columns if c.startswith(("mawd_", "perm_better_", "ratio_", "r_real", "stationary", "ok"))]
    s = d.groupby(["cK", "rule", "share"])[cols].mean()
    s.to_csv(OUT / "s3_abm_summary.csv")
    pd.set_option("display.width", 250)
    print(s.round(3).to_string())
