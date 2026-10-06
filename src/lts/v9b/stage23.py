"""Iteration 9b, stages 2-3: world-level industry test with the R18 benchmark (WIOD) and the additive decomposition
of the South -> North flow (EXIOBASE) (pre_registration_v9b.md, stages 2-3).

Outputs: results/v9b/s2_*.csv, s3_*.csv
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from ..v8.rule import decide
from ..v9.stage34 import (GOVL, YEARS, contents, mawd, qmap, t1, world_industries, year_system)
from .stage1 import OUT

B = 2000


# ---------------------------------------------------------------- stage 2 (a): national vs world for lab and cap (R18)
def national_vs_world(y, cont, cells, W, k):
    rows = []
    keep = ~cells.code.isin(GOVL) & (cells.x > 0) & (cells.country != "ROW")
    for c, g in cells[keep].groupby("country"):
        idx = g.index.to_numpy()
        lam = cont[k][idx]
        wld = g.code.map(W[k]).to_numpy()
        if (lam <= 0).any() or len(idx) < 20 or not np.isfinite(wld).all():
            continue
        x = g.x.to_numpy()
        rows.append(dict(year=y, cand=k, country=c, mawd_national=mawd(lam, x)[0], mawd_world_eq=mawd(qmap(wld, lam), x)[0],
                         world_better=mawd(qmap(wld, lam), x)[0] < mawd(lam, x)[0]))
    return rows


def run_years(years=YEARS):
    nat, store = [], {}
    ppp = None
    for y in years:
        A, cells = year_system(y)
        cont = contents(A, cells)
        W, X = world_industries(cont, cells)
        for k in ("hours", "lab", "cap"):
            nat += national_vs_world(y, cont, cells, W, k)
        pidx = cont["_pidx"]
        keep = ~cells.code.isin(GOVL) & (cells.x > 0) & np.isfinite(pidx) & (pidx > 0) & (cells.country != "ROW")
        P = (cells.x[keep].groupby(cells.code[keep]).sum() / (cells.x[keep] / pidx[keep.values]).groupby(cells.code[keep]).sum())
        store[y] = dict(W=W, X=X, P=P)
        if y == 2005:
            keepw = ~cells.code.isin(GOVL) & (cells.x > 0)
            pl = cells.pl1.where(np.isfinite(cells.pl1) & (cells.pl1 > 0), 1.0)
            xt = (cells.x / pl)[keepw]
            Xt = xt.groupby(cells.code[keepw]).sum()
            ppp = dict(W=W, Xt=Xt.reindex(W.index), share_weight1=float((~(np.isfinite(cells.pl1) & (cells.pl1 > 0)))[keepw].mean()))
        pd.DataFrame(nat).to_csv(OUT / "s2_national_vs_world.csv", index=False)
        print("s2", y, flush=True)
    return pd.DataFrame(nat), store, ppp


def outcome44(N):
    piv = N.groupby(["cand", "country"]).world_better.mean().gt(0.5).unstack("cand")
    piv = piv.dropna()
    sh = piv.mean()
    rng = np.random.default_rng(44)
    rows = []
    for other in ("lab", "cap"):
        est = float(sh["hours"] - sh[other])
        bs = []
        a, b = piv["hours"].to_numpy(float), piv[other].to_numpy(float)
        for _ in range(B):
            ii = rng.integers(0, len(a), len(a))
            bs.append(a[ii].mean() - b[ii].mean())
        lo, hi = np.quantile(bs, [0.05, 0.95])
        rows.append(dict(outcome=f"44: share(hours) - share({other}), world beats national", est=est, ci90_lo=lo, ci90_hi=hi,
                         share_hours=float(sh["hours"]), share_other=float(sh[other]), n=len(piv),
                         label=decide(est, lo, hi, 0, 0.1, ">")))
    return pd.DataFrame(rows)


def outcome45(store):
    wins = ((2000, 2005), (2005, 2010), (2009, 2014))
    keys = None
    for a, b in wins:
        k = store[a]["W"].index.intersection(store[b]["W"].index).intersection(store[a]["P"].index).intersection(store[b]["P"].index)
        keys = k if keys is None else keys.intersection(k)
    keys = np.array(sorted(keys))

    def r2s(sel):
        out = {}
        for a, b in wins:
            sa, sb = store[a], store[b]
            dP = np.log(sb["P"].reindex(sel) / sa["P"].reindex(sel)).to_numpy()
            wt = sb["X"].reindex(sel).to_numpy()
            for c in ("hours", "lab", "cap", "energy", "flat_phys"):
                dz = np.log(sb["W"][c].reindex(sel) / sa["W"][c].reindex(sel)).to_numpy()
                out[(a, c)] = t1(dP, dz, wt)
        return out

    def stat(r):
        return float(np.mean([r[(a, "hours")] - max(r[(a, "cap")], r[(a, "energy")]) for a, _ in wins]))
    r = r2s(keys)
    est = stat(r)
    rng = np.random.default_rng(45)
    bs = [stat(r2s(keys[rng.integers(0, len(keys), len(keys))])) for _ in range(B)]
    lo, hi = np.quantile(bs, [0.05, 0.95])
    detail = pd.DataFrame([dict(window=f"{a}-{a + 5 if a != 2009 else 2014}", cand=c, r2=v) for (a, c), v in r.items()])
    return dict(outcome="45: R2(hours) - max(R2 cap, R2 energy), world industries, 5-year changes, mean of windows",
                est=est, ci90_lo=lo, ci90_hi=hi, n=len(keys), label=decide(est, lo, hi, 0, 0.05, ">")), detail


def outcome46(ppp):
    W, Xt = ppp["W"], ppp["Xt"].to_numpy()
    h = W.hours.to_numpy()
    eq = {k: qmap(W[k].to_numpy(), h) for k in ("hours", "lab", "cap", "energy")}
    lv = [dict(cand=k, mawd_eq_ppp=mawd(v, Xt)[0], mawd_raw_ppp=mawd(W[k].to_numpy(), Xt)[0]) for k, v in eq.items()]
    lv.append(dict(cand="flat", mawd_raw_ppp=mawd(W["flat"].to_numpy(), Xt)[0]))
    pt = {k: mawd(v, Xt)[0] for k, v in eq.items()}
    est = pt["hours"] - min(pt["cap"], pt["energy"])
    rng = np.random.default_rng(46)
    bs = []
    for _ in range(B):
        ii = rng.integers(0, len(h), len(h))
        m = {k: mawd(v[ii], Xt[ii])[0] for k, v in eq.items()}
        bs.append(m["hours"] - min(m["cap"], m["energy"]))
    lo, hi = np.quantile(bs, [0.05, 0.95])
    return dict(outcome="46: MAWD(hours) - min(cap, energy), world industries, PPP weights 2005, equal dispersion", est=est,
                ci90_lo=lo, ci90_hi=hi, n=len(h), share_cells_weight1=ppp["share_weight1"],
                label=decide(est, lo, hi, 0, 0.01, "<")), pd.DataFrame(lv)


# ---------------------------------------------------------------- stage 3: additive decomposition (EXIOBASE 2017)
def rows_like(F, *keys):
    r = F.index.astype(str)
    m = np.zeros(len(r), bool)
    for k in keys:
        m |= r.str.startswith(k)
    return F[m].sum(0).to_numpy()


def adj_weights(f, pl_year=2017, adjust_services=True):
    from ..v9.stage5 import ISO2_3, pld_levels, product_sector
    sec = product_sector(f["idx"])
    pls = pld_levels(pl_year)
    reg = f["reg"]
    x, hrs = f["x"], f["hrs"]
    pl = np.array([pls.get((ISO2_3.get(r, ""), s), np.nan) for r, s in zip(reg, sec)])
    if not adjust_services:
        pl[sec == "svc"] = np.nan
    q = x / pl
    ok = np.isfinite(q) & (hrs > 0) & (q > 0)
    prod = np.array(f["idx"].get_level_values(1))
    df = pd.DataFrame({"p": prod, "q": np.where(ok, q, 0), "H": np.where(ok, hrs, 0)})
    pib = (df.groupby("p").q.transform("sum") / df.groupby("p").H.transform("sum")).to_numpy()
    w = np.where(ok, (q / hrs) / pib, 1.0)
    return np.where(np.isfinite(w), w, 1.0), float(1 - (ok * hrs).sum() / hrs.sum())


def decompose_additive(f, F, pl_year=2017, adjust_services=True, reference="north production"):
    x, reg, sN, X, regions = f["x"], f["reg"], f["sN"], f["X"], f["regions"]
    isN = f["isN"]
    hrs = f["hrs"]
    W_ = f["comp"]
    P_ = rows_like(F, "Operating surplus: Remaining net operating surplus", "Operating surplus: Rents on land",
                   "Operating surplus: Royalties on resources")
    R_ = rows_like(F, "Operating surplus: Consumption of fixed capital", "Other net taxes on production")
    w, sh1 = adj_weights(f, pl_year, adjust_services)
    rate = lambda v: np.divide(v, x, out=np.zeros(len(x)), where=x > 0)  # noqa: E731
    XN = X[:, sN].sum(1)                                              # output for North final demand
    XS = X[:, ~sN].sum(1)
    flowSN = lambda v: float((rate(v) * XN)[~isN].sum())            # noqa: E731
    flowNS = lambda v: float((rate(v) * XS)[isN].sum())             # noqa: E731
    H, Hs = flowSN(hrs), flowSN(hrs * w)
    Wf, Pf, Rf = flowSN(W_), flowSN(P_), flowSN(R_)
    if reference == "north production":
        hN = hrs[isN].sum()
        wN, pN, rN = W_[isN].sum() / hN, P_[isN].sum() / hN, R_[isN].sum() / hN
    else:                                                             # North production for exports to the South
        hN = flowNS(hrs)
        wN, pN, rN = flowNS(W_) / hN, flowNS(P_) / hN, flowNS(R_) / hN
    G = H * (wN + pN + rN) - (Wf + Pf + Rf)
    a = (H - Hs) * (wN + pN + rN)
    b = Hs * wN - Wf
    c = Hs * pN - Pf
    d = Hs * rN - Rf
    return dict(pl_year=pl_year, adjust_services=adjust_services, reference=reference, H_SN=H, Hstar_SN=Hs,
                VA_SN_MEUR=Wf + Pf + Rf, W_SN=Wf, P_SN=Pf, R_SN=Rf, wN=wN, pN=pN, rN=rN, VA_at_north_rates=H * (wN + pN + rN),
                G=G, a_productivity=a, b_wage=b, c_profit=c, d_rest=d, check=G - (a + b + c + d), share_a=a / G,
                share_b=b / G, share_c=c / G, share_d=d / G, share_hours_weight1=sh1)


def stage3():
    from ..v9.stage5 import NORTH, NORTH_NARROW, flows, load
    A, idx, Y, x, F = load(2017)
    rows = []
    for nm, north in (("IMF advanced", NORTH), ("narrow North", NORTH_NARROW)):
        f = flows(A, idx, Y, x, F, north)
        specs = [dict(), dict(adjust_services=False), dict(pl_year=2011), dict(reference="north exports")] \
            if nm == "IMF advanced" else [dict()]
        for sp in specs:
            r = decompose_additive(f, F, **sp)
            r["north"] = nm
            rows.append(r)
            print(nm, sp, {k: round(v, 3) for k, v in r.items() if k.startswith("share")}, flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    part = sys.argv[1] if len(sys.argv) > 1 else "all"
    if part in ("world", "all"):
        N, store, ppp = run_years()
        o44 = outcome44(N)
        o45, d45 = outcome45(store)
        o46, l46 = outcome46(ppp)
        O = pd.concat([o44, pd.DataFrame([o45, o46])], ignore_index=True)
        O.to_csv(OUT / "s2_outcomes.csv", index=False)
        d45.to_csv(OUT / "s2_changes_world.csv", index=False)
        l46.to_csv(OUT / "s2_ppp_levels_2005.csv", index=False)
        print(O.round(4).to_string(), "\n", d45.round(3).to_string(), "\n", l46.round(3).to_string())
    if part in ("unequal", "all"):
        D = stage3()
        D.to_csv(OUT / "s3_decomposition_additive.csv", index=False)
        print(D.round(3).T.to_string())
