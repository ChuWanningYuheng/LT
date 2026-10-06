"""Iteration 9, stage 1.1: tails of iteration 8 (pre_registration_v9.md, stage 1.1).

* B3 panel without aggregate / sub-industry overlaps; outcomes 6, 17, 19 and the b table recomputed.
* Disaggregation gradient: paired cluster bootstrap over summary groups (exploratory).
* Phases: US counted once, cluster bootstrap over phase windows, mid-phase window variant.
* Real revaluation of capital in crisis vs other years (outcome 20).
Outputs: results/v9/s1_*.csv
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy import stats

from ..v4 import b_tests
from ..v4.b_data import ALT
from ..v6.series import ROOT
from ..v8.rule import decide

OUT = ROOT / "results" / "v9"
OUT.mkdir(parents=True, exist_ok=True)
AGG = {a: [k for k, v in ALT.items() if v == a] for a in set(ALT.values())}


# ---------------------------------------------------------------- panel without overlaps
def noovl_panel():
    p = pd.read_csv(ROOT / "results" / "v4" / "b_panel.csv.gz")
    drop = np.zeros(len(p), bool)
    present = p.groupby(["geo", "year"]).ind.apply(set)
    for i, r in enumerate(p[["geo", "year", "ind"]].itertuples(index=False)):
        subs = AGG.get(r.ind)
        if subs and present[(r.geo, r.year)] & set(subs):
            drop[i] = True
    out = p[~drop]
    out.to_csv(OUT / "b_panel.csv.gz", index=False)
    return out, int(drop.sum())


def use_noovl():
    """Point the iteration-4 loader (and everything built on it) to the panel without overlaps."""
    b_tests.OUT = OUT


def recompute():
    from ..v7.extra import k2
    from ..v7.part1 import klems_panel, make_a, threshold
    from ..v8.stage2 import bfit, klems
    rows = []
    for tag in ("v4 panel", "no overlaps"):
        if tag == "no overlaps":
            use_noovl()
        p = b_tests.load("primary")
        r = b_tests.b31(p)
        rows.append(dict(panel=tag, item="outcome 6: B3.1 beta", est=r["beta"], se=r["se"], n=r["n"]))
        kp = klems_panel()
        for k in k2(kp):
            rows.append(dict(panel=tag, item=k["check"], est=k["est"], se=k["se"], n=k["n"]))
        a = kp[kp.U_lag.notna()].copy()
        _, m0, ms = threshold(a, make_a())
        rows.append(dict(panel=tag, item="iteration 7: m*_0 / m*_s (a)", est=m0, se=ms, n=len(a)))
        d = klems()
        f = bfit(d, np.log(d.PI / d.W), np.log(d.K / d.W))
        rows.append(dict(panel=tag, item="outcome 19: b KLEMS (log, PI > 0)", est=f["b"], se=f["se"], n=f["n"]))
    b_tests.OUT = ROOT / "results" / "v4"
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- gradient, paired bootstrap
def gradient_paired(B=2000, coverage_fix=False, E_override=None, tag="", indep=False):
    from ..core import leontief_inverse
    from ..v2.placebo_cost import metrics_mat
    from ..v8.stage3 import bea_detail_system, concordance, detail_employment, level_maps
    inds, Z, m, x, comp = bea_detail_system()
    E, matched, info = detail_employment(inds, comp, coverage_fix)
    if E_override is not None:
        E = E_override.reindex(inds).fillna(E)
    gov = np.array([i.startswith(("S00", "GSLG")) or i in ("531HSO", "531HST", "814000") for i in inds])
    ev_det = matched.to_numpy() & ~gov & (x > 0)
    cc = concordance()
    summ_of = {i: cc.summary.get(i, i) for i in inds}
    lev_obj = {}
    for lev, keys in level_maps(inds).items():
        labs = list(dict.fromkeys(keys))
        G = np.zeros((len(labs), len(inds)))
        for j, k in enumerate(keys):
            G[labs.index(k), j] = 1
        xg, Zg, mg, Eg = G @ x, G @ Z @ G.T, G @ m, G @ E.to_numpy()
        ev = (G @ ev_det.astype(float)) == G.sum(1)
        ok = xg > 0
        A = np.divide(Zg, xg, out=np.zeros_like(Zg), where=xg > 0)
        L = leontief_inverse(A)
        mu = np.divide(mg, xg, out=np.zeros(len(xg)), where=ok) @ L

        def z(rows):
            lam = np.atleast_2d(rows) @ L
            sc = (xg.sum() - mu @ xg) / (lam @ xg)
            return sc[:, None] * lam + mu[None]
        l = np.divide(Eg, xg, out=np.zeros(len(xg)), where=ok)
        rng = np.random.default_rng(sum(map(ord, f"BEA2017_{lev}")))
        idx = np.where(ev)[0]
        perm = np.tile(l, (1000, 1))
        for p_ in range(1000):
            perm[p_, idx] = l[idx][rng.permutation(len(idx))]
        first = {lab: inds[int(np.argmax(G[i]))] for i, lab in enumerate(labs)}
        grp = np.array([summ_of[first[lab]] for lab in labs])[ev]
        lev_obj[lev] = dict(Zh=z(l)[:, ev], Zf=z(ok.astype(float))[:, ev], Zp=z(perm)[:, ev], xe=xg[ev], grp=grp)

    def stats_on(o, ii):
        h = metrics_mat(o["Zh"][:, ii], o["xe"][ii])
        f = metrics_mat(o["Zf"][:, ii], o["xe"][ii])
        pm = metrics_mat(o["Zp"][:, ii], o["xe"][ii])
        return dict(perm_mawd=float((pm["mawd"] < h["mawd"][0]).mean()), perm_d=float((pm["d"] < h["d"][0]).mean()),
                    fmh=float(f["mawd"][0] - h["mawd"][0]))
    point = {lev: stats_on(o, np.arange(len(o["xe"]))) for lev, o in lev_obj.items()}
    groups = np.unique(np.concatenate([o["grp"] for o in lev_obj.values()]))
    members = {lev: {g: np.where(o["grp"] == g)[0] for g in groups} for lev, o in lev_obj.items()}
    rng = np.random.default_rng(909)
    draws = []
    for _ in range(B):
        gs = rng.choice(groups, len(groups), replace=True)
        r = {}
        for lev in lev_obj:
            ii = np.concatenate([members[lev][g] for g in gs])
            r[lev] = stats_on(lev_obj[lev], ii) if len(ii) > 5 else None
        if all(v is not None for v in r.values()):
            draws.append(r)
    ind_draws = {}
    if indep:                                                  # pre-registered scheme of iteration 8 (independent levels)
        for lev, o in lev_obj.items():
            J = len(o["xe"])
            I = np.random.default_rng(7 + J).integers(0, J, (B, J))
            ind_draws[lev] = [stats_on(o, I[b]) for b in range(B)]
    rows = []
    for k, theta_d, direction in (("perm_mawd", 0.1, "<"), ("perm_d", 0.1, "<"), ("fmh", 0.02, ">")):
        est = point["detail"][k] - point["summary"][k]
        if indep:
            di = np.array([a[k] - b_[k] for a, b_ in zip(ind_draws["detail"], ind_draws["summary"])])
            ilo, ihi = np.quantile(di, [0.05, 0.95])
        dd = np.array([d["detail"][k] - d["summary"][k] for d in draws])
        lo, hi = np.quantile(dd, [0.05, 0.95])
        rows.append(dict(variant=tag or ("coverage fix" if coverage_fix else "main"), stat=k, detail=point["detail"][k],
                         naics3=point["naics3"][k], summary=point["summary"][k], diff=est, ci90_lo=lo, ci90_hi=hi,
                         boot_median=float(np.median(dd)), n_draws=len(dd),
                         label=decide(est, lo, hi, 0, theta_d, direction),
                         **(dict(indep_lo=ilo, indep_hi=ihi, label_prereg=decide(est, ilo, ihi, 0, theta_d, direction))
                            if indep else {})))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- phases
def phases_v9(B=2000, mid=False):
    from ..v6.tests import contrib, logs
    from ..v8.stage4 import EU_CRISES, MAIN8, US_PEAKS, phases_from, series_all
    s = series_all()
    rows = []
    sets = [("USA_NFC", US_PEAKS)] + [(g, EU_CRISES) for g in MAIN8 if g != "USA"]
    for geo, cr in sets:
        g = s[(s.geo == geo) & (s.variant == "main")].set_index("year").sort_index()
        if geo == "USA_NFC":
            g = g.loc[1951:]
            bounds = [1950] + US_PEAKS + [2025]
            ph = [(a + 1, b) for a, b in zip(bounds[:-1], bounds[1:]) if b - a >= 3]
        else:
            ph = phases_from(list(g.index), cr, int(g.index.max()))
        for a, b in ph:
            if mid:
                a = a + (b - a + 1) // 2
            x = g.loc[a:b]
            if len(x) < 3:
                continue
            a, b = int(x.index.min()), int(x.index.max())
            sl = np.polyfit(x.index, x.rM, 1)[0]
            dk = np.log1p(x.k.iloc[-1]) - np.log1p(x.k.iloc[0])
            cl = f"US{a}" if geo == "USA_NFC" else f"EU{b}"
            rows.append(dict(geo=geo, start=a, end=b, cluster=cl, rM_falls=bool(sl < 0), k_rises=bool(dk > 0)))
    P = pd.DataFrame(rows)
    cl = P.cluster.unique()
    rng = np.random.default_rng(404)
    out = []
    for col in ("rM_falls", "k_rises"):
        by = P.groupby("cluster")[col].agg(["sum", "size"])
        bs = []
        for _ in range(B):
            c = rng.choice(cl, len(cl), replace=True)
            t = by.loc[c]
            bs.append(t["sum"].sum() / t["size"].sum())
        lo, hi = np.quantile(bs, [0.05, 0.95])
        est = P[col].mean()
        out.append(dict(window="mid-phase to peak" if mid else "after peak to peak", item=col, k=int(P[col].sum()),
                        n=len(P), clusters=len(cl), share=est, ci90_lo=lo, ci90_hi=hi,
                        label=decide(est, lo, hi, 0.5, 0.15, ">")))
    return P, pd.DataFrame(out)


# ---------------------------------------------------------------- real revaluation
def real_revaluation():
    from ..v6.series import fa
    from ..v8.stage4 import EU_CRISES, MAIN8, US_PEAKS, series_all
    s = series_all()
    pk = {g: s[(s.geo == g) & (s.variant == "main")].set_index("year").P_K for g in MAIN8}
    K = fa(4, "FAAt401-A", 37)
    I = fa(4, "FAAt407-A", 37)
    D = fa(4, "FAAt404-A", 37)
    us = pd.DataFrame({"K": K, "I": I, "D": D}).loc[1951:2024]
    nom = (us.K - us.K.shift() - (us.I - us.D)) / us.K.shift()
    infl = pk["USA"].pct_change()
    data = [("USA_NFC", (nom - infl.reindex(nom.index)), {y + 1 for y in US_PEAKS} | set(US_PEAKS))]
    for geo in MAIN8:
        if geo == "USA":
            continue
        g = s[(s.geo == geo) & (s.variant == "main")].set_index("year").sort_index()
        rv = (g.K - g.K.shift() - (g.UIGT - g.UKCT)) / g.K.shift() - g.P_K.pct_change()
        data.append((geo, rv, set(EU_CRISES)))
    rows = []
    for geo, rv, cr in data:
        rv = rv.replace([np.inf, -np.inf], np.nan).dropna()
        c, o = rv[rv.index.isin(cr)], rv[~rv.index.isin(cr)]
        t = stats.ttest_ind(c, o, equal_var=False)
        se = np.sqrt(c.var(ddof=1) / len(c) + o.var(ddof=1) / len(o))
        rows.append(dict(geo=geo, real_reval_crisis=c.mean(), real_reval_other=o.mean(), diff=c.mean() - o.mean(),
                         se=se, df=t.df, p=t.pvalue, n_crisis=len(c), n_other=len(o)))
    R = pd.DataFrame(rows)
    u = R.set_index("geo").loc["USA_NFC"]                                   # journal 1: AMECO real revaluation = 0
    tq_u = stats.t.ppf(0.95, u.df)
    us_row = dict(item="outcome 20 (journal 1): US NFC real revaluation, crisis - other", est=u["diff"], se=u.se,
                  ci90_lo=u["diff"] - tq_u * u.se, ci90_hi=u["diff"] + tq_u * u.se, n=int(u.n_crisis + u.n_other),
                  label=decide(u["diff"], u["diff"] - tq_u * u.se, u["diff"] + tq_u * u.se, 0, 0.01, "<"))
    d = R["diff"].to_numpy()
    G = len(d)
    se = d.std(ddof=1) / np.sqrt(G)
    tq = stats.t.ppf(0.95, G - 1)
    est = d.mean()
    summ = dict(item="mean of 8 series (AMECO real revaluation = 0 by construction)", est=est, se=se, ci90_lo=est - tq * se,
                ci90_hi=est + tq * se, n=G, label=decide(est, est - tq * se, est + tq * se, 0, 0.01, "<"))
    return R, [us_row, summ]


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else "all"
    pd.set_option("display.width", 250)
    if part in ("panel", "all"):
        p, nd = noovl_panel()
        print("dropped overlapping aggregate rows:", nd, flush=True)
        R = recompute()
        R.to_csv(OUT / "s1_noovl.csv", index=False)
        print(R.round(4).to_string(), flush=True)
    if part in ("phases", "all"):
        P1, S1 = phases_v9()
        P2, S2 = phases_v9(mid=True)
        pd.concat([P1.assign(window="after peak"), P2.assign(window="mid")]).to_csv(OUT / "s1_phases.csv", index=False)
        S = pd.concat([S1, S2])
        S.to_csv(OUT / "s1_phases_summary.csv", index=False)
        print(S.round(3).to_string(), flush=True)
    if part in ("reval", "all"):
        R, summ = real_revaluation()
        R.to_csv(OUT / "s1_real_reval.csv", index=False)
        pd.DataFrame(summ).to_csv(OUT / "s1_real_reval_summary.csv", index=False)
        print(R.round(4).to_string(), "\n", summ, flush=True)
    if part in ("gradient", "all"):
        G = pd.concat([gradient_paired(), gradient_paired(coverage_fix=True)])
        G.to_csv(OUT / "s1_gradient_paired.csv", index=False)
        print(G.round(4).to_string(), flush=True)
