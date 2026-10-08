"""Iteration 10, stage 5: distribution of firm profit rates (Farjoun-Machover vs Laplace/Subbotin), SEC FSDS 10-K.

Outputs: results/v10/s5_*.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

from ..v6.series import ROOT
from ..v8.rule import decide, share_row

OUT = ROOT / "results" / "v10"
OUT.mkdir(parents=True, exist_ok=True)
SEC = ROOT / "data" / "raw" / "v10" / "sec" / "sec_firm_fy_all.csv"
YEARS = range(2011, 2026)


def sample():
    d = pd.read_csv(SEC, low_memory=False)
    d = d[(d.countryba == "US") & d.fy.between(2011, 2025)].copy()
    d["sic"] = pd.to_numeric(d.sic, errors="coerce")
    d = d[~d.sic.between(6000, 6799) & d.sic.notna()]
    d = d[d.Assets >= 1e6]
    rev = d.Revenues.fillna(d.RevenueFromContractWithCustomerExcludingAssessedTax).fillna(d.SalesRevenueNet)
    d["rev"] = rev
    d["r_op"] = d.OperatingIncomeLoss / d.Assets
    d["r_ni"] = d.NetIncomeLoss / d.Assets
    d["r_pp"] = d.OperatingIncomeLoss / d.PropertyPlantAndEquipmentNet.where(d.PropertyPlantAndEquipmentNet > 0)
    d["r_cost"] = d.OperatingIncomeLoss / (rev - d.OperatingIncomeLoss).where(rev - d.OperatingIncomeLoss > 0)
    d["r_lab"] = d.OperatingIncomeLoss / d.LaborAndRelatedExpense.where(d.LaborAndRelatedExpense > 0)
    return d


# ---------------------------------------------------------------- asymmetric Subbotin (Bottazzi) density
def aep_logpdf(x, m, al, ar, bl, br):
    z = np.where(x < m, ((m - x) / al) ** bl / bl, ((x - m) / ar) ** br / br)
    A = al * bl ** (1 / bl) * special.gamma(1 + 1 / bl) + ar * br ** (1 / br) * special.gamma(1 + 1 / br)
    return -z - np.log(A)


def fit_aep(x):
    m0, s0 = np.median(x), np.std(x)

    def nll(p):
        m, la, ra, lb, rb = p
        al, ar, bl, br = np.exp(la), np.exp(ra), np.exp(lb), np.exp(rb)
        v = aep_logpdf(x, m, al, ar, bl, br)
        return -np.sum(v) if np.all(np.isfinite(v)) else 1e12
    best = None
    for b0 in (0.0, np.log(0.7)):
        r = optimize.minimize(nll, [m0, np.log(s0 / 2), np.log(s0 / 2), b0, b0], method="Nelder-Mead",
                              options={"maxiter": 6000, "xatol": 1e-6, "fatol": 1e-6})
        if best is None or r.fun < best.fun:
            best = r
    return -best.fun, best.x


def fits(x):
    n = len(x)
    res = {}

    def add(name, ll, k, params):
        res[name] = dict(ll=ll, k=k, bic=k * np.log(n) - 2 * ll, aic=2 * k - 2 * ll, params=params)
    p = stats.norm.fit(x)
    add("normal", stats.norm.logpdf(x, *p).sum(), 2, p)
    p = stats.laplace.fit(x)
    add("laplace", stats.laplace.logpdf(x, *p).sum(), 2, p)
    p = stats.laplace_asymmetric.fit(x)
    add("asym_laplace", stats.laplace_asymmetric.logpdf(x, *p).sum(), 3, p)
    p = stats.gennorm.fit(x)
    add("subbotin", stats.gennorm.logpdf(x, *p).sum(), 3, p)
    try:
        ll, p = fit_aep(x)
        add("asym_subbotin", ll, 5, tuple(p))
    except Exception:                                              # noqa: BLE001
        pass
    # shifted gamma: loc free, start below the minimum
    best = None
    for a0 in (2.0, 5.0, 20.0):
        try:
            p = stats.gamma.fit(x, a0, loc=x.min() - 0.05, scale=np.std(x) / np.sqrt(a0))
            ll = stats.gamma.logpdf(x, *p).sum()
            if np.isfinite(ll) and (best is None or ll > best[0]):
                best = (ll, p)
        except Exception:                                          # noqa: BLE001
            continue
    if best is not None:
        add("gamma_shift", best[0], 3, best[1])
    return res


def by_year(d, col="r_op", trim=1.0):
    rows, shapes = [], []
    for y in YEARS:
        x = d.loc[d.fy == y, col].dropna().to_numpy()
        x = x[np.abs(x) <= trim]
        if len(x) < 1000:
            continue
        f = fits(x)
        q = np.quantile(x, [0.05, 0.25, 0.5, 0.75, 0.95])
        rec = dict(year=y, measure=col, n=len(x), q05=q[0], q25=q[1], q50=q[2], q75=q[3], q95=q[4],
                   subbotin_beta=f["subbotin"]["params"][0])
        for k, v in f.items():
            rec[f"bic_{k}"] = v["bic"]
        best = min(f, key=lambda k: f[k]["bic"])
        rec["best"] = best
        tent = [k for k in ("laplace", "asym_laplace", "subbotin", "asym_subbotin") if k in f]
        rec["tent_beats_gamma"] = ("gamma_shift" in f) and min(f[k]["bic"] for k in tent) < f["gamma_shift"]["bic"]
        # gamma on the positive part (FM variant; separate sample)
        xp = x[x > 0]
        pg = stats.gamma.fit(xp, floc=0)
        rec["gamma_pos_shape"] = pg[0]
        rows.append(rec)
        z = (x - q[2]) / (q[3] - q[1])
        shapes.append((y, z))
        print(col, y, len(x), best, round(rec["subbotin_beta"], 3), flush=True)
    R = pd.DataFrame(rows)
    ks = [dict(year=b[0], ks=stats.ks_2samp(a[1], b[1]).statistic) for a, b in zip(shapes[:-1], shapes[1:])]
    return R, pd.DataFrame(ks)


def qcd(v):
    q1, q3 = np.quantile(v, [0.25, 0.75])
    return (q3 - q1) / (q3 + q1)


def outcome57(d, B=1000, seed=57):
    e = d[(d.OperatingIncomeLoss > 0) & (d.LaborAndRelatedExpense > 0) & (d.Assets > 0)].copy()
    e["a"] = e.OperatingIncomeLoss / e.Assets
    e["l"] = e.OperatingIncomeLoss / e.LaborAndRelatedExpense
    per = []
    for y, g in e.groupby("fy"):
        if len(g) < 100:
            continue
        per.append(dict(year=y, n=len(g), qcd_labour=qcd(g.l), qcd_assets=qcd(g.a), diff=qcd(g.l) - qcd(g.a)))
    P = pd.DataFrame(per)
    est = float(P["diff"].median())
    rng = np.random.default_rng(seed)
    groups = {y: g[["a", "l"]].to_numpy() for y, g in e.groupby("fy") if y in set(P.year)}
    bs = []
    for _ in range(B):
        diffs = []
        for y, a in groups.items():
            s = a[rng.integers(0, len(a), len(a))]
            diffs.append(qcd(s[:, 1]) - qcd(s[:, 0]))
        bs.append(np.median(diffs))
    lo, hi = np.quantile(bs, [0.05, 0.95])
    return P, dict(outcome="57: median over years of QCD(op. income / labour cost) - QCD(op. income / assets)", est=est,
                   ci90_lo=lo, ci90_hi=hi, n_firm_years=len(e), label=decide(est, lo, hi, 0, 0.05, "<"))


SIC_SECTOR = [((100, 999), "agr"), ((1000, 1499), "min"), ((1500, 1799), "con"), ((2000, 3999), "man"),
              ((4000, 4799), "trn"), ((4800, 4899), "inf"), ((4900, 4999), "util"), ((5000, 5199), "whl"),
              ((5200, 5999), "ret"), ((7000, 7099), "acc"), ((7200, 7299), "oth"), ((7300, 7399), "adm"),
              ((7800, 7999), "art"), ((8000, 8099), "hlt"), ((8200, 8299), "edu"), ((8700, 8799), "mps")]


def labour_terciles(d):
    from ..v9.stage7 import build
    D, _ = build()
    s = D[D.year == 2017].set_index("code")
    ls = (s.comp / s.va).to_dict()

    def sector(sic):
        for (a, b), c in SIC_SECTOR:
            if a <= sic <= b:
                return c
        return None
    d = d.copy()
    d["sector"] = d.sic.map(sector)
    d["lshare"] = d.sector.map(ls)
    e = d.dropna(subset=["lshare", "r_op"])
    e = e[e.r_op.abs() <= 1]
    cuts = e.drop_duplicates("sector").lshare.quantile([1 / 3, 2 / 3]).to_numpy()
    e["tercile"] = np.where(e.lshare <= cuts[0], "low", np.where(e.lshare <= cuts[1], "mid", "high"))
    rows = []
    for (y, t), g in e[e.r_op > 0].groupby(["fy", "tercile"]):
        rows.append(dict(year=y, tercile=t, n=len(g), qcd_r_op=qcd(g.r_op), median_r_op=g.r_op.median()))
    T = pd.DataFrame(rows)
    return T, e.groupby("tercile").sector.unique().apply(list).to_dict()


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    d = sample()
    out = []
    for col in ("r_op", "r_ni", "r_pp", "r_cost"):
        R, K = by_year(d, col)
        R.to_csv(OUT / f"s5_fits_{col}.csv", index=False)
        K.to_csv(OUT / f"s5_ks_{col}.csv", index=False)
        if col == "r_op":
            o56 = share_row("56: share of years where Laplace/Subbotin beats shifted gamma by BIC (r_op)", R.tent_beats_gamma,
                            0.5, 0.15, ">")
            out.append(o56)
        print(R.drop(columns=[c for c in R.columns if c.startswith("bic_")]).round(3).to_string(), flush=True)
    R1, _ = by_year(d.assign(r_op=d.r_op.where(d.r_op.between(d.r_op.quantile(0.01), d.r_op.quantile(0.99)))), "r_op", 10)
    R1.to_csv(OUT / "s5_fits_r_op_trim1pct.csv", index=False)
    P, o57 = outcome57(d)
    P.to_csv(OUT / "s5_qcd_labour_assets.csv", index=False)
    out.append(o57)
    T, terc = labour_terciles(d)
    T.to_csv(OUT / "s5_labour_terciles.csv", index=False)
    pd.Series({k: ",".join(v) for k, v in terc.items()}).to_csv(OUT / "s5_tercile_sectors.csv")
    pd.DataFrame(out).to_csv(OUT / "s5_outcomes.csv", index=False)
    print(pd.DataFrame(out).round(4).to_string(), "\n", P.round(3).to_string(), "\n",
          T.groupby("tercile").qcd_r_op.median().round(3).to_string())
