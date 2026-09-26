"""Iteration 4, B0b: labour vs aggregate costs with equal dispersion (pre-registration v4, journal 8).

Candidates (direct coefficients per unit of output, FIGARO labels): hours, wage bill (labour income),
CFC, net capital stock (EU KLEMS K_GFCF split within KLEMS industries by FIGARO CFC, A-V4-B0B-K),
GOS corrected for mixed income, GOS uncorrected, non-labour primary costs (B2A3G + D29X39).
Equalisation: (a) rank-preserving quantile map onto the hours coefficients; (b) power x^alpha matching
the log-SD of hours; (c) all candidates (hours included) powered to the median log-SD of the candidates.
Metrics MAWD and d on the v3 evaluation set and without B, K64-66, L, J61, J62_63.
1000 permutations of each equalised vector (v3 stage-P permutations).  Industry bootstrap (1000 draws
per country, same draw for all years) of the mean-over-years metric difference candidate - hours.
Outputs results/v4/b0b_levels.csv (per country-year-candidate), b0b_boot.csv (per country).
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from .. import oecd
from ..codes import divisions
from ..economy import MERGES, _group_matrix, build, national_blocks
from ..figaro import INDUSTRIES
from ..v3.placebo_disp import COUNTRIES, YEARS, Draws, Setup
from .b_data import divs as kdivs
from .splits import OUT

PREREG_EXCL = {"B", "K64", "K65", "K66", "L", "J61", "J62_63"}
CANDS = ["hours", "wagebill", "cfc", "capital", "gos_mi", "gos", "nonlabour"]
N_BOOT = 1000
ISO2K = {"AUT": "AT", "CZE": "CZ", "DEU": "DE", "ESP": "ES", "FRA": "FR", "ITA": "IT", "NLD": "NL", "POL": "PL",
         "GBR": "UK", "USA": "US", "JPN": "JP"}
_PANEL = None


def panel():
    global _PANEL
    if _PANEL is None:
        _PANEL = pd.read_csv(OUT / "b_panel.csv.gz")
    return _PANEL


def label_rows(c3, y, labels):
    """B2A3G and D29X39 aggregated to the economy's labels (same merges as economy.build)."""
    c2 = oecd.ISO3_TO_2[c3]
    b = national_blocks(c2, y)
    names, G = _group_matrix(INDUSTRIES, MERGES.get(c2, []))
    keep = (G @ b["x"]) > 0
    out = {}
    for it in ("B2A3G", "D29X39"):
        v = (G @ b["V"][b["varows"].index(it)])[keep]
        out[it] = pd.Series(v, index=[n for n, k in zip(names, keep) if k]).reindex(labels).to_numpy()
    return out


def capital(e, c3, y):
    """K per label: KLEMS K_GFCF of connected groups of (labels, KLEMS industries), split by CFC."""
    geo = ISO2K.get(c3)
    if geo is None:
        return None
    p = panel()
    g = p[(p.geo == geo) & (p.year == min(y, 2021))]
    if g.empty:
        return None
    kind = [(r.ind, kdivs(r.ind), r.K) for r in g.itertuples()]
    ldiv = [set().union(*[divisions(m) or set() for m in lab.split("+")]) for lab in e.labels]
    # union-find over labels (0..n-1) and KLEMS industries (n..)
    n = e.n
    par = list(range(n + len(kind)))

    def f(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a
    for i, ld in enumerate(ldiv):
        for k, (_, kd, _) in enumerate(kind):
            if ld & kd:
                par[f(i)] = f(n + k)
    K = np.full(n, np.nan)
    groups = {}
    for i in range(n):
        groups.setdefault(f(i), []).append(i)
    for root, labs in groups.items():
        ks = [k for k in range(len(kind)) if f(n + k) == root]
        if not ks:
            continue
        cover = set().union(*[kind[k][1] for k in ks])
        if not set().union(*[ldiv[i] for i in labs]) <= cover:
            continue
        Ktot = np.nansum([kind[k][2] for k in ks])
        c = e.cfc[labs]
        K[labs] = Ktot * c / c.sum() if c.sum() > 0 else np.nan
    return K


def candidates(e, c3, y):
    x = e.x
    rows = label_rows(c3, y, e.labels)
    se_li = e.labour_income - e.wages
    raw = {"hours": e.hours, "wagebill": e.labour_income, "cfc": e.cfc,
           "gos_mi": rows["B2A3G"] - se_li, "gos": rows["B2A3G"], "nonlabour": rows["B2A3G"] + rows["D29X39"]}
    K = capital(e, c3, y)
    if K is not None and np.isfinite(K).mean() > 0.9:
        raw["capital"] = np.where(np.isfinite(K), K, np.nanmedian(K / x) * x)
    coef, nfix = {}, {}
    for k, v in raw.items():
        c = np.divide(v, x, out=np.zeros(e.n), where=x > 0)
        pos = c > 0
        nfix[k] = int((~pos & (x > 0)).sum())
        if pos.any():
            c = np.where(pos, c, c[pos].min())
        coef[k] = c
    return coef, nfix


def equalise(coef, ev):
    """dict method -> dict cand -> equalised coefficient vector."""
    h = coef["hours"]
    lsd = lambda v: np.log(v[ev]).std()
    out = {"a_quantile": {}, "b_power": {}, "c_common": {}}
    hs = np.sort(h[ev])
    target = np.median([lsd(v) for v in coef.values()])
    for k, v in coef.items():
        qa = h.copy()
        ranks = np.argsort(np.argsort(v[ev], kind="stable"), kind="stable")
        qa[ev] = hs[ranks]
        out["a_quantile"][k] = qa
        out["b_power"][k] = v ** (lsd(h) / lsd(v))            # T, U transformed like the rest (journal 9)
        out["c_common"][k] = v ** (target / lsd(v))
    return out


def metrics3(Z, x):
    """Z: (..., J) ratios on evaluated industries; x: (..., J). Returns mawd, d arrays (...)."""
    Z = Z * x.sum(-1, keepdims=True) / (Z * x).sum(-1, keepdims=True)
    w = x / x.sum(-1, keepdims=True)
    q = 1 / Z
    mawd = (np.abs(q - 1) * w).sum(-1)
    cos = q.sum(-1) / (np.linalg.norm(q, axis=-1) * np.sqrt(Z.shape[-1]))
    d = np.sqrt(np.clip(2 * (1 - cos), 0, None))
    return {"mawd": mawd, "d": d}


def run_country(c3):
    rows, boot_store = [], {}
    D = None
    rng = np.random.default_rng(99 + sum(map(ord, c3)))
    idx_boot = None
    for y in YEARS:
        e, _ = build(c3, y)
        S = Setup(e)
        if D is None:
            D = Draws(c3, e.n, S.base_mask)
        ev = S.base_mask
        masks = {"all": ev, "no_rent": S.mask(PREREG_EXCL)}
        coef, nfix = candidates(e, c3, y)
        eq = equalise(coef, ev)
        for mname, m in masks.items():
            J = int(m.sum())
            key = (mname, J)
            if key not in boot_store:
                boot_store[key] = rng.integers(0, J, (N_BOOT, J))
            I = boot_store[key]
            for meth, vecs in eq.items():
                for cand, l in vecs.items():
                    z = S.z(l[None])[0]
                    zm, xm = z[m], S.x[m]
                    met = metrics3(zm[None], xm[None])
                    mb = metrics3(zm[I], xm[I])
                    Zp = S.z(l[D.perm])[:, m]
                    mp = metrics3(Zp, np.broadcast_to(xm, Zp.shape))
                    rows.append(dict(country=c3, year=y, mask=mname, method=meth, cand=cand,
                                     mawd=float(met["mawd"][0]), d=float(met["d"][0]),
                                     perm_better_mawd=float((mp["mawd"] < met["mawd"][0]).mean()),
                                     perm_better_d=float((mp["d"] < met["d"][0]).mean()),
                                     nfix=nfix.get(cand, 0),
                                     boot_mawd=mb["mawd"].astype(np.float32), boot_d=mb["d"].astype(np.float32)))
        print("b0b", c3, y, flush=True)
    return rows


def summarise(rows):
    lev = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("boot")} for r in rows])
    df = pd.DataFrame(rows)
    out = []
    for (c, mask, meth), g in df.groupby(["country", "mask", "method"]):
        hrs = g[g.cand == "hours"].set_index("year")
        for cand in CANDS:
            gc = g[g.cand == cand].set_index("year")
            if gc.empty:
                continue
            yrs = sorted(set(gc.index) & set(hrs.index))
            for met in ("mawd", "d"):
                bc = np.vstack(gc.loc[yrs, f"boot_{met}"].to_list()).mean(0)
                bh = np.vstack(hrs.loc[yrs, f"boot_{met}"].to_list()).mean(0)
                dlt = bc - bh
                lo, hi = np.quantile(dlt, [0.025, 0.975])
                out.append(dict(country=c, mask=mask, method=meth, cand=cand, metric=met, n_years=len(yrs),
                                mean=float(gc.loc[yrs, met].mean()),
                                ci_lo=float(np.quantile(bc, 0.025)), ci_hi=float(np.quantile(bc, 0.975)),
                                delta=float(gc.loc[yrs, met].mean() - hrs.loc[yrs, met].mean()),
                                delta_lo=float(lo), delta_hi=float(hi),
                                hours_sig_better=bool(lo > 0), cand_sig_better=bool(hi < 0),
                                perm_better=float(gc.loc[yrs, f"perm_better_{met}"].mean())))
    return lev, pd.DataFrame(out)


def run(countries=COUNTRIES):
    L, Bt = [], []
    for c in countries:
        lev, bt = summarise(run_country(c))
        L.append(lev)
        Bt.append(bt)
        pd.concat(L).to_csv(OUT / "b0b_levels.csv", index=False)
        pd.concat(Bt).to_csv(OUT / "b0b_boot.csv", index=False)


if __name__ == "__main__":
    run(sys.argv[1].split(",") if len(sys.argv) > 1 else COUNTRIES)
