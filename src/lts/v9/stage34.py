"""Iteration 9, stages 3-4: world values and hours vs costs on the world market (pre_registration_v9.md).

World MRIO = WIOD 2016 (43 countries + RoW, 56 industries, 2000-2014) with SEA.
Vertically integrated contents per dollar of output of every country-industry:
  hours (a), hours weighted by physical productivity (b1 PLD 2005 benchmark, b2 PLD 2023 sectors, b3 PWT price level),
  labour compensation LAB (c), capital stock K, energy (purchases from B, C19, D35 as a basis), flat vector.
Stage 3: world industries (output-weighted averages); national vs world-average; 5-year changes.
Stage 4: cells (country, traded industry); levels at equal dispersion and 5-year changes; country transfers T_c.
Outputs: results/v9/s3_*.csv, s4_*.csv
"""
from __future__ import annotations

import sys
from functools import lru_cache

import numpy as np
import pandas as pd
import scipy.linalg as sla

from ..revisit.wiod import RAW, sea, wiot
from ..v2.placebo_cost import metrics_mat
from ..v8.rule import decide
from .stage1 import OUT

R9 = OUT.parents[1] / "data" / "raw" / "v9"
YEARS = list(range(2000, 2015))
ENERGY = {"B", "C19", "D35"}
GOVL = {"O84", "P85", "Q", "T", "U", "L68"}
PLD35 = {"A01": "AtB", "A02": "AtB", "A03": "AtB", "B": "C", "C10-C12": "15t16", "C13-C15": "17t18", "C16": "20",
         "C17": "21t22", "C18": "21t22", "C19": "23", "C20": "24", "C21": "24", "C22": "25", "C23": "26", "C24": "27t28",
         "C25": "27t28", "C26": "30t33", "C27": "30t33", "C28": "29", "C29": "34t35", "C30": "34t35", "C31_C32": "36t37",
         "C33": "36t37", "D35": "E", "E36": "E", "E37-E39": "E", "F": "F", "G45": "50", "G46": "51", "G47": "52",
         "H49": "60", "H50": "61", "H51": "62", "H52": "63", "H53": "64", "I": "H", "J58": "71t74", "J59_J60": "71t74",
         "J61": "64", "J62_J63": "71t74", "K64": "J", "K65": "J", "K66": "J", "L68": "70", "M69_M70": "71t74",
         "M71": "71t74", "M72": "71t74", "M73": "71t74", "M74_M75": "71t74", "N": "71t74", "O84": "L", "P85": "M",
         "Q": "N", "R_S": "O", "T": "P"}
PLD12 = {"A": "agr", "B": "min", "C": "man", "D": "pu", "E": "pu", "F": "con", "G": "trd", "H": "tra", "I": "trd",
         "J": "bus", "K": "fin", "L": "dwe", "M": "bus", "N": "bus", "O": "pub", "P": "pub", "Q": "pub", "R": "oth",
         "S": "oth", "T": "oth", "U": "oth"}


def traded(code):
    return code[0] in "AC" or code == "B"


# ---------------------------------------------------------------- inputs
@lru_cache(None)
def xr():
    d = pd.read_excel(RAW / "wiod" / "Exchange_Rates.xlsx", sheet_name="EXR", header=None).iloc[4:]
    d = d[d[1].notna()]
    out = {}
    for _, r in d.iterrows():
        for j, y in enumerate(YEARS):
            out[(str(r[1]).strip(), y)] = float(r[2 + j])           # US$ per unit of local currency
    return out


@lru_cache(None)
def sea_wide():
    s = sea()
    s["year"] = s.year.astype(int)
    w = s.pivot_table(index=["country", "code", "year"], columns="variable", values="v").reset_index()
    w["xr"] = [xr().get((c, y), np.nan) for c, y in zip(w.country, w.year)]
    return w


@lru_cache(None)
def pld2005():
    d = pd.read_excel(R9 / "ggdc_benchmark_2005.xlsx", sheet_name="GO_35Industry", header=None)
    codes = [str(c).strip() for c in d.iloc[2, 2:]]
    out = {}
    for i in range(3, d.shape[0]):
        iso = d.iloc[i, 1]
        if not isinstance(iso, str) or len(iso) != 3:
            continue
        for j, c in enumerate(codes):
            v = pd.to_numeric(d.iloc[i, 2 + j], errors="coerce")
            if np.isfinite(v):
                out[(iso, c)] = float(v)
    return out


@lru_cache(None)
def pld2023():
    d = pd.read_excel(R9 / "pld2023_dataset.xlsx", sheet_name="Data")
    d["pl"] = d.PPP_y / d.xr
    return {(r.countrycode, int(r.year), r.sector): r.pl for r in d.itertuples() if np.isfinite(r.pl)}


@lru_cache(None)
def pwt_pl():
    p = pd.read_excel(R9 / "pwt110.xlsx", sheet_name="Data")
    col = "pl_gdpo" if "pl_gdpo" in p.columns else "pl_gdpe"
    return {(r.countrycode, int(r.year)): getattr(r, col) for r in p.itertuples() if np.isfinite(getattr(r, col))}


# ---------------------------------------------------------------- one year
def year_system(y):
    w = wiot(y)
    countries, codes = w["countries"], w["codes"]
    n = len(countries) * len(codes)
    x = w["x"]
    A = np.divide(w["Z"], x, out=np.zeros_like(w["Z"]), where=x > 0)
    cells = pd.DataFrame({"country": np.repeat(countries, len(codes)), "code": np.tile(codes, len(countries)), "x": x})
    s = sea_wide()
    s = s[s.year == y]
    cells = cells.merge(s, on=["country", "code"], how="left")
    usd = lambda v: v * cells.xr                                     # national currency -> US$ mn
    cells["H"] = cells.H_EMPE * np.where(cells.EMPE > 0, cells.EMP / cells.EMPE, 1.0)
    cells["LAB$"] = usd(cells.LAB)
    cells["K$"] = usd(cells.K)
    cells["GO$"] = usd(cells.GO)
    # RoW: per-dollar averages of the 43 countries, same industry
    for v in ("H", "LAB$", "K$"):
        rate = cells[cells.country != "ROW"].groupby("code").apply(
            lambda g: np.nansum(g[v]) / np.nansum(g["GO$"].where(g[v].notna())), include_groups=False)
        cells[v + "_x"] = np.where(cells.country == "ROW", cells.code.map(rate),
                                   np.divide(cells[v], cells["GO$"], out=np.full(n, np.nan), where=cells["GO$"] > 0))
    for v in ("H", "LAB$", "K$"):
        cells[v + "_x"] = cells[v + "_x"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # price levels (US = 1) for physical productivity
    cells["pl1"] = [pl_b1(c, k, y, s) for c, k in zip(cells.country, cells.code)]
    p23 = pld2023()
    bm = 2005 if y <= 2008 else 2011
    cells["pl2"] = [pl_b2(c, k, y, bm, s, p23) for c, k in zip(cells.country, cells.code)]
    pp = pwt_pl()
    cells["pl3"] = [pp.get((c, y), np.nan) / pp.get(("USA", y), np.nan) for c in cells.country]
    for v in ("pl1", "pl2", "pl3"):
        cells["w_" + v] = adj_weights(cells, v)
    return A, cells


def pl_b1(c, code, y, s):
    base = pld2005().get((c, PLD35.get(code, "")), np.nan)
    if not np.isfinite(base):
        return np.nan
    return base * rel_change(c, code, y, 2005, s)


def pl_b2(c, code, y, bm, s, p23):
    base = p23.get((c, bm, PLD12.get(code[0], "")), np.nan)
    if not np.isfinite(base):
        return np.nan
    return base * rel_change(c, code, y, bm, s)


_PI = None


def rel_change(c, code, y, base, s=None):
    """(GO_PI * xr)_t / (GO_PI * xr)_base, relative to the US same industry."""
    global _PI
    if _PI is None:
        w = sea_wide()
        _PI = {(r.country, r.code, r.year): r.GO_PI * r.xr for r in w.itertuples() if np.isfinite(r.GO_PI * r.xr)}
    try:
        return (_PI[(c, code, y)] / _PI[(c, code, base)]) / (_PI[("USA", code, y)] / _PI[("USA", code, base)])
    except (KeyError, ZeroDivisionError):
        return np.nan


def adj_weights(cells, plcol):
    """h* / h = pi / pi_bar; pi = (GO$/PL)/H; pi_bar per industry over cells with PL; weight 1 elsewhere."""
    q = cells["GO$"] / cells[plcol]
    ok = np.isfinite(q) & (cells.H > 0) & (q > 0)
    pib = (q.where(ok).groupby(cells.code).transform("sum") / cells.H.where(ok).groupby(cells.code).transform("sum"))
    wgt = (q / cells.H) / pib
    return np.where(ok & np.isfinite(wgt), wgt, 1.0)


def contents(A, cells):
    """Vertically integrated contents per dollar (rows) via one LU of (I - A)^T."""
    n = len(cells)
    I = np.eye(n)
    lu = sla.lu_factor((I - A).T)
    h = cells.H_x.to_numpy()
    rows = {"hours": h, "lab": cells["LAB$_x"].to_numpy(), "cap": cells["K$_x"].to_numpy(),
            "flat": (cells.x > 0).astype(float).to_numpy()}
    for v in ("pl1", "pl2", "pl3"):
        rows["hours_adj_" + v] = h * cells["w_" + v].to_numpy()
    pidx = (cells.GO_PI / 100 * cells.xr / cells.country.map(lambda c: xr().get((c, 2010), np.nan))).to_numpy()
    rows["flat_phys"] = np.where(np.isfinite(pidx) & (pidx > 0), 1 / pidx, 0.0)
    out = {k: sla.lu_solve(lu, v) for k, v in rows.items()}
    # energy basis: rows of energy industries made primary
    en = cells.code.isin(ENERGY).to_numpy()
    An = A.copy()
    An[en, :] = 0.0
    aE = A[en, :].sum(0)
    out["energy"] = sla.solve((I - An).T, aE)
    out["_pidx"] = pidx                                              # cell price index in US$, 2010 = 1
    return out


# ---------------------------------------------------------------- metrics helpers
def qmap(v, target):
    """Rank-preserving map of v onto the distribution of target (same length)."""
    r = np.argsort(np.argsort(v, kind="stable"), kind="stable")
    return np.sort(target)[r]


def mawd(z, x):
    m = metrics_mat(np.atleast_2d(z), x)
    return float(m["mawd"][0]), float(m["d"][0])


# ---------------------------------------------------------------- stage 3
def world_industries(cont, cells):
    keep = ~cells.code.isin(GOVL) & (cells.x > 0)
    agg = {}
    for k, v in cont.items():
        if k.startswith("_"):
            continue
        agg[k] = (pd.Series(v * cells.x)[keep.values].groupby(cells.code[keep.values]).sum()
                  / cells.x[keep].groupby(cells.code[keep]).sum())
    X = cells.x[keep].groupby(cells.code[keep]).sum()
    return pd.DataFrame(agg), X


def stage3_levels(y, W, X, n_perm=1000, seed=3):
    h = W.hours.to_numpy()
    rows = []
    for k in W.columns:
        v = W[k].to_numpy()
        raw = mawd(v, X.to_numpy())
        eq = mawd(qmap(v, h), X.to_numpy()) if k != "flat" else raw
        rows.append(dict(year=y, cand=k, mawd=raw[0], d=raw[1], mawd_eq=eq[0], d_eq=eq[1]))
    rng = np.random.default_rng(seed)
    Pm = np.array([mawd(h[rng.permutation(len(h))], X.to_numpy())[0] for _ in range(n_perm)])
    rows.append(dict(year=y, cand="perm placebos (median)", mawd=float(np.median(Pm)),
                     share_better_than_hours=float((Pm < mawd(h, X.to_numpy())[0]).mean())))
    return rows


def stage3_national(y, cont, cells, W):
    rows = []
    keep = ~cells.code.isin(GOVL) & (cells.x > 0) & (cells.country != "ROW")
    for c, g in cells[keep].groupby("country"):
        idx = g.index.to_numpy()
        lam = cont["hours"][idx]
        wld = g.code.map(W.hours).to_numpy()
        if (lam <= 0).any() or len(idx) < 20:
            continue
        x = g.x.to_numpy()
        m_nat = mawd(lam, x)[0]
        m_wld_eq = mawd(qmap(wld, lam), x)[0]
        rows.append(dict(year=y, country=c, mawd_national=m_nat, mawd_world_eq=m_wld_eq, mawd_world_raw=mawd(wld, x)[0],
                         world_better=m_wld_eq < m_nat))
    return rows


# ---------------------------------------------------------------- stage 4
def stage4_cells(y, cont, cells, B=2000, seed=4):
    keep = cells.code.map(traded) & (cells.country != "ROW") & (cells.x > 0)
    for k in ("hours", "hours_adj_pl1"):
        keep &= np.isfinite(cont[k]) & (cont[k] > 0)
    c = cells[keep].copy()
    idx = c.index.to_numpy()
    cand = {k: cont[k][idx] for k in ("hours", "hours_adj_pl1", "hours_adj_pl2", "hours_adj_pl3", "lab", "cap",
                                      "energy", "flat")}
    tgt = cand["hours_adj_pl1"]
    x = c.x.to_numpy()
    eq = {k: (qmap(v, tgt) if k not in ("flat",) else v) for k, v in cand.items()}
    rows = []
    for k in cand:
        r = mawd(cand[k], x)
        e = mawd(eq[k], x)
        rows.append(dict(year=y, cand=k, mawd=r[0], d=r[1], mawd_eq=e[0], d_eq=e[1], n_cells=len(c)))
    # bootstrap by country for the outcome differences (equal dispersion)
    ctry = c.country.to_numpy()
    uc = np.unique(ctry)
    members = {u: np.where(ctry == u)[0] for u in uc}
    rng = np.random.default_rng(seed + y)
    diffs = {"30 b-c": [], "31 b-min(cap,energy)": [], "32 a-b": []}
    for _ in range(B):
        ii = np.concatenate([members[u] for u in rng.choice(uc, len(uc), replace=True)])
        m = {k: mawd(eq[k][ii], x[ii])[0] for k in ("hours", "hours_adj_pl1", "lab", "cap", "energy")}
        diffs["30 b-c"].append(m["hours_adj_pl1"] - m["lab"])
        diffs["31 b-min(cap,energy)"].append(m["hours_adj_pl1"] - min(m["cap"], m["energy"]))
        diffs["32 a-b"].append(m["hours"] - m["hours_adj_pl1"])
    pt = {r["cand"]: r["mawd_eq"] for r in rows}
    est = {"30 b-c": pt["hours_adj_pl1"] - pt["lab"], "31 b-min(cap,energy)": pt["hours_adj_pl1"] - min(pt["cap"], pt["energy"]),
           "32 a-b": pt["hours"] - pt["hours_adj_pl1"]}
    outc = []
    for k, d in diffs.items():
        lo, hi = np.quantile(d, [0.05, 0.95])
        direction = ">" if k.startswith("32") else "<"
        outc.append(dict(year=y, outcome=k, est=est[k], ci90_lo=lo, ci90_hi=hi, label=decide(est[k], lo, hi, 0, 0.01, direction)))
    # transfers T_c (4.4)
    lam = cand["hours_adj_pl1"]
    m_ = x.sum() / (lam * x).sum()
    c["T"] = x * (m_ * lam - 1)
    T = c.groupby("country").agg(T=("T", "sum"), x=("x", "sum")).reset_index().assign(year=y)
    share1 = float(np.mean(c.w_pl1.to_numpy() == 1.0))
    return rows, outc, T, dict(year=y, share_cells_weight1=share1, n_cells=len(c))


# ---------------------------------------------------------------- changes (5 years)
def changes(store, stage):
    rows = []
    for y0, y1 in ((2000, 2005), (2005, 2010), (2009, 2014)):
        a, b = store[y0], store[y1]
        if stage == 3:
            keys = a["W"].index.intersection(b["W"].index)
            Pa, Pb = a["P"].reindex(keys), b["P"].reindex(keys)
            wts = b["X"].reindex(keys).to_numpy()
            dP = np.log(Pb / Pa).to_numpy()
            for k in a["W"].columns:
                dz = np.log(b["W"][k].reindex(keys) / a["W"][k].reindex(keys)).to_numpy()
                rows.append(dict(window=f"{y0}-{y1}", cand=k, r2=t1(dP, dz, wts)))
        else:
            ca, cb = a["cells"], b["cells"]
            m = ca.merge(cb, on=["country", "code"], suffixes=("_a", "_b"))
            dP = np.log(m.p_b / m.p_a).to_numpy()
            wts = m.x_b.to_numpy()
            for k in [c[:-2] for c in m.columns if c.endswith("_a") and c.startswith("z_")]:
                dz = np.log(m[k + "_b"] / m[k + "_a"]).to_numpy()
                rows.append(dict(window=f"{y0}-{y1}", cand=k[2:], r2=t1(dP, dz, wts)))
    return pd.DataFrame(rows)


def t1(dP, dz, w):
    """T1 at beta = 1: content per real unit = z * P, so dP - d(content per real unit) = -dz."""
    ok = np.isfinite(dP) & np.isfinite(dz) & (w > 0)
    dP, dz, w = dP[ok], dz[ok], w[ok] / w[ok].sum()
    dPr = dP - w @ dP
    dzr = dz - w @ dz
    return float(1 - (w @ dzr ** 2) / (w @ dPr ** 2))


def run(years=YEARS):
    L3, N3, L4, O4, T4, info = [], [], [], [], [], []
    st3, st4 = {}, {}
    for y in years:
        A, cells = year_system(y)
        cont = contents(A, cells)
        W, X = world_industries(cont, cells)
        L3 += stage3_levels(y, W, X)
        N3 += stage3_national(y, cont, cells, W)
        # world industry price index (2010 = 1, US$): output-weighted harmonic mean of cell indices
        pidx = cont["_pidx"]
        keep = ~cells.code.isin(GOVL) & (cells.x > 0) & np.isfinite(pidx) & (pidx > 0) & (cells.country != "ROW")
        P = (cells.x[keep].groupby(cells.code[keep]).sum()
             / (cells.x[keep] / pidx[keep.values]).groupby(cells.code[keep]).sum())
        st3[y] = dict(W=W, X=X, P=P)
        r, o, T, inf = stage4_cells(y, cont, cells)
        L4 += r
        O4 += o
        T4.append(T)
        info.append(inf)
        tr = cells.code.map(traded) & (cells.country != "ROW") & (cells.x > 0) & np.isfinite(pidx) & (pidx > 0)
        cdf = cells[tr][["country", "code", "x"]].copy()
        cdf["p"] = pidx[tr.values]
        for k in ("hours", "hours_adj_pl1", "lab", "cap", "energy", "flat", "flat_phys"):
            cdf["z_" + k] = cont[k][tr.values]
        st4[y] = dict(cells=cdf)
        print("s34", y, flush=True)
        pd.DataFrame(L3).to_csv(OUT / "s3_world_levels.csv", index=False)
        pd.DataFrame(N3).to_csv(OUT / "s3_national_vs_world.csv", index=False)
        pd.DataFrame(L4).to_csv(OUT / "s4_levels.csv", index=False)
        pd.DataFrame(O4).to_csv(OUT / "s4_outcomes_by_year.csv", index=False)
        pd.concat(T4).to_csv(OUT / "s4_transfers.csv", index=False)
        pd.DataFrame(info).to_csv(OUT / "s4_info.csv", index=False)
    if all(y in st3 for y in (2000, 2005, 2009, 2010, 2014)):
        changes(st3, 3).to_csv(OUT / "s3_changes.csv", index=False)
        changes(st4, 4).to_csv(OUT / "s4_changes.csv", index=False)


if __name__ == "__main__":
    yrs = [int(a) for a in sys.argv[1:]] or YEARS
    run(yrs)
