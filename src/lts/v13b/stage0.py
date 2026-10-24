"""Iteration 13b, stage 0 (pre_registration_v13b.md): supervisors by the 47 codes of 13a, the qualification
equivalent w_eq from March CPS cells, the managers' premium P and pi. No outcomes.

0.2-0.3  q(g, t) = w_eq / w_rf and rho(g, t) = weekly hours of supervisors / rank-and-file, by industry group g and
         earnings year t, from the CPS extracts of lts.v13b.cps (cell ladder 0.3); definitions: main, broad, ces
         (composition by CES); Hilferding ratio qH = mean psi of supervisors / mean psi of rank-and-file.
0.4      W_eq,j = W_rf,j ((1 - sN) / sN) rho_g q_g;  P_j = W_sup,j - W_eq,j;  pi = sum P / sum W_eq (productive codes).
         Supervisor hours L_s,j = H_rf,j ((1 - sN) / sN) rho_g (main) or (1 - sH) H_emp,j (variant 'hours = CES').

Usage: PYTHONPATH=src python3 -P -m lts.v13b.stage0 [cps|premium|all]
Outputs: results/v13b/s0_qrho.csv, s0_cells.csv, s0_counts.csv, s0_premium_codes.csv, s0_pi.csv, s0_checks.csv
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from ..v6.series import ROOT
from ..v12b import us_io as u
from ..v13a import stage0 as s13
from . import cps

OUT = ROOT / "results" / "v13b"
V13A = ROOT / "results" / "v13a"
CODES = u.CODES
PROD = [c for c in CODES if c not in u.UNPRODUCTIVE and not c.startswith("G")]
PROD_GROUPS = ["G1", "G2", "G3", "G4", "G5", "G6", "G9", "G10"]
GOODS = ["G2", "G3", "G4", "G5"]
MIN_CELL = 25
SEAMS = [1971, 1982, 1991, 2002, 2010, 2019]          # first earnings year of a new occupation classification
MISSING = [1962, 1964, 1969]                          # earnings years without a March file at NBER


# ================================================================ 0.2-0.3 CPS: q and rho
def load(survey_year, variant=""):
    f = cps.CACHE / f"cps_{survey_year}{variant}.parquet"
    d = pd.read_parquet(f)
    d["H"] = d.weeks * d.hrs_year_week
    d["w"] = d.wage / d.H
    ok = d.H > 0
    lo, hi = wquantile(d.w[ok], d.wgt[ok], [0.01, 0.99])
    d["use_w"] = ok & (d.w >= lo) & (d.w <= hi)
    d["e5"] = pd.cut(d.S, [-1, 11.99, 12.01, 15.99, 16.01, 99], labels=False)
    d["e3"] = pd.cut(d.S, [-1, 12.01, 15.99, 99], labels=False)
    d["a4"] = pd.cut(d.age, [17, 29, 39, 49, 64], labels=False)
    d["a2"] = pd.cut(d.age, [17, 39, 64], labels=False)
    d["psi"] = 1 + cps.KAPPA * np.maximum(0.0, d.S - 12)
    return d


def wquantile(x, w, qs):
    x, w = np.asarray(x, float), np.asarray(w, float)
    o = np.argsort(x)
    cw = np.cumsum(w[o]) / w.sum()
    return [float(x[o][np.searchsorted(cw, q)]) for q in qs]


def sup_flag(d, definition):
    if definition == "main":
        return d.sup.to_numpy()
    if definition == "broad":
        return d.sup_broad.to_numpy()
    if definition == "ces":
        return np.where(d.grp.isin(GOODS), d.nonprod, d.sup)
    raise ValueError(definition)


def _cellmeans(rf, keys):
    """Rank-and-file mean hourly wage (sum wage / sum hours, weighted) and unweighted count by cell."""
    g = rf.assign(ww=rf.wgt * rf.wage, wh=rf.wgt * rf.H).groupby(keys, observed=True)
    return pd.DataFrame(dict(w=g.ww.sum() / g.wh.sum(), n=g.size()))


LADDER = [["grp", "e5", "a4", "female"], ["grp", "e5", "a2", "female"], ["grp", "e5", "a2"], ["grp", "e3", "a2"],
          ["e3", "a2"]]


def qrho_year(survey_year, definition, pooled=None, variant=""):
    """q, rho, qH by group for one survey year. pooled: rank-and-file of the neighbouring years (ladder step 5)."""
    d = load(survey_year, variant)
    d["s"] = sup_flag(d, definition)
    d = d[d.grp.notna()]
    rf = d[~d.s & d.use_w]
    sp = d[d.s & d.use_w]
    tables = [_cellmeans(rf, k) for k in LADDER[:4]]
    tables.append(_cellmeans(rf[rf.grp.isin(PROD_GROUPS)], LADDER[4]))
    if pooled is not None:
        tables.append(_cellmeans(pooled[pooled.grp.isin(PROD_GROUPS)], LADDER[4]))
    cells = sp.assign(h=sp.wgt * sp.H).groupby(["grp", "e5", "a4", "female", "e3", "a2"], observed=True).h.sum() \
        .reset_index()
    w_c, step = np.full(len(cells), np.nan), np.full(len(cells), -1)
    for i, (keys, T) in enumerate(zip(LADDER + [LADDER[4]], tables)):
        idx = pd.MultiIndex.from_frame(cells[keys]) if len(keys) > 1 else cells[keys[0]]
        v = T.reindex(idx)
        take = np.isnan(w_c) & (v.n.to_numpy() >= MIN_CELL)
        w_c[take] = v.w.to_numpy()[take]
        step[take] = i
    last = tables[-1].reindex(pd.MultiIndex.from_frame(cells[LADDER[4]]))
    rest = np.isnan(w_c)
    w_c[rest] = last.w.to_numpy()[rest]
    step[rest] = 9                                        # still below the minimum after step 5 (flag)
    cells["w_rf_cell"], cells["step"] = w_c, step
    rows, crow = [], []
    for g in sorted(d.grp.unique()):
        cg = cells[cells.grp == g]
        rg, sg = rf[rf.grp == g], sp[sp.grp == g]
        if cg.h.sum() <= 0 or len(rg) == 0:
            continue
        w_eq = float((cg.h * cg.w_rf_cell).sum() / cg.h.sum())
        w_rf = float((rg.wgt * rg.wage).sum() / (rg.wgt * rg.H).sum())
        at = d[(d.grp == g) & d.hrs_week.notna()]
        hs = at[at.s]
        hr = at[~at.s]
        rho = float(np.average(hs.hrs_week, weights=hs.wgt) / np.average(hr.hrs_week, weights=hr.wgt)) \
            if len(hs) and len(hr) else np.nan
        qH = float(np.average(sg.psi, weights=sg.wgt * sg.H) / np.average(rg.psi, weights=rg.wgt * rg.H))
        rows.append(dict(eyear=survey_year - 1, grp=g, definition=definition, variant=variant or "main", q=w_eq / w_rf,
                         rho=rho, qH=qH, w_eq=w_eq, w_rf=w_rf, n_sup=len(sg), n_rf=len(rg),
                         N_sup_w=float(d[(d.grp == g) & d.s].wgt.sum()), N_rf_w=float(d[(d.grp == g) & ~d.s].wgt.sum()),
                         **{f"share_step{k}": float(cg.h[cg.step == k].sum() / cg.h.sum()) for k in (0, 1, 2, 3, 4, 5, 9)}))
    crow = cells.assign(eyear=survey_year - 1, definition=definition)
    return pd.DataFrame(rows), crow


def cps_all():
    years = cps.SURVEY_YEARS
    out, cells = [], []
    cache = {}

    def rf_of(y):
        if y not in cache:
            dd = load(y)
            cache[y] = dd
        return cache[y]

    for definition in ("main", "broad", "ces"):
        for i, y in enumerate(years):
            nb = [z for z in (y - 1, y + 1) if z in years]
            pooled = []
            for z in [y] + nb:
                dz = rf_of(z)
                dz = dz.assign(s=sup_flag(dz, definition))
                pooled.append(dz[~dz.s & dz.use_w & dz.grp.notna()])
            r, c = qrho_year(y, definition, pd.concat(pooled))
            out.append(r)
            if definition == "main":
                cells.append(c)
            print(definition, y, flush=True)
    r, _ = qrho_year(2014, "main", None, "_3x8")
    out.append(r)
    Q = pd.concat(out, ignore_index=True)
    Q.to_csv(OUT / "s0_qrho.csv", index=False)
    pd.concat(cells).to_csv(OUT / "s0_cells.csv", index=False)
    return Q


# ================================================================ series by group and year (interpolation, seams)
def series(Q, definition="main", col="q", interp=True, seams=False, back1961=False, years=range(1948, 2024),
           v2014="main"):
    """ln-linear interpolation over the missing earnings years; optional seam neutralisation (the change in the first
    year of a new occupation classification replaced by the mean change of the neighbouring years); optional
    constant 1961 values before 1961."""
    q = Q[(Q.definition == definition) & (Q.variant == "main")].pivot_table(index="eyear", columns="grp", values=col)
    if v2014 == "3x8":
        alt = Q[(Q.variant == "_3x8")].pivot_table(index="eyear", columns="grp", values=col)
        q.loc[2013] = alt.loc[2013]
    lq = np.log(q.sort_index())
    if seams:
        dl = lq.diff()
        for s in SEAMS:
            if s in dl.index and s - 1 in dl.index and s + 1 in dl.index:
                dl.loc[s] = (dl.loc[s - 1] + dl.loc[s + 1]) / 2
        lq = pd.concat([lq.iloc[[0]], lq.iloc[0] + dl.iloc[1:].cumsum()])
    full = lq.reindex(range(min(years[0], int(lq.index.min())), max(years[-1], int(lq.index.max())) + 1))
    if interp:
        full = full.interpolate(limit_area="inside")
    if back1961:
        full = full.bfill()
    return np.exp(full).reindex(years)


# ================================================================ 0.4 premium by code
def code_inputs(ces_variant="main"):
    sh = pd.read_csv(V13A / "s0_ces_shares.csv", dtype={"code": str})
    cap = ces_variant == "sW capped at 1"
    sh = sh[sh.variant == ("main" if cap else ces_variant)].copy()
    if cap:
        sh["sW"] = sh.sW.clip(upper=1.0)
    piv = {k: sh.pivot_table(index="year", columns="code", values=k).reindex(columns=PROD) for k in ("sN", "sW", "sH")}
    W = pd.read_csv(V13A / "s0_wages_47.csv", index_col=0).reindex(columns=PROD)
    H = pd.read_csv(V13A / "s0_hours_47.csv", index_col=0).reindex(columns=PROD)
    se = pd.read_csv(V13A / "s0_selfemployed_share.csv", index_col=0).reindex(columns=PROD).fillna(0.0)
    return piv, W, H, se


def premium(Q, definition="main", hours="cps", qcol="q", interp=True, seams=False, back1961=False,
            ces_variant="main", v2014="main", years=range(1948, 2024)):
    """Per code and year: W_eq, P, L_s; aggregate pi over productive codes with supervisors (sN < 1)."""
    piv, W, H, se = code_inputs(ces_variant)
    q = series(Q, definition, qcol, interp, seams, back1961, years, v2014)
    rho = series(Q, definition, "rho", interp, seams, back1961, years, v2014)
    rows = []
    for y in years:
        if y not in W.index or y not in piv["sN"].index:
            continue
        for c in PROD:
            g = cps.CODE_GROUP[c]
            sN, sW, sH = (float(piv[k].loc[y, c]) if np.isfinite(piv[k].loc[y, c]) else np.nan for k in ("sN", "sW", "sH"))
            Wc = float(W.loc[y, c]) if np.isfinite(W.loc[y, c]) else np.nan
            Hemp = float(H.loc[y, c] * (1 - se.loc[y, c])) if y in H.index else np.nan
            if not np.isfinite(sN) or sN >= 1 or not np.isfinite(Wc):
                continue
            qg, rg = float(q.loc[y, g]), float(rho.loc[y, g])
            W_rf, W_sup = sW * Wc, (1 - sW) * Wc
            ratio_N = (1 - sN) / sN
            H_rf = sH * Hemp
            if hours == "cps":
                L_s = H_rf * ratio_N * rg
                W_eq = W_rf * ratio_N * rg * qg
            else:                                              # variant: supervisor hours = CES remainder
                L_s = (1 - sH) * Hemp
                W_eq = W_rf / sH * (1 - sH) * qg if sH > 0 else np.nan
            rows.append(dict(year=y, code=c, grp=g, W=Wc, W_rf=W_rf, W_sup=W_sup, W_eq=W_eq, P=W_sup - W_eq,
                             H_emp=Hemp, H_rf=H_rf, L_s=L_s, sN=sN, sW=sW, sH=sH, q=qg, rho=rg))
    D = pd.DataFrame(rows)
    D["pi"] = D.P / D.W_eq
    return D


VARIANTS = {"main": dict(),
            "broad supervisors": dict(definition="broad"),
            "composition by CES": dict(definition="ces"),
            "hours = CES remainder": dict(hours="ces"),
            "Hilferding w_eq": dict(qcol="qH"),
            "seams neutralised": dict(seams=True),
            "CPS years only": dict(interp=False),
            "q, rho of 1961 before 1961": dict(back1961=True),
            "sW capped at 1": dict(ces_variant="sW capped at 1"),
            "splice 1998": dict(ces_variant="splice 1998"),
            "ASEC 2014 3/8": dict(v2014="3x8")}


def pi_all(Q):
    out, codes = [], []
    for name, kw in VARIANTS.items():
        D = premium(Q, **kw)
        A = D.groupby("year")[["P", "W_eq", "W_sup", "L_s", "H_rf"]].sum()
        A["pi"] = A.P / A.W_eq
        A["variant"] = name
        out.append(A.reset_index())
        if name == "main":
            codes.append(D)
    P = pd.concat(out, ignore_index=True)
    P.to_csv(OUT / "s0_pi.csv", index=False)
    pd.concat(codes).to_csv(OUT / "s0_premium_codes.csv", index=False)
    return P


# ================================================================ descriptive: counts CPS vs CES, checks
GROUP_SCOPES = {"G2": ["10000000"], "G3": ["20000000"], "G4": ["31000000"], "G5": ["32000000"],
                "G6": ["44220000", "43000000", "50000000"], "G9": ["60540000"],
                "G10": ["65000000", "70000000", "80000000"]}


def counts(Q):
    A = s13.ces_annual()
    rows = []
    for g, scs in GROUP_SCOPES.items():
        for y in range(1961, 2025):
            v = [A.loc[(s, y)] if (s, y) in A.index else None for s in scs]
            if any(x is None for x in v):
                continue
            n_sup = sum(float(x["01"] - x["06"]) for x in v)
            rows.append(dict(grp=g, year=y, ces_N_sup_thousand=n_sup))
    C = pd.DataFrame(rows)
    q = Q[(Q.definition == "main") & (Q.variant == "main")][["eyear", "grp", "N_sup_w"]]
    C = C.merge(q.rename(columns={"eyear": "year"}), on=["grp", "year"], how="left")
    C.to_csv(OUT / "s0_counts.csv", index=False)
    return C


def checks(Q, D):
    rows = []
    Dm = D[D.year.between(1961, 2023)]
    rows.append(dict(check="L_r + L_s vs H_emp (productive codes with supervisors, 1961-2023): median ratio",
                     value=float(((Dm.H_rf + Dm.L_s) / Dm.H_emp).median())))
    rows.append(dict(check="share of code-years with pi < 0 (main, 1961-2023)", value=float((Dm.pi < 0).mean())))
    q = Q[(Q.definition == "main") & (Q.variant == "main")]
    rows.append(dict(check="share of supervisor hours in cells closed after step 3 (main, all years)",
                     value=float((q.share_step4 + q.share_step5 + q.share_step9).mean())))
    rows.append(dict(check="cells still below the minimum after step 5 (share of supervisor hours, max over years)",
                     value=float(q.share_step9.max())))
    pd.DataFrame(rows).to_csv(OUT / "s0_checks.csv", index=False)


def main(part="all"):
    OUT.mkdir(parents=True, exist_ok=True)
    if part in ("cps", "all"):
        Q = cps_all()
    else:
        Q = pd.read_csv(OUT / "s0_qrho.csv")
    if part in ("premium", "all"):
        pi_all(Q)
        D = pd.read_csv(OUT / "s0_premium_codes.csv")
        counts(Q)
        checks(Q, D)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
