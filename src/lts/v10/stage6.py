"""Iteration 10, stage 6: household labour in the symmetric framing (pre_registration_v10.md, stage 6; journal 12).

d = hours of unpaid domestic work / hours of market work (ATUS for the US by year, HETUS 2010/2000 for European
countries). Closed system with one bundle per hour (A-CLOSE-U) plus d hours of simple labour per hour of labour
(block M[n+i, n+j] = d * H_i / sum H). Outcome 2 (share of commodity bases better than labour, MAWD, capital, imports
at price, subset all) recomputed with d = 0 and with d; share of the labour part L_X = v_X^sym - O_X (descriptive).

Usage: PYTHONPATH=src python -P -m lts.v10.stage6
Outputs: results/v10/s6_*.csv
"""
from __future__ import annotations

import types
import zipfile

import numpy as np
import pandas as pd

from ..core import Economy, content, spectral_radius
from ..economy import build
from ..levels import mask_for
from ..metrics import ratio_metrics
from ..v6.series import ROOT
from ..v8.rule import mean_row
from .stage1 import es_json

OUT = ROOT / "results" / "v10"
R10 = ROOT / "data" / "raw" / "v10"
V3 = ("USA", "DEU", "MEX", "FRA", "ITA", "ESP", "NLD", "AUT", "POL", "CZE", "KOR", "JPN", "GBR")
ISO2 = {"DEU": "DE", "FRA": "FR", "ITA": "IT", "ESP": "ES", "NLD": "NL", "AUT": "AT", "POL": "PL", "CZE": "CZ",
        "GBR": "UK"}
YEARS = range(2010, 2023)


# ================================================================ d
def atus_d():
    z = zipfile.ZipFile(R10 / "bls_atussum-0324.zip")
    d = pd.read_csv(z.open("atussum_0324.dat"), low_memory=False)
    dom = [c for c in d.columns if c.startswith(("t02", "t03", "t07"))]
    work = [c for c in d.columns if c.startswith("t0501")]
    d["w"] = np.where(d.TUYEAR == 2020, d.TU20FWGT, d.TUFNWGTP)
    d = d[d.w > 0]
    d["dom"] = d[dom].sum(axis=1)
    d["work"] = d[work].sum(axis=1)
    g = d.groupby("TUYEAR")
    out = pd.DataFrame({"dom_min": g.apply(lambda x: np.average(x.dom, weights=x.w), include_groups=False),
                        "work_min": g.apply(lambda x: np.average(x.work, weights=x.w), include_groups=False)})
    out["d"] = out.dom_min / out.work_min
    return out


def hhmm(s):
    try:
        h, m = str(s).split(":")
        return int(h) + int(m) / 60
    except ValueError:
        return np.nan


def hetus_d():
    d = es_json(R10 / "es_tus_00selfstat.json")
    d = d[(d.unit == "TIME_SP") & (d.sex == "T") & (d.wstatus == "POP") & d.acl00.isin(["AC3", "AC1A"])]
    d["h"] = d.value.map(hhmm)
    p = d.pivot_table(index=["geo", "time"], columns="acl00", values="h").reset_index()
    p["d"] = p.AC3 / p.AC1A
    p = p.sort_values("time").groupby("geo").last()
    return p


# ================================================================ closed system with domestic labour
def patch(e: Economy, d: float):
    base = Economy.system

    def system(self, capital, closed, imports, l):
        M, Mm, lab = base(self, capital, closed, imports, l)
        if not closed or d <= 0:
            return M, Mm, lab
        n = self.n
        M = M.copy()
        s = self.hours / self.hours.sum()
        M[n:, n:] += d * np.outer(s, np.ones(n))
        Mt = M + Mm if imports == "competitive" else M
        if spectral_radius(Mt) > 0.995:                       # scale the bundle down (A-CLOSE-U)
            Bd = M[:n, n:].copy()
            Bm = Mm[:n, n:].copy()
            lo, hi = 0.0, 1.0
            for _ in range(40):
                mid = (lo + hi) / 2
                T = M.copy()
                T[:n, n:] = Bd * mid
                if imports == "competitive":
                    T = T + Mm
                lo, hi = (mid, hi) if spectral_radius(T) <= 0.995 else (lo, mid)
            M[:n, n:] = Bd * lo
            Mm = Mm.copy()
            Mm[:n, n:] = Bm * lo
            self.meta["_d_bundle_scale"] = lo
        return M, Mm, lab
    e.system = types.MethodType(system, e)
    return e


def shares(e, capital=True):
    """outcome-2 share and labour part of symmetric values, closed system with one bundle per hour"""
    labels = list(e.labels)
    m = mask_for(labels, set())
    zl = e.basis_ratios("L", capital, "uniform", "price", l=e.l())
    ml = ratio_metrics(zl, e.x, m)["mawd"]
    better, lshare = [], []
    Mo, _, _ = Economy.system(e, capital, False, "price", e.l())
    Mc, _, lab = e.system(capital, "uniform", "price", e.l())
    D, _ = e.dep_coeffs()
    for k in range(e.n):
        if e.A[k].sum() + (D[k].sum() if capital else 0) <= 0:
            continue
        z = e.basis_ratios(k, capital, "uniform", "price", l=e.l())
        better.append(ratio_metrics(z, e.x, m)["mawd"] < ml)
        lam_c = content(Mc, Mc[k, :], zero_row=k)[:e.n]
        lam_o = content(Mo, Mo[k, :], zero_row=k)
        tot_c, tot_o = (lam_c * e.x)[m].sum(), (lam_o * e.x)[m].sum()
        lshare.append(1 - tot_o / tot_c if tot_c > 0 else np.nan)
    return float(np.mean(better)), float(np.nanmedian(lshare)), ml


def run(resume=False):
    A = atus_d()
    A.to_csv(OUT / "s6_atus_d.csv")
    Hd = hetus_d()
    Hd.to_csv(OUT / "s6_hetus_d.csv")
    print(A.round(3).to_string(), "\n", Hd.round(3).to_string(), flush=True)
    rows = []
    done = set()
    if resume and (OUT / "s6_outcome2.csv").exists():                 # continue an interrupted run (complete countries)
        prev = pd.read_csv(OUT / "s6_outcome2.csv")
        done = {c for c, g in prev.groupby("country") if len(g) == len(YEARS)}
        rows = prev[prev.country.isin(done)].to_dict("records")
    for c in V3:
        if c in done:
            continue
        if c == "USA":
            dmap = {y: float(A.d.get(y, A.d.loc[A.index[A.index <= y].max()])) for y in YEARS}
            src = "ATUS"
        elif ISO2.get(c) in Hd.index:
            v = float(Hd.loc[ISO2[c], "d"])
            dmap = {y: v for y in YEARS}
            src = f"HETUS {Hd.loc[ISO2[c], 'time']}"
        else:
            continue
        for y in YEARS:
            e0, _ = build(c, y)
            s0, l0, ml0 = shares(e0)
            e1, _ = build(c, y)
            e1 = patch(e1, dmap[y])
            note = ""
            try:
                s1, l1, ml1 = shares(e1)
            except ValueError as exc:                        # journal 14: d >= 1 -> not productive
                s1 = l1 = ml1 = np.nan
                note = f"not productive: {exc}"[:80]
            rows.append(dict(country=c, year=y, d=dmap[y], d_source=src, share_d0=s0, share_d=s1, labour_part_d0=l0,
                             labour_part_d=l1, mawd_labour_d0=ml0, mawd_labour_d=ml1,
                             bundle_scale=e1.meta.get("_d_bundle_scale", 1.0), note=note))
            print("s6", c, y, round(s0, 3), s1, flush=True)
        pd.DataFrame(rows).to_csv(OUT / "s6_outcome2.csv", index=False)
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "s6_outcome2.csv", index=False)
    by = R.groupby("country")[["share_d0", "share_d", "labour_part_d0", "labour_part_d", "d"]].mean()
    ok = by.share_d.notna()
    out = [mean_row("2 [d = 0, countries with d]: share of commodity bases better than labour, symmetric",
                    by.share_d0.to_numpy(), 0.5, 0.10, "<"),
           mean_row("2 [d = 0, countries where the system with d is productive]", by.share_d0[ok].to_numpy(), 0.5, 0.10, "<"),
           mean_row("2 [with household labour d, productive countries only]", by.share_d[ok].to_numpy(), 0.5, 0.10, "<")]
    O = pd.DataFrame(out)
    O.to_csv(OUT / "s6_outcomes.csv", index=False)
    by.to_csv(OUT / "s6_by_country.csv")
    pd.set_option("display.width", 250)
    print(by.round(3).to_string(), "\n", O.round(4).to_string(), flush=True)


if __name__ == "__main__":
    import sys
    run(resume="resume" in sys.argv[1:])
