"""Iteration 2, stages 2-3: prices of production with a uniform (skill-adjusted) wage, and reduction of
complex labour without wages.

Labour vectors l (hours per unit of output, possibly skill-weighted):
  hours        unweighted hours
  wagebill     hours x relative hourly labour income (= labour income / (wbar x)); CIRCULAR, comparison only
  edu_years    hours x sum_e s_je * S_e/9                       (A-EDU, "years" scheme)
  ck_<...>     Cockshott training-time reduction: kappa_e = 1 + Y_e * H_study * (1 + teach) / (H_year * T_life),
               Y_e = S_e - 9 ('b9') or S_e ('all'); teach = teacher hours per student hour
Education structure s_je (share of HOURS by education, section level): EU KLEMS 2023 labour accounts
(Share_E, education 1=high, 2=medium, 3=low; A-EDU-KLEMS) where available (USA, DEU and EU countries),
otherwise OECD TiMBC employment shares (MEX). S = 16.5 (high), 12.5 (medium), 9 (low) years.

Anchors (z per unit of output, market price 1):
  LV_<l>        labour values with labour vector l (open system, imports at price)
  PPa           prices of production, actual industry labour income (iteration 1 "pp_actual_wage")
  PPb           prices of production, one wage per hour  (l = hours)
  PPc_<l>       prices of production, one wage per skill-adjusted hour (l = edu_years / ck_central)
All at the actual flow profit rate (capped at 0.99 R, A-RMAX).
"""
from __future__ import annotations

import itertools
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ..codes import BEA_TO_NACE, SECTIONS, divisions
from ..core import Economy, leontief_inverse

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "v2"
S_YEARS = {"high": 16.5, "medium": 12.5, "low": 9.0}
KLEMS_EDU = {1: "high", 2: "medium", 3: "low"}
ISO3_TO_KLEMS = {"USA": "US", "DEU": "DE", "FRA": "FR", "ITA": "IT", "ESP": "ES", "NLD": "NL", "AUT": "AT",
                 "POL": "PL", "CZE": "CZ", "JPN": "JP", "GBR": "UK"}

CK_GRID = [dict(H_study=hs, T_life=tl, count=cnt, teach=te)
           for hs, tl, cnt, te in itertools.product([1000, 1600, 2000], [35, 40, 45], ["b9", "all"], [0.0, 1 / 15])]
CK_CENTRAL = dict(H_study=1600, T_life=40, count="b9", teach=0.0)
H_YEAR = 1800.0


def ck_name(p):
    return f"ck_{p['H_study']}_{p['T_life']}_{p['count']}_{'t' if p['teach'] else 'n'}"


def ck_kappa(p: dict) -> dict:
    out = {}
    for e, S in S_YEARS.items():
        Y = max(S - 9.0, 0.0) if p["count"] == "b9" else S
        out[e] = 1 + Y * p["H_study"] * (1 + p["teach"]) / (H_YEAR * p["T_life"])
    return out


@lru_cache(maxsize=None)
def _klems():
    p = ROOT / "data" / "raw" / "euklems" / "labour_accounts.csv"
    d = pd.read_csv(p)
    d = d[d.education.isin([1, 2, 3])]
    g = d.groupby(["geo_code", "nace_r2_code", "year", "education"]).Share_E.sum().reset_index()
    return g


def section_shares(country3: str, year: int) -> tuple[pd.DataFrame, str]:
    """Hours shares by education (columns high/medium/low), index = NACE section letter."""
    geo = ISO3_TO_KLEMS.get(country3)
    if geo is not None:
        g = _klems()
        g = g[g.geo_code == geo]
        if not g.empty:
            y = min(g.year.unique(), key=lambda t: abs(t - year))
            w = g[g.year == y].pivot_table(index="nace_r2_code", columns="education", values="Share_E")
            w.columns = [KLEMS_EDU[c] for c in w.columns]
            w = w.div(w.sum(1), axis=0)
            return w, f"EU KLEMS {y}"
    from .. import oecd
    sh = oecd.education_shares(country3, year)        # by FIGARO industry
    sh.columns = ["low", "medium", "high"]
    sec = sh.groupby([c[0] for c in sh.index]).mean()
    return sec, "OECD TiMBC"


def label_section(label: str, source: str) -> str:
    part = label.split("+")[0]
    if source == "bea":
        d = BEA_TO_NACE[part][0]
        return next(L for L, r in SECTIONS.items() if d in r)
    return part[0]


def skill_index(e: Economy, country3: str, year: int, source: str, kappa: dict) -> np.ndarray:
    sec, _ = section_shares(country3, year)
    k = pd.Series(kappa)
    idx = (sec[list(kappa)] * k).sum(1)
    out = np.array([idx.get(label_section(l, source), np.nan) for l in e.labels])
    return np.where(np.isfinite(out), out, np.nanmean(out))


def labour_vectors(e: Economy, country3: str, year: int, source: str) -> dict:
    h = e.l()
    li = e.labour_income
    wbar = li.sum() / e.hours.sum()
    rel = np.divide(li / e.hours, wbar, out=np.ones(e.n), where=e.hours > 0)
    out = {"hours": h, "wagebill": h * rel,
           "edu_years": h * skill_index(e, country3, year, source, {k: v / 9.0 for k, v in S_YEARS.items()})}
    for p in CK_GRID:
        out[ck_name(p)] = h * skill_index(e, country3, year, source, ck_kappa(p))
    return out


def lv_z(e: Economy, M, Mm, l):
    L = leontief_inverse(M)
    lam, mu = l @ L, Mm.sum(0) @ L
    x = e.x
    return (x.sum() - mu @ x) / (lam @ x) * lam + mu


def anchors(e: Economy, country3: str, year: int, source: str, capital: bool) -> dict:
    lv = labour_vectors(e, country3, year, source)
    M, Mm, _ = e.system(capital, False, "price", e.l())
    r = e.actual_profit_rate(capital)
    R = e.max_profit_rate(capital, "price")
    r = min(r, 0.99 * R)
    out = {f"LV_{k}": lv_z(e, M, Mm, v) for k, v in lv.items()}
    out["PPa"] = e.prices_of_production(r, capital, "price", "actual")
    out["PPb"] = e.prices_of_production(r, capital, "price", "uniform", l=lv["hours"])
    for k in ["edu_years", ck_name(CK_CENTRAL)] + [ck_name(p) for p in CK_GRID]:
        out[f"PPc_{k}"] = e.prices_of_production(r, capital, "price", "uniform", l=lv[k])
    return out


def run(source="figaro"):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    if source == "bea":
        from .. import bea
        items = [("USA_BEA", y, lambda y=y: bea.build(y)[0]) for y in range(1998, 2024)]
    else:
        from ..economy import build
        items = [(c, y, lambda c=c, y=y: build(c, y)[0]) for c in ("USA", "DEU", "MEX") for y in range(2010, 2024)]
    for c, y, mk in items:
        e = mk()
        c3 = "USA" if c == "USA_BEA" else c
        for cap in (True, False):
            for name, z in anchors(e, c3, y, source, cap).items():
                rows.append(pd.DataFrame(dict(country=c, year=y, capital=cap, basis=name, industry=e.labels,
                                              z=z, x=e.x, P=e.meta["price_index"])))
        print(source, c, y, flush=True)
    df = pd.concat(rows, ignore_index=True)
    df.to_parquet(OUT / f"anchors_{source}.parquet")
    return df


if __name__ == "__main__":
    import sys
    run(sys.argv[1] if len(sys.argv) > 1 else "figaro")
