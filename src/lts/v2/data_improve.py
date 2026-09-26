"""Iteration 2, stage 5: data improvements for the US long series (BEA, 1998-2023).

Variants (A-DATA-V2-*):
  base       iteration-1 BEA economy (FTE-based hours, economy-wide GFCF composition for depreciation)
  bls        hours from BLS "Labor productivity, detailed industries" (hours worked, all workers, NAICS) for
             BEA industries fully covered by BLS aggregates (~35 of 69); the others keep FTE-based hours
             rescaled by the ratio BLS/FTE hours on the covered industries of the same year
  capflow    depreciation allocated to supplying commodities with the investment composition of the using
             industry's column in the BEA 1997 capital flow table (180 commodities x 22 industries, producers'
             prices), mapped to BEA summary commodities by NAICS prefix; government industries keep the
             economy-wide composition; domestic/imported split by the year's GFCF import share per commodity
  bls+capflow both
Outputs: results/v2/data_improve.csv (labour percentile among commodity bases, open & symmetric-uniform,
MAWD and d; MAWD of LV, PPa, PPb), subset all and core.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .. import bea
from ..core import Economy, leontief_inverse
from ..levels import SUBSETS, mask_for
from ..metrics import ratio_metrics

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "results" / "v2"

NAICS = {'111CA': ['111', '112'], '113FF': ['113', '114', '115'], '211': ['211'], '212': ['212'], '213': ['213'],
         '22': ['22'], '23': ['23'], '321': ['321'], '327': ['327'], '331': ['331'], '332': ['332'], '333': ['333'],
         '334': ['334'], '335': ['335'], '3361MV': ['3361', '3362', '3363'], '3364OT': ['3364', '3365', '3366', '3369'],
         '337': ['337'], '339': ['339'], '311FT': ['311', '312'], '313TT': ['313', '314'], '315AL': ['315', '316'],
         '322': ['322'], '323': ['323'], '324': ['324'], '325': ['325'], '326': ['326'], '42': ['42'], '441': ['441'],
         '445': ['445'], '452': ['452'], '4A0': ['442', '443', '444', '446', '447', '448', '451', '453', '454'],
         '481': ['481'], '482': ['482'], '483': ['483'], '484': ['484'], '485': ['485'], '486': ['486'],
         '487OS': ['487', '488', '492'], '493': ['493'], '511': ['511', '516'], '512': ['512'],
         '513': ['513', '515', '517'], '514': ['514', '518', '519'], '521CI': ['521', '522'], '523': ['523'],
         '524': ['524'], '525': ['525'], 'HS': ['531'], 'ORE': ['531'], '532RL': ['532', '533'], '5411': ['5411'],
         '5415': ['5415'], '5412OP': ['541'], '55': ['55'], '561': ['561'], '562': ['562'], '61': ['61'],
         '621': ['621'], '622': ['622'], '623': ['623'], '624': ['624'], '711AS': ['711', '712'], '713': ['713'],
         '721': ['721'], '722': ['722'], '81': ['81']}
# BLS aggregate codes that fully cover a BEA industry
BLS_COVER = {'211': ['211'], '212': ['212'], '213': ['213'], '22': ['221'], '321': ['321'], '327': ['327'],
             '331': ['331'], '332': ['332'], '333': ['333'], '334': ['334'], '335': ['335'],
             '3361MV': ['3361', '3362', '3363'], '3364OT': ['3364', '3365', '3366', '3369'], '337': ['337'],
             '339': ['339'], '311FT': ['311', '312'], '313TT': ['313', '314'], '315AL': ['315', '316'],
             '322': ['322'], '323': ['323'], '324': ['324'], '325': ['325'], '326': ['326'], '42': ['42'],
             '441': ['441'], '445': ['445'], '452': ['452'],
             '4A0': ['442', '443', '444', '446', '447', '448', '451', '453', '454'], '481': ['481'], '484': ['484'],
             '493': ['493'], '511': ['511'], '721': ['721'], '722': ['722']}


@lru_cache(maxsize=None)
def bls_hours() -> pd.DataFrame:
    d = pd.read_excel(RAW / "bls" / "labor-productivity-detailed-industries.xlsx", sheet_name="MachineReadable")
    h = d[(d.Measure == "Hours worked") & (d.Units == "Millions of hours")]
    return h.assign(NAICS=h.NAICS.astype(str)).pivot_table(index="Year", columns="NAICS", values="Value")


def cft_code(bea_code: str) -> str:
    """CFT 22-industry code of a BEA summary industry (NAICS based)."""
    p = NAICS[bea_code][0]
    if p[:2] in ("31", "32", "33"):
        p3 = int(p[:3])
        return "31" if p3 <= 316 else ("32" if p3 <= 327 else "33")
    if p[:2] in ("44", "45"):
        return "4A"
    return p[:2]


@lru_cache(maxsize=None)
def capflow_matrix() -> pd.DataFrame:
    d = pd.read_excel(RAW / "bea" / "flow1997.xls", sheet_name="180x22Combined", header=None)
    codes = [str(c).strip() for c in d.iloc[3, 3:].tolist()]
    body = d.iloc[4:184, :].copy()
    body = body[pd.to_numeric(body[0], errors="coerce").notna()]
    comm = body[1].astype(str).str.strip()
    vals = body.iloc[:, 3:3 + len(codes)].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    vals.columns = codes
    vals.index = comm
    return vals


def commodity_to_bea(code: str, bea_codes: list[str]) -> str | None:
    digits = "".join(ch for ch in code if ch.isdigit())
    best, blen = None, 0
    for b in bea_codes:
        for p in NAICS.get(b, []):
            if digits.startswith(p) and len(p) > blen:
                best, blen = b, len(p)
    if best is None and digits.startswith("23"):
        best = "23"
    return best


def capflow_shares(labels: list[str]) -> np.ndarray:
    """(n commodities x n industries) investment composition by using industry (columns sum to 1)."""
    cf = capflow_matrix()
    parts = {lab: lab.split("+") for lab in labels}
    flat = [p for ps in parts.values() for p in ps if p in NAICS]
    idx = {lab: i for i, lab in enumerate(labels)}
    part_to_label = {p: lab for lab, ps in parts.items() for p in ps}
    comp = {}
    for col in cf.columns:
        v = np.zeros(len(labels))
        for comm, val in cf[col].items():
            b = commodity_to_bea(comm, flat)
            if b is not None and val > 0:
                v[idx[part_to_label[b]]] += val
        comp[col] = v / v.sum() if v.sum() > 0 else None
    S = np.full((len(labels), len(labels)), np.nan)
    for j, lab in enumerate(labels):
        p = lab.split("+")[0]
        if p in NAICS:
            c = comp.get(cft_code(p))
            if c is not None:
                S[:, j] = c
    return S


class CapflowEconomy(Economy):
    def dep_coeffs(self):
        d = np.divide(self.cfc, self.x, out=np.zeros(self.n), where=self.x > 0)
        tot = self.gfcf_dom + self.gfcf_imp
        imp_sh = np.divide(self.gfcf_imp, tot, out=np.zeros(self.n), where=tot > 0)
        base = tot / tot.sum()
        S = self.meta["capflow"]
        S = np.where(np.isnan(S), base[:, None], S)
        D = S * (1 - imp_sh)[:, None] * d[None, :]
        Dm = S * imp_sh[:, None] * d[None, :]
        return D, Dm


def variant(e: Economy, year: int, use_bls: bool, use_cf: bool) -> tuple[Economy, dict]:
    info = {}
    hours = e.hours.copy()
    if use_bls:
        H = bls_hours()
        if year in H.index:
            cov = np.zeros(e.n, bool)
            new = hours.copy()
            for i, lab in enumerate(e.labels):
                p = lab.split("+")
                if len(p) == 1 and p[0] in BLS_COVER and all(c in H.columns for c in BLS_COVER[p[0]]):
                    v = H.loc[year, BLS_COVER[p[0]]]
                    if v.notna().all():
                        new[i] = v.sum() * 1e6
                        cov[i] = True
            f = new[cov].sum() / hours[cov].sum()
            new[~cov] = hours[~cov] * f
            info.update(bls_covered=int(cov.sum()), bls_factor=float(f),
                        bls_share_hours_covered=float(new[cov].sum() / new.sum()))
            hours = new
    cls = CapflowEconomy if use_cf else Economy
    meta = dict(e.meta)
    if use_cf:
        meta["capflow"] = capflow_shares(e.labels)
        info["capflow_cols_mapped"] = int((~np.isnan(meta["capflow"][0])).sum())
    e2 = cls(labels=e.labels, A=e.A, Am=e.Am, x=e.x, hours=hours, wages=e.wages, cfc=e.cfc, gfcf_dom=e.gfcf_dom,
             gfcf_imp=e.gfcf_imp, hh_dom=e.hh_dom, hh_imp=e.hh_imp, va=e.va, labour_income=e.labour_income, meta=meta)
    return e2, info


def evaluate(e: Economy) -> list[dict]:
    rows = []
    M, Mm, _ = e.system(True, False, "price", e.l())
    zL = e.basis_ratios("L", True, False, "price")
    r = min(e.actual_profit_rate(True), 0.99 * e.max_profit_rate(True, "price"))
    zPa = e.prices_of_production(r, True, "price", "actual")
    zPb = e.prices_of_production(r, True, "price", "uniform")
    ks = [k for k in range(e.n) if M[k].sum() > 0]
    zo = {k: e.basis_ratios(k, True, False, "price") for k in ks}
    zs = {k: e.basis_ratios(k, True, "uniform", "price") for k in ks}
    for sub in ("all", "core"):
        m = mask_for(e.labels, SUBSETS[sub], bea.BEA_ROLE)
        rL = ratio_metrics(zL, e.x, m)
        r = dict(subset=sub, mawd_LV=rL["mawd"], d_LV=rL["d"], mawd_PPa=ratio_metrics(zPa, e.x, m)["mawd"],
                 mawd_PPb=ratio_metrics(zPb, e.x, m)["mawd"])
        for name, zz in (("open", zo), ("sym", zs)):
            for met in ("mawd", "d"):
                vals = np.array([ratio_metrics(z, e.x, m)[met] for z in zz.values()])
                r[f"share_better_{name}_{met}"] = float((vals < rL[met]).mean())
        rows.append(r)
    return rows


def run():
    rows = []
    for y in range(1998, 2024):
        e0, _ = bea.build(y)
        for name, ub, uc in (("base", False, False), ("bls", True, False), ("capflow", False, True),
                             ("bls+capflow", True, True)):
            e, info = variant(e0, y, ub, uc)
            for r in evaluate(e):
                rows.append(dict(year=y, variant=name, **info, **r))
        print("data_improve", y, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "data_improve.csv", index=False)
    return df


if __name__ == "__main__":
    run()
