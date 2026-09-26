"""Iteration 4, part B: industry panel for the profit tests (EU KLEMS 2023 + Eurostat/OECD CFC).

Per country-year-industry (finest non-overlapping KLEMS industries, 2000-2021):
  VA, GO, II, COMP, EMP, EMPE, H, HE (KLEMS national accounts), K (K_GFCF, net current replacement cost),
  K_tang = K_GFCF - K_Soft_DB - K_RD - K_OIPP, K_nonNA (INTAN), I_nonNA, VAadj (intangibles analytical),
  cfc_sh = P51C/B1G, tax_sh = D29X39/B1G (Eurostat nama_10_a64; OECD Table 6 for US, JP and where
  Eurostat is missing).  Ratios are applied to KLEMS VA so that units and vintages stay KLEMS's (A-V4-B-RATIO).
Derived (pre-registered): W_se = COMP/HE (H - HE); W = COMP + W_se; CFC = cfc_sh VA; TAX = tax_sh VA;
  PI = VA - COMP - W_se - CFC - TAX (net, mixed-income corrected); PI_nomi = VA - COMP - CFC - TAX;
  r = PI/K.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .. import oecd
from ..codes import SECTIONS

ROOT = Path(__file__).resolve().parents[3]
KL = ROOT / "data" / "raw" / "euklems"
RAW4 = ROOT / "data" / "raw" / "v4"
OUT = ROOT / "results" / "v4"

FINE = ["A", "B", "C10-C12", "C13-C15", "C16-C18", "C19", "C20", "C21", "C22-C23", "C24-C25", "C26", "C27",
        "C28", "C29-C30", "C31-C33", "D", "E", "F", "G45", "G46", "G47", "H49", "H50", "H51", "H52", "H53", "I",
        "J58-J60", "J61", "J62-J63", "K", "L", "M", "N", "O", "P", "Q86", "Q87-Q88", "R", "S", "T", "U"]
ALT = {"C20": "C20-C21", "C21": "C20-C21", "C26": "C26-C27", "C27": "C26-C27", "D": "D-E", "E": "D-E",
       "R": "R-S", "S": "R-S", "Q86": "Q", "Q87-Q88": "Q", "M": "M-N", "N": "M-N"}
RENT = {"B", "D", "E", "D-E", "K", "L", "L68A"}                     # pre-registered exclusions (B2.4)
GOV = {"O", "P", "Q", "Q86", "Q87-Q88", "T", "U", "O-Q"}            # B2.5
COUNTRIES = ["AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "EL", "ES", "FI", "FR", "HR", "HU", "IE", "IT",
             "LT", "LU", "LV", "MT", "NL", "PL", "PT", "RO", "SE", "SI", "SK", "UK", "US", "JP"]
LETTERS = "ABCDEFGHIJKLMNOPQRSTU"


def divs(code: str) -> frozenset:
    """NACE code (Eurostat/KLEMS/OECD style) -> set of 2-digit divisions."""
    c = code.replace("_", "+")
    out = set()
    for part in c.split("+"):
        m = re.fullmatch(r"([A-U])(\d{2})?(?:-([A-U])?(\d{2}))?A?", part)
        m2 = re.fullmatch(r"([A-U])-([A-U])", part)
        if m2:
            for L in LETTERS[LETTERS.index(m2.group(1)):LETTERS.index(m2.group(2)) + 1]:
                out |= set(SECTIONS[L])
        elif m and m.group(2) is None and m.group(4) is None:
            out |= set(SECTIONS[m.group(1)])
        elif m:
            lo = int(m.group(2))
            hi = int(m.group(4)) if m.group(4) else lo
            out |= set(range(lo, hi + 1))
        elif re.fullmatch(r"(\d{2})", part):
            out.add(int(part))
        else:
            return frozenset()
    return frozenset(out)


@lru_cache(None)
def _klems():
    na = pd.read_csv(KL / "national_accounts.csv", low_memory=False)
    ca = pd.read_csv(KL / "capital_accounts.csv", low_memory=False,
                     usecols=["nace_r2_code", "geo_code", "year", "K_GFCF", "K_Soft_DB", "K_RD", "K_OIPP", "K_Rstruc"])
    ia = pd.read_csv(KL / "intangibles_analytical.csv", low_memory=False,
                     usecols=["nace_r2_code", "geo_code", "year", "K_NonNatAcc", "I_NonNatAcc", "VAadj", "VA_CP"])
    ia = ia.rename(columns={"VA_CP": "VA_ia"})
    key = ["nace_r2_code", "geo_code", "year"]
    d = na.merge(ca, on=key, how="left").merge(ia, on=key, how="left")
    return d.rename(columns={"nace_r2_code": "ind", "geo_code": "geo"})


def _eurostat(item):
    d = pd.read_csv(RAW4 / f"nama64_{item}.tsv.gz", sep="\t")
    k = d.columns[0]
    parts = d[k].str.split(",", expand=True)
    parts.columns = ["freq", "unit", "nace", "item", "geo"]
    long = pd.concat([parts, d.iloc[:, 1:]], axis=1).melt(id_vars=list(parts.columns), var_name="year", value_name="v")
    long["year"] = long.year.str.strip().astype(int)
    long["v"] = pd.to_numeric(long.v.astype(str).str.extract(r"(-?[\d.]+)")[0], errors="coerce")
    return long[["geo", "nace", "year", "v"]].dropna()


@lru_cache(None)
def na_items():
    """Long frame geo, nace, year, P51C, D29X39, B1G from Eurostat, plus US/JP (and UK gaps) from OECD."""
    frames = []
    for it in ("P51C", "D29X39", "B1G"):
        frames.append(_eurostat(it).assign(item=it))
    e = pd.concat(frames).pivot_table(index=["geo", "nace", "year"], columns="item", values="v").reset_index()
    e["src"] = "Eurostat"
    o = []
    for c3, c2 in (("USA", "US"), ("JPN", "JP"), ("GBR", "UK")):
        cols = {}
        for it in ("P51C", "D29X39", "B1G"):
            s = oecd.series(c3, "table6", it, "XDC", "V")
            if not s.empty:
                cols[it] = s.stack()
        if cols:
            f = pd.DataFrame(cols).reset_index()
            f.columns = ["year", "nace"] + list(cols)
            f["geo"], f["src"] = c2, "OECD T6"
            o.append(f)
    o = pd.concat(o)
    o["nace"] = o.nace.str.replace("T", "-", regex=False).where(~o.nace.str.fullmatch(r"[A-U]T[A-U]"), o.nace)
    e = pd.concat([e, o[~o.set_index(["geo", "year"]).index.isin(e.set_index(["geo", "year"]).index)]])
    return e


def ratio_for(nai: pd.DataFrame, target: frozenset):
    """Sum P51C, D29X39, B1G over the finest codes that exactly partition `target`."""
    cand = [(c, divs(c)) for c in nai.nace.unique()]
    cand = [(c, s) for c, s in cand if s and s <= target]
    exact = [c for c, s in cand if s == target]
    if exact:
        g = nai[nai.nace == exact[0]]
    else:
        maximal = [(c, s) for c, s in cand if not any(s < s2 for _, s2 in cand)]
        if not maximal or set().union(*[s for _, s in maximal]) != set(target) or \
                sum(len(s) for _, s in maximal) != len(target):
            return None
        g = nai[nai.nace.isin([c for c, _ in maximal])]
    s = g[["P51C", "D29X39", "B1G"]].sum(min_count=1)
    if not np.isfinite(s.B1G) or s.B1G <= 0:
        return None
    return float(s.P51C / s.B1G), float(s.D29X39 / s.B1G) if np.isfinite(s.D29X39) else np.nan


def build_panel(years=range(2000, 2022)):
    kl = _klems()
    nai = na_items()
    rows = []
    for geo in COUNTRIES:
        g = kl[(kl.geo == geo) & kl.year.isin(years)]
        if g.empty:
            continue
        ng = nai[nai.geo == geo]
        for y, gy in g.groupby("year"):
            byind = gy.set_index("ind")
            ny = ng[ng.year == y]
            chosen = []
            for ind in FINE:
                ok = lambda i: i in byind.index and all(np.isfinite(byind.loc[i, v]) for v in
                                                         ("VA_CP", "COMP", "H_EMP", "H_EMPE", "K_GFCF", "GO_CP"))
                if ok(ind):
                    chosen.append(ind)
                elif ind in ALT and ok(ALT[ind]) and ALT[ind] not in chosen:
                    chosen.append(ALT[ind])
            for ind in chosen:
                r = byind.loc[ind]
                rt = ratio_for(ny, divs(ind)) if not ny.empty else None
                rows.append(dict(geo=geo, year=y, ind=ind, VA=r.VA_CP, GO=r.GO_CP, II=r.II_CP, COMP=r.COMP,
                                 EMP=r.EMP, EMPE=r.EMPE, H=r.H_EMP, HE=r.H_EMPE, GO_Q=r.GO_Q,
                                 K=r.K_GFCF, K_tang=r.K_GFCF - np.nansum([r.K_Soft_DB, r.K_RD, r.K_OIPP]),
                                 K_nonNA=r.K_NonNatAcc, I_nonNA=r.I_NonNatAcc, VAadj=r.VAadj, VA_ia=r.VA_ia,
                                 cfc_sh=rt[0] if rt else np.nan, tax_sh=rt[1] if rt else np.nan))
        print("panel", geo, flush=True)
    p = pd.DataFrame(rows)
    p["W_se"] = np.where(p.HE > 0, p.COMP / p.HE * (p.H - p.HE).clip(lower=0), 0.0)
    p["W"] = p.COMP + p.W_se
    p["CFC"] = p.cfc_sh * p.VA
    p["TAX"] = p.tax_sh.fillna(0) * p.VA
    p["PI"] = p.VA - p.W - p.CFC - p.TAX
    p["PI_nomi"] = p.VA - p.COMP - p.CFC - p.TAX
    p["rent"] = p.ind.isin(RENT)
    p["gov"] = p.ind.isin(GOV)
    return p


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    p = build_panel()
    p.to_csv(OUT / "b_panel.csv.gz", index=False)
    print(p.groupby("geo").agg(n=("ind", "size"), cfc_ok=("cfc_sh", lambda s: s.notna().mean())).to_string())
