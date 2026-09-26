"""Iteration 4, A1: which industries' employment is reconstructed by splitting an aggregate, and
alternative splits (donor-median structure; Eurostat SBS persons employed)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .. import oecd
from ..codes import divisions
from ..economy import build, national_blocks
from ..figaro import INDUSTRIES
from ..v3.placebo_disp import COUNTRIES, YEARS

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "v4"
FIG_DIV = {j: divisions(j) for j in INDUSTRIES}


def parent_divs(code: str) -> frozenset:
    return divisions(code) or frozenset()


def label_split(labels, split_kids: dict) -> np.ndarray:
    """A label is 'split' if one of its members got persons from an aggregate that is not contained
    in the label itself (merged USA labels that cover the whole aggregate are observed)."""
    out = []
    for lab in labels:
        ld = set().union(*[FIG_DIV.get(m) or set() for m in lab.split("+")])
        s = False
        for m in lab.split("+"):
            if m in split_kids and m != "U":
                if not parent_divs(split_kids[m]) <= ld:
                    s = True
        out.append(s)
    return np.array(out)


@lru_cache(None)
def fig_persons(country: str, year: int):
    """FIGARO-level persons and split log of the default (output) split."""
    c2 = oecd.ISO3_TO_2[country]
    xs = pd.Series(national_blocks(c2, year)["x"], index=INDUSTRIES)
    lab = oecd.labour_block(country, year, xs)
    return lab["persons"], lab["log"]["split_kids"], lab["log"]["hpp_fallback_list"]


def donor_splitter(country: str, year: int) -> pd.Series:
    """Median over donor countries (the other 12, where the aggregate's children are all observed)
    of each child's share of persons in the target's aggregate.  NaN where no donor exists."""
    _, kids, _ = fig_persons(country, year)
    out = pd.Series(np.nan, index=INDUSTRIES)
    for kid, pcode in kids.items():
        if kid == "U":
            continue
        P = parent_divs(pcode)
        members = [j for j in INDUSTRIES if FIG_DIV[j] and FIG_DIV[j] <= P]
        shares = []
        for d in COUNTRIES:
            if d == country:
                continue
            pers, dk, _ = fig_persons(d, year)
            if any(m in dk for m in members):
                continue
            tot = pers.reindex(members).sum()
            if tot > 0 and np.isfinite(pers.get(kid, np.nan)):
                shares.append(pers[kid] / tot)
        if len(shares) >= 1:
            out[kid] = float(np.median(shares))
    return out


@lru_cache(None)
def _sbs():
    """Eurostat SBS persons employed (V16110) by NACE division, 2008-2020 (sbs_na_sca_r2) and
    2021+ (sbs_ovw_act, EMP_NR... if present)."""
    frames = []
    for f in ("sbs_na_sca_r2.tsv.gz",):
        p = ROOT / "data" / "raw" / "v4" / f
        if not p.exists():
            continue
        d = pd.read_csv(p, sep="\t")
        k = d.columns[0]
        parts = d[k].str.split(",", expand=True)
        parts.columns = k.split("\\")[0].split(",")
        d = pd.concat([parts, d.iloc[:, 1:]], axis=1)
        d = d[d.indic_sb == "V16110"]
        long = d.melt(id_vars=list(parts.columns), var_name="year", value_name="v")
        long["year"] = long.year.str.strip().astype(int)
        long["v"] = pd.to_numeric(long.v.astype(str).str.extract(r"([\d.]+)")[0], errors="coerce")
        frames.append(long[["geo", "nace_r2", "year", "v"]])
    return pd.concat(frames) if frames else pd.DataFrame(columns=["geo", "nace_r2", "year", "v"])


def sbs_splitter(country: str, year: int) -> pd.Series:
    """Persons employed from SBS summed over the divisions of each FIGARO industry (NaN if any
    division missing or outside SBS coverage)."""
    from ..v3.weights import ISO2
    g = ISO2.get(country)
    s = _sbs()
    out = pd.Series(np.nan, index=INDUSTRIES)
    if g is None or s.empty:
        return out
    s = s[(s.geo == g)]
    if s.empty:
        return out
    yrs = sorted(s.year.unique())
    y = min(yrs, key=lambda t: abs(t - year))
    s = s[s.year == y].set_index("nace_r2").v
    for j in INDUSTRIES:
        ds = FIG_DIV[j]
        if not ds:
            continue
        vals = [s.get(f"{j[0]}{d:02d}", np.nan) for d in ds]
        if all(np.isfinite(vals)):
            out[j] = float(np.sum(vals))
    return out


def split_table(countries=COUNTRIES, years=YEARS):
    rows = []
    for c in countries:
        for y in years:
            e, info = build(c, y)
            lg = info["labour_log"]
            sp = label_split(e.labels, lg["split_kids"])
            hp = np.array([any(m in lg["hpp_fallback_list"] for m in lab.split("+")) for lab in e.labels])
            ev = np.array([not any(m in ("T", "U") for m in lab.split("+")) for lab in e.labels])
            rows.append(dict(country=c, year=y, n_eval=int(ev.sum()), n_split=int((sp & ev).sum()),
                             n_hpp=int((hp & ev).sum()),
                             split_x_share=float(e.x[sp & ev].sum() / e.x[ev].sum()),
                             split_h_share=float(e.hours[sp & ev].sum() / e.hours[ev].sum()),
                             split_list=" ".join(np.array(e.labels)[sp & ev]),
                             hpp_list=" ".join(np.array(e.labels)[hp & ev])))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    t = split_table()
    t.to_csv(OUT / "a1_split_lists.csv", index=False)
    print(t.groupby("country")[["n_eval", "n_split", "n_hpp", "split_x_share", "split_h_share"]].mean().round(3))
