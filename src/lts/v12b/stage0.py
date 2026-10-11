"""Iteration 12b, stage 0: China hours in the WIOD world system (pre_registration_v12b.md, stage 0).

WIOD SEA has no employee hours for China, and lts.v9.stage34.year_system gives China zero hours (rest-of-world
averages are computed without China). Variants of the imputation (hours in millions):
  (a) H = EMP (thousands) x PWT 11 avh / 1000, same in every industry (main);
  (b) H = EMP x median H/EMP of the same industry-year over WIOD countries with data.
H_x of China and the rest-of-world per-dollar hours (now including China) are recomputed; productivity weights too.
The code of iterations 9, 9b, 10 is rerun unchanged with this year_system and outputs redirected to
results/v12b/s0_<variant>/{v9,v9b,v10}: outcomes 28-33 (+ T_c transfers), 44-46, 54, 55.

Usage: PYTHONPATH=src python -P -m lts.v12b.stage0 [a|b] [v9|v9b|v10_54|v10_55|compare ...]
"""
from __future__ import annotations

import shutil
import sys

import numpy as np
import pandas as pd

from ..v6.series import ROOT
from ..v9 import stage34 as s34

OUT = ROOT / "results" / "v12b"
ORIG_YEAR_SYSTEM = s34.year_system
_AVH = None


def avh_china(y):
    global _AVH
    if _AVH is None:
        p = pd.read_excel(ROOT / "data" / "raw" / "v9" / "pwt110.xlsx", sheet_name="Data",
                          usecols=["countrycode", "year", "avh"])
        _AVH = p[p.countrycode == "CHN"].set_index("year").avh
    return float(_AVH.loc[y])


def patch(cells, y, variant):
    cells = cells.copy()
    m = (cells.country == "CHN") & (cells.EMP > 0)
    if variant == "a":
        cells.loc[m, "H"] = cells.loc[m, "EMP"] * avh_china(y) / 1000
    else:
        ok = ~cells.country.isin(["CHN", "ROW"]) & (cells.EMP > 0) & (cells.H > 0)
        med = (cells.H[ok] / cells.EMP[ok]).groupby(cells.code[ok]).median()
        cells.loc[m, "H"] = cells.loc[m, "EMP"] * cells.loc[m, "code"].map(med).fillna(float(med.median()))
    n = len(cells)
    rate = cells[cells.country != "ROW"].groupby("code").apply(
        lambda g: np.nansum(g["H"]) / np.nansum(g["GO$"].where(g["H"].notna())), include_groups=False)
    hx = np.where(cells.country == "ROW", cells.code.map(rate),
                  np.divide(cells.H, cells["GO$"], out=np.full(n, np.nan), where=cells["GO$"] > 0))
    cells["H_x"] = pd.Series(hx, index=cells.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    for v in ("pl1", "pl2", "pl3"):
        cells["w_" + v] = s34.adj_weights(cells, v)
    return cells


def make_year_system(variant):
    def year_system(y):
        A, cells = ORIG_YEAR_SYSTEM(y)
        return A, patch(cells, y, variant)
    return year_system


def setup(variant):
    from ..v9b import stage23
    from ..v10 import stage3 as v10s3, stage4 as v10s4
    ys = make_year_system(variant)
    s34.year_system = ys
    stage23.year_system = ys
    base = OUT / f"s0_{variant}"
    dirs = {k: base / k for k in ("v9", "v9b", "v10")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    s34.OUT = dirs["v9"]
    stage23.OUT = dirs["v9b"]
    v10s3.OUT = dirs["v10"]
    v10s4.OUT = dirs["v10"]
    return dirs


def run_v9():
    s34.run()
    s34.outcomes()


def run_v9b():
    from ..v9b import stage23
    N, store, ppp = stage23.run_years()
    o44 = stage23.outcome44(N)
    o45, d45 = stage23.outcome45(store)
    o46, l46 = stage23.outcome46(ppp)
    O = pd.concat([o44, pd.DataFrame([o45, o46])], ignore_index=True)
    O.to_csv(stage23.OUT / "s2_outcomes.csv", index=False)
    d45.to_csv(stage23.OUT / "s2_changes_world.csv", index=False)


def run_v10_54():
    from ..v10 import stage3 as v10s3
    v10s3.run_placebo()


def run_v10_55(dirs):
    from ..v10 import stage4 as v10s4
    for f in ("s4_flags_all.csv.gz", "s4_o1_country_years.csv", "s4_o3_shares.csv", "s4_o15_shares.csv"):
        shutil.copy(ROOT / "results" / "v10" / f, dirs["v10"] / f)
    D = v10s4.load_flags()
    v10s4.run_wiod28(D)
    v10s4.summary()


def compare():
    T = pd.read_csv(ROOT / "results" / "v12" / "outcomes_table.csv", dtype={"n": str}).set_index("n")
    rows = []
    for variant in ("a", "b"):
        base = OUT / f"s0_{variant}"
        new = {}
        f = base / "v9" / "s34_outcomes.csv"
        if f.exists():
            s = pd.read_csv(f).set_index("outcome")
            new["28"] = s.loc["28: countries where world-average labour beats national (majority of years)"]
            new["29"] = s.loc["29 (2014)"]
            new["30"] = s.loc["30 b-c (2005)"]
            new["31"] = s.loc["31 b-min(cap,energy) (2005)"]
            new["32"] = s.loc["32 a-b (2005)"]
            new["33"] = s.loc["33: R2(b) - R2(c), 5-year changes, mean of windows"]
        f = base / "v9b" / "s2_outcomes.csv"
        if f.exists():
            s = pd.read_csv(f)
            new["44"] = s[s.outcome.str.startswith("44: share(hours) - share(lab)")].iloc[0]
            new["45"] = s[s.outcome.str.startswith("45:")].iloc[0]
            new["46"] = s[s.outcome.str.startswith("46:")].iloc[0]
        f = base / "v10" / "s3_outcomes.csv"
        if f.exists():
            new["54"] = pd.read_csv(f).iloc[0]
        f = base / "v10" / "s4_outcomes.csv"
        if f.exists():
            x = pd.read_csv(f).iloc[0]
            new["55"] = pd.Series(dict(est=x.est, ci90_lo=np.nan, ci90_hi=np.nan, label=x.label, changed=x.get("changed")))
        for n, x in new.items():
            o = T.loc[n]
            rows.append(dict(variant=variant, n=n, est_old=o.est, lo_old=o.lo, hi_old=o.hi, label_old=o.label,
                             est_new=x.est, lo_new=x.get("ci90_lo", np.nan), hi_new=x.get("ci90_hi", np.nan),
                             label_new=x.label, label_changed=str(x.label) != str(o.label),
                             extra=str(x.get("changed", "")) if n == "55" else ""))
    C = pd.DataFrame(rows)
    C.to_csv(OUT / "s0_compare.csv", index=False)
    # T_c transfers: old vs new
    tr = []
    for variant in ("a", "b"):
        f = OUT / f"s0_{variant}" / "v9" / "s4_transfers.csv"
        if f.exists():
            tr.append(pd.read_csv(f).assign(variant=variant))
    if tr:
        old = pd.read_csv(ROOT / "results" / "v9" / "s4_transfers.csv").assign(variant="old (China hours = 0)")
        pd.concat([old] + tr, ignore_index=True).to_csv(OUT / "s0_transfers_compare.csv", index=False)
    pd.set_option("display.width", 250)
    print(C.round(4).to_string())
    return C


if __name__ == "__main__":
    args = sys.argv[1:]
    variant = args[0] if args and args[0] in ("a", "b") else "a"
    parts = [a for a in args if a not in ("a", "b")] or ["v9", "v9b", "v10_54", "v10_55", "compare"]
    if parts != ["compare"]:
        dirs = setup(variant)
    if "v9" in parts:
        run_v9()
    if "v9b" in parts:
        run_v9b()
    if "v10_54" in parts:
        run_v10_54()
    if "v10_55" in parts:
        run_v10_55(dirs)
    if "compare" in parts:
        compare()
