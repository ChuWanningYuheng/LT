"""Iteration 2, stages 2-3 analysis of results/v2/anchors_{figaro,bea}.parquet.

Outputs (results/v2/):
  pp_levels.csv      MAWD / d / cv_w of every anchor, averaged over years (subset all & core, capital T/F)
  reduction_gap.csv  G = (err_hours - err_red) / (err_hours - err_wagebill) for every reduction variant
  pp_dyn_T1.csv, pp_dyn_T4.csv, pp_dyn_T3.csv   dynamics, out-of-sample, pairwise encompassing weights
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..dynamics import encompassing, prepare, t1_constrained_r2, t4_out_of_sample
from ..levels import mask_for
from ..metrics import ratio_metrics
from .reduction_pp import CK_CENTRAL, ck_name

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "v2"
CKC = ck_name(CK_CENTRAL)
KEY = ["LV_hours", "LV_wagebill", "LV_edu_years", f"LV_{CKC}", "PPa", "PPb", "PPc_edu_years", f"PPc_{CKC}"]


def load():
    from ..bea import BEA_ROLE
    out = []
    for src in ("figaro", "bea"):
        d = pd.read_parquet(OUT / f"anchors_{src}.parquet")
        d["source"] = src
        out.append(d)
    return pd.concat(out, ignore_index=True), BEA_ROLE


def levels(df, role_bea):
    rows = []
    for (c, y, cap, b), g in df.groupby(["country", "year", "capital", "basis"], sort=False):
        rm = role_bea if c == "USA_BEA" else None
        for sub, excl in (("all", set()), ("core", {"B", "K64", "K65", "K66", "L", "O84", "P85", "Q86", "Q87_88"})):
            m = ratio_metrics(g.z.to_numpy(), g.x.to_numpy(), mask_for(g.industry.tolist(), excl, rm))
            rows.append(dict(country=c, year=y, capital=cap, basis=b, subset=sub, mawd=m["mawd"], d=m["d"], cv_w=m["cv_w"]))
    lv = pd.DataFrame(rows)
    avg = lv.groupby(["country", "capital", "subset", "basis"])[["mawd", "d", "cv_w"]].mean().reset_index()
    avg.to_csv(OUT / "pp_levels.csv", index=False)
    # reduction gap
    g = []
    for (c, cap, sub), a in avg.groupby(["country", "capital", "subset"]):
        a = a.set_index("basis")
        for met in ("mawd", "d"):
            h, w = a.loc["LV_hours", met], a.loc["LV_wagebill", met]
            for b in a.index:
                if b.startswith("LV_") and b not in ("LV_hours", "LV_wagebill"):
                    g.append(dict(country=c, capital=cap, subset=sub, metric=met, variant=b[3:], err_hours=h,
                                  err_wagebill=w, err_variant=a.loc[b, met], G=(h - a.loc[b, met]) / (h - w)))
    pd.DataFrame(g).to_csv(OUT / "reduction_gap.csv", index=False)
    return avg


def dynamics(df, role_bea):
    t1, t4, t3 = [], [], []
    for src, splits in (("figaro", [(2016, 2017, 2023)]), ("bea", [(2008, 2009, 2014), (2008, 2009, 2023)])):
        d0 = df[(df.source == src) & (df.capital == True) & df.basis.isin(KEY)]
        rm = role_bea if src == "bea" else None
        for sub in ("all", "core"):
            d = prepare(d0, sub, rm)
            t1.append(t1_constrained_r2(d, horizons=(1, 5)).assign(subset=sub))
            t4.append(t4_out_of_sample(d, splits).assign(subset=sub))
            for c, gc in d.groupby("country"):
                for lv, pp in [("LV_hours", "PPa"), ("LV_hours", "PPb"), ("LV_hours", "PPc_edu_years"),
                               ("LV_hours", f"PPc_{CKC}"), ("LV_edu_years", "PPc_edu_years")]:
                    for col, on in (("d_lz_rel", "changes"), ("lz_rel", "levels")):
                        r = encompassing(gc[gc.basis.isin([lv, pp])], [lv, pp], col)
                        if r:
                            t3.append(dict(country=c, subset=sub, lv=lv, pp=pp, on=on, w_lv=r[f"w_{lv}"], n=r["n"]))
    pd.concat(t1).to_csv(OUT / "pp_dyn_T1.csv", index=False)
    pd.concat(t4).to_csv(OUT / "pp_dyn_T4.csv", index=False)
    pd.DataFrame(t3).to_csv(OUT / "pp_dyn_T3.csv", index=False)


def run():
    df, role = load()
    levels(df, role)
    dynamics(df, role)


if __name__ == "__main__":
    run()
