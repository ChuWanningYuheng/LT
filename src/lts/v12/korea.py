"""Iteration 12: check of Korea in the FIGARO replication of iteration 11 (pre_registration_v12.md, «Проверка Кореи»).

(a) Korea rebuilt with the donor-median employment split (iteration 4, A1(b), lts.v4.splits.donor_splitter);
(b) Korea without its split industries (lts.v4.splits.label_split, union over years);
(c) without Korea.
Signals as in lts.v11.review.figaro_signals (current economies; 50 permutations fixed per country).

Usage: PYTHONPATH=src python -P -m lts.v12.korea
Outputs: results/v12/k_*.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..economy import build
from ..levels import labour_vectors
from ..v4.splits import donor_splitter, label_split
from ..v6.series import ROOT
from ..v11 import review as rv
from ..v11 import stage1 as s1

OUT = ROOT / "results" / "v12"
OUT11 = ROOT / "results" / "v11"
YEARS = range(2010, 2023)


def kor_rows(splitter=False):
    rows, prow = [], []
    perms = None
    for y in YEARS:
        sp = donor_splitter("KOR", y) if splitter else None
        e, info = build("KOR", y, persons_splitter=sp) if splitter else build("KOR", y)
        if perms is None:
            rng = np.random.default_rng(11 + sum(map(ord, "KOR")))
            perms = np.vstack([rng.permutation(e.n) for _ in range(50)])
        lv = labour_vectors(e, "KOR", y)
        S = s1.Setup(e)
        r = e.actual_profit_rate(True)
        pp = e.prices_of_production(r, True, "price", "actual")
        z = {k: S.z(lv[k][None])[0] for k in ("hours", "edu_years", "edu_train")}
        zP = S.z(lv["hours"][perms])
        rows.append(pd.DataFrame(dict(country="KOR", year=y, ind=e.labels, x=e.x, hours=e.hours, persons=e.meta["persons"],
                                      z_hours=z["hours"], z_edu_years=z["edu_years"], z_edu_train=z["edu_train"],
                                      pp_actual_wage=pp)))
        prow.append(pd.DataFrame(zP.T, columns=[f"placebo_perm_{k:03d}" for k in range(50)]))
    return pd.concat([pd.concat(rows, ignore_index=True), pd.concat(prow, ignore_index=True)], axis=1)


def split_info():
    out, union = [], set()
    for y in YEARS:
        e, info = build("KOR", y)
        kids = info["labour_log"]["split_kids"]
        labs = np.array(e.labels)[label_split(e.labels, kids)]
        union |= set(labs)
        out.append(dict(year=y, split_kids=";".join(f"{k}<-{v}" for k, v in kids.items()), split_labels=";".join(labs)))
    return pd.DataFrame(out), sorted(union)


def summarize(F, name):
    sv, spp = "Sz_hours", "Spp_actual_wage"
    r = s1.labelled(s1.dd(F, "R_hours", sv=sv, spp=spp, B=rv.B_RV), 0.02, ">")
    out = [dict(item=f"{name}: DD", **r)]
    for k, v in rv.group_coefs(F, sv=sv, spp=spp).items():
        out.append(dict(item=f"{name}: DD model {k}", **v))
    pc = [c for c in F.columns if c.startswith("Splacebo_perm_")]
    fp = np.array([s1.dd(F, "R_hours", sv=c, spp=spp, boot=False)["est"] for c in pc])
    out.append(dict(item=f"{name}: placebo DD mean; share >= DD", est=float(fp.mean()), share_ge=float((fp >= r["est"]).mean())))
    g = s1.groups(F)
    out.append(dict(item=f"{name}: Korean self-employed cells", est=int(((g == "se") & (F.country == "KOR")).sum())))
    return out


def run():
    sig = pd.read_parquet(OUT11 / "s1_signals.parquet")
    W = pd.read_parquet(OUT11 / "rv_figaro_signals.parquet")
    info, union = split_info()
    info.to_csv(OUT / "k_split_info.csv", index=False)
    print(info.to_string(), "\nunion:", union, flush=True)
    rows = []
    F = rv.figaro_panel_new(W, sig)
    rows += summarize(F, "baseline (iteration 11 review, recomputed signals)")
    Wa = pd.concat([W[W.country != "KOR"], kor_rows(splitter=True)], ignore_index=True)
    rows += summarize(rv.figaro_panel_new(Wa, sig), "(a) Korea with donor-median split")
    Fb = F[~((F.country == "KOR") & F.ind.isin(union))]
    rows += summarize(Fb, "(b) Korea without split industries")
    rows += summarize(F[F.country != "KOR"], "(c) without Korea")
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "k_results.csv", index=False)
    pd.set_option("display.width", 250)
    print(R[[c for c in ("item", "est", "ci90_lo", "ci90_hi", "n", "G", "label", "share_ge") if c in R]].round(4).to_string())


if __name__ == "__main__":
    run()
