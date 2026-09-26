"""Iteration 4, A1: stage P of v3 re-run (a) without industries with split employment (and without
split + national-average hours per person), (b) with employment split by donor-median structure.
Only countries with split industries change (JPN, KOR, USA); the other ten are identical by
construction (no split industries, verified in a1_split_lists.csv)."""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from ..economy import build
from ..v3.placebo_disp import YEARS, Draws, Setup, shares
from ..v3.reduction import commodity_Z
from ..v2.placebo_cost import metrics_mat
from .splits import OUT, donor_splitter, label_split

AFFECTED = ("JPN", "KOR", "USA")


def pct_bases(e, S, mask, l):
    ZB = commodity_Z(e, S)
    mB = metrics_mat(ZB[:, mask], S.x[mask])
    ml = metrics_mat(S.z(l[None])[:, mask], S.x[mask])
    return {m: float((mB[m] < ml[m][0]).mean()) for m in ("mawd", "d")}, {m: float(ml[m][0]) for m in ("mawd", "d")}


def run(countries=AFFECTED, years=YEARS):
    rows = []
    for c in countries:
        # union over years of split / hpp-fallback labels (constant exclusion mask for the country)
        split_u, hpp_u = set(), set()
        for y in years:
            e, info = build(c, y)
            lg = info["labour_log"]
            split_u |= set(np.array(e.labels)[label_split(e.labels, lg["split_kids"])])
            hpp_u |= {lab for lab in e.labels if any(m in lg["hpp_fallback_list"] for m in lab.split("+"))}
        Dv3 = Da = Dah = Db = None
        for y in years:
            # ---------------- default build (v3) and exclusion variants
            e, _ = build(c, y)
            S = Setup(e)
            base = S.base_mask
            ma = base & ~np.isin(e.labels, list(split_u))
            mah = ma & ~np.isin(e.labels, list(hpp_u))
            if Dv3 is None:
                Dv3 = Draws(c, e.n, base)
                Da = Draws(c, e.n, ma)
                Dah = Draws(c, e.n, mah)
            l = e.l()
            for name, m, Dp in (("v3", base, Dv3), ("A1a_nosplit", ma, Da), ("A1a_nosplit_nohpp", mah, Dah)):
                if m.sum() < 8:
                    continue
                for r in shares(S, Dp, l, {"m": m}, kinds=("perm",)):
                    rows.append(dict(country=c, year=y, variant=name, n_eval=int(m.sum()), **r))
                for r in shares(S, Dv3, l, {"m": m}, kinds=("lnorm", "mix")):
                    rows.append(dict(country=c, year=y, variant=name, n_eval=int(m.sum()), **r))
                pb, err = pct_bases(e, S, m, l)
                for met in ("mawd", "d"):
                    rows.append(dict(country=c, year=y, variant=name, n_eval=int(m.sum()), kind="bases",
                                     metric=met, share_better=pb[met], labour=err[met]))
            # ---------------- donor-median split
            sp = donor_splitter(c, y)
            e2, info2 = build(c, y, persons_splitter=sp)
            S2 = Setup(e2)
            if Db is None:
                Db = Draws(c, e2.n, S2.base_mask)
            for r in shares(S2, Db, e2.l(), {"m": S2.base_mask}):
                rows.append(dict(country=c, year=y, variant="A1b_donor", n_eval=int(S2.base_mask.sum()),
                                 fallback=len(info2["labour_log"].get("split_fallback", [])), **r))
            pb, err = pct_bases(e2, S2, S2.base_mask, e2.l())
            for met in ("mawd", "d"):
                rows.append(dict(country=c, year=y, variant="A1b_donor", n_eval=int(S2.base_mask.sum()), kind="bases",
                                 metric=met, share_better=pb[met], labour=err[met],
                                 fallback=len(info2["labour_log"].get("split_fallback", []))))
            print("a1", c, y, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "a1_placebo.csv", index=False)


if __name__ == "__main__":
    run(sys.argv[1].split(",") if len(sys.argv) > 1 else AFFECTED)
