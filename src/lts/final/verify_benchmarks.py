"""Verification file for numbers already reported in v3 (review remarks 1-2) and v4: metrics of the
flat vector (equal direct input per unit of output) and of powers of hours, all 13 countries, 2010-2022,
v3 specification.  Output results/final/benchmarks_flat_power.csv."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..economy import build
from ..v2.placebo_cost import metrics_mat
from ..v3.placebo_disp import COUNTRIES, PREREG_EXCL, YEARS, Setup
from ..v3.weights import wage_rel


def run():
    rows = []
    for c in COUNTRIES:
        for y in YEARS:
            e, _ = build(c, y)
            S = Setup(e)
            h = e.l()
            vecs = {"hours": h, "flat": np.where(e.x > 0, 1.0, 0.0), "wagebill": e.l(wage_rel(e)),
                    "hours^0.5": np.where(h > 0, h ** 0.5, 0), "hours^0.75": np.where(h > 0, h ** 0.75, 0)}
            for mname, m in (("all", S.base_mask), ("no_prereg", S.mask(PREREG_EXCL))):
                for k, v in vecs.items():
                    ml = metrics_mat(S.z(v[None])[:, m], S.x[m])
                    rows.append(dict(country=c, year=y, mask=mname, vector=k,
                                     mawd=float(ml["mawd"][0]), d=float(ml["d"][0])))
        print("verify", c, flush=True)
    pd.DataFrame(rows).to_csv("results/final/benchmarks_flat_power.csv", index=False)


if __name__ == "__main__":
    run()
