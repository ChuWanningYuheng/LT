"""Diagnostic (not pre-registered): how much of the hours -> wage-bill gap can any weight that varies only
across ISIC sections (the resolution of the ILO/TiMBC skill data) close?  Upper bound: the country's own
relative hourly labour income averaged (hours-weighted) within sections; and TiMBC aggregates."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..economy import build
from ..v2.placebo_cost import metrics_mat
from . import weights as W
from .placebo_disp import COUNTRIES, OUT, YEARS, Setup


def group_mean(w, h, keys):
    k = pd.Series(keys)
    num = pd.Series(w * h).groupby(k).transform("sum")
    den = pd.Series(h).groupby(k).transform("sum")
    return (num / den).values


def run():
    rows = []
    for c in COUNTRIES:
        for y in YEARS:
            e, _ = build(c, y)
            S = Setup(e)
            m = S.base_mask
            wr = W.wage_rel(e)
            sec = [W.label_sections(l)[0] if W.label_sections(l) else l for l in e.labels]
            vecs = {"hours": np.ones(e.n), "wagebill": wr, "wage_section_mean": group_mean(wr, e.hours, sec)}
            for name, w in vecs.items():
                ml = metrics_mat(S.z(e.l(w)[None])[:, m], S.x[m])
                for met in ("mawd", "d"):
                    rows.append(dict(country=c, year=y, variant=name, metric=met, value=float(ml[met][0])))
    pd.DataFrame(rows).to_csv(OUT / "section_bound.csv", index=False)


if __name__ == "__main__":
    run()
