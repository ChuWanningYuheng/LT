"""Stage P / 3.1 artefact check for the USA on an independent source: BEA IO tables (69 industries,
FTE-based hours), 2010-2022, same placebo families and evaluation as placebo_disp."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import bea
from ..codes import BEA_TO_NACE
from .placebo_disp import OUT, Draws, Setup, shares

EXCL_DIVS = set(range(5, 10)) | {61, 62, 63, 64, 65, 66, 68}


def run(years=range(2010, 2023)):
    rows, D = [], None
    for y in years:
        e, _ = bea.build(y)
        S = Setup(e)
        if D is None:
            D = Draws("USA_BEA", e.n, S.base_mask)
            labels0 = list(e.labels)
        assert list(e.labels) == labels0
        divs = {lab: set().union(*[set(BEA_TO_NACE[m]) for m in lab.split("+")]) for lab in e.labels}
        pre = S.base_mask & np.array([not (divs[l] and divs[l] <= EXCL_DIVS) for l in e.labels])
        for r in shares(S, D, e.l(), {"all": S.base_mask, "no_prereg": pre}):
            rows.append(dict(country="USA_BEA", year=y, **r))
        print("bea_check", y, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "placebo_disp_bea.csv", index=False)


if __name__ == "__main__":
    run()
