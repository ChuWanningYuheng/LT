"""Iteration 13a: descriptive additions after the results (journal item 6, 'после результата').

* estimates (not only labels) of outcomes 110, 111, 115 in the lambda-before-1947 variants (stage 1 wrote labels only);
* Kliman (2009): his 'persistent fall' concerns the post-war years - author's series and our analogue on 1947-2007
  (the pre-registered window, from 1929, starts in the Depression).

Usage: PYTHONPATH=src python -P -m lts.v13a.extra
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..v6.series import ROOT
from . import est
from . import stage1 as s1
from . import stage2 as s2

OUT = ROOT / "results" / "v13a"


def run():
    V = pd.read_csv(OUT / "s1_lambda_variants.csv")
    rows = []
    pcons = s1.u.nipa_series().DPCERG
    for var, g in V.groupby("lambda_pre1947"):
        g = g.set_index("year").sort_index()
        for no, y, d_ in ((110, np.log(g.k), ">"), (111, np.log(g.r), "<"),
                          (115, np.log(s1.okishio(g, s1.PHASES, "hist", pcons)), "<")):
            r = est.within_trend(y, s1.PHASES, 0.002, d_)
            r.update(est.phase_slopes(y, s1.PHASES))
            rows.append(dict(item=f"{no}: lambda before 1947 = {var}", **r))
    ka, ko = s2.kliman_author(), s2.kliman_ours()
    c = s2.compare_series("Kliman (2009): PIH / HCFA, 1947-2007", ko, ka.PIH_HCFA, (1947, 2007))
    rows.append(dict(item=c["work"], est=c["trend_author"], ci90_lo=c["trend_author_lo"], ci90_hi=c["trend_author_hi"],
                     label_full=c["label_author"], ours=c["trend_ours"], ours_lo=c["trend_ours_lo"],
                     ours_hi=c["trend_ours_hi"], diff=c["diff"], diff_lo=c["diff_lo"], diff_hi=c["diff_hi"],
                     reproduced=c["reproduced"]))
    pd.DataFrame(rows).to_csv(OUT / "s1_extra_after_results.csv", index=False)
    print(pd.DataFrame(rows)[["item", "est", "ci90_lo", "ci90_hi", "label_full"]].round(5).to_string())


if __name__ == "__main__":
    run()
