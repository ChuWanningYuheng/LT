"""Iteration 13a: descriptive additions after the results (journal items 6 and 7, 'после результата').

Item 6:
* estimates (not only labels) of outcomes 110, 111, 115 in the lambda-before-1947 variants (stage 1 wrote labels only);
* Kliman (2009): his 'persistent fall' concerns the post-war years - author's series and our analogue on 1947-2007
  (the pre-registered window, from 1929, starts in the Depression).
Item 7 (post-calculation review):
* R3: 117, 125, 126, 127 in the lambda-before-1947 variants; readings 110 x 111 and 125 x 111 by variant;
* R2: CES variant 'splice 1998' (wages 1948-1997 spliced to NIPA 6.3D); R5: sW capped at 1; managers in V and L;
  omega* and the PNS share of productive compensation in A' = 1964-1973.

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
        lab = {}
        for no, y, d_, dl in ((110, np.log(g.k), ">", 0.002), (111, np.log(g.r), "<", 0.002),
                              (115, np.log(s1.okishio(g, s1.PHASES, "hist", pcons)), "<", 0.002),
                              (117, np.log(g.r / g.r_P), "<", 0.001), (125, np.log(g.CL_P), ">", 0.002)):
            r = est.within_trend(y, s1.PHASES, dl, d_)
            r.update(est.phase_slopes(y, s1.PHASES))
            rows.append(dict(item=f"{no}: lambda before 1947 = {var}", **r))
            lab[no] = r["label_full"]
        for no, y, d_ in ((126, np.log(g.r), ">"), (127, np.log(g.CL_P), "<")):
            r = est.two_phase_diff(y, s1.PHASES["A"], s1.PHASES["B"], 0.002, d_)
            rows.append(dict(item=f"{no}: lambda before 1947 = {var}", **r))
        rows.append(dict(item=f"reading 110 x 111: lambda before 1947 = {var} (main labels of the variant)",
                         label_full=s1.reading_110_111(lab[110], lab[111])))
        rows.append(dict(item=f"reading 125 x 111: lambda before 1947 = {var} (main labels of the variant)",
                         label_full=s1.reading_110_111(lab[125], lab[111])))
    # R2, R5: variants of V
    D, A1, PB = 0.002, s1.A_PRIME, s1.PHASES["B"]
    for vn, kw in (("splice 1998", dict(ces_variant="splice 1998")), ("sW capped at 1", dict(ces_variant="sW capped at 1")),
                   ("managers in V and L (who=all)", dict(cfg_key=s1.config_key({**s1.MAIN, "who": "all"})))):
        m = s1.main_series(**kw)
        for no, y, d_ in ((110, np.log(m.k), ">"), (111, np.log(m.r), "<"), ("e", np.log(m.e), ">")):
            r = est.within_trend(y, s1.PHASES, D, d_)
            r.update(est.phase_slopes(y, s1.PHASES))
            rows.append(dict(item=f"{no}: V variant = {vn}", **r))
        for no, y, a_, d_ in ((116, np.log(m.e), A1, ">"), (126, np.log(m.r), s1.PHASES["A"], ">")):
            r = est.two_phase_diff(y, a_, PB, D, d_)
            rows.append(dict(item=f"{no}: V variant = {vn}", **r))
    # R5: omega* and the PNS share of productive compensation in A'
    ms = s1.main_series()
    sh = {}
    for y_ in s1.YEARS:
        _, sW = s1.labour(y_, "hours", "pns")
        comp = s1.u.compensation(y_).to_numpy(float)
        sh[y_] = float((comp * sW * s1.PROD).sum() / (comp * s1.PROD).sum())
    sh = pd.Series(sh)
    for nm, y in (("ln omega*", np.log(ms.omega)), ("ln PNS share of productive compensation", np.log(sh))):
        yy = y.loc[A1[0]:A1[1]]
        rows.append(dict(item=f"A' 1964-1973: OLS slope of {nm}", **est.single_trend(yy, D, "<")))
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
