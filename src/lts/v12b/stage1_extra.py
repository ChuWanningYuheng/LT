"""Iteration 12b, stage 1: descriptive items not computed in the first run (pre_registration_v12b.md, journal 10).

"New interpretation" variant V_NI = L x W_prod / net product; pooled within-phase slopes (phase effects, clusters =
phases); trends of the money rates of iterations 6 and 9b on the same years.

Usage: PYTHONPATH=src python -P -m lts.v12b.stage1_extra
Outputs: results/v12b/s1_extra.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..v6.series import ROOT
from ..v8.rule import decide
from ..v11.stage1 import reg
from .stage1 import US_PEAKS, trend

OUT = ROOT / "results" / "v12b"


def run():
    S = pd.read_csv(OUT / "s1_us_series.csv")
    m = S[S.variant == "main"].set_index("year").sort_index()
    rows = []
    V = m.L * m.W_prod / m.gdp
    e_ni, r_ni, k_ni = (m.L - V) / V, (m.L - V) / (m.C_h + V), m.C_h / V
    for nm, s, d_ in (("NI: ln r", r_ni, "<"), ("NI: ln e", e_ni, ">"), ("NI: ln k", k_ni, ">"),
                      ("NI: ln(r_NI / r_h)", r_ni / m.r_h, "<"), ("NI: ln(r_NI / r_$)", r_ni / m["r_$"], "<")):
        rows.append(dict(item=nm, **trend(np.log(s), 4, 0.002, d_)))
    lv = pd.DataFrame({"r_NI": r_ni, "e_NI": e_ni, "k_NI": k_ni, "V_NI/V_h": V / m.V_h})
    # phases
    bounds = [1948] + US_PEAKS + [2023]
    ph = pd.Series(index=m.index, dtype=object)
    for a, b in zip(bounds[:-1], bounds[1:]):
        ph.loc[(m.index >= a) & (m.index <= b) & ph.isna()] = f"{a}-{b}"
    d = pd.DataFrame({"t": m.index.astype(float), "phase": ph})
    for k_, dr in (("r_h", "<"), ("k_h", ">"), ("e_h", ">"), ("r_$", "<")):
        d["y"] = np.log(m[k_])
        r = reg(d, "y", ["t"], 0, fe=("phase",), cl="phase")
        rows.append(dict(item=f"phases: pooled within-phase slope of ln {k_} (phase effects, clusters = phases)", **r,
                         theta0=0.0, delta=0.002, direction=dr,
                         label=decide(r["est"], r["ci90_lo"], r["ci90_hi"], 0, 0.002, dr)))
    c = pd.read_csv(OUT / "s1_compare_money.csv").set_index("year")
    for col in ("r_NFC (iter 6)", "r_m (9b)", "r_h", "r_$"):
        s = c[col].dropna()
        rows.append(dict(item=f"trend ln {col}, same years {int(s.index.min())}-{int(s.index.max())}",
                         **trend(np.log(s), 4, 0.002, "<")))
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "s1_extra.csv", index=False)
    lv.to_csv(OUT / "s1_extra_ni_levels.csv")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 90)
    print(D[["item", "est", "ci90_lo", "ci90_hi", "label"]].round(5).to_string())
    print(lv.loc[[1948, 1973, 1997, 2023]].round(4).to_string())


if __name__ == "__main__":
    run()
