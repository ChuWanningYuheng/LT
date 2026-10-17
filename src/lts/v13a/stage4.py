"""Iteration 13a, stage 4 (pre_registration_v13a.md 'Этап 4'): the long series of Dumenil-Levy and the BEA
insurance error.

(a) Dumenil-Levy 1869-2014 (uslt4x, r = (NDP - W L) / KN), descriptive: phases and windows of the literature fixed
    before the data; common slope of ln r within phases (CI as in stage 1, no label); recoveries d ln r between the
    phases 1889->1898, 1929->1946, 1973->1983. Before 1889 only ten-year averages are known for some variables
    (Dumenil & Levy 2016 note) - written next to the numbers.
(b) BEA insurance error (12b, section 1; review R8): lts.bea with K_SECTION_SELF_RATIO = True (self-employed /
    employees ratio of section K for divisions 64-66, not of K65); outcomes 7 (iteration 4, B0) and 18 (iteration 7,
    part 2, test 3) recomputed with the same code; table 'before - after - does the label change'.

Usage: PYTHONPATH=src python -P -m lts.v13a.stage4 [dl|bea|all]
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from .. import bea
from ..v6.series import ROOT
from ..v8.rule import share_row
from ..v8.stage1 import no_ci_row
from . import est
from . import stage0 as s0

OUT = ROOT / "results" / "v13a"
DL_PHASES = {"1869-1889": (1869, 1889), "1898-1913": (1898, 1913), "1920-1929": (1920, 1929),
             "1946-1973": (1946, 1973), "1983-2007": (1983, 2007)}
DL_WINDOWS = [(1890, 1897), (1914, 1919), (1930, 1945), (1974, 1982), (2008, 2014)]
DL_RECOVERIES = [(1889, 1898), (1929, 1946), (1973, 1983)]


def dl():
    d = s0.uslt4x()
    r = d["R"].where(d["R"] > 0).dropna()
    r = r.loc[1869:2014]
    lr = np.log(r)
    rows = []
    o = est.within_trend(lr, DL_PHASES, 0.002, "<")
    o.update(est.phase_slopes(lr, DL_PHASES))
    o["label"] = o["label_full"] = "описательно (без метки)"
    rows.append(dict(item="Duménil–Lévy: common slope of ln r within the phases of the literature", **o))
    o2 = est.within_trend(lr.loc[1889:], {k: v for k, v in DL_PHASES.items() if v[0] >= 1889}, 0.002, "<")
    o2["label"] = o2["label_full"] = "описательно (без метки)"
    rows.append(dict(item="same, without 1869-1889 (ten-year averages before 1889)", **o2))
    for a, b in DL_RECOVERIES:
        rows.append(dict(item=f"recovery d ln r {a}->{b}", est=float(lr.loc[b] - lr.loc[a])))
    for a, b in DL_WINDOWS:
        w = lr.loc[a:b]
        rows.append(dict(item=f"window {a}-{b}: d ln r from the year before to the year after",
                         est=float(lr.loc[min(b + 1, lr.index.max())] - lr.loc[a - 1])))
    pd.DataFrame(rows).to_csv(OUT / "s4_dumenil_levy.csv", index=False)
    print(pd.DataFrame(rows)[["item", "est"]].round(4).to_string())


def outcome7(t1):
    """As iteration 4 (lts.v4.b_summary.b0) and iteration 8 (outcome 7): share of country tables where labour is
    informative against the physical flat benchmark, subset 'core', h = 5."""
    rows = []
    for (c, s, h), g in t1.groupby(["country", "subset", "h"]):
        L = g[g.basis == "labour"].r2_w.iloc[0]
        P = g[g.basis.str.startswith("perm")].r2_w
        rows.append(dict(country=c, subset=s, h=int(h), labour=L, flat_phys=g[g.basis == "flat_phys"].r2_w.iloc[0],
                         perm_share_ge_labour=(P >= L).mean()))
    r = pd.DataFrame(rows)
    r["informative_vs_phys"] = (r.perm_share_ge_labour <= 0.05) & (r.labour > r.flat_phys)
    s = r.query("subset=='core' and h==5")
    return share_row("Итерация 4, Б0: труд информативен против физически плоского эталона, h = 5",
                     s.informative_vs_phys.astype(bool), 0.5, 0.15, ">"), r


def bea_fix():
    from ..v4 import b0
    from ..v7 import extra
    old_t1 = pd.read_csv(ROOT / "results" / "v4" / "b0_t1.csv")
    old7, old_summary = outcome7(old_t1)
    bea.K_SECTION_SELF_RATIO = True
    t_bea = b0.run_country("USA_BEA", "bea")
    new_t1 = pd.concat([old_t1[old_t1.country != "USA_BEA"], t_bea], ignore_index=True)
    new_t1.to_csv(OUT / "s4_b0_t1_fixed.csv", index=False)
    new7, new_summary = outcome7(new_t1)
    old_pw = pd.read_csv(ROOT / "results" / "v7" / "p2_power.csv")
    new_pw = extra.power(ROOT / "results" / "v7" / "p2_occupations.csv")
    bea.K_SECTION_SELF_RATIO = False
    new_pw.to_csv(OUT / "s4_p2_power_fixed.csv", index=False)
    def o18(pw):
        p = pw.query("vector=='psi Shaikh'")
        return no_ci_row("Итерация 7, ч. 2, тест 3: выигрыш MAWD ψ Шейха против c2 (среднее по BEA и FIGARO)",
                         p.gain_vs_c2_mawd.mean(), 0, 0.005, ">")
    old18, new18 = o18(old_pw), o18(new_pw)
    rows = []
    for no, o, n_ in ((7, old7, new7), (18, old18, new18)):
        rows.append(dict(outcome=no, name=o["outcome"], est_before=o["est"], ci_before=f"{o.get('ci90_lo', np.nan)}; "
                         f"{o.get('ci90_hi', np.nan)}", label_before=o["label"], est_after=n_["est"],
                         ci_after=f"{n_.get('ci90_lo', np.nan)}; {n_.get('ci90_hi', np.nan)}", label_after=n_["label"],
                         label_changes=o["label"] != n_["label"]))
    bea_rows = new_summary[new_summary.country == "USA_BEA"].assign(version="after").to_dict("records") + \
        old_summary[old_summary.country == "USA_BEA"].assign(version="before").to_dict("records")
    pd.DataFrame(bea_rows).to_csv(OUT / "s4_b0_usa_bea.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "s4_insurance_fix.csv", index=False)
    print(pd.DataFrame(rows).to_string())


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else "all"
    if part in ("dl", "all"):
        dl()
    if part in ("bea", "all"):
        bea_fix()
