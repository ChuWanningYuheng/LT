"""Iteration 11: descriptive checks after the results (pre_registration_v11.md, journal 3). Labels do not change.

(a) outcome 72: abnormal pre-peak change of ln F split into ln MV, ln Aaa and -ln NOS (each with its own mean drift);
(b) stage 1: WIOD main DD restricted to the 13 FIGARO countries, to signal years 2008-2010, and both;
    FIGARO group components (b_v, b_pp by group);
(c) outcome 69: cointegration among countries where ADF does not reject a unit root in both r and i_real.

Usage: PYTHONPATH=src python -P -m lts.v11.checks
Outputs: results/v11/ck_*.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..v8.stage4 import US_PEAKS
from . import stage1 as s1
from . import stage2 as s2

OUT = s1.OUT


def f_decomposition():
    Z = s2.z1()
    aaa = s2.fred("AAA") / 100
    comp = {"ln MV": Z.MV, "ln Aaa": aaa.reindex(Z.index), "ln NOS (sign flipped)": 1 / Z.NOS,
            "ln F (total)": Z.MV * aaa.reindex(Z.index) / Z.NOS}
    rows = []
    for name, x in comp.items():
        x = x.loc[1945:2024].dropna()
        gbar = float(np.log(x).diff().loc[1946:2024].mean())
        pre = [s2.abnormal(x, P, gbar)[0] for P in US_PEAKS]
        r = s2.mean_ci(pre, US_PEAKS, 0.05, ">")
        rows.append(dict(component=name, **r, by_peak=";".join(f"{P}:{v:.3f}" for P, v in zip(US_PEAKS, pre))))
    return pd.DataFrame(rows)


def stage1_subsets():
    sig = pd.read_parquet(OUT / "s1_signals.parquet")
    d, _ = s1.prepare(sig)
    P3 = s1.build_panel(d, 3, s_years=range(2001, 2011))
    rows = []
    for name, m in (("13 FIGARO countries", P3.country.isin(s1.FIG_COUNTRIES)),
                    ("signal years 2008-2010", P3.year.between(2008, 2010)),
                    ("13 FIGARO countries, 2008-2010", P3.country.isin(s1.FIG_COUNTRIES) & P3.year.between(2008, 2010)),
                    ("other 28 countries", ~P3.country.isin(s1.FIG_COUNTRIES))):
        p = P3[m]
        a = s1.labelled(s1.dd(p, "R_hours"), 0.02, ">")
        b = s1.labelled(s1.dd(p, "R_hours", which="corp"), 0.02, ">")
        rows.append(dict(item=f"WIOD {name}: DD", **a))
        rows.append(dict(item=f"WIOD {name}: corporate b_pp - b_v", **b))
        print(name, round(a["est"], 4), a["label"], flush=True)
    F = s1.figaro_panel(sig)
    g = s1.groups(F)
    for grp in ("se", "corp"):
        sub = F[g == grp]
        for k, v in s1.coefs(sub, "R_hours", ["Sz_hours", "Spp_actual_wage"]).items():
            rows.append(dict(item=f"FIGARO group {grp}: b_{k} (joint)", **v))
    rows.append(dict(item="FIGARO cells by group", est=np.nan, n_se=int((g == "se").sum()), n_corp=int((g == "corp").sum()),
                     n_mid=int((g == "mid").sum()), n_na=int((g == "na").sum())))
    return pd.DataFrame(rows)


def coint_i1():
    T = pd.read_csv(OUT / "s2_69_tests.csv")
    T = T[~T.iso.str.startswith("US NFC")]
    both = T[(T.adf_r_p >= 0.10) & (T.adf_i_p >= 0.10)]
    return pd.DataFrame([dict(item="countries with unit root not rejected in both r and i_real (ADF p >= 0.10)",
                              n=len(both), cointegrated=int(both.eg_coint.sum()), countries=",".join(both.iso)),
                         dict(item="countries with i_real stationary by ADF (p < 0.10)", n=int((T.adf_i_p < 0.10).sum())),
                         dict(item="countries with r stationary by ADF (p < 0.10)", n=int((T.adf_r_p < 0.10).sum()))])


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    A = f_decomposition()
    A.to_csv(OUT / "ck_72_decomposition.csv", index=False)
    print(A.round(4).to_string(), flush=True)
    C = coint_i1()
    C.to_csv(OUT / "ck_69_i1.csv", index=False)
    print(C.to_string(), flush=True)
    B = stage1_subsets()
    B.to_csv(OUT / "ck_s1_subsets.csv", index=False)
    print(B[["item", "est", "ci90_lo", "ci90_hi", "n", "G", "label"]].round(4).to_string(), flush=True)
