"""Iteration 10, stage 11: Okishio vs Marx (pre_registration_v10.md, stage 11; journal 10).

PWT 11, OECD countries, non-overlapping 5-year windows 1950-2020.
11.1 Marx-biased technical change: d ln(Y/L) > 0 and d ln(Y/K) < 0 (outcome 62).
11.2 d ln r = d ln(1 - omega) + d ln(Y/K) (nominal, current PPP); windows with falling r where Y/K accounts for more
     than half of the fall and omega does not rise (outcome 63). Descriptive: Basu (real Y/K vs relative price of K);
     business sector (BSDB) variant of 11.1.

Usage: PYTHONPATH=src python -P -m lts.v10.stage11
Outputs: results/v10/s11_*.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..v6.series import ROOT
from ..v8.rule import share_row
from .stage9 import OECD38, bsdb

OUT = ROOT / "results" / "v10"
WINDOWS = [(a, a + 5) for a in range(1950, 2020, 5)]


def pwt():
    p = pd.read_excel(ROOT / "data" / "raw" / "v9" / "pwt110.xlsx", sheet_name="Data")
    p = p[p.countrycode.isin(OECD38)].copy()
    p["Y"], p["K"], p["L"] = p.rgdpna, p.rnna, p.emp * p.avh
    p["yk_nom"] = p.cgdpo / p.cn
    p["omega"] = p.labsh
    p["r"] = (1 - p.labsh) * p.yk_nom
    p["pk_rel"] = p.pl_k / p.pl_gdpo
    return p.set_index(["countrycode", "year"]).sort_index()


def windows(p):
    rows = []
    for c in p.index.get_level_values(0).unique():
        x = p.loc[c]
        for a, b in WINDOWS:
            if a not in x.index or b not in x.index:
                continue
            s, e = x.loc[a], x.loc[b]
            rec = dict(country=c, start=a, end=b)
            if np.isfinite([s.Y, e.Y, s.K, e.K, s.L, e.L]).all() and min(s.L, e.L, s.K, e.K) > 0:
                rec["dln_yl"] = np.log(e.Y / e.L) - np.log(s.Y / s.L)
                rec["dln_yk_real"] = np.log(e.Y / e.K) - np.log(s.Y / s.K)
                rec["marx_bias"] = bool(rec["dln_yl"] > 0 and rec["dln_yk_real"] < 0)
            if np.isfinite([s.r, e.r, s.yk_nom, e.yk_nom, s.omega, e.omega]).all() and min(s.r, e.r) > 0:
                rec["dln_r"] = np.log(e.r / s.r)
                rec["dln_1mw"] = np.log((1 - e.omega) / (1 - s.omega))
                rec["dln_yk_nom"] = np.log(e.yk_nom / s.yk_nom)
                rec["d_omega"] = e.omega - s.omega
                if np.isfinite([s.pk_rel, e.pk_rel]).all() and min(s.pk_rel, e.pk_rel) > 0:
                    rec["dln_pk_rel"] = np.log(e.pk_rel / s.pk_rel)
            rows.append(rec)
    return pd.DataFrame(rows)


def bsdb_windows():
    b = pd.read_stata(ROOT / "data" / "raw" / "v10" / "oecd_bsdb.dta")
    b["geo"] = b.cty
    a = b.groupby(["geo", "year"])[["gdpbv", "kbv", "etb", "hrs"]].mean().reset_index()
    a["year"] = a.year.astype(int)
    rows = []
    for g, x in a.groupby("geo"):
        x = x.set_index("year")
        for s0, e0 in [(y, y + 5) for y in range(1960, 1995, 5)]:
            if s0 not in x.index or e0 not in x.index:
                continue
            s, e = x.loc[s0], x.loc[e0]
            L0, L1 = s.etb, e.etb
            if not np.isfinite([s.gdpbv, e.gdpbv, s.kbv, e.kbv, L0, L1]).all() or min(L0, L1, s.kbv, e.kbv) <= 0:
                continue
            dyl = np.log(e.gdpbv / L1) - np.log(s.gdpbv / L0)
            dyk = np.log(e.gdpbv / e.kbv) - np.log(s.gdpbv / s.kbv)
            rows.append(dict(geo=g, start=s0, end=e0, dln_yl=dyl, dln_yk_real=dyk, marx_bias=bool(dyl > 0 and dyk < 0)))
    return pd.DataFrame(rows)


def run():
    p = pwt()
    W = windows(p)
    W.to_csv(OUT / "s11_windows.csv", index=False)
    w62 = W.dropna(subset=["marx_bias"])
    out = [share_row("62: share of country-windows with Marx-biased technical change (PWT 11, OECD, 5-year)",
                     w62.marx_bias.astype(bool), 0.5, 0.1, ">", countries=w62.country.nunique())]
    w63 = W.dropna(subset=["dln_r"])
    fall = w63[w63.dln_r < -0.05].copy()
    fall["marx"] = (fall.dln_yk_nom < 0.5 * fall.dln_r) & (fall.d_omega <= 0)
    fall["yk_dominant"] = fall.dln_yk_nom < 0.5 * fall.dln_r
    out.append(share_row("63: among windows with falling r, share where Y/K gives > half of the fall and omega does not rise",
                         fall.marx, 0.5, 0.15, ">", windows_falling=len(fall), windows_all=len(w63)))
    fall.to_csv(OUT / "s11_falling_r_windows.csv", index=False)
    desc = []
    for (a, b), lab in (((1950, 1975), "1950-1975"), ((1975, 2000), "1975-2000"), ((2000, 2020), "2000-2020")):
        x = w62[(w62.start >= a) & (w62.end <= b)]
        desc.append(dict(item=f"share Marx-biased, windows within {lab}", est=float(x.marx_bias.mean()), n=len(x)))
    desc.append(dict(item="falling-r windows: share where Y/K gives > half of the fall (omega unrestricted)",
                     est=float(fall.yk_dominant.mean()), n=len(fall)))
    desc.append(dict(item="falling-r windows: share with omega not rising", est=float((fall.d_omega <= 0).mean()), n=len(fall)))
    bw = w63.dropna(subset=["dln_pk_rel"])
    desc.append(dict(item="Basu: mean d ln(real Y/K) over windows", est=float(W.dln_yk_real.mean()), n=int(W.dln_yk_real.notna().sum())))
    desc.append(dict(item="Basu: mean d ln(pl_k / pl_gdpo) over windows", est=float(bw.dln_pk_rel.mean()), n=len(bw)))
    desc.append(dict(item="Basu: corr(d ln nominal Y/K, d ln real Y/K)",
                     est=float(W[["dln_yk_nom", "dln_yk_real"]].dropna().corr().iloc[0, 1]), n=int(W[["dln_yk_nom", "dln_yk_real"]].dropna().shape[0])))
    B = bsdb_windows()
    B.to_csv(OUT / "s11_bsdb_windows.csv", index=False)
    desc.append(dict(item="BSDB business sector 1960-1995 (persons): share Marx-biased (descriptive)",
                     est=float(B.marx_bias.mean()), n=len(B)))
    O = pd.DataFrame(out)
    O.to_csv(OUT / "s11_outcomes.csv", index=False)
    D = pd.DataFrame(desc)
    D.to_csv(OUT / "s11_descriptive.csv", index=False)
    pd.set_option("display.width", 250)
    print(O.round(4).to_string(), "\n", D.round(4).to_string(), flush=True)


if __name__ == "__main__":
    run()
