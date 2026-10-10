"""Iteration 12, stage 3: reproduction schemes — disproportion between departments I and II before recessions
(pre_registration_v12.md, stage 3). Outcome 83.

WIOD 2000-2014: share of a country-industry's output going to department-I uses (intermediate use of all countries +
GFCF + inventories, codes 60-61); department I if the share >= 0.5 in the country's 2000 table. R_t = ln(X_I / X_II),
D_t = R_t - R_{t-3}; benchmark G_t = d3 ln(GFCF / GDP). Recession onsets from PWT rgdpna; event: onset in t+1 or t+2.

Usage: PYTHONPATH=src python -P -m lts.v12.stage3
Outputs: results/v12/s3_*.csv
"""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd

from ..revisit import wiod
from ..v6.series import ROOT
from ..v8.rule import decide
from ..v11.stage1 import reg

OUT = ROOT / "results" / "v12"
YEARS = range(2000, 2015)


def year_table(y):
    import pyreadr
    f = glob.glob(str(wiod.RAW / "wiod" / "wiot" / f"WIOT{y}_*.RData"))[0]
    r = pyreadr.read_r(f)
    d = r[list(r)[0]]
    N = wiod.N
    body = d.iloc[:44 * N]
    countries = list(dict.fromkeys(body.Country))
    codes = list(body.IndustryCode.iloc[:N])
    Zc = [c for c in d.columns if c[:3] in countries and c[3:].isdigit() and 1 <= int(c[3:]) <= N]
    Z = body[Zc].to_numpy(float)
    inter = Z.sum(1)
    fd = {k: body[[f"{c}{k}" for c in countries]].to_numpy(float) for k in range(57, 62)}
    x = d[d.IndustryCode == "GO"][Zc].to_numpy(float).ravel()
    va = d[d.IndustryCode == "VA"][Zc].to_numpy(float).ravel()
    dept1_use = inter + fd[60].sum(1) + fd[61].sum(1)
    uses = dept1_use + fd[57].sum(1) + fd[58].sum(1) + fd[59].sum(1)
    cells = pd.DataFrame({"country": np.repeat(countries, N), "code": np.tile(codes, len(countries)), "x": x, "va": va,
                          "share_I": np.divide(dept1_use, uses, out=np.full(len(x), np.nan), where=uses > 0)})
    gfcf = pd.Series(fd[60].sum(0), index=countries)                  # GFCF purchases of each destination country
    return cells, gfcf


def build():
    rows, cls = [], {}
    for y in YEARS:
        cells, gfcf = year_table(y)
        for c, g in cells.groupby("country"):
            if c == "ROW":
                continue
            if c not in cls:                                          # classification fixed by the 2000 table
                cls[c] = dict(zip(g.code, g.share_I >= 0.5))
            isI = g.code.map(cls[c]).fillna(False).astype(bool)
            XI, XII = g.x[isI].sum(), g.x[~isI].sum()
            XIc = (g.x * g.share_I.fillna(0)).sum()
            rows.append(dict(country=c, year=y, X_I=XI, X_II=XII, X_I_cont=XIc, X_II_cont=g.x.sum() - XIc,
                             GDP=g.va.sum(), GFCF=gfcf[c], n_I=int(isI.sum())))
        print("s3", y, flush=True)
    D = pd.DataFrame(rows)
    D["R"] = np.log(D.X_I / D.X_II)
    D["R_cont"] = np.log(D.X_I_cont / D.X_II_cont)
    D["inv"] = np.log(D.GFCF / D.GDP)
    D = D.sort_values(["country", "year"])
    for v in ("R", "R_cont", "inv"):
        D["d3_" + v] = D.groupby("country")[v].diff(3)
    pd.DataFrame([dict(country=c, code=k, dept_I=v) for c, m in cls.items() for k, v in m.items()]).to_csv(
        OUT / "s3_classification_2000.csv", index=False)
    return D


def onsets():
    p = pd.read_excel(ROOT / "data" / "raw" / "v9" / "pwt110.xlsx", sheet_name="Data", usecols=["countrycode", "year", "rgdpna"])
    p = p.sort_values(["countrycode", "year"])
    p["g"] = p.groupby("countrycode").rgdpna.transform(lambda s: np.log(s).diff())
    p["g_prev"] = p.groupby("countrycode").g.shift(1)
    p["onset"] = ((p.g < 0) & (p.g_prev >= 0)).astype(float)
    return p.rename(columns={"countrycode": "country"})[["country", "year", "onset", "g"]]


def run():
    D = build()
    O = onsets()
    k = O.set_index(["country", "year"]).onset
    D["Y"] = np.fmax(k.reindex(pd.MultiIndex.from_arrays([D.country, D.year + 1])).to_numpy(),
                     k.reindex(pd.MultiIndex.from_arrays([D.country, D.year + 2])).to_numpy())
    D["yr"] = D.year.astype(str)
    D.to_csv(OUT / "s3_panel.csv", index=False)
    s = D[D.year.between(2003, 2014)].dropna(subset=["Y", "d3_R", "d3_inv"]).copy()
    for v in ("d3_R", "d3_R_cont", "d3_inv"):
        s[v + "_z"] = s[v] / s[v].std()
    rows, desc = [], []
    r = reg(s, "Y", ["d3_R_z", "d3_inv_z"], 0, fe=("country",), cl="country")
    rows.append(dict(outcome="83: LPM recession onset in t+1/t+2 on D_t = d3 ln(X_I/X_II) (std), with d3 ln(GFCF/GDP), "
                             "country FE, WIOD 2003-2014", **r, theta0=0.0, delta=0.02, direction=">",
                     label=decide(r["est"], r["ci90_lo"], r["ci90_hi"], 0, 0.02, ">")))
    rg = reg(s, "Y", ["d3_R_z", "d3_inv_z"], 1, fe=("country",), cl="country")
    desc.append(dict(item="83: beta_G (accelerator benchmark) in the same model", **rg))
    ra = reg(s, "Y", ["d3_R_z"], 0, fe=("country",), cl="country")
    desc.append(dict(item="83 variant: D alone (no investment control)", **ra))
    ry = reg(s, "Y", ["d3_R_z", "d3_inv_z"], 0, fe=("country", "yr"), cl="country")
    desc.append(dict(item="83 variant: with year effects", **ry))
    rc = reg(s, "Y", ["d3_R_cont_z", "d3_inv_z"], 0, fe=("country",), cl="country")
    desc.append(dict(item="83 variant: continuous allocation of output to departments", **rc))
    # event profile around onsets (2003-2014)
    prof = []
    on = O[(O.onset == 1) & O.year.between(2003, 2016) & O.country.isin(D.country.unique())]
    kd = D.set_index(["country", "year"])
    for kk in range(-3, 3):
        idx = pd.MultiIndex.from_arrays([on.country, on.year + kk])
        prof.append(dict(k=kk, D_mean=float(kd.d3_R.reindex(idx).mean()), G_mean=float(kd.d3_inv.reindex(idx).mean()),
                         n=int(kd.d3_R.reindex(idx).notna().sum())))
    P = pd.DataFrame(prof)
    P.to_csv(OUT / "s3_event_profile.csv", index=False)
    desc.append(dict(item="events: recession onsets 2003-2016 in WIOD countries", est=int(len(on)),
                     n=int(on.country.nunique())))
    pd.DataFrame(rows).to_csv(OUT / "s3_outcomes.csv", index=False)
    pd.DataFrame(desc).to_csv(OUT / "s3_descriptive.csv", index=False)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows)[["outcome", "est", "ci90_lo", "ci90_hi", "n", "G", "label"]].round(4).to_string())
    print(pd.DataFrame(desc).round(4).to_string())
    print(P.round(4).to_string())


if __name__ == "__main__":
    run()
