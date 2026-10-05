"""Iteration 6, stage 4 extension (exploratory; pre_registration_v6.md, journal 9).

OECD ICIO 1995-2022 (81 countries x 50 industries): omega_VA (foreign value added embodied in imported
intermediates / own value added) and omega_PWT (compensation proxied by VA x Penn World Table labour share).
k* panel as in stage 4 on 1995-2022.  Canada: drift of AMECO capital against StatCan 36-10-0096.
Outputs: results/v6/s4i_omega.csv, s4i_validation.csv, s4i_panel.csv, s4i_canada.csv
"""
from __future__ import annotations

import glob
import zipfile

import numpy as np
import pandas as pd

from .series import MAIN8, OUT, RAW, ameco_country, ameco_raw, derive, finish
from .stage4 import GEO2, embodied, kstar_panel
from .tests import trend

GD = RAW / "v6" / "gdrive"
MERGE = {"CN1": "CHN", "CN2": "CHN", "MX1": "MEX", "MX2": "MEX"}
YEARS = range(1995, 2023)


def icio_year(y):
    for f in sorted(glob.glob(str(GD / "*.zip"))):
        z = zipfile.ZipFile(f)
        for name in (f"{y}_SML.csv", f"{y}.csv"):
            if name in z.namelist():
                d = pd.read_csv(z.open(name), index_col=0)
                break
        else:
            continue
        break
    else:
        raise FileNotFoundError(y)
    lab = lambda s: MERGE.get(s[:3], s[:3]) + s[3:]                    # noqa: E731
    body = [i for i in d.index if "_" in i]
    cols = [c for c in d.columns if c in set(body)]
    Z = d.loc[body, cols].rename(index=lab, columns=lab)
    Z = Z.T.groupby(level=0, sort=False).sum().T.groupby(level=0, sort=False).sum()
    va = d.loc["VA", cols].rename(index=lab).groupby(level=0, sort=False).sum()
    x = d.loc["OUT", cols].rename(index=lab).groupby(level=0, sort=False).sum()
    labels = list(Z.columns)
    Z = Z.loc[labels, labels].to_numpy(float)
    return labels, Z.clip(min=0), va.loc[labels].to_numpy(float), x.loc[labels].to_numpy(float)


def pwt_labsh():
    p = pd.read_stata(RAW / "v6" / "pwt1001.dta", columns=["countrycode", "year", "labsh"])
    p = p[p.year >= 1995].pivot(index="year", columns="countrycode", values="labsh")
    p = p.reindex(YEARS).ffill()                                         # 2020-2022: 2019 value
    return p


def icio_omega():
    ls = pwt_labsh()
    rows = []
    for y in YEARS:
        labels, Z, va, x = icio_year(y)
        countries = list(dict.fromkeys(l.split("_")[0] for l in labels))
        n = len(labels) // len(countries)
        with np.errstate(divide="ignore", invalid="ignore"):
            A = np.where(x[None, :] > 0, Z / x[None, :], 0.0)
            v = np.where(x > 0, va / x, 0.0)
        med = float(ls.loc[y].median())
        sh = np.array([ls.loc[y].get(c, np.nan) for c in countries])
        sh = np.where(np.isnan(sh), med, sh)
        if "ROW" in countries:                                           # output-weighted mean of others
            xc = np.array([x[i * n:(i + 1) * n].sum() for i in range(len(countries))])
            ok = np.array([c != "ROW" for c in countries])
            sh[countries.index("ROW")] = float((sh[ok] * xc[ok]).sum() / xc[ok].sum())
        w = v * np.repeat(sh, n)
        for name, coef in (("ICIO_VA", v), ("ICIO_PWT", w)):
            res = embodied(A, x, coef, None, countries, n, list(GEO2))
            for c3, r in res.items():
                rows.append(dict(source=name, year=y, geo=c3, **r))
        print("ICIO", y, flush=True)
    return pd.DataFrame(rows)


def validation(om):
    old = pd.read_csv(OUT / "s4_omega.csv")
    rows = []
    for src in ("FIGARO", "WIOD"):
        a = old[old.source == src][["geo", "year", "omega"]]
        for new in ("ICIO_PWT", "ICIO_VA"):
            b = om[om.source == new][["geo", "year", "omega"]]
            m = a.merge(b, on=["geo", "year"], suffixes=("_old", "_icio"))
            m = m[m.geo.isin(MAIN8)]
            dm = m.groupby("geo")[["omega_old", "omega_icio"]].transform(lambda s: s - s.mean())
            rows.append(dict(ref=src, icio=new, n=len(m), years=f"{m.year.min()}-{m.year.max()}",
                             corr_level=m.omega_old.corr(m.omega_icio), corr_within=dm.omega_old.corr(dm.omega_icio),
                             ratio_median=(m.omega_icio / m.omega_old).median()))
    return pd.DataFrame(rows)


def canada():
    raw = ameco_raw()
    d = finish(derive(ameco_country(raw, "CAN"), 1.0))
    z = zipfile.ZipFile(GD / "36100096-eng.zip")
    sc = pd.read_csv(z.open("36100096.csv"), low_memory=False)
    sc = sc[(sc.GEO == "Canada") & (sc.Prices == "Current prices") & (sc.Industry == "Total all industries") &
            (sc["Flows and stocks"] == "Geometric end-year net stock") & (sc.Assets == "Total non-residential")]
    ks = sc.set_index("REF_DATE").VALUE.astype(float) / 1e3                # millions -> billions (AMECO units)
    ks.index = ks.index.astype(int)
    rows = []
    ratio = (d.K / ks).dropna()
    for a, b in ((1961, 2024), (1991, 2024), (1995, 2021)):
        r = np.log(ratio.loc[a:b])
        g = np.polyfit(r.index, r.to_numpy(), 1)[0]
        rows.append(dict(kind="drift", y0=a, y1=b, ratio_first=float(np.exp(r.iloc[0])), ratio_last=float(np.exp(r.iloc[-1])),
                         g=g))
    p = d[d.PI.notna()].loc[1991:2024].copy()
    for name, K in (("ameco_K", p.K), ("statcan_index_K", p.K.loc[1991] * ks.loc[1991:2024] / ks.loc[1991])):
        q = p.copy()
        q["K"] = K
        q = finish(q)
        for series in ("r", "rM"):
            for a in (1991, 1996):
                for b in (2024, 2019):
                    for lag in (2, 4, 8):
                        sl, se, pv = trend(q.loc[a:b, series].to_numpy(), lag)
                        rows.append(dict(kind="trend", variant=name, series=series, y0=a, y1=b, lag=lag, slope=sl, p=pv,
                                         first=q[series].iloc[0], last=q[series].iloc[-1]))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ca = canada()
    ca.to_csv(OUT / "s4i_canada.csv", index=False)
    print(ca.round(4).to_string(), flush=True)
    om = icio_omega()
    om.to_csv(OUT / "s4i_omega.csv", index=False)
    print(om.pivot_table(index="year", columns=["source", "geo"], values="omega").round(3).to_string())
    va = validation(om)
    va.to_csv(OUT / "s4i_validation.csv", index=False)
    print(va.round(3).to_string())
    s = pd.read_csv(OUT / "series.csv")
    kp = kstar_panel(s, om)
    kp.to_csv(OUT / "s4i_panel.csv", index=False)
    print(kp.round(3).to_string())
