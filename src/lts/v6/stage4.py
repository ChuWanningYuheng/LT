"""Iteration 6, stage 4 (exploratory): foreign labour behind corporate profits (pre_registration_v6.md, §4).

4.1 GDP vs GNI view: r with PI + net primary income from abroad (AMECO UBRA) and, for US NFC, with
    corporate profits from the rest of the world (NIPA B933RC - B934RC).  Trends as in T1.
4.2 Extended organic composition: omega_c = W*_c / W_c, where W*_c is the foreign compensation embodied
    (through the world Leontief inverse of intermediate inputs only) in country c's imported intermediate
    inputs, valued at the supplying country-industry compensation per unit of output, and W_c is c's own
    compensation in the same table.  k* = K / (W (1 + omega)).  FIGARO 2010-2024; WIOD 2000-2014.
    Hours embodied H*_c are reported for WIOD (SEA hours of employees scaled to all employed).
Outputs: results/v6/s4_omega.csv, s4_gni_trends.csv, s4_panel.csv
"""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd

from .. import figaro
from ..core import leontief_inverse
from ..figaro import INDUSTRIES
from ..v5.stage1 import ols_fwl
from .series import MAIN8, OUT, RAW
from .tests import trend

GEO2 = {"USA": "US", "GBR": "GB", "FRA": "FR", "DEU": "DE", "ITA": "IT", "NLD": "NL", "SWE": "SE", "AUS": "AU",
        "JPN": "JP", "CAN": "CA"}
N = len(INDUSTRIES)


def embodied(A, x, wcoef, hcoef, countries, n, targets):
    """For each target country c: foreign compensation (and hours) embodied in c's intermediate imports."""
    Lw = leontief_inverse(A, check=False)
    Z = A * x[None, :]
    out = {}
    for c in targets:
        if c not in countries:
            continue
        ci = countries.index(c)
        sl = slice(ci * n, (ci + 1) * n)
        m = Z[:, sl].sum(1)
        m[sl] = 0.0                                   # only imported intermediates
        tot = Lw @ m
        foreign = np.ones(len(x), bool)
        foreign[sl] = False
        Wstar = float((wcoef * tot)[foreign].sum())
        Hstar = float((hcoef * tot)[foreign].sum()) if hcoef is not None else np.nan
        Wown = float((wcoef[sl] * x[sl]).sum())
        out[c] = dict(Wstar=Wstar, Wown=Wown, omega=Wstar / Wown, Hstar=Hstar, imports=float(m.sum()))
    return out


def figaro_omega(years=range(2010, 2025)):
    rows = []
    for y in years:
        d = figaro.load(y)
        C = [str(c) for c in d["countries"]]
        x = d["Z"].sum(1) + d["F"].sum(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            A = np.where(x[None, :] > 0, d["Z"] / x[None, :], 0.0)
            w = np.where(x > 0, d["V"][list(d["varows"]).index("D1")] / x, 0.0)
        res = embodied(A, x, w, None, C, N, list(GEO2.values()))
        for c2, r in res.items():
            rows.append(dict(source="FIGARO", year=y, geo=[k for k, v in GEO2.items() if v == c2][0], **r))
        print("FIGARO", y, flush=True)
    return rows


def wiod_omega(years=range(2000, 2015)):
    import pyreadr
    sea = pd.read_excel(RAW / "v6" / "wiod" / "Socio_Economic_Accounts.xlsx", sheet_name="DATA")
    xr = pd.read_excel(RAW / "v6" / "wiod" / "Exchange_Rates.xlsx", sheet_name="EXR", header=3)
    xr = xr.set_index("Acronym")
    rows = []
    for y in years:
        f = glob.glob(str(RAW / "v6" / "wiod" / "wiot" / f"WIOT{y}_*.RData"))[0]
        r = pyreadr.read_r(f)
        d = r[list(r)[0]]
        body = d.iloc[:2464]                               # 44 countries x 56 industries
        cols = [c for c in d.columns if c[:3].isalpha() and c[3:].isdigit() and 1 <= int(c[3:]) <= 56]
        Z = body[cols].to_numpy(float)
        x = d[d.IndustryCode == "GO"][cols].to_numpy(float).ravel()
        countries = list(dict.fromkeys(c[:3] for c in cols))
        n = 56
        with np.errstate(divide="ignore", invalid="ignore"):
            A = np.where(x[None, :] > 0, Z / x[None, :], 0.0)
        s = sea[sea.variable.isin(["COMP", "GO", "H_EMPE", "EMP", "EMPE"])][["country", "variable", "code", y]]
        s = s.pivot_table(index=["country", "code"], columns="variable", values=y)
        codes = list(body.IndustryCode.iloc[:n])
        w, h = np.zeros(len(x)), np.zeros(len(x))
        for i, c in enumerate(countries):
            if c == "ROW":
                continue
            sc = s.loc[c].reindex(codes) if c in s.index.get_level_values(0) else None
            if sc is None:
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                w[i * n:(i + 1) * n] = np.nan_to_num((sc.COMP / sc.GO).to_numpy())
                rate = xr.loc[c, f"_{y}"] if c in xr.index else np.nan
                hrs = sc.H_EMPE * (sc.EMP / sc.EMPE)
                h[i * n:(i + 1) * n] = np.nan_to_num((hrs / (sc.GO * rate)).to_numpy())
        # ROW: output-weighted average of non-ROW coefficients (WIOD SEA has no ROW)
        nr = [i for i, c in enumerate(countries) if c != "ROW"]
        ri = countries.index("ROW")
        xs = np.array([x[i * n:(i + 1) * n] for i in nr])
        for arr in (w, h):
            V = np.array([arr[i * n:(i + 1) * n] for i in nr])
            arr[ri * n:(ri + 1) * n] = np.nansum(V * xs, 0) / np.maximum(xs.sum(0), 1e-12)
        res = embodied(A, x, w, h, countries, n, list(GEO2))
        for c3, rr in res.items():
            rows.append(dict(source="WIOD", year=y, geo=c3, **rr))
        print("WIOD", y, flush=True)
    return rows


def gni_trends(s):
    rows = []
    for geo in MAIN8 + ["USA_NFC"]:
        g = s[(s.geo == geo) & (s.variant == "main")].set_index("year").sort_index()
        if geo == "USA_NFC":
            g = g.loc[1951:]
            gg = s[(s.geo == geo) & (s.variant == "gni")].set_index("year").sort_index().loc[1951:]
            rgni = gg.r
        else:
            rgni = g.PI_gni / g.K
        for name, y in (("gdp", g.r), ("gni", rgni)):
            y = y.dropna()
            for lag in (2, 4, 8):
                sl, se, p = trend(y.to_numpy(), lag)
                rows.append(dict(geo=geo, view=name, lag=lag, slope=sl, p=p, y0=int(y.index.min()), y1=int(y.index.max()),
                                 mean=y.mean()))
    return pd.DataFrame(rows)


def kstar_panel(s, om):
    rows = []
    for src, g in om.groupby("source"):
        m = s[(s.variant == "main") & s.geo.isin(g.geo.unique())][["geo", "year", "PI", "K", "W", "rM", "k"]]
        d = m.merge(g[["geo", "year", "omega"]], on=["geo", "year"])
        d["kstar"] = d.K / (d.W * (1 + d.omega))
        d["lnrM"] = np.log(d.rM)
        d["ln1k"], d["ln1ks"] = np.log1p(d.k), np.log1p(d.kstar)
        d = d.sort_values(["geo", "year"])
        for c in ("lnrM", "ln1k", "ln1ks"):
            d["d_" + c] = d.groupby("geo")[c].diff()
        d = d.dropna(subset=["d_lnrM"])
        for x in ("d_ln1k", "d_ln1ks"):
            r = ols_fwl(d, "d_lnrM", x, [], ["geo", "year"], B=9999)
            fe = d[["d_lnrM", x]].copy()
            for c in ("d_lnrM", x):
                fe[c] = fe[c] - d.groupby("geo")[c].transform("mean") - d.groupby("year")[c].transform("mean") + fe[c].mean()
            r2 = np.corrcoef(fe.d_lnrM, fe[x])[0, 1] ** 2
            rows.append(dict(source=src, x=x, r2_within=r2, **r))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    s = pd.read_csv(OUT / "series.csv")
    gt = gni_trends(s)
    gt.to_csv(OUT / "s4_gni_trends.csv", index=False)
    print(gt.round(4).to_string())
    om = pd.DataFrame(figaro_omega() + wiod_omega())
    om.to_csv(OUT / "s4_omega.csv", index=False)
    print(om.pivot_table(index="year", columns=["source", "geo"], values="omega").round(3).to_string())
    kp = kstar_panel(s, om)
    kp.to_csv(OUT / "s4_panel.csv", index=False)
    print(kp.round(3).to_string())
