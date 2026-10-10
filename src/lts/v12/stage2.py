"""Iteration 12, stage 2: unequal exchange — transfer through productivity in two frames; the circle (pre_registration_v12.md).

2.1 EXIOBASE 2017 (2021): the 9b gap in world-average hours (WAH) of the same product, split into the North's premium
    (1/k - 1) H, the South's deficit H - H*, and exchange at equal productivity H* - E; frame (a) transfer T_a = H* - E,
    frame (b) (market value at the average) T_b = H - E.
2.2 WIOD 2000-2014: balanced outflow of embodied hours O_c (raw) and S_c (productivity-adjusted, w_pl1) towards the
    other 42 WIOD countries; response: 5-year change of ln productivity per hour relative to the US (PWT 11).
Outcomes 81, 82.

Usage: PYTHONPATH=src python -P -m lts.v12.stage2 [frames|flows|circle|all]
Outputs: results/v12/s2_*.csv
"""
from __future__ import annotations

import glob
import sys

import numpy as np
import pandas as pd
import scipy.linalg as sla

from ..revisit import wiod
from ..v6.series import ROOT
from ..v8.rule import decide
from ..v11.stage1 import reg, widest, wild_fast

OUT = ROOT / "results" / "v12"
YEARS = range(2000, 2015)
AB = {"A01", "A02", "A03", "B"}


# ================================================================ 2.1 frames
def frames():
    from ..v9.stage5 import NORTH, NORTH_NARROW, flows, load
    from ..v9b.stage23 import adj_weights, decompose_additive, rows_like
    rows = []
    for year, pl_year in ((2017, 2017), (2021, 2017)):                # PLD 2023 has 2005, 2011, 2017 only (journal 2)
        try:
            A, idx, Y, x, F = load(year)
        except FileNotFoundError:
            continue
        for nm, north in (("IMF advanced", NORTH), ("narrow North", NORTH_NARROW)):
            f = flows(A, idx, Y, x, F, north)
            r = decompose_additive(f, F, pl_year=pl_year)
            used_pl = pl_year
            w, _ = adj_weights(f, used_pl)
            K1 = r["north_adj_hours_per_hour"]
            H, Hs = r["H_SN"], r["Hstar_SN"]
            vN = r["wN"] + r["pN"] + r["rN"]                          # money per actual northern hour
            muN = vN / K1                                             # money per northern WAH
            W_ = f["comp"]
            P_ = rows_like(F, "Operating surplus: Remaining net operating surplus", "Operating surplus: Rents on land",
                           "Operating surplus: Royalties on resources")
            R_ = rows_like(F, "Operating surplus: Consumption of fixed capital", "Other net taxes on production")
            muW = float((W_ + P_ + R_).sum() / (f["hrs"] * w).sum())  # world money per WAH
            for mu_name, mu in (("North (main)", muN), ("world", muW)):
                E = r["VA_SN_MEUR"] / mu
                G = H * K1 - E
                prem, deficit, exch = H * (K1 - 1), H - Hs, Hs - E
                rows.append(dict(year=year, pl_year=used_pl, north=nm, mu=mu_name, H=H, Hstar=Hs, E=E, G_star=G,
                                 north_premium=prem, south_deficit=deficit, exchange_equal_prod=exch,
                                 share_premium=prem / G, share_deficit=deficit / G, share_exchange=exch / G,
                                 T_a=exch, T_b=H - E, share_T_a=exch / G, share_T_b=(H - E) / G, ratio_Ta_Tb=exch / (H - E),
                                 check_9b_bcd=r["share2_b"] + r["share2_c"] + r["share2_d"], K1=K1))
                print(year, nm, mu_name, {k: round(v, 3) for k, v in rows[-1].items() if isinstance(v, float) and
                                          k.startswith(("share", "ratio", "check"))}, flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "s2_frames.csv", index=False)
    return D


# ================================================================ 2.2 flows
def final_demand(y):
    """WIOT final demand by destination country (codes 57-61 summed), n x countries, US$ mn"""
    import pyreadr
    f = glob.glob(str(wiod.RAW / "wiod" / "wiot" / f"WIOT{y}_*.RData"))[0]
    r = pyreadr.read_r(f)
    d = r[list(r)[0]]
    body = d.iloc[:44 * wiod.N]
    countries = list(dict.fromkeys(body.Country))
    FD = np.column_stack([body[[f"{c}{k}" for k in range(57, 62)]].to_numpy(float).sum(1) for c in countries])
    return FD, countries


def flows_year(y):
    from ..v9.stage34 import year_system
    A, cells = year_system(y)
    W = wiod.wiot(y)
    FD, countries = final_demand(y)
    n = len(cells)
    lu = sla.lu_factor(np.eye(n) - A)
    X = sla.lu_solve(lu, FD)                                          # output required by each destination's FD
    x = W["x"]
    v = np.divide(W["va"], x, out=np.zeros(n), where=x > 0)
    h = cells.H_x.to_numpy()
    hs = h * cells.w_pl1.to_numpy()
    ctry = cells.country.to_numpy()
    code = cells.code.to_numpy()
    wiod_c = [c for c in countries if c != "ROW"]
    Hc = cells.assign(Hs=cells.H * cells.w_pl1).groupby("country")[["H", "Hs"]].sum()
    rows = []
    for c in wiod_c:
        ic = ctry == c
        P = [d for d in wiod_c if d != c]
        jP = np.isin(np.array(countries), P)
        iP = np.isin(ctry, P)
        Xexp = X[:, jP].sum(1)                                        # output for partners' final demand
        Ximp = X[:, countries.index(c)]                               # output for c's final demand
        rec = dict(country=c, year=y)
        for nm, hh, HH in (("raw", h, Hc.loc[c, "H"]), ("adj", hs, Hc.loc[c, "Hs"])):
            HX = float((hh * Xexp)[ic].sum())
            VX = float((v * Xexp)[ic].sum())
            HM = float((hh * Ximp)[iP].sum())
            VM = float((v * Ximp)[iP].sum())
            rec.update({f"HX_{nm}": HX, f"VX": VX, f"HM_{nm}": HM, f"VM": VM, f"H_{nm}": HH,
                        f"O_{nm}": (HX / VX - HM / VM) * VX / HH if VX > 0 and VM > 0 and HH > 0 else np.nan,
                        f"N_{nm}": (HX - HM) / HH if HH > 0 else np.nan})
        ab = np.isin(code, list(AB)) & ic
        rec["commodity_share_X"] = float((v * Xexp)[ab].sum() / rec["VX"]) if rec["VX"] > 0 else np.nan
        rows.append(rec)
    wiod.wiot.cache_clear()
    return rows


def all_flows():
    rows = []
    for y in YEARS:
        rows += flows_year(y)
        print("flows", y, flush=True)
        pd.DataFrame(rows).to_csv(OUT / "s2_flows.csv", index=False)
    return pd.DataFrame(rows)


# ================================================================ 2.2 circle
def pwt():
    p = pd.read_excel(ROOT / "data" / "raw" / "v9" / "pwt110.xlsx", sheet_name="Data",
                      usecols=["countrycode", "year", "rgdpo", "emp", "avh", "hc", "csh_i"])
    p["prod"] = np.where(p.avh.notna(), p.rgdpo / (p.emp * p.avh), p.rgdpo / p.emp)
    p["per_worker"] = p.avh.isna()
    us = p[p.countrycode == "USA"].set_index("year")
    p["lnRP"] = np.log(p["prod"]) - np.log(p.year.map(us["prod"]))
    p["lnP"] = np.log(p["prod"])
    return p.rename(columns={"countrycode": "country"})


def wdi(ind):
    import json
    w = json.load(open(ROOT / "data" / "raw" / "v12" / f"wb_{ind}.json"))[1]
    return pd.Series({(r["countryiso3code"], int(r["date"])): r["value"] for r in w if r["value"] is not None})


def panel(h=5):
    F = pd.read_csv(OUT / "s2_flows.csv")
    p = pwt()
    k = p.set_index(["country", "year"])
    d = F.merge(p[["country", "year", "lnRP", "lnP", "hc", "csh_i", "per_worker"]], on=["country", "year"], how="left")
    fut = k.lnRP.reindex(pd.MultiIndex.from_arrays([d.country, d.year + h])).to_numpy()
    d["dRP"] = fut - d.lnRP
    # relative to the hours-weighted mean of the WIOD sample (variant)
    hw = F[["country", "year", "H_raw"]]
    pm = p.merge(hw[hw.year == 2000][["country", "H_raw"]], on="country")
    mean = pm.groupby("year").apply(lambda g: np.average(g.lnP, weights=g.H_raw), include_groups=False)
    d["lnRPm"] = d.lnP - d.year.map(mean)
    futm = (k.lnP.reindex(pd.MultiIndex.from_arrays([d.country, d.year + h])).to_numpy()
            - (d.year + h).map(mean).to_numpy())
    d["dRPm"] = futm - d.lnRPm
    agr, ind = wdi("NV.AGR.TOTL.ZS"), wdi("NV.IND.TOTL.ZS")
    d["agr"] = [agr.get((c, y), np.nan) for c, y in zip(d.country, d.year)]
    d["ind"] = [ind.get((c, y), np.nan) for c, y in zip(d.country, d.year)]
    c00 = F[F.year == 2000].set_index("country").commodity_share_X
    d["commodity_exporter"] = d.country.map(c00 > 0.25)
    d["yr"] = d.year.astype(str)
    d["one"] = "all"
    return d[d.country != "USA"]


CTRL = ["lnRP", "hc", "csh_i", "agr", "ind"]


def est(d, x, y="dRP", ctrl=CTRL, fe=("yr",), B=9999):
    d = d.dropna(subset=[x, y] + ctrl).copy()
    d[x] = d[x] / d[x].std()
    r = reg(d, y, [x] + ctrl, 0, fe=fe, cl="country", B=B)
    r["countries"] = int(d.country.nunique())
    return r


def iv(d, x, insts, y="dRP", ctrl=CTRL):
    """2SLS with year effects, CR1 by country (exploratory; point estimate, SE, first-stage F)"""
    from ..v11.stage1 import demean
    d = d.dropna(subset=[x, y] + ctrl + insts).copy()
    d[x] = d[x] / d[x].std()
    M = demean(d[[y, x] + ctrl + insts].to_numpy(float), [d.yr.to_numpy()])
    yy, xe, C, Z = M[:, 0], M[:, 1:2], M[:, 2:2 + len(ctrl)], M[:, 2 + len(ctrl):]
    Zf = np.c_[Z, C]
    pi = np.linalg.lstsq(Zf, xe, rcond=None)[0]
    xh = Zf @ pi
    Xh = np.c_[xh, C]
    b = np.linalg.lstsq(Xh, yy, rcond=None)[0]
    e = yy - np.c_[xe, C] @ b
    gi = pd.factorize(d.country)[0]
    G = gi.max() + 1
    XtX = np.linalg.inv(Xh.T @ Xh)
    meat = sum(np.outer(Xh[gi == g].T @ e[gi == g], Xh[gi == g].T @ e[gi == g]) for g in range(G))
    se = float(np.sqrt((XtX @ meat @ XtX)[0, 0] * G / (G - 1)))
    # first-stage F on excluded instruments (homoskedastic, descriptive)
    e1 = xe[:, 0] - Zf @ pi[:, 0]
    pr = np.linalg.lstsq(C, xe[:, 0], rcond=None)[0]
    e0 = xe[:, 0] - C @ pr
    q = Z.shape[1]
    Fst = ((e0 @ e0 - e1 @ e1) / q) / ((e1 @ e1) / (len(yy) - Zf.shape[1]))
    from scipy import stats
    tq = stats.t.ppf(0.95, G - 1)
    return dict(est=float(b[0]), se=se, ci90_lo=float(b[0] - tq * se), ci90_hi=float(b[0] + tq * se), n=len(d), G=int(G),
                first_stage_F=float(Fst))


def circle():
    d = panel(5)
    rows, desc = [], []
    for nm, x, lab in (("81: beta of balanced outflow O_c (raw hours) on 5-year relative productivity growth", "O_raw", "81"),
                       ("82: beta of strict outflow S_c (productivity-adjusted hours)", "O_adj", "82")):
        r = est(d, x)
        rows.append(dict(outcome=nm, **r, theta0=0.0, delta=0.01, direction="<",
                         label=decide(r["est"], r["ci90_lo"], r["ci90_hi"], 0, 0.01, "<")))
    d10 = panel(10)
    for x in ("O_raw", "O_adj"):
        variants = {
            "10-year horizon": (d10, x, "dRP", ("yr",)),
            "relative to WIOD hours-weighted mean": (d, x, "dRPm", ("yr",)),
            "non-overlapping windows (2000, 2005, 2010)": (d[d.year.isin([2000, 2005, 2010])], x, "dRP", ("yr",)),
            "without China": (d[d.country != "CHN"], x, "dRP", ("yr",)),
            "without commodity exporters": (d[~d.commodity_exporter.fillna(False).astype(bool)], x, "dRP", ("yr",)),
            "country fixed effects": (d, x, "dRP", ("yr", "country")),
        }
        for vn, (dd, xx, yy, fe) in variants.items():
            r = est(dd, xx, y=yy, fe=fe)
            desc.append(dict(item=f"{x}: {vn}", **r, label=decide(r["est"], r["ci90_lo"], r["ci90_hi"], 0, 0.01, "<")))
    for x in ("N_raw", "N_adj"):
        r = est(d, x)
        desc.append(dict(item=f"{x} (unbalanced net flow)", **r, label=decide(r["est"], r["ci90_lo"], r["ci90_hi"], 0, 0.01, "<")))
    # exploratory 2SLS
    geo = pd.read_excel(ROOT / "data" / "raw" / "v12" / "cepii_geo_cepii.xls")
    ll = geo.drop_duplicates("iso3").set_index("iso3").landlocked
    d["landlocked"] = d.country.map(ll).astype(float)
    c00 = d[d.year == 2000].set_index("country").commodity_share_X
    d["commodity_share_2000"] = d.country.map(c00)
    for x in ("O_raw", "O_adj"):
        try:
            r = iv(d, x, ["commodity_share_2000", "landlocked"])
            desc.append(dict(item=f"{x}: 2SLS (instruments: commodity export share 2000, landlocked), exploratory", **r))
        except Exception as exc:                                      # noqa: BLE001
            desc.append(dict(item=f"{x}: 2SLS failed: {exc}"))
    # descriptive: who flows out
    s = d.groupby("country")[["O_raw", "O_adj", "N_raw", "commodity_share_X"]].mean().sort_values("O_raw")
    s.to_csv(OUT / "s2_flows_by_country.csv")
    O = pd.DataFrame(rows)
    O.to_csv(OUT / "s2_outcomes.csv", index=False)
    D = pd.DataFrame(desc)
    D.to_csv(OUT / "s2_descriptive.csv", index=False)
    pd.set_option("display.width", 250)
    print(O[["outcome", "est", "ci90_lo", "ci90_hi", "n", "G", "label"]].round(4).to_string())
    print(D.round(4).to_string())
    print(s.round(3).to_string())


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("frames", "all"):
        frames()
    if what in ("flows", "all"):
        all_flows()
    if what in ("circle", "all"):
        circle()
