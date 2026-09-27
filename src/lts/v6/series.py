"""Iteration 6, stage 1-2: long profit-rate series (pre_registration_v6.md, sections 0-1).

AMECO (total economy, 1960-2024) for USA, GBR, FRA, DEU (West Germany 1960-1990 + Germany 1991-2024),
ITA, NLD, SWE, AUS; JPN and CAN as variants.  US nonfinancial corporations from NIPA T11400 + BEA Fixed
Assets tables 4.1-4.6.
Output: results/v6/series.csv (long: geo, variant, year, columns)
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "results" / "v6"
MAIN8 = ["USA", "GBR", "FRA", "DEU", "ITA", "NLD", "SWE", "AUS"]
EXTRA = ["JPN", "CAN"]
YEARS = range(1960, 2025)
CODES = ["UOND", "UWCD", "UKCT", "OKND", "PIGT", "UVGD", "OVGD", "UTVT", "NETD", "NWTD", "NSTD", "NLHT", "AVGDGP",
         "UBRA", "UIGT", "UOGD", "OIGT"]


# ---------------------------------------------------------------- AMECO
def ameco_raw():
    fr = []
    for f in sorted(glob.glob(str(RAW / "v6" / "ameco" / "AMECO*.TXT"))):
        d = pd.read_csv(f, sep=";", dtype=str, encoding="latin-1")
        d.columns = [c.strip() for c in d.columns]
        fr.append(d)
    a = pd.concat(fr)
    p = a.CODE.str.split(".", expand=True)
    a = a[(p[3] == "0") & (p[4] == "0")].copy()          # national currency / default unit (not EUR, PPS)
    a["geo"], a["var"] = a.CODE.str.split(".").str[0], a.CODE.str.split(".").str[-1]
    a = a[a["var"].isin(CODES)].drop_duplicates(["geo", "var"])
    yrs = [c for c in a.columns if c.isdigit()]
    long = a.melt(id_vars=["geo", "var"], value_vars=yrs, var_name="year", value_name="v")
    long["year"] = long.year.astype(int)
    long["v"] = pd.to_numeric(long.v, errors="coerce")
    return long.pivot_table(index=["geo", "year"], columns="var", values="v").reset_index()


def ameco_country(raw, geo):
    return raw[raw.geo == geo].set_index("year").reindex(YEARS)


def derive(d, mixed_correction=True):
    d = d.copy()
    ratio = d.NETD / d.NWTD
    d["W_se"] = d.UWCD * (ratio - 1) if mixed_correction else 0.0
    d["PI"] = d.UOND - d.W_se
    d["W"] = d.UWCD + d.W_se
    pk = (d.PIGT / 100).combine_first(d.UIGT / d.OIGT)   # D_W has no PIGT: implicit GFCF deflator (A-V6-PK)
    d["PIGT"] = pk * 100
    d["K"] = d.OKND * pk
    d["Y"] = d.UWCD + d.UOND
    d["Q_Y"], d["P_Y"] = d.OVGD, d.UVGD / d.OVGD
    d["Q_K"], d["P_K"] = d.OKND, d.PIGT / 100
    d["H"] = d.NLHT
    d["gap"] = d.AVGDGP
    d["PI_gni"] = d.PI + d.UBRA
    return d


def hist_cost(d, delta_scale=1.0, init_scale=1.0, y0=1960):
    """Perpetual inventory at historical cost from GFCF at current prices (UIGT)."""
    dl = (d.UKCT / d.K.shift()).loc[y0 + 1:].median() * delta_scale
    k = pd.Series(np.nan, index=d.index)
    k.loc[y0] = d.K.loc[y0] * init_scale
    for t in range(y0 + 1, d.index.max() + 1):
        k.loc[t] = k.loc[t - 1] * (1 - dl) + d.UIGT.loc[t]
    return k, dl


def finish(d):
    d["r"], d["e"], d["k"] = d.PI / d.K, d.PI / d.W, d.K / d.W
    d["rM"] = d.PI / (d.K + d.W)
    d["share"] = d.PI / d.Y
    d["tech"] = d.Q_K / d.H
    d["w_hour"] = d.W / d.H
    return d


def ameco_panel():
    raw = ameco_raw()
    rows = []
    for geo in MAIN8 + EXTRA:
        src = "D_W" if geo == "DEU" else geo
        base = ameco_country(raw, src)
        if geo == "DEU":                               # splice (a): West Germany to 1990, Germany from 1991
            de = ameco_country(raw, "DEU")
            base = pd.concat([base.loc[:1990], de.loc[1991:]])
            dw91 = ameco_country(raw, "D_W").loc[1991]
        for vname, mc in (("main", True), ("no_mi", False)):
            d = finish(derive(base, mc))
            d = d[d.PI.notna() & d.K.notna() & d.W.notna()]
            if d.empty:
                continue
            out = d.assign(geo=geo, variant=vname)
            rows.append(out)
            if vname == "main" and d.UIGT.notna().all() and d.index.min() == 1960:
                for tag, ds, isc in (("hc", 1.0, 1.0), ("hc_d07", 0.7, 1.0), ("hc_d13", 1.3, 1.0),
                                     ("hc_i07", 1.0, 0.7), ("hc_i13", 1.0, 1.3)):
                    k, dl = hist_cost(d, ds, isc)
                    h = d.copy()
                    h["K"] = k
                    h["delta"] = dl
                    h = finish(h)
                    rows.append(h.loc[1975:].assign(geo=geo, variant=tag))
                    if tag == "hc":            # profit net of historical-cost depreciation
                        g = h.copy()
                        g["PI"] = g.PI + g.UKCT - dl * k.shift()
                        rows.append(finish(g).loc[1975:].assign(geo=geo, variant="hc_hcdep"))
        if geo == "DEU":                               # splice (b): shift West-German segment to match 1991
            d = finish(derive(base, True))
            dw = finish(derive(pd.DataFrame([dw91]).set_index(pd.Index([1991])), True))
            b = d.copy()
            for c in ("r", "rM", "e", "k", "share"):
                shift = d.loc[1991, c] - dw.loc[1991, c]
                b.loc[:1990, c] = d.loc[:1990, c] + shift
            rows.append(b.assign(geo=geo, variant="deu_shift"))
            rows.append(d.loc[:1990].assign(geo=geo, variant="dw_only"))
    p = pd.concat(rows).reset_index().rename(columns={"index": "year"})
    return p


# ---------------------------------------------------------------- US NFC
def nipa(codes):
    d = pd.read_csv(RAW / "v5" / "NipaDataA.txt", dtype=str)
    d.columns = ["code", "year", "v"]
    d = d[d.code.isin(codes)]
    d["v"] = pd.to_numeric(d.v.str.replace(",", ""), errors="coerce")
    d["year"] = d.year.astype(int)
    return d.pivot(index="year", columns="code", values="v")


def fa(section, sheet, line):
    x = pd.read_excel(RAW / "v6" / "bea_fa" / f"Section{section}All_xls.xlsx", sheet_name=sheet, header=None)
    hdr = x.index[x.iloc[:, 0].astype(str).str.strip() == "Line"][0]
    h = pd.to_numeric(x.iloc[hdr], errors="coerce")
    cols = [i for i in range(2, x.shape[1]) if np.isfinite(h.iloc[i])]
    row = x[pd.to_numeric(x.iloc[:, 0], errors="coerce") == line].iloc[0]
    return pd.Series(pd.to_numeric(row.iloc[cols], errors="coerce").values, index=h.iloc[cols].astype(int).values)


def us_nfc():
    n = nipa(["A455RC", "B455RX", "B456RC", "A460RC", "W325RC", "W326RC", "A457RC", "W322RC", "B933RC", "B934RC"])
    d = pd.DataFrame({"VA": n.A455RC, "Q_Yg": n.B455RX, "CFC_nipa": n.B456RC, "W": n.A460RC, "PI": n.W326RC,
                      "Y": n.A457RC, "PI_allcorp": n.W322RC, "row_rec": n.B933RC, "row_pay": n.B934RC})
    L = {"K": 37, "K_eq": 38, "K_st": 39, "K_ip": 40, "K_allcorp": 17}
    for k, ln in L.items():
        d[k] = fa(4, "FAAt401-A", ln)
    for k, ln in (("Khc", 37), ("Khc_eq", 38), ("Khc_st", 39)):
        d[k] = fa(4, "FAAt403-A", ln)
    d["CFC_cc"], d["CFC_hc"] = fa(4, "FAAt404-A", 37), fa(4, "FAAt406-A", 37)
    qi = fa(4, "FAAt402-A", 37)
    d["Q_K"] = qi * d.K.loc[2017] / qi.loc[2017]
    d["P_K"] = d.K / d.Q_K
    d["Q_Y"], d["P_Y"] = d.Q_Yg, d.VA / d.Q_Yg
    raw = ameco_raw()
    d["H"] = ameco_country(raw, "USA").NLHT.reindex(d.index)
    gap = ameco_country(raw, "USA").AVGDGP
    d["gap"] = gap.reindex(d.index)
    d = d[(d.index >= 1946) & (d.index <= 2024)]
    rows = []
    specs = {"main": dict(), "no_ip": dict(K=d.K_eq + d.K_st), "hc": dict(K=d.Khc),
             "hc_no_ip": dict(K=d.Khc_eq + d.Khc_st), "hc_hcdep": dict(K=d.Khc, PI=d.PI + d.CFC_cc - d.CFC_hc),
             "gni": dict(PI=d.PI + d.row_rec - d.row_pay)}
    for v, over in specs.items():
        x = d.copy()
        for c, s in over.items():
            x[c] = s
        rows.append(finish(x).assign(geo="USA_NFC", variant=v))
    return pd.concat(rows).reset_index().rename(columns={"index": "year"})


KLEMS_GEO = {"FRA": "FR", "DEU": "DE", "ITA": "IT", "NLD": "NL", "SWE": "SE", "GBR": "UK", "USA": "US"}


def drift_variants(p):
    """Journal 4 (exploratory): AMECO capital drift vs official stocks; 'kdrift' and USA 'bea_K'."""
    ca = pd.read_csv(RAW / "euklems" / "capital_accounts.csv", usecols=["nace_r2_code", "geo_code", "year", "K_GFCF"],
                     low_memory=False)
    tot = ca[ca.nace_r2_code == "TOT"]
    bea = fa(1, "FAAt101-A", 2) / 1000
    rows, info = [], []
    for geo in MAIN8:
        m = p[(p.geo == geo) & (p.variant == "main")].set_index("year").copy()
        if geo == "USA":
            ref = bea.reindex(m.index)
            yrs = m.index
        elif geo in KLEMS_GEO:
            ref = tot[tot.geo_code == KLEMS_GEO[geo]].set_index("year").K_GFCF.reindex(m.index) / 1000
            yrs = m.index[(m.index >= 1995) & (m.index <= 2021)]
        else:
            info.append(dict(geo=geo, g=np.nan, note="no official reference stock"))
            continue
        lr = np.log(m.loc[yrs, "K"] / ref.loc[yrs]).dropna()
        g = np.polyfit(lr.index.to_numpy(float), lr.to_numpy(), 1)[0]
        info.append(dict(geo=geo, g=g, y0=int(lr.index.min()), y1=int(lr.index.max()),
                         ratio_first=float(np.exp(lr.iloc[0])), ratio_last=float(np.exp(lr.iloc[-1]))))
        k = m.copy()
        k["K"] = m.K * np.exp(-g * (m.index - 2021))
        rows.append(finish(k).assign(geo=geo, variant="kdrift"))
        if geo == "USA":
            b = m.copy()
            b["K"] = ref
            rows.append(finish(b).assign(geo=geo, variant="bea_K"))
    pd.DataFrame(info).to_csv(OUT / "kdrift_info.csv", index=False)
    return pd.concat(rows).reset_index().rename(columns={"index": "year"})


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    a = ameco_panel()
    a = pd.concat([a, drift_variants(a)], ignore_index=True)
    u = us_nfc()
    s = pd.concat([a, u], ignore_index=True)
    keep = ["geo", "variant", "year", "PI", "W", "K", "Y", "r", "rM", "e", "k", "share", "Q_Y", "P_Y", "Q_K", "P_K",
            "H", "tech", "w_hour", "gap", "PI_gni", "UBRA", "UIGT", "UKCT", "delta", "OVGD", "W_se"]
    s = s[[c for c in keep if c in s.columns]]
    s.to_csv(OUT / "series.csv", index=False)
    print(s.groupby(["geo", "variant"]).year.agg(["min", "max", "size"]).to_string())
