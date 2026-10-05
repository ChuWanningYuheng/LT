"""Iteration 7, part 1.3(7): national checks of the KLEMS non-national-accounts intangibles profile.

UK (ONS, investment by SIC division 1997-2023): stocks by perpetual inventory with StatCan (Gu & Macdonald
2020, table 2) depreciation rates: organisational capital 0.40, branding (advertising) 0.60, design 0.20,
financial product innovation 0.20; initial stock I_1997 / (g + delta), g = 0.04.  Compared with KLEMS UK
K_OrgCap + K_Brand + K_Design + K_NFP (no training: ONS has none) by KLEMS industry.
NL (CBS marketing assets 2013-2024, flows): compared with KLEMS NL I_Brand.
Output: results/v7/p1_profiles.csv, p1_profiles_detail.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..v4.b_data import divs
from ..v6.series import ROOT
from .part1 import FINE_ORDER, KL, OUT

NAT = ROOT / "data" / "raw" / "v7" / "national"
ONS_ASSETS = {"Organisational Capital": 0.40, "Branding": 0.60, "Design": 0.20, "Financial Product Innovation": 0.20}


def to_fine(div_set):
    for k in FINE_ORDER:
        if div_set and div_set <= divs(k):
            return k
    return None


def ons_stocks(g=0.04):
    x = pd.ExcelFile(NAT / "ons_intangibles_dec25.xlsx")
    out = []
    for sheet, dl in ONS_ASSETS.items():
        d = pd.read_excel(x, sheet, header=None)
        h = d.index[d.iloc[:, 0].astype(str).str.strip() == "Industry"][0]          # first table = Total
        years = [int(v) for v in d.iloc[h, 1:] if str(v).strip().isdigit()]
        body = d.iloc[h + 1:]
        stop = body.index[body.iloc[:, 0].isna() | ~body.iloc[:, 0].astype(str).str.strip().str[:2].str.isdigit()]
        body = body.loc[:stop[0] - 1] if len(stop) else body
        for _, row in body.iterrows():
            code = str(row.iloc[0]).replace("\xa0", "").strip()
            dv = frozenset(int(c) for c in code.replace(" ", "").split(",") if c.isdigit())
            inv = pd.to_numeric(pd.Series(row.iloc[1:1 + len(years)].to_numpy(), index=years), errors="coerce").fillna(0)
            k, ks = inv.iloc[0] / (g + dl), []
            for y, i in inv.items():
                k = k * (1 - dl) + i if y > years[0] else k
                ks.append(k)
            out.append(pd.DataFrame({"year": years, "K": ks, "I": inv.to_numpy(), "asset": sheet, "divs": [dv] * len(years)}))
    o = pd.concat(out)
    o["ind"] = o.divs.map(to_fine)
    return o.groupby(["ind", "year"])[["K", "I"]].sum().reset_index()


def compare():
    ia = pd.read_csv(KL / "intangibles_analytical.csv", low_memory=False,
                     usecols=["nace_r2_code", "geo_code", "year", "K_OrgCap", "K_Brand", "K_Design", "K_NFP", "I_Brand"])
    ia = ia.rename(columns={"nace_r2_code": "ind", "geo_code": "geo"})
    panel = pd.read_csv(ROOT / "results" / "v4" / "b_panel.csv.gz")[["geo", "ind", "year", "K", "W"]]
    rows, det = [], []
    # UK
    uk = ia[ia.geo == "UK"].assign(U_klems=lambda d: d[["K_OrgCap", "K_Brand", "K_Design", "K_NFP"]].fillna(0).sum(axis=1))
    o = ons_stocks()
    m = o.rename(columns={"K": "U_ons"}).merge(uk[["ind", "year", "U_klems"]], on=["ind", "year"])
    m = m.merge(panel[panel.geo == "UK"][["ind", "year", "K", "W"]], on=["ind", "year"])
    m = m[(m.K > 0) & (m.year.between(2000, 2021))]
    det.append(m.assign(country="UK"))
    for y in (2005, 2012, 2019):
        s = m[m.year == y]
        rows.append(dict(country="UK", year=y, n_ind=len(s),
                         corr_UK_ratio=np.corrcoef(s.U_ons / s.K, s.U_klems / s.K)[0, 1],
                         spearman=(s.U_ons / s.K).rank().corr((s.U_klems / s.K).rank()),
                         level_ratio_total=s.U_ons.sum() / s.U_klems.sum(),
                         corr_with_lnKW_ons=np.corrcoef(s.U_ons / s.K, np.log(s.K / s.W))[0, 1],
                         corr_with_lnKW_klems=np.corrcoef(s.U_klems / s.K, np.log(s.K / s.W))[0, 1]))
    # NL brand flows
    x = pd.read_excel(NAT / "cbs_marketing_2013_2024.xlsx", "Lopende prijzen", header=None)
    hdr = x.iloc[1].astype(str).tolist()
    body = x.iloc[2:].copy()
    body.columns = hdr
    body = body[body["Type"].astype(str).str.contains("Totaal", case=False)]
    cbs = []
    for c in hdr[2:]:
        code = c.split(" ")[0]
        dv = frozenset()
        if code.replace("-", "").isdigit():
            a, *b = code.split("-")
            dv = frozenset(range(int(a), int(b[0] if b else a) + 1))
        elif len(code) == 1 and code.isalpha():
            dv = divs(code)
        ind = to_fine(dv) if dv else None
        if ind:
            for _, r in body.iterrows():
                cbs.append(dict(ind=ind, year=int(r["Jaar"]), I_cbs=pd.to_numeric(r[c], errors="coerce")))
    cbs = pd.DataFrame(cbs).groupby(["ind", "year"]).I_cbs.sum().reset_index()
    nl = ia[ia.geo == "NL"][["ind", "year", "I_Brand"]]
    mm = cbs.merge(nl, on=["ind", "year"]).merge(panel[panel.geo == "NL"][["ind", "year", "K", "W"]], on=["ind", "year"])
    mm = mm[mm.K > 0]
    det.append(mm.assign(country="NL"))
    for y in (2013, 2017, 2021):
        s = mm[mm.year == y]
        rows.append(dict(country="NL (brand flows)", year=y, n_ind=len(s),
                         corr_UK_ratio=np.corrcoef(s.I_cbs / s.K, s.I_Brand / s.K)[0, 1],
                         spearman=(s.I_cbs / s.K).rank().corr((s.I_Brand / s.K).rank()),
                         level_ratio_total=s.I_cbs.sum() / s.I_Brand.sum(),
                         corr_with_lnKW_ons=np.corrcoef(s.I_cbs / s.K, np.log(s.K / s.W))[0, 1],
                         corr_with_lnKW_klems=np.corrcoef(s.I_Brand / s.K, np.log(s.K / s.W))[0, 1]))
    return pd.DataFrame(rows).rename(columns={"corr_UK_ratio": "corr_U/K_national_vs_klems",
                                              "corr_with_lnKW_ons": "corr(U/K_national, lnKW)",
                                              "corr_with_lnKW_klems": "corr(U/K_klems, lnKW)"}), pd.concat(det)


if __name__ == "__main__":
    r, d = compare()
    r.to_csv(OUT / "p1_profiles.csv", index=False)
    d.drop(columns=[c for c in d.columns if c == "divs"]).to_csv(OUT / "p1_profiles_detail.csv", index=False)
    print(r.round(3).to_string())
