"""Iteration 9, stage 7: Marxian profit rate with productive / unproductive labour (US, BEA, 1997-2024).

r_m = s / (C + v); s = NDP - compensation in productive industries; v = that compensation;
C = current-cost net stock of private fixed assets of productive industries (pre_registration_v9.md, journal 9).
Outputs: results/v9/s7_*.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from ..v8.rule import decide
from .stage1 import OUT

ROOT = OUT.parents[1]
FA = ROOT / "data" / "raw" / "v6" / "bea_fa"
VA = ROOT / "data" / "raw" / "v9" / "bea_ValueAdded.xlsx"
YEARS = range(1997, 2025)

SECTORS = {"agr": "Agriculture, forestry, fishing, and hunting", "min": "Mining", "util": "Utilities",
           "con": "Construction", "man": "Manufacturing", "whl": "Wholesale trade", "ret": "Retail trade",
           "trn": "Transportation and warehousing", "inf": "Information", "fin": "Finance and insurance",
           "rea": "Real estate and rental and leasing", "leg": "Legal services",
           "csd": "Computer systems design and related services",
           "mps": "Miscellaneous professional, scientific, and technical services",
           "mgt": "Management of companies and enterprises", "adm": "Administrative and support services",
           "wst": "Waste management and remediation services", "edu": "Educational services",
           "hlt": "Health care and social assistance", "art": "Arts, entertainment, and recreation",
           "acc": "Accommodation and food services", "oth": "Other services, except government", "gov": "Government"}
FA_NAME = {"hlt": "Health care and social assistance", "oth": "Other services, except government"}
UNPROD = {"main": {"whl", "ret", "fin", "rea", "leg", "mgt", "adm", "gov"}}
UNPROD["V2 transport unproductive"] = UNPROD["main"] | {"trn"}
UNPROD["V3 information + computer design unproductive"] = UNPROD["main"] | {"inf", "csd"}
UNPROD["V4 all professional services unproductive"] = UNPROD["main"] | {"csd", "mps"}
UNPROD["V5 Tsoulfidis-Paitaridis classification (journal 12)"] = UNPROD["main"] | {"mps", "wst"}


def norm(s):
    return " ".join(str(s).replace("\\", " ").split()).lower().rstrip("0123456789 ").strip()


def va_components():
    x = pd.read_excel(VA, sheet_name="TVA113-A", header=None)
    years = [int(v) for v in x.iloc[7, 3:] if str(v).isdigit()]
    rows = x.iloc[8:].reset_index(drop=True)
    out = {}
    for i in range(len(rows) - 3):
        name = norm(rows.iloc[i, 1])
        if norm(rows.iloc[i + 1, 1]) == "compensation of employees":
            vals = {}
            for k, comp in zip((1, 2, 3), ("comp", "tax", "gos")):
                vals[comp] = pd.to_numeric(rows.iloc[i + k, 3:3 + len(years)], errors="coerce").to_numpy()
            if name not in out:
                out[name] = pd.DataFrame(vals, index=years)
    return out


def fa_table(sheet):
    x = pd.read_excel(FA / "Section3All_xls.xlsx", sheet_name=sheet, header=None)
    hdr = x.iloc[7].tolist()
    years = {j: int(float(v)) for j, v in enumerate(hdr) if str(v).replace(".0", "").isdigit()}
    out = {}
    for i in range(8, len(x)):
        name = norm(x.iloc[i, 1])
        if name and name not in out:
            out[name] = pd.Series({y: pd.to_numeric(x.iloc[i, j], errors="coerce") for j, y in years.items()})
    return out


def depreciation_total():
    x = pd.read_excel(FA / "Section1All_xls.xlsx", sheet_name="FAAt103-A", header=None)
    hdr = x.iloc[7].tolist()
    years = {j: int(float(v)) for j, v in enumerate(hdr) if str(v).replace(".0", "").isdigit()}
    for i in range(8, len(x)):
        if norm(x.iloc[i, 1]) == "fixed assets":
            return pd.Series({y: pd.to_numeric(x.iloc[i, j], errors="coerce") for j, y in years.items()})
    raise KeyError("fixed assets line")


def build():
    va = va_components()
    k = fa_table("FAAt301ESI-A")
    dep = depreciation_total()
    gdp = va["gross domestic product"]
    rows = []
    for code, name in SECTORS.items():
        v = va.get(norm(name))
        kk = k.get(norm(FA_NAME.get(code, name)))
        if v is None:
            raise KeyError(name)
        for y in YEARS:
            rows.append(dict(code=code, year=y, comp=v.comp.get(y), va=v[["comp", "tax", "gos"]].loc[y].sum(),
                             K=(kk.get(y) if kk is not None else np.nan) if code != "gov" else 0.0))
    D = pd.DataFrame(rows)
    tot = pd.DataFrame({"gdp": gdp[["comp", "tax", "gos"]].sum(1), "comp_all": gdp.comp}).loc[list(YEARS)]
    tot["dep"] = dep.reindex(tot.index)
    tot["ndp"] = tot.gdp - tot.dep
    tot["K_private"] = k[norm("Private fixed assets")].reindex(tot.index)
    return D, tot


def trend(s):
    t = np.arange(len(s))
    f = sm.OLS(s.to_numpy(dtype=float), sm.add_constant(t)).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
    return f.params[1], f.bse[1]


def run():
    D, tot = build()
    out, series = [], []
    for var, unp in UNPROD.items():
        prod = D[~D.code.isin(unp)]
        v = prod.groupby("year").comp.sum()
        C = prod.groupby("year").K.sum()
        s = tot.ndp - v
        rm = s / (C + v)
        r = (tot.ndp - tot.comp_all) / tot.K_private
        series.append(pd.DataFrame({"variant": var, "year": rm.index, "r_m": rm.values, "r": r.values, "s": s.values,
                                    "v": v.values, "C": C.values, "e_m": (s / v).values,
                                    "unprod_share_comp": (1 - v / tot.comp_all).values}))
        b, se = trend(rm)
        tq = 1.645
        out.append(dict(variant=var, item="outcome 37: trend of r_m per year", est=b, se=se, ci90_lo=b - tq * se,
                        ci90_hi=b + tq * se, label=decide(b, b - tq * se, b + tq * se, 0, 0.0002, "<")))
        b2, se2 = trend(rm - r)
        out.append(dict(variant=var, item="outcome 38: trend of r_m - trend of r", est=b2, se=se2, ci90_lo=b2 - tq * se2,
                        ci90_hi=b2 + tq * se2, label=decide(b2, b2 - tq * se2, b2 + tq * se2, 0, 0.0002, "<")))
        b3, se3 = trend(r)
        out.append(dict(variant=var, item="conventional r: trend per year", est=b3, se=se3))
    S = pd.concat(series)
    S.to_csv(OUT / "s7_series.csv", index=False)
    O = pd.DataFrame(out)
    O.to_csv(OUT / "s7_outcomes.csv", index=False)
    return S, O


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    S, O = run()
    print(S[S.variant == "main"].round(4).to_string())
    print(O.round(5).to_string())
