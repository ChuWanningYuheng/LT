"""Iteration 7, part 1 (v): US SG&A and net PP&E by SIC from SEC Financial Statement Data Sets.

For fiscal years 2012, 2016, 2019, 2021: 10-K filings (fp = FY) in the four quarterly files of the following
calendar year; latest filing per company; tags SellingGeneralAndAdministrativeExpense (fallback
GeneralAndAdministrativeExpense + SellingAndMarketingExpense), PropertyPlantAndEquipmentNet; USD, no
co-registrant, qtrs = 4 (flows) or 0 (stock), ddate at fiscal year end.
Output: data/raw/v7/sec/sec_firm_fy.csv (firm-level extract), results/v7/p1_sec_sic2.csv
"""
from __future__ import annotations

import io
import time
import urllib.request
import zipfile

import numpy as np
import pandas as pd

from ..v6.series import ROOT

RAW = ROOT / "data" / "raw" / "v7" / "sec"
RAW.mkdir(parents=True, exist_ok=True)
UA = "LT-research-script/1.0 (noreply@anthropic.com)"
TAGS = ["SellingGeneralAndAdministrativeExpense", "GeneralAndAdministrativeExpense", "SellingAndMarketingExpense",
        "PropertyPlantAndEquipmentNet", "Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"]
FYS = (2012, 2016, 2019, 2021)


def fetch(q):
    url = f"https://www.sec.gov/files/dera/data/financial-statement-data-sets/{q}.zip"
    for k in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            return urllib.request.urlopen(req, timeout=600).read()
        except Exception as exc:  # noqa: BLE001
            print("retry", q, exc, flush=True)
            time.sleep(2 ** (k + 1))
    raise RuntimeError(q)


def extract(fy):
    rows = []
    for q in range(1, 5):
        z = zipfile.ZipFile(io.BytesIO(fetch(f"{fy + 1}q{q}")))
        sub = pd.read_csv(z.open("sub.txt"), sep="\t", dtype=str, usecols=["adsh", "cik", "sic", "form", "fy", "fp", "period", "filed"])
        sub = sub[(sub.form == "10-K") & (sub.fp == "FY") & (sub.fy == str(fy))]
        keep = set(sub.adsh)
        parts = []
        for ch in pd.read_csv(z.open("num.txt"), sep="\t", dtype=str, chunksize=2_000_000,
                              usecols=["adsh", "tag", "version", "coreg", "ddate", "qtrs", "uom", "value"]):
            ch = ch[ch.adsh.isin(keep) & ch.tag.isin(TAGS) & (ch.uom == "USD") & ch.coreg.isna()]
            parts.append(ch)
        num = pd.concat(parts)
        num = num.merge(sub, on="adsh")
        num = num[num.ddate == num.period]                       # value at / for the fiscal year end
        num = num[((num.tag == "PropertyPlantAndEquipmentNet") & (num.qtrs == "0")) |
                  ((num.tag != "PropertyPlantAndEquipmentNet") & (num.qtrs == "4"))]
        rows.append(num)
        print("sec", fy, q, len(sub), len(num), flush=True)
    d = pd.concat(rows)
    d["value"] = pd.to_numeric(d.value, errors="coerce")
    d = d.sort_values("filed").groupby(["cik", "tag"]).last().reset_index()   # latest filing per company
    w = d.pivot_table(index=["cik", "sic"], columns="tag", values="value", aggfunc="last").reset_index()
    w["fy"] = fy
    return w


def build():
    firms = pd.concat([extract(fy) for fy in FYS], ignore_index=True)
    firms.to_csv(RAW / "sec_firm_fy.csv", index=False)
    return firms


def by_sic2(firms):
    f = firms.copy()
    sga = f.get("SellingGeneralAndAdministrativeExpense")
    alt = f.get("GeneralAndAdministrativeExpense", 0).fillna(0) + f.get("SellingAndMarketingExpense", 0).fillna(0)
    f["SGA"] = sga.where(sga.notna(), alt.where(alt > 0))
    f["PPE"] = f.PropertyPlantAndEquipmentNet
    f["sic"] = pd.to_numeric(f.sic, errors="coerce")
    f = f[f.SGA.notna() & (f.SGA > 0) & (f.PPE > 0) & f.sic.notna()]
    f = f[~f.sic.between(6000, 6799) & ~f.sic.between(4900, 4999)]
    f["sic2"] = (f.sic // 100).astype(int)
    g = f.groupby(["fy", "sic2"]).agg(SGA=("SGA", "sum"), PPE=("PPE", "sum"), n=("cik", "size")).reset_index()
    g["sga_ppe"] = g.SGA / g.PPE
    return g


if __name__ == "__main__":
    firms = build()
    g = by_sic2(firms)
    g.to_csv(ROOT / "results" / "v7" / "p1_sec_sic2.csv", index=False)
    print(g.groupby("sic2").sga_ppe.median().round(2).to_string())
