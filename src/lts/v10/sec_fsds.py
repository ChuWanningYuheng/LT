"""Iteration 10, stage 0: firm-year extract from SEC Financial Statement Data Sets (2009q2-2026q2).

All 10-K / 10-K/A filings with fp = FY in every quarterly archive; tags below; USD (shares/pure for employees);
no co-registrant; no dimensional rows (segments empty; journal 7); flows with qtrs = 4, stocks with qtrs = 0;
ddate at the fiscal year end (period); keys with conflicting values left after these filters are dropped;
latest filing per company x fiscal year x tag. Archives are not kept (as in iteration 7).

Output: data/raw/v10/sec/sec_fy_<quarter>.csv (per archive) and data/raw/v10/sec/sec_firm_fy_all.csv
Usage: PYTHONPATH=src python -m lts.v10.sec_fsds [build|merge]
"""
from __future__ import annotations

import csv
import datetime as dt
import glob
import hashlib
import io
import sys
import time
import urllib.request
import zipfile

import pandas as pd

from ..v6.series import ROOT

RAW = ROOT / "data" / "raw" / "v10" / "sec"
RAW.mkdir(parents=True, exist_ok=True)
UA = "LT-research-script/1.0 (noreply@anthropic.com)"
FLOWS = ["NetIncomeLoss", "ProfitLoss", "OperatingIncomeLoss",
         "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
         "Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet", "CostsAndExpenses",
         "CostOfRevenue", "CostOfGoodsAndServicesSold", "OperatingExpenses", "LaborAndRelatedExpense",
         "SalariesAndWages", "InterestExpense", "DepreciationDepletionAndAmortization",
         "PaymentsToAcquirePropertyPlantAndEquipment", "SellingGeneralAndAdministrativeExpense"]
STOCKS = ["Assets", "PropertyPlantAndEquipmentNet", "StockholdersEquity", "EntityNumberOfEmployees"]
TAGS = set(FLOWS + STOCKS)


def quarters():
    out = [f"2009q{q}" for q in (2, 3, 4)]
    for y in range(2010, 2027):
        for q in range(1, 5):
            if (y, q) <= (2026, 2):
                out.append(f"{y}q{q}")
    return out


def fetch(q):
    url = f"https://www.sec.gov/files/dera/data/financial-statement-data-sets/{q}.zip"
    for k in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            return urllib.request.urlopen(req, timeout=900).read(), url
        except Exception as exc:  # noqa: BLE001
            print("retry", q, exc, flush=True)
            time.sleep(2 ** (k + 1))
    raise RuntimeError(q)


def extract(q, force=False):
    out = RAW / f"sec_fy_{q}.csv"
    if out.exists() and not force:
        return
    data, url = fetch(q)
    z = zipfile.ZipFile(io.BytesIO(data))
    sub = pd.read_csv(z.open("sub.txt"), sep="\t", dtype=str, quoting=csv.QUOTE_NONE,
                      usecols=["adsh", "cik", "name", "sic", "countryba", "stprba", "form", "fy", "fp", "period", "filed"])
    sub = sub[sub.form.isin(["10-K", "10-K/A"]) & (sub.fp == "FY")]
    keep = set(sub.adsh)
    parts = []
    for ch in pd.read_csv(z.open("num.txt"), sep="\t", dtype=str, chunksize=2_000_000, quoting=csv.QUOTE_NONE,
                          usecols=["adsh", "tag", "coreg", "segments", "ddate", "qtrs", "uom", "value"]):
        ch = ch[ch.adsh.isin(keep) & ch.tag.isin(TAGS) & ch.coreg.isna() & ch.segments.isna()]
        parts.append(ch)
    num = pd.concat(parts).merge(sub, on="adsh")
    num = num[num.ddate == num.period]
    isflow = num.tag.isin(FLOWS)
    num = num[(isflow & (num.qtrs == "4")) | (~isflow & (num.qtrs == "0"))]
    num = num[(num.uom == "USD") | (num.tag == "EntityNumberOfEmployees")]
    key = ["adsh", "tag", "ddate", "qtrs", "uom"]
    nv = num.groupby(key).value.transform("nunique")
    n_conf = int((nv > 1).sum())
    num = num[nv == 1].drop_duplicates(key)
    num.drop(columns=["coreg", "segments"]).to_csv(out, index=False)
    with open(ROOT / "data" / "raw" / "MANIFEST.csv", "a", newline="") as f:
        csv.writer(f).writerow(["v10", f"sec/sec_fy_{q}.csv", url, "public domain (SEC)",
                                "iteration 10, extract of 10-K FY values (archive not kept)", len(data),
                                hashlib.sha256(data).hexdigest(), dt.datetime.now().isoformat(timespec="seconds")])
    print("sec", q, len(sub), len(num), "conflicting rows dropped:", n_conf, flush=True)


def merge():
    fr = [pd.read_csv(f, dtype=str) for f in sorted(glob.glob(str(RAW / "sec_fy_*.csv")))]
    d = pd.concat(fr, ignore_index=True)
    d["value"] = pd.to_numeric(d.value, errors="coerce")
    d = d.sort_values("filed").groupby(["cik", "fy", "tag"]).last().reset_index()
    meta = d.sort_values("filed").groupby(["cik", "fy"])[["name", "sic", "countryba", "stprba", "period"]].last()
    w = d.pivot_table(index=["cik", "fy"], columns="tag", values="value", aggfunc="last")
    w = meta.join(w).reset_index()
    w.to_csv(RAW / "sec_firm_fy_all.csv", index=False)
    print("merged", w.shape, flush=True)
    return w


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else "build"
    if part in ("build", "rebuild"):
        for q in quarters():
            try:
                extract(q, force=part == "rebuild")
            except Exception as exc:  # noqa: BLE001
                print("FAILED", q, exc, flush=True)
        merge()
    else:
        merge()
