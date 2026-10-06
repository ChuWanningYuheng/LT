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


def _main():
    import sys
    out = ROOT / "results" / "v7"
    if len(sys.argv) > 1 and sys.argv[1] == "threshold":
        r, c, pv = threshold_v()
        r.to_csv(out / "p1_sec_threshold.csv", index=False)
        c.to_csv(out / "p1_sec_curves.csv", index=False)
        pv.to_csv(out / "p1_sec_profile.csv", index=False)
        print(r.round(3).to_string())
        return
    firms = build()
    g = by_sic2(firms)
    g.to_csv(out / "p1_sec_sic2.csv", index=False)
    print(g.groupby("sic2").sga_ppe.median().round(2).to_string())


# ---------------------------------------------------------------- profile (v): SIC -> KLEMS, threshold on US rows
SIC3 = {271: "J58-J60", 272: "J58-J60", 273: "J58-J60", 274: "J58-J60", 731: "M", 365: "C26", 366: "C26",
        384: "C31-C33", 385: "C31-C33",                                  # review R13
        283: "C21", 357: "C26", 367: "C26", 737: "J62-J63", 481: "J61", 482: "J61", 483: "J58-J60", 484: "J58-J60",
        551: "G45", 552: "G45", 553: "G45", 555: "G45", 556: "G45", 557: "G45", 581: "I", 701: "I", 704: "I"}
SIC2 = {**{k: "A" for k in range(1, 10)}, **{k: "B" for k in range(10, 15)}, **{k: "F" for k in range(15, 18)},
        20: "C10-C12", 21: "C10-C12", 22: "C13-C15", 23: "C13-C15", 31: "C13-C15", 24: "C16-C18", 26: "C16-C18",
        27: "C16-C18", 25: "C31-C33", 39: "C31-C33", 28: "C20", 29: "C19", 30: "C22-C23", 32: "C22-C23",
        33: "C24-C25", 34: "C24-C25", 35: "C28", 36: "C27", 37: "C29-C30", 38: "C26",
        40: "H49", 41: "H49", 42: "H49", 46: "H49", 44: "H50", 45: "H51", 47: "H52", 43: "H53", 48: "J61",
        50: "G46", 51: "G46", **{k: "G47" for k in range(52, 60)}, 58: "I", 70: "I", 72: "S", 73: "N", 75: "G45",
        76: "S", 78: "J58-J60", 79: "R", 81: "M", 87: "M"}


def sic_to_klems(sic):
    return SIC3.get(int(sic) // 10, SIC2.get(int(sic) // 100))


def profile_v(firms, epw_params):
    f = firms.copy()
    sga = f.get("SellingGeneralAndAdministrativeExpense")
    alt = f.get("GeneralAndAdministrativeExpense", 0).fillna(0) + f.get("SellingAndMarketingExpense", 0).fillna(0)
    f["SGA"] = sga.where(sga.notna(), alt.where(alt > 0))
    f["PPE"] = f.PropertyPlantAndEquipmentNet
    f["sic"] = pd.to_numeric(f.sic, errors="coerce")
    f = f[f.SGA.notna() & (f.SGA > 0) & (f.PPE > 0) & f.sic.notna()]
    f = f[~f.sic.between(6000, 6799) & ~f.sic.between(4900, 4999)]
    f["ind"] = f.sic.map(sic_to_klems)
    g = epw_params.set_index("sic").gamma
    f["gamma_epw"] = f.sic.astype(int).map(g)
    f["gamma_epw"] = f.gamma_epw.fillna(epw_params.gamma.median())
    f = f[f.ind.notna()]
    agg = f.groupby(["fy", "ind"]).apply(lambda s: pd.Series(dict(
        SGA=s.SGA.sum(), PPE=s.PPE.sum(), gSGA=(s.gamma_epw * s.SGA).sum(), n=len(s))), include_groups=False)
    return agg.reset_index()


def threshold_v():
    from .part1 import klems_panel, slope, threshold
    firms = pd.read_csv(RAW / "sec_firm_fy.csv")
    epw = pd.read_csv(ROOT / "data" / "raw" / "v7" / "epw" / "capital_accum_parameters_2023.csv")
    pv = profile_v(firms, epw)
    prof = pv.groupby("ind")[["SGA", "PPE", "gSGA"]].sum()                 # pooled over the four fiscal years
    p = klems_panel()
    us = p[p.geo == "US"].copy()
    us = us.merge(prof, left_on="ind", right_index=True, how="inner")
    us["q_klems"] = us.K_nonNA / us.K
    res, curves = [], []
    b0, p0, _ = slope(us.assign(_r=us.r, _x=us.x0), "_r", "_x")
    for gname in ("0.3", "0.5", "1.0", "EPW"):
        for dl in (0.05, 0.15, 0.20):
            num = us.gSGA if gname == "EPW" else float(gname) * us.SGA
            q = num / (0.04 + dl) / us.PPE                                  # O / PPE as U / K
            fn = (lambda qq: (lambda d, m: ((d.PI + 0.04 * m * qq.loc[d.index] * d.K) / (d.K * (1 + m * qq.loc[d.index])),
                                            np.log(d.K * (1 + m * qq.loc[d.index]) / d.W))))(q)
            t, m0, ms = threshold(us, fn)
            curves.append(t.assign(gamma=gname, delta=dl))
            res.append(dict(gamma=gname, delta=dl, beta0=b0, p0=p0, m_zero=m0, m_insig=ms, n=len(us),
                            median_U_over_K=float(q.groupby(us.ind).first().median()),
                            corr_q_sec_vs_klems=float(np.corrcoef(q.groupby(us.ind).first(),
                                                                  us.groupby("ind").q_klems.mean())[0, 1]),
                            corr_q_sec_lnKW=float(np.corrcoef(q.groupby(us.ind).first(),
                                                              us.groupby("ind").x0.mean())[0, 1])))
    return pd.DataFrame(res), pd.concat(curves), pv


if __name__ == "__main__":
    _main()
