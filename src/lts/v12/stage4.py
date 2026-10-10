"""Iteration 12, stage 4: the labour process — deskilling (Braverman) vs polarisation (pre_registration_v12.md, stage 4).

O*NET versions 5.0 (2003), 10.0 (2006), 15.0 (2010), 20.1 (2015-16), 25.0 (2020), 30.0 (2025):
within-occupation change of autonomy (Freedom to Make Decisions, Structured versus Unstructured Work), pace control
(Pace Determined by Speed of Equipment) and routine (Importance of Repeating Same Tasks), CX scale, occupations whose
rating date differs between versions; standardised by the cross-occupation SD in the base version.
Between occupations: OEWS May 2004 vs May 2023 employment shares by Job Zone (O*NET 10.0 and 30.0).
Outcomes 87-89.

Usage: PYTHONPATH=src python -P -m lts.v12.stage4
Outputs: results/v12/s4_*.csv
"""
from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd

from ..v6.series import ROOT
from ..v8.rule import decide

OUT = ROOT / "results" / "v12"
R12 = ROOT / "data" / "raw" / "v12"
VERS = {"5.0": ("onet_db_50.zip", "db_50/WorkContext.txt", "db_50/onetsoc_job_zones.txt"),
        "10.0": ("onet_db_10_0.zip", "db_10_0/Work Context.txt", "db_10_0/Job Zones.txt"),
        "15.0": ("onet_db_15_0.zip", "db_15_0/Work Context.txt", "db_15_0/Job Zones.txt"),
        "20.1": ("onet_db_20_1_text.zip", "db_20_1_text/Work Context.txt", "db_20_1_text/Job Zones.txt"),
        "25.0": ("onet_db_25_0_text.zip", "db_25_0_text/Work Context.txt", "db_25_0_text/Job Zones.txt"),
        "30.0": ("onet_db_30_0_text.zip", "db_30_0_text/Work Context.txt", "db_30_0_text/Job Zones.txt")}
EL = {"4.C.3.a.4": "freedom", "4.C.3.b.8": "structured", "4.C.3.d.3": "pace", "4.C.3.b.7": "repeat"}


def read(zipname, path):
    z = zipfile.ZipFile(R12 / zipname)
    return pd.read_csv(io.StringIO(z.open(path).read().decode("latin-1")), sep="\t", low_memory=False)


def work_context(v):
    f, wc, _ = VERS[v]
    d = read(f, wc)
    d = d[(d["Scale ID"] == "CX") & d["Element ID"].isin(EL)]
    d = d.assign(item=d["Element ID"].map(EL))
    src = "Domain Source" if "Domain Source" in d else "Source"
    out = d.pivot_table(index="O*NET-SOC Code", columns="item", values="Data Value", aggfunc="first")
    dates = d.pivot_table(index="O*NET-SOC Code", columns="item", values="Date", aggfunc="first")
    srcs = d.pivot_table(index="O*NET-SOC Code", columns="item", values=src, aggfunc="first")
    return out, dates, srcs


def job_zones(v):
    f, _, jz = VERS[v]
    d = read(f, jz)
    return d.set_index("O*NET-SOC Code")["Job Zone"]


def boot_mean(v, B=2000, seed=40):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    rng = np.random.default_rng(seed)
    m = v[rng.integers(0, len(v), (B, len(v)))].mean(1)
    return float(v.mean()), float(np.quantile(m, 0.05)), float(np.quantile(m, 0.95)), len(v)


def within(v0, v1, rerated=True):
    a, da, sa = work_context(v0)
    b, db, sb = work_context(v1)
    common = a.index.intersection(b.index)
    rows = {}
    for item in EL.values():
        if item not in a or item not in b:
            continue
        sd = a[item].std()
        ch = (b.loc[common, item] - a.loc[common, item]) / sd
        if rerated:
            ch = ch[(db.loc[common, item].astype(str) != da.loc[common, item].astype(str))]
        rows[item] = ch
    C = pd.DataFrame(rows)
    if "structured" in C:
        C["autonomy"] = C[["freedom", "structured"]].mean(1, skipna=False)
    else:
        C["autonomy"] = C["freedom"]
    jz0, jz1 = job_zones(v0), job_zones(v1)
    jc = jz0.index.intersection(jz1.index)
    C = C.join((jz1.loc[jc] - jz0.loc[jc]).rename("job_zone"), how="left")
    return C, len(common)


def oews(year):
    if year == 2004:
        z = zipfile.ZipFile(R12 / "bls_oesm04nat.zip")
        d = pd.read_excel(z.open("national_may2004_dl.xls"))
    else:
        z = zipfile.ZipFile(R12 / "bls_oesm23nat.zip")
        d = pd.read_excel(z.open("oesm23nat/national_M2023_dl.xlsx"))
    d.columns = [c.upper() for c in d.columns]
    grp = next(c for c in ("O_GROUP", "OCC_GROUP", "GROUP") if c in d)  # 2004: "group", empty for detailed rows
    d = d[d[grp].fillna("detailed").astype(str).str.lower().eq("detailed") & d.OCC_CODE.astype(str).str.match(r"^\d\d-\d{4}$")]
    d = d[~d.OCC_CODE.str.endswith("0000")]
    d["emp"] = pd.to_numeric(d.TOT_EMP, errors="coerce")
    return d[["OCC_CODE", "emp"]].dropna()


def between():
    rows = []
    for year, v in ((2004, "10.0"), (2023, "30.0")):
        o = oews(year)
        jz = job_zones(v)
        soc = jz.groupby(jz.index.str[:7]).mean().round()
        o["jz"] = o.OCC_CODE.map(soc)
        cov = o.emp[o.jz.notna()].sum() / o.emp.sum()
        o = o.dropna(subset=["jz"])
        rows.append(dict(year=year, version=v, coverage=float(cov), n_occ=len(o),
                         **{f"share_jz{int(k)}": float(o.emp[o.jz == k].sum() / o.emp.sum()) for k in range(1, 6)}))
        o.assign(year=year).to_csv(OUT / f"s4_oews_{year}.csv", index=False)
    S = pd.DataFrame(rows)
    # bootstrap of the change in JZ-3 share (occupations resampled within each year)
    a = pd.read_csv(OUT / "s4_oews_2004.csv")
    b = pd.read_csv(OUT / "s4_oews_2023.csv")
    rng = np.random.default_rng(89)
    def share3(d, idx):
        e = d.emp.to_numpy()[idx]
        return e[d.jz.to_numpy()[idx] == 3].sum() / e.sum()
    bs = [share3(b, rng.integers(0, len(b), len(b))) - share3(a, rng.integers(0, len(a), len(a))) for _ in range(2000)]
    est = float(S.share_jz3.iloc[1] - S.share_jz3.iloc[0])
    lo, hi = np.quantile(bs, [0.05, 0.95])
    return S, dict(outcome="89: change of the employment share of Job Zone 3 (middle), OEWS 2004 -> 2023", est=est,
                   ci90_lo=float(lo), ci90_hi=float(hi), n=len(a) + len(b), theta0=0, delta=0.02, direction="<",
                   label=decide(est, lo, hi, 0, 0.02, "<"))


def run():
    rows, desc = [], []
    for v0, v1, main in (("15.0", "30.0", True), ("10.0", "15.0", False), ("20.1", "30.0", False), ("5.0", "30.0", False)):
        C, ncommon = within(v0, v1)
        C.to_csv(OUT / f"s4_within_{v0}_{v1}.csv")
        for item in ("autonomy", "freedom", "structured", "pace", "repeat", "job_zone"):
            if item not in C:
                continue
            est, lo, hi, n = boot_mean(C[item])
            r = dict(item=f"{v0} -> {v1}: mean within-occupation change of {item} (SD units; re-rated occupations)",
                     est=est, ci90_lo=lo, ci90_hi=hi, n=n, common_codes=ncommon)
            desc.append(r)
            if main and item == "autonomy":
                rows.append(dict(outcome="87: mean within-occupation change of autonomy, O*NET 15.0 -> 30.0", est=est,
                                 ci90_lo=lo, ci90_hi=hi, n=n, theta0=0, delta=0.1, direction="<",
                                 label=decide(est, lo, hi, 0, 0.1, "<")))
            if main and item == "pace":
                rows.append(dict(outcome="88: mean within-occupation change of pace control, O*NET 15.0 -> 30.0", est=est,
                                 ci90_lo=lo, ci90_hi=hi, n=n, theta0=0, delta=0.1, direction=">",
                                 label=decide(est, lo, hi, 0, 0.1, ">")))
        print(v0, v1, ncommon, flush=True)
    # sources of ratings by version (comparability)
    src = []
    for v in VERS:
        _, _, s = work_context(v)
        if "freedom" in s:
            src.append(dict(version=v, **s["freedom"].value_counts(normalize=True).round(3).to_dict()))
    pd.DataFrame(src).to_csv(OUT / "s4_rating_sources.csv", index=False)
    S, r89 = between()
    S.to_csv(OUT / "s4_jobzone_shares.csv", index=False)
    rows.append(r89)
    O = pd.DataFrame(rows)
    O.to_csv(OUT / "s4_outcomes.csv", index=False)
    D = pd.DataFrame(desc)
    D.to_csv(OUT / "s4_descriptive.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 110)
    print(O.round(4).to_string())
    print(D.round(4).to_string())
    print(S.round(4).to_string())
    print(pd.DataFrame(src).to_string())


if __name__ == "__main__":
    run()
