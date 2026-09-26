"""Download all raw data used in the study.

Every source is listed in SOURCES with its URL, licence and a short note. Files are
written to data/raw/<group>/ and a manifest (size, sha256, download time) is written to
data/raw/MANIFEST.csv so that results can be tied to exact file versions.

Usage:  python -m lts.download [--only figaro,oecd,bea] [--force]

Note on WIOD 2016: the WIOD files are hosted on dataverse.nl, which (Sept 2026) answers
automated requests with a "BotStopper: Access Denied" page. We did not try to circumvent
this. FIGARO + OECD National Accounts + BEA replace it (see README / report).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"

EUROSTAT_BULK = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/{ds}?format=TSV&compressed=true"
OECD_SDMX = "https://sdmx.oecd.org/public/rest/data/{flow},/{key}?{params}format=csv"

# OECD countries for which national labour (hours) + capital consumption data are pulled.
# ISO3 codes. The focus countries are USA, DEU, MEX; the others are used for robustness.
OECD_COUNTRIES = ["USA", "DEU", "MEX", "FRA", "ITA", "ESP", "NLD", "AUT", "POL", "CZE",
                  "KOR", "JPN", "TUR", "GBR"]

SOURCES = []

# --- Eurostat FIGARO inter-country industry-by-industry IO tables (2026 edition) -------------
for i, yrs in enumerate(["2010-2013", "2014-2017", "2018-2021", "2022-2024"], start=1):
    ds = f"naio_10_fcp_ii{i}"
    SOURCES.append(dict(group="figaro", file=f"{ds}.tsv.gz", url=EUROSTAT_BULK.format(ds=ds),
                        licence="Eurostat re-use policy (CC BY 4.0)",
                        note=f"FIGARO inter-country IO table, industry by industry, current prices, MIO_EUR, {yrs}"))

# --- OECD National Accounts (SNA 2008) --------------------------------------------------------
for c in OECD_COUNTRIES:
    SOURCES.append(dict(group="oecd", file=f"table6_{c}.csv",
                        url=OECD_SDMX.format(flow="OECD.SDD.NAD,DSD_NAMAIN10@DF_TABLE6",
                                             key=f"A.{c}..........", params="startPeriod=1995&"),
                        licence="OECD Terms and Conditions (CC BY 4.0 since 2024)",
                        note="Table 6: output, value added and components (D1, P51C, B2A3G/N), current & volume, by ISIC4 activity"))
    SOURCES.append(dict(group="oecd", file=f"table7_{c}.csv",
                        url=OECD_SDMX.format(flow="OECD.SDD.NAD,DSD_NAMAIN10@DF_TABLE7",
                                             key=f"A.{c}..........", params="startPeriod=1995&"),
                        licence="OECD Terms and Conditions (CC BY 4.0 since 2024)",
                        note="Table 7: employment (persons, jobs, hours) by ISIC4 activity"))
    SOURCES.append(dict(group="oecd", file=f"stan2025_{c}.csv",
                        url=OECD_SDMX.format(flow="OECD.STI.PIE,DSD_STAN@DF_STAN_2025",
                                             key=f"A.{c}....", params="startPeriod=1995&"),
                        licence="OECD Terms and Conditions (CC BY 4.0 since 2024)",
                        note="STAN 2025: employees/self-employed/hours by ISIC4 activity (fallback when Table 7 lacks them)"))
SOURCES.append(dict(group="oecd", file="timbc_2025_empn.csv",
                    url=OECD_SDMX.format(flow="OECD.STI.PIE,DSD_TIMBC_2025@DF_TIMBC_2025",
                                         key="EMPN...W.PS.A.", params=""),
                    licence="OECD Terms and Conditions (CC BY 4.0 since 2024)",
                    note="TiMBC 2025: employment by industry (ICIO aggregates) and workforce characteristic (ISCED/ISCO/age/sex)"))

# --- US BEA -----------------------------------------------------------------------------------
SOURCES += [
    dict(group="bea", file="AllTablesIO.zip",
         url="https://apps.bea.gov/industry/iTables%20Static%20Files/AllTablesIO.zip",
         licence="US Government work, public domain",
         note="BEA annual IO: Make/Use before redefinitions, producer prices, summary level, 1997-2023"),
    dict(group="bea", file="ImportMatrices_Before_Redefinitions_SUM_1997-2023.xlsx",
         url="https://apps.bea.gov/industry/xls/io-annual/ImportMatrices_Before_Redefinitions_SUM_1997-2023.xlsx",
         licence="US Government work, public domain", note="BEA import matrices, summary"),
    dict(group="bea", file="GrossOutput.xlsx",
         url="https://apps.bea.gov/industry/Release/XLS/GDPxInd/GrossOutput.xlsx",
         licence="US Government work, public domain", note="BEA GDP by industry: gross output and chain-type price indexes"),
    dict(group="bea", file="NipaDataA.txt", url="https://apps.bea.gov/national/Release/TXT/NipaDataA.txt",
         licence="US Government work, public domain", note="NIPA annual flat file (Table 6.5D FTE employees by industry)"),
    dict(group="bea", file="SeriesRegister.txt", url="https://apps.bea.gov/national/Release/TXT/SeriesRegister.txt",
         licence="US Government work, public domain", note="NIPA series register"),
]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path, tries: int = 3) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for k in range(tries):
        r = subprocess.run(["curl", "-sS", "-L", "--fail", "-m", "3600", "-A", "Mozilla/5.0 (research script)",
                            "-o", str(tmp), url])
        if r.returncode == 0 and tmp.stat().st_size > 1000:
            tmp.rename(dest)
            return
        time.sleep(2 ** (k + 1))
    raise RuntimeError(f"download failed: {url}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="figaro,oecd,bea")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    groups = set(a.only.split(","))
    rows = []
    for s in SOURCES:
        dest = RAW / s["group"] / s["file"]
        if s["group"] in groups and (a.force or not dest.exists()):
            print(f"-> {s['group']}/{s['file']}", file=sys.stderr)
            try:
                fetch(s["url"], dest)
            except RuntimeError as e:  # e.g. OECD Table 7 does not exist for TUR
                if s["group"] == "oecd":
                    print(f"   WARNING: {e} (skipped; country dropped where needed)", file=sys.stderr)
                    continue
                raise
        if not dest.exists():
            continue
        rows.append([s["group"], s["file"], s["url"], s["licence"], s["note"], dest.stat().st_size, sha256(dest),
                     dt.datetime.fromtimestamp(dest.stat().st_mtime).isoformat(timespec="seconds")])
    import csv
    with open(RAW / "MANIFEST.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "file", "url", "licence", "note", "bytes", "sha256", "retrieved"])
        w.writerows(rows)
    print(f"manifest: {len(rows)} files", file=sys.stderr)


if __name__ == "__main__":
    main()
