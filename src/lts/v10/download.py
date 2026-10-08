"""Iteration 10, stage 0: downloads into data/raw/v10/ and rows in data/raw/MANIFEST.csv (group v10).

Usage: PYTHONPATH=src python -m lts.v10.download [name ...]   (no names: everything not yet present)
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import sys
import time
import urllib.request

from ..v6.series import ROOT

RAW = ROOT / "data" / "raw" / "v10"
RAW.mkdir(parents=True, exist_ok=True)
UA = "LT-research-script/1.0 (noreply@anthropic.com)"
ES = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id="
WB = "https://api.worldbank.org/v2/country/all/indicator/"
WAY = "https://web.archive.org/web/"
BEA = "public domain (BEA)"

FILES = [
    # stage 1
    ("bea_SAGDP.zip", "https://apps.bea.gov/regional/zip/SAGDP.zip", BEA, "BEA Regional, state GDP and components (annual)"),
    ("bea_SQGDP.zip", "https://apps.bea.gov/regional/zip/SQGDP.zip", BEA, "BEA Regional, state GDP (quarterly)"),
    ("fred_MCOILWTICO.csv", FRED + "MCOILWTICO", "FRED terms", "WTI crude oil price, monthly"),
    ("fred_PNGASEUUSDM.csv", FRED + "PNGASEUUSDM", "FRED terms", "natural gas price, Europe (IMF), monthly"),
    ("fred_UNRATE.csv", FRED + "UNRATE", "FRED terms", "US unemployment rate, monthly"),
    ("es_namq_10_a10_B1G.json", ES + "namq_10_a10?format=JSON&na_item=B1G&unit=CP_MEUR&s_adj=SCA", "Eurostat (CC BY 4.0)",
     "quarterly GVA by A*10, current prices"),
    ("es_namq_10_a10_D1.json", ES + "namq_10_a10?format=JSON&na_item=D1&unit=CP_MEUR&s_adj=SCA", "Eurostat (CC BY 4.0)",
     "quarterly compensation of employees by A*10"),
    ("es_nama_10_a64_B2A3G.json", ES + "nama_10_a64?format=JSON&na_item=B2A3G&unit=CP_MEUR", "Eurostat (CC BY 4.0)",
     "gross operating surplus and mixed income by A*64, annual"),
    ("bea_SAGDP_SIC.zip", "https://apps.bea.gov/regional/zip/SAGDP_SIC.zip", BEA, "BEA Regional, state GDP and components, SIC basis"),
    ("bh_oil_supply_shocks.xlsx", "https://drive.google.com/uc?export=download&id=1OsA8btgm2rmDucUFngiLkwv4uywTDmya",
     "authors' website (Baumeister)", "Baumeister & Hamilton (2019) monthly structural oil supply shocks"),
    ("bh_oil_demand_shocks.xlsx", "https://drive.google.com/uc?export=download&id=1neFXLrIvGwggebQRwjmtrWK-dfQZ9NH8",
     "authors' website (Baumeister)", "Baumeister & Hamilton (2019) monthly structural oil demand shocks"),
    # stage 2
    ("nlrb.db", "https://labordata.bunkum.us/nlrb.db", "public records (NLRB) compiled by labordata.bunkum.us",
     "NLRB case data 2010+, SQLite (elections, tallies, participants)"),
    # stage 6
    ("es_tus_00selfstat.json", ES + "tus_00selfstat?format=JSON&lang=EN", "Eurostat (CC BY 4.0)",
     "HETUS 2000/2010: time use by self-declared labour status"),
    ("es_tus_00age.json", ES + "tus_00age?format=JSON&lang=EN", "Eurostat (CC BY 4.0)", "HETUS 2000/2010: time use by age"),
    ("bea_household_production_expanded.xlsx",
     "https://www.bea.gov/sites/default/files/2026-06/household-production-expanded-indicators-data.xlsx", BEA,
     "BEA household production expanded indicators 1965-2024 (Bridgman, WP2026-15)"),
    ("bea_household_production_2022.xlsx", "https://www.bea.gov/sites/default/files/2022-02/household-production-data.xlsx",
     BEA, "BEA value of household production 1947-2020 (SCB February 2022)"),
    # stage 9-11
    ("oecd_bsdb.dta", WAY + "20220414224011id_/http://fmwww.bc.edu/ec-p/data/oecd/bsdb.dta",
     "OECD data via Boston College archive (Wayback)", "OECD Business Sector Data Base (BSDB), Stata"),
    ("euklems09_all_countries_09I.zip", WAY + "20211105162150id_/http://euklems.net/data/09i/all_countries_09I.zip",
     "EU KLEMS (euklems.net, Wayback)", "EU KLEMS November 2009 release, basic files, all countries"),
    ("euklems09_all_capital_09I.zip", WAY + "20211105172643id_/http://euklems.net/data/09i/all_capital_09I.zip",
     "EU KLEMS (euklems.net, Wayback)", "EU KLEMS November 2009 release, capital input files"),
    ("oecd_isdb.dta", WAY + "20220308051535id_/http://fmwww.bc.edu/ec-p/data/oecd/oecd_isdb.dta",
     "OECD data via Boston College archive (Wayback)", "OECD International Sectoral Database (ISDB), Stata"),
    # stage 10
    ("wb_CMO_Historical_Data_Monthly.xlsx",
     "https://thedocs.worldbank.org/en/doc/74e8be41ceb20fa0da750cda2f6b9e4e-0050012026/related/CMO-Historical-Data-Monthly.xlsx",
     "World Bank (CC BY 4.0)", "World Bank Pink Sheet, monthly commodity prices"),
    ("wb_CMO_Historical_Data_Annual.xlsx",
     "https://thedocs.worldbank.org/en/doc/74e8be41ceb20fa0da750cda2f6b9e4e-0050012026/related/CMO-Historical-Data-Annual.xlsx",
     "World Bank (CC BY 4.0)", "World Bank Pink Sheet, annual commodity prices"),
    ("bis_WS_SPP_csv_flat.zip", "https://data.bis.org/static/bulk/WS_SPP_csv_flat.zip", "BIS terms of use",
     "BIS selected residential property prices"),
    ("wb_NY.GDP.TOTL.RT.ZS.json", WB + "NY.GDP.TOTL.RT.ZS?format=json&per_page=20000", "World Bank (CC BY 4.0)",
     "total natural resources rents, % of GDP"),
    # stage 6
    # stage 8 (added with journal 12)
    ("fred_WPU11.csv", FRED + "WPU11", "FRED terms", "US PPI by commodity: machinery and equipment, monthly"),
    ("bls_atussum-0324.zip", "https://www.bls.gov/tus/datafiles/atussum-0324.zip", "public domain (BLS)",
     "ATUS 2003-2024 activity summary file"),
]


def fetch(name, url, licence, note, tries=4):
    out = RAW / name
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=900) as r, open(out, "wb") as f:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
            break
        except Exception as e:                                   # noqa: BLE001
            print("retry", name, e, flush=True)
            time.sleep(2 ** (k + 2))
    else:
        print("FAILED", name, url, flush=True)
        if out.exists():
            out.unlink()
        return False
    data = out.read_bytes()
    with open(ROOT / "data" / "raw" / "MANIFEST.csv", "a", newline="") as f:
        csv.writer(f).writerow(["v10", name, url, licence, f"iteration 10, {note}", len(data),
                                hashlib.sha256(data).hexdigest(), dt.datetime.now().isoformat(timespec="seconds")])
    print("ok", name, len(data), flush=True)
    return True


if __name__ == "__main__":
    want = set(sys.argv[1:])
    for name, url, lic, note in FILES:
        if (want and name not in want) or (not want and (RAW / name).exists()):
            continue
        fetch(name, url, lic, note)
