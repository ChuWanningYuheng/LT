"""Iteration 12, stage 0: downloads into data/raw/v12/ and rows in data/raw/MANIFEST.csv (group v12).

Usage: PYTHONPATH=src python -P -m lts.v12.download [name ...]   (no names: everything not yet present)
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import sys
import time
import urllib.request

from ..v6.series import ROOT

RAW = ROOT / "data" / "raw" / "v12"
RAW.mkdir(parents=True, exist_ok=True)
UA = "LT-research-script/1.0 (noreply@anthropic.com)"
ES = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
OECD = "https://sdmx.oecd.org/public/rest/data/"
WB = "https://api.worldbank.org/v2/country/all/indicator/{}?format=json&per_page=20000&date=1960:2024"
ONET = "https://www.onetcenter.org/dl_files/"
OES = "https://www.bls.gov/oes/special-requests/"

FILES = [
    ("es_nasa_10_nf_tr_S11_property.json",
     ES + "nasa_10_nf_tr?format=JSON&lang=EN&sector=S11&unit=CP_MNAC&na_item=D4&na_item=D42&na_item=D43&na_item=D44"
          "&na_item=B2A3N&na_item=D1&direct=RECV&direct=PAID", "Eurostat (CC BY 4.0)",
     "non-financial corporations: property income (D4, D42, D43 reinvested earnings on FDI, D44), NOS, compensation"),
    ("oecd_fdi_inc_aggr.csv", OECD + "OECD.DAF.INV,DSD_FDI@DF_FDI_INC_AGGR,/.T_D4P_F+T_D43S_F5+T_D42S_F5.USD_EXC.DO+DI...D.S1.W.._T.A"
     "?format=csvfilewithlabels",
     "OECD (CC BY 4.0)", "FDI income (total, reinvested earnings, dividends), USD, world counterpart, BMD4"),
    ("oecd_cit.csv", OECD + "OECD.CTP.TPS,DSD_TAX_CIT@DF_CIT,/all?format=csvfilewithlabels", "OECD (CC BY 4.0)",
     "corporate income tax statutory rates (Corporate Tax Statistics)"),
    ("oecd_epl.csv", OECD + "OECD.ELS.JAI,DSD_EPL@DF_EPL,/all?format=csvfilewithlabels", "OECD (CC BY 4.0)",
     "strictness of employment protection (EPL)"),
    ("wb_NV.AGR.TOTL.ZS.json", WB.format("NV.AGR.TOTL.ZS"), "World Bank (CC BY 4.0)", "agriculture value added, % GDP"),
    ("wb_NV.IND.TOTL.ZS.json", WB.format("NV.IND.TOTL.ZS"), "World Bank (CC BY 4.0)", "industry value added, % GDP"),
    ("wb_NE.EXP.GNFS.ZS.json", WB.format("NE.EXP.GNFS.ZS"), "World Bank (CC BY 4.0)", "exports of goods and services, % GDP"),
    ("cepii_geo_cepii.xls", "https://www.cepii.fr/distance/geo_cepii.xls", "CEPII (Etalab open licence)",
     "GeoDist country file (landlocked, area, coordinates)"),
    ("dorn_occ1990dd_task_alm.zip", "https://www.ddorn.net/data/occ1990dd_task_alm.zip", "D. Dorn data page",
     "Autor-Levy-Murnane task measures (DOT) by occ1990dd"),
    ("onet_db_50.zip", ONET + "db_50.zip", "O*NET (CC BY 4.0)", "O*NET 5.0 database"),
    ("onet_db_10_0.zip", ONET + "db_10_0.zip", "O*NET (CC BY 4.0)", "O*NET 10.0 database"),
    ("onet_db_15_0.zip", ONET + "db_15_0.zip", "O*NET (CC BY 4.0)", "O*NET 15.0 database"),
    ("onet_db_20_1_text.zip", ONET + "database/db_20_1_text.zip", "O*NET (CC BY 4.0)", "O*NET 20.1 database"),
    ("onet_db_25_0_text.zip", ONET + "database/db_25_0_text.zip", "O*NET (CC BY 4.0)", "O*NET 25.0 database"),
    ("onet_db_30_0_text.zip", ONET + "database/db_30_0_text.zip", "O*NET (CC BY 4.0)", "O*NET 30.0 database"),
    ("bls_oesm04nat.zip", OES + "oesm04nat.zip", "BLS (public domain)", "OEWS national, May 2004"),
    ("bls_oesm10nat.zip", OES + "oesm10nat.zip", "BLS (public domain)", "OEWS national, May 2010"),
    ("bls_oesm16nat.zip", OES + "oesm16nat.zip", "BLS (public domain)", "OEWS national, May 2016"),
    ("bls_oesm23nat.zip", OES + "oesm23nat.zip", "BLS (public domain)", "OEWS national, May 2023"),
    ("bls_soc_2000_to_2010_crosswalk.xls", "https://www.bls.gov/soc/soc_2000_to_2010_crosswalk.xls", "BLS (public domain)",
     "SOC 2000 to 2010 crosswalk"),
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
        csv.writer(f).writerow(["v12", name, url, licence, f"iteration 12, {note}", len(data),
                                hashlib.sha256(data).hexdigest(), dt.datetime.now().isoformat(timespec="seconds")])
    print("ok", name, len(data), flush=True)
    return True


if __name__ == "__main__":
    want = set(sys.argv[1:])
    for name, url, lic, note in FILES:
        if (want and name not in want) or (not want and (RAW / name).exists()):
            continue
        fetch(name, url, lic, note)
