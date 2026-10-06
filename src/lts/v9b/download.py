"""Iteration 9b, stage 0: downloads into data/raw/v9b/ and rows in data/raw/MANIFEST.csv (group v9b).

Usage: PYTHONPATH=src python -m lts.v9b.download [name ...]   (no names: everything not yet present)
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import sys
import time
import urllib.request

from ..v6.series import ROOT

RAW = ROOT / "data" / "raw" / "v9b"
RAW.mkdir(parents=True, exist_ok=True)
UA = "LT-research-script/1.0 (noreply@anthropic.com)"
ES = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
BEA = "public domain (BEA)"

FILES = [
    ("bea_GDPbyInd_VA_SIC.xls", "https://apps.bea.gov/industry/xls/GDPbyInd_VA_SIC.xls", BEA,
     "BEA GDP by industry, SIC basis: value added, components, employment, 1947-1997"),
    ("bea_NipaDataA.txt", "https://apps.bea.gov/national/Release/TXT/NipaDataA.txt", BEA, "NIPA annual, all series"),
    ("bea_SeriesRegister.txt", "https://apps.bea.gov/national/Release/TXT/SeriesRegister.txt", BEA, "NIPA series register"),
    ("bea_TablesRegister.txt", "https://apps.bea.gov/national/Release/TXT/TablesRegister.txt", BEA, "NIPA tables register"),
    ("klems2025_national_accounts.csv", "https://www.dropbox.com/s/nkz7mdp0onken1j/national%20accounts.csv?dl=1",
     "EUKLEMS & INTANProd (Luiss) terms", "EUKLEMS & INTANProd 2025 release, national accounts"),
    ("klems2025_capital_accounts.csv", "https://www.dropbox.com/s/sp2p4m86et66nfg/capital%20accounts.csv?dl=1",
     "EUKLEMS & INTANProd (Luiss) terms", "EUKLEMS & INTANProd 2025 release, capital accounts"),
    ("z1_csv_files.zip", "https://www.federalreserve.gov/releases/z1/current/z1_csv_files.zip", "public domain (Federal Reserve)",
     "Financial Accounts of the US (Z.1), all tables, CSV"),
    ("es_nrg_ind_id.json", ES + "nrg_ind_id?format=JSON&siec=TOTAL", "Eurostat (CC BY 4.0)",
     "energy import dependency by country"),
    ("es_nama_10_a64_B2A3N.json", ES + "nama_10_a64?format=JSON&na_item=B2A3N&unit=CP_MEUR", "Eurostat (CC BY 4.0)",
     "net operating surplus and mixed income by A*64 (check of Eurostat profit)"),
] + [(f"oesm{y % 100:02d}in4.zip", f"https://www.bls.gov/oes/special-requests/oesm{y % 100:02d}in4.zip", "public domain (BLS)",
      f"OEWS national industry-specific estimates, May {y}") for y in range(2012, 2025)]


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
            time.sleep(2 ** (k + 1))
    else:
        print("FAILED", name, url, flush=True)
        return False
    data = out.read_bytes()
    with open(ROOT / "data" / "raw" / "MANIFEST.csv", "a", newline="") as f:
        csv.writer(f).writerow(["v9b", name, url, licence, f"iteration 9b, {note}", len(data),
                                hashlib.sha256(data).hexdigest(), dt.datetime.now().isoformat(timespec="seconds")])
    print("ok", name, len(data), flush=True)
    return True


if __name__ == "__main__":
    want = set(sys.argv[1:])
    for name, url, lic, note in FILES:
        if (want and name not in want) or (not want and (RAW / name).exists()):
            continue
        fetch(name, url, lic, note)
