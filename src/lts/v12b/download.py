"""Iteration 12b, stage 0: downloads into data/raw/v12b/ and rows in data/raw/MANIFEST.csv (group v12b).

Usage: PYTHONPATH=src python -P -m lts.v12b.download [name ...]   (no names: everything not yet present)
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import sys
import time
import urllib.request

from ..v6.series import ROOT

RAW = ROOT / "data" / "raw" / "v12b"
RAW.mkdir(parents=True, exist_ok=True)
UA = "LT-research-script/1.0 (noreply@anthropic.com)"
BEAIO = "https://apps.bea.gov/industry/xls/io-annual/"
ES = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
EIA = "https://www.eia.gov/totalenergy/data/browser/csv.php?tbl="
PEFA_GEO = ("DE", "FR", "IT", "ES", "NL", "AT", "PL", "CZ")

FILES = [
    ("bea_IOUse_Before_Redefinitions_PRO_1947-1962_Summary.xlsx", BEAIO + "IOUse_Before_Redefinitions_PRO_1947-1962_Summary.xlsx",
     "BEA (public domain)", "use tables before redefinitions, producers' prices, 1947-1962, 46 industries (NAICS)"),
    ("bea_IOUse_Before_Redefinitions_PRO_1963-1996_Summary.xlsx", BEAIO + "IOUse_Before_Redefinitions_PRO_1963-1996_Summary.xlsx",
     "BEA (public domain)", "use tables before redefinitions, producers' prices, 1963-1996, 65 industries (NAICS)"),
    ("bea_IOMake_Before_Redefinitions_1947-1962_Summary.xlsx", BEAIO + "IOMake_Before_Redefinitions_1947-1962_Summary.xlsx",
     "BEA (public domain)", "make tables before redefinitions, 1947-1962"),
    ("bea_IOMake_Before_Redefinitions_1963-1996_Summary.xlsx", BEAIO + "IOMake_Before_Redefinitions_1963-1996_Summary.xlsx",
     "BEA (public domain)", "make tables before redefinitions, 1963-1996"),
    ("bea_AllTablesHist.zip", "https://apps.bea.gov/industry/iTables%20Static%20Files/AllTablesHist.zip",
     "BEA (public domain)", "GDP by industry 1947-1997 (NAICS): value added, components, gross output"),
    ("eia_T01.02.csv", EIA + "T01.02", "EIA (public domain)", "primary energy production by source, 1949-"),
    ("eia_T01.03.csv", EIA + "T01.03", "EIA (public domain)", "primary energy consumption by source, 1949-"),
    ("es_env_ac_pefa04.json", ES + "env_ac_pefa04?format=JSON&lang=EN&unit=TJ" + "".join(f"&geo={g}" for g in PEFA_GEO),
     "Eurostat (CC BY 4.0)", "physical energy flow accounts, key indicators by NACE (natural energy inputs, net domestic use)"),
    ("maito2014_mpra55894.pdf", "https://mpra.ub.uni-muenchen.de/55894/1/MPRA_paper_55894.pdf", "MPRA (author)",
     "Maito (2014), The historical transience of capital: the downward trend in the rate of profit since XIX century"),
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
        csv.writer(f).writerow(["v12b", name, url, licence, f"iteration 12b, {note}", len(data),
                                hashlib.sha256(data).hexdigest(), dt.datetime.now().isoformat(timespec="seconds")])
    print("ok", name, len(data), flush=True)
    return True


if __name__ == "__main__":
    want = set(sys.argv[1:])
    for name, url, lic, note in FILES:
        if (want and name not in want) or (not want and (RAW / name).exists()):
            continue
        fetch(name, url, lic, note)
