"""Iteration 11, stage 0: downloads into data/raw/v11/ and rows in data/raw/MANIFEST.csv (group v11).

Usage: PYTHONPATH=src python -P -m lts.v11.download [name ...]   (no names: everything not yet present)
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import sys
import time
import urllib.request

from ..v6.series import ROOT

RAW = ROOT / "data" / "raw" / "v11"
RAW.mkdir(parents=True, exist_ok=True)
UA = "LT-research-script/1.0 (noreply@anthropic.com)"
FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id="
JST = "https://www.macrohistory.net/app/download/"
OECD = "https://sdmx.oecd.org/public/rest/data/OECD.SDD.STES,DSD_STES@DF_FINMARK,/"

FILES = [
    ("JSTdatasetR6.dta", JST + "9834512469/JSTdatasetR6.dta", "CC BY-NC-SA 4.0 (Jorda-Schularick-Taylor Macrohistory Database)",
     "JST Macrohistory Database R6 (18 countries, 1870-2020), Stata"),
    ("JST_documentationR6.pdf", JST + "9834516169/JST_documentationR6.pdf", "macrohistory.net", "JST R6 documentation"),
    ("JSTcrisis_chronology.pdf", JST + "9844625569/JSTcrisis_chronology.pdf", "macrohistory.net", "JST crisis chronology"),
    ("JST_RORE_Documentation_R6.pdf", JST + "9918957869/JST_RORE_Documentation_R6.pdf", "macrohistory.net",
     "JST rate of return on everything, documentation"),
    ("fred_GS10.csv", FRED + "GS10", "FRED terms", "10-year Treasury constant maturity, monthly"),
    ("fred_TB3MS.csv", FRED + "TB3MS", "FRED terms", "3-month Treasury bill, monthly"),
    ("fred_FEDFUNDS.csv", FRED + "FEDFUNDS", "FRED terms", "effective federal funds rate, monthly"),
    ("fred_AAA.csv", FRED + "AAA", "FRED terms", "Moody's Aaa corporate bond yield, monthly"),
    ("fred_BAA.csv", FRED + "BAA", "FRED terms", "Moody's Baa corporate bond yield, monthly"),
    ("fred_CPIAUCNS.csv", FRED + "CPIAUCNS", "FRED terms", "CPI-U not seasonally adjusted, monthly (from 1913)"),
    ("oecd_finmark_annual_IRLT.csv", OECD + ".A.IRLT.PA.....?format=csv", "OECD (CC BY 4.0)",
     "long-term interest rates, annual"),
    ("oecd_finmark_annual_IR3TIB.csv", OECD + ".A.IR3TIB.PA.....?format=csv", "OECD (CC BY 4.0)",
     "3-month interbank rates, annual"),
]


def fetch(name, url, licence, note, tries=4):
    out = RAW / name
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=600) as r, open(out, "wb") as f:
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
        csv.writer(f).writerow(["v11", name, url, licence, f"iteration 11, {note}", len(data),
                                hashlib.sha256(data).hexdigest(), dt.datetime.now().isoformat(timespec="seconds")])
    print("ok", name, len(data), flush=True)
    return True


if __name__ == "__main__":
    want = set(sys.argv[1:])
    for name, url, lic, note in FILES:
        if (want and name not in want) or (not want and (RAW / name).exists()):
            continue
        fetch(name, url, lic, note)
