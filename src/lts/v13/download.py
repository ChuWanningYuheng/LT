"""Iteration 13, stage D: data downloads only (no calculations).

Files go to data/raw/v13/<source>/ (gitignored); files larger than 100 MB are moved to EXTERNAL/<source>/ outside the
repository. Every file gets a row in data/raw/MANIFEST.csv (group v13; URL, size, sha256, retrieval time, contents).
Every attempt (also failed ones) is appended to data/raw/v13/attempts.jsonl (URL, HTTP status or error).

Usage: PYTHONPATH=src python -P -m lts.v13.download <source> [<source> ...]
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from ..v6.series import ROOT

RAW = ROOT / "data" / "raw" / "v13"
EXTERNAL = ROOT.parent / "LT_external" / "v13"
MANIFEST = ROOT / "data" / "raw" / "MANIFEST.csv"
ATTEMPTS = RAW / "attempts.jsonl"
UA = "LT-research-script/1.0 (noreply@anthropic.com)"
BIG = 100 * 1024 * 1024
BLS_CE = "https://download.bls.gov/pub/time.series/ce/"

CE_SECTORS = ["00a.TotalNonfarm.Employment", "05a.TotalPrivate.Employment", "05b.TotalPrivate.AllEmployeeHoursAndEarnings",
              "05c.TotalPrivate.ProductionEmployeeHoursAndEarnings"]
for code, nm in (("10", "MiningAndLogging"), ("20", "Construction"), ("30", "Manufacturing"),
                 ("31", "ManufacturingDurableGoods"), ("32", "ManufacturingNondurableGoods"),
                 ("40", "TradeTransportationAndUtilities"), ("41", "WholesaleTrade"), ("42", "RetailTrade"),
                 ("43", "TransportationAndWarehousingAndUtilities"), ("50", "Information"), ("55", "FinancialActivities"),
                 ("60", "ProfessionalBusinessServices"), ("65", "EducationAndHealthCare"), ("70", "LeisureAndHospitality"),
                 ("80", "OtherServices")):
    CE_SECTORS += [f"{code}a.{nm}.Employment", f"{code}b.{nm}.AllEmployeeHoursAndEarnings",
                   f"{code}c.{nm}.ProductionEmployeeHoursAndEarnings"]
CE_SECTORS.append("90a.Government.Employment")

SOURCES = {
    "bls_ces": [(f, BLS_CE + f, "BLS (public domain)",
                 "BLS CES national, metadata" if not f.startswith("ce.data") else f"BLS CES national, data file {f}")
                for f in ["ce.txt", "ce.series", "ce.industry", "ce.datatype", "ce.supersector", "ce.period",
                          "ce.seasonal", "ce.footnote"] + [f"ce.data.{s}" for s in CE_SECTORS]],
}


def log_attempt(url, status, note=""):
    RAW.mkdir(parents=True, exist_ok=True)
    with open(ATTEMPTS, "a") as f:
        f.write(json.dumps(dict(time=dt.datetime.now().isoformat(timespec="seconds"), url=url, status=status,
                                note=note), ensure_ascii=False) + "\n")


def probe(url, method="GET", headers=None):
    """status code (int) or error string; logs the attempt."""
    h = {"User-Agent": UA}
    h.update(headers or {})
    try:
        req = urllib.request.Request(url, headers=h, method=method)
        with urllib.request.urlopen(req, timeout=120) as r:
            st = r.status
    except urllib.error.HTTPError as e:
        st = e.code
    except Exception as e:                                       # noqa: BLE001
        st = f"{type(e).__name__}: {e}"
    log_attempt(url, st, "probe")
    return st


def fetch(source, name, url, licence, note, tries=4, headers=None):
    d = RAW / source
    d.mkdir(parents=True, exist_ok=True)
    out = d / name
    h = {"User-Agent": UA}
    h.update(headers or {})
    st = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=h)
            with urllib.request.urlopen(req, timeout=1800) as r, open(out, "wb") as f:
                st = r.status
                shutil.copyfileobj(r, f, 1 << 20)
            break
        except urllib.error.HTTPError as e:
            st = e.code
            log_attempt(url, st, f"fetch {source}/{name}")
            if e.code in (401, 403, 404, 410):
                break
        except Exception as e:                                   # noqa: BLE001
            st = f"{type(e).__name__}: {e}"
            log_attempt(url, st, f"fetch {source}/{name}")
        time.sleep(2 ** (k + 1))
    else:
        st = st or "failed"
    if not out.exists() or st != 200:
        if out.exists():
            out.unlink()
        print("FAILED", source, name, st, flush=True)
        return False
    log_attempt(url, 200, f"fetch {source}/{name}")
    sha, size = hashlib.sha256(), out.stat().st_size
    with open(out, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            sha.update(b)
    rel = f"v13/{source}/{name}"
    if size > BIG:
        (EXTERNAL / source).mkdir(parents=True, exist_ok=True)
        shutil.move(str(out), EXTERNAL / source / name)
        rel = f"EXTERNAL:LT_external/v13/{source}/{name}"
    with open(MANIFEST, "a", newline="") as f:
        csv.writer(f).writerow(["v13", rel, url, licence, f"iteration 13 stage D, {note}", size, sha.hexdigest(),
                                dt.datetime.now().isoformat(timespec="seconds")])
    print("ok", source, name, size, flush=True)
    return True


def annotate(source, text):
    """append a content description to the MANIFEST notes of this source's v13 rows"""
    rows = list(csv.reader(open(MANIFEST, newline="")))
    for r in rows[1:]:
        if r[0] == "v13" and (r[1].startswith(f"v13/{source}/") or r[1].startswith(f"EXTERNAL:LT_external/v13/{source}/")) \
                and text not in r[4]:
            r[4] = f"{r[4]}; {text}"
    with open(MANIFEST, "w", newline="") as f:
        csv.writer(f).writerows(rows)


DESCRIPTIONS = {
    "bls_ces": ("contents: CES national establishment survey; monthly M01-M12 (+M13 annual average for NSA), 1939-2026; "
                "850 industry codes (CES codes ~ NAICS, supersectors to 6-digit); datatypes 01 all employees and 06 "
                "production & nonsupervisory employees (thousands), 02/07 average weekly hours, 03/08 average hourly "
                "earnings (USD), 56/57/81/82 aggregate weekly hours and payrolls; all-employee hours and earnings only "
                "from 2006; production/nonsupervisory hours from 1939 (manufacturing), 1947 (mining, construction), "
                "1964 (total private), mostly 1990 for detailed industries; S = seasonally adjusted, U = not"),
}


def run(sources):
    for s in sources:
        for name, url, lic, note in SOURCES[s]:
            if (RAW / s / name).exists() or (EXTERNAL / s / name).exists():
                continue
            fetch(s, name, url, lic, note)


if __name__ == "__main__":
    if sys.argv[1:2] == ["annotate"]:
        for src in sys.argv[2:]:
            annotate(src, DESCRIPTIONS[src])
    else:
        run(sys.argv[1:])
