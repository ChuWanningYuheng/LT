"""Iteration 13b, stage D: data downloads only (no calculations).

Files go to data/raw/v13b/<source>/ (gitignored); files larger than 100 MB are moved to EXTERNAL/<source>/ outside the
repository. Every file gets a row in data/raw/MANIFEST.csv (group v13b; URL, size, sha256, retrieval time, contents).
Every attempt (also failed ones) is appended to data/raw/v13b/attempts.jsonl (URL, HTTP status or error).
Sites that refuse scripted downloads (HTTP 403 with a bot notice) are recorded, not worked around.

Usage: PYTHONPATH=src python -P -m lts.v13b.download <source> [<source> ...] | all
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

from ..v6.series import ROOT

RAW = ROOT / "data" / "raw" / "v13b"
EXTERNAL = ROOT.parent / "LT_external" / "v13b"
MANIFEST = ROOT / "data" / "raw" / "MANIFEST.csv"
ATTEMPTS = RAW / "attempts.jsonl"
UA = "LT-research-script/1.0 (noreply@anthropic.com)"
BIG = 100 * 1024 * 1024
BLS = "https://download.bls.gov/pub/time.series/"


def log_attempt(url, status, note=""):
    RAW.mkdir(parents=True, exist_ok=True)
    with open(ATTEMPTS, "a") as f:
        f.write(json.dumps(dict(time=dt.datetime.now().isoformat(timespec="seconds"), url=url, status=status,
                                note=note)) + "\n")


def fetch(source, name, url, licence, note, tries=4, secret_params=None):
    """Download url to RAW/source/name; secret_params (e.g. an API key) are added to the request but never logged."""
    d = RAW / source
    d.mkdir(parents=True, exist_ok=True)
    out = d / name
    full = url
    if secret_params:
        full = url + ("&" if "?" in url else "?") + urllib.parse.urlencode(secret_params)
    st = None
    for k in range(tries):
        try:
            req = urllib.request.Request(full, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=3600) as r, open(out, "wb") as f:
                st = r.status
                expected = r.headers.get("Content-Length")
                shutil.copyfileobj(r, f, 1 << 20)
            if expected is not None and out.stat().st_size != int(expected):
                raise IOError(f"truncated: {out.stat().st_size} of {expected} bytes")
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
    rel = f"v13b/{source}/{name}"
    if size > BIG:
        (EXTERNAL / source).mkdir(parents=True, exist_ok=True)
        shutil.move(str(out), EXTERNAL / source / name)
        rel = f"EXTERNAL:LT_external/v13b/{source}/{name}"
    with open(MANIFEST, "a", newline="") as f:
        csv.writer(f).writerow(["v13b", rel, url, licence, f"iteration 13b stage D, {note}", size, sha.hexdigest(),
                                dt.datetime.now().isoformat(timespec="seconds")])
    print("ok", source, name, size, flush=True)
    return True


def probe(url):
    """Status of a URL without saving the body (for sources that refuse scripts)."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=60) as r:
            st = r.status
    except urllib.error.HTTPError as e:
        st = e.code
    except Exception as e:                                       # noqa: BLE001
        st = f"{type(e).__name__}: {e}"
    log_attempt(url, st, "probe")
    print("probe", st, url, flush=True)
    return st


# ---------------------------------------------------------------- sources
def eia():
    lic = "EIA (public domain)"
    fetch("eia", "MER_T09_01.csv", "https://www.eia.gov/totalenergy/data/browser/csv.php?tbl=T09.01", lic,
          "EIA Monthly Energy Review table 9.1 Crude Oil Price Summary (domestic first purchase price from 1949 annual, "
          "landed and imported prices; nominal $ per barrel)")
    fetch("eia", "MER_T09_01.xlsx", "https://www.eia.gov/totalenergy/data/browser/xls.php?tbl=T09.01", lic,
          "EIA MER table 9.1, spreadsheet version with notes")
    fetch("eia", "steo_real_prices.xlsx", "https://www.eia.gov/outlooks/steo/realprices/real_prices.xlsx", lic,
          "EIA Short-Term Energy Outlook real energy prices (monthly, nominal and real crude oil prices)")


def energy_institute():
    probe("https://www.energyinst.org/statistical-review/resources-and-data-downloads")
    lic = "Our World in Data (CC BY 4.0), data from Energy Institute Statistical Review of World Energy"
    fetch("ei_owid", "crude-oil-prices.csv", "https://ourworldindata.org/grapher/crude-oil-prices.csv", lic,
          "crude oil prices since 1861 ($ per barrel, nominal and constant), Energy Institute Statistical Review of "
          "World Energy as republished by Our World in Data (the Energy Institute site refuses scripted downloads)")
    fetch("ei_owid", "crude-oil-prices.metadata.json",
          "https://ourworldindata.org/grapher/crude-oil-prices.metadata.json", lic,
          "metadata of the OWID crude oil price series (source citation, units, processing)")


def ceo_pay():
    for u in ("https://www.epi.org/chart/ceo-pay-ceo-pay-ceo-compensation-over-time-1a/",
              "https://www.epi.org/blog/ceo-pay-increased-in-2024-and-is-now-281-times-that-of-the-typical-worker-new-"
              "epi-landing-page-has-all-the-details/"):
        probe(u)
    lic = "Federal Reserve Board (public), authors"
    fetch("frydman_saks", "figure_data.html", "https://www.federalreserve.gov/pubs/feds/2007/200735/figure_data.html",
          lic, "Frydman & Saks, FEDS 2007-35, yearly data behind the figures, 1936-2005 (HTML tables)")
    fetch("frydman_saks", "200735pap.pdf", "https://www.federalreserve.gov/pubs/feds/2007/200735/200735pap.pdf", lic,
          "Frydman & Saks, Executive Compensation: A New View from a Long-Term Perspective, 1936-2005, FEDS 2007-35")
    fetch("frydman_saks", "trends_rfs2010.pdf", "https://www.mit.edu/~frydman/trends_rfs2010.pdf", lic,
          "Frydman & Saks (2010), Review of Financial Studies version (author copy)")


def wid():
    fetch("wid", "wid_all_data.zip", "https://wid.world/bulk_download/wid_all_data.zip", "WID (CC BY 4.0)",
          "World Inequality Database, full bulk download (all countries and variables, incl. sptinc / sfiinc top shares)")


def pss():
    fetch("pss2014", "Files_nberwp.zip", "https://eml.berkeley.edu/~saez/Files_nberwp.zip", "authors (research use)",
          "Piketty, Saez & Stantcheva (2014), data and programs of the NBER WP version (top marginal tax rates, top income "
          "shares, growth by OECD country)")
    fetch("pss2014", "piketty-saez-stantchevaAEJ14.pdf", "https://eml.berkeley.edu/~saez/piketty-saez-stantchevaAEJ14.pdf",
          "authors", "Piketty, Saez & Stantcheva (2014), AEJ: Economic Policy 6(1), paper")
    probe("https://www.openicpsr.org/")


def oecd_tax():
    fetch("oecd_tax", "PIT_TOP_EARN_THRESH.csv",
          "https://sdmx.oecd.org/public/rest/data/OECD.CTP.TPS,DSD_TAX_PIT@DF_PIT_TOP_EARN_THRESH,1.0/all"
          "?format=csvfilewithlabels", "OECD (terms and conditions)",
          "OECD Tax Database table I.7: top statutory personal income tax rate and top marginal rate, thresholds, 2000-")


def dfa():
    lic = "Federal Reserve Board (public)"
    fetch("fed_dfa", "dfa.zip", "https://www.federalreserve.gov/releases/z1/dataviz/download/zips/dfa.zip", lic,
          "Fed Distributional Financial Accounts, all levels and shares by income, wealth, age, generation (1989-)")
    fetch("fed_dfa", "dfa_public_code.zip",
          "https://www.federalreserve.gov/releases/z1/dataviz/download/zips/dfa_public_code.zip", lic,
          "Fed DFA public code and documentation")


def psz():
    lic = "authors (research use)"
    base = "https://gabriel-zucman.eu/files/"
    for nm, note in (("PSZ2022AppendixTablesII(Distrib).xlsx", "PSZ distributional national accounts, appendix tables II "
                      "(distribution: income shares and averages by group, pre-tax and post-tax, 1913-)"),
                     ("PSZ2022AppendixTablesI(Aggreg).xlsx", "PSZ appendix tables I (aggregates)"),
                     ("PSZ2018MainData.xlsx", "PSZ (2018) main data"),
                     ("PSZCodebook.pdf", "PSZ codebook"), ("PSZ2018DataAppendix.pdf", "PSZ (2018) data appendix"),
                     ("PSZUpdates.pdf", "PSZ updates note")):
        fetch("psz", nm, base + urllib.parse.quote(nm), lic, note)


def nyfed():
    lic = "Federal Reserve Bank of New York / Equifax (public tables)"
    fetch("nyfed_ccp", "hhd_c_report_2026q2.xlsx",
          "https://www.newyorkfed.org/medialibrary/interactives/householdcredit/data/xls/hhd_c_report_2026q2.xlsx", lic,
          "NY Fed Household Debt and Credit Report data, 2026Q2 (CCP/Equifax; national series and state-level debt per "
          "capita tables)")
    fetch("nyfed_ccp", "area_report_by_year.xlsx",
          "https://www.newyorkfed.org/medialibrary/interactives/householdcredit/data/xls/area_report_by_year.xlsx", lic,
          "NY Fed HHDC, state-level household debt statistics by year (CCP/Equifax; all states)")
    lic2 = "Federal Reserve Board (public); source FRBNY Consumer Credit Panel / Equifax"
    for nm in ("household-debt-by-state.zip", "household-debt-by-county.zip", "household-debt.zip"):
        fetch("fed_hh_debt", nm, "https://www.federalreserve.gov/releases/z1/dataviz/download/zips/" + nm, lic2,
              f"Fed Board, household debt-to-income by geography from the FRBNY CCP/Equifax, 1999- ({nm})")
    fetch("nyfed_ccp", "report_by_age.xls",
          "https://www.newyorkfed.org/medialibrary/interactives/householdcredit/data/xls/report_by_age.xls", lic,
          "NY Fed HHDC, debt by age")


def frank():
    lic = "author (research use)"
    for nm, note in (("Frank_States_2023.xlsx", "Frank (Sommeiller-Price) top income shares by US state, annual (1917-)"),
                     ("Frank_StateGini_2023.xlsx", "Frank state Gini and related inequality measures")):
        fetch("frank_states", nm, "https://profiles.shsu.edu/eco_mwf/" + nm, lic, note)


def fhfa():
    lic = "FHFA (public)"
    for nm, note in (("hpi_at_state.csv", "FHFA all-transactions house price index by state, quarterly, 1975-"),
                     ("hpi_po_state.xlsx", "FHFA purchase-only HPI by state"),
                     ("hpi_exp_state.xlsx", "FHFA expanded-data HPI by state")):
        fetch("fhfa", nm, "https://www.fhfa.gov/hpi/download/quarterly_datasets/" + nm, lic, note)


def bls_states():
    lic = "BLS (public domain)"
    for nm in ("la.txt", "la.series", "la.area", "la.area_type", "la.measure", "la.period", "la.seasonal",
               "la.data.3.AllStatesS", "la.data.2.AllStatesU"):
        fetch("bls_laus", nm, BLS + "la/" + nm, lic, f"BLS LAUS {nm} (states: unemployment, labour force)")
    for nm in ("sm.txt", "sm.series", "sm.area", "sm.state", "sm.industry", "sm.data_type", "sm.supersector",
               "sm.data.54.TotalNonFarm.All", "sm.data.55.TotalNonFarmStateWide.All"):
        fetch("bls_sm", nm, BLS + "sm/" + nm, lic, f"BLS CES state and metro {nm} (employment by state)")


def bea_sainc():
    fetch("bea_sainc", "SAINC.zip", "https://apps.bea.gov/regional/zip/SAINC.zip", "BEA (public domain)",
          "BEA regional state annual personal income (SAINC tables: personal income, population, per capita, "
          "components), bulk file")


def cps_probe():
    """CPS microdata: IPUMS needs an API key (IPUMS_API_KEY); NBER refuses scripted downloads."""
    print("IPUMS_API_KEY set:", bool(os.environ.get("IPUMS_API_KEY")), flush=True)
    for u in ("https://data.nber.org/morg/annual/morg79.dta", "https://data.nber.org/cps/cpsmar68.zip",
              "https://www2.census.gov/programs-surveys/cps/datasets/"):
        probe(u)


def census_asec(years=range(1998, 2026)):
    """CPS ASEC (March supplement) public-use files from the Census Bureau, 1998- (earlier years only via NBER or
    IPUMS). Per year: the ASCII public-use zip (about 30 MB; the CSV zip, where published, is about 150 MB), plus record
    layouts and documentation."""
    import re
    base = "https://www2.census.gov/programs-surveys/cps/datasets/{}/march/"
    lic = "U.S. Census Bureau (public domain)"
    for y in years:
        url = base.format(y)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=120) as r:
                html = r.read().decode("utf-8", "ignore")
        except Exception as e:                                   # noqa: BLE001
            log_attempt(url, f"{type(e).__name__}: {e}", "listing")
            print("FAILED listing", y, flush=True)
            continue
        names = sorted(set(re.findall(r'href="([^"?/][^"]*\.(?:zip|pdf|txt))"', html)))
        data = [n for n in names if n.endswith(".zip") and "REPWGT" not in n.upper() and "sas" not in n.lower()
                and not n.endswith("csv.zip")] or [n for n in names if n.endswith("csv.zip")]
        docs = [n for n in names if n.endswith((".pdf", ".txt"))]
        for n in data + docs:
            fetch("census_asec", f"{y}_{n}", url + n, lic,
                  f"CPS ASEC {y} public-use file from the Census Bureau ({n})")


SOURCES = dict(eia=eia, energy_institute=energy_institute, ceo_pay=ceo_pay, wid=wid, pss=pss, oecd_tax=oecd_tax,
               dfa=dfa, psz=psz, nyfed=nyfed, frank=frank, fhfa=fhfa, bls_states=bls_states, bea_sainc=bea_sainc,
               cps_probe=cps_probe, census_asec=census_asec)


if __name__ == "__main__":
    names = sys.argv[1:] or ["all"]
    for n in (SOURCES if names == ["all"] else names):
        SOURCES[n]()
