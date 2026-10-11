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
    "jhs": [(n, "https://www.worldklems.net/documents/" + u, "WorldKLEMS / Jorgenson-Ho-Samuels (research use)", d) for n, u, d in (
        ("usa_wk_mar_2017.xlsx", "31/usa_wk_mar_2017.xlsx", "WorldKLEMS US basic file, March 2017 release"),
        ("usa_wk_apr_2013.xlsx", "30/usa_wk_apr_2013.xlsx", "WorldKLEMS US basic file, April 2013 release"),
        ("USA_WK_apr_2013_labour.xlsx", "35/USA_WK_apr_2013_labour.xlsx", "WorldKLEMS US labour file, April 2013"),
        ("jorgenson_ho_samuels.USProductionAccount.pdf", "38/jorgenson_ho_samuels.USProductionAccount.pdf",
         "Jorgenson, Ho, Samuels: A prototype industry-level production account for the US (paper)"),
        ("usa_wk_apr_2013_source.pdf", "41/usa_wk_apr_2013_source.pdf", "April 2013 release: additional note"),
        ("usa_wk_11_2010.xls", "29/usa_wk_11_2010.xls", "WorldKLEMS US basic file, November 2010 release"),
        ("usa_wk_11_2010_add.xls", "24/usa_wk_11_2010_add.xls", "WorldKLEMS US additional file, November 2010"),
        ("usa_wk_11_2010_source.pdf", "40/usa_wk_11_2010_source.pdf", "November 2010 release: sources and notes"))],
    "bea_bls_ilpa": [(u.split("/")[-1], "https://www.bea.gov" + u, "BEA/BLS (public domain)", d) for u, d in (
        ("/sites/default/files/2026-04/BEA-BLS-industry-level-production-account-1997-2024.xlsx",
         "BEA-BLS integrated industry-level production account (KLEMS), production account tables 1997-2024"),
        ("/sites/default/files/2026-04/BEA-BLS-industry-level-production-account-1998-2024-CONTR.xlsx",
         "BEA-BLS ILPA, contributions tables 1998-2024"),
        ("/sites/default/files/2026-04/BEA-BLS-industry-level-production-account-Expanded-Capital-Detail-1997-2024.xlsx",
         "BEA-BLS ILPA, production account tables with detailed capital, 1997-2024"),
        ("/sites/default/files/2026-04/BEA-BLS-industry-level-production-account-Expanded-Capital-Detail-1998-2024-CONTR.xlsx",
         "BEA-BLS ILPA, contributions tables with detailed capital, 1998-2024"),
        ("/sites/default/files/2020-11/Technical-Document-Toward-a-BEA-BLS-Integrated-Industry-level-Production-Account-for-1947-2016.pdf",
         "Technical document: Toward a BEA-BLS integrated industry-level production account for 1947-2016"),
        ("/sites/default/files/2020-11/KLEMS-Research-Tables-BEA-BLS-Integrated-Industry-Level-Production-Account-1947-2016_0.xlsx",
         "KLEMS research tables, BEA-BLS integrated industry-level production account 1947-2016"),
        ("/sites/default/files/2020-11/KLEMS-Research-Tables-Underlying-Data_0.xlsx",
         "KLEMS research tables 1947-2016: underlying data"),
        ("/sites/default/files/2023-09/BEA-BLS-industry-level-production-account-archive-data-2022.zip",
         "BEA-BLS ILPA archive data, 2022 vintage"),
        ("/sites/default/files/2023-09/BEA-BLS-industry-level-production-account-archive-data-2021.zip",
         "BEA-BLS ILPA archive data, 2021 vintage"))],
    "bls_productivity": [(u.split("/")[-1], "https://www.bls.gov" + u, "BLS (public domain)", d) for u, d in (
        ("/productivity/tables/major-industry-total-factor-productivity-klems.xlsx",
         "BLS total factor productivity for major industries (KLEMS)"),
        ("/productivity/tables/hours-employment-detailed-industries.xlsx", "BLS hours and employment, detailed industries"),
        ("/productivity/tables/labor-productivity-detailed-industries.xlsx", "BLS labor productivity, detailed industries"),
        ("/productivity/tables/historical-labor-productivity-detailed-industries-discontinued.xlsx",
         "BLS historical labor productivity, detailed industries (discontinued series)"),
        ("/productivity/tables/total-factor-productivity-manufacturing-and-transportation-detailed-industries.xlsx",
         "BLS TFP, manufacturing and transportation detailed industries"),
        ("/productivity/tables/total-factor-productivity-major-sectors.xlsx", "BLS TFP, major sectors"),
        ("/productivity/tables/total-factor-productivity-major-sectors-historical.xlsx", "BLS TFP, major sectors, historical"),
        ("/productivity/tables/total-factor-productivity-capital-details-major-sectors-and-industries.xlsx",
         "BLS TFP capital details, major sectors and industries"),
        ("/productivity/tables/rental-prices-major-industries.xlsx", "BLS capital rental prices, major industries"),
        ("/productivity/tables/major-sector-industries-nonprofit-owner-occupied-capital-measures.xlsx",
         "BLS capital measures incl. nonprofit and owner-occupied"),
        ("/productivity/tables/labor-productivity-manufacturing-sectors-historical-sic-1949-2003.xlsx",
         "BLS labor productivity, manufacturing sectors, historical SIC 1949-2003"),
        ("/productivity/tables/total-factor-productivity-manufacturing-historical-sic-1949-2001.txt",
         "BLS TFP, manufacturing, historical SIC 1949-2001"),
        ("/web/prod2/total-economy-hours-employment.xlsx", "BLS total economy hours and employment"),
        ("/productivity/articles-and-research/industry-production-account-capital.xlsx",
         "BLS research: industry production account, capital"),
        ("/productivity/articles-and-research/industry-production-account-capital-expanded-detail.xlsx",
         "BLS research: industry production account, capital, expanded detail"))],
    "bea_fa_detail": [(n, "https://apps.bea.gov/national/FA2004/Details/xls/" + n, "BEA (public domain)", d) for n, d in (
        ("detailnonres_stk1.xlsx", "BEA fixed assets detailed data: private nonresidential net stocks, current cost, asset x industry"),
        ("detailnonres_stk2.xlsx", "BEA fixed assets detailed data: private nonresidential net stocks, fixed (chained) cost"),
        ("detailnonres_dep1.xlsx", "BEA fixed assets detailed data: depreciation, current cost"),
        ("detailnonres_dep2.xlsx", "BEA fixed assets detailed data: depreciation, fixed cost"),
        ("detailnonres_inv1.xlsx", "BEA fixed assets detailed data: investment, current dollars"),
        ("detailnonres_inv2.xlsx", "BEA fixed assets detailed data: investment, fixed cost"),
        ("DetailNonres_rate.xlsx", "BEA fixed assets detailed data: implied rates of depreciation"),
        ("detailresidential.xlsx", "BEA fixed assets detailed data: residential"),
        ("DetailCDG.xlsx", "BEA fixed assets detailed data: consumer durable goods"))],
    "bea_fa_tables": [(n, "https://apps.bea.gov/national/FixedAssets/Release/" + u, "BEA (public domain)", d) for n, u, d in (
        ("Section2All_xls.xlsx", "XLS/Section2All_xls.xlsx", "BEA Fixed Assets section 2 (by type of asset), all tables"),
        ("Section3All_xls.xlsx", "XLS/Section3All_xls.xlsx", "BEA Fixed Assets section 3 (private by industry): 3.1-3.9 E/S/I"),
        ("Section4All_xls.xlsx", "XLS/Section4All_xls.xlsx", "BEA Fixed Assets section 4 (private nonresidential by legal form)"),
        ("Section6All_xls.xlsx", "XLS/Section6All_xls.xlsx", "BEA Fixed Assets section 6 (by legal form and industry)"),
        ("FixedAssets.txt", "TXT/FixedAssets.txt", "BEA Fixed Assets flat file, all tables (series code, period, value)"),
        ("FA_TablesRegister.txt", "TXT/TablesRegister.txt", "BEA Fixed Assets tables register"),
        ("FA_SeriesRegister.txt", "TXT/SeriesRegister.txt", "BEA Fixed Assets series register"))],
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
    "jhs": ("contents: Jorgenson-Ho-Samuels US industry production account as published on worldklems.net. "
            "Mar 2017 (usa_wk_mar_2017.xlsx): 65 BEA/NAICS industries, 1947-2014, nominal and constant gross output, "
            "capital, labour, intermediate input (no hours). Apr 2013 (usa_wk_apr_2013.xlsx): 40 ISIC Rev.3 codes incl. "
            "aggregates, 1947-2010, WorldKLEMS variables GO, II, VA, EMP (thousands), H_EMP (millions of hours), LAB, "
            "CAP (USD millions), price and volume indices 2005=100. Labour file Apr/May 2013: compensation per hour, "
            "persons and weekly hours by industry (31 ISIC) x sex x class (employee, self-employed/family) x education "
            "x age, 1947-2010. Nov 2010 (usa_wk_11_2010*.xls): 40 codes, 1960-2007, incl. EMPE, H_EMPE"),
    "bea_bls_ilpa": ("contents: (1) BEA-BLS integrated industry-level production account, release 2026-04-27, 63 industries "
                     "(BEA summary, NAICS) + aggregates, 1997-2024: quantity indexes 2017=100 (gross output, VA, energy, "
                     "materials, services, capital by type art/R&D/IT/software/other, labour college/non-college, labour "
                     "input, labour hours) and nominal compensation (USD millions) by input; hours only as an index. "
                     "(2) Expanded capital detail 1997-2024 (computers, communication, instruments, transport, other "
                     "equipment, structures). (3) Research account 1947-2016 (Eldridge et al. 2020): underlying data, "
                     "44 industries 1947-1963 and 63 industries 1963-2016: nominal gross output, intermediate input, "
                     "capital input by type, college / non-college labour input, quantity indexes and total hours (hrs; "
                     "units not stated in the file); tables 11.1-11.7. (4) archive vintages 2021 and 2022 (1987-2020)"),
    "nipa_hours_fred": ("contents: hours worked by full-time and part-time employees (NIPA tables 6.9B/C/D lines as "
                        "republished by FRED), millions of hours, annual; sector level only (about 20 lines): totals "
                        "B4701-B4703 1948-2022; sector lines J47xx 1948-1969, B47xx 1987-2000 (SIC), N47xx 2000-2022 "
                        "(NAICS); no sector series for 1970-1986 on FRED; employees only (no self-employed)"),
    "bls_productivity": ("contents: BLS Office of Productivity and Technology tables (machine-readable sheets). Major "
                         "industries TFP (KLEMS): 81 NAICS industries/aggregates, 1987-2024, 51 measures incl. hours "
                         "worked (millions), labor composition, capital input, combined inputs, intermediate/energy "
                         "inputs; hours and employment for detailed industries: 774 NAICS codes (2- to 6-digit), "
                         "1987-2025, all workers (millions of hours, thousands of jobs); labor productivity detailed "
                         "industries 1987-2025 (hours, labor compensation, output); capital details: productive capital "
                         "stock, wealth stock, investment, rental prices, depreciation rates by asset category, 84 sectors/"
                         "industries, 1987-2024; total economy hours and employment by class (employees, proprietors, "
                         "unpaid family workers), quarterly 1947-2026; historical SIC manufacturing 1949-2003 (labor "
                         "productivity) and 1949-2001 (TFP, text file)"),
    "bea_fa_detail": ("contents: BEA fixed assets detailed data (vintage updated 2026-09-30), private nonresidential, "
                      "74 industry sheets (BEA FA industry codes, e.g. 110C, 2110) x about 100 NIPA asset types; "
                      "investment 1901-2025, net stocks and depreciation 1925-2025; millions of dollars; current cost "
                      "(stk1, dep1, inv1) and fixed (chained 2017) cost (stk2, dep2, inv2); implied depreciation rates; "
                      "residential and consumer durables files. No historical-cost detail at asset x industry level"),
    "bea_fa_tables": ("contents: BEA Fixed Assets published tables, files created 2026-09-28: section 2 (by asset type), "
                      "section 3 (private by industry; 3.1 current-cost net stock, 3.2 quantity indexes, 3.3 historical-"
                      "cost net stock, 3.4 current-cost depreciation, 3.6 historical-cost depreciation, 3.7 historical-"
                      "cost investment, 3.8-3.9; E/S/I and total), section 4 (nonresidential by legal form), section 6 "
                      "(by legal form and industry); annual 1947-2025 (section 3), millions of dollars, year-end; flat file "
                      "FixedAssets.txt with all series and registers"),
}


def fred_hours():
    """NIPA tables 6.9B (B), 6.9C (J), 6.9D (N): hours worked by full-time and part-time employees by industry, as
    republished by FRED (series {B,J,N}47NNC0A222NBEA, millions of hours). Enumerates NN = 01..99; titles from the
    series page."""
    src = "nipa_hours_fred"
    titles = []
    for pre in ("B", "J", "N"):
        for nn in range(1, 100):
            sid = f"{pre}47{nn:02d}C0A222NBEA"
            if (RAW / src / f"{sid}.csv").exists():
                continue
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
            ok = fetch(src, f"{sid}.csv", url, "BEA via FRED (public domain)",
                       f"NIPA table 6.9{dict(B='B', J='C', N='D')[pre]} line via FRED, series {sid}", tries=2)
            time.sleep(0.5)
            if ok:
                try:
                    req = urllib.request.Request(f"https://fred.stlouisfed.org/series/{sid}", headers={"User-Agent": UA})
                    html = urllib.request.urlopen(req, timeout=120).read().decode("utf-8", "replace")
                    import re
                    m = re.search(r"<title>(.*?)</title>", html, re.S)
                    titles.append((sid, m.group(1).strip() if m else ""))
                except Exception as e:                           # noqa: BLE001
                    titles.append((sid, f"title not read: {e}"))
                time.sleep(0.5)
    if titles:
        with open(RAW / src / "_titles.csv", "a", newline="") as f:
            csv.writer(f).writerows(titles)


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
    elif sys.argv[1:2] == ["fred_hours"]:
        fred_hours()
    else:
        run(sys.argv[1:])
