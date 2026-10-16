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
import re
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
    "shaikh_capitalism": [(nm, "https://drive.google.com/uc?export=download&id=" + fid, "A. Shaikh (author's public Google Drive)",
                           "Shaikh (2016) Capitalism: data tables of the book appendices, from realecon.org/data -> Google Drive "
                           "folder 1_EonoTKayfETIoTJZQoKedr5ygOykjMd") for fid, nm in (
        ("1_tWL4sxALcGMOX5C7cVBhCr-LWWAgnN9", "Appendix_2.2_Data_Tables.xlsx"),
        ("1_sXzj5CsdBK9cb1qBfkxCuGPaZhowncL", "Appendix_4.2_Numerical_Calculations_Production_and_Costs.docx"),
        ("1_oWibfhadka_kRXr-8BRNmeRbjit7n-W", "Appendix_5.3_Data_Tables.xlsx"),
        ("1_kbJELg5AJRDBqe8ecxbzg9BlnboMRnu", "Appendix_6.8_Data_Tables_Corrected.xlsx"),
        ("1_k9jOPEPR8V1mzHfzZuI3Ne2CjqHvMpJ", "Appendix_6.8_Data_Tables.xlsx"),
        ("1_jxrQNY76JMqYVGhMeu4XSKrEJ-8dROb", "Appendix_7.2_Data_Tables.xlsx"),
        ("1_jBDkc6_lmwE_HiNgwftoKRMNzhavUU3", "Appendix_8.1_Data_Tables.xlsx"),
        ("1_ijlLBhpnwzwH5PFm7wla2-DygySh0-V", "Appendix_9.3_Data_Tables.xls"),
        ("1__h4fhmMQT3ZUJ8VBlI3IqYZHJ07tDu2", "Appendix_10.2_Data_Tables.xlsx"),
        ("1_Yu3QbgpcdyXWMNql0H-SoETZALOAhX7", "Appendix_11.2_Data_Tables.xlsx"),
        ("1_RMLimPqh1J6BE6UY4c3QCats52BzTK0", "Appendix_12.2_Data_Tables.xlsx"),
        ("1_N_lrsP6R-CTzzMiw3aAnp3JKmvSvcA2", "Appendix_14.3_Data_Tables.xlsx"),
        ("1_KYIpzqwlRuh7NxOeHUP2CchgzxM5Yii", "Appendix_15.2_Data_Tables.xlsx"),
        ("1_J2FGi0yM1kqxQofKEcgWqMkdVNJsCnn", "Appendix_16.2_Data_Tables.xlsx"),
        ("1_xhrE4RpUiahIxUZLaQttl5FZv2YzXs6", "Appendix_17.2_Data_Tables.xls"))],
    "literature": [(n, u, lic, d) for n, u, lic, d in (
        ("basu_manolakos_2010_umass_wp2010-04.pdf", "https://www.umass.edu/economics/publications/2010-04.pdf",
         "author working paper", "Basu & Manolakos (2010) UMass Econ WP 2010-04, working version of RRPE 45(1) 2013"),
        ("basu_huato_larajauregui_wasner_umass_wp318.pdf",
         "https://scholarworks.umass.edu/server/api/core/bitstreams/69ff54a4-6554-48b9-9b5b-58008cb25725/content",
         "author working paper", "Basu, Huato, Lara Jauregui, Wasner, World Profit Rates 1960-2019, UMass Econ WP 318"),
        ("tsoulfidis_paitaridis_2017_mpra84035.pdf", "https://mpra.ub.uni-muenchen.de/84035/1/MPRA_paper_84035.pdf",
         "MPRA (author)", "Tsoulfidis & Paitaridis, MPRA 84035 (working version of the 2019 follow-up on unproductive activities "
         "in the US 1964-2015); NOT the 2012 RRPE paper"),
        ("kliman_2009_persistent_fall_data_graphs.xls", "http://akliman.squarespace.com/storage/Persistent%20Fall%20Data%20Graphs.xls",
         "author (research use; see 'citation' sheet)", "Kliman (2009) The Persistent Fall in Profitability: spreadsheet "
         "accompanying the paper (US corporations 1929-2007); not the 2012 book tables"),
        ("mpra_81542.pdf", "https://mpra.ub.uni-muenchen.de/81542/1/MPRA_paper_81542.pdf", "MPRA (author)",
         "MPRA 81542 (found in search for Paitaridis-Tsoulfidis 2012; contents recorded in availability file)"))],
    "dumenil_levy": [(n, "https://www.cepremap.fr/membres/dlevy/" + n, "authors (research use)", d) for n, d in (
        ("dle2016e.htm", "Dumenil & Levy (2016) note: historical trends of technology and distribution in the US economy since 1869"),
        ("dle2016e.pdf", "Dumenil & Levy (2016) PDF of the note"),
        ("uslt.txt", "Dumenil & Levy long-term US data table (private nonresidential economy, since 1869)"),
        ("dle1994e.pdf", "Dumenil & Levy (1994) sources and construction of the series 1869-1992"))],
    "exiobase": [(n, f"https://zenodo.org/records/23039074/files/{n}?download=1",
                  "EXIOBASE 3 (customized CC BY-NC-SA, non-commercial; see LICENSE.txt)",
                  "EXIOBASE 3.11.2 (Zenodo record 23039074, 2026-09-30) " + d) for n, d in (
        [("LICENSE.txt", "licence"), ("Changelog.txt", "changelog"), ("hashes.csv", "file hashes")]
        + [(f"IOT_{y}_ixi.zip", f"industry-by-industry MRIO with satellite accounts, {y}") for y in range(1995, 2025)])],
    "pwt_docs": [(n, u, "GGDC (CC BY 4.0)", d) for n, u, d in (
        ("pwt110_user_guide_to_data_files.pdf", "https://dataverse.nl/api/access/datafile/554025", "PWT 11.0 user guide to data files"),
        ("pwt110_whatsnew.pdf", "https://dataverse.nl/api/access/datafile/554031", "What is new in PWT 11.0"),
        ("pwt110_capital_detail.xlsx", "https://dataverse.nl/api/access/datafile/554029",
         "PWT 11.0 capital detail (capital stocks by asset type)"),
        ("the_next_generation_of_the_penn_world_table.pdf",
         "https://www.rug.nl/ggdc/docs/the_next_generation_of_the_penn_world_table.pdf",
         "Feenstra, Inklaar, Timmer (2015) The Next Generation of the Penn World Table"),
        ("pwt91_capitalservices_ipmrevision.pdf", "https://www.rug.nl/ggdc/docs/pwt91_capitalservices_ipmrevision.pdf",
         "Capital services in PWT 9.1 (capital methodology note)"))],
    "capital_matrices": [(f"Kbar_exio_v3_8_2_{y}_cfc_{k}.mat",
                          f"https://zenodo.org/records/7073276/files/Kbar_exio_v3_8_2_{y}_cfc_{k}.mat?download=1",
                          "CC BY 4.0", f"Sodersten et al. capital use matrices for EXIOBASE 3.8.2, {y}, {k}")
                         for y in range(1995, 2021) for k in ("pxi", "pxp")],
    "oecd_tim": [("timbc_2023_empn.csv",
                  "https://sdmx.oecd.org/public/rest/data/OECD.STI.PIE,DSD_TIMBC_2023@DF_TIMBC_2023,/EMPN...W.PS.A.?format=csv",
                  "OECD (CC BY 4.0)", "TiMBC 2023 edition: employment by industry (ICIO 2023 aggregates) and workforce "
                  "characteristic")] + [
        (f"tim_{e}_empn_labr.csv",
         f"https://sdmx.oecd.org/public/rest/data/OECD.STI.PIE,DSD_TIM_{e}@DF_TIM_{e},/EMPN+LABR...W..A?format=csv",
         "OECD (CC BY 4.0)", f"TiM {e} edition (trade in employment): employment EMPN and compensation of employees LABR "
         f"by ICIO {e} industry") for e in (2023, 2025)],
    # NIPA tables 6.7-6.9 were discontinued on 2024-07-25 (BEA API: "Table was discontinued due to budget constraints");
    # the last vintage that still contains them is archived at apps.bea.gov/histdata (NIPA, 2024 Q1 third estimate)
    "nipa_hours_archive": [(
        "nipa_section6_vintage_2024-06-28.xlsx",
        "https://apps.bea.gov/HistData/Files/Releases/GDP_and_PI/2024/Q1/Third_June-28-2024/Section6all_xls.xlsx",
        "BEA (public domain)", "BEA data archive, NIPA section 6 as published 2024-06-28 (last vintage before tables "
        "6.7-6.9 were discontinued on 2024-07-25)")],
    # files the user uploaded to a shared Google Drive folder (1rHZwXJnBYZHA79CpmDE0XvU5XwesT22X), 2026-10-16
    "user_gdrive": [(n, f"https://drive.usercontent.google.com/download?id={i}&export=download&confirm=t", lic,
                     "supplied by the user via Google Drive folder 1rHZwXJnBYZHA79CpmDE0XvU5XwesT22X; " + d)
                    for n, i, lic, d in (
        ("mohun_2005_cje_bei043.pdf", "1lLgIJe4nOCvzHGUVD2vsoS9qRNMkzrwM", "publisher copyright (research copy)",
         "Mohun (2005) On measuring the wealth of nations: the US economy, 1964-2001, CJE 29(5), doi:10.1093/cje/bei043"),
        ("mohun_2014_rrpe_0486613413506080.pdf", "1SOM_IPNee1d9XjgI6hcEui4bZlI7eX6d", "publisher copyright (research copy)",
         "Mohun (2014) Unproductive Labor in the U.S. Economy 1964-2010, RRPE 46(3), doi:10.1177/0486613413506080"),
        ("paitaridis_tsoulfidis_2012_rrpe_0486613411423899.pdf", "1h9P0TIlhKmB6qEranC9K4fXM6JRgOU6U",
         "publisher copyright (research copy)", "Paitaridis & Tsoulfidis (2012) The Growth of Unproductive Activities, the "
         "Rate of Profit, and the Phase-Change of the U.S. Economy, RRPE 44(2), doi:10.1177/0486613411423899"),
        ("icio_1995-2000_SML.zip", "1pTMQR9sn1tml8D4kRD5Rwlqs9Unq6MFL", "OECD (CC BY 4.0)", "OECD ICIO SML 1995-2000"),
        ("icio_2001-2005_SML.zip", "1OZQpGjz_xxtIbkNtuWPTUdzxkOGJnIg9", "OECD (CC BY 4.0)", "OECD ICIO SML 2001-2005"),
        ("icio_2006-2010_SML.zip", "1fQp4TyanT5r7uFJPsCGQZujoobh2edVF", "OECD (CC BY 4.0)", "OECD ICIO SML 2006-2010"),
        ("icio_2011-2015_SML.zip", "1FNJQwWSoh9rol_GlPTzvEpt6u50FELLf", "OECD (CC BY 4.0)", "OECD ICIO SML 2011-2015"),
        ("icio_2016-2020_SML.zip", "17RYNv0rhbG0aI8S-6rBpEoKHIuwBWnIM", "OECD (CC BY 4.0)", "OECD ICIO SML 2016-2020"))],
    "wiod_energy": [(n, "https://dataverse.nl/api/access/datafile/" + i, "WIOD (CC BY 4.0)",
                     "WIOD 2013 release (dataverse.nl doi:10.34894/XDTAUZ) " + d) for n, i, d in (
        ("wiod2013_Environmental_Accounts.zip", "199110", "environmental accounts (energy use, emissions, other), 1995-2009"),
        ("wiod2013_Environmental_Sources.pdf", "199109", "sources and methods of the environmental accounts"))],
    "wiod_energy_docs": [(n, u, "authors / JRC (documentation)", d) for n, u, d in (
        ("kulionis_2018_arxiv_1810.07112.pdf", "https://arxiv.org/pdf/1810.07112",
         "Kulionis, Constructing energy accounts for WIOD 2016 release (arXiv 1810.07112), methodology of the TIB dataset"),
        ("jrc116234_wiod_based_energy_and_co2_accounts.pdf",
         "https://publications.jrc.ec.europa.eu/repository/bitstream/JRC116234/jrc116234_wiod_based_energy_and_co2_accounts_final_pubsy.pdf",
         "Corsatea et al. (2019) WIOD environmental accounts update 2000-2016, JRC116234 / EUR 29727 EN (report only)"))],
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


_ZENODO = {}


def zenodo_md5(url):
    """expected md5 of a Zenodo record file (from the record API), or None for other URLs"""
    m = re.match(r"https://zenodo\.org/records/(\d+)/files/([^?]+)", url)
    if not m:
        return None
    rec, name = m.groups()
    if rec not in _ZENODO:
        req = urllib.request.Request(f"https://zenodo.org/api/records/{rec}", headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=120) as r:
            _ZENODO[rec] = {f["key"]: f["checksum"].split(":", 1)[1] for f in json.load(r)["files"]}
    return _ZENODO[rec].get(name)


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
                expected = r.headers.get("Content-Length")
                shutil.copyfileobj(r, f, 1 << 20)
            # a dropped connection can end the copy without an exception: check completeness
            if expected is not None and out.stat().st_size != int(expected):
                raise IOError(f"truncated: {out.stat().st_size} of {expected} bytes")
            md5 = zenodo_md5(url)
            if md5 is not None:
                got = hashlib.md5()
                with open(out, "rb") as f:
                    for b in iter(lambda: f.read(1 << 20), b""):
                        got.update(b)
                if got.hexdigest() != md5:
                    raise IOError(f"md5 mismatch: {got.hexdigest()} != {md5}")
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
        st = st if isinstance(st, str) else "failed"
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


def register_local(source, name, path, origin, licence, note):
    """copy a file the user supplied (chat upload, shared folder) into RAW/<source>/ and add its MANIFEST row;
    origin is where it came from (no download URL exists for this session)"""
    d = RAW / source
    d.mkdir(parents=True, exist_ok=True)
    out = d / name
    shutil.copyfile(path, out)
    sha, size = hashlib.sha256(), out.stat().st_size
    with open(out, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            sha.update(b)
    with open(MANIFEST, "a", newline="") as f:
        csv.writer(f).writerow(["v13", f"v13/{source}/{name}", origin, licence, f"iteration 13 stage D, {note}", size,
                                sha.hexdigest(), dt.datetime.now().isoformat(timespec="seconds")])
    log_attempt(origin, "user-supplied file registered", f"register {source}/{name}")
    print("ok", source, name, size, flush=True)


def annotate(source, text):
    """append a content description to the MANIFEST notes of this source's v13 rows (and set the licence column to
    the one in SOURCES, which was corrected for EXIOBASE after the first rows were written)"""
    licence = {u: lic for _, u, lic, _ in SOURCES.get(source, [])}
    rows = list(csv.reader(open(MANIFEST, newline="")))
    for r in rows[1:]:
        if r[0] == "v13" and (r[1].startswith(f"v13/{source}/") or r[1].startswith(f"EXTERNAL:LT_external/v13/{source}/")):
            r[3] = licence.get(r[2], r[3])
            if text not in r[4]:
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
    "shaikh_capitalism": ("contents: 15 files from Shaikh's public folder 'Shaikh Capitalism Data' (dated 2024-01-24): data "
                          "tables for book appendices 2.2, 4.2 (docx), 5.3, 6.8 (original and corrected: business accounts, "
                          "noncorporate measures, GPIM capital stock tests, net and gross capital, corporate profit rates, "
                          "capacity utilization), 7.2 (Salter data, US industry and OECD profit rates), 8.1, 9.3 (US IO "
                          "tables 1947-1998, vertically integrated unit labour costs), 10.2, 11.2, 12.2, 14.3, 15.2, 16.2, 17.2"),
    "literature": ("contents: Kliman (2009) Persistent Fall spreadsheet (sheets citation, sources/defs, basic data, ROP "
                   "data, misc graph data; US corporations 1929-2007; use requires citation per sheet 'citation'); "
                   "Tsoulfidis, Tsimis, Paitaridis MPRA 84035 (2018, unproductive activities in the US 1964-2015); "
                   "Tsoulfidis & Paitaridis MPRA 81542 (2017, capital intensity, unproductive activities and the Great "
                   "Recession). Not obtained: Basu & Manolakos WP 2010-04 (404), Basu et al. WP 318 (403)"),
    "dumenil_levy": ("contents: Dumenil & Levy (2016) long-term data for the US private nonresidential economy, annual "
                     "1869-2015 (uslt.txt): net domestic product (current and 2009 $ bn), compensation of employees + "
                     "self-employed ($ bn), FTE employees + self-employed (millions), net stock of fixed capital (current "
                     "and 2009 $ bn), total hours worked (billions), GDP ($ bn); note (htm, pdf) and 1994 sources paper"),
    "exiobase": ("contents: EXIOBASE 3.11.2 industry-by-industry MRIO, one zip per year 1995-2024; 49 regions (44 countries "
                 "+ 5 rest-of-world) x 163 industries; Z, Y, x (M.EUR, current prices); satellite folders incl. "
                 "employment/F.txt: persons (1000 p) and hours (M.hr) by skill (low, medium, high) x sex, and "
                 "factor_inputs/F.txt: compensation of employees by skill, consumption of fixed capital, rents on land, "
                 "royalties on resources, remaining net operating surplus, taxes (M.EUR); no capital stocks; files over "
                 "100 MB kept outside the repository; md5 checked against the Zenodo record; licence: customized "
                 "CC BY-NC-SA derivative for non-commercial academic use, publications must not include raw data points "
                 "or coefficients (LICENSE.txt)"),
    "capital_matrices": ("contents: Sodersten, Wood et al. capital use matrices Kbar (use of capital goods by industries, "
                         "M.EUR, from EXIOBASE consumption of fixed capital benchmarked to World Bank CFC), MATLAB .mat, "
                         "1995-2020, pxi and pxp; built on EXIOBASE 3.8.2, not on 3.11.2 (version mismatch; the newer "
                         "3.10.2-based record zenodo 20762989, pxp 1995-2022, is access-restricted)"),
    "pwt_docs": ("contents: PWT 11.0 documentation; user guide states cn (capital stock) is built from investment in 4 "
                 "asset groups: structures (including residential and non-residential), machinery, transport equipment, "
                 "other assets; capital detail file: 180 countries, 1950-2023, investment, net stock, consumption and "
                 "capital services by these 4 assets (current national prices and price indices)"),
    "oecd_tim": ("contents: OECD SDMX extracts. TiMBC 2023: employment (persons, thousands) for 67 areas x 18 ICIO 2023 "
                 "activity aggregates x 12 workforce characteristics (ISCO, ISCED, age, sex groups and total), 2008-2020, "
                 "no hours. TiM 2023 / 2025: employment EMPN (persons, thousands) and compensation of employees LABR "
                 "(USD million and share of value added) for 93 / 97 areas x 70 / 80 ICIO activities and aggregates, "
                 "1995-2020 / 1995-2022, no hours"),
    "nipa_hours_archive": ("contents: NIPA section 6 workbook as published 2024-06-28 (BEA data archive), sheets incl. "
                           "T60900B-A 1948-1987, T60900C-A 1987-2000, T60900D-A 2000-2022 (hours worked by full-time and "
                           "part-time employees, millions of hours, about 20 sector lines: BEA published these tables at "
                           "sector level only), T60700A-D (self-employed persons), T60800A-D (persons engaged in "
                           "production, 6.8D about 100 lines); 6.9 data as published 2023-09-29; tables 6.7-6.9 were "
                           "discontinued 2024-07-25 and are not available from the BEA API"),
    "user_gdrive": ("contents: three article PDFs (Mohun 2005, 17 pages; Mohun 2014, 26 pages; Paitaridis & Tsoulfidis "
                    "2012) and OECD ICIO 2023 edition, SML version (no China / Mexico split), one csv per year 1995-2020 "
                    "in five zips: 76 economies + ROW (77 areas) x 45 industries (ISIC Rev. 4, codes as A01_02, "
                    "B05_06, C10T12), final demand columns HFCE, NPISH, GGFC, GFCF, INVNT, DPABR; rows TLS, VA, OUT; "
                    "USD million; edition identified from industry codes and years (matches TiM 2023)"),
    "wiod_energy": ("contents: WIOD 2013 release (May 2012) environmental accounts, one xls per country (40 countries "
                    "+ RoW), sheets 1995-2009: EU gross energy use and EM emission relevant energy use (TJ) by 35 WIOD "
                    "industries (NACE Rev. 1.1 / ISIC Rev. 3) + households x 26 energy commodities, losses, total; CO2, "
                    "AIR, land, materials, water archives; matches WIOD 2013 (35 industries), not WIOD 2016 (56 "
                    "industries) used in the project"),
    "wiod_energy_docs": ("contents: methodology documents only (Kulionis, arXiv 1810.07112; JRC116234 / EUR 29727 EN, WIOD "
                         "environmental accounts 2000-2016); the WIOD 2016 energy account data files were not obtained "
                         "(the WIOD 2013 release accounts are in wiod_energy)"),
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
