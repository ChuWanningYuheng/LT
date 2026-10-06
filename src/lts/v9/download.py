"""Iteration 9, stage 0: downloads into data/raw/v9/ and rows in data/raw/MANIFEST.csv.

Usage: PYTHONPATH=src python -m lts.v9.download [name ...]   (no names: everything not yet present)
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import sys
import time
import urllib.request

from ..v6.series import ROOT

RAW = ROOT / "data" / "raw" / "v9"
RAW.mkdir(parents=True, exist_ok=True)
UA = "LT-research-script/1.0 (noreply@anthropic.com)"
OECD = "https://sdmx.oecd.org/public/rest/data/"
ES = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
ESB = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/"
FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id="
CC = "CC BY 4.0"

FILES = [
    # stage 2: wages, prices, unemployment, institutions (OECD)
    ("oecd_hou_ear.csv", OECD + "OECD.SDD.TPS,DSD_EAR@DF_HOU_EAR,/all?format=csv", CC, "OECD hourly earnings"),
    ("oecd_prices_all.csv", OECD + "OECD.SDD.TPS,DSD_PRICES@DF_PRICES_ALL,/.M+Q+A.N.CPI.IX._T+CP01+CP045_0722.N.?format=csv", CC,
     "OECD CPI index: all items, food, energy"),
    ("oecd_ulc_q.csv", OECD + "OECD.SDD.TPS,DSD_PDB@DF_PDB_ULC_Q,/all?format=csv", CC, "OECD ULC and productivity, quarterly"),
    ("oecd_une_q.csv", OECD + "OECD.SDD.TPS,DSD_LFS@DF_IALFS_UNE_Q,/all?format=csv", CC, "OECD unemployment, quarterly"),
    ("oecd_cbc.csv", OECD + "OECD.ELS.SAE,DSD_TUD_CBC@DF_CBC,/all?format=csv", CC, "OECD/AIAS collective bargaining coverage"),
    ("oecd_tud.csv", OECD + "OECD.ELS.SAE,DSD_TUD_CBC@DF_TUD,/all?format=csv", CC, "OECD/AIAS trade union density"),
    ("oecd_rmw.csv", OECD + "OECD.ELS.SAE,DSD_EARNINGS@RMW,/all?format=csv", CC, "OECD real minimum wages"),
    ("oecd_av_an_wage.csv", OECD + "OECD.ELS.SAE,DSD_EARNINGS@AV_AN_WAGE,/all?format=csv", CC, "OECD average annual wages"),
    # stage 2: Eurostat
    ("es_hicp_midx.json", ES + "prc_hicp_midx?unit=I15&coicop=CP00&coicop=CP01&coicop=NRG", CC, "HICP monthly index: all, food, energy"),
    ("es_lci_q.json", ES + "lc_lci_r2_q?unit=I20&s_adj=SCA&lcstruct=D1_D4_MD5&nace_r2=B-S", CC, "labour cost index, wages, quarterly"),
    ("es_lci_q_wag.json", ES + "lc_lci_r2_q?unit=I20&s_adj=SCA&lcstruct=D11&nace_r2=B-S", CC, "labour cost index, wages and salaries D11"),
    ("es_ppi_m.json", ES + "sts_inppd_m?unit=I21&s_adj=NSA&indic_bt=PRC_PRR_DOM" + "".join(
        f"&nace_r2={n}" for n in ("B", "B05", "B06", "B07", "C10", "C19", "C20", "C24", "C17", "C16", "C23", "D", "D35")),
     CC, "PPI domestic, monthly, selected NACE"),
    ("es_une_q.json", ES + "une_rt_q?s_adj=SA&age=Y15-74&sex=T&unit=PC_ACT", CC, "unemployment rate, quarterly"),
    ("es_slack_q.json", ES + "lfsi_sla_q?s_adj=SA&sex=T&age=Y15-74&unit=PC_ELF&wstatus=SLACK", CC, "labour market slack, quarterly"),
    # stage 2: US
    *[(f"fred_{s}.csv", FRED + s, "public domain (via FRED)", f"FRED {s}") for s in
      ("AHETPI", "CPIAUCSL", "CPIUFDSL", "CPIENGSL", "UNRATE", "U6RATE", "WPU01", "WPU0561", "WPU1011", "WPU1012",
       "WPU1017", "PCU324110324110", "PCU311311", "PCU3311103311105", "PPIENG", "WPU02", "WPU061", "PCU325110325110",
       "OPHNFB", "COMPRNFB", "PNFIC1")],
    ("bls_ci.data.1.AllData", "https://download.bls.gov/pub/time.series/ci/ci.data.1.AllData", "public domain (BLS)", "ECI all data"),
    ("bls_ci.series", "https://download.bls.gov/pub/time.series/ci/ci.series", "public domain (BLS)", "ECI series"),
    # stages 3-6
    ("pld2023_dataset.xlsx", "https://dataverse.nl/api/access/datafile/383800", CC, "GGDC Productivity Level Database 2023"),
    ("ggdc_benchmark_2005.xlsx", "https://www.rug.nl/ggdc/docs/benchmark_2005.xlsx", CC,
     "GGDC PLD 2005 benchmark (WIOD industries; Inklaar & Timmer 2014)"),
    ("pwt110.xlsx", "https://dataverse.nl/api/access/datafile/554105", CC, "Penn World Table 11.0"),
    ("hickel2024_MOESM4.xlsx", "https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41467-024-"
     "49687-y/MediaObjects/41467_2024_49687_MOESM4_ESM.xlsx", CC, "Hickel et al. 2024 Nat Commun, supplementary data"),
    ("hickel2024_MOESM1.pdf", "https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41467-024-"
     "49687-y/MediaObjects/41467_2024_49687_MOESM1_ESM.pdf", CC, "Hickel et al. 2024, supplementary information"),
    ("exiobase_IOT_2011_ixi.zip", "https://zenodo.org/records/5589597/files/IOT_2011_ixi.zip?download=1", CC,
     "EXIOBASE 3.8.2 IOT 2011 ixi"),
    ("exiobase_IOT_2017_ixi.zip", "https://zenodo.org/records/5589597/files/IOT_2017_ixi.zip?download=1", CC,
     "EXIOBASE 3.8.2 IOT 2017 ixi"),
    ("exiobase_IOT_2017_pxp.zip", "https://zenodo.org/records/5589597/files/IOT_2017_pxp.zip?download=1", CC,
     "EXIOBASE 3.8.2 IOT 2017 pxp"),
    ("exiobase_IOT_2021_pxp.zip", "https://zenodo.org/records/5589597/files/IOT_2021_pxp.zip?download=1", CC,
     "EXIOBASE 3.8.2 IOT 2021 pxp"),
    # stage 7-8
    ("bea_ValueAdded.xlsx", "https://apps.bea.gov/industry/Release/XLS/GDPxInd/ValueAdded.xlsx", "public domain (BEA)",
     "BEA GDP by industry: value added"),
    ("bea_GrossOutput.xlsx", "https://apps.bea.gov/industry/Release/XLS/GDPxInd/GrossOutput.xlsx", "public domain (BEA)",
     "BEA GDP by industry: gross output"),
    ("bea_IntermediateInputs.xlsx", "https://apps.bea.gov/industry/Release/XLS/GDPxInd/IntermediateInputs.xlsx",
     "public domain (BEA)", "BEA GDP by industry: intermediate inputs"),
    ("bls_ip.data.1.AllData", "https://download.bls.gov/pub/time.series/ip/ip.data.1.AllData", "public domain (BLS)",
     "BLS labour productivity, detailed industries (hours)"),
    *[(f"bls_ip.{k}", f"https://download.bls.gov/pub/time.series/ip/ip.{k}", "public domain (BLS)", f"BLS ip {k}")
      for k in ("series", "industry", "measure", "sector", "duration")],
    ("mpra_81542.pdf", "https://mpra.ub.uni-muenchen.de/81542/1/MPRA_paper_81542.pdf", "author (MPRA)",
     "Tsoulfidis & Paitaridis 2017, MPRA 81542 (classification and estimates, Appendix 1)"),
    ("mpra_84035.pdf", "https://mpra.ub.uni-muenchen.de/84035/1/MPRA_paper_84035.pdf", "author (MPRA)",
     "Paitaridis & Tsoulfidis, MPRA 84035"),
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
            time.sleep(2 ** (k + 1))
    else:
        print("FAILED", name, url, flush=True)
        return False
    data = out.read_bytes()
    with open(ROOT / "data" / "raw" / "MANIFEST.csv", "a", newline="") as f:
        csv.writer(f).writerow(["v9", name, url, licence, f"iteration 9, {note}", len(data),
                                hashlib.sha256(data).hexdigest(), dt.datetime.now().isoformat(timespec="seconds")])
    print("ok", name, len(data), flush=True)
    return True


if __name__ == "__main__":
    want = set(sys.argv[1:])
    for name, url, lic, note in FILES:
        if (want and name not in want) or (not want and (RAW / name).exists()):
            continue
        fetch(name, url, lic, note)
