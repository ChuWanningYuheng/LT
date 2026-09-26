"""Download the additional public data of iteration 3 into data/raw/v3/ (+ MANIFEST_v3.csv with sha256).

BLS requires a descriptive User-Agent.  Eurofound (EWCS) answers 429 to scripted requests and the EWCS
micro data require registration (UK Data Service): not downloaded (A-V3-EWCS).
"""
from __future__ import annotations

import hashlib
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw" / "v3"
UA = "LT-research-script/1.0 (noreply@anthropic.com)"
C13 = "USA+GBR+JPN+KOR+NLD+POL+AUT+CZE+DEU+ESP+FRA+ITA+MEX"
ES = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/{}?format=TSV&compressed=true"
ILO = "https://sdmx.ilo.org/rest/data/ILO,{},1.0/" + C13 + ".A{}?startPeriod=2008&format=csv"
OECD = "https://sdmx.oecd.org/public/rest/data/{}/all?format=csvfilewithlabels"

FILES = {
    "oesm22nat.zip": "https://www.bls.gov/oes/special-requests/oesm22nat.zip",
    "db_29_0_text.zip": "https://www.onetcenter.org/dl_files/database/db_29_0_text.zip",
    "ISCO_SOC_Crosswalk.xls": "https://www.bls.gov/soc/ISCO_SOC_Crosswalk.xls",
    "soc_2010_to_2018_crosswalk.xlsx": "https://www.bls.gov/soc/2018/soc_2010_to_2018_crosswalk.xlsx",
    "ilo_eco_ocu.csv": ILO.format("DF_EMP_TEMP_ECO_OCU_NB", "......"),
    "ilo_EHRA_SEX_OCU_CUR_NB.csv": ILO.format("DF_EAR_EHRA_SEX_OCU_CUR_NB", "......."),
    "ilo_EMTA_SEX_OCU_CUR_NB.csv": ILO.format("DF_EAR_EMTA_SEX_OCU_CUR_NB", "......."),
    "eag_rel_upper.csv": OECD.format("OECD.EDU.IMEP,DSD_EAG_LSO_EA@DF_LSO_EARN_REL_UPPER,"),
    "uoe_stud_totals.csv": OECD.format("OECD.EDU.IMEP,DSD_EAG_UOE_NON_FIN_STUD@DF_UOE_NF_STUD_TOTALS,"),
    "cbc.csv": "https://sdmx.oecd.org/public/rest/data/OECD.ELS.SAE,DSD_TUD_CBC@DF_CBC,/all?startPeriod=2008&format=csvfilewithlabels",
    "pop_wb.json": "https://api.worldbank.org/v2/country/USA;GBR;JPN;KOR;NLD;POL;AUT;CZE;DEU;ESP;FRA;ITA;MEX/"
                   "indicator/SP.POP.TOTL?date=2008:2023&format=json&per_page=1000",
    **{f"{t}.tsv.gz": ES.format(t) for t in ("lfsa_eisn2", "earn_ses18_14", "earn_ses18_16", "earn_ses18_47",
                                              "earn_ses18_54", "educ_uoe_enra02", "lfsa_egised")},
}


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    rows = ["file,url,sha256"]
    for name, url in FILES.items():
        out = RAW / name
        if not out.exists():
            subprocess.run(["curl", "-sS", "-f", "-A", UA, "--max-time", "600", "-o", str(out), url], check=True)
        rows.append(f"{name},{url},{hashlib.sha256(out.read_bytes()).hexdigest()}")
        print("ok", name, flush=True)
    with zipfile.ZipFile(RAW / "oesm22nat.zip") as z:
        z.extractall(RAW)
    with zipfile.ZipFile(RAW / "db_29_0_text.zip") as z:
        (RAW / "onet").mkdir(exist_ok=True)
        for m in ("Job Zones.txt", "Job Zone Reference.txt"):
            (RAW / "onet" / m).write_bytes(z.read(f"db_29_0_text/{m}"))
    (RAW / "MANIFEST_v3.csv").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
