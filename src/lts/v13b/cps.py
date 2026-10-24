"""Iteration 13b, stage 0.2-0.3: March CPS / ASEC microdata, survey years 1962-2025 (pre_registration_v13b.md).

Fixed-width readers by era. Positions and codes are taken from the documentation in data/raw/v13b (NBER March CPS
descriptions cpsmarYY.pdf / .ddf, Census technical documentation cpsmarYY.pdf, ASEC data dictionaries); the notes
next to each era name the document. One row per person: civilian, 18-64, private wage and salary worker, employed in
the survey week, positive wage and salary income and weeks worked last year. The file of survey year t describes
earnings year t - 1 (column `eyear`); occupation and industry are those of the current (survey-week) job.

Harmonised columns: eyear, wgt, age, female, S (years of schooling), grp (industry group G1-G10, None outside),
sup (supervisor, main definition), sup_broad, nonprod (non-production occupation; variant 'composition by CES'),
wage (annual wage and salary income), weeks, hrs_week (hours last week, at work only), hrs_year_week (weekly hours
used for annual hours: usual hours last year from survey year 1976, hours last week before), topcoded (flag).

Usage: PYTHONPATH=src python3 -P -m lts.v13b.cps [years...]  ->  data/raw/v13b_cache/cps_<year>.parquet (not in git)
"""
from __future__ import annotations

import io
import sys
import zipfile

import numpy as np
import pandas as pd

from ..v6.series import ROOT

RAW = ROOT / "data" / "raw" / "v13b"
EXT = ROOT.parent / "LT_external" / "v13b"
CACHE = ROOT / "data" / "raw" / "v13b_cache"
SURVEY_YEARS = [1962, 1964, 1966, 1967, 1968, 1969] + list(range(1971, 2026))
KAPPA = (1600 + 347) / (40 * 1800)            # Hilferding, iterations 3 and 7 (pre-registration 0.3)

zipfile.ZipInfo._decodeExtra = lambda self, *a, **k: None   # 21 NBER zips: corrupt extra field, CRC verified


# ------------------------------------------------------------------ files
def lines(year):
    """Raw person-file lines (bytes, without line ends) of a survey year."""
    yy = f"{year % 100:02d}"
    if year <= 1997:
        z = zipfile.ZipFile(RAW / "nber_cps" / f"cpsmar{yy}.zip")
    else:
        names = {1998: "1998_mar98supp.zip", 1999: "1999_mar99supp.zip", 2000: "2000_mar00supp.zip",
                 2001: "2001_mar01supp.zip", 2002: "2002_mar02supp.zip", 2003: "2003_asec2003.zip",
                 2004: "2004_asec2004.zip", 2005: "2005_asec2005_pubuse.zip", 2006: "2006_asec2006_pubuse.zip",
                 2007: "2007_asec2007_pubuse_tax2.zip", 2014: "2014_asec2014_pubuse_tax_fix_5x8_2017.zip",
                 2016: "2016_asec2016_pubuse_v3.zip"}
        nm = names.get(year, f"{year}_asec{year}_pubuse.zip")
        z = zipfile.ZipFile(RAW / "census_asec" / nm)
    with z.open(z.infolist()[0]) as f:
        for ln in io.BufferedReader(f, 1 << 22):
            yield ln.rstrip(b"\r\n")


def lines_2014_3x8():
    z = zipfile.ZipFile(RAW / "census_asec" / "2014_asec2014_pubuse_3x8_rerun_v2.zip")
    with z.open(z.infolist()[0]) as f:
        for ln in io.BufferedReader(f, 1 << 22):
            yield ln.rstrip(b"\r\n")


# ------------------------------------------------------------------ era layouts: name -> (start, length), 1-based
L62 = dict(pop=(124, 1), esr=(15, 2), hrs=(31, 3), ind=(37, 2), occ=(41, 2), occmaj=(43, 2), cow=(46, 1),
           wgt=(67, 12), age=(95, 2), colsex=(8, 1), wage=(133, 6), wrec=(163, 2), match=(173, 1), wks=(181, 1),
           edlev=(205, 1), edgr=(206, 1), edfin=(207, 1))
# 1964-1967: the description gives 0-based positions; stored here as 1-based (+1)
L64 = dict(wgt=(21, 10), age=(34, 2), agerec=(36, 1), sex=(38, 1), esr=(48, 2), hrs=(51, 2), cow=(55, 1),
           ind=(67, 2), occ=(71, 2), occmaj=(73, 2), educ=(79, 2), wks=(103, 1), wage=(131, 10))
L68 = dict(pop=(1, 1), age=(29, 2), sex=(32, 1), grade=(57, 2), gradec=(59, 1), wage=(67, 6), wks=(131, 1),
           esr=(152, 1), hrs=(153, 2), occ3=(172, 3), cow=(175, 1), occ=(179, 2), occmaj=(185, 2), ind=(187, 2),
           ind3=(169, 3), wgt=(205, 12))
L76 = dict(ppos=(7, 2), esr=(12, 1), hrs=(16, 2), ind3=(49, 3), indrec=(52, 2), occ3=(56, 3), cow=(59, 1),
           sex=(108, 1), age=(110, 2), grade=(115, 2), gradec=(117, 1), wgt=(118, 11), wks=(134, 2), uhrs=(136, 2),
           wage=(191, 5))
L88 = dict(rec=(1, 1), age=(15, 2), sex=(20, 1), hga=(22, 2), hgc=(24, 1), wgt=(66, 8), hrs=(76, 2), occ3=(106, 3),
           cow=(109, 1), esr=(145, 1), ind=(155, 2), wks=(171, 2), uhrs=(181, 2), wage=(243, 6))
L03 = dict(L88, ind4=(87, 4), occ4=(91, 4))
L03.pop("occ3")
L11 = dict(rec=(1, 1), age=(19, 2), sex=(24, 1), hga=(25, 2), wgt=(155, 8), hrs=(163, 2), ind4=(168, 4),
           occ4=(172, 4), cow=(176, 1), esr=(200, 1), wks=(258, 2), uhrs=(268, 2), wage=(364, 7))


def layout_2019(year):
    """2019-2025: positions from the Census person-record format file of the year."""
    d = {}
    for ln in open(RAW / "census_asec" / f"{year}_persfmt.txt"):
        p = ln.split()
        if len(p) >= 3 and p[1].isdigit() and p[2].isdigit():
            d[p[0]] = (int(p[2]), int(p[1]))
    m = dict(rec="PRECORD", age="A_AGE", sex="A_SEX", hga="A_HGA", wgt="MARSUPWT", hrs="A_HRS1", ind4="PEIOIND",
             occ4="PEIOOCC", cow="A_CLSWKR", esr="A_LFSR", wks="WKSWORK", uhrs="HRSWK", wage="WSAL_VAL")
    return {k: d[v] for k, v in m.items()}


def era(year):
    if year == 1962:
        return "62"
    if year <= 1967:
        return "64"
    if year <= 1971:
        return "68"
    if year <= 1975:
        return "72"
    if year <= 1987:
        return "76"
    if year <= 2002:
        return "88"
    if year <= 2010:
        return "03"
    if year <= 2018:
        return "11"
    return "19"


# ------------------------------------------------------------------ occupation codes (pre-registration, table 0.2)
def rng(a, b):
    return set(range(a, b + 1))


OCC = {
    # 1970 census codes (cpsmar72.pdf, pp. 70-71): managers and administrators exc. farm 201-245; foremen n.e.c. 441;
    # professional 001-195, sales 260-285, clerical 301-395 -> non-production
    "70": dict(sup=rng(201, 245) | {441}, broad=rng(201, 245) | {441}, nonprod=rng(1, 395) | {441}),
    # 1980 census codes (cpsmar84.pdf, appendix B1)
    "80": dict(sup=rng(3, 19) | {503, 633, 613, 803, 843, 863} | rng(553, 558),
               broad=rng(3, 19) | {503, 633, 613, 803, 843, 863} | rng(553, 558) | {243} | rng(303, 307)
               | rng(413, 415) | {433, 448, 456},
               nonprod=rng(3, 389)),
    # 1990 census codes (census_cps_techdocs/cpsmar96.pdf)
    "90": dict(sup=rng(3, 22) | {503, 628, 613, 803, 843, 864} | rng(553, 558),
               broad=rng(3, 22) | {503, 628, 613, 803, 843, 864} | rng(553, 558) | {243} | rng(303, 307)
               | rng(413, 415) | {433, 448, 456},
               nonprod=rng(3, 389)),
    # 2002 census codes (cpsmar03.pdf) and 2010 codes (cpsmar11.pdf)
    "00": dict(sup=rng(10, 430) | {6200, 7000, 7700, 9000},
               broad=rng(10, 430) | {6200, 7000, 7700, 9000, 3700, 3710, 3720, 3730, 4010, 4200, 4210, 4300, 4320,
                                     4700, 4710, 5000},
               nonprod=rng(10, 3540) | rng(4700, 5940)),
    # 2018 codes (cpsmar20.pdf)
    "18": dict(sup=rng(10, 440) | {6200, 7000, 7700, 9005},
               broad=rng(10, 440) | {6200, 7000, 7700, 9005, 3700, 3710, 3720, 3725, 3735, 4010, 4200, 4210, 4330,
                                     4700, 4710, 5000},
               nonprod=rng(10, 3550) | rng(4700, 5940)),
}
for _d in OCC.values():
    _d["nonprod"] = _d["nonprod"] | _d["broad"]


def occ_class(year):
    """Occupation classification of the survey year."""
    if year <= 1971:
        return "60"
    if year <= 1982:
        return "70"
    if year <= 1991:
        return "80"
    if year <= 2002:
        return "90"
    if year <= 2019:
        return "00"
    return "18"


# ------------------------------------------------------------------ industry groups (pre-registration, table 0.3)
GROUPS = {"G1": ["111CA", "113FF"], "G2": ["211", "212", "213"], "G3": ["23"],
          "G4": ["321", "327", "331", "332", "333", "334", "335", "3361MV", "3364OT", "337", "339"],
          "G5": ["311FT", "313TT", "315AL", "322", "323", "324", "325", "326"],
          "G6": ["22", "48", "493", "51"], "G7": ["42", "44RT"], "G8": ["52", "531", "532RL"],
          "G9": ["54", "55", "56"], "G10": ["61", "62", "71", "721", "722", "81"]}
CODE_GROUP = {c: g for g, cs in GROUPS.items() for c in cs}


def _m(d):
    return {k: g for g, ks in d.items() for k in ks}


# 1962 recode 25 (intermediate industry) 00-19
IND62 = _m({"G1": [0, 1], "G2": [2], "G3": [3], "G4": [4], "G5": [5], "G6": [6, 7, 8], "G7": [9, 10], "G8": [11],
            "G9": [12, 17], "G10": [13, 14, 15, 16]})
# 1964-1967 intermediate industry 01-19
IND64 = _m({"G1": [1, 2], "G2": [3], "G3": [4], "G4": [5], "G5": [6], "G6": [7, 8, 9], "G7": [10, 11], "G8": [12],
            "G9": [13, 18], "G10": [14, 15, 16, 17]})
# 1968-1975 major industry P187-188 (00 = agriculture, private household, never worked: not used)
IND68 = _m({"G2": [1], "G3": [2], "G4": [3], "G5": [4], "G6": [5, 6, 7], "G7": [8, 9], "G8": [10], "G9": [11, 18],
            "G10": [12, 13, 14, 15, 16, 17], "G1": [19]})
# 1983-1987 industry recode P52 01-47 (cpsmar84.pdf)
IND83 = _m({"G1": [1, 45], "G2": [2], "G3": [3], "G4": list(range(4, 18)), "G5": list(range(18, 28)),
            "G6": [28, 29, 30], "G7": [31, 32], "G8": [33, 34], "G9": [36, 37, 44],
            "G10": [35, 38, 39, 40, 41, 42, 43]})
# 1988-2002 major industry A-MJIND 01-23 (cpsmar89.ddf, cpsmar96.pdf)
IND88 = _m({"G1": [1, 21], "G2": [2], "G3": [3], "G4": [4], "G5": [5], "G6": [6, 7, 8], "G7": [9, 10], "G8": [11],
            "G9": [13, 20], "G10": [12, 14, 15, 16, 17, 18, 19]})


def ind4_group(c):
    """2003-2025 census industry codes (cpsmar03/11/20/25.pdf, appendix A)."""
    if 170 <= c <= 290:
        return "G1"
    if 370 <= c <= 490:
        return "G2"
    if c == 770:
        return "G3"
    if 2470 <= c <= 3990:
        return "G4"
    if 1070 <= c <= 2390:
        return "G5"
    if 570 <= c <= 690 or 6070 <= c <= 6390 or 6470 <= c <= 6780:
        return "G6"
    if 4070 <= c <= 5790:
        return "G7"
    if 6870 <= c <= 7190:
        return "G8"
    if 7270 <= c <= 7790:
        return "G9"
    if 7860 <= c <= 9290:
        return "G10"
    return None


# ------------------------------------------------------------------ education: years of schooling S
def s_grade(grade, done):
    """1968-1987: highest grade attended 01 none, 02-09 elementary 1-8, 10-13 high school 1-4, 14-19 college 1-6+."""
    s = np.where(grade <= 1, 0, grade - 1).astype(float)
    return np.where((done == 2) & (s > 0), s - 1, s)


def s_hga88(hga, hgc):
    """1988-1991: A-HGA 00 none, 01-08 E1-E8, 09-12 H1-H4, 13-18 C1-C6+; A-HGC 1 completed, 2 not."""
    s = hga.astype(float)
    return np.where((hgc == 2) & (s > 0), s - 1, s)


HGA92 = {31: 0, 32: 2.5, 33: 5.5, 34: 7.5, 35: 9, 36: 10, 37: 11, 38: 11, 39: 12, 40: 13, 41: 14, 42: 14, 43: 16,
         44: 18, 45: 19, 46: 20}
EDUC64 = {1: 0, 2: 2.5, 3: 6, 4: 8, 5: 10, 6: 12, 7: 14, 8: 16, 10: 18}
WKS_MID = {1: 7, 2: 20, 3: 33, 4: 43.5, 5: 48.5, 6: 51}       # 1-13, 14-26, 27-39, 40-47, 48-49, 50-52


# ------------------------------------------------------------------ parsing
def _num(col):
    return pd.to_numeric(pd.Series(col).str.strip().replace("", np.nan), errors="coerce")


def raw_frame(year, layout, keep, src=None):
    rows = {k: [] for k in layout}
    for ln in (src if src is not None else lines(year)):
        if not keep(ln):
            continue
        for k, (a, n) in layout.items():
            rows[k].append(ln[a - 1:a - 1 + n].decode("latin-1"))
    return pd.DataFrame(rows)


def _signed(s):
    """Income field with an alphanumeric minus sign / complemented negatives: negatives -> NaN (not used)."""
    v = pd.to_numeric(s.str.strip(), errors="coerce")
    return v


BR62 = {1: (1, 499), 2: (500, 999), 3: (1000, 1499), 4: (1500, 1999), 5: (2000, 2499), 6: (2500, 2999),
        7: (3000, 3499), 8: (3500, 3999), 9: (4000, 4499), 10: (4500, 4999), 11: (5000, 5999), 12: (6000, 6999),
        13: (7000, 7999), 14: (8000, 9999), 15: (10000, 14999), 16: (15000, 24999), 17: (25000, 99999)}


def _bracket_ok(w, rec):
    """1962: the amount field stores losses complemented; keep amounts consistent with the income-bracket recode
    (163-164; 18-19 = not allocated / not in the income sample)."""
    lo = rec.map({k: v[0] for k, v in BR62.items()})
    hi = rec.map({k: v[1] for k, v in BR62.items()})
    return (w >= lo) & (w <= hi)


def parse(year, src=None):
    e = era(year)
    if e == "62":
        r = raw_frame(year, L62, lambda ln: len(ln) >= 240 and ln[123:124] == b"0", src)
        n = {k: _num(r[k]) for k in r}
        emp = n["esr"].isin([0, 1, 2])
        d = pd.DataFrame(dict(wgt=n["wgt"], age=n["age"], female=n["colsex"].isin([1, 3]),
                              S=np.select([n["edlev"] == 0, n["edlev"] == 1, n["edlev"] == 2, n["edlev"] == 3],
                                          [0, n["edgr"], 8 + n["edgr"], 12 + n["edgr"]], np.nan),
                              priv=n["cow"] == 0, emp=emp, wage=n["wage"].where(_bracket_ok(n["wage"], n["wrec"])),
                              weeks=n["wks"].map({1: 7, 2: 20, 3: 33, 4: 43.5, 5: 48.5, 6: 51})
                              .where(n["match"] == 0),
                              hrs_week=n["hrs"].where((n["hrs"] >= 1) & (n["hrs"] <= 99) & n["esr"].isin([0, 1])),
                              grp=n["ind"].map(IND62)))
        d["S"] = np.where((n["edfin"] == 7) & (d.S > 0), d.S - 1, d.S)
        d["sup"] = n["occ"].isin([7, 16])
        d["sup_broad"] = d.sup
        d["nonprod"] = n["occmaj"].isin([0, 2, 3, 4]) | d.sup
        d["occ"] = n["occ"]
    elif e == "64":
        r = raw_frame(year, L64, lambda ln: ln[:5] == b"00036" and ln[5:10] != b"22222", src)
        n = {k: _num(r[k]) for k in r}
        d = pd.DataFrame(dict(wgt=n["wgt"], age=n["age"], female=n["sex"] == 2, S=n["educ"].map(EDUC64),
                              priv=n["cow"] == 1, emp=n["esr"].isin([1, 2, 3]),
                              wage=_signed(r["wage"]), weeks=n["wks"].map({2: 7, 3: 20, 4: 33, 5: 43.5, 6: 48.5, 7: 51}),
                              hrs_week=n["hrs"].where((n["hrs"] >= 1) & (n["hrs"] <= 98) & n["esr"].isin([1, 2])),
                              grp=n["ind"].map(IND64)))
        d["sup"] = n["occ"].isin([8, 17])
        d["sup_broad"] = d.sup
        d["nonprod"] = n["occmaj"].isin([1, 3, 4, 5]) | d.sup
        d["occ"] = n["occ"]
        d["agerec"] = n["agerec"]
    elif e in ("68", "72"):
        r = raw_frame(year, L68, lambda ln: ln[:1] == b"1", src)
        n = {k: _num(r[k]) for k in r}
        d = pd.DataFrame(dict(wgt=n["wgt"], age=n["age"], female=n["sex"] == 2, S=s_grade(n["grade"], n["gradec"]),
                              priv=n["cow"] == 1, emp=n["esr"].isin([1, 2]),
                              wage=n["wage"].where(n["wage"] < 99999),
                              weeks=n["wks"].map({2: 7, 3: 20, 4: 33, 5: 43.5, 6: 48.5, 7: 51}),
                              hrs_week=n["hrs"].where((n["hrs"] >= 1) & (n["esr"] == 1)),
                              grp=n["ind"].map(IND68)))
        if e == "68":
            # recode P179-180 of 1968-1971 is not the 1963-1967 list: by the cross-table with the 3-digit codes and
            # the class of worker, 07 = salaried managers (1960 codes 250-290; 1971: 1970 codes 201-245), 16 = foremen
            # n.e.c. (430; 1971: 441) - stage-0 check (i), journal item 5
            d["sup"] = n["occ"].isin([7, 16])
            d["nonprod"] = n["occmaj"].isin([1, 2, 3, 4]) | d.sup
            d["occ"] = n["occ"]
        else:
            o = OCC["70"]
            d["sup"] = n["occ3"].isin(o["sup"])
            d["nonprod"] = n["occ3"].isin(o["nonprod"])
            d["occ"] = n["occ3"]
        d["sup_broad"] = d.sup
        d["occ3"] = n["occ3"]
        d["ind3"] = n["ind3"]
        d["ind_major"] = n["ind"]
    elif e == "76":
        # person records: position 7-8 = 01-39 (with record type '3' at 331 from 1977; the 1976 file has no type)
        r = raw_frame(year, L76, lambda ln: len(ln) >= 200 and ln[6:8].isdigit() and 1 <= int(ln[6:8]) <= 39, src)
        n = {k: _num(r[k]) for k in r}
        o = OCC[occ_class(year)]
        d = pd.DataFrame(dict(wgt=n["wgt"], age=n["age"], female=n["sex"] == 2, S=s_grade(n["grade"], n["gradec"]),
                              priv=n["cow"] == 1, emp=n["esr"].isin([1, 2]), wage=n["wage"],
                              weeks=n["wks"].where(n["wks"] >= 1),
                              hrs_week=n["hrs"].where((n["hrs"] >= 1) & (n["esr"] == 1)),
                              uhrs=n["uhrs"].where(n["uhrs"] >= 1)))
        d["grp"] = n["indrec"].map(IND83) if year >= 1983 else None
        d["sup"] = n["occ3"].isin(o["sup"])
        d["sup_broad"] = n["occ3"].isin(o["broad"])
        d["nonprod"] = n["occ3"].isin(o["nonprod"])
        d["occ"] = n["occ3"]
        d["ind3"] = n["ind3"]
    else:
        lay = {"88": L88, "03": L03, "11": L11}.get(e) or layout_2019(year)
        r = raw_frame(year, lay, lambda ln: ln[:1] == b"3", src)
        n = {k: _num(r[k]) for k in r}
        o = OCC[occ_class(year)]
        if e == "88" and year <= 1991:
            S = s_hga88(n["hga"], n["hgc"])
        else:
            S = n["hga"].map(HGA92)
        occ = n["occ3"] if e == "88" else n["occ4"]
        d = pd.DataFrame(dict(wgt=n["wgt"], age=n["age"], female=n["sex"] == 2, S=S, priv=n["cow"] == 1,
                              emp=n["esr"].isin([1, 2]), wage=n["wage"], weeks=n["wks"].where(n["wks"] >= 1),
                              hrs_week=n["hrs"].where((n["hrs"] >= 1) & (n["esr"] == 1)),
                              uhrs=n["uhrs"].where(n["uhrs"] >= 1)))
        d["grp"] = n["ind"].map(IND88) if e == "88" else n["ind4"].map(lambda c: ind4_group(int(c))
                                                                       if pd.notna(c) else None)
        d["sup"] = occ.isin(o["sup"])
        d["sup_broad"] = occ.isin(o["broad"])
        d["nonprod"] = occ.isin(o["nonprod"])
        d["occ"] = occ
    d["year"] = year
    d["eyear"] = year - 1
    if "uhrs" not in d:
        d["uhrs"] = np.nan
    d["hrs_year_week"] = d.uhrs if year >= 1976 else d.hrs_week
    return d


def sample(d):
    """Pre-registration 0.2: 18-64, private wage and salary, employed in the survey week, wage > 0, weeks >= 1."""
    m = (d.age >= 18) & (d.age <= 64) & d.priv & d.emp & (d.wage > 0) & (d.weeks >= 1) & d.S.notna() & (d.wgt > 0)
    return d[m].copy()


def topcode(d, year):
    """Survey years up to 1995: the largest wage value is a top-code threshold if at least 0.05% of earners have it
    (journal: rule for thresholds not printed in the descriptions); x 1.5. From 1996 the public files carry cell
    means / rank-swapped values - kept as is."""
    d["topcoded"] = False
    d["wage"] = d.wage.astype(float)
    if year <= 1995 and len(d):
        mx = d.wage.max()
        if (d.wage == mx).mean() >= 0.0005:
            d.loc[d.wage == mx, "topcoded"] = True
            d.loc[d.topcoded, "wage"] = d.loc[d.topcoded, "wage"] * 1.5
    return d


def ind70_crosswalk():
    """1970 census industry code -> major industry recode P187-188, from the survey years 1972-1975 (documented
    recode); used to group the 3-digit industry of survey years 1976-1982."""
    rows = []
    for y in (1972, 1973, 1974, 1975):
        d = parse(y)
        rows.append(d.loc[d.ind3.notna() & d.ind_major.notna(), ["ind3", "ind_major", "wgt"]])
    x = pd.concat(rows)
    t = x.groupby(["ind3", "ind_major"]).wgt.sum().reset_index()
    t = t.sort_values("wgt").groupby("ind3").tail(1)
    tot = x.groupby("ind3").wgt.sum()
    t["share_mode"] = t.wgt.to_numpy() / tot.reindex(t.ind3).to_numpy()
    t["grp"] = t.ind_major.map(IND68)
    return t.set_index("ind3")


def build(years=SURVEY_YEARS):
    CACHE.mkdir(parents=True, exist_ok=True)
    xw = None
    diag = []
    for y in years:
        d = parse(y)
        if era(y) == "76" and y <= 1982:
            if xw is None:
                xw = ind70_crosswalk()
                xw.to_csv(ROOT / "results" / "v13b" / "s0_cps_ind70_xwalk.csv")
            d["grp"] = d.ind3.map(xw.grp)
        n_all = len(d)
        s = topcode(sample(d), y)
        diag.append(dict(year=y, era=era(y), occ_class=occ_class(y), n_records=n_all, n_sample=len(s),
                         share_female=float(s.female.mean()) if len(s) else np.nan,
                         share_grp_missing=float(s.grp.isna().mean()) if len(s) else np.nan,
                         share_sup=float(s.sup.mean()) if len(s) else np.nan,
                         share_sup_broad=float(s.sup_broad.mean()) if len(s) else np.nan,
                         share_nonprod=float(s.nonprod.mean()) if len(s) else np.nan,
                         share_hrs_week=float(s.hrs_week.notna().mean()) if len(s) else np.nan,
                         share_topcoded=float(s.topcoded.mean()) if len(s) else np.nan,
                         age_min=float(s.age.min()) if len(s) else np.nan, S_mean=float(s.S.mean()) if len(s) else np.nan))
        cols = ["year", "eyear", "wgt", "age", "female", "S", "grp", "sup", "sup_broad", "nonprod", "wage", "weeks",
                "hrs_week", "hrs_year_week", "topcoded", "occ"]
        s[cols].to_parquet(CACHE / f"cps_{y}.parquet", index=False)
        print(diag[-1], flush=True)
    if 2014 in years:
        d = parse(2014, src=lines_2014_3x8())
        s = topcode(sample(d), 2014)
        s.to_parquet(CACHE / "cps_2014_3x8.parquet", index=False)
    return pd.DataFrame(diag)


if __name__ == "__main__":
    ys = [int(a) for a in sys.argv[1:]] or SURVEY_YEARS
    D = build(ys)
    D.to_csv(ROOT / "results" / "v13b" / f"s0_cps_parse{'_' + '_'.join(map(str, ys)) if sys.argv[1:] else ''}.csv",
             index=False)
