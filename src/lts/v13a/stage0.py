"""Iteration 13a, stage 0: data preparation, no outcomes (pre_registration_v13a.md, stage 0).

0.1 hours by the 47 codes of iteration 12b, 1947-2023 (BEA-BLS research account 1947-2016 + index 1997-2024),
    units check against NIPA 6.9B/C/D and JHS 2013 H_EMP;
0.2 production and nonsupervisory (PNS) shares of hours and payroll (CES) by code;
0.3 capital vintages from FA Detailed Data (investment by asset x industry since 1901, BEA implied depreciation rates),
    checks against FA tables 3.3 (historical cost) and the detailed current-cost stocks;
0.4 index of hours per current dollar of private NDP before 1947 (Dumenil-Levy uslt4x);
0.5 productive / unproductive scope of FA industries.

Usage: PYTHONPATH=src python -P -m lts.v13a.stage0 [hours|units|ces|vintages|lambda|all]
Outputs: results/v13a/s0_*.csv, results/v13a/s0_vintages.npz
"""
from __future__ import annotations

import re
import sys
from functools import lru_cache

import numpy as np
import pandas as pd

from ..v6.series import ROOT
from ..v12b import us_io as u

RAW = ROOT / "data" / "raw"
V13 = RAW / "v13"
OUT = ROOT / "results" / "v13a"
OUT.mkdir(parents=True, exist_ok=True)
CODES = u.CODES
YEARS = range(1947, 2024)
ARCH = V13 / "nipa_hours_archive" / "nipa_section6_vintage_2024-06-28.xlsx"
RES = V13 / "bea_bls_ilpa" / "KLEMS-Research-Tables-Underlying-Data_0.xlsx"
ACC = V13 / "bea_bls_ilpa" / "BEA-BLS-industry-level-production-account-1997-2024.xlsx"


def norm(s):
    return " ".join(str(s).replace("\\", " ").split()).lower().rstrip("0123456789 ").strip()


# ================================================================ 0.1 hours
# research account industries -> 47 codes (a list: split within by compensation)
RES63 = {1: ["111CA"], 2: ["113FF"], 3: ["211"], 4: ["212"], 5: ["213"], 6: ["22"], 7: ["23"], 8: ["321"], 9: ["327"],
         10: ["331"], 11: ["332"], 12: ["333"], 13: ["334"], 14: ["335"], 15: ["3361MV"], 16: ["3364OT"], 17: ["337"],
         18: ["339"], 19: ["311FT"], 20: ["313TT"], 21: ["315AL"], 22: ["322"], 23: ["323"], 24: ["324"], 25: ["325"],
         26: ["326"], 27: ["42"], 28: ["44RT"], **{k: ["48"] for k in range(29, 36)}, 36: ["493"],
         **{k: ["51"] for k in range(37, 41)}, **{k: ["52"] for k in range(41, 45)}, 45: ["531"], 46: ["532RL"],
         47: ["54"], 48: ["54"], 49: ["54"], 50: ["55"], 51: ["56"], 52: ["56"], 53: ["61"], 54: ["62"], 55: ["62"],
         56: ["62"], 57: ["71"], 58: ["71"], 59: ["721"], 60: ["722"], 61: ["81"], 62: ["GFG", "GFE"],
         63: ["GSLG", "GSLE"]}
# 1947-1963 sheet: aggregated industries -> components on the 1963-2016 sheet (for the 1963 link) and 47 codes
RES44_PARTS = {2936: list(range(29, 37)), 3740: list(range(37, 41)), 4144: list(range(41, 45)), 4749: [47, 48, 49],
               5152: [51, 52], 5456: [54, 55, 56], 5758: [57, 58]}


@lru_cache(None)
def research():
    out = {}
    for sh in ("1947-1963", "1963-2016"):
        d = pd.read_excel(RES, sheet_name=sh, header=1)
        out[sh] = d.pivot_table(index="yr", columns="indnum", values="hrs", aggfunc="sum")
    return out


@lru_cache(None)
def acc_index():
    """1997-2024 account: 'Labor Hours_Quantity' index by industry description (63 rows)."""
    d = pd.read_excel(ACC, sheet_name="Labor Hours_Quantity", header=None)
    hdr = d.index[d.iloc[:, 0].astype(str).str.strip() == "Industry Description"][0]
    years = [int(float(v)) for v in d.iloc[hdr, 1:] if pd.notna(pd.to_numeric(v, errors="coerce"))]
    body = d.iloc[hdr + 1:, :1 + len(years)].copy()
    body.columns = ["desc"] + years
    body = body[body.desc.notna()]
    body["desc"] = body.desc.map(norm)
    return body.set_index("desc").apply(lambda c: pd.to_numeric(c, errors="coerce"))


ACC_ALIAS = {"hospitals and nursing and residential care": "hospitals and nursing and residential care facilities"}


def research_names():
    """Research-account industry names, normalised to the names of the 1997-2024 account."""
    d = pd.read_excel(RES, sheet_name="1963-2016", header=1)
    return d.groupby("indnum").Description.first().map(norm).replace(ACC_ALIAS)


def _split(total, codes, year):
    """Distribute a research-account industry total over 47 codes by compensation (only within the industry)."""
    if len(codes) == 1:
        return {codes[0]: total}
    w = u.compensation(min(max(year, 1948), 2023))[codes].clip(lower=0)
    w = w / w.sum() if w.sum() > 0 else pd.Series(1.0 / len(codes), index=codes)
    return {c: total * float(w[c]) for c in codes}


def hours_47(return_research=False):
    """Hours by 47 codes 1947-2023 (units of the research account 'hrs')."""
    R = research()
    old, new = R["1947-1963"], R["1963-2016"]
    # link 1947-1962 to the 1963-2016 sheet by the 1963 ratio within each old industry
    link = {}
    for ind in old.columns:
        parts = RES44_PARTS.get(ind, [ind])
        link[ind] = float(new.loc[1963, parts].sum() / old.loc[1963, ind])
    rows = {}
    for y in range(1947, 1963):
        rec = {}
        for ind in old.columns:
            codes = sorted({c for p in RES44_PARTS.get(ind, [ind]) for c in RES63[p]}, key=CODES.index)
            for c, v in _split(old.loc[y, ind] * link[ind], codes, y).items():
                rec[c] = rec.get(c, 0.0) + v
        rows[y] = rec
    # 1963-2016 by industry; 2017-2023: 2016 level x account index ratio
    idx = acc_index()
    names = research_names()
    ext = {}
    for ind in new.columns:
        nm = names[ind]
        ext[ind] = idx.loc[nm] if nm in idx.index else None
    for y in range(1963, 2024):
        rec = {}
        for ind in new.columns:
            if y <= 2016:
                val = new.loc[y, ind]
            else:
                e = ext[ind]
                val = new.loc[2016, ind] * float(e[y] / e[2016]) if e is not None and y in e.index else np.nan
            for c, v in _split(val, RES63[ind], y).items():
                rec[c] = rec.get(c, 0.0) + v
        rows[y] = rec
    H = pd.DataFrame(rows).T.reindex(columns=CODES)
    H.index.name = "year"
    missing = [names[i] for i, e in ext.items() if e is None]
    return (H, link, missing) if return_research else H


def hours_seam():
    """1997-2016: annual growth of research hrs vs the 1997-2024 account index, by research industry."""
    R = research()["1963-2016"]
    idx = acc_index()
    names = research_names()
    rows = []
    for ind in R.columns:
        nm = names[ind]
        if nm not in idx.index:
            rows.append(dict(indnum=ind, industry=nm, matched=False))
            continue
        g1 = np.log(R.loc[1997:2016, ind]).diff().dropna()
        g2 = np.log(idx.loc[nm, 1997:2016].astype(float)).diff().dropna()
        rows.append(dict(indnum=ind, industry=nm, matched=True, mean_growth_research=float(g1.mean()),
                         mean_growth_account=float(g2.mean()), mean_abs_diff=float((g1.values - g2.values).__abs__().mean())))
    S = pd.DataFrame(rows)
    priv = [i for i in R.columns if i not in (62, 63)]
    tot_r = np.log(R.loc[1997:2016, priv].sum(1)).diff().mean()
    tot_a = np.log(sum(R.loc[2016, i] * idx.loc[names[i], 1997:2016].astype(float) / float(idx.loc[names[i], 2016])
                       for i in priv if names[i] in idx.index)).diff().mean()
    return S, float(tot_r), float(tot_a)


# ================================================================ NIPA archive (section 6, vintage 2024-06-28)
@lru_cache(None)
def arch(table):
    d = pd.read_excel(ARCH, sheet_name=f"{table}-A", header=None)
    hdr = d.index[d.iloc[:, 0].astype(str).str.strip() == "Line"][0]
    years = {i: int(float(v)) for i, v in d.iloc[hdr].items() if i >= 2 and str(v).replace(".0", "").strip().isdigit()}
    body = d.iloc[hdr + 1:]
    body = body[pd.to_numeric(body.iloc[:, 0], errors="coerce").notna()]
    vals = body[list(years)].apply(lambda c: pd.to_numeric(c, errors="coerce"))
    vals.columns = list(years.values())
    vals.index = pd.MultiIndex.from_arrays([pd.to_numeric(body.iloc[:, 0]).astype(int), body.iloc[:, 1].map(norm)],
                                           names=["line", "label"])
    return vals


def _first(tab, label, after=None):
    """Row of the first line with this label (after a given line)."""
    lab = tab.index.get_level_values("label")
    lines = tab.index.get_level_values("line")
    for (ln, lb), row in tab.iterrows():
        if lb == label and (after is None or ln > after):
            return row
    raise KeyError(label)


def sic_groups(table):
    """SIC tables (6.3B/C, 6.5B/C, 6.8B/C, 6.9B/C): the 12b bridge groups (lts.v12b.us_io.GROUPS)."""
    t = arch(table)
    g = {"AGR": _first(t, "agriculture, forestry, and fishing"), "MIN": _first(t, "mining"),
         "CON": _first(t, "construction"), "TRN": _first(t, "transportation"),
         "UTL": _first(t, "electric, gas, and sanitary services"), "WHL": _first(t, "wholesale trade"),
         "RET": _first(t, "retail trade"), "FIRE": _first(t, "finance, insurance, and real estate")}
    comm = None
    for lab in ("communications", "communication"):
        try:
            comm = _first(t, lab)
            break
        except KeyError:
            pass
    mp = _first(t, "motion pictures")
    g["MFGINF"] = _first(t, "manufacturing") + comm + mp
    g["SRV"] = _first(t, "services") - mp
    lines = t.index.get_level_values("line")
    labs = list(t.index.get_level_values("label"))
    fed = [ln for ln, lb in zip(lines, labs) if lb == "federal"][0]
    sl = [ln for ln, lb in zip(lines, labs) if lb == "state and local"][0]
    gg = [ln for ln, lb in zip(lines, labs) if lb == "general government"]
    ge = [ln for ln, lb in zip(lines, labs) if lb == "government enterprises"]
    byline = t.droplevel("label")
    g["FEDG"] = byline.loc[[x for x in gg if fed < x < sl][0]]
    g["FEDE"] = byline.loc[[x for x in ge if fed < x < sl][0]]
    g["SLG"] = byline.loc[[x for x in gg if x > sl][0]]
    g["SLE"] = byline.loc[[x for x in ge if x > sl][0]]
    return pd.DataFrame(g)


# NAICS detail (6.3D / 6.8D lines) -> 47 codes
NAICS_LINE = {**{k: v for k, v in u.FA_IND.items()},
              "farms": "111CA", "forestry, fishing, and related activities": "113FF",
              "food and beverage and tobacco products": "311FT", "motor vehicle and parts dealers": "44RT",
              "food and beverage stores": "44RT", "general merchandise stores": "44RT", "other retail": "44RT",
              "rail transportation": "48", "truck transportation": "48", "transit and ground passenger transportation": "48",
              "pipeline transportation": "48", "other transportation and support activities": "48",
              "air transportation": "48", "water transportation": "48", "warehousing and storage": "493",
              "publishing industries (includes software)": "51", "motion picture and sound recording industries": "51",
              "broadcasting and telecommunications": "51", "information and data processing services": "51",
              "federal reserve banks, credit intermediation, and related activities": "52",
              "securities, commodity contracts, and investments": "52",
              "insurance carriers and related activities": "52", "funds, trusts, and other financial vehicles": "52",
              "real estate": "531", "rental and leasing services and lessors of intangible assets": "532RL",
              "legal services": "54", "computer systems design and related services": "54",
              "miscellaneous professional, scientific, and technical services": "54",
              "management of companies and enterprises": "55", "administrative and support services": "56",
              "waste management and remediation services": "56", "educational services": "61",
              "ambulatory health care services": "62", "hospitals": "62", "nursing and residential care facilities": "62",
              "hospitals and nursing and residential care facilities": "62", "social assistance": "62",
              "performing arts, spectator sports, museums, and related activities": "71",
              "amusements, gambling, and recreation industries": "71", "accommodation": "721",
              "food services and drinking places": "722", "other services, except government": "81"}


def naics_47(table):
    """NAICS archive table (6.3D, 6.8D) -> 47 codes (leaf lines; parents with mapped children skipped)."""
    return _naics_from(arch(table))


# ================================================================ wages and salaries (NIPA 6.3) by 47 codes
@lru_cache(None)
def flat_63d():
    """Current NIPA flat file, table 6.3D (1998-), by 47 codes via line labels in the series register."""
    reg = pd.read_csv(RAW / "v9b" / "bea_SeriesRegister.txt", dtype=str)
    reg.columns = ["code", "label", "metric", "calc", "scale", "tables", "parents"]
    rows = []
    for _, r in reg.iterrows():
        for tl in str(r.tables).split("|"):
            m = re.match(r"T60300D:(\d+)$", tl)
            if m:
                rows.append((int(m.group(1)), r.code, norm(r.label)))
    lines = pd.DataFrame(rows, columns=["line", "code", "label"]).sort_values("line")
    d = pd.read_csv(RAW / "v9b" / "bea_NipaDataA.txt", dtype=str)
    d.columns = ["code", "period", "value"]
    d = d[d.code.isin(lines.code)]
    d["v"] = pd.to_numeric(d.value.str.replace(",", ""), errors="coerce")
    P = d.pivot_table(index="code", columns="period", values="v")
    P.columns = P.columns.astype(int)
    t = P.reindex(lines.code)
    t.index = pd.MultiIndex.from_arrays([lines.line, lines.label], names=["line", "label"])
    return t


def wages_47():
    """Wages and salaries by 47 codes, $ million, 1948-2023: 1948-1997 SIC groups (6.3B to 1987, 6.3C 1988-1997)
    allocated within groups by the 12b compensation allocator; 1998-2023 NAICS lines (current flat file 6.3D)."""
    gB, gC = sic_groups("T60300B"), sic_groups("T60300C")
    G = pd.concat([gB.loc[:1987], gC.loc[1988:1997]])
    rows = {}
    for y in range(1948, 1998):
        alloc = u.comp_allocator(y).clip(lower=0)
        rec = pd.Series(0.0, index=CODES)
        for g, cs in u.GROUPS.items():
            w = alloc[cs]
            rec[cs] = float(G.loc[y, g]) * (w / w.sum() if w.sum() > 0 else 1.0 / len(cs))
        rows[y] = rec
    D = _naics_from(flat_63d())
    for y in range(1998, 2024):
        rows[y] = D.loc[y] if y in D.index else pd.Series(np.nan, index=CODES)
    W = pd.DataFrame(rows).T
    W.index.name = "year"
    return W


def _naics_from(t):
    """NAICS table (index line, label) -> 47 codes. A mapped line is treated as a parent and skipped when the lines
    right after it map to the same code and sum to its value (within 0.5% in every year with data)."""
    out = pd.DataFrame(0.0, index=t.columns, columns=CODES)
    labs = list(t.index.get_level_values("label"))
    lines = list(t.index.get_level_values("line"))
    vals = t.to_numpy(float)
    fed = lines[labs.index("federal")]
    sl = lines[labs.index("state and local")]
    codes = [NAICS_LINE.get(lb) for lb in labs]
    for i, (ln, lb) in enumerate(zip(lines, labs)):
        if lb in ("general government", "government enterprises"):
            code = ("GFG" if lb == "general government" else "GFE") if fed < ln < sl else (
                "GSLG" if lb == "general government" else "GSLE")
            out[code] += np.nan_to_num(vals[i])
            continue
        c = codes[i]
        if c is None:
            continue
        j = i + 1
        while j < len(labs) and codes[j] == c:
            j += 1
        if j > i + 1:
            child = np.nansum(vals[i + 1:j], axis=0)
            ok = np.isfinite(vals[i]) & (np.abs(vals[i]) > 0)
            if ok.any() and np.all(np.abs(child[ok] / vals[i][ok] - 1) < 0.005):
                continue
        out[c] += np.nan_to_num(vals[i])
    return out


# ================================================================ self-employed share of persons engaged
SE_SECTOR_SIC = {"AGR": "agriculture, forestry, and fishing", "MIN": "mining", "CON": "construction",
                 "MFGINF": "manufacturing", "TRN": "transportation and public utilities",
                 "UTL": "transportation and public utilities", "WHL": "wholesale trade", "RET": "retail trade",
                 "FIRE": "finance, insurance, and real estate", "SRV": "services"}


def se_share():
    """Self-employed / persons engaged by 47 code, 1948-2023 (NIPA 6.7 / 6.8; SIC sectors to 1997, NAICS sectors
    1998-2022; 2023 = 2022). Government: 0."""
    rows = {}
    for tab7, tab8, yrs in (("T60700B", "T60800B", range(1948, 1988)), ("T60700C", "T60800C", range(1988, 1998))):
        s7, s8 = arch(tab7), arch(tab8)
        for y in yrs:
            rec = pd.Series(0.0, index=CODES)
            for g, cs in u.GROUPS.items():
                if g not in SE_SECTOR_SIC:
                    continue
                lab = SE_SECTOR_SIC[g]
                se = float(_first(s7, lab)[y])
                pe = float(_first(s8, lab)[y]) if lab != "transportation and public utilities" else float(
                    _first(s8, "transportation and public utilities")[y])
                rec[cs] = se / pe if pe > 0 else 0.0
            rows[y] = rec
    s7, s8 = arch("T60700D"), arch("T60800D")
    sec7 = {lb: row for (ln, lb), row in s7.iterrows()}
    p8 = naics_47("T60800D")
    sector_codes = {
        "agriculture, forestry, fishing, and hunting": ["111CA", "113FF"], "mining": ["211", "212", "213"],
        "utilities": ["22"], "construction": ["23"], "manufacturing": u.MFG, "wholesale trade": ["42"],
        "retail trade": ["44RT"], "transportation and warehousing": ["48", "493"], "information": ["51"],
        "finance and insurance": ["52"], "real estate and rental and leasing": ["531", "532RL"],
        "professional, scientific, and technical services": ["54"],
        "management of companies and enterprises": ["55"],
        "administrative and waste management services": ["56"], "educational services": ["61"],
        "health care and social assistance": ["62"], "arts, entertainment, and recreation": ["71"],
        "accommodation and food services": ["721", "722"], "other services, except government": ["81"]}
    for y in range(1998, 2024):
        yy = min(y, max(s7.columns))
        rec = pd.Series(0.0, index=CODES)
        for lab, cs in sector_codes.items():
            row = sec7.get(lab)
            if row is None:
                continue
            pe = float(p8.loc[yy, cs].sum())
            rec[cs] = float(row[yy]) / pe if pe > 0 else 0.0
        rows[y] = rec
    S = pd.DataFrame(rows).T
    S.index.name = "year"
    return S


# ================================================================ 0.1 units check
@lru_cache(None)
def jhs_hemp():
    d = pd.read_excel(V13 / "jhs" / "usa_wk_apr_2013.xlsx", sheet_name="DATA")
    d = d[d.Variable.astype(str).str.strip() == "H_EMP"]
    ycols = [c for c in d.columns if str(c).startswith("_")]
    out = d.set_index(d.code.astype(str).str.strip())[ycols]
    out.columns = [int(str(c)[1:]) for c in ycols]
    return out.apply(lambda c: pd.to_numeric(c, errors="coerce"))


def nipa_hours_private():
    s = {}
    for tab, yrs in (("T60900B", range(1948, 1988)), ("T60900C", range(1988, 2000)), ("T60900D", range(2000, 2023))):
        t = arch(tab)
        row = None
        for (ln, lb), r in t.iterrows():
            if lb.startswith("private industries"):
                row = r
                break
        for y in yrs:
            s[y] = float(row[y])
    return pd.Series(s)


def units_check():
    H = hours_47()
    priv = [c for c in CODES if not c.startswith("G")]
    hp = H[priv].sum(1)
    nip = nipa_hours_private()
    jh = jhs_hemp()
    tot = jh.loc["TOT"] if "TOT" in jh.index else None
    gov = [c for c in jh.index if c.startswith("L") or c == "L"]  # ISIC L = public administration
    rows = []
    for y in range(1948, 2023):
        r = dict(year=y, hrs_private=float(hp.get(y, np.nan)), nipa69_private_employees=float(nip.get(y, np.nan)))
        if tot is not None and y in tot.index:
            r["jhs_H_EMP_total"] = float(tot[y])
            r["jhs_H_EMP_publicadmin_L"] = float(jh.loc["L", y]) if "L" in jh.index else np.nan
        rows.append(r)
    U = pd.DataFrame(rows)
    res = {}
    for scale in (1.0, 1e3, 1e-3):
        a = (U.hrs_private * scale / U.nipa69_private_employees).median()
        b = (U.hrs_private * scale / (U.jhs_H_EMP_total - U.jhs_H_EMP_publicadmin_L)).median()
        res[scale] = (float(a), float(b))
    return U, res


# ================================================================ 0.2 CES shares
CES = V13 / "bls_ces"
# 47 code -> preference list of CES scopes; a scope is a tuple of CES industry codes (summed)
SCOPES = {
    "211": [("10211000",), ("10210000",), ("10000000",)], "212": [("10212000",), ("10210000",), ("10000000",)],
    "213": [("10213000",), ("10210000",), ("10000000",)], "22": [("44220000",), ("40000000",)],
    "23": [("20000000",)],
    **{c: [(d,), ("31000000",)] for c, d in (("321", "31321000"), ("327", "31327000"), ("331", "31331000"),
                                              ("332", "31332000"), ("333", "31333000"), ("334", "31334000"),
                                              ("335", "31335000"), ("337", "31337000"), ("339", "31339000"))},
    "3361MV": [("31336100", "31336300"), ("31000000",)], "3364OT": [("31336400",), ("31000000",)],
    **{c: [(d,), ("32000000",)] for c, d in (("311FT", "32311000"), ("313TT", "32314000"), ("322", "32322000"),
                                              ("323", "32323000"), ("325", "32325000"), ("326", "32326000"))},
    "315AL": [("32000000",)], "324": [("32000000",)],
    "48": [("43481000", "43484000", "43485000", "43488000", "43492000"), ("43000000",), ("40000000",)],
    "493": [("43493000",), ("43000000",), ("40000000",)],
    "51": [("50000000",)], "54": [("60540000",), ("60000000",)], "61": [("65000000",)],
    "62": [("65620000",), ("65000000",)], "71": [("70710000",), ("70000000",)], "721": [("70721000",), ("70000000",)],
    "722": [("70722000",), ("70000000",)], "81": [("80000000",)]}
# codes covered by each scope (for the wage denominator); partial scopes are not used for the payroll share
COVER = {"06000000": ["211", "212", "213", "23", "321", "327", "331", "332", "333", "334", "335", "3361MV", "3364OT",
                      "337", "339", "311FT", "313TT", "315AL", "322", "323", "324", "325", "326"],
         "10000000": ["211", "212", "213"], "10210000": ["211", "212", "213"], "40000000": ["22", "42", "44RT", "48", "493"],
         "43000000": ["48", "493"], "31000000": ["321", "327", "331", "332", "333", "334", "335", "3361MV", "3364OT", "337", "339"],
         "32000000": ["311FT", "313TT", "315AL", "322", "323", "324", "325", "326"], "60000000": ["54", "55", "56"],
         "65000000": ["61", "62"], "70000000": ["71", "721", "722"], "50000000": ["51"], "80000000": ["81"],
         "20000000": ["23"], "44220000": ["22"], "60540000": ["54"], "65620000": ["62"], "70710000": ["71"],
         "70721000": ["721"], "70722000": ["722"], "10211000": ["211"], "10212000": ["212"], "10213000": ["213"],
         "43493000": ["493"], **{d: [c] for c, d in (("321", "31321000"), ("327", "31327000"), ("331", "31331000"),
                                                     ("332", "31332000"), ("333", "31333000"), ("334", "31334000"),
                                                     ("335", "31335000"), ("337", "31337000"), ("339", "31339000"),
                                                     ("322", "32322000"), ("323", "32323000"), ("325", "32325000"),
                                                     ("326", "32326000"))}}
PARTIAL = {("31336100", "31336300"), ("31336400",), ("32311000",), ("32314000",),
           ("43481000", "43484000", "43485000", "43488000", "43492000")}
AGR = ["111CA", "113FF"]


@lru_cache(None)
def ces_annual():
    """Annual averages (NSA, M13) of data types 01, 02, 06, 07, 08 by CES industry code."""
    ser = pd.read_csv(CES / "ce.series", sep="\t", dtype=str)
    ser.columns = [c.strip() for c in ser.columns]
    for c in ser.columns:
        ser[c] = ser[c].str.strip()
    ser = ser[(ser.seasonal == "U") & ser.data_type_code.isin(["01", "02", "06", "07", "08"])]
    want = set(ser.series_id)
    frames = []
    for f in sorted(CES.glob("ce.data.*")):
        if f.name == "ce.data.0.AllCESSeries":
            continue
        d = pd.read_csv(f, sep="\t", dtype=str)
        d.columns = [c.strip() for c in d.columns]
        d["series_id"] = d.series_id.str.strip()
        d = d[d.series_id.isin(want) & (d.period.str.strip() == "M13")]
        frames.append(d)
    d = pd.concat(frames).drop_duplicates(["series_id", "year"])
    d["v"] = pd.to_numeric(d.value.str.strip(), errors="coerce")
    d["year"] = d.year.astype(int)
    d = d.merge(ser[["series_id", "industry_code", "data_type_code"]], on="series_id")
    return d.pivot_table(index=["industry_code", "year"], columns="data_type_code", values="v")


def scope_vals(scope, year):
    """PNS employment, AE employment, PNS weekly hours/worker, AE weekly hours/worker, PNS weekly payroll ($ thousand)."""
    A = ces_annual()
    tot = dict(n06=0.0, n01=0.0, h06=0.0, h01=0.0, pay06=0.0, ok_h01=True)
    for c in scope:
        if (c, year) not in A.index:
            return None
        r = A.loc[(c, year)]
        if not all(np.isfinite(r.get(k, np.nan)) for k in ("01", "06", "07", "08")):
            return None
        tot["n06"] += r["06"]
        tot["n01"] += r["01"]
        tot["h06"] += r["06"] * r["07"]
        tot["pay06"] += r["06"] * r["07"] * r["08"]
        if np.isfinite(r.get("02", np.nan)):
            tot["h01"] += r["01"] * r["02"]
        else:
            tot["ok_h01"] = False
    return tot


def _scope_series(scope, W):
    """Raw yearly sN, rho (2006+ where AE hours exist) and sW (full-coverage scopes only) of one CES scope."""
    recs = {}
    for y in range(1939, 2027):
        v = scope_vals(scope, y)
        if v is None:
            continue
        sN = v["n06"] / v["n01"]
        rho = (v["h06"] / v["n06"]) / (v["h01"] / v["n01"]) if v["ok_h01"] and v["h01"] > 0 else np.nan
        sW = np.nan
        cov = COVER.get(scope[0]) if len(scope) == 1 and scope not in PARTIAL else None
        if cov is not None and y in W.index:
            den = float(W.loc[y, cov].sum())
            if np.isfinite(den) and den > 0:
                sW = v["pay06"] * 52 / 1000 / den
        recs[y] = dict(sN=sN, rho=rho, sW=sW)
    return pd.DataFrame(recs).T if recs else pd.DataFrame(columns=["sN", "rho", "sW"])


def _linked(series_list, link=True):
    """Combine scope series (most detailed first): years of the detailed series as they are; other years from the
    next coarser series, scaled by the mean ratio detailed / coarser over the first 3 common years (link=True)."""
    out = pd.Series(dtype=float)
    scale = 1.0
    prev = None
    for ser in series_list:
        ser = ser.dropna()
        if ser.empty:
            continue
        if prev is not None and link:
            common = sorted(set(prev.index) & set(ser.index))[:3]
            if common:
                scale = scale * float(np.mean([prev[c] / ser[c] for c in common if ser[c] != 0]))
        add = ser * scale
        out = pd.concat([out, add[~add.index.isin(out.index)]]).sort_index()
        prev = ser
    return out


def ces_shares(link=True, rho_mode="2006-2010", pre1964="1964", agr="one", years=range(1948, 2024)):
    """PNS shares by 47 code and year: sN (employment), rho (hours per worker PNS / AE), sH = sN * rho,
    sW (PNS payroll / NIPA wages and salaries). Variants: link, rho_mode ('2006-2010' or 'one'), pre1964 ('1964' or
    'goods-trend'), agr ('one' or 'private')."""
    W = wages_47()
    cache = {}

    def ser(sc):
        if sc not in cache:
            cache[sc] = _scope_series(sc, W)
        return cache[sc]

    goods = ser(("06000000",))
    pref_all = dict(SCOPES)
    if agr == "private":
        for c in AGR:
            pref_all[c] = [("05000000",)]
    rows = []
    for code, pref in pref_all.items():
        S = {k: _linked([ser(sc)[k] for sc in pref if not (k == "sW" and sc in PARTIAL) and k in ser(sc)], link)
             for k in ("sN", "sW", "rho")}
        rho = S["rho"]
        first = {k: (int(v.index.min()) if len(v) else None) for k, v in S.items()}
        for y in years:
            rec = dict(year=y, code=code)
            for k in ("sN", "sW"):
                v = S[k]
                if y in v.index:
                    val = float(v[y])
                elif first[k] is not None and y < first[k]:
                    f0 = first[k]
                    val = float(v[f0])
                    if pre1964 == "goods-trend" and f0 > 1948 and y in goods.index and f0 in goods.index:
                        g = goods[k]
                        if np.isfinite(g.get(y, np.nan)) and np.isfinite(g.get(f0, np.nan)):
                            val *= float(g[y] / g[f0])
                else:
                    val = float(v[v.index <= y].iloc[-1]) if len(v[v.index <= y]) else np.nan
                rec[k] = val
            if rho_mode == "one":
                rec["rho"] = 1.0
            elif y >= 2006 and y in rho.index:
                rec["rho"] = float(rho[y])
            else:
                rec["rho"] = float(rho.loc[2006:2010].mean()) if len(rho.loc[2006:2010]) else 1.0
            rows.append(rec)
    for c in AGR:
        if agr == "one":
            for y in years:
                rows.append(dict(year=y, code=c, sN=1.0, sW=1.0, rho=1.0))
    D = pd.DataFrame(rows)
    # sW is not capped: PNS compensation = sW x comp = CES PNS payroll x (comp / wages) of the code; sW > 1 marks
    # codes whose pre-1998 NIPA wages (SIC groups allocated to NAICS codes) are too low (journal, stage 0)
    D["sW_gt1"] = D.sW > 1
    D["sN"] = D.sN.clip(0, 1)
    D["sH"] = (D.sN * D.rho).clip(0, 1)
    return D


# ================================================================ 0.3 capital vintages
FAD = V13 / "bea_fa_detail"
# detailed asset -> producing industry (47 codes), following lts.v12b.us_io.ASSET_LINES by name
ASSET_PROD = {
    **{a: "334" for a in ("EP1A", "EP1B", "EP1C", "EP1D", "EP1E", "EP1F", "EP1G", "EP1H", "EP20", "EP35", "EP36")},
    "EP34": "339", "EP31": "333", "EP12": "333", "EI11": "325", "EI12": "332",
    **{a: "333" for a in ("EI21", "EI22", "EI30", "EI40", "EI50", "EO30", "EO21", "EO40", "EO22", "EO50", "EO60")},
    "EI60": "335", "EO71": "335", "EO72": "335", "EO80": "339",
    **{a: "3361MV" for a in ("ET11", "ET12", "ET20")}, **{a: "3364OT" for a in ("ET30", "ET40", "ET50")},
    "EO11": "337", "EO12": "337",
    "SOMO": "321", "SM01": "213", "SM02": "213",
    "ENS1": "51", "ENS2": "54", "ENS3": "54",
    **{a: "51" for a in ("AE10", "AE20", "AE30", "AE40", "AE50")}}
# FA detailed industries -> 47 codes; FA-level scope (iteration 9: legal services unproductive, waste management
# productive; real estate - nonresidential assets only)
FA_DET = {"110C": "111CA", "113F": "113FF", "2110": "211", "2120": "212", "2130": "213", "2211": "22", "2212": "22",
          "2213": "22", "2300": "23", "3210": "321", "3270": "327", "3310": "331", "3320": "332", "3330": "333",
          "3340": "334", "3350": "335", "336M": "3361MV", "336O": "3364OT", "3370": "337", "3380": "339", "3390": "339",
          "3110": "311FT", "3120": "311FT", "313T": "313TT", "315A": "315AL", "3220": "322", "3230": "323",
          "3240": "324", "3250": "325", "3260": "326", "4210": "42", "4220": "42", "4410": "44RT", "4450": "44RT",
          "4520": "44RT", "442R": "44RT", "4810": "48", "4820": "48", "4830": "48", "4840": "48", "4850": "48",
          "4860": "48", "487S": "48", "4930": "493", "5110": "51", "5120": "51", "5130": "51", "5140": "51",
          "5210": "52", "5221": "52", "5223": "52", "5229": "52", "5230": "52", "524A": "52", "5243": "52",
          "5250": "52", "5310": "531", "5320": "532RL", "5411": "54", "5415": "54", "5412": "54", "5500": "55",
          "5610": "56", "5620": "56", "6100": "61", "6210": "62", "622H": "62", "6230": "62", "6240": "62",
          "711A": "71", "7130": "71", "7210": "721", "7220": "722", "8100": "81"}
FA_UNPROD = {"4210", "4220", "4410", "4450", "4520", "442R", "5210", "5221", "5223", "5229", "5230", "524A", "5243",
             "5250", "5310", "5320", "5411", "5500", "5610"}
TOTALS = {"EQ00", "ST00", "IP00", "ES00", "EQUI", "STRU", "IPP"}


def asset_producer(a):
    a = a.strip()
    if a in ASSET_PROD:
        return ASSET_PROD[a]
    if a.startswith("S"):
        return "23"
    if a.startswith("RD"):
        return "54"
    raise KeyError(a)


@lru_cache(None)
def fa_dataset(name):
    """FA Detailed 'Datasets' sheet -> DataFrame indexed by (industry, asset), columns = years."""
    d = pd.read_excel(FAD / f"{name}.xlsx", sheet_name="Datasets", header=0)
    sid = d.iloc[:, 0].astype(str).str.strip()
    years = [int(float(c)) for c in d.columns[1:]]
    vals = d.iloc[:, 1:].apply(lambda c: pd.to_numeric(c, errors="coerce"))
    vals.columns = years
    vals.index = pd.MultiIndex.from_arrays([sid.str[3:7], sid.str[8:].str.replace(".A", "", regex=False).str.strip()],
                                           names=["ind", "asset"])
    vals = vals[~vals.index.get_level_values("asset").isin(TOTALS)]
    return vals[~vals.index.duplicated()]


def vintage_inputs(last=2023):
    """Aligned arrays over (industry, asset) rows and years 1901..last: I1 current-$ investment, I2 chained,
    delta implied rate (1925 rate before 1925), K1 current-cost stock, K2 chained stock, D2 chained depreciation."""
    I1 = fa_dataset("detailnonres_inv1")
    keys = I1.index
    years = list(range(1901, last + 1))
    def al(name):
        X = fa_dataset(name).reindex(index=keys)
        return X.reindex(columns=years)
    I1 = I1.reindex(columns=years).fillna(0.0)
    I2 = al("detailnonres_inv2")
    K1, K2, D2 = al("detailnonres_stk1"), al("detailnonres_stk2"), al("detailnonres_dep2")
    # DetailNonres_rate has implied rates only for asset aggregates (741 rows), not by detailed asset: the
    # pre-registered fallback - implied rate from chained series, d(t) = D2(t) / (K2(t-1) + I2(t) / 2)
    den = K2.shift(1, axis=1) + I2.fillna(0.0) / 2
    R = (D2 / den).where(den > 0)
    R = R.clip(lower=0.0, upper=1.0)
    asset = keys.get_level_values("asset")
    med = R.groupby(asset).transform("median")          # asset median across industries, same year
    R = R.fillna(med)
    R.loc[:, [y for y in years if y <= 1925]] = np.nan  # before 1926: first available rate (1926)
    R = R.T.bfill().T
    R = R.fillna(R.groupby(asset).transform("median")).fillna(0.1)
    return dict(keys=keys, years=years, I1=I1, I2=I2, delta=R, K1=K1, K2=K2, D2=D2)


def vintage_stock(I, delta, weight=None):
    """Net stock by recursion Z(t) = Z(t-1) (1 - d(t)) + I(t) w(t) (1 - d(t)/2): w = 1 historical cost in dollars;
    w = lambda(t) gives historical cost in hours of the purchase year (stage 1)."""
    I = np.asarray(I, float)
    d = np.asarray(delta, float)
    w = np.ones_like(I) if weight is None else np.asarray(weight, float)
    Z = np.zeros_like(I)
    prev = np.zeros(I.shape[0])
    for j in range(I.shape[1]):
        prev = prev * (1 - d[:, j]) + I[:, j] * w[:, j] * (1 - d[:, j] / 2)
        Z[:, j] = prev
    return Z


@lru_cache(None)
def fa_table_by_industry(sheet):
    """FA section 3/4 published table (e.g. FAAt303ESI-A) by normalised line name."""
    sec = int(sheet[4])
    x = pd.read_excel(V13 / "bea_fa_tables" / f"Section{sec}All_xls.xlsx", sheet_name=sheet, header=None)
    hdr = x.index[x.iloc[:, 0].astype(str).str.strip() == "Line"][0]
    h = pd.to_numeric(x.iloc[hdr], errors="coerce")
    cols = [i for i in range(2, x.shape[1]) if np.isfinite(h.iloc[i])]
    rows = x.iloc[hdr + 1:]
    rows = rows[pd.to_numeric(rows.iloc[:, 0], errors="coerce").notna()]
    vals = rows.iloc[:, cols].apply(lambda c: pd.to_numeric(c, errors="coerce"))
    vals.columns = h.iloc[cols].astype(int).to_numpy()
    vals.index = pd.MultiIndex.from_arrays([pd.to_numeric(rows.iloc[:, 0]).astype(int), rows.iloc[:, 1].map(norm)])
    return vals


def vintage_check():
    v = vintage_inputs()
    keys, years = v["keys"], v["years"]
    Hd = vintage_stock(v["I1"].to_numpy(), v["delta"].to_numpy())
    # current cost by recursion with investment prices p = I1 / I2 (asset x industry)
    p = (v["I1"] / v["I2"]).replace([np.inf, -np.inf], np.nan)
    p = p.T.ffill().bfill().T.to_numpy()
    I = v["I1"].to_numpy()
    dd = v["delta"].to_numpy()
    Cc = np.zeros_like(I)
    prev = np.zeros(I.shape[0])
    for j in range(I.shape[1]):
        reval = np.divide(p[:, j], p[:, j - 1], out=np.ones(I.shape[0]), where=(j > 0) & (p[:, j - 1] > 0)) if j else 1.0
        prev = prev * reval * (1 - dd[:, j]) + I[:, j] * (1 - dd[:, j] / 2)
        Cc[:, j] = prev
    ind = keys.get_level_values("ind")
    rows = []
    t43 = fa_table_by_industry("FAAt403-A")
    tot43 = t43.iloc[0]
    K1 = v["K1"].to_numpy()
    for j, y in enumerate(years):
        if y < 1925:
            continue
        rows.append(dict(year=y, hist_model=float(Hd[:, j].sum()), hist_BEA_table43=float(tot43.get(y, np.nan)),
                         curr_model=float(Cc[:, j].sum()), curr_BEA_detail=float(np.nansum(K1[:, j]))))
    C = pd.DataFrame(rows)
    C["hist_dev"] = C.hist_model / C.hist_BEA_table43 - 1
    C["curr_dev"] = C.curr_model / C.curr_BEA_detail - 1
    # by 47 code: model historical cost vs table 3.3ESI (industries named as in lts.v12b.us_io.FA_IND); real estate
    # excluded (3.3 includes residential assets)
    t33 = fa_table_by_industry("FAAt303ESI-A")
    by33 = {}
    for (ln, lb), row in t33.iterrows():
        c = u.FA_IND.get(lb) or (u.FA_IND.get("health care and social assistance") if lb == "health and social assistance" else None)
        if c and lb not in by33:
            by33[lb] = (c, row)
    tab = pd.DataFrame({y: {} for y in years})
    agg33 = {}
    for lb, (c, row) in by33.items():
        agg33[c] = agg33.get(c, 0) + row
    code_of = np.array([FA_DET[i] for i in ind])
    rows_c = []
    for c, row in agg33.items():
        if c == "531":
            continue
        m = code_of == c
        for j, y in enumerate(years):
            if y >= 1948 and y in row.index and np.isfinite(row[y]) and row[y] > 0:
                rows_c.append(dict(code=c, year=y, dev=float(Hd[m, j].sum() / row[y] - 1)))
    Dc = pd.DataFrame(rows_c)
    by_code = Dc.groupby("code").dev.agg(lambda x: float(np.abs(x).max())).rename("max_abs_dev_1948_2023")
    by_code_med = Dc.groupby("code").dev.agg(lambda x: float(np.abs(x).median())).rename("median_abs_dev")
    C.attrs["by_code"] = pd.concat([by_code, by_code_med], axis=1)
    return C


# ================================================================ 0.4 lambda index before 1947
@lru_cache(None)
def uslt4x():
    """Dumenil-Levy USLT (version 4x), first table: AN KN KN_R L W GDP GDP_R NDP NDP_R I_R R U P; -999999 = missing."""
    txt = (V13 / "dumenil_levy" / "uslt4x.txt").read_text(encoding="latin-1").splitlines()
    hdr, rows = None, []
    for line in txt:
        tok = line.split()
        if not tok:
            continue
        if tok[0] == "AN":
            if hdr is not None:
                break                      # second table
            hdr = tok
            continue
        if hdr is not None and re.fullmatch(r"\d{4}", tok[0]) and len(tok) == len(hdr):
            rows.append([float(t) for t in tok])
    D = pd.DataFrame(rows, columns=hdr).set_index("AN")
    D.index = D.index.astype(int)
    return D.replace(-999999.0, np.nan)


def lambda_index():
    """(L / NDP)_v / (L / NDP)_1947, private economy, hours per current dollar of NDP (main variant of 0.4)."""
    D = uslt4x()
    ratio = D["L"] / D["NDP"]
    return (ratio / ratio.loc[1947]).loc[1901:1947]


# ================================================================ run
def run(part="all"):
    if part in ("hours", "all"):
        H, link, missing = hours_47(return_research=True)
        H.to_csv(OUT / "s0_hours_47.csv")
        S, tr, ta = hours_seam()
        S.to_csv(OUT / "s0_hours_seam.csv", index=False)
        pd.DataFrame([dict(item="link ratio 1963 (" + str(k) + ")", value=v) for k, v in link.items()]
                     + [dict(item="private mean annual growth 1997-2016, research", value=tr),
                        dict(item="private mean annual growth 1997-2016, 1997-2024 account", value=ta)]
                     ).to_csv(OUT / "s0_hours_link.csv", index=False)
        rows = []
        for ind, codes in RES63.items():
            rows.append(dict(sheet="1963-2016", indnum=ind, codes="+".join(codes), split_by_compensation=len(codes) > 1))
        for ind, parts in RES44_PARTS.items():
            codes = sorted({c for p_ in parts for c in RES63[p_]}, key=CODES.index)
            rows.append(dict(sheet="1947-1963", indnum=ind, codes="+".join(codes), split_by_compensation=len(codes) > 1))
        pd.DataFrame(rows).to_csv(OUT / "s0_hours_map.csv", index=False)
        print("hours done", flush=True)
    if part in ("units", "all"):
        U, res = units_check()
        H = hours_47()
        jh = jhs_hemp()
        U["hrs_all_industries"] = U.year.map(H.sum(1))
        U["jhs_H_EMP_TOT"] = U.year.map(jh.loc["TOT"])
        U.to_csv(OUT / "s0_hours_units.csv", index=False)
        r_all = (U.hrs_all_industries / U.jhs_H_EMP_TOT).median()
        r_emp = (U.hrs_private / U.nipa69_private_employees).median()
        decision = "all employed (incl. self-employed)" if 0.95 <= r_all <= 1.05 else (
            "employees only" if 0.95 <= r_emp <= 1.05 else "STOP: ask the user")
        pd.DataFrame([dict(check="median hrs(all industries) / JHS H_EMP TOT 1948-2010", value=r_all),
                      dict(check="median hrs(private) / NIPA 6.9 private employees 1948-2022", value=r_emp),
                      dict(check="decision", value=decision)]).to_csv(OUT / "s0_hours_units_decision.csv", index=False)
        print("units:", decision, flush=True)
    if part in ("ces", "all"):
        variants = {"main": {}, "unlinked": dict(link=False), "rho=1": dict(rho_mode="one"),
                    "pre-1964 goods trend": dict(pre1964="goods-trend"), "agriculture = private": dict(agr="private")}
        frames = []
        for nm, kw in variants.items():
            D = ces_shares(**kw)
            D["variant"] = nm
            frames.append(D)
        pd.concat(frames).to_csv(OUT / "s0_ces_shares.csv", index=False)
        rows = [dict(code=c, preference="; ".join("+".join(sc) for sc in pref)) for c, pref in SCOPES.items()]
        pd.DataFrame(rows).to_csv(OUT / "s0_ces_map.csv", index=False)
        wages_47().to_csv(OUT / "s0_wages_47.csv")
        se_share().to_csv(OUT / "s0_selfemployed_share.csv")
        print("ces done", flush=True)
    if part in ("vintages", "all"):
        v = vintage_inputs()
        keys = v["keys"]
        np.savez_compressed(OUT / "s0_vintages.npz", ind=np.array(keys.get_level_values("ind")),
                            asset=np.array(keys.get_level_values("asset")), years=np.array(v["years"]),
                            I1=v["I1"].to_numpy(), I2=v["I2"].to_numpy(), delta=v["delta"].to_numpy(),
                            K1=v["K1"].to_numpy(), K2=v["K2"].to_numpy(), D2=v["D2"].to_numpy())
        C = vintage_check()
        C.to_csv(OUT / "s0_vintage_check.csv", index=False)
        C.attrs["by_code"].to_csv(OUT / "s0_vintage_check_by_code.csv")
        am = sorted(set(keys.get_level_values("asset")))
        pd.DataFrame([dict(asset=a, producer=asset_producer(a)) for a in am]).to_csv(OUT / "s0_asset_map.csv", index=False)
        pd.DataFrame([dict(fa_industry=i, code=c, scope="unproductive" if i in FA_UNPROD else "productive")
                      for i, c in FA_DET.items()]).to_csv(OUT / "s0_fa_industry_map.csv", index=False)
        print("vintages done", flush=True)
    if part in ("lambda", "all"):
        lambda_index().rename("index_L_per_NDP_1947=1").to_csv(OUT / "s0_lambda_index_pre1947.csv")
        print("lambda index done", flush=True)


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "all")
