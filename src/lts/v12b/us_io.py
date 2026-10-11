"""Iteration 12b, stage 1: US data assembly at the common 47-industry level, 1948-2023 (pre_registration_v12b.md 1.1;
implementation details: journal 1).

For each year: total-use industry-by-industry coefficients A (industry technology, imports competitive), net product n,
household consumption c, private fixed investment composition, value added, industry output x; persons engaged by
industry (12 SIC/NAICS bridge groups, allocated within groups); compensation; primary energy row (EIA);
productive capital stock by asset type mapped to producing industries (BEA FA 2.1, 3.1E/S/I); real stocks (3.2E/S/I).
"""
from __future__ import annotations

import zipfile
from functools import lru_cache

import numpy as np
import pandas as pd

from ..v6.series import ROOT, fa, nipa

RAW = ROOT / "data" / "raw"
R12B = RAW / "v12b"
FAD = RAW / "v6" / "bea_fa"
YEARS = range(1948, 2024)

CODES = ["111CA", "113FF", "211", "212", "213", "22", "23", "321", "327", "331", "332", "333", "334", "335", "3361MV",
         "3364OT", "337", "339", "311FT", "313TT", "315AL", "322", "323", "324", "325", "326", "42", "44RT", "48", "493",
         "51", "52", "531", "532RL", "54", "55", "56", "61", "62", "71", "721", "722", "81", "GFG", "GFE", "GSLG", "GSLE"]
AGG = {"441": "44RT", "445": "44RT", "452": "44RT", "4A0": "44RT", "481": "48", "482": "48", "483": "48", "484": "48",
       "485": "48", "486": "48", "487OS": "48", "511": "51", "512": "51", "513": "51", "514": "51", "521CI": "52",
       "523": "52", "524": "52", "525": "52", "HS": "531", "ORE": "531", "5411": "54", "5415": "54", "5412OP": "54",
       "561": "56", "562": "56", "621": "62", "622": "62", "622HO": "62", "623": "62", "624": "62", "711AS": "71",
       "713": "71", "GFGD": "GFG", "GFGN": "GFG"}
MFG = ["321", "327", "331", "332", "333", "334", "335", "3361MV", "3364OT", "337", "339", "311FT", "313TT", "315AL",
       "322", "323", "324", "325", "326"]
GROUPS = {"AGR": ["111CA", "113FF"], "MIN": ["211", "212", "213"], "CON": ["23"], "MFGINF": MFG + ["51"],
          "TRN": ["48", "493"], "UTL": ["22"], "WHL": ["42"], "RET": ["44RT", "722"], "FIRE": ["52", "531"],
          "SRV": ["532RL", "54", "55", "56", "61", "62", "71", "721", "81"], "FEDG": ["GFG"], "FEDE": ["GFE"],
          "SLG": ["GSLG"], "SLE": ["GSLE"]}
UNPRODUCTIVE = {"42", "44RT", "52", "531", "532RL", "55", "56", "GFG", "GFE", "GSLG", "GSLE"}
NECESSITIES = ["111CA", "311FT", "313TT", "315AL", "22", "324", "531", "48", "62"]
# BEA FA table 2.1 leaf lines (private nonresidential) -> producing industry; broad type E / S / I
ASSET_LINES = {
    5: ("334", "E"), 6: ("334", "E"), 7: ("339", "E"), 8: ("334", "E"), 9: ("333", "E"), 10: ("333", "E"),
    12: ("332", "E"), 13: ("333", "E"), 14: ("333", "E"), 15: ("333", "E"), 16: ("333", "E"), 17: ("335", "E"),
    20: ("3361MV", "E"), 21: ("3361MV", "E"), 22: ("3361MV", "E"), 23: ("3364OT", "E"), 24: ("3364OT", "E"),
    25: ("3364OT", "E"), 27: ("337", "E"), 28: ("333", "E"), 29: ("333", "E"), 30: ("333", "E"), 31: ("333", "E"),
    32: ("335", "E"), 33: ("339", "E"),
    38: ("23", "S"), 41: ("23", "S"), 42: ("23", "S"), 43: ("23", "S"), 44: ("23", "S"), 45: ("23", "S"),
    46: ("23", "S"), 47: ("23", "S"), 48: ("23", "S"), 51: ("23", "S"), 52: ("23", "S"), 53: ("23", "S"),
    55: ("213", "S"), 56: ("213", "S"), 58: ("23", "S"), 59: ("23", "S"), 60: ("23", "S"), 61: ("23", "S"),
    63: ("23", "S"), 64: ("23", "S"), 65: ("23", "S"), 66: ("23", "S"),
    79: ("51", "I"), 80: ("54", "I"), 81: ("54", "I"), 85: ("54", "I"), 86: ("54", "I"), 87: ("54", "I"),
    88: ("54", "I"), 89: ("54", "I"), 90: ("54", "I"), 91: ("54", "I"), 93: ("54", "I"), 94: ("54", "I"),
    96: ("54", "I"), 97: ("54", "I"), 99: ("51", "I"), 100: ("51", "I"), 101: ("51", "I"), 102: ("51", "I"),
    103: ("51", "I")}
BROAD_TOTAL = {"E": 3, "S": 36, "I": 77}
# BEA FA section 3 industry names (normalised) -> 47 codes (depreciation variant) and productive flag
FA_IND = {
    "farms": "111CA", "forestry, fishing, and related activities": "113FF", "oil and gas extraction": "211",
    "mining, except oil and gas": "212", "support activities for mining": "213", "utilities": "22",
    "construction": "23", "wood products": "321", "nonmetallic mineral products": "327", "primary metals": "331",
    "fabricated metal products": "332", "machinery": "333", "computer and electronic products": "334",
    "electrical equipment, appliances, and components": "335", "motor vehicles, bodies and trailers, and parts": "3361MV",
    "other transportation equipment": "3364OT", "furniture and related products": "337",
    "miscellaneous manufacturing": "339", "food and beverage and tobacco products": "311FT",
    "textile mills and textile product mills": "313TT", "apparel and leather and allied products": "315AL",
    "paper products": "322", "printing and related support activities": "323", "petroleum and coal products": "324",
    "chemical products": "325", "plastics and rubber products": "326", "wholesale trade": "42", "retail trade": "44RT",
    "air transportation": "48", "railroad transportation": "48", "water transportation": "48",
    "truck transportation": "48", "transit and ground passenger transportation": "48",
    "pipeline transportation": "48", "other transportation and support activities": "48",
    "warehousing and storage": "493", "information": "51", "finance and insurance": "52", "real estate": "531",
    "rental and leasing services and lessors of intangible assets": "532RL",
    "professional, scientific, and technical services": "54", "management of companies and enterprises": "55",
    "administrative and waste management services": "56", "educational services": "61",
    "health care and social assistance": "62", "arts, entertainment, and recreation": "71", "accommodation": "721",
    "food services and drinking places": "722", "other services, except government": "81"}


def norm(s):
    return " ".join(str(s).replace("\\", " ").split()).lower().rstrip("0123456789 ").strip()


def num(v):
    return pd.to_numeric(pd.Series(v), errors="coerce").fillna(0.0).to_numpy()


# ================================================================ IO tables
@lru_cache(None)
def _hist_book(kind, period):
    f = R12B / f"bea_IO{'Use' if kind == 'use' else 'Make'}_Before_Redefinitions{'_PRO' if kind == 'use' else ''}_{period}_Summary.xlsx"
    return pd.read_excel(f, sheet_name=None, header=None)


def _table(kind, year):
    if year >= 1997:
        from ..bea import _table as bt
        return bt(kind, year)
    period = "1947-1962" if year <= 1962 else "1963-1996"
    d = _hist_book(kind, period)[str(year)]
    cols = d.iloc[6].tolist()
    body = d.iloc[7:].copy()
    body.columns = ["code", "name"] + [str(c).strip() for c in cols[2:]]
    body = body[body["code"].notna()]
    body["code"] = body["code"].astype(str).str.strip()
    body = body.set_index("code").drop(columns="name")
    body = body.loc[:, [c for c in body.columns if c not in ("nan", "None")]]
    body = body.loc[~body.index.duplicated(), ~body.columns.duplicated()]
    return body.apply(lambda c: pd.to_numeric(c, errors="coerce")).fillna(0.0)


def agg_index(idx):
    return [AGG.get(c, c) for c in idx]


@lru_cache(None)
def year_tables(year):
    """Aggregated (47) use (commodity x industry/FD), make (industry x commodity), industry output x, VA, comp."""
    U = _table("use", year)
    V = _table("make", year)
    fdc = [c for c in U.columns if c.startswith("F")]
    U = U.groupby(agg_index(U.index)).sum().T.groupby(agg_index(U.columns)).sum().T
    V = V.groupby(agg_index(V.index)).sum().T.groupby(agg_index(V.columns)).sum().T
    Uc = U.reindex(index=CODES, columns=CODES).fillna(0.0)
    Vm = V.reindex(index=CODES, columns=CODES).fillna(0.0)
    tot_row = "T008" if "T008" in U.index else None
    x = U.loc[tot_row, CODES].to_numpy(float) if tot_row else Vm.sum(1).to_numpy(float)
    va = U.loc["T006", CODES].to_numpy(float) if "T006" in U.index else U.reindex(["V001", "V002", "V003"])[CODES].sum().to_numpy(float)
    comp = U.loc["V001", CODES].to_numpy(float) if "V001" in U.index else np.full(len(CODES), np.nan)
    fd = U.reindex(index=CODES, columns=fdc).fillna(0.0)
    return dict(U=Uc, V=Vm, x=x, va=va, comp=comp, fd=fd, gdp=float(fd.to_numpy().sum()))


def system(year):
    t = year_tables(year)
    V, U, x = t["V"].to_numpy(float), t["U"].to_numpy(float), t["x"]
    q = V.sum(0)
    D = np.divide(V, q, out=np.zeros_like(V), where=q > 0)        # industry x commodity market shares
    A = np.divide(D @ U, x, out=np.zeros_like(U), where=x > 0)
    A = np.maximum(A, 0.0)
    fd = t["fd"]
    to_ind = lambda cols: D @ fd[[c for c in cols if c in fd.columns]].to_numpy(float).sum(1)
    n = to_ind(list(fd.columns))
    c = to_ind(["F010"])
    inv_cols = [c_ for c_ in fd.columns if c_.startswith("F02") and c_ != "F02R"]
    inv = to_ind(inv_cols)
    return dict(A=A, D=D, n=n, c=c, inv=inv, x=x, va=t["va"], comp=t["comp"], gdp=t["gdp"])


# ================================================================ labour
@lru_cache(None)
def sic_blocks():
    """SIC persons engaged and compensation by bridge group, 1947-1997 (72SIC to 1987, 87SIC 1988-1997)."""
    out = {}
    for sheet_e, sheet_c in (("72SIC_Employment", "72SIC_Components of VA"), ("87SIC_Employment", "87SIC_Components of VA")):
        for sheet, start in ((sheet_e, "Persons engaged"), (sheet_c, "Compensation of employees")):
            d = pd.read_excel(RAW / "v9b" / "bea_GDPbyInd_VA_SIC.xls", sheet_name=sheet, header=None)
            years = [int(float(v)) for v in d.iloc[0, 2:] if str(v).replace(".0", "").isdigit()]
            i0 = d.index[d.iloc[:, 0].astype(str).str.startswith(start)][0]
            blk = d.iloc[i0 + 1:]
            code = blk.iloc[0, 0]
            blk = blk[blk.iloc[:, 0] == code]
            names = [" ".join(str(v).split()) for v in blk.iloc[:, 1]]
            vals = blk.iloc[:, 2:2 + len(years)].apply(lambda c: pd.to_numeric(c, errors="coerce"))
            vals.index = names
            vals.columns = years
            # federal / state-local general government and enterprises by position
            pos = {}
            for i, nm in enumerate(names):
                pos.setdefault(nm, []).append(i)
            g = {}
            for grp, line in (("AGR", "Agriculture, forestry, and fishing"), ("MIN", "Mining"), ("CON", "Construction"),
                              ("TRN", "Transportation"), ("UTL", "Electric, gas, and sanitary services"),
                              ("WHL", "Wholesale trade"), ("RET", "Retail trade"),
                              ("FIRE", "Finance, insurance, and real estate")):
                g[grp] = vals.iloc[pos[line][0]]
            g["MFGINF"] = (vals.iloc[pos["Manufacturing"][0]] + vals.iloc[pos["Communications"][0]]
                           + vals.iloc[pos["Motion pictures"][0]])
            g["SRV"] = vals.iloc[pos["Services"][0]] - vals.iloc[pos["Motion pictures"][0]]
            fed, sl = pos["Federal"][0], pos["State and local"][0]
            gg, ge = pos["General government"], pos["Government enterprises"]
            g["FEDG"] = vals.iloc[[i for i in gg if i > fed and i < sl][0]]
            g["FEDE"] = vals.iloc[[i for i in ge if i > fed and i < sl][0]]
            g["SLG"] = vals.iloc[[i for i in gg if i > sl][0]]
            g["SLE"] = vals.iloc[[i for i in ge if i > sl][0]]
            G = pd.DataFrame(g)
            kind = "persons" if sheet == sheet_e else "comp"
            prev = out.get(kind)
            if prev is None:
                out[kind] = G
            else:
                out[kind] = pd.concat([prev[prev.index < 1988], G[G.index >= 1988]])
    return out


@lru_cache(None)
def naics_comp_hist():
    """NAICS compensation by 47 codes, 1987-1996 (BEA GDPbyInd 1947-1997, Components; earlier years suppressed)."""
    z = zipfile.ZipFile(R12B / "bea_AllTablesHist.zip")
    d = pd.read_excel(z.open("GDPbyInd_VA_1947-1997.xlsx"), sheet_name="Components", header=None)
    years = [int(v) for v in d.iloc[5, 2:] if str(v).isdigit()]
    names = [norm(v) for v in d.iloc[:, 1]]
    lookup = {"farms": "111CA", "forestry, fishing, and related activities": "113FF", "oil and gas extraction": "211",
              "mining, except oil and gas": "212", "support activities for mining": "213", "utilities": "22",
              "construction": "23", "wood products": "321", "nonmetallic mineral products": "327",
              "primary metals": "331", "fabricated metal products": "332", "machinery": "333",
              "computer and electronic products": "334", "electrical equipment, appliances, and components": "335",
              "motor vehicles, bodies and trailers, and parts": "3361MV", "other transportation equipment": "3364OT",
              "furniture and related products": "337", "miscellaneous manufacturing": "339",
              "food and beverage and tobacco products": "311FT", "textile mills and textile product mills": "313TT",
              "apparel and leather and allied products": "315AL", "paper products": "322",
              "printing and related support activities": "323", "petroleum and coal products": "324",
              "chemical products": "325", "plastics and rubber products": "326", "wholesale trade": "42",
              "retail trade": "44RT", "transportation and warehousing": "48+493", "information": "51",
              "finance and insurance": "52", "real estate": "531",
              "rental and leasing services and lessors of intangible assets": "532RL",
              "professional, scientific, and technical services": "54",
              "management of companies and enterprises": "55",
              "administrative and waste management services": "56", "educational services": "61",
              "health care and social assistance": "62", "arts, entertainment, and recreation": "71",
              "accommodation": "721", "food services and drinking places": "722",
              "other services, except government": "81", "federal": "GF", "state and local": "GSL"}
    out = {}
    for i, nm in enumerate(names):
        if nm in lookup and lookup[nm] not in out and i + 1 < len(names) and names[i + 1] == "compensation of employees":
            out[lookup[nm]] = pd.to_numeric(d.iloc[i + 1, 2:2 + len(years)], errors="coerce").to_numpy()
    return pd.DataFrame(out, index=years)


def comp_allocator(year):
    """Allocator within bridge groups ~ compensation by 47 codes (1987+: actual; earlier: VA_t x (comp/VA)_1987)."""
    t = year_tables(year)
    if year >= 1997:
        return pd.Series(t["comp"], index=CODES)
    if year >= 1987:
        return hist_comp_47(year)
    t87 = year_tables(1987)
    share = np.divide(hist_comp_47(1987).to_numpy(), t87["va"], out=np.zeros(len(CODES)), where=t87["va"] > 0)
    return pd.Series(np.clip(t["va"], 0, None) * share, index=CODES)


def hist_comp_47(year):
    H = naics_comp_hist().loc[year]
    t = year_tables(year)
    va = pd.Series(t["va"], index=CODES)
    out = pd.Series(0.0, index=CODES)
    for k, v in H.items():
        if k == "48+493":
            parts = ["48", "493"]
        elif k == "GF":
            parts = ["GFG", "GFE"]
        elif k == "GSL":
            parts = ["GSLG", "GSLE"]
        else:
            parts = [k]
        w = va[parts].clip(lower=0)
        out[parts] = v * (w / w.sum() if w.sum() > 0 else 1.0 / len(parts))
    return out


SECTION_OF = [(range(1, 4), "A"), (range(5, 10), "B"), (range(10, 34), "C"), (range(35, 36), "D"),
              (range(36, 40), "E"), (range(41, 44), "F"), (range(45, 48), "G"), (range(49, 54), "H"),
              (range(55, 57), "I"), (range(58, 64), "J"), (range(64, 67), "K"), (range(68, 69), "L"),
              (range(69, 76), "M"), (range(77, 83), "N"), (range(84, 85), "O"), (range(85, 86), "P"),
              (range(86, 89), "Q"), (range(90, 94), "R"), (range(94, 97), "S"), (range(97, 99), "T")]


def section(div):
    for r, s in SECTION_OF:
        if div in r:
            return s
    return None


@lru_cache(None)
def nipa_groups(year):
    """1998+: persons (thousands) = NIPA 6.5D FTE x (1 + self-employed / employees of the NACE SECTION, OECD Table 7
    USA); section level because division ratios are implausible for insurance (K65: SAL 15 000 vs SELF 154 000;
    journal 1). Government: no self-employed."""
    from ..bea import FTE_SERIES, _nipa
    from ..codes import BEA_TO_NACE
    from .. import oecd
    nip = _nipa()
    t7 = oecd.series("USA", "table7", "SELF", "JB")
    s7 = oecd.series("USA", "table7", "SAL", "JB")
    yy = year if year in t7.index else min(t7.index, key=lambda t: abs(t - year))
    m = {}
    for lbl, ser in FTE_SERIES.items():
        fte = float(nip.loc[year, ser]) / 1000.0 if year in nip.index else np.nan
        parts = lbl.split("+")
        if lbl.startswith("G"):
            ratio = 0.0
        else:
            secs = sorted({section(dv) for p_ in parts for dv in BEA_TO_NACE[p_]} - {None})
            num_, den_ = sum(float(t7.loc[yy].get(sc, np.nan)) for sc in secs), sum(float(s7.loc[yy].get(sc, np.nan)) for sc in secs)
            ratio = num_ / den_ if np.isfinite(num_) and den_ > 0 else 0.0
        code = "531" if lbl == "HS+ORE" else ("GFG" if lbl == "GFGD+GFGN" else AGG.get(parts[0], parts[0]))
        m[code] = m.get(code, 0.0) + fte * (1 + ratio)
    return pd.Series(m)


def persons(year, method="group"):
    """Persons engaged by 47 codes. method 'group': bridge groups allocated within by compensation (all years);
    'detail' (1998+): NIPA FTE x (1 + self ratio) directly by industry."""
    alloc = comp_allocator(year).clip(lower=0)
    if year >= 1998:
        direct = nipa_groups(year).reindex(CODES).fillna(0.0)
        if method == "detail":
            return direct
        grp_tot = {g: float(direct[c].sum()) for g, c in GROUPS.items()}
    else:
        P = sic_blocks()["persons"]
        grp_tot = {g: float(P.loc[year, g]) for g in GROUPS}
    out = pd.Series(0.0, index=CODES)
    for g, cs in GROUPS.items():
        w = alloc[cs]
        out[cs] = grp_tot[g] * (w / w.sum() if w.sum() > 0 else 1.0 / len(cs))
    return out


def compensation(year):
    """Compensation by 47 codes: 1987+ NAICS (actual); before: SIC group totals allocated by the allocator."""
    if year >= 1987:
        return comp_allocator(year)
    C = sic_blocks()["comp"]
    alloc = comp_allocator(year).clip(lower=0)
    out = pd.Series(0.0, index=CODES)
    for g, cs in GROUPS.items():
        w = alloc[cs]
        out[cs] = float(C.loc[year, g]) * (w / w.sum() if w.sum() > 0 else 1.0 / len(cs))
    return out


# ================================================================ energy (EIA, quadrillion Btu)
@lru_cache(None)
def eia():
    e = pd.read_csv(R12B / "eia_T01.02.csv")
    e = e[e.YYYYMM.astype(str).str.endswith("13")].copy()
    e["year"] = e.YYYYMM.astype(str).str[:4].astype(int)
    e["v"] = pd.to_numeric(e.Value, errors="coerce")
    return e.pivot_table(index="year", columns="MSN", values="v")


def energy_row(year, x):
    E = eia().loc[year]
    prod = {"211": E.PAPRBUS + E.NGPRBUS + E.NLPRBUS, "212": E.CLPRBUS,
            "22": E.NUETBUS + E.HVTCBUS + E.GETCBUS + E.SOTCBUS + E.WYTCBUS, "113FF": E.BMPRBUS}
    row = np.zeros(len(CODES))
    for k, v in prod.items():
        j = CODES.index(k)
        row[j] = float(np.nan_to_num(v)) / x[j] if x[j] > 0 else 0.0
    return row, float(sum(np.nan_to_num(v) for v in prod.values()))


# ================================================================ capital (BEA FA)
@lru_cache(None)
def fa_lines(sheet):
    sec = int(sheet[4])
    x = pd.read_excel(FAD / f"Section{sec}All_xls.xlsx", sheet_name=sheet, header=None)
    hdr = x.index[x.iloc[:, 0].astype(str).str.strip() == "Line"][0]
    h = pd.to_numeric(x.iloc[hdr], errors="coerce")
    cols = [i for i in range(2, x.shape[1]) if np.isfinite(h.iloc[i])]
    rows = x.iloc[hdr + 1:]
    rows = rows[pd.to_numeric(rows.iloc[:, 0], errors="coerce").notna()]
    vals = rows.iloc[:, cols].apply(lambda c: pd.to_numeric(c, errors="coerce"))
    vals.columns = h.iloc[cols].astype(int).to_numpy()
    vals.index = pd.to_numeric(rows.iloc[:, 0]).astype(int)
    names = pd.Series([norm(v) for v in rows.iloc[:, 1]], index=vals.index)
    return vals, names


def fa_by_name(sheet):
    vals, names = fa_lines(sheet)
    out = {}
    for ln, nm in names.items():
        if nm and nm not in out:
            out[nm] = vals.loc[ln]
    return out


# productive / unproductive sectors at FA level (iteration 9, lts.v9.stage7)
FA_SECTORS_PROD = ["agriculture, forestry, fishing, and hunting", "mining", "utilities", "construction", "manufacturing",
                   "transportation and warehousing", "information", "computer systems design and related services",
                   "miscellaneous professional, scientific, and technical services",
                   "waste management and remediation services", "educational services",
                   "health care and social assistance", "arts, entertainment, and recreation",
                   "accommodation and food services", "other services, except government"]
FA_SECTORS_UNPROD = ["wholesale trade", "retail trade", "finance and insurance", "real estate and rental and leasing",
                     "legal services", "management of companies and enterprises", "administrative and support services"]
def _get(d, nm):
    s = d.get(nm)
    if s is None and nm == "health care and social assistance":
        s = d.get("health and social assistance")
    return s


def _sector_sum(d, names, year):
    tot = 0.0
    for nm in names:
        s = _get(d, nm)
        if s is None:
            raise KeyError(nm)
        tot += float(np.nan_to_num(s.get(year, np.nan)))
    return tot


def capital(year, scope="productive"):
    """Current-cost net stock of private nonresidential fixed assets mapped to producing industries (47 vector),
    by broad type; scope 'productive' (iteration-9 productive industries) or 'all'."""
    v21, _ = fa_lines("FAAt201-A")
    out = {}
    for typ, tl in BROAD_TOTAL.items():
        lines = [ln for ln, (c, t) in ASSET_LINES.items() if t == typ]
        comp = v21.loc[lines, year].clip(lower=0)
        comp = comp / comp.sum()
        if scope == "all":
            total = float(v21.loc[tl, year])
        else:
            total = _sector_sum(fa_by_name({"E": "FAAt301E-A", "S": "FAAt301S-A", "I": "FAAt301I-A"}[typ]),
                                FA_SECTORS_PROD, year)
        vec = np.zeros(len(CODES))
        for ln, sh in comp.items():
            vec[CODES.index(ASSET_LINES[ln][0])] += total * sh
        out[typ] = vec
    return out


def real_capital(year, scope="productive"):
    """Real stock (2017 $) by broad type: sum over sectors of current-cost 2017 x quantity index / index 2017."""
    out = {}
    for typ in ("E", "S", "I"):
        if scope == "all":
            q, _ = fa_lines("FAAt202-A")
            v, _ = fa_lines("FAAt201-A")
            tl = BROAD_TOTAL[typ]
            out[typ] = float(v.loc[tl, 2017] * q.loc[tl, year] / q.loc[tl, 2017])
            continue
        kc = fa_by_name({"E": "FAAt301E-A", "S": "FAAt301S-A", "I": "FAAt301I-A"}[typ])
        kq = fa_by_name({"E": "FAAt302E-A", "S": "FAAt302S-A", "I": "FAAt302I-A"}[typ])
        tot = 0.0
        for nm in FA_SECTORS_PROD:
            c, qq = _get(kc, nm), _get(kq, nm)
            if c is None or qq is None or not np.isfinite(qq.get(year, np.nan)) or not np.isfinite(qq.get(2017, np.nan)) or qq.get(2017) == 0:
                continue
            tot += float(c.loc[2017] * qq.loc[year] / qq.loc[2017])
        out[typ] = tot
    return out


def depreciation(year):
    """Current-cost depreciation by 47 codes (FA 3.4E + 3.4S + 3.4I by industry); government 0."""
    out = np.zeros(len(CODES))
    for sh in ("FAAt304E-A", "FAAt304S-A", "FAAt304I-A"):
        d = fa_by_name(sh)
        for nm, code in FA_IND.items():
            s = d.get(nm)
            if s is not None:
                out[CODES.index(code)] += float(np.nan_to_num(s.get(year, np.nan)))
    return out


# ================================================================ NIPA
@lru_cache(None)
def nipa_series():
    return nipa(["A065RC", "DPCERX", "A191RD", "DPCERG"])


@lru_cache(None)
def inventories():
    """Private inventories, end of year (Q4), current dollars (NIPA 5.8.5, quarterly file)."""
    d = pd.read_csv(RAW / "v8" / "NipaDataQ.txt", dtype=str)
    d.columns = ["code", "period", "v"]
    reg = pd.read_csv(RAW / "bea" / "SeriesRegister.txt", dtype=str) if (RAW / "bea" / "SeriesRegister.txt").exists() else None
    code = None
    if reg is not None:
        r = reg[reg.iloc[:, 5].astype(str).str.contains("T50805B:1(?:\\||$)", regex=True)]
        if len(r):
            code = r.iloc[0, 0]
    if code is None:
        return pd.Series(dtype=float)
    s = d[(d.code == code) & d.period.str.endswith("Q4")].copy()
    s["year"] = s.period.str[:4].astype(int)
    return pd.to_numeric(s.set_index("year").v.str.replace(",", ""), errors="coerce")
