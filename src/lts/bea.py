"""US long series (1998-2023) from BEA annual IO tables (summary level, before redefinitions).

Industry-by-industry coefficients from Make/Use under the industry-technology assumption
(A = D U / g, D = market-share matrix) -- A-BEA-IxI. Imports from BEA import matrices.
Labour: NIPA Table 6.5D full-time-equivalent employees x (1 + self-employed/employees ratio of
the corresponding ISIC activity in OECD Table 7 USA) -- A-BEA-L (hours per FTE assumed equal across
industries). CFC: OECD Table 6 USA P51C/P1 by ISIC activity. Prices: BEA chain-type price indexes
for gross output (TGO104).
"""
from __future__ import annotations

import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import oecd
from .codes import BEA_TO_NACE, divisions
from .core import Economy

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "bea"

# merged groups (A-BEA-MERGE): housing (imputed rents, no employees) with other real estate,
# federal general government defense + nondefense (one FTE series)
BEA_MERGES = [["HS", "ORE"], ["GFGD", "GFGN"]]

FTE_SERIES = {
    "111CA": "N4305C", "113FF": "N4306C", "211": "N4308C", "212": "N4309C", "213": "N4310C", "22": "N4311C",
    "23": "N4312C", "321": "N4315C", "327": "N4316C", "331": "N4317C", "332": "N4318C", "333": "N4319C",
    "334": "N4320C", "335": "N4321C", "3361MV": "N4322C", "3364OT": "N4323C", "337": "N4324C", "339": "N4325C",
    "311FT": "N4327C", "313TT": "N4328C", "315AL": "N4329C", "322": "N4330C", "323": "N4331C", "324": "N4332C",
    "325": "N4333C", "326": "N4334C", "42": "N4335C", "441": "N4339C", "445": "N4340C", "452": "N4341C",
    "4A0": "N4342C", "481": "N4344C", "482": "N4345C", "483": "N4346C", "484": "N4347C", "485": "N4348C",
    "486": "N4349C", "487OS": "N4350C", "493": "N4351C", "511": "N4353C", "512": "N4354C", "513": "N4355C",
    "514": "N4356C", "521CI": "N4358C", "523": "N4359C", "524": "N4360C", "525": "N4361C", "HS+ORE": "N4363C",
    "532RL": "N4364C", "5411": "N4366C", "5415": "N4367C", "5412OP": "N4368C", "55": "N4369C", "561": "N4371C",
    "562": "N4372C", "61": "N4373C", "621": "N4375C", "622": "N4390C", "623": "N4391C", "624": "N4392C",
    "711AS": "N4394C", "713": "N4395C", "721": "N4397C", "722": "N4398C", "81": "N4399C",
    "GFGD+GFGN": "A4378C", "GFE": "B4381C", "GSLG": "A4383C", "GSLE": "B4386C",
}
PRICE_LINE = {
    "111CA": 4, "113FF": 5, "211": 7, "212": 8, "213": 9, "22": 10, "23": 11, "321": 14, "327": 15, "331": 16,
    "332": 17, "333": 18, "334": 19, "335": 20, "3361MV": 21, "3364OT": 22, "337": 23, "339": 24, "311FT": 26,
    "313TT": 27, "315AL": 28, "322": 29, "323": 30, "324": 31, "325": 32, "326": 33, "42": 34, "441": 36,
    "445": 37, "452": 38, "4A0": 39, "481": 41, "482": 42, "483": 43, "484": 44, "485": 45, "486": 46,
    "487OS": 47, "493": 48, "511": 50, "512": 51, "513": 52, "514": 53, "521CI": 56, "523": 57, "524": 58,
    "525": 59, "HS+ORE": 61, "532RL": 64, "5411": 67, "5415": 68, "5412OP": 69, "55": 70, "561": 72, "562": 73,
    "61": 75, "621": 77, "622": 78, "623": 79, "624": 80, "711AS": 83, "713": 84, "721": 86, "722": 87,
    "81": 88, "GFGD+GFGN": 91, "GFE": 94, "GSLG": 96, "GSLE": 97,
}
# FIGARO-equivalent labels used for exclusion sets / named bases
BEA_ROLE = {"22": "D35", "324": "C19", "211": "B", "212": "B", "213": "B", "331": "C24", "111CA": "A01",
            "325": "C20", "311FT": "C10-12", "484": "H49", "521CI": "K64", "523": "K66", "524": "K65",
            "525": "K64", "HS+ORE": "L", "GFGD+GFGN": "O84", "GSLG": "O84", "61": "P85", "621": "Q86",
            "622": "Q86", "623": "Q87_88", "624": "Q87_88"}


def _num(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(lambda c: pd.to_numeric(c, errors="coerce")).fillna(0.0)


@lru_cache(maxsize=None)
def _sheets(kind: str) -> dict:
    if kind == "use":
        f = zipfile.ZipFile(RAW / "AllTablesIO.zip").open("IOUse_Before_Redefinitions_PRO_1997-2023_Summary.xlsx")
    elif kind == "make":
        f = zipfile.ZipFile(RAW / "AllTablesIO.zip").open("IOMake_Before_Redefinitions_PRO_1997-2023_Summary.xlsx")
    else:
        f = RAW / "ImportMatrices_Before_Redefinitions_SUM_1997-2023.xlsx"
    return pd.read_excel(f, sheet_name=None, header=None)


def _table(kind: str, year: int) -> pd.DataFrame:
    d = _sheets(kind)[str(year)]
    cols = d.iloc[5].tolist()
    body = d.iloc[7:].copy()
    body.columns = ["code", "name"] + cols[2:]
    body = body[body["code"].notna() & (body["code"].astype(str).str.len() <= 8)]
    body["code"] = body["code"].astype(str).str.strip()
    body = body.set_index("code").drop(columns="name")
    body.columns = [str(c).strip() for c in body.columns]
    body = body.loc[:, [c for c in body.columns if c != "nan"]]
    body = body.loc[~body.index.duplicated(), ~body.columns.duplicated()]
    return _num(body)


@lru_cache(maxsize=None)
def _nipa() -> pd.DataFrame:
    d = pd.read_csv(RAW / "NipaDataA.txt", dtype=str)
    d.columns = ["code", "year", "value"]
    d = d[d.code.isin(set(FTE_SERIES.values()))]
    d["value"] = pd.to_numeric(d.value.str.replace(",", ""), errors="coerce") * 1000.0
    d["year"] = d.year.astype(int)
    return d.pivot_table(index="year", columns="code", values="value")


@lru_cache(maxsize=None)
def _prices() -> pd.DataFrame:
    x = pd.read_excel(RAW / "GrossOutput.xlsx", sheet_name="TGO104-A", header=None)
    hdr = x.iloc[7].tolist()
    body = x.iloc[8:]
    body = body[pd.to_numeric(body[0], errors="coerce").notna()]
    body.index = body[0].astype(int)
    out = body[[i for i, h in enumerate(hdr) if isinstance(h, float) and h > 1900]]
    out.columns = [int(hdr[i]) for i in out.columns]
    return out.apply(pd.to_numeric, errors="coerce")


def _nace_lookup(row: pd.Series, divs: set[int], mode="index"):
    """Value of the smallest ISIC activity in `row` containing `divs`."""
    best = None
    for code, v in row.dropna().items():
        s = divisions(code)
        if s and divs <= s and (best is None or len(s) < best[0]):
            best = (len(s), v)
    return best[1] if best else np.nan


def build(year: int) -> tuple[Economy, dict]:
    U = _table("use", year)
    V = _table("make", year)
    IM = _table("imp", year)
    inds = [c for c in V.index if c in BEA_TO_NACE]
    comms = [c for c in V.columns if c in U.index]
    Vm = V.loc[inds, comms]
    q = Vm.sum(0)
    D = Vm.div(q.where(q != 0), axis=1).fillna(0.0)            # industry x commodity market shares
    Uc = U.reindex(index=comms, columns=inds).fillna(0.0)
    IMc = IM.reindex(index=comms, columns=inds).fillna(0.0)
    # industry output: sum of make rows (consistent with D) ; check against use-table total
    x = Vm.sum(1)
    Udom = Uc - IMc   # a few small negative cells (scrap) are kept so that column identities hold
    Zdom = D.values @ Udom.values
    Zimp = D.values @ IMc.values
    va_rows = U.reindex(index=["V001", "V002", "V003"], columns=inds).fillna(0.0)
    va = va_rows.sum(0).values
    comp = va_rows.loc["V001"].values
    gos = va_rows.loc["V003"].values
    fd = lambda cols, tab: tab.reindex(index=comms, columns=[c for c in cols if c in tab.columns]).fillna(0).sum(1)
    gf_cols = ["F02S", "F02E", "F02N", "F02R", "F06S", "F06E", "F06N", "F07S", "F07E", "F07N", "F10S", "F10E", "F10N"]
    pce_tot, pce_imp = fd(["F010"], U), fd(["F010"], IM)
    gf_tot, gf_imp = fd(gf_cols, U), fd(gf_cols, IM)
    gc_cols = ["F06C", "F07C", "F10C"]
    gc_tot, gc_imp = fd(gc_cols, U), fd(gc_cols, IM)
    gov_dom = D.values @ (gc_tot - gc_imp).clip(lower=0).values
    gov_imp = D.values @ gc_imp.values
    hh_dom = D.values @ (pce_tot - pce_imp).clip(lower=0).values
    hh_imp = D.values @ pce_imp.values
    gfcf_dom = D.values @ (gf_tot - gf_imp).clip(lower=0).values
    gfcf_imp = D.values @ gf_imp.values
    # merges ----------------------------------------------------------------------------------------
    names, G = [], []
    gid = {}
    for grp in BEA_MERGES:
        for m in grp:
            gid[m] = len(names)
        names.append("+".join(grp))
    for c in inds:
        if c not in gid:
            gid[c] = len(names)
            names.append(c)
    G = np.zeros((len(names), len(inds)))
    for j, c in enumerate(inds):
        G[gid[c], j] = 1
    xg = G @ x.values
    A = (G @ Zdom @ G.T) / xg
    Am = (G @ Zimp @ G.T) / xg
    # A-BEA-NEG: tiny negative coefficients (scrap sales, max |a| ~ 0.003) set to zero; value added
    # reduced by the same amount so that column identities still hold.
    neg = (np.minimum(A, 0).sum(0) + np.minimum(Am, 0).sum(0)) * xg
    A, Am = np.maximum(A, 0), np.maximum(Am, 0)
    # labour -------------------------------------------------------------------------------------------
    nipa = _nipa()
    fte = np.array([nipa.loc[year, FTE_SERIES[n]] if year in nipa.index else np.nan for n in names])
    t7 = oecd.series("USA", "table7", "SELF", "JB")
    s7 = oecd.series("USA", "table7", "SAL", "JB")
    yy = year if year in t7.index else min(t7.index, key=lambda t: abs(t - year))
    ratio_row = (t7.loc[yy] / s7.loc[yy]).replace([np.inf, -np.inf], np.nan)
    self_ratio = []
    for n in names:
        divs = set().union(*[set(BEA_TO_NACE[m]) for m in n.split("+")])
        r = _nace_lookup(ratio_row, divs)
        self_ratio.append(0.0 if n.startswith("G") else (r if np.isfinite(r) else 0.0))
    self_ratio = np.array(self_ratio)
    persons_fte = fte * (1 + self_ratio)
    hours = persons_fte * 1800.0     # scale irrelevant for ratios; A-BEA-L
    # CFC ratio from OECD Table 6 USA -----------------------------------------------------------------
    p1 = oecd.series("USA", "table6", "P1", "XDC", "V")
    cf = oecd.series("USA", "table6", "P51C", "XDC", "V")
    yy = year if year in cf.index else min(cf.index, key=lambda t: abs(t - year))
    cfr = (cf.loc[yy] / p1.loc[yy]).replace([np.inf, -np.inf], np.nan)
    cfc_ratio = np.array([_nace_lookup(cfr, set().union(*[set(BEA_TO_NACE[m]) for m in n.split("+")]))
                          for n in names])
    gos_g = G @ gos
    cfc = np.minimum(np.nan_to_num(cfc_ratio, nan=np.nanmedian(cfc_ratio)) * xg, np.clip(gos_g, 0, None))
    comp_g = G @ comp
    li = np.minimum(comp_g * (1 + self_ratio), np.clip(G @ va - cfc, comp_g, None))
    # prices ---------------------------------------------------------------------------------------------
    pr = _prices()
    price = np.array([pr.loc[PRICE_LINE[n], year] if year in pr.columns else np.nan for n in names])
    e = Economy(labels=names, A=A, Am=Am, x=xg, hours=hours, wages=comp_g, cfc=cfc,
                gfcf_dom=G @ gfcf_dom, gfcf_imp=G @ gfcf_imp, hh_dom=G @ hh_dom, hh_imp=G @ hh_imp,
                va=G @ va + neg, labour_income=li, meta=dict(price_index=price, persons=persons_fte,
                                                              gov_dom=G @ gov_dom, gov_imp=G @ gov_imp))
    colsum = e.A.sum(0) + e.Am.sum(0) + e.va / e.x
    info = dict(country="USA_BEA", year=year, n=e.n, max_colsum_gap=float(np.abs(colsum - 1).max()),
                use_vs_make_output_gap=float(np.abs((U.reindex(columns=inds).loc[["V001", "V002", "V003"]].sum(0)
                                                     + Uc.sum(0)).values / x.values - 1).max()),
                fte_missing=int(np.isnan(fte).sum()))
    return e, info
