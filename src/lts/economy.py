"""Build `core.Economy` objects (one country-year) from FIGARO + OECD data.

Decisions that affect results are flagged with the ID used in ASSUMPTIONS.md.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from . import figaro, oecd
from .core import Economy
from .figaro import INDUSTRIES

# A-MERGE: industries that the national labour data do not report separately (zero/missing
# employment) are merged with the industry that absorbs them in the source. Evidence: OECD
# Table 7 USA reports no persons for C21, J61, M71-M73, N78-N79, R93, S95-S96, A03 while the
# listed absorbers carry the combined employment (e.g. M74_75 = 6.5m persons, 35 h per kEUR).
MERGES = {
    "US": [["C20", "C21"], ["J59_60", "J61"], ["M71", "M72", "M73", "M74_75"], ["N78", "N79", "N80-82"],
           ["R90-92", "R93"], ["S94", "S95", "S96"], ["A02", "A03"]],
}
# A-EXCL-BASE: excluded from every metric (not from the IO system): no market output / no output.
EXCLUDE_BASE = {"T", "U"}
# A-ROBUST: robustness exclusions
EXCLUDE_RENT_FIN_MINING = {"B", "K64", "K65", "K66", "L"}
EXCLUDE_GOV = {"O84", "P85", "Q86", "Q87_88"}

# Alternative commodity bases (FIGARO codes)
BASES = {"electricity": "D35", "oil_refined": "C19", "mining_energy": "B", "metals": "C24",
         "agriculture": "A01", "chemicals": "C20", "food": "C10-12", "transport": "H49", "finance": "K64"}


@lru_cache(maxsize=64)
def _fig(year: int):
    return figaro.load(year)


def _group_matrix(labels: list[str], merges: list[list[str]]) -> tuple[list[str], np.ndarray]:
    """Aggregation matrix G (n_groups x n) and group labels (first member + '+' ...)."""
    gid = {}
    names = []
    for grp in merges:
        for m in grp:
            gid[m] = len(names)
        names.append("+".join(grp))
    for lab in labels:
        if lab not in gid:
            gid[lab] = len(names)
            names.append(lab)
    G = np.zeros((len(names), len(labels)))
    for j, lab in enumerate(labels):
        G[gid[lab], j] = 1.0
    return names, G


def national_blocks(country2: str, year: int) -> dict:
    """Domestic / imported flows of one country from the FIGARO world table (money, MIO_EUR)."""
    d = _fig(year)
    C = list(d["countries"])
    N, K = len(INDUSTRIES), len(figaro.FD)
    c = C.index(country2)
    rows_dom = slice(c * N, (c + 1) * N)
    cols = slice(c * N, (c + 1) * N)
    Z = d["Z"]
    Zc = Z[:, cols]                           # all origins -> country c industries
    Zdom = Zc[rows_dom, :]
    Zimp = Zc.reshape(len(C), N, N).sum(0) - Zdom
    Fc = d["F"][:, c * K:(c + 1) * K]
    Fdom = Fc[rows_dom, :]
    Fimp = Fc.reshape(len(C), N, K).sum(0) - Fdom
    V = d["V"][:, cols]
    x = Z[rows_dom, :].sum(1) + d["F"][rows_dom, :].sum(1)
    return dict(Zdom=Zdom, Zimp=Zimp, Fdom=Fdom, Fimp=Fimp, V=V, x=x, varows=list(d["varows"]),
                fd=list(d["fd"]))


def build(country3: str, year: int, cfc_fallback_country: str = "USA") -> tuple[Economy, dict]:
    c2 = oecd.ISO3_TO_2[country3]
    b = national_blocks(c2, year)
    x = b["x"]
    xs = pd.Series(x, index=INDUSTRIES)
    lab = oecd.labour_block(country3, year, xs)
    info = dict(country=country3, year=year, labour_log=lab["log"])
    # capital consumption -------------------------------------------------------------------------
    D1 = b["V"][b["varows"].index("D1")]
    GOS = b["V"][b["varows"].index("B2A3G")]
    cfc_ratio = lab["cfc_ratio"]
    if lab["log"].get("cfc_source") == "B2A3G - B2A3N" and (cfc_ratio.fillna(0) == 0).all():
        # A-CAP-MEX: no CFC by industry (net = gross surplus in OECD Table 6). Borrow CFC/GOS ratios
        # by industry from the fallback country (same year) and apply them to own GOS.
        fb = build_cfc_gos_ratio(cfc_fallback_country, year)
        cfc = (fb * np.clip(GOS, 0, None)).to_numpy()
        info["cfc_note"] = f"CFC/GOS ratios borrowed from {cfc_fallback_country}"
    else:
        cfc = (cfc_ratio.fillna(cfc_ratio.median()) * xs).to_numpy()
    cfc = np.minimum(cfc, np.clip(GOS, 0, None))  # CFC cannot exceed gross operating surplus
    # labour income incl. imputed self-employed (A-LI) ------------------------------------------------
    es = lab["employee_share"].reindex(INDUSTRIES).fillna(1.0).clip(0.2, 1.0).to_numpy()
    va = b["V"].sum(0)
    li = np.minimum(D1 / es, np.clip(va - cfc, D1, None))
    hours = lab["hours"].reindex(INDUSTRIES).fillna(0).to_numpy()
    price = lab["price_index"].reindex(INDUSTRIES).to_numpy()
    # merges -------------------------------------------------------------------------------------------
    names, G = _group_matrix(INDUSTRIES, MERGES.get(c2, []))
    agg = lambda v: G @ v
    aggM = lambda M: G @ M @ G.T
    xg = agg(x)
    Zd, Zi = aggM(b["Zdom"]), aggM(b["Zimp"])
    P3 = b["fd"].index("P3_S14")
    P51 = b["fd"].index("P51G")
    with np.errstate(invalid="ignore", divide="ignore"):
        A = np.where(xg > 0, Zd / xg, 0.0)
        Am = np.where(xg > 0, Zi / xg, 0.0)
        pg = np.where(xg > 0, (G @ (np.nan_to_num(price) * x)) / xg, np.nan)
    keep = xg > 0  # drop industries without output (U)
    idx = np.where(keep)[0]
    e = Economy(labels=[names[i] for i in idx], A=A[np.ix_(idx, idx)], Am=Am[np.ix_(idx, idx)], x=xg[idx],
                hours=agg(hours)[idx], wages=agg(D1)[idx], cfc=agg(cfc)[idx],
                gfcf_dom=agg(b["Fdom"][:, P51])[idx], gfcf_imp=agg(b["Fimp"][:, P51])[idx],
                hh_dom=agg(b["Fdom"][:, P3])[idx], hh_imp=agg(b["Fimp"][:, P3])[idx],
                va=agg(va)[idx], labour_income=agg(li)[idx],
                meta=dict(price_index=pg[idx], persons=agg(lab["persons"].reindex(INDUSTRIES).fillna(0).to_numpy())[idx]))
    # identity check: column sums of coefficients + value added share = 1
    colsum = e.A.sum(0) + e.Am.sum(0) + e.va / e.x
    info["max_colsum_gap"] = float(np.abs(colsum - 1).max())
    info["n"] = e.n
    return e, info


@lru_cache(maxsize=64)
def build_cfc_gos_ratio(country3: str, year: int) -> pd.Series:
    c2 = oecd.ISO3_TO_2[country3]
    b = national_blocks(c2, year)
    lab = oecd.labour_block(country3, year, pd.Series(b["x"], index=INDUSTRIES))
    GOS = pd.Series(b["V"][b["varows"].index("B2A3G")], index=INDUSTRIES)
    cfc = lab["cfc_ratio"] * pd.Series(b["x"], index=INDUSTRIES)
    r = (cfc / GOS.where(GOS > 0)).clip(0, 1)
    return r.fillna(r.median())


def education_weights(country3: str, year: int, labels: list[str], scheme: str,
                      label_divs: dict | None = None) -> np.ndarray:
    """Reduction coefficients for skilled labour from education structure (A-EDU).

    'train' : kappa_e = 1 + (S_e - S_L)/40  -- extra years of schooling amortised over a 40-year
              working life (a 'training-time' reduction in the spirit of Hilferding / Cockshott).
    'years' : kappa_e = S_e / S_L          -- proportional to years of schooling (steeper).
    S_L, S_M, S_H = 9, 12.5, 16.5 years (ISCED 0-2, 3-4, 5-8).
    label_divs: optional {label: set of NACE divisions} for non-FIGARO classifications (BEA).
    """
    from .codes import divisions
    S = {"ISCED_L": 9.0, "ISCED_M": 12.5, "ISCED_H": 16.5}
    if scheme == "train":
        kappa = {k: 1 + (v - 9.0) / 40.0 for k, v in S.items()}
    elif scheme == "years":
        kappa = {k: v / 9.0 for k, v in S.items()}
    else:
        raise ValueError(scheme)
    sh = oecd.education_shares(country3, year)          # by FIGARO industry
    w = (sh * pd.Series(kappa)).sum(1).where(sh.notna().all(1))
    out = []
    for lab in labels:
        if label_divs is None:
            vals = [w.get(m, np.nan) for m in lab.split("+")]
        else:
            d = label_divs[lab]
            vals = [w[j] for j in INDUSTRIES if divisions(j) and divisions(j) & d]
        out.append(np.nanmean(vals) if len(vals) and np.isfinite(vals).any() else np.nan)
    out = np.array(out)
    return np.where(np.isfinite(out), out, np.nanmean(out))
