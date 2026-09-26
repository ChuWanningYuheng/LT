"""Iteration 3, stage 1: reduction weights for heterogeneous labour that use no industry wages
of the country (variants 1.1a-1.3b, 1.5b) and frozen / transplanted wage coefficients (1.5a, 1.5b).

Every function returns a weight per industry label of an Economy (mean 1 over hours), or None if the
data for the country do not exist.  Data sources (data/raw/v3/, see download_v3.py):
  ILO: employment by ISIC4 section x ISCO-08 major group; mean hourly (monthly fallback) earnings by
       ISCO-08 major group
  OECD EAG: earnings relative to upper secondary (25-64, full-time, both sexes)
  OECD TiMBC: employment by ISCED level and industry aggregate (lts.oecd.education_shares)
  O*NET 29.0 Job Zones + BLS SOC 2010<->2018 and ISCO-08<->SOC 2010 crosswalks + OEWS 2022 national
  OECD UOE enrolments, World Bank population
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .. import oecd
from ..codes import SECTIONS, divisions

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw" / "v3"
OUTV3 = ROOT / "results" / "v3"
ISCO = [str(i) for i in range(10)]
S_YEARS = {"ISCED_L": 9.0, "ISCED_M": 12.5, "ISCED_H": 16.5}      # as in iteration 2 (A-EDU)
JZ_YEARS = {1: 0.25, 2: 1.0, 3: 1.5, 4: 3.0, 5: 5.0}               # pre-registered (JZ5 in {4,5,6})
EAG_LEV = {"ISCED_L": "ISCED11A_0T2", "ISCED_M": "ISCED11A_3_4", "ISCED_H": "ISCED11A_5T8"}
DIV2SEC = {d: s for s, ds in SECTIONS.items() for d in ds}


def label_sections(label: str, label_divs: dict | None = None) -> list[str]:
    ds = label_divs[label] if label_divs else set().union(*[divisions(p) or set() for p in label.split("+")])
    secs = sorted({DIV2SEC[d] for d in ds if d in DIV2SEC})
    return secs


def nearest(years, y):
    return min(years, key=lambda t: abs(t - y))


def normalise(w: np.ndarray, hours: np.ndarray) -> np.ndarray:
    return w / (w @ hours / hours.sum())


def to_labels(sec_vals: pd.Series, labels, label_divs=None) -> np.ndarray:
    out = []
    for lab in labels:
        v = [sec_vals.get(s, np.nan) for s in label_sections(lab, label_divs)]
        v = [a for a in v if np.isfinite(a)]
        out.append(np.mean(v) if v else np.nan)
    out = np.array(out, dtype=float)
    return np.where(np.isfinite(out), out, np.nanmean(out))


# ---------------------------------------------------------------- occupation structure (ILO)
@lru_cache(None)
def _ilo_eco_ocu():
    d = pd.read_csv(RAW / "ilo_eco_ocu.csv", low_memory=False)
    d = d[d.ECO.str.match(r"ECO_ISIC4_[A-U]$") & d.OCU.str.match(r"OCU_ISCO08_\d$")]
    d = d.assign(sec=d.ECO.str[-1], occ=d.OCU.str[-1])
    return d[["REF_AREA", "TIME_PERIOD", "sec", "occ", "OBS_VALUE"]]


def occ_shares(country: str, year: int) -> pd.DataFrame | None:
    d = _ilo_eco_ocu()
    d = d[d.REF_AREA == country]
    if d.empty:
        return None
    y = nearest(sorted(d.TIME_PERIOD.unique()), year)
    t = d[d.TIME_PERIOD == y].pivot_table(index="sec", columns="occ", values="OBS_VALUE", aggfunc="sum")
    t = t.reindex(columns=ISCO).fillna(0)
    return t.div(t.sum(1), axis=0)


@lru_cache(None)
def _ilo_earn():
    frames = []
    for f, kind in (("ilo_EHRA_SEX_OCU_CUR_NB.csv", "hourly"), ("ilo_EMTA_SEX_OCU_CUR_NB.csv", "monthly")):
        d = pd.read_csv(RAW / f, low_memory=False)
        d = d[(d.SEX == "SEX_T") & (d.CUR == "CUR_TYPE_LCU") & d.OCU.str.match(r"OCU_ISCO08_\d$")]
        frames.append(d.assign(kind=kind, occ=d.OCU.str[-1])[["REF_AREA", "TIME_PERIOD", "occ", "OBS_VALUE", "kind"]])
    return pd.concat(frames)


def occ_wage(country: str, year: int) -> tuple[pd.Series, str] | tuple[None, None]:
    """National mean earnings by ISCO major group relative to the employment-weighted mean
    (hourly if available, else monthly -- A-V3-OCCWAGE)."""
    d = _ilo_earn()
    for kind in ("hourly", "monthly"):
        g = d[(d.REF_AREA == country) & (d.kind == kind)]
        g = g.groupby("TIME_PERIOD").filter(lambda t: t.occ.nunique() >= 8)
        if not g.empty:
            y = nearest(sorted(g.TIME_PERIOD.unique()), year)
            w = g[g.TIME_PERIOD == y].groupby("occ").OBS_VALUE.mean().reindex(ISCO)
            return w, f"{kind}:{y}"
    return None, None


# ---------------------------------------------------------------- O*NET job zones -> ISCO
@lru_cache(None)
def isco_prep_years(jz5: float = 5.0) -> pd.Series:
    """Mean preparation time (years) by ISCO-08 major group: O*NET Job Zone per SOC 2018 detailed
    occupation, weighted by OEWS 2022 national employment, via SOC 2018 -> SOC 2010 -> ISCO-08
    (many-to-many links split equally, A-V3-XWALK).  Also writes the crosswalk table."""
    jz = pd.read_csv(RAW / "onet" / "Job Zones.txt", sep="\t")
    jz["soc18"] = jz["O*NET-SOC Code"].str[:7]
    jz = jz.groupby("soc18")["Job Zone"].mean()
    oe = pd.read_excel(RAW / "oesm22nat" / "national_M2022_dl.xlsx")
    oe = oe[oe.O_GROUP == "detailed"][["OCC_CODE", "TOT_EMP"]]
    oe["TOT_EMP"] = pd.to_numeric(oe.TOT_EMP, errors="coerce")
    x18 = pd.read_excel(RAW / "soc_2010_to_2018_crosswalk.xlsx", header=8).iloc[:, :4]
    x18.columns = ["soc10", "t10", "soc18", "t18"]
    xi = pd.read_excel(RAW / "ISCO_SOC_Crosswalk.xls", header=6).iloc[:, :4]
    xi.columns = ["isco", "tisco", "part", "soc10"]
    xi["isco"] = xi.isco.astype(str).str.zfill(4)
    df = oe.rename(columns={"OCC_CODE": "soc18"}).merge(jz.rename("jz"), left_on="soc18", right_index=True, how="left")
    df = df.merge(x18[["soc18", "soc10"]], on="soc18", how="left")
    df["n10"] = df.groupby("soc18").soc10.transform("count").clip(lower=1)
    df = df.merge(xi[["soc10", "isco"]], on="soc10", how="left")
    df["nisco"] = df.groupby(["soc18", "soc10"]).isco.transform("count").clip(lower=1)
    df["emp"] = df.TOT_EMP / df.n10 / df.nisco
    OUTV3.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTV3 / "crosswalk_soc2018_soc2010_isco08_jobzone.csv", index=False)
    yrs = dict(JZ_YEARS)
    yrs[5] = jz5
    df = df.dropna(subset=["jz", "isco", "emp"])
    df["P"] = np.interp(df.jz, list(yrs), list(yrs.values()))
    df["major"] = df.isco.str[0]
    P = df.groupby("major").apply(lambda g: np.average(g.P, weights=g.emp), include_groups=False)
    P = P.reindex(ISCO)
    P["0"] = P.get("0") if np.isfinite(P.get("0", np.nan)) else np.nanmean(P.values)  # armed forces
    return P


# ---------------------------------------------------------------- education (EAG relative earnings)
@lru_cache(None)
def _eag():
    d = pd.read_csv(RAW / "eag_rel_upper.csv", low_memory=False)
    return d[(d.SEX == "_T") & (d.AGE == "Y25T64") & d.OBS_VALUE.notna()]


ISO2 = {"AUT": "AT", "CZE": "CZ", "DEU": "DE", "ESP": "ES", "FRA": "FR", "ITA": "IT", "NLD": "NL",
        "POL": "PL", "GBR": "UK"}


@lru_cache(None)
def _ses_edu():
    d = pd.read_csv(RAW / "earn_ses18_16.tsv.gz", sep="\t")
    k = d.columns[0]
    parts = d[k].str.split(",", expand=True)
    parts.columns = k.split("\\")[0].split(",")
    d = pd.concat([parts, d.iloc[:, 1:]], axis=1)
    d = d[(d.indic_se == "ERN") & (d.sex == "T") & (d.unit == "EUR") & (d.nace_r2 == "B-S_X_O")
          & (d.sizeclas == "GE10")]
    val = pd.to_numeric(d.iloc[:, -1].astype(str).str.extract(r"([\d.]+)")[0], errors="coerce")
    return pd.Series(val.values, index=pd.MultiIndex.from_arrays([d.geo, d.isced11]))


def rel_earnings_ses(country: str) -> pd.Series | None:
    """Relative hourly earnings by attainment from SES 2018 (B-S excl. O, enterprises 10+)."""
    g = ISO2.get(country)
    s = _ses_edu()
    if g is None or g not in s.index.get_level_values(0):
        return None
    s = s[g]
    lv = {"ISCED_L": "ED0-2", "ISCED_M": "ED3_4", "ISCED_H": "ED5-8"}
    if not all(v in s.index and np.isfinite(s[v]) for v in lv.values()):
        return None
    return pd.Series({k: s[v] / s["ED3_4"] for k, v in lv.items()})


def rel_earnings(country: str, year: int) -> tuple[pd.Series, str] | tuple[None, None]:
    d = _eag()
    d = d[d.REF_AREA == country]
    for wt in ("FT", "_T"):
        g = d[(d.WORK_TIME_ARNGMNT == wt) & d.ATTAINMENT_LEV.isin(EAG_LEV.values())]
        g = g.groupby("TIME_PERIOD").filter(lambda t: t.ATTAINMENT_LEV.nunique() == 3)
        if not g.empty:
            y = nearest(sorted(g.TIME_PERIOD.unique()), year)
            s = g[g.TIME_PERIOD == y].groupby("ATTAINMENT_LEV").OBS_VALUE.mean()
            return pd.Series({k: s[v] / 100 for k, v in EAG_LEV.items()}), f"{wt}:{y}"
    s = rel_earnings_ses(country)                   # A-V3-RE: FRA is not in EAG earnings data
    if s is not None:
        return s, "SES2018"
    return None, None


def mincer_beta(re: pd.Series) -> float:
    s = np.array([S_YEARS[k] for k in re.index])
    return float(np.polyfit(s, np.log(re.values), 1)[0])


# ---------------------------------------------------------------- Hilferding inputs
@lru_cache(None)
def _enrol():
    d = pd.read_csv(RAW / "uoe_stud_totals.csv", low_memory=False)
    d = d[(d.MEASURE == "ENRL") & (d.SEX == "_T") & (d.MOBILITY == "_T")
          & d.EDUCATION_LEV.isin([f"ISCED11_{i}" for i in range(1, 9)])]
    return d.groupby(["REF_AREA", "TIME_PERIOD"]).OBS_VALUE.sum()


def students(country: str, year: int) -> tuple[float, int]:
    s = _enrol().loc[country]
    y = nearest(list(s.index), year)
    return float(s[y]), y


@lru_cache(None)
def _pop():
    d = json.load(open(RAW / "pop_wb.json"))[1]
    return {(r["countryiso3code"], int(r["date"])): r["value"] for r in d if r["value"]}


def population(country: str, year: int) -> float:
    p = _pop()
    y = nearest([yy for (c, yy) in p if c == country], year)
    return float(p[(country, y)])


# ---------------------------------------------------------------- variants
def w_edu_price(e, country, year, label_divs=None, source="eag"):
    """1.1a: sum_e s_je RE_e, RE national relative earnings by attainment (no industry effects).
    source='ses': SES 2018 relative earnings (sensitivity, EU countries)."""
    if source == "ses":
        re, src = rel_earnings_ses(country), "SES2018"
    else:
        re, src = rel_earnings(country, year)
    sh = oecd.education_shares(country, year)
    if re is None or sh.isna().all().all():
        return None, "no data"
    w_ind = (sh * re).sum(1).where(sh.notna().all(1))
    return _fig_to_labels(w_ind, e, label_divs), f"EAG {src}"


def _fig_to_labels(w_ind: pd.Series, e, label_divs):
    return normalise(_map_ind(w_ind, e, label_divs), e.hours)


def _map_ind(w_ind: pd.Series, e, label_divs):
    """FIGARO-industry values -> Economy labels (mean over members; missing -> mean)."""
    out = []
    for lab in e.labels:
        if label_divs is None:
            vals = [w_ind.get(m, np.nan) for m in lab.split("+")]
        else:
            d = label_divs[lab]
            vals = [w_ind[j] for j in w_ind.index if divisions(j) and divisions(j) & d]
        vals = [v for v in vals if np.isfinite(v)]
        out.append(np.mean(vals) if vals else np.nan)
    out = np.array(out, dtype=float)
    return np.where(np.isfinite(out), out, np.nanmean(out))


def w_occ_price(e, country, year, label_divs=None):
    """1.1b: sum_o s_so w_o / wbar, w_o national mean earnings of ISCO major group o."""
    sh = occ_shares(country, year)
    w, src = occ_wage(country, year)
    if sh is None or w is None:
        return None, "no data"
    w = w.fillna(np.nanmean(w.values))
    sec = sh @ w
    return normalise(to_labels(sec, e.labels, label_divs), e.hours), f"ILO {src}"


def prep_years_by_section(country, year, jz5=5.0):
    sh = occ_shares(country, year)
    if sh is None:
        return None
    return sh @ isco_prep_years(jz5)


def w_jobzone_time(e, country, year, label_divs=None, H=1600, T=40, jz5=5.0):
    """1.2a: 1 + P_j H / (T 1800)."""
    P = prep_years_by_section(country, year, jz5)
    if P is None:
        return None, "no data"
    w = 1 + to_labels(P, e.labels, label_divs) * H / (T * 1800)
    return normalise(w, e.hours), "ILO+O*NET"


def w_jobzone_price(e, country, year, label_divs=None, jz5=5.0):
    """1.2b: exp(beta P_j), beta = national return to a year of schooling from EAG."""
    P = prep_years_by_section(country, year, jz5)
    re, src = rel_earnings(country, year)
    if P is None or re is None:
        return None, "no data"
    b = mincer_beta(re)
    return normalise(np.exp(b * to_labels(P, e.labels, label_divs)), e.hours), f"beta={b:.3f} ({src})"


def hilferding_E(e, country, year, capital=True):
    """Vertically integrated hours of education (P85) per student-year (A-V3-HILF)."""
    from ..core import leontief_inverse
    M, _, _ = e.system(capital, False, "price", e.l())
    v = e.l() @ leontief_inverse(M)
    j = [i for i, lab in enumerate(e.labels) if "P85" in lab.split("+")]
    if not j:
        return np.nan, "no P85"
    j = j[0]
    fc = e.hh_dom[j] + e.meta.get("gov_dom", np.zeros(e.n))[j]
    n, ys = students(country, year)
    return float(v[j] * fc / n), f"students {ys}"


def w_hilferding(e, country, year, label_divs=None, H=1600, T=40, subsistence=False):
    """1.3a/b: 1 + Y_j (H + E [+ C]) / (T 1800); Y_j years of schooling beyond 9 (TiMBC shares);
    C = hours embodied in household consumption per head per year (at the economy's hours per unit of
    value added for the imported part)."""
    sh = oecd.education_shares(country, year)
    if sh.isna().all().all():
        return None, "no data"
    Y = (sh * pd.Series({k: v - 9.0 for k, v in S_YEARS.items()})).sum(1).where(sh.notna().all(1))
    E, src = hilferding_E(e, country, year)
    extra = H + E
    if subsistence:
        from ..core import leontief_inverse
        M, _, _ = e.system(True, False, "price", e.l())
        v = e.l() @ leontief_inverse(M)
        hpm = e.hours.sum() / e.va.sum()
        C = (v @ e.hh_dom + e.hh_imp.sum() * hpm) / population(country, year)
        extra += C
    Yl = _map_ind(Y, e, label_divs)
    w = 1 + Yl * extra / (T * 1800)
    return normalise(w, e.hours), f"E={E:.0f} h/student-year ({src})" + (f", C={extra - H - E:.0f}" if subsistence else "")


def wage_rel(e) -> np.ndarray:
    li = e.labour_income
    return np.divide(li / e.hours, li.sum() / e.hours.sum(), out=np.ones(e.n), where=e.hours > 0)


def fig_wage_rel(e) -> pd.Series:
    """Relative hourly labour income by FIGARO industry (merged labels spread to their members)."""
    w = wage_rel(e)
    d = {}
    for lab, v in zip(e.labels, w):
        for m in lab.split("+"):
            d[m] = v
    return pd.Series(d)
