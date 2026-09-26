"""OECD National Accounts (Tables 6 and 7), STAN 2025 and TiMBC loaders, mapped to FIGARO industries.

All mappings are by sets of 2-digit NACE/ISIC divisions (see codes.py). When an OECD series is
not available at the FIGARO level, the value of the smallest available parent aggregate (minus
children observed directly) is split among the unobserved FIGARO children using a "splitter"
(persons employed if available at a finer level, otherwise FIGARO gross output). Every such
imputation is counted and reported (see `Mapped.log`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .codes import divisions
from .figaro import INDUSTRIES

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "oecd"

ISO3_TO_2 = {"USA": "US", "DEU": "DE", "MEX": "MX", "FRA": "FR", "ITA": "IT", "ESP": "ES", "NLD": "NL",
             "AUT": "AT", "POL": "PL", "CZE": "CZ", "KOR": "KR", "JPN": "JP", "TUR": "TR", "GBR": "UK",
             "ARG": "AR", "AUS": "AU", "BEL": "BE", "BGR": "BG", "BRA": "BR", "CAN": "CA", "CHE": "CH",
             "CHN": "CN", "CYP": "CY", "DNK": "DK", "EST": "EE", "FIN": "FI", "GRC": "EL", "HRV": "HR",
             "HUN": "HU", "IDN": "ID", "IND": "IN", "IRL": "IE", "LTU": "LT", "LUX": "LU", "LVA": "LV",
             "MLT": "MT", "NOR": "NO", "PRT": "PT", "ROU": "RO", "RUS": "RU", "SAU": "SA", "SVK": "SK",
             "SVN": "SI", "SWE": "SE"}
ISO2_TO_3 = {v: k for k, v in ISO3_TO_2.items()}

FIG_DIV = {c: divisions(c) for c in INDUSTRIES}


@lru_cache(maxsize=None)
def _read(name: str) -> pd.DataFrame:
    p = RAW / name
    if not p.exists():
        return pd.DataFrame()
    d = pd.read_csv(p, low_memory=False)
    d["value"] = d["OBS_VALUE"] * 10.0 ** d["UNIT_MULT"].fillna(0)
    return d


def series(country: str, table: str, transaction: str, unit: str, price_base: str | None = None,
           measure_col: str = "TRANSACTION") -> pd.DataFrame:
    """Wide frame: index = year, columns = OECD activity code."""
    d = _read(f"{table}_{country}.csv")
    if d.empty:
        return pd.DataFrame()
    m = (d[measure_col] == transaction) & (d["UNIT_MEASURE"] == unit)
    if price_base is not None:
        m &= d["PRICE_BASE"] == price_base
    if "SECTOR" in d:
        m &= d["SECTOR"].isin(["S1", "_Z"])
    x = d[m].pivot_table(index="TIME_PERIOD", columns="ACTIVITY", values="value", aggfunc="first")
    return x


@dataclass
class Mapped:
    values: pd.Series
    log: dict = field(default_factory=dict)


def to_figaro(row: pd.Series, splitter: pd.Series | None, mode: str = "flow") -> Mapped:
    """Map one year's OECD activity values to the 64 FIGARO industries.

    mode 'flow'  : additive quantities (hours, persons, money). Unobserved children share the parent
                   residual in proportion to `splitter` (indexed by FIGARO industry).
    mode 'index' : intensive quantities (price indices, ratios): unobserved children take the value
                   of their smallest observed parent.
    """
    row = row.dropna()
    avail = {c: divisions(c) for c in row.index}
    avail = {c: s for c, s in avail.items() if s}
    by_set: dict[frozenset, str] = {}
    for c, s in avail.items():  # prefer the shortest code for identical sets
        if s not in by_set or len(c) < len(by_set[s]):
            by_set[s] = c
    out = pd.Series(np.nan, index=INDUSTRIES)
    exact = 0
    for j, s in FIG_DIV.items():
        if s in by_set:
            out[j] = row[by_set[s]]
            exact += 1
    parents = sorted(by_set.items(), key=lambda kv: len(kv[0]))
    imputed = 0
    for s_par, code in parents:
        kids = [j for j, s in FIG_DIV.items() if s <= s_par]
        if not kids or set().union(*[FIG_DIV[j] for j in kids]) != s_par:
            continue  # parent is not a union of FIGARO industries
        todo = [j for j in kids if np.isnan(out[j])]
        if not todo:
            continue
        if mode == "index":
            out[todo] = row[code]
        else:
            resid = row[code] - out[[j for j in kids if j not in todo]].sum()
            resid = max(resid, 0.0)
            w = splitter.reindex(todo).fillna(0.0) if splitter is not None else pd.Series(1.0, index=todo)
            if w.sum() <= 0:
                w = pd.Series(1.0, index=todo)
            out[todo] = resid * w / w.sum()
        imputed += len(todo)
    return Mapped(out, dict(exact=exact, imputed=imputed, missing=int(out.isna().sum())))


def _row(df: pd.DataFrame, year: int) -> pd.Series:
    if df.empty or year not in df.index:
        return pd.Series(dtype=float)
    return df.loc[year]


def _section_of(code: str) -> str:
    return code[0]


def _map_with_fallback(row: pd.Series, splitter, mode="flow") -> pd.Series:
    return to_figaro(row, splitter, mode).values


def labour_block(country: str, year: int, x_fig: pd.Series) -> dict:
    """Hours (all persons), persons, employee share, CFC/output ratio and output deflator by FIGARO industry.

    persons : first available of Table 7 persons, Table 7 jobs, STAN persons.
    hours   : persons x hours-per-person, where hours per person is taken at the finest OECD activity
              where both hours and persons exist and the ratio is plausible (900-3300 h/yr), else the
              ISIC section, else the national average; total rescaled to national total hours.
    """
    log = {}
    xpos = x_fig > 0
    cands = []
    for tab, tr, un, mc in [("table7", "EMP", "PS", "TRANSACTION"), ("table7", "EMP", "JB", "TRANSACTION"),
                            ("stan2025", "EMP", "PS", "MEASURE")]:
        r = _row(series(country, tab, tr, un, measure_col=mc), year)
        if len(r.dropna()) < 5:
            continue
        m = to_figaro(r.replace(0, np.nan), x_fig)
        bad = int(((m.values.fillna(0) <= 0) & xpos).sum())
        cands.append((bad, len(cands), f"{tab} {tr} {un}", m))
    if not cands:
        raise ValueError(f"no employment data for {country} {year}")
    # Prefer official Table 7 (first candidate); STAN detail contains OECD estimates that can be
    # implausible (e.g. USA K65 insurance 0.13m persons). Zero industries are merged (economy.MERGES).
    bad, _, src, persons = cands[0]
    log["persons_source"], log["persons_zero_with_output"] = src, bad
    persons_v = persons.values.fillna(0.0)
    # hours per person ----------------------------------------------------------------------------
    hpp_codes = {}
    for tab, trh, trp, unp, mc in [("table7", "EMP", "EMP", "PS", "TRANSACTION"),
                                   ("table7", "SAL", "SAL", "PS", "TRANSACTION"),
                                   ("table7", "SAL", "SAL", "JB", "TRANSACTION"),
                                   ("stan2025", "EMP", "EMP", "PS", "MEASURE")]:
        h = _row(series(country, tab, trh, "H", measure_col=mc), year)
        pp = _row(series(country, tab, trp, unp, measure_col=mc), year)
        if h.empty or pp.empty:
            continue
        ratio = (h / pp).replace([np.inf, -np.inf], np.nan).dropna()
        ratio = ratio[(ratio > 900) & (ratio < 3300)]
        for c, v in ratio.items():
            hpp_codes.setdefault(c, v)
    tot_h = _row(series(country, "table7", "EMP", "H"), year).get("_T", np.nan)
    if not np.isfinite(tot_h):
        tot_h = _row(series(country, "stan2025", "EMP", "H", measure_col="MEASURE"), year).get("_T", np.nan)
    nat = tot_h / persons_v.sum() if np.isfinite(tot_h) else hpp_codes.get("_T", 1800.0)
    hpp_codes.pop("_T", None)
    hpp = to_figaro(pd.Series(hpp_codes, dtype=float), None, mode="index").values if hpp_codes else \
        pd.Series(np.nan, index=INDUSTRIES)
    n_fallback = int(hpp.isna().sum())
    hpp = hpp.fillna(nat)
    hours = persons_v * hpp
    if np.isfinite(tot_h) and hours.sum() > 0:
        log["hours_rescale"] = float(tot_h / hours.sum())
        hours = hours * tot_h / hours.sum()
    log["hpp_national_fallback"] = n_fallback
    # employee share --------------------------------------------------------------------------------
    sal = None
    for tab, un, mc in [("table7", "PS", "TRANSACTION"), ("table7", "JB", "TRANSACTION"), ("stan2025", "PS", "MEASURE")]:
        r = _row(series(country, tab, "SAL", un, measure_col=mc), year)
        if len(r.dropna()) >= 5:
            e = _row(series(country, tab, "EMP", un, measure_col=mc), year)
            ratio = (r / e).replace([np.inf, -np.inf], np.nan).dropna().clip(0, 1)
            sal = to_figaro(ratio, None, mode="index").values
            log["employee_share_source"] = f"{tab} {un}"
            break
    emp_share = sal.fillna(sal.median()) if sal is not None else pd.Series(1.0, index=INDUSTRIES)
    # capital consumption as share of output ------------------------------------------------------
    p1 = series(country, "table6", "P1", "XDC", "V")
    cfc = series(country, "table6", "P51C", "XDC", "V")
    if cfc.empty or year not in cfc.index:
        cfc = series(country, "table6", "B2A3G", "XDC", "V") - series(country, "table6", "B2A3N", "XDC", "V")
        log["cfc_source"] = "B2A3G - B2A3N"
    else:
        log["cfc_source"] = "P51C"
    ratio = (_row(cfc, year) / _row(p1, year)).replace([np.inf, -np.inf], np.nan).dropna()
    cfc_ratio = to_figaro(ratio, None, mode="index").values
    log["cfc_missing"] = int(cfc_ratio.isna().sum())
    # output price index -------------------------------------------------------------------------
    defl = series(country, "table6", "P1", "IX", "DR")
    price = to_figaro(_row(defl, year), None, mode="index")
    log["price"] = price.log
    return dict(hours=hours, persons=persons_v, employee_share=emp_share,
                cfc_ratio=cfc_ratio, price_index=price.values, log=log)


# --- TiMBC education structure --------------------------------------------------------------------
EDU = ["ISCED_L", "ISCED_M", "ISCED_H"]


def education_shares(country3: str, year: int) -> pd.DataFrame:
    """Shares of low/medium/high education in employment for each FIGARO industry (from the
    smallest TiMBC aggregate containing it). Years outside 2008-2022 use the nearest year."""
    d = _read("timbc_2025_empn.csv")
    d = d[(d.REF_AREA == country3) & d.CHARACTERISTIC.isin(EDU + ["_T"])]
    if d.empty:
        return pd.DataFrame(np.nan, index=INDUSTRIES, columns=EDU)
    yrs = sorted(d.TIME_PERIOD.unique())
    y = min(yrs, key=lambda t: abs(t - year))
    w = d[d.TIME_PERIOD == y].pivot_table(index="ACTIVITY", columns="CHARACTERISTIC", values="OBS_VALUE")
    sh = w[EDU].div(w[EDU].sum(1), axis=0)
    aggs = {a: divisions(a) for a in sh.index}
    out = pd.DataFrame(np.nan, index=INDUSTRIES, columns=EDU)
    for j, s in FIG_DIV.items():
        cands = [(len(v), a) for a, v in aggs.items() if v and s <= v]
        if cands:
            out.loc[j] = sh.loc[min(cands)[1]].values
    return out


def employment_persons_timbc(country3: str, year: int, x_fig: pd.Series) -> pd.Series:
    """Persons employed by FIGARO industry from TiMBC aggregates (used for foreign countries in the
    multi-regional variant). Split within aggregates by FIGARO gross output."""
    d = _read("timbc_2025_empn.csv")
    d = d[(d.REF_AREA == country3) & (d.CHARACTERISTIC == "_T")]
    if d.empty:
        return pd.Series(np.nan, index=INDUSTRIES)
    yrs = sorted(d.TIME_PERIOD.unique())
    y = min(yrs, key=lambda t: abs(t - year))
    row = d[d.TIME_PERIOD == y].set_index("ACTIVITY")["value"]
    return to_figaro(row, x_fig).values
