"""Iteration 5, stage 1-2 data: country-year aggregates of the market sector (EU KLEMS 2023 +
Eurostat/OECD CFC shares), 1995-2021.

Industry rows (finest non-overlapping KLEMS industries, fallbacks to aggregates as in iteration 4) carry
VA, VA_PYP, GO, II, COMP, H, HE, K (K_GFCF), K_PYP (Kpyp_GFCF), ICT capital (K_IT + K_CT + K_Soft_DB),
intangible parts (K_Soft_DB, K_RD, K_OIPP), non-NA intangibles (K_NonNatAcc, I_NonNatAcc, VAadj),
CFC and D29X39 shares of B1G.  Missing CFC shares: nearest year of the same industry, else the share of
the industry's NACE section (A-V5-CFC).

Aggregates (sums over the chosen industries) for variants of the sector definition:
  main      : all industries except L (incl. L68A imputed rent), O, P, Q, T, U
  all       : every industry (with real estate and government)
  no_fin    : main without K
  no_mining : main without B
  no_mi     : main without the mixed-income correction (W = COMP)
  tangible  : main, K without Soft_DB, R&D, other IPP
  intangibles : main, K + non-national-accounts intangibles, profit adjusted as in iteration 4
Chain-linked volumes: Q_t/Q_{t-1} = sum X_PYP,t / sum X_CP,t-1 over industries present in both years.
Output: results/v5/agg_industry.csv.gz, results/v5/agg_country_year.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..v4.b_data import ALT, COUNTRIES, FINE, KL, divs, na_items, ratio_for

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "v5"
YEARS = range(1995, 2022)
EXCL_MAIN = {"L", "L68A", "O", "P", "Q", "Q86", "Q87-Q88", "O-Q", "T", "U"}
RENT_REG = {"B", "D", "E", "D-E", "K", "L", "L68A", "O", "P", "Q", "Q86", "Q87-Q88", "O-Q", "T", "U", "R", "S", "R-S"}
MAIN = lambda i: i not in EXCL_MAIN


def _no_mi(s):
    return s.assign(PI=s.PI_nomi, W=s.COMP)


def _tangible(s):
    f = 1 - s.K_intNA / s.K
    return s.assign(K=s.K * f, K_PYP=s.K_PYP * f)   # A-V5-TANG: tangible K_PYP by the same price change


def _intang(s):
    s = s.dropna(subset=["K_nonNA", "PI_int"])
    return s.assign(K=s.K + s.K_nonNA, K_PYP=s.K_PYP * (s.K + s.K_nonNA) / s.K, PI=s.PI_int)


# name: (industry filter, transform of industry rows)
VARIANTS = {"main": (MAIN, None), "all": (lambda i: True, None),
            "no_fin": (lambda i: MAIN(i) and i != "K", None), "no_mining": (lambda i: MAIN(i) and i != "B", None),
            "no_mi": (MAIN, _no_mi), "tangible": (MAIN, _tangible), "intangibles": (MAIN, _intang)}


def load_klems():
    na = pd.read_csv(KL / "national_accounts.csv", low_memory=False)
    ca = pd.read_csv(KL / "capital_accounts.csv", low_memory=False,
                     usecols=["nace_r2_code", "geo_code", "year", "K_GFCF", "Kpyp_GFCF", "Kq_GFCF", "K_IT", "K_CT", "K_Soft_DB",
                              "K_RD", "K_OIPP", "K_Rstruc"])
    ia = pd.read_csv(KL / "intangibles_analytical.csv", low_memory=False,
                     usecols=["nace_r2_code", "geo_code", "year", "K_NonNatAcc", "I_NonNatAcc", "VAadj", "VA_CP"])
    ia = ia.rename(columns={"VA_CP": "VA_ia"})
    key = ["nace_r2_code", "geo_code", "year"]
    d = na.merge(ca, on=key, how="left").merge(ia, on=key, how="left")
    return d.rename(columns={"nace_r2_code": "ind", "geo_code": "geo"})


def industry_rows():
    kl = load_klems()
    nai = na_items()
    # iteration-4 na_items() turns the OECD code R_S_T into "R_S_-" (and _T into "_-"); undo it here
    nai["nace"] = nai.nace.replace({"R_S_-": "R_S_T", "_-": "_T"})
    rows = []
    req = ("VA_CP", "COMP", "H_EMP", "H_EMPE", "K_GFCF", "GO_CP")
    for geo in COUNTRIES:
        g = kl[(kl.geo == geo) & kl.year.isin(YEARS)]
        ng = nai[nai.geo == geo]
        for y, gy in g.groupby("year"):
            byind = gy.set_index("ind")
            ny = ng[ng.year == y]
            chosen = []
            ok = lambda i: i in byind.index and all(np.isfinite(byind.loc[i, v]) for v in req)
            for ind in FINE:
                if ok(ind):
                    chosen.append(ind)
                elif ind in ALT and ok(ALT[ind]) and ALT[ind] not in chosen:
                    chosen.append(ALT[ind])
            for ind in chosen:
                r = byind.loc[ind]
                rt = ratio_for(ny, divs(ind)) if not ny.empty else None
                rows.append(dict(geo=geo, year=y, ind=ind, VA=r.VA_CP, VA_PYP=r.VA_PYP, GO=r.GO_CP, II=r.II_CP,
                                 COMP=r.COMP, H=r.H_EMP, HE=r.H_EMPE, K=r.K_GFCF, K_PYP=r.Kpyp_GFCF, VA_Q=r.VA_Q, K_Q=r.Kq_GFCF,
                                 K_ICT=np.nansum([r.K_IT, r.K_CT, r.K_Soft_DB]),
                                 K_intNA=np.nansum([r.K_Soft_DB, r.K_RD, r.K_OIPP]),
                                 K_nonNA=r.K_NonNatAcc, I_nonNA=r.I_NonNatAcc, VAadj=r.VAadj, VA_ia=r.VA_ia,
                                 cfc_sh=rt[0] if rt else np.nan, tax_sh=rt[1] if rt else np.nan))
        print("industry rows", geo, flush=True)
    p = pd.DataFrame(rows)
    # A-V5-CFC: fill missing CFC shares -- nearest year of the same industry, else the NACE section share
    p = p.sort_values(["geo", "ind", "year"])
    p["cfc_src"] = np.where(p.cfc_sh.notna(), "own", "")
    for col in ("cfc_sh", "tax_sh"):
        p[col] = p.groupby(["geo", "ind"])[col].transform(lambda s: s.ffill().bfill())
    p.loc[(p.cfc_src == "") & p.cfc_sh.notna(), "cfc_src"] = "nearest_year"
    miss = p.cfc_sh.isna()
    if miss.any():
        for ix in p.index[miss]:
            r = p.loc[ix]
            sec = r.ind[0]
            ny = nai[(nai.geo == r.geo) & (nai.year == r.year)]
            rt = ratio_for(ny, divs(sec)) if not ny.empty else None
            if rt:
                p.loc[ix, ["cfc_sh", "tax_sh", "cfc_src"]] = [rt[0], rt[1], "section"]
                continue
            # smallest national-accounts aggregate containing the section (JP: D_E, R_S_T)
            if ny.empty:
                continue
            tgt = divs(sec)
            enc = [(len(divs(c)), c) for c in ny.nace.unique() if divs(c) and tgt <= divs(c)]
            for _, c in sorted(enc):
                rt = ratio_for(ny, divs(c))
                if rt:
                    p.loc[ix, ["cfc_sh", "tax_sh", "cfc_src"]] = [rt[0], rt[1], "enclosing"]
                    break
    p["tax_sh"] = p.tax_sh.fillna(0)
    # A-V5-PYP: where previous-year-price values are missing (US; PT capital has neither), rebuild them from
    # the industry volume index: X_PYP,t = X_CP,t-1 * Q_t / Q_t-1 (consecutive years only)
    g = p.groupby(["geo", "ind"])
    cons = g.year.diff() == 1
    for pyp, cp, q in (("VA_PYP", "VA", "VA_Q"), ("K_PYP", "K", "K_Q")):
        alt = g[cp].shift() * p[q] / g[q].shift()
        fill = p[pyp].isna() & cons & np.isfinite(alt)
        p.loc[fill, pyp] = alt[fill]
        p[pyp + "_src"] = np.where(fill, "volume_index", np.where(p[pyp].notna(), "klems_pyp", ""))
    p["W_se"] = np.where(p.HE > 0, p.COMP / p.HE * (p.H - p.HE).clip(lower=0), 0.0)
    p["W"] = p.COMP + p.W_se
    p["CFC"] = p.cfc_sh * p.VA
    p["TAX"] = p.tax_sh * p.VA
    p["PI"] = p.VA - p.W - p.CFC - p.TAX
    p["PI_nomi"] = p.VA - p.COMP - p.CFC - p.TAX
    dep = p.I_nonNA - p.groupby(["geo", "ind"]).K_nonNA.diff()
    p["PI_int"] = p.PI + (p.VAadj - p.VA_ia) - dep
    return p


def chain(d, cp, pyp):
    """Chain-linked volume index (first year = nominal level) for an aggregate over industries present
    in consecutive years with both the previous-year-price and the current-price value."""
    years = sorted(d.year.unique())
    q = {years[0]: d[d.year == years[0]][cp].sum()}
    for a, b in zip(years[:-1], years[1:]):
        da, db = d[d.year == a].set_index("ind"), d[d.year == b].set_index("ind")
        common = da.index.intersection(db.index)
        common = common[np.isfinite(db.loc[common, pyp].to_numpy(float)) & np.isfinite(da.loc[common, cp].to_numpy(float))
                        & (da.loc[common, cp].to_numpy(float) > 0)]
        num, den = db.loc[common, pyp].sum(), da.loc[common, cp].sum()
        q[b] = q[a] * num / den if den > 0 and np.isfinite(num) and num > 0 else np.nan
    return pd.Series(q)


COVER_MIN = 0.95  # A-V5-COV: country-years with CFC-covered VA share of the variant below this are dropped


def aggregates(p):
    rows = []
    fin_share = p[p.ind.isin(["K"])].groupby(["geo", "year"]).VA.sum() / p.groupby(["geo", "year"]).VA.sum()
    for geo, g in p.groupby("geo"):
        for vname, (keep, tf) in VARIANTS.items():
            s = g[g.ind.map(keep) & g.cfc_sh.notna()]
            if tf is not None:
                s = tf(s)
            if s.empty:
                continue
            qy, qk = chain(s, "VA", "VA_PYP"), chain(s, "K", "K_PYP")
            a = s.groupby("year")[["VA", "GO", "II", "COMP", "W_se", "W", "H", "K", "K_ICT", "K_intNA", "K_nonNA",
                                   "CFC", "TAX", "PI", "PI_nomi", "PI_int"]].sum(min_count=1)
            a["n_ind"] = s.groupby("year").ind.nunique()
            a["cover"] = a.VA / g[g.ind.map(keep)].groupby("year").VA.sum()
            a["Q_Y"], a["Q_K"] = qy, qk
            a["geo"], a["variant"] = geo, vname
            a["fin_share"] = fin_share.loc[geo] if geo in fin_share.index.get_level_values(0) else np.nan
            rows.append(a.reset_index())
    d = pd.concat(rows, ignore_index=True)
    # A-V5-COV: drop incomplete coverage and degenerate capital (BG has K only for 2016, K = 0)
    d = d[(d.cover >= COVER_MIN) & (d.K > 0) & (d.W > 0)].copy()
    d["P_Y"], d["P_K"] = d.VA / d.Q_Y, d.K / d.Q_K
    d["r"], d["e"], d["k"] = d.PI / d.K, d.PI / d.W, d.K / d.W
    d["rM"] = d.PI / (d.K + d.W)
    d["share_PI"] = d.PI / d.VA
    d["pcm"] = d.PI / d.GO
    d["mu"] = d.GO / (d.II + d.W + d.CFC) - 1
    d["ict_share"] = d.K_ICT / d.K
    d["w_hour"] = d.W / d.H
    return d


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    p = industry_rows()
    p.to_csv(OUT / "agg_industry.csv.gz", index=False)
    d = aggregates(p)
    d.to_csv(OUT / "agg_country_year.csv", index=False)
    print(p.cfc_src.value_counts())
    m = d[d.variant == "main"]
    print(m.groupby("geo").agg(y0=("year", "min"), y1=("year", "max"), r_mean=("r", "mean"), e=("e", "mean"), k=("k", "mean")).round(3).to_string())
