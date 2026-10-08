"""Iteration 10, stage 9: long business-sector capital series (pre_registration_v10.md, stage 9; journal 10).

9.1 sources table; 9.2 splices (BSDB -> EU KLEMS 2009 -> EU KLEMS 2025); 9.3 business-sector r, trends with
different ends, phases, profit squeeze, PWT variant (outcome 58); 9.4 Marxian r_m on KLEMS industries (outcome 59).

Usage: PYTHONPATH=src python -P -m lts.v10.stage9
Outputs: results/v10/s9_*.csv
"""
from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd
import statsmodels.api as sm

from ..v6.series import ROOT
from ..v8.rule import share_row
from ..v8.stage4 import EU_CRISES, phases_from
from ..v9b.stage1 import trend

OUT = ROOT / "results" / "v10"
R10 = ROOT / "data" / "raw" / "v10"
R9B = ROOT / "data" / "raw" / "v9b"
ISO3 = {"AT": "AUT", "BE": "BEL", "BG": "BGR", "CY": "CYP", "CZ": "CZE", "DE": "DEU", "DK": "DNK", "EE": "EST",
        "EL": "GRC", "ES": "ESP", "FI": "FIN", "FR": "FRA", "HR": "HRV", "HU": "HUN", "IE": "IRL", "IT": "ITA",
        "JP": "JPN", "LT": "LTU", "LU": "LUX", "LV": "LVA", "MT": "MLT", "NL": "NLD", "PL": "POL", "PT": "PRT",
        "RO": "ROU", "SE": "SWE", "SI": "SVN", "SK": "SVK", "UK": "GBR", "US": "USA"}
K09_ISO = {"GER": "DEU", "UK": "GBR", "USA-NAICS": "USA"}
BSDB_ISO = {"IRE": "IRL"}
OECD38 = ["AUS", "AUT", "BEL", "CAN", "CHL", "COL", "CRI", "CZE", "DNK", "EST", "FIN", "FRA", "DEU", "GRC", "HUN", "ISL",
          "IRL", "ISR", "ITA", "JPN", "KOR", "LVA", "LTU", "LUX", "MEX", "NLD", "NZL", "NOR", "POL", "PRT", "SVK", "SVN",
          "ESP", "SWE", "CHE", "TUR", "GBR", "USA"]


# ================================================================ sources
def bsdb():
    b = pd.read_stata(R10 / "oecd_bsdb.dta")
    b["geo"] = b.cty.replace(BSDB_ISO)
    a = b.groupby(["geo", "year"])[["gdpb", "wsse", "eep", "kbv", "pib", "unr"]].mean().reset_index()
    a["year"] = a.year.astype(int)
    a["comp"] = a.wsse * a.eep / 1e6
    a["K"] = a.kbv * a.pib / 100
    a["r"] = (a.gdpb - a.comp) / a.K
    a["e"] = (a.gdpb - a.comp) / a.comp
    return a.dropna(subset=["r"])[["geo", "year", "r", "e", "gdpb", "comp", "K"]]


def k09_long(name):
    raw = zipfile.ZipFile(R10 / f"euklems09_all_{name}_09I.zip").read(f"all_{name}_09I.txt").decode("latin1")
    d = pd.read_csv(io.StringIO(raw.replace('"', "")), dtype=str)
    d = d.melt(id_vars=["country", "var", "code"], var_name="year", value_name="v")
    d["year"] = d.year.str.lstrip("_").astype(int)
    d["v"] = pd.to_numeric(d.v, errors="coerce")
    d["geo"] = d.country.replace(K09_ISO)
    return d


def klems09():
    b = k09_long("countries")
    c = k09_long("capital")
    w = b[b["var"].isin(["VA", "COMP", "LAB"])].pivot_table(index=["geo", "code", "year"], columns="var", values="v")
    k = c[c["var"].isin(["K_GFCF", "Ip_GFCF"])].pivot_table(index=["geo", "code", "year"], columns="var", values="v")
    k["Knom"] = k.K_GFCF * k.Ip_GFCF / 100
    return w.join(k[["Knom"]], how="left").reset_index()


def klems25():
    na = pd.read_csv(R9B / "klems2025_national_accounts.csv", usecols=["nace_r2_code", "geo_code", "year", "VA_CP", "COMP"],
                     low_memory=False)
    ca = pd.read_csv(R9B / "klems2025_capital_accounts.csv", usecols=["nace_r2_code", "geo_code", "year", "K_GFCF"],
                     low_memory=False)
    m = na.merge(ca, on=["nace_r2_code", "geo_code", "year"], how="left")
    m = m[m.geo_code.isin(ISO3)].copy()
    m["geo"] = m.geo_code.map(ISO3)
    return m.rename(columns={"nace_r2_code": "code", "VA_CP": "VA", "K_GFCF": "Knom"})


# ================================================================ 9.3 business-sector r
NONMARKET09 = ["70", "L", "M", "N", "P", "Q"]


def market09(K09):
    rows = []
    for (geo, y), g in K09.groupby(["geo", "year"]):
        g = g.set_index("code")
        if "TOT" not in g.index:
            continue
        t = g.loc["TOT"]
        ex = g.reindex(NONMARKET09)[["VA", "COMP", "Knom"]].sum(min_count=1)
        need = g.reindex(["70", "L", "M", "N"])
        if need[["VA", "COMP", "Knom"]].isna().any().any():
            continue
        va, comp, k = t.VA - ex.VA, t.COMP - ex.COMP, t.Knom - ex.Knom
        if not np.isfinite([va, comp, k]).all() or k <= 0:
            continue
        rows.append(dict(geo=geo, year=int(y), r=(va - comp) / k, e=(va - comp) / comp))
    return pd.DataFrame(rows)


def market25(K25):
    m = K25[K25.code == "MARKT"].dropna(subset=["VA", "COMP", "Knom"])
    m = m[m.Knom > 0]
    return pd.DataFrame(dict(geo=m.geo, year=m.year.astype(int), r=(m.VA - m.COMP) / m.Knom, e=(m.VA - m.COMP) / m.COMP))


def splice_r(parts, years_1995=(1995,), y70=1970):
    """parts: dict source -> Series (year -> r). Returns spliced series and splice factors."""
    out, fac = None, {}
    s25, s09, sb = parts.get("K25"), parts.get("K09"), parts.get("BSDB")
    cur = s25
    if s09 is not None and cur is not None:
        common = [y for y in years_1995 if y in s09.index and y in cur.index]
        if common:
            f = float(np.mean([cur[y] / s09[y] for y in common]))
            fac["K09->K25"] = f
            cur = pd.concat([s09[s09.index < min(cur.index)] * f, cur])
    elif s09 is not None and cur is None:
        cur = s09
    if sb is not None:
        if cur is None:
            cur = sb
        else:
            anchor = y70 if (s09 is not None and y70 in cur.index and y70 in sb.index) else \
                (1995 if (1995 in cur.index and 1995 in sb.index) else None)
            if anchor is None:
                common = sorted(set(cur.index) & set(sb.index))
                anchor = common[0] if common else None
            if anchor is not None:
                f = float(cur[anchor] / sb[anchor])
                fac["BSDB->"] = f
                fac["BSDB anchor"] = anchor
                cur = pd.concat([sb[sb.index < min(cur.index)] * f, cur])
    out = cur.sort_index() if cur is not None else None
    return out, fac


def ends_variants(s):
    s = s.dropna()
    a, b = s.index.min(), s.index.max()
    v = {"full": s, "start+5": s.loc[a + 5:], "end-5": s.loc[:b - 5], "both": s.loc[a + 5:b - 5]}
    return {k: trend(x) for k, x in v.items() if len(x) >= 15}


def run_r(B, M09, M25, variant="main"):
    yrs95 = (1995,) if variant == "main" else tuple(range(1995, 2001))
    geos = sorted(set(B.geo) | set(M09.geo) | set(M25.geo))
    series, facs, T = {}, [], []
    for g in geos:
        parts = {}
        for name, D in (("BSDB", B), ("K09", M09), ("K25", M25)):
            x = D[D.geo == g].set_index("year").r.sort_index()
            if len(x):
                parts[name] = x
        s, fac = splice_r(parts, yrs95)
        if s is None:
            continue
        s = s[~s.index.duplicated()]
        series[g] = s
        facs.append(dict(geo=g, variant=variant, sources="+".join(parts), first=int(s.index.min()), last=int(s.index.max()),
                         n=len(s), **fac))
        if len(s) >= 30:
            ev = ends_variants(s)
            rob = all(v["label"] == "подтверждено" for v in ev.values())
            T.append(dict(geo=g, variant=variant, n=len(s), years=f"{s.index.min()}-{s.index.max()}", est=ev["full"]["est"],
                          ci90_lo=ev["full"]["ci90_lo"], ci90_hi=ev["full"]["ci90_hi"], label=ev["full"]["label"],
                          labels_ends=";".join(f"{k}:{v['label'][:4]}" for k, v in ev.items()), robust_ends=rob))
    return series, pd.DataFrame(facs), pd.DataFrame(T)


def phases_r(series):
    rows = []
    for g, s in series.items():
        if len(s) < 30:
            continue
        for a, b in phases_from(list(s.index), EU_CRISES, int(s.index.max())):
            x = s.loc[a:b]
            if len(x) < 3:
                continue
            rows.append(dict(geo=g, start=a, end=b, slope=np.polyfit(x.index, x.values, 1)[0]))
    P = pd.DataFrame(rows)
    P["falls"] = P.slope < 0
    return P


def profit_squeeze(B, M09, M25):
    from ..v6 import series as S6
    S6.CODES = list(dict.fromkeys(S6.CODES + ["ZUTN"]))
    raw = S6.ameco_raw()
    u = raw[["geo", "year", "ZUTN"]].dropna()
    rows = []
    for src, D in (("BSDB", B), ("K09", M09), ("K25", M25)):
        x = D[["geo", "year", "e"]].copy()
        x["src"] = src
        rows.append(x)
    d = pd.concat(rows).merge(u, on=["geo", "year"], how="left").sort_values(["src", "geo", "year"])
    d = d[d.e > 0]
    g = d.groupby(["src", "geo"])
    d["dlne"] = g.e.transform(lambda v: np.log(v).diff())
    d["u1"] = g.ZUTN.shift()
    d = d.dropna(subset=["dlne", "u1"])
    d["unit"] = d.src + "_" + d.geo
    X = d[["dlne", "u1"]] - d.groupby("unit")[["dlne", "u1"]].transform("mean")
    f = sm.OLS(X.dlne, X[["u1"]]).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.geo)[0]})
    return pd.DataFrame([dict(item="d ln e on unemployment(t-1), unit = country x source FE, clusters = countries",
                              coef=f.params.u1, se=f.bse.u1, ci90_lo=f.params.u1 - 1.645 * f.bse.u1,
                              ci90_hi=f.params.u1 + 1.645 * f.bse.u1, n=len(d), countries=d.geo.nunique())])


def pwt_r():
    p = pd.read_excel(ROOT / "data" / "raw" / "v9" / "pwt110.xlsx", sheet_name="Data")
    p = p[p.countrycode.isin(OECD38)]
    p["r"] = (1 - p.labsh) * p.cgdpo / p.cn
    T = []
    for g, x in p.groupby("countrycode"):
        s = x.set_index("year").r.dropna()
        if len(s) < 30:
            continue
        t = trend(s)
        T.append(dict(geo=g, n=len(s), years=f"{s.index.min()}-{s.index.max()}", est=t["est"], label=t["label"]))
    return pd.DataFrame(T)


# ================================================================ 9.4 Marxian r_m
UNP09 = ["G", "J", "70", "71", "74", "L"]
UNP09_FB = ["G", "J", "K", "L"]
UNP25 = ["G", "K", "L", "N", "O"]


def rm_panel(D, unp, unp_fb=None):
    rows = []
    for (geo, y), g in D.groupby(["geo", "year"]):
        g = g.set_index("code")
        if "TOT" not in g.index:
            continue
        use = unp
        if not set(use) <= set(g.index) or g.loc[use, ["COMP", "Knom"]].isna().any().any():
            if unp_fb is None or not set(unp_fb) <= set(g.index) or g.loc[unp_fb, ["COMP", "Knom"]].isna().any().any():
                continue
            use = unp_fb
        t = g.loc["TOT"]
        if not np.isfinite([t.VA, t.COMP, t.Knom]).all():
            continue
        v = t.COMP - g.loc[use, "COMP"].sum()
        C = t.Knom - g.loc[use, "Knom"].sum()
        rows.append(dict(geo=geo, year=int(y), r_m=(t.VA - v) / (C + v), unp="+".join(use)))
    return pd.DataFrame(rows)


def run_rm(K09, K25):
    a = rm_panel(K09, UNP09, UNP09_FB)
    b = rm_panel(K25, UNP25)
    T, F = [], []
    for g in sorted(set(a.geo) | set(b.geo)):
        parts = {}
        x = a[a.geo == g].set_index("year").r_m.sort_index()
        if len(x):
            parts["K09"] = x
        x = b[b.geo == g].set_index("year").r_m.sort_index()
        if len(x):
            parts["K25"] = x
        s, fac = splice_r(parts)
        if s is None:
            continue
        s = s[~s.index.duplicated()]
        F.append(dict(geo=g, sources="+".join(parts), first=int(s.index.min()), last=int(s.index.max()), n=len(s), **fac))
        if len(s) >= 30:
            ev = ends_variants(s)
            T.append(dict(geo=g, n=len(s), years=f"{s.index.min()}-{s.index.max()}", est=ev["full"]["est"],
                          ci90_lo=ev["full"]["ci90_lo"], ci90_hi=ev["full"]["ci90_hi"], label=ev["full"]["label"],
                          labels_ends=";".join(f"{k}:{v['label'][:4]}" for k, v in ev.items()),
                          robust_ends=all(v["label"] == "подтверждено" for v in ev.values())))
    return pd.DataFrame(T), pd.DataFrame(F), pd.concat([a.assign(src="K09"), b.assign(src="K25")])


def sources_table():
    return pd.DataFrame([
        dict(source="OECD BSDB", years="1960-1997 (quarterly)", sector="business sector", stock="KBV: business capital stock, volume; x PIB",
             gross_net="gross (OECD BSDB definition; documentation not available)", valuation="replacement (deflated by PIB)"),
        dict(source="OECD ISDB", years="1960-1997", sector="industries", stock="KTVO (suffix meanings undocumented)",
             gross_net="not used in the main series", valuation="-"),
        dict(source="EU KLEMS 2009", years="1970-2007", sector="industries (NACE 1); market = TOT - 70, L-Q",
             stock="K_GFCF (real, 1995 prices) x Ip_GFCF", gross_net="net (PIM, geometric)", valuation="current replacement"),
        dict(source="EU KLEMS 2025", years="1995-2021", sector="MARKT; industries (NACE 2)", stock="K_GFCF",
             gross_net="net", valuation="current replacement"),
        dict(source="PWT 11", years="1950-2023", sector="total economy", stock="cn (current PPP)", gross_net="net (PIM)",
             valuation="current PPP"),
    ])


def run():
    sources_table().to_csv(OUT / "s9_sources.csv", index=False)
    B = bsdb()
    K09 = klems09()
    K25 = klems25()
    M09, M25 = market09(K09), market25(K25)
    allT, allF = [], []
    for variant in ("main", "mean ratio 1995-2000"):
        series, F, T = run_r(B, M09, M25, variant)
        allT.append(T)
        allF.append(F)
        if variant == "main":
            main_series = series
    T = pd.concat(allT, ignore_index=True)
    F = pd.concat(allF, ignore_index=True)
    T.to_csv(OUT / "s9_r_trends.csv", index=False)
    F.to_csv(OUT / "s9_r_splices.csv", index=False)
    pd.concat({g: s for g, s in main_series.items()}, names=["geo", "year"]).rename("r").to_csv(OUT / "s9_r_series.csv")
    Tm = T[T.variant == "main"]
    out = [share_row("58: share of countries with a 'confirmed' falling business-sector r (>= 30 years)",
                     Tm.label == "подтверждено", 0.5, 0.15, ">", robust_ends=int(Tm.robust_ends.sum()))]
    Tv = T[T.variant != "main"]
    out.append(share_row("58 (variant: splice on mean ratio 1995-2000)", Tv.label == "подтверждено", 0.5, 0.15, ">"))
    P = phases_r(main_series)
    P.to_csv(OUT / "s9_phases.csv", index=False)
    Pw = pwt_r()
    Pw.to_csv(OUT / "s9_pwt_trends.csv", index=False)
    out.append(share_row("PWT 11 total economy (descriptive): share with 'confirmed' falling r", Pw.label == "подтверждено",
                         0.5, 0.15, ">"))
    PS = profit_squeeze(B, M09, M25)
    PS.to_csv(OUT / "s9_profit_squeeze.csv", index=False)
    Tr, Fr, Pr = run_rm(K09, K25)
    Tr.to_csv(OUT / "s9_rm_trends.csv", index=False)
    Fr.to_csv(OUT / "s9_rm_splices.csv", index=False)
    Pr.to_csv(OUT / "s9_rm_panel.csv", index=False)
    out.append(share_row("59: share of countries with a 'confirmed' falling Marxian r_m (KLEMS 2009 + 2025, >= 30 years)",
                         Tr.label == "подтверждено", 0.5, 0.15, ">", robust_ends=int(Tr.robust_ends.sum())))
    O = pd.DataFrame(out)
    l58, l59 = O.label.iloc[0], O.label.iloc[-1]
    rob = (Tm.robust_ends.mean() > 0.5) and (Tr.robust_ends.mean() > 0.5)
    gen = "подтверждено" if (l58 == "подтверждено" and l59 == "подтверждено" and rob) else \
        ("частично" if "подтверждено" in (l58, l59) else "нет")
    O = pd.concat([O, pd.DataFrame([dict(outcome="generalisation of the law of the tendency over the long period (9b rule)",
                                         label=gen)])], ignore_index=True)
    O.to_csv(OUT / "s9_outcomes.csv", index=False)
    pd.set_option("display.width", 250)
    print(F[F.variant == "main"].round(3).to_string(), "\n", Tm.round(5).to_string(), "\n", Tr.round(5).to_string(), "\n",
          P.groupby("geo").falls.mean().round(2).to_dict(), "\n", PS.round(4).to_string(), "\n", O.round(4).to_string(),
          flush=True)


if __name__ == "__main__":
    run()
