"""Iteration 12b, stage 3: manufacturing (pre_registration_v12b.md, stage 3; outcomes 101-103; journal 6).

US 1948-2023 (NAICS): VA of manufacturing industries from the IO tables (T006); compensation: 1987+ NAICS, before:
NAICS VA x labour share of SIC manufacturing (journal 6); depreciation and current-cost stock: BEA FA 3.4 / 3.1 (E+S+I),
manufacturing; r = (VA - comp - CFC) / K, r_m = (VA - comp - CFC) / (K + comp). Private sector: same definitions with
all private industries (FA line 1).
r_h of manufacturing from the stage-1 system: (L_mfg - V_h,mfg) / (C_h,mfg + V_h,mfg), V_h,mfg = lambda_h . b x
W_mfg / W_prod, C_h,mfg = lambda_h . K_mfg (FA manufacturing E/S/I, type composition of table 2.1).
Europe: EU KLEMS 2009 (1970-2007, D) and 2025 (1995-2021, C), gross r = (VA - LAB) / K (2025: LAB = COMP x EMP/EMPE;
no depreciation in KLEMS 2025), spliced on 1995-2000 (journal 6).

Usage: PYTHONPATH=src python -P -m lts.v12b.stage3
Outputs: results/v12b/s3_*.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..v6.series import ROOT
from ..v8.rule import decide, share_row
from . import us_io as u
from .stage1 import robust, trend

OUT = ROOT / "results" / "v12b"
R9B = ROOT / "data" / "raw" / "v9b"
NON_EUROPE = {"USA", "JPN", "AUS"}


def sic_mfg_share():
    """labour share (comp / VA) of SIC manufacturing and SIC private industries, 1947-1997"""
    out = {}
    for sh_va, sh_c, yrs in (("72SIC_VA, GO, II", "72SIC_Components of VA", None), ("87SIC_VA, GO, II", "87SIC_Components of VA", None)):
        for sheet, start in ((sh_va, "Value added"), (sh_c, "Compensation of employees")):
            d = pd.read_excel(ROOT / "data" / "raw" / "v9b" / "bea_GDPbyInd_VA_SIC.xls", sheet_name=sheet, header=None)
            years = [int(float(v)) for v in d.iloc[0, 2:] if str(v).replace(".0", "").isdigit()]
            i0 = d.index[d.iloc[:, 0].astype(str).str.startswith(start)][0]
            blk = d.iloc[i0 + 1:]
            code = blk.iloc[0, 0]
            blk = blk[blk.iloc[:, 0] == code]
            names = [" ".join(str(v).split()) for v in blk.iloc[:, 1]]
            vals = blk.iloc[:, 2:2 + len(years)].apply(lambda c: pd.to_numeric(c, errors="coerce"))
            vals.index, vals.columns = names, years
            key = "va" if sheet == sh_va else "comp"
            for nm, lab in (("Manufacturing", "mfg"), ("Private industries", "priv")):
                s = vals.loc[nm] if nm in vals.index else vals.iloc[[i for i, n in enumerate(names) if n.startswith(nm)][0]]
                prev = out.get((key, lab))
                out[(key, lab)] = s if prev is None else pd.concat([prev[prev.index < 1988], s[s.index >= 1988]])
    return {lab: out[("comp", lab)] / out[("va", lab)] for lab in ("mfg", "priv")}


def fa_sector(sheet_prefix, name, year):
    tot = 0.0
    for t in ("E", "S", "I"):
        d = u.fa_by_name(f"{sheet_prefix}{t}-A")
        s = d.get(name)
        if s is None:
            vals, names = u.fa_lines(f"{sheet_prefix}{t}-A")
            s = vals.loc[names.index[0]] if name == "private fixed assets" else None
        tot += float(np.nan_to_num(s.get(year, np.nan))) if s is not None else np.nan
    return tot


def us_series():
    ls = sic_mfg_share()
    gov = {"GFG", "GFE", "GSLG", "GSLE"}
    rows = []
    for y in u.YEARS:
        t = u.year_tables(y)
        va = pd.Series(t["va"], index=u.CODES)
        comp = u.compensation(y) if y >= 1987 else None
        va_m, va_p = float(va[u.MFG].sum()), float(va[[c for c in u.CODES if c not in gov]].sum())
        if y >= 1987:
            w_m, w_p = float(comp[u.MFG].sum()), float(comp[[c for c in u.CODES if c not in gov]].sum())
        else:
            w_m, w_p = va_m * float(ls["mfg"].loc[y]), va_p * float(ls["priv"].loc[y])
        rows.append(dict(year=y, VA_mfg=va_m, W_mfg=w_m, CFC_mfg=fa_sector("FAAt304", "manufacturing", y),
                         K_mfg=fa_sector("FAAt301", "manufacturing", y), VA_priv=va_p, W_priv=w_p,
                         CFC_priv=fa_sector("FAAt304", "private fixed assets", y),
                         K_priv=fa_sector("FAAt301", "private fixed assets", y), **mfg_hours(y)))
    d = pd.DataFrame(rows).set_index("year")
    for s in ("mfg", "priv"):
        prof = d[f"VA_{s}"] - d[f"W_{s}"] - d[f"CFC_{s}"]
        d[f"r_{s}"] = prof / d[f"K_{s}"]
        d[f"rm_{s}"] = prof / (d[f"K_{s}"] + d[f"W_{s}"])
    return d


def mfg_hours(y):
    s = u.system(y)
    A, c, x = s["A"], s["c"], s["x"]
    P = u.persons(y).to_numpy()
    W = u.compensation(y).to_numpy()
    prod = np.array([cd not in u.UNPRODUCTIVE for cd in u.CODES])
    mf = np.isin(u.CODES, u.MFG)
    lam = np.divide(P * prod, x, out=np.zeros(len(x)), where=x > 0) @ np.linalg.inv(np.eye(len(x)) - A)
    Wp = float((W * prod).sum())
    omega = Wp / float(u.nipa_series().loc[y, "A065RC"])
    V_all = float(lam @ (omega * c))
    v21, _ = u.fa_lines("FAAt201-A")
    Kv = np.zeros(len(x))
    for typ, tl in u.BROAD_TOTAL.items():
        lines = [ln for ln, (cd, tp) in u.ASSET_LINES.items() if tp == typ]
        comp = v21.loc[lines, y].clip(lower=0)
        comp = comp / comp.sum()
        tot = float(np.nan_to_num(u.fa_by_name(f"FAAt301{typ}-A").get("manufacturing").get(y, np.nan)))
        for ln, sh in comp.items():
            Kv[u.CODES.index(u.ASSET_LINES[ln][0])] += tot * sh
    L_m = float(P[mf].sum())
    V_m = V_all * float(W[mf].sum()) / Wp
    C_m = float(lam @ Kv)
    return dict(L_mfg=L_m, Vh_mfg=V_m, Ch_mfg=C_m, rh_mfg=(L_m - V_m) / (C_m + V_m))


def europe():
    from ..v10.stage9 import k09_long, K09_ISO
    b = k09_long("countries")
    c = k09_long("capital")
    w = b[b["var"].isin(["VA", "LAB"]) & b.code.isin(["D", "TOT"])].pivot_table(index=["geo", "code", "year"], columns="var", values="v")
    k = c[c["var"].isin(["K_GFCF", "Ip_GFCF", "D_GFCF"]) & c.code.isin(["D", "TOT"])].pivot_table(
        index=["geo", "code", "year"], columns="var", values="v")
    k["Knom"] = k.K_GFCF * k.Ip_GFCF
    k["Dnom"] = k.D_GFCF * k.Ip_GFCF if "D_GFCF" in k else np.nan
    k09 = w.join(k[["Knom", "Dnom"]], how="left").reset_index()
    k09["r"] = (k09.VA - k09.LAB) / k09.Knom
    k09["r_net"] = (k09.VA - k09.LAB - k09.Dnom) / k09.Knom
    na = pd.read_csv(R9B / "klems2025_national_accounts.csv", usecols=["nace_r2_code", "geo_code", "year", "VA_CP", "COMP", "EMP", "EMPE"],
                     low_memory=False)
    ca = pd.read_csv(R9B / "klems2025_capital_accounts.csv", usecols=["nace_r2_code", "geo_code", "year", "K_GFCF"], low_memory=False)
    m = na.merge(ca, on=["nace_r2_code", "geo_code", "year"], how="left")
    m = m[m.nace_r2_code.isin(["C", "TOT"])].copy()
    m["LAB"] = m.COMP * m.EMP / m.EMPE
    m["r"] = (m.VA_CP - m.LAB) / m.K_GFCF
    iso2 = {"UK": "GBR", "EL": "GRC"}
    m["geo"] = m.geo_code.map(lambda g: iso2.get(g, None))
    from ..v10.stage9 import ISO3
    m["geo"] = m.geo_code.map(ISO3).fillna(m.geo)
    rows = []
    for geo in sorted((set(k09.geo.dropna()) & set(m.geo.dropna())) - NON_EUROPE):
        for code09, code25, lab in (("D", "C", "mfg"), ("TOT", "TOT", "total")):
            s09 = k09[(k09.geo == geo) & (k09.code == code09)].set_index("year").r.dropna()
            s25 = m[(m.geo == geo) & (m.nace_r2_code == code25)].set_index("year").r.dropna()
            s09, s25 = s09[np.isfinite(s09) & (s09 > 0)], s25[np.isfinite(s25) & (s25 > 0)]
            if s25.empty or s09.empty:
                continue
            common = [y for y in range(1995, 2001) if y in s09.index and y in s25.index]
            if not common:
                continue
            f = float(np.mean([s25[y] / s09[y] for y in common]))
            sp = pd.concat([s09[s09.index < s25.index.min()] * f, s25]).sort_index()
            for y, v in sp.items():
                rows.append(dict(geo=geo, sector=lab, year=int(y), r=float(v), splice=f))
    return pd.DataFrame(rows)


def run():
    d = us_series()
    d.to_csv(OUT / "s3_us_manufacturing.csv")
    out, desc = [], []
    o = trend(np.log(d.r_mfg), 4, 0.002, "<")
    o.update(robust(np.log(d.r_mfg), 0.002, "<"))
    out.append(dict(outcome="101: US manufacturing, trend ln r (net, current cost), 1948-2023", **o))
    o = trend(np.log(d.r_mfg / d.r_priv), 4, 0.002, "<")
    out.append(dict(outcome="102: US, trend ln(r_manufacturing / r_private sector)", **o))
    for nm, s, dr in (("ln r private sector", d.r_priv, "<"), ("ln r_m manufacturing", d.rm_mfg, "<"),
                      ("ln r_m private", d.rm_priv, "<"), ("ln r_h manufacturing", d.rh_mfg, "<")):
        desc.append(dict(item=f"US trend {nm}", **trend(np.log(s), 4, 0.002, dr)))
    for a, b in ((1948, 1973), (1973, 1997), (1997, 2023)):
        desc.append(dict(item=f"US trend ln r manufacturing {a}-{b}", **trend(np.log(d.r_mfg.loc[a:b]), 4, 0.002, "<")))
    E = europe()
    E.to_csv(OUT / "s3_europe.csv", index=False)
    labs, labs_t, rel = [], [], []
    for geo, g in E.groupby("geo"):
        gm = g[g.sector == "mfg"].set_index("year").r
        gt = g[g.sector == "total"].set_index("year").r
        if len(gm) >= 30:
            t_ = trend(np.log(gm), 4, 0.002, "<")
            labs.append(t_["label"] == "подтверждено")
            desc.append(dict(item=f"Europe {geo}: trend ln r manufacturing ({int(gm.index.min())}-{int(gm.index.max())})", **t_))
            common = gm.index.intersection(gt.index)
            if len(common) >= 30:
                t2 = trend(np.log(gm.loc[common] / gt.loc[common]), 4, 0.002, "<")
                rel.append(t2["label"] == "подтверждено")
                desc.append(dict(item=f"Europe {geo}: trend ln(r_mfg / r_total)", **t2))
        if len(gt) >= 30:
            labs_t.append(trend(np.log(gt), 4, 0.002, "<")["label"] == "подтверждено")
    r103 = share_row("103: Europe (EU KLEMS 2009 + 2025, >= 30 years), share of countries with a confirmed falling r in "
                     "manufacturing", np.array(labs), 0.5, 0.15, ">")
    out.append(r103)
    desc.append(share_row("Europe: share of countries with a confirmed falling r, total economy", np.array(labs_t), 0.5, 0.15, ">"))
    desc.append(share_row("Europe: share of countries where r_mfg falls faster than r_total (confirmed)", np.array(rel), 0.5, 0.15, ">"))
    O = pd.DataFrame(out)
    O.to_csv(OUT / "s3_outcomes.csv", index=False)
    D = pd.DataFrame(desc)
    D.to_csv(OUT / "s3_descriptive.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 90)
    print(O.round(5).to_string())
    print(D.round(5).to_string())
    print(d.loc[[1948, 1960, 1973, 1980, 1997, 2007, 2023], ["r_mfg", "r_priv", "rm_mfg", "rh_mfg"]].round(4).to_string())


if __name__ == "__main__":
    run()
