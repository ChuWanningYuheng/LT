"""Iteration 9b, stage 1: Marxian profit rate - ends, long US series 1947-2024, other countries, OEWS functions,
decomposition, external criterion, Moseley (pre_registration_v9b.md, stage 1).

Outputs: results/v9b/s1_*.csv
"""
from __future__ import annotations

import glob
import sys
import zipfile

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from ..v8.rule import decide, share_row
from ..v9.stage7 import FA, SECTORS, FA_NAME, UNPROD, fa_table, depreciation_total, norm, build as build97
from ..v6.series import ROOT

OUT = ROOT / "results" / "v9b"
OUT.mkdir(parents=True, exist_ok=True)
R = ROOT / "data" / "raw" / "v9b"
TQ = 1.645
VARS = ["main", "V2 transport unproductive", "V3 information + computer design unproductive",
        "V4 all professional services unproductive", "V5 Tsoulfidis-Paitaridis classification (journal 12)"]
SHORT = dict(zip(VARS, ["main", "V2", "V3", "V4", "V5"]))


# ---------------------------------------------------------------- trends (pre-registration: wider of HAC and t(n-2))
def trend(y):
    y = pd.Series(y).dropna().astype(float)
    t = np.arange(len(y))
    X = sm.add_constant(t)
    h = sm.OLS(y.to_numpy(), X).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
    o = sm.OLS(y.to_numpy(), X).fit()
    b = float(o.params[1])
    w_hac, w_ols = TQ * h.bse[1], stats.t.ppf(0.95, len(y) - 2) * o.bse[1]
    w = max(w_hac, w_ols)
    return dict(est=b, se_hac=float(h.bse[1]), se_ols=float(o.bse[1]), ci90_lo=b - w, ci90_hi=b + w, n=len(y),
                label=decide(b, b - w, b + w, 0, 0.0002, "<"))


# ---------------------------------------------------------------- NIPA
def nipa(code):
    d = pd.read_csv(R / "bea_NipaDataA.txt", dtype=str)
    d.columns = ["code", "period", "value"]
    d = d[d.code == code]
    return pd.Series(pd.to_numeric(d.value.str.replace(",", ""), errors="coerce").to_numpy(), index=d.period.astype(int))


# ---------------------------------------------------------------- SIC compensation 1947-1997
SIC_UNPROD = {
    "main": ["Wholesale trade", "Retail trade", "Finance, insurance, and real estate", "Business services", "Legal services",
             "Government"],
}
SIC_UNPROD["V2 transport unproductive"] = SIC_UNPROD["main"] + ["Transportation"]
SIC_UNPROD["V3 information + computer design unproductive"] = SIC_UNPROD["main"] + ["Communications", "Printing and publishing",
                                                                                     "Motion pictures"]
SIC_UNPROD["V4 all professional services unproductive"] = SIC_UNPROD["main"] + ["Miscellaneous professional services",
                                                                                "Other services"]
SIC_UNPROD["V5 Tsoulfidis-Paitaridis classification (journal 12)"] = SIC_UNPROD["V4 all professional services unproductive"]


def sic_comp(sheet):
    d = pd.read_excel(R / "bea_GDPbyInd_VA_SIC.xls", sheet_name=sheet, header=None)
    years = [int(v) for v in d.iloc[0, 2:] if str(v).isdigit()]
    c = d[d[0] == "COMP"]
    out = {}
    for _, row in c.iterrows():
        name = str(row[1]).strip()
        if name not in out:                                   # first occurrence (government sub-lines repeat)
            out[name] = pd.to_numeric(row.iloc[2:2 + len(years)], errors="coerce").set_axis(years)
    return out


def sic_v(var):
    res = {}
    for sh in ("72SIC_Components of VA", "87SIC_Components of VA"):
        c = sic_comp(sh)
        allc = c["All industries"]
        un = sum(c[k] for k in SIC_UNPROD[var] if k in c)
        res[sh[:5]] = pd.DataFrame({"v": allc - un, "comp_all": allc})
    return res


# ---------------------------------------------------------------- NAICS (1997+) and capital (1947+)
def naics_v():
    D, tot = build97()
    out = {}
    for var, unp in UNPROD.items():
        if var not in VARS:
            continue
        prod = D[~D.code.isin(unp)]
        out[var] = pd.DataFrame({"v": prod.groupby("year").comp.sum(), "comp_all": tot.comp_all})
    return out, D, tot


def capital_long():
    k = fa_table("FAAt301ESI-A")
    rows = {}
    for code, name in SECTORS.items():
        if code == "gov":
            continue
        s = k.get(norm(FA_NAME.get(code, name)))
        if s is None:
            raise KeyError(name)
        rows[code] = s
    K = pd.DataFrame(rows)
    return K, k[norm("Private fixed assets")]


def splice(old, new, year):
    """ratio splice: old series scaled by new/old in the overlap year"""
    f = new.loc[year] / old.loc[year]
    return pd.concat([old[old.index < year] * f, new[new.index >= year]]), float(f)


def long_series(raw=False):
    nv, D97, tot97 = naics_v()
    K, Kpriv = capital_long()
    gdp = nipa("A191RC")
    dep = depreciation_total()
    years = range(1947, 2025)
    ndp = (gdp - dep).reindex(years)
    out, breaks = [], []
    for var in VARS:
        s = sic_v(var)
        a, b, n = s["72SIC"], s["87SIC"], nv[var]
        br = {}
        for col in ("v", "comp_all"):
            if raw:
                x = pd.concat([a[col][a.index < 1987], b[col][(b.index >= 1987) & (b.index < 1997)], n[col][n.index >= 1997]])
                br[col] = (np.nan, np.nan)
            else:
                ab, f1 = splice(a[col], b[col], 1987)
                x, f2 = splice(ab, n[col], 1997)
                br[col] = (f1, f2)
            br[col + "_series"] = x
        for yr, old, new in ((1987, a, b), (1997, b, n)):
            breaks.append(dict(variant=var, year=yr, share_prod_old=float(old.v.loc[yr] / old.comp_all.loc[yr]),
                               share_prod_new=float(new.v.loc[yr] / new.comp_all.loc[yr])))
        unp = UNPROD[var]
        C = K[[c for c in K.columns if c not in unp]].sum(1)
        v = br["v_series"].reindex(years)
        ca = br["comp_all_series"].reindex(years)
        Cc = C.reindex(years)
        s_ = ndp - v
        rm = s_ / (Cc + v)
        r = (ndp - ca) / Kpriv.reindex(years)
        out.append(pd.DataFrame({"variant": var, "year": list(years), "r_m": rm.values, "r": r.values, "e_m": (s_ / v).values,
                                 "kappa": (Cc / ndp).values, "nu": (v / ndp).values, "C": Cc.values, "v": v.values,
                                 "ndp": ndp.values, "comp_all": ca.values, "K_private": Kpriv.reindex(years).values,
                                 "unprod_share_comp": (1 - v / ca).values}))
    B = pd.DataFrame(breaks)
    B["gap_pp"] = 100 * (B.share_prod_new - B.share_prod_old)
    return pd.concat(out, ignore_index=True), B


# ---------------------------------------------------------------- 1.1 ends
def ends(L):
    rows = []
    for var in VARS:
        x = L[L.variant == var].set_index("year")
        for y0 in range(1997, 2003):
            for y1 in range(2019, 2025):
                w = x.loc[y0:y1]
                t37, t38 = trend(w.r_m), trend(w.r_m - w.r)
                rows.append(dict(variant=SHORT[var], start=y0, end=y1, est37=t37["est"], lo37=t37["ci90_lo"],
                                 hi37=t37["ci90_hi"], label37=t37["label"], est38=t38["est"], lo38=t38["ci90_lo"],
                                 hi38=t38["ci90_hi"], label38=t38["label"]))
    E = pd.DataFrame(rows)
    S = E.groupby("variant").agg(n=("label37", "size"), k37=("label37", lambda s: int((s == "подтверждено").sum())),
                                 k38=("label38", lambda s: int((s == "подтверждено").sum())),
                                 min37=("est37", "min"), max37=("est37", "max"), min38=("est38", "min"),
                                 max38=("est38", "max")).reset_index()
    S["robust37"] = S.k37 == S.n
    S["robust38"] = S.k38 == S.n
    return E, S


# ---------------------------------------------------------------- 1.2 long outcomes
def long_outcomes(L, Lraw):
    rows = []
    for tag, data in (("ratio splice", L), ("raw splice", Lraw)):
        for var in VARS:
            x = data[data.variant == var].set_index("year")
            for (a, b) in ((1947, 2024), (1947, 1973), (1973, 1997), (1997, 2024)):
                w = x.loc[a:b]
                for item, y in (("39: trend r_m", w.r_m), ("40: trend r_m - r", w.r_m - w.r)):
                    t = trend(y)
                    rows.append(dict(splice=tag, variant=SHORT[var], period=f"{a}-{b}", item=item, **t))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 1.3 other countries (Eurostat; KLEMS gross)
def es_load():
    from ..v8.stage4 import es, R8
    it = {k: es(R8 / f"a64_{k}.json") for k in ("B1G", "D1", "P51C")}
    st = es(R8 / "eurostat_nama_10_nfa_st.json")
    st = st[st.asset10 == "N11N"]
    return it, st


EU_UNPROD = {"main": ["G", "K", "L", ("M69_M70", "M69-M71"), "N", "O"]}
EU_UNPROD["V2"] = EU_UNPROD["main"] + ["H"]
EU_UNPROD["V3"] = EU_UNPROD["main"] + ["J"]
EU_UNPROD["V4"] = ["G", "K", "L", "M", "N", "O"]
EU_UNPROD["V5"] = EU_UNPROD["V4"] + [("E37-E39", "E")]
EU_UNPROD["V6"] = EU_UNPROD["main"] + ["P", "Q"]
EXCL = {"EU27_2020", "EU28", "EU15", "EA", "EA19", "EA20", "EA21", "EA12", "EU"}


def eu_country_panel():
    it, st = es_load()
    piv = lambda d: d.pivot_table(index=["geo", "time"], columns="nace_r2", values="value")  # noqa: E731
    D1, B1G, P51C, N = piv(it["D1"]), piv(it["B1G"]), piv(it["P51C"]), piv(st)
    out = []
    for var, unp in EU_UNPROD.items():
        for (geo, time) in D1.index:
            if geo in EXCL:
                continue
            try:
                d, n = D1.loc[(geo, time)], N.loc[(geo, time)]
                b, p = B1G.loc[(geo, time)], P51C.loc[(geo, time)]
            except KeyError:
                continue
            un_d, un_n, ok = 0.0, 0.0, True
            for u in unp:
                opts = u if isinstance(u, tuple) else (u,)
                hit = [o for o in opts if np.isfinite(d.get(o, np.nan)) and np.isfinite(n.get(o, np.nan))]
                if not hit:
                    ok = False
                    break
                un_d += d[hit[0]]
                un_n += n[hit[0]]
            need = [d.get("TOTAL"), n.get("TOTAL"), b.get("TOTAL"), p.get("TOTAL"), n.get("O")]
            if not ok or not all(np.isfinite(v) for v in need):
                continue
            ndp = b["TOTAL"] - p["TOTAL"]
            v = d["TOTAL"] - un_d
            C = n["TOTAL"] - un_n
            out.append(dict(variant=var, geo=geo, year=int(time), r_m=(ndp - v) / (C + v),
                            r=(ndp - d["TOTAL"]) / (n["TOTAL"] - n["O"]), nu=v / ndp, kappa=C / ndp,
                            unprod_share_comp=un_d / d["TOTAL"]))
    return pd.DataFrame(out)


def longest_run(years):
    ys = sorted(years)
    best, cur = [ys[0]], [ys[0]]
    for a in ys[1:]:
        cur = cur + [a] if a == cur[-1] + 1 else [a]
        if len(cur) > len(best):
            best = cur
    return best


def eu_outcomes(P):
    rows = []
    main = P[P.variant == "main"]
    keep = {}
    for geo, g in main.groupby("geo"):
        run = longest_run(g.year.tolist())
        if len(run) >= 20:
            keep[geo] = (run[0], run[-1])
    for var, g0 in P.groupby("variant"):
        for geo, (a, b) in keep.items():
            x = g0[(g0.geo == geo) & g0.year.between(a, b)].set_index("year").sort_index()
            if len(x) < 20:
                continue
            t37, t38 = trend(x.r_m), trend(x.r_m - x.r)
            rows.append(dict(variant=var, geo=geo, y0=int(x.index.min()), y1=int(x.index.max()), n=len(x),
                             rm_first=x.r_m.iloc[0], rm_last=x.r_m.iloc[-1], est37=t37["est"], lo37=t37["ci90_lo"],
                             hi37=t37["ci90_hi"], label37=t37["label"], est38=t38["est"], lo38=t38["ci90_lo"],
                             hi38=t38["ci90_hi"], label38=t38["label"]))
    C = pd.DataFrame(rows)
    shares = []
    for var, g in C.groupby("variant"):
        for item, col in (("41: share of countries with r_m trend 'подтверждено'", "label37"),
                          ("42: share of countries with (r_m - r) trend 'подтверждено'", "label38")):
            row = share_row(item, g[col] == "подтверждено", 0.5, 0.15, ">")
            row["variant"] = var
            row["share_negative_point"] = float((g["est" + col[-2:]] < 0).mean())
            shares.append(row)
    return C, pd.DataFrame(shares)


def klems_gross():
    na = pd.read_csv(R / "klems2025_national_accounts.csv", usecols=["nace_r2_code", "geo_code", "year", "VA_CP", "COMP"],
                     low_memory=False)
    ca = pd.read_csv(R / "klems2025_capital_accounts.csv", usecols=["nace_r2_code", "geo_code", "year", "K_GFCF"],
                     low_memory=False)
    m = na.merge(ca, on=["nace_r2_code", "geo_code", "year"], how="left")
    unp = ["G", "K", "L", "N", "O"]
    rows = []
    for (geo, y), g in m.groupby(["geo_code", "year"]):
        if geo.startswith(("EU", "EA")):
            continue
        g = g.set_index("nace_r2_code")
        if "TOT" not in g.index or not set(unp) <= set(g.index):
            continue
        tot = g.loc["TOT"]
        un_c = g.loc[unp, "COMP"].sum(min_count=len(unp))
        un_k = g.loc[unp, "K_GFCF"].sum(min_count=len(unp))
        if not np.isfinite([tot.VA_CP, tot.COMP, tot.K_GFCF, un_c, un_k]).all():
            continue
        v = tot.COMP - un_c
        C = tot.K_GFCF - un_k
        rows.append(dict(geo=geo, year=int(y), r_m_g=(tot.VA_CP - v) / (C + v), r_g=(tot.VA_CP - tot.COMP) / tot.K_GFCF))
    P = pd.DataFrame(rows)
    T = []
    for geo, x in P.groupby("geo"):
        x = x.set_index("year").sort_index()
        if len(x) < 20:
            continue
        t1, t2 = trend(x.r_m_g), trend(x.r_m_g - x.r_g)
        T.append(dict(geo=geo, n=len(x), est37g=t1["est"], label37g=t1["label"], est38g=t2["est"], label38g=t2["label"]))
    T = pd.DataFrame(T)
    sh = [share_row("KLEMS gross: share with r_m^g trend 'подтверждено'", T.label37g == "подтверждено", 0.5, 0.15, ">"),
          share_row("KLEMS gross: share with (r_m^g - r^g) trend 'подтверждено'", T.label38g == "подтверждено", 0.5, 0.15, ">")]
    return P, T, pd.DataFrame(sh)


# ---------------------------------------------------------------- 1.4 OEWS
OEWS_MAP = {"agr": ["11"], "min": ["21"], "util": ["22"], "con": ["23"], "man": ["31-33"], "trn": ["48-49"], "inf": ["51"],
            "csd": ["54"], "mps": ["54"], "wst": ["562"], "edu": ["61"], "hlt": ["62"], "art": ["71"], "acc": ["72"],
            "oth": ["81"]}
FUN = {"main": {"major": ["11-0000", "13-0000", "41-0000"], "detailed": ["33-9032"]},
       "wide": {"major": ["11-0000", "13-0000", "41-0000", "43-0000", "23-0000"], "detailed": ["33-9032"]}}


def _oews_frame(zf, pattern):
    z = zipfile.ZipFile(zf)
    names = [n for n in z.namelist() if pattern in n.split("/")[-1] and "owner" not in n]
    frames = []
    for nm in names:
        with z.open(nm) as fh:
            frames.append(pd.read_excel(fh))
    d = pd.concat(frames)
    d.columns = [c.upper() for c in d.columns]
    d = d.rename(columns={"O_GROUP": "OCC_GROUP"})
    d["NAICS"] = d.NAICS.astype(str).str.strip()
    for c in ("TOT_EMP", "A_MEAN"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["WB"] = d.TOT_EMP * d.A_MEAN
    return d


def _naics_key(code):
    c = str(code).replace(".0", "")
    c = c.rstrip("0") if c.isdigit() and len(c) == 6 else c
    return c


def oews_shares():
    rows = []
    for y in range(2012, 2025):
        zf = R / f"oesm{y % 100:02d}in4.zip"
        sec = _oews_frame(zf, "natsector")
        d3 = _oews_frame(zf, "nat3d")
        both = pd.concat([sec, d3])
        both["key"] = both.NAICS.map(_naics_key)
        for code, keys in OEWS_MAP.items():
            k = keys[0]
            g = both[both.key == k]
            if g.empty and "-" in k:
                g = both[both.key == k.split("-")[0]]
            g = g.drop_duplicates(subset=["OCC_CODE"])
            tot = g[g.OCC_CODE == "00-0000"].WB.sum()
            for fv, spec in FUN.items():
                un = g[g.OCC_CODE.isin(spec["major"] + spec["detailed"])].WB.sum()
                rows.append(dict(year=y, code=code, oews=k, variant=fv, share=un / tot if tot > 0 else np.nan,
                                 total_wb=tot))
    return pd.DataFrame(rows)


def oews_adjusted(S):
    D, tot = build97()
    gdp_dep = tot.ndp
    K, _ = capital_long()
    rows = []
    for fv in ("main", "wide"):
        sh = S[S.variant == fv].pivot_table(index="year", columns="code", values="share")
        for y in range(2012, 2025):
            prod = D[(D.year == y) & ~D.code.isin(UNPROD["main"])].set_index("code")
            v = prod.comp.sum()
            d = sh.loc[y].reindex(prod.index)
            if d.isna().any():
                raise ValueError(f"missing OEWS share {y}: {list(d[d.isna()].index)}")
            v2 = (prod.comp * (1 - d)).sum()
            C = K.loc[y, [c for c in K.columns if c not in UNPROD["main"]]].sum()
            ndp = gdp_dep.loc[y]
            rows.append(dict(variant=fv, year=y, r_m=(ndp - v) / (C + v), r_m_adj=(ndp - v2) / (C + v2),
                             fun_share_of_v=1 - v2 / v))
    A = pd.DataFrame(rows)
    T = []
    for fv, g in A.groupby("variant"):
        g = g.set_index("year")
        for col in ("r_m", "r_m_adj", "fun_share_of_v"):
            t = trend(g[col])
            T.append(dict(variant=fv, series=col, first=g[col].iloc[0], last=g[col].iloc[-1], **t))
    return A, pd.DataFrame(T)


# ---------------------------------------------------------------- 1.5 decomposition + Basu
def decompose(L, periods=((1997, 2024), (1947, 2024), (1947, 1973), (1973, 1997))):
    x = L[L.variant == "main"].set_index("year")
    q = fa_table("FAAt302ESI-A")
    k = fa_table("FAAt301ESI-A")
    py = nipa("A191RD")
    py = py / py.loc[2017]
    real = 0
    for code, name in SECTORS.items():
        if code == "gov" or code in UNPROD["main"]:
            continue
        nm = norm(FA_NAME.get(code, name))
        real = real + k[nm].loc[2017] * q[nm] / q[nm].loc[2017]
    real = real.reindex(x.index)
    PK = x.C / real
    rows = []
    for a, b in periods:
        w = x.loc[a:b]
        nu, ka = w.nu.to_numpy(), w.kappa.to_numpy()
        nub, kab = (nu[1:] + nu[:-1]) / 2, (ka[1:] + ka[:-1]) / 2
        ck = float(np.sum(-np.diff(ka) / (kab + nub)))
        cn = float(np.sum(-np.diff(nu) * (1 / (1 - nub) + 1 / (kab + nub))))
        tot = float(np.log(w.r_m.iloc[-1] / w.r_m.iloc[0]))
        dk = float(np.log(w.kappa.iloc[-1] / w.kappa.iloc[0]))
        ndp_real = w.ndp / py.reindex(w.index)
        d_vol = float(np.log((real.loc[b] / ndp_real.loc[b]) / (real.loc[a] / ndp_real.loc[a])))
        d_price = float(np.log((PK.loc[b] / py.loc[b]) / (PK.loc[a] / py.loc[a])))
        rows.append(dict(period=f"{a}-{b}", dln_rm=tot, contrib_kappa=ck, contrib_nu=cn, residual=tot - ck - cn,
                         dln_kappa=dk, dln_K_real_over_NDP_real=d_vol, dln_PK_over_PY=d_price,
                         kappa_first=w.kappa.iloc[0], kappa_last=w.kappa.iloc[-1], nu_first=w.nu.iloc[0], nu_last=w.nu.iloc[-1],
                         em_first=w.e_m.iloc[0], em_last=w.e_m.iloc[-1]))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 1.6 external criterion
def external(L):
    from ..v8.stage4 import measures_us, oos_measure
    m, lnq, rec = measures_us()
    m["r_m"] = L[L.variant == "main"].set_index("year").r_m.reindex(m.index)
    rows = []
    for c in ("r", "r_n", "rM", "all_assets", "r_m"):
        rows += oos_measure(m, c, lnq, rec)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 1.7 Moseley
def moseley(L):
    x = L[L.variant == "main"].set_index("year")
    t = trend(x.unprod_share_comp)
    w = t["ci90_hi"] - t["est"]
    o43 = dict(item="43: trend of unproductive share of compensation, 1947-2024", est=t["est"], ci90_lo=t["ci90_lo"],
               ci90_hi=t["ci90_hi"], n=t["n"], label=decide(t["est"], t["est"] - w, t["est"] + w, 0, 0.0005, ">"))
    ups = (x.comp_all - x.v) / x.ndp
    kk = x.K_private / x.ndp
    nu = x.nu
    rows = []
    for a, b in ((1947, 1979), (1979, 1997), (1997, 2024), (1947, 2024)):
        w_ = slice(a, b)
        n_, u_, k_ = nu.loc[w_].to_numpy(), ups.loc[w_].to_numpy(), kk.loc[w_].to_numpy()
        nb, ub, kb = (n_[1:] + n_[:-1]) / 2, (u_[1:] + u_[:-1]) / 2, (k_[1:] + k_[:-1]) / 2
        cn = float(np.sum(-np.diff(n_) / kb))
        cu = float(np.sum(-np.diff(u_) / kb))
        ck = float(np.sum(-(1 - nb - ub) * np.diff(k_) / kb ** 2))
        dr = float(x.r.loc[b] - x.r.loc[a])
        rows.append(dict(period=f"{a}-{b}", d_r=dr, contrib_nu=cn, contrib_unprod=cu, contrib_k=ck, residual=dr - cn - cu - ck,
                         unprod_share_first=x.unprod_share_comp.loc[a], unprod_share_last=x.unprod_share_comp.loc[b]))
    return o43, pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    part = sys.argv[1] if len(sys.argv) > 1 else "all"
    if part in ("long", "all"):
        L, B = long_series()
        Lr, _ = long_series(raw=True)
        L.to_csv(OUT / "s1_long_series.csv", index=False)
        Lr.to_csv(OUT / "s1_long_series_raw.csv", index=False)
        B.to_csv(OUT / "s1_splice_breaks.csv", index=False)
        print(B.round(4).to_string(), flush=True)
        E, S = ends(L)
        E.to_csv(OUT / "s1_ends.csv", index=False)
        S.to_csv(OUT / "s1_ends_summary.csv", index=False)
        print(S.round(5).to_string(), flush=True)
        O = long_outcomes(L, Lr)
        O.to_csv(OUT / "s1_long_outcomes.csv", index=False)
        print(O[O.variant == "main"].round(5).to_string(), flush=True)
        Dc = decompose(L)
        Dc.to_csv(OUT / "s1_decomposition.csv", index=False)
        print(Dc.round(4).to_string(), flush=True)
        o43, M = moseley(L)
        pd.DataFrame([o43]).to_csv(OUT / "s1_moseley_outcome.csv", index=False)
        M.to_csv(OUT / "s1_moseley.csv", index=False)
        print(o43, "\n", M.round(4).to_string(), flush=True)
    if part in ("countries", "all"):
        P = eu_country_panel()
        P.to_csv(OUT / "s1_eu_panel.csv", index=False)
        C, Sh = eu_outcomes(P)
        C.to_csv(OUT / "s1_eu_trends.csv", index=False)
        Sh.to_csv(OUT / "s1_eu_outcomes.csv", index=False)
        print(C[C.variant == "main"].round(5).to_string(), "\n", Sh.round(3).to_string(), flush=True)
        KP, KT, KS = klems_gross()
        KP.to_csv(OUT / "s1_klems_gross_panel.csv", index=False)
        KT.to_csv(OUT / "s1_klems_gross_trends.csv", index=False)
        KS.to_csv(OUT / "s1_klems_gross_outcomes.csv", index=False)
        print(KT.round(5).to_string(), "\n", KS.round(3).to_string(), flush=True)
    if part in ("oews", "all"):
        S = oews_shares()
        S.to_csv(OUT / "s1_oews_shares.csv", index=False)
        A, T = oews_adjusted(S)
        A.to_csv(OUT / "s1_oews_adjusted.csv", index=False)
        T.to_csv(OUT / "s1_oews_trends.csv", index=False)
        print(T.round(5).to_string(), flush=True)
    if part in ("external", "all"):
        L = pd.read_csv(OUT / "s1_long_series.csv")
        X = external(L)
        X.to_csv(OUT / "s1_external.csv", index=False)
        print(X.round(3).to_string(), flush=True)
