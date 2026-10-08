"""Iteration 10, stage 1: labour-power shocks vs supplier shocks (pre_registration_v10.md, stage 1; journal 2-3).

1.1 US industry groups 1947-2025: local projections of profit shares on oil and labour shocks, Q ratios
    (outcomes 47-49), moving-block bootstrap, permutation placebo; variants (Kaenzig, Kilian, horizons, A with suppliers)
1.2 right-to-work laws: stacked DiD on state GOS shares (outcome 50), event study, enactment-year and SIC variants
1.3 synthetic control on the AMECO profit share (100 - ALCD2): Hartz IV (outcome 51), Hartz I-II, minimum wages
1.4 gas shock 2021-2022, EU A*10 (descriptive)
1.5 Section 232 steel tariffs (descriptive)

Usage: PYTHONPATH=src python -P -m lts.v10.stage1 [us|rtw|sc|gas|steel ...]
Outputs: results/v10/s1_*.csv
"""
from __future__ import annotations

import glob
import json
import sys
import zipfile

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import optimize

from ..v6.series import ROOT
from ..v8.rule import decide, CONF, REF, UNINF
from ..v9.stage7 import va_components
from ..v9b.stage1 import nipa
from ..v9b.stage45 import wild_t, widest

OUT = ROOT / "results" / "v10"
OUT.mkdir(parents=True, exist_ok=True)
R10 = ROOT / "data" / "raw" / "v10"
SIC_XLS = ROOT / "data" / "raw" / "v9b" / "bea_GDPbyInd_VA_SIC.xls"
USE = ROOT / "data" / "raw" / "bea" / "io" / "IOUse_Before_Redefinitions_PRO_1997-2023_Summary.xlsx"
H = range(0, 6)
H_MAIN = 2
B_BOOT, BLOCK, N_PERM = 2000, 4, 1000
YRS = np.arange(1940, 2035)


def equiv(est, lo, hi, theta0, delta):
    """equivalence prediction: confirmed if the CI is inside the zone, refuted if entirely outside"""
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return UNINF
    if lo >= theta0 - delta and hi <= theta0 + delta:
        return CONF
    if hi < theta0 - delta or lo > theta0 + delta:
        return REF
    return UNINF


def es_json(path):
    d = json.load(open(path))
    ids, size = d["id"], d["size"]
    cats = [sorted(d["dimension"][k]["category"]["index"].items(), key=lambda x: x[1]) for k in ids]
    mult = [int(np.prod(size[i + 1:])) for i in range(len(size))]
    rows = []
    for k, v in d["value"].items():
        k = int(k)
        rows.append({idn: cats[i][(k // mult[i]) % size[i]][0] for i, idn in enumerate(ids)} | {"value": v})
    return pd.DataFrame(rows)


# ================================================================ 1.1 US industry groups
def snorm(s):
    return " ".join(str(s).replace("/2/", "").replace("/3/", "").split()).lower()


def sic_components(sheet):
    d = pd.read_excel(SIC_XLS, sheet_name=sheet, header=None)
    years = [int(v) for v in d.iloc[0, 2:] if str(v).isdigit()]
    out = {}
    for code, comp in (("COMP", "comp"), ("TXPIXS", "tax"), ("GOS", "gos")):
        seen = set()
        for _, row in d[d[0] == code].iterrows():
            name = snorm(row[1])
            if name in seen:                                   # government sub-lines repeat
                continue
            seen.add(name)
            out.setdefault(name, {})[comp] = pd.to_numeric(row.iloc[2:2 + len(years)], errors="coerce").set_axis(
                years).astype(float)
    return {k: pd.DataFrame(v) for k, v in out.items() if len(v) == 3}


TRN = ["481", "482", "483", "484", "485", "486", "487OS", "493"]
DUR = ["321", "327", "331", "332", "333", "334", "335", "3361MV", "3364OT", "337", "339"]
NDUR_X_PET = ["311FT", "313TT", "315AL", "322", "323", "325", "326"]
SRV = ["5411", "5415", "5412OP", "55", "561", "562", "61", "621", "622", "623", "624", "711AS", "713", "721", "722", "81"]
OIL = ["oil and gas extraction", "petroleum and coal products"]


def G(sp, sm_, npl, nmi, cols):
    return dict(sic_plus=sp, sic_minus=sm_, naics_plus=npl, naics_minus=nmi, cols=cols)


BROAD = {
    "agr": G(["agriculture, forestry, and fishing"], [], ["agriculture, forestry, fishing, and hunting"], [],
             ["111CA", "113FF"]),
    "min_x_oil": G(["mining"], ["oil and gas extraction"], ["mining"], ["oil and gas extraction"], ["212", "213"]),
    "util": G(["electric, gas, and sanitary services"], [], ["utilities"], [], ["22"]),
    "con": G(["construction"], [], ["construction"], [], ["23"]),
    "dur": G(["durable goods"], [], ["durable goods"], [], DUR),
    "ndur_x_pet": G(["nondurable goods"], ["petroleum and coal products"], ["nondurable goods"],
                    ["petroleum and coal products"], NDUR_X_PET),
    "whl": G(["wholesale trade"], [], ["wholesale trade"], [], ["42"]),
    "ret": G(["retail trade"], [], ["retail trade"], [], ["441", "445", "452", "4A0"]),
    "trn": G(["transportation"], [], ["transportation and warehousing"], [], TRN),
    "inf": G(["communications"], [], ["information"], [], ["511", "512", "513", "514"]),
    "fin": G(["finance, insurance, and real estate"], ["real estate"], ["finance and insurance"], [],
             ["521CI", "523", "524", "525"]),
    "srv": G(["services"], [], ["professional and business services",
                                "educational services, health care, and social assistance",
                                "arts, entertainment, recreation, accommodation, and food services",
                                "other services, except government"], [], SRV),
}
FIXED = {
    "oil_sup": G(OIL, [], OIL, [], ["211", "324"]),
    "oil_con": G(["transportation", "electric, gas, and sanitary services", "chemicals and allied products", "farms"], [],
                 ["transportation and warehousing", "utilities", "chemical products", "farms"], [],
                 TRN + ["22", "325", "111CA"]),
    "A": G(["private industries"], OIL, ["private industries"], OIL, "private_x_oil"),
    "A_all": G(["private industries"], [], ["private industries"], [], "private"),
}


def combine(src, plus, minus):
    for n in plus + minus:
        if n not in src:
            raise KeyError(n)
    f = sum(src[n][["comp", "tax", "gos"]] for n in plus)
    if minus:
        f = f - sum(src[n][["comp", "tax", "gos"]] for n in minus)
    return f


def use_2017():
    x = pd.read_excel(USE, sheet_name="2017", header=None)
    codes = [str(c) if isinstance(c, str) else None for c in x.iloc[5]]
    rows = x.iloc[6:]

    def row(code):
        r = rows[rows[0].astype(str) == code]
        return pd.Series(pd.to_numeric(r.iloc[0], errors="coerce").fillna(0).to_numpy(), index=codes)
    oil = row("211") + row("324")
    va = row("V001") + row("V002") + row("V003")
    comp = row("V001")
    private = codes[2:codes.index("GFGD")]
    return oil, va, comp, private


def group_levels(spec, s72, s87, nai):
    a = combine(s72, spec["sic_plus"], spec["sic_minus"]).loc[1947:1987]
    b = combine(s87, spec["sic_plus"], spec["sic_minus"]).loc[1987:1997]
    c = combine(nai, spec["naics_plus"], spec["naics_minus"]).loc[1997:2025]
    out, fac = {}, {}
    for v in ("gos", "va", "comp"):
        sa, sb, sc = ((d.sum(axis=1) if v == "va" else d[v]) for d in (a, b, c))
        f1 = sc.loc[1997] / sb.loc[1997]
        sb = sb * f1
        f2 = sb.loc[1987] / sa.loc[1987]
        sa = sa * f2
        out[v] = pd.concat([sa[sa.index < 1987], sb[sb.index < 1997], sc])
        fac[v] = (f2, f1)
    return pd.DataFrame(out), fac


def us_groups():
    s72, s87 = sic_components("72SIC_Components of VA"), sic_components("87SIC_Components of VA")
    nai = va_components()
    oil, va, comp, private = use_2017()
    for spec in list(BROAD.values()) + list(FIXED.values()):
        if isinstance(spec["cols"], list):
            miss = [c for c in spec["cols"] if c not in private]
            if miss:
                raise KeyError(miss)
    # labour shares of the 12 broad groups, NAICS 2017 -> labour-consumer group (top quartile = 3 of 12)
    lab = {}
    for k, spec in BROAD.items():
        f = combine(nai, spec["naics_plus"], spec["naics_minus"]).loc[2017]
        lab[k] = f.comp / f.sum()
    lab = pd.Series(lab).sort_values(ascending=False)
    top = list(lab.index[:3])
    print("labour shares 2017 (NAICS):", lab.round(3).to_dict(), "-> labour consumers:", top, flush=True)
    groups = dict(FIXED)
    groups["lab_con"] = G(sum((BROAD[k]["sic_plus"] for k in top), []), sum((BROAD[k]["sic_minus"] for k in top), []),
                          sum((BROAD[k]["naics_plus"] for k in top), []), sum((BROAD[k]["naics_minus"] for k in top), []),
                          sum((BROAD[k]["cols"] for k in top), []))
    for k in BROAD:
        groups["broad_" + k] = BROAD[k]
    series, shares, facs = {}, [], []
    for g, spec in groups.items():
        L, fac = group_levels(spec, s72, s87, nai)
        if L.isna().any().any():
            raise ValueError(f"missing values in group {g}: {L[L.isna().any(axis=1)].index.tolist()}")
        series[g] = L
        cols = spec["cols"]
        if cols == "private_x_oil":
            cols = [c for c in private if c not in ("211", "324")]
        elif cols == "private":
            cols = private
        n17 = combine(nai, spec["naics_plus"], spec["naics_minus"]).loc[2017]
        shares.append(dict(group=g, s_oil=oil[cols].sum() / va[cols].sum(), s_lab=n17.comp / n17.sum(),
                           s_lab_use=comp[cols].sum() / va[cols].sum(), va2017_use=va[cols].sum(),
                           splice_gos_1987=fac["gos"][0], splice_gos_1997=fac["gos"][1],
                           splice_va_1987=fac["va"][0], splice_va_1997=fac["va"][1]))
    S = pd.DataFrame(shares).set_index("group")
    pi = pd.DataFrame({g: 100 * L.gos / L.va for g, L in series.items()})
    ls = pd.DataFrame({g: 100 * L.comp / L.va for g, L in series.items()})
    lab.rename("labour_share_2017").to_csv(OUT / "s1_us_broad_labour_shares.csv")
    return pi, ls, S, top


def fred(path):
    d = pd.read_csv(path)
    return pd.Series(pd.to_numeric(d.iloc[:, 1], errors="coerce").to_numpy(), index=pd.to_datetime(d.iloc[:, 0]))


def annual(s, how="sum", need=12):
    s = s.dropna()
    g = s.groupby(s.index.year).agg([how, "count"])
    return g.loc[g["count"] >= need, how]


def shocks():
    x = pd.read_excel(R10 / "bh_oil_supply_shocks.xlsx", header=None).iloc[2:, :2].dropna()
    bh = annual(pd.Series(x[1].astype(float).to_numpy(), index=pd.to_datetime(x[0])))
    k = pd.read_csv(R10 / "kaenzig2021_oil_supply_surprises.csv")
    ka = annual(pd.Series(k.oil_supply_surprise_contract15.to_numpy(),
                          index=pd.to_datetime(k.month.str.replace("M", "-") + "-01"))).loc[1984:2017]
    k = pd.read_csv(R10 / "kilian_proxy_from_kaenzig.csv")
    ki = annual(pd.Series(k.kilian_proxy_1.to_numpy(),
                          index=pd.to_datetime(k.month.str.replace("M", "-") + "-01"))).loc[1974:2004]
    wti = annual(fred(R10 / "fred_MCOILWTICO.csv"), "mean")
    cpi = annual(fred(ROOT / "data" / "raw" / "v9" / "fred_CPIAUCSL.csv"), "mean")
    drw = np.log(wti / cpi).diff().dropna()
    signs, out = [], {}
    for name, s in (("bh", bh), ("kaenzig", ka), ("kilian", ki)):
        j = pd.concat([drw, s], axis=1, keys=["d", "s"]).dropna()
        f = sm.OLS(j.d, sm.add_constant(j.s)).fit()
        sign = 1.0 if f.params["s"] > 0 else -1.0
        signs.append(dict(shock=name, slope=f.params["s"], t=f.tvalues["s"], n=len(j), years=f"{j.index.min()}-{j.index.max()}",
                          sign_applied=sign))
        out[name] = s * sign
    u = annual(fred(R10 / "fred_UNRATE.csv"), "mean")
    out["lab"] = -u.diff().dropna()
    dly = np.log(nipa("A191RX")).diff()
    w = annual(fred(ROOT / "data" / "raw" / "v9" / "fred_COMPRNFB.csv"), "mean", need=4)
    return out, dly, 100 * np.log(w), pd.DataFrame(signs)


def arr(s):
    return pd.Series(s).reindex(YRS).to_numpy(float)


def design(pi, shock, dly, h, T):
    dpi = np.r_[np.nan, np.diff(pi)]
    y = pi[T + h] - pi[T - 1]
    X = np.column_stack([np.ones(len(T)), shock[T], shock[T - 1], shock[T - 2], dpi[T - 1], dpi[T - 2], dly[T - 1],
                         dly[T - 2]])
    return y, X


def common_T(pis, shocks_, dly, h):
    ok = np.ones(len(YRS), bool)
    pos = np.arange(len(YRS))
    valid = (pos >= 2) & (pos + h < len(YRS))
    T = pos[valid]
    ok = np.ones(len(T), bool)
    for p in pis:
        for s in shocks_:
            y, X = design(p, s, dly, h, T)
            ok &= np.isfinite(y) & np.isfinite(X).all(1)
    return T[ok]


def beta(y, X):
    return np.linalg.lstsq(X, y, rcond=None)[0][1]


def beta_hac(y, X, h):
    f = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": h + 1})
    return float(f.params[1]), float(f.bse[1])


def q_ratio(bA, sA, bC, sC):
    return (bA / sA) / (bC / sC)


def block_idx(n, rng):
    starts = rng.integers(0, n - BLOCK + 1, int(np.ceil(n / BLOCK)))
    return np.concatenate([np.arange(s, s + BLOCK) for s in starts])[:n]


def q_set(P, SH, dly, S, T, h, oil="bh", A="A", idx=None):
    """Q_oil, Q_lab and D on rows T (optionally resampled by idx)"""
    b = {}
    for g in (A, "oil_con", "lab_con"):
        for k in (oil, "lab"):
            y, X = design(P[g], SH[k], dly, h, T)
            if idx is not None:
                y, X = y[idx], X[idx]
            b[g, k] = beta(y, X)
    qo = q_ratio(b[A, oil], S.loc[A, "s_oil"], b["oil_con", oil], S.loc["oil_con", "s_oil"])
    ql = q_ratio(b[A, "lab"], S.loc[A, "s_lab"], b["lab_con", "lab"], S.loc["lab_con", "s_lab"])
    return qo, ql, qo - ql


def run_us():
    pi, ls, S, top = us_groups()
    sh, dly, lnw, signs = shocks()
    pi.round(4).to_csv(OUT / "s1_us_profit_shares.csv")
    ls.round(4).to_csv(OUT / "s1_us_labour_shares.csv")
    S.to_csv(OUT / "s1_us_cost_shares.csv")
    signs.to_csv(OUT / "s1_us_shock_signs.csv", index=False)
    pd.DataFrame(sh).to_csv(OUT / "s1_us_shocks_annual.csv")
    print(S.round(4).to_string(), "\n", signs.to_string(), flush=True)
    P = {g: arr(pi[g]) for g in pi}
    SH = {k: arr(v) for k, v in sh.items()}
    DLY = arr(dly)
    P["real_hourly_comp"] = arr(lnw)

    # ---- LP coefficients (HAC), all groups, all shocks, h = 0..5, on the common sample of the main pair
    lp_rows, q_rows = [], []
    variants = {"main (BH)": ("bh", "A"), "Kaenzig": ("kaenzig", "A"), "Kilian": ("kilian", "A"),
                "A with suppliers (BH)": ("bh", "A_all")}
    for vname, (oil, A) in variants.items():
        for h in H:
            T = common_T([P[A], P["oil_con"], P["lab_con"]], [SH[oil], SH["lab"]], DLY, h)
            for g in [A, "oil_con", "lab_con", "oil_sup", "real_hourly_comp"] + (["A_all"] if A == "A" else []):
                for k in (oil, "lab"):
                    y, X = design(P[g], SH[k], DLY, h, T)
                    m = np.isfinite(y) & np.isfinite(X).all(1)
                    b, se = beta_hac(y[m], X[m], h)
                    lp_rows.append(dict(variant=vname, h=h, group=g, shock=k, beta=b, se_hac=se, n=int(m.sum()),
                                        years=f"{YRS[T][m].min()}-{YRS[T][m].max()}"))
            qo, ql, d = q_set(P, SH, DLY, S, T, h, oil, A)
            rng = np.random.default_rng(470 + h)
            bs = np.array([q_set(P, SH, DLY, S, T, h, oil, A, block_idx(len(T), rng)) for _ in range(B_BOOT)])
            lo, hi = np.nanquantile(bs, 0.05, axis=0), np.nanquantile(bs, 0.95, axis=0)
            q_rows.append(dict(variant=vname, h=h, n=len(T), years=f"{YRS[T].min()}-{YRS[T].max()}", Q_oil=qo,
                               Q_oil_lo=lo[0], Q_oil_hi=hi[0], Q_lab=ql, Q_lab_lo=lo[1], Q_lab_hi=hi[1], D=d, D_lo=lo[2],
                               D_hi=hi[2]))
            print(vname, h, len(T), round(qo, 3), (round(lo[0], 3), round(hi[0], 3)), round(ql, 3),
                  (round(lo[1], 3), round(hi[1], 3)), round(d, 3), flush=True)
    LP = pd.DataFrame(lp_rows)
    Q = pd.DataFrame(q_rows)
    LP.to_csv(OUT / "s1_us_lp.csv", index=False)
    Q.to_csv(OUT / "s1_us_q.csv", index=False)

    # ---- permutation placebo for outcome 49 (main, h = 2)
    T = common_T([P["A"], P["oil_con"], P["lab_con"]], [SH["bh"], SH["lab"]], DLY, H_MAIN)
    d_obs = q_set(P, SH, DLY, S, T, H_MAIN)[2]
    rng = np.random.default_rng(4949)
    perm = []
    for _ in range(N_PERM):
        SHp = dict(SH)
        for k in ("bh", "lab"):
            v = SH[k].copy()
            ok = np.isfinite(v)
            v[ok] = rng.permutation(v[ok])
            SHp[k] = v
        perm.append(q_set(P, SHp, DLY, S, T, H_MAIN)[2])
    perm = np.array(perm)
    pd.Series(perm, name="D_placebo").to_csv(OUT / "s1_us_placebo.csv", index=False)
    p_one = float(np.mean(perm <= d_obs))
    p_two = float(np.mean(np.abs(perm) >= abs(d_obs)))

    m = Q[(Q.variant == "main (BH)") & (Q.h == H_MAIN)].iloc[0]
    out = [dict(outcome="47: Q_oil (BH, h=2)", est=m.Q_oil, ci90_lo=m.Q_oil_lo, ci90_hi=m.Q_oil_hi, n=m.n, theta0=1,
                delta=0.25, direction="<", label=decide(m.Q_oil, m.Q_oil_lo, m.Q_oil_hi, 1, 0.25, "<")),
           dict(outcome="48: Q_labour (h=2), equivalence ~1", est=m.Q_lab, ci90_lo=m.Q_lab_lo, ci90_hi=m.Q_lab_hi, n=m.n,
                theta0=1, delta=0.25, direction="=", label=equiv(m.Q_lab, m.Q_lab_lo, m.Q_lab_hi, 1, 0.25)),
           dict(outcome="49: Q_oil - Q_labour (BH, h=2)", est=m.D, ci90_lo=m.D_lo, ci90_hi=m.D_hi, n=m.n, theta0=0,
                delta=0.25, direction="<", label=decide(m.D, m.D_lo, m.D_hi, 0, 0.25, "<"), p_placebo_one=p_one,
                p_placebo_two=p_two)]
    O = pd.DataFrame(out)
    O.to_csv(OUT / "s1_outcomes_us.csv", index=False)
    print(O.round(4).to_string(), "\n", Q.round(3).to_string(), flush=True)
    return O


# ================================================================ 1.2 right-to-work
ABBR = {"Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA", "Colorado": "CO",
        "Connecticut": "CT", "Delaware": "DE", "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA",
        "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY",
        "Louisiana": "LA", "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN",
        "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH",
        "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY", "North Carolina": "NC", "North Dakota": "ND",
        "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI",
        "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT",
        "Virginia": "VA", "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY"}
TREAT = {"OK": 2002, "IN": 2013, "MI": 2013, "WI": 2016, "WV": 2017, "KY": 2018}
ENACT = {"OK": 2001, "IN": 2012, "MI": 2012, "WI": 2015, "WV": 2016, "KY": 2017}
CONTROL = ["AK", "CA", "CO", "CT", "DE", "HI", "IL", "ME", "MD", "MA", "MN", "MT", "NH", "NJ", "NM", "NY", "OH", "OR",
           "PA", "RI", "VT", "WA", "MO", "DC"]
SIC_EVENTS = {"LA": 1976, "ID": 1985}


def state_panel(sic=False):
    z = zipfile.ZipFile(R10 / ("bea_SAGDP_SIC.zip" if sic else "bea_SAGDP.zip"))
    names = z.namelist()
    vals = {}
    for t, scale in (("SAGDP2", 1000.0), ("SAGDP4", 1.0), ("SAGDP7", 1.0)):
        f = [n for n in names if n.startswith(t + ("S" if sic else "") + "__ALL_AREAS")][0]
        d = pd.read_csv(z.open(f), dtype=str, encoding="latin1")
        d = d[d.LineCode.astype(str).str.strip() == "2"].copy()
        d["st"] = d.GeoName.astype(str).str.replace("*", "", regex=False).str.strip().map(ABBR)
        d = d.dropna(subset=["st"])
        yc = [c for c in d.columns if c.strip().isdigit()]
        m = d.melt(id_vars=["st"], value_vars=yc, var_name="year", value_name="v")
        m["year"] = m.year.astype(int)
        m["v"] = pd.to_numeric(m.v, errors="coerce") * scale
        vals[t] = m.set_index(["st", "year"]).v
    P = pd.DataFrame({"va": vals["SAGDP2"], "comp": vals["SAGDP4"], "gos": vals["SAGDP7"]}).reset_index()
    P["gos_share"] = 100 * P.gos / P.va
    P["comp_share"] = 100 * P.comp / P.va
    return P


def demean2(M, a, b, tol=1e-11, it=5000):
    M = np.asarray(M, float).copy()
    ia, ib = pd.factorize(a)[0], pd.factorize(b)[0]
    na, nb = np.bincount(ia), np.bincount(ib)
    for _ in range(it):
        prev = M.copy()
        for g, n in ((ia, na), (ib, nb)):
            for j in range(M.shape[1]):
                M[:, j] -= (np.bincount(g, M[:, j]) / n)[g]
        if np.abs(M - prev).max() < tol:
            break
    return M


def stack(P, events, controls, window=5):
    parts = []
    for st, E in events.items():
        sub = P[P.st.isin([st] + controls) & P.year.between(E - window, E + window)].copy()
        if st == "MI":
            sub = sub[~((sub.st == "MI") & (sub.year >= 2024))]
        sub["stk"] = st
        sub["D"] = ((sub.st == st) & (sub.year >= E)).astype(float)
        sub["rel"] = np.where(sub.st == st, sub.year - E, np.nan)
        parts.append(sub)
    return pd.concat(parts, ignore_index=True)


def did(S, outcome, B=9999, seed=50):
    S = S.dropna(subset=[outcome])
    u, t = S.st + "_" + S.stk, S.year.astype(str) + "_" + S.stk
    M = demean2(np.column_stack([S[outcome], S.D]), u, t)
    b, se, lo_b, hi_b, p, G = wild_t(M[:, [1]], M[:, 0], S.st.to_numpy(), 0, B=B, seed=seed)
    lo, hi, lab = widest(b, se, G, lo_b, hi_b, 0.5, ">")
    return dict(est=b, se=se, ci90_lo=lo, ci90_hi=hi, wild_lo=lo_b, wild_hi=hi_b, p_wild=p, G=G, n=len(S), label=lab)


def event_study(S, outcome):
    S = S.dropna(subset=[outcome])
    rels = [r for r in range(-5, 6) if r != -1]
    D = np.column_stack([(S.rel == r).astype(float) for r in rels])
    u, t = S.st + "_" + S.stk, S.year.astype(str) + "_" + S.stk
    M = demean2(np.column_stack([S[outcome], D]), u, t)
    y, X = M[:, 0], M[:, 1:]
    keep = X.std(0) > 0
    X = X[:, keep]
    XtX = np.linalg.pinv(X.T @ X)
    b = XtX @ X.T @ y
    e = y - X @ b
    gi = pd.factorize(S.st)[0]
    meat = sum(np.outer(X[gi == g].T @ e[gi == g], X[gi == g].T @ e[gi == g]) for g in range(gi.max() + 1))
    G = gi.max() + 1
    V = XtX @ meat @ XtX * G / (G - 1)
    return pd.DataFrame({"rel": np.array(rels)[keep], "beta": b, "se_cr1": np.sqrt(np.diag(V))})


def run_rtw():
    P = state_panel()
    P.to_csv(OUT / "s1_rtw_state_panel.csv", index=False)
    rows, es_rows = [], []
    for vname, ev in (("first full year (main)", TREAT), ("enactment year", ENACT)):
        S = stack(P, ev, CONTROL)
        for oc in ("gos_share", "comp_share"):
            r = did(S, oc)
            if oc == "comp_share":                      # descriptive; the label rule of outcome 50 does not apply
                r["label"] = "описательно"
            rows.append(dict(variant=vname, outcome=oc, **r))
            print(vname, oc, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
            e = event_study(S, oc)
            e["variant"], e["outcome"] = vname, oc
            es_rows.append(e)
    # per-state effects (descriptive)
    for st, E in TREAT.items():
        S = stack(P, {st: E}, CONTROL)
        r = did(S, "gos_share", B=1999)
        r["label"] = "описательно (один кластер лечения)"
        rows.append(dict(variant=f"single state {st} {E}", outcome="gos_share", **r))
    Ps = state_panel(sic=True)
    Ps.to_csv(OUT / "s1_rtw_state_panel_sic.csv", index=False)
    S = stack(Ps, SIC_EVENTS, CONTROL)
    for oc in ("gos_share", "comp_share"):
        r = did(S, oc)
        r["label"] = "описательно"
        rows.append(dict(variant="SIC 1963-1997: LA 1976, ID 1985 (descriptive)", outcome=oc, **r))
        e = event_study(S, oc)
        e["variant"], e["outcome"] = "SIC", oc
        es_rows.append(e)
    Rr = pd.DataFrame(rows)
    Rr.to_csv(OUT / "s1_rtw_did.csv", index=False)
    pd.concat(es_rows).to_csv(OUT / "s1_rtw_event_study.csv", index=False)
    m = Rr[(Rr.variant == "first full year (main)") & (Rr.outcome == "gos_share")].iloc[0]
    o = dict(outcome="50: RTW effect on state private GOS share, years 0..+5 (pp)", est=m.est, ci90_lo=m.ci90_lo,
             ci90_hi=m.ci90_hi, n=m.n, theta0=0, delta=0.5, direction=">", label=m.label, G=m.G, p_wild=m.p_wild)
    print(Rr.round(4).to_string(), flush=True)
    return pd.DataFrame([o])


def trend_adjusted(S, outcome, B=9999, seed=51):
    """post hoc (journal 4): treated-state linear trend fitted on the pre-window, mean deviation over years 0..+5.
    Regressors: D (coefficient = mean post effect), treated x rel, and (1[rel=r] - 1[rel=0]) for r = 1..5."""
    S = S.dropna(subset=[outcome])
    tr = S.rel.notna().astype(float).to_numpy()
    rel = np.nan_to_num(S.rel.to_numpy())
    cols = [S.D.to_numpy(), tr * rel] + [((S.rel == r).astype(float) - (S.rel == 0).astype(float)).to_numpy()
                                          for r in range(1, 6)]
    u, t = S.st + "_" + S.stk, S.year.astype(str) + "_" + S.stk
    M = demean2(np.column_stack([S[outcome].to_numpy()] + cols), u, t)
    b, se, lo_b, hi_b, p, G = wild_t(M[:, 1:], M[:, 0], S.st.to_numpy(), 0, B=B, seed=seed)
    lo, hi, lab = widest(b, se, G, lo_b, hi_b, 0.5, ">")
    return dict(est=b, se=se, ci90_lo=lo, ci90_hi=hi, p_wild=p, G=G, n=len(S), label=lab)


def run_rtw_trend():
    P = pd.read_csv(OUT / "s1_rtw_state_panel.csv")
    rows = []
    for vname, ev in (("first full year (main)", TREAT), ("enactment year", ENACT)):
        S = stack(P, ev, CONTROL)
        for oc in ("gos_share", "comp_share"):
            r = trend_adjusted(S, oc)
            rows.append(dict(variant=vname, outcome=oc, **r))
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "s1_rtw_trend_adjusted.csv", index=False)
    print(R.round(4).to_string(), flush=True)


# ================================================================ 1.3 synthetic control
DONORS = ["AUT", "BEL", "DNK", "FIN", "FRA", "DEU", "GRC", "IRL", "ITA", "LUX", "NLD", "PRT", "ESP", "SWE", "GBR", "NOR",
          "CHE", "USA", "CAN", "JPN", "AUS"]


def profit_share_ameco():
    rows = []
    for f in sorted(glob.glob(str(ROOT / "data" / "raw" / "v6" / "ameco" / "AMECO*.TXT"))):
        d = pd.read_csv(f, sep=";", dtype=str, encoding="latin1")
        d = d[d.CODE.str.endswith(".1.0.0.0.ALCD2", na=False)]
        for _, r in d.iterrows():
            for y in [c for c in d.columns if c.strip().isdigit()]:
                v = pd.to_numeric(r[y], errors="coerce")
                if np.isfinite(v):
                    rows.append((r.CODE.split(".")[0], int(y), v))
    a = pd.DataFrame(rows, columns=["geo", "year", "v"]).drop_duplicates(["geo", "year"])
    P = a.pivot(index="year", columns="geo", values="v")
    return 100 - P.loc[:2024]


def sc_weights(y1, Y0):
    k = Y0.shape[1]
    r = optimize.minimize(lambda w: np.sum((y1 - Y0 @ w) ** 2), np.full(k, 1 / k), bounds=[(0, 1)] * k,
                          constraints=({"type": "eq", "fun": lambda w: w.sum() - 1},), method="SLSQP",
                          options={"maxiter": 2000, "ftol": 1e-12})
    return r.x


def synth(P, treated, pre, post):
    pre, post = list(pre), list(post)
    D = P.reindex(pre + post)[[treated] + [d for d in DONORS if d != treated]].dropna(axis=1)
    pool = [u for u in D.columns if u != treated]

    def one(t, pool_):
        w = sc_weights(D.loc[pre, t].to_numpy(), D.loc[pre, pool_].to_numpy())
        gap = D[t] - D[pool_].to_numpy() @ w
        r_pre = np.sqrt(np.mean(gap.loc[pre] ** 2))
        r_post = np.sqrt(np.mean(gap.loc[post] ** 2))
        return gap, pd.Series(w, index=pool_), r_post / r_pre
    gap, w, ratio = one(treated, pool)
    ratios = [ratio] + [one(d, [u for u in pool if u != d])[2] for d in pool]
    p = float(np.mean(np.array(ratios) >= ratio))
    return gap, w, ratio, p, len(ratios)


SC_CASES = [("Hartz IV 2005 (outcome 51)", "DEU", range(1991, 2005), range(2005, 2013)),
            ("Hartz I-II 2003 (variant)", "DEU", range(1991, 2003), range(2003, 2013)),
            ("Germany minimum wage 2015", "DEU", range(2005, 2015), range(2015, 2020)),
            ("UK NMW 1999", "GBR", range(1989, 1999), range(1999, 2004)),
            ("UK NLW 2016", "GBR", range(2006, 2016), range(2016, 2020)),
            ("Spain minimum wage +22% 2019", "ESP", range(2009, 2019), range(2019, 2020))]


def run_sc():
    P = profit_share_ameco()
    rows, gaps = [], []
    for name, tr, pre, post in SC_CASES:
        gap, w, ratio, p, n = synth(P, tr, pre, post)
        g = float(gap.loc[list(post)].mean())
        rows.append(dict(case=name, treated=tr, pre=f"{min(pre)}-{max(pre)}", post=f"{min(post)}-{max(post)}",
                         post_gap_pp=g, rmspe_pre=float(np.sqrt(np.mean(gap.loc[list(pre)] ** 2))), ratio=ratio,
                         p_rank=p, n_units=n, weights="; ".join(f"{k} {v:.2f}" for k, v in w[w > 0.01].items())))
        gg = gap.rename("gap").reset_index().rename(columns={"index": "year"})
        gg["case"] = name
        gaps.append(gg)
        print(rows[-1], flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "s1_sc.csv", index=False)
    pd.concat(gaps).to_csv(OUT / "s1_sc_gaps.csv", index=False)
    m = R.iloc[0]
    g, p = m.post_gap_pp, m.p_rank
    lab = CONF if (g > 1 and p <= 0.10) else REF if ((g < 0 and p <= 0.10) or abs(g) <= 0.5) else UNINF
    return pd.DataFrame([dict(outcome="51: Hartz IV synthetic-control gap in profit share 2005-2012 (pp)", est=g,
                              ci90_lo=np.nan, ci90_hi=np.nan, n=m.n_units, theta0=0, delta=0.5, direction=">",
                              label=lab, p_rank=p)])


# ================================================================ 1.4 gas shock
def run_gas():
    from .. import figaro
    b = es_json(R10 / "es_namq_10_a10_B1G.json")
    d = es_json(R10 / "es_namq_10_a10_D1.json")
    x = pd.concat([b.assign(item="B1G"), d.assign(item="D1")])
    x = x[x.nace_r2.isin(["TOTAL", "B-E", "C"])]
    W = x.pivot_table(index=["geo", "time"], columns=["item", "nace_r2"], values="value")
    pi = pd.DataFrame(index=W.index)
    pi["A"] = 100 * (W["B1G", "TOTAL"] - W["D1", "TOTAL"]) / W["B1G", "TOTAL"]
    pi["C"] = 100 * (W["B1G", "C"] - W["D1", "C"]) / W["B1G", "C"]
    sup_va = W["B1G", "B-E"] - W["B1G", "C"]
    pi["sup"] = 100 * (sup_va - (W["D1", "B-E"] - W["D1", "C"])) / sup_va
    pi = pi.reset_index()
    post = [f"{y}-Q{q}" for y in (2022,) for q in range(1, 5)] + ["2023-Q1", "2023-Q2"]
    pre = [f"2019-Q{q}" for q in range(1, 5)]
    f = figaro.load(2019)
    ind, cty = list(f["industries"]), list(f["countries"])
    N = len(ind)
    Z, V = f["Z"], f["V"][:3].sum(0)
    rows_bd = [c * N + ind.index(p) for c in range(len(cty)) for p in ("B", "D35")]
    out = []
    for g, s in pi.groupby("geo"):
        s = s.set_index("time")[["A", "C", "sup"]]
        if not (set(pre) | set(post)) <= set(s.index) or s.loc[pre + post].isna().any().any():
            continue
        dlt = s.loc[post].mean() - s.loc[pre].mean()
        rec = dict(geo=g, d_pi_A=dlt.A, d_pi_C=dlt.C, d_pi_sup=dlt.sup)
        if g in cty:
            gi = cty.index(g)
            colC = [gi * N + j for j, c in enumerate(ind) if c.startswith("C")]
            colA = [gi * N + j for j in range(N)]
            rec["s_C"] = Z[np.ix_(rows_bd, colC)].sum() / V[colC].sum()
            rec["s_A"] = Z[np.ix_(rows_bd, colA)].sum() / V[colA].sum()
            rec["Q_gas"] = q_ratio(dlt.A, rec["s_A"], dlt.C, rec["s_C"])
        out.append(rec)
    R = pd.DataFrame(out)
    R.to_csv(OUT / "s1_gas.csv", index=False)
    print(R.round(3).to_string(), "\nmedians:", R.median(numeric_only=True).round(3).to_dict(), flush=True)


# ================================================================ 1.5 Section 232
def run_steel():
    v = va_components()
    groups = {"primary metals (supplier)": ["primary metals"],
              "consumers: fabricated metals, machinery, motor vehicles": ["fabricated metal products", "machinery",
                                                                          "motor vehicles, bodies and trailers, and parts"],
              "A: private industries": ["private industries"]}
    rows = []
    for name, ns in groups.items():
        f = sum(v[n][["comp", "tax", "gos"]] for n in ns)
        pi = 100 * f.gos / f.sum(axis=1)
        rows.append(dict(group=name, pi_2015_2017=pi.loc[2015:2017].mean(), pi_2018_2019=pi.loc[2018:2019].mean(),
                         change=pi.loc[2018:2019].mean() - pi.loc[2015:2017].mean()))
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "s1_steel.csv", index=False)
    print(R.round(3).to_string(), flush=True)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    parts = sys.argv[1:] or ["us", "rtw", "sc", "gas", "steel"]
    outs = []
    if "us" in parts:
        outs.append(run_us())
    if "rtw" in parts:
        outs.append(run_rtw())
    if "sc" in parts:
        outs.append(run_sc())
    if "gas" in parts:
        run_gas()
    if "steel" in parts:
        run_steel()
    if "rtw_trend" in parts:
        run_rtw_trend()
    if outs:
        O = pd.concat(outs, ignore_index=True)
        prev = OUT / "s1_outcomes.csv"
        if prev.exists() and set(parts) != {"us", "rtw", "sc", "gas", "steel"}:
            old = pd.read_csv(prev)
            O = pd.concat([old[~old.outcome.str[:3].isin(O.outcome.str[:3])], O], ignore_index=True)
        O.sort_values("outcome").to_csv(prev, index=False)
        print(O.round(4).to_string())
