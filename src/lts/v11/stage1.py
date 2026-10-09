"""Iteration 11, stage 1: the law of value as regulator of the allocation of labour (pre_registration_v11.md, stage 1).

Signals (country x industry x year t-1, WIOD 2016 national systems; open system, fixed capital, imports at price):
  S_v  = -ln z_L     (price / value; labour basis; sum z x = sum x)
  S_pp = -ln p_pp    (price / price of production; uniform r = actual, actual wages; flow-based mark-up)
  variants: S_ppU (uniform wage), S_r (stock profit rate (CAP - delta K)/K minus the country-year mean)
  placebos: flat vector (l = 1), 200 permutations of l fixed per country (seed 11)
Responses from t to t+h (t = signal year + 1): d ln hours (SEA), d ln EMP, d ln GO_QI, d ln self-employed,
  investment (EU KLEMS 2025, ln mean I over t+1..t+h minus ln mean I over t-2..t).
Outcomes 64-67; test 1 (joint and both orthogonalisation orders); placebos; variants; FIGARO replication with
reduction of complex labour (edu_years, edu_train).

Usage: PYTHONPATH=src python -P -m lts.v11.stage1 [signals|figaro_hours|estimate|all]
Outputs: results/v11/s1_*.csv, s1_signals.parquet, s1_perm.parquet
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy import stats

from ..revisit import wiod
from ..v3.placebo_disp import Setup
from ..v6.series import ROOT
from ..v8.rule import decide

OUT = ROOT / "results" / "v11"
OUT.mkdir(parents=True, exist_ok=True)
R9B = ROOT / "data" / "raw" / "v9b"
YEARS = range(2000, 2015)
NPERM = 200
EXCL = {"B", "K64", "K65", "K66", "L68", "O84", "P85", "Q", "T", "U"}
AGRI = {"A01", "A02", "A03"}
SE_HI, CORP_LO = 0.25, 0.10                     # main thresholds; variant 0.40 / 0.05
WEBB = np.array([-np.sqrt(1.5), -1, -np.sqrt(0.5), np.sqrt(0.5), 1, np.sqrt(1.5)])
KLEMS_MAP = {"A01": "A", "A02": "A", "A03": "A", "C10-C12": "C10-C12", "C13-C15": "C13-C15", "C16": "C16-C18",
             "C17": "C16-C18", "C18": "C16-C18", "C19": "C19", "C20": "C20", "C21": "C21", "C22": "C22-C23",
             "C23": "C22-C23", "C24": "C24-C25", "C25": "C24-C25", "C26": "C26", "C27": "C27", "C28": "C28",
             "C29": "C29-C30", "C30": "C29-C30", "C31_C32": "C31-C33", "C33": "C31-C33", "D35": "D", "E36": "E",
             "E37-E39": "E", "F": "F", "G45": "G45", "G46": "G46", "G47": "G47", "H49": "H49", "H50": "H50",
             "H51": "H51", "H52": "H52", "H53": "H53", "I": "I", "J58": "J58-J60", "J59_J60": "J58-J60", "J61": "J61",
             "J62_J63": "J62-J63", "M69_M70": "M", "M71": "M", "M72": "M", "M73": "M", "M74_M75": "M", "N": "N",
             "R_S": "R-S"}
KLEMS_ISO = {"AUT": "AT", "BEL": "BE", "BGR": "BG", "CYP": "CY", "CZE": "CZ", "DEU": "DE", "DNK": "DK", "EST": "EE",
             "GRC": "EL", "ESP": "ES", "FIN": "FI", "FRA": "FR", "HRV": "HR", "HUN": "HU", "IRL": "IE", "ITA": "IT",
             "JPN": "JP", "LTU": "LT", "LUX": "LU", "LVA": "LV", "MLT": "MT", "NLD": "NL", "POL": "PL", "PRT": "PT",
             "ROU": "RO", "SWE": "SE", "SVN": "SI", "SVK": "SK", "GBR": "UK", "USA": "US"}


# ================================================================ inference
def demean(M, groups, tol=1e-10, maxit=2000, w=None):
    """Two-way (or k-way) within transformation by alternating projections (weighted means if w); M: (n, p)."""
    M = np.array(M, float, copy=True)
    w = np.ones(len(M)) if w is None else np.asarray(w, float)
    codes = [pd.factorize(g)[0] for g in groups]
    counts = [np.bincount(c, weights=w) for c in codes]
    for _ in range(maxit):
        delta = 0.0
        for c, n in zip(codes, counts):
            means = np.vstack([np.bincount(c, weights=w * M[:, k]) for k in range(M.shape[1])]).T / n[:, None]
            M -= means[c]
            delta = max(delta, np.abs(means).max())
        if delta < tol:
            break
    return M


def wild_fast(X, y, groups, j, B=9999, seed=7):
    """Same estimator as lts.v9b.stage45.wild_t (CR1, Webb weights, unrestricted bootstrap-t, same draw order),
    vectorised over replications. Returns b_j, se_j, 90% bootstrap-t CI, symmetric p, G."""
    gi = pd.factorize(groups)[0]
    G = gi.max() + 1
    XtX = np.linalg.inv(X.T @ X)
    b = XtX @ X.T @ y
    e = y - X @ b
    C = np.zeros((G, len(y)))
    C[gi, np.arange(len(y))] = 1.0
    Q = C @ (X * e[:, None])                                   # G x k cluster scores
    meat = Q.T @ Q
    V = XtX @ meat @ XtX * G / (G - 1)
    se = np.sqrt(np.diag(V))
    v = XtX[j]
    a = X @ v
    P = C @ (a * e)                                            # G
    R = C @ (a[:, None] * X)                                   # G x k
    rng = np.random.default_rng(seed)
    W = np.vstack([WEBB[rng.integers(0, 6, G)] for _ in range(B)])      # B x G, same order as wild_t
    D = W @ Q @ XtX                                            # B x k: b* - b
    Sv = W * P[None] - D @ R.T                                 # B x G
    se_b = np.sqrt((Sv ** 2).sum(1) * G / (G - 1))
    ts = D[:, j] / se_b
    lo, hi = b[j] - np.quantile(ts, 0.95) * se[j], b[j] - np.quantile(ts, 0.05) * se[j]
    p = float(np.mean(np.abs(ts) >= abs(b[j] / se[j])))
    return float(b[j]), float(se[j]), float(lo), float(hi), p, int(G)


def widest(b, se, G, lo_b, hi_b):
    tq = stats.t.ppf(0.95, G - 1)
    return min(b - tq * se, lo_b), max(b + tq * se, hi_b)


def reg(d, y, xs, j, fe=("ind", "cy"), cl="ind", B=9999, scale=1.0, boot=True, w=None):
    """OLS (WLS if w names a weight column) of y on xs after absorbing fe; coefficient xs[j] times scale
    (scale < 0 flips the sign) with the wider of t(G-1) and the wild cluster bootstrap-t CI."""
    d = d.dropna(subset=[y] + xs + ([w] if w else []))
    ww = d[w].to_numpy(float) if w else None
    M = demean(d[[y] + xs].to_numpy(float), [d[f].to_numpy() for f in fe], w=ww)
    if w:
        M = M * np.sqrt(ww)[:, None]
    yy, X = M[:, 0], M[:, 1:]
    if not boot:
        bb = np.linalg.lstsq(X, yy, rcond=None)[0]
        return dict(est=float(bb[j] * scale), n=len(d))
    b, se, lo_b, hi_b, p, G = wild_fast(X, yy, d[cl].to_numpy(), j, B=B)
    lo, hi = widest(b, se, G, lo_b, hi_b)
    lo, hi = sorted((lo * scale, hi * scale))
    return dict(est=b * scale, se=se * abs(scale), ci90_lo=lo, ci90_hi=hi, p_wild=p, G=G, n=len(d),
                mde=2.8 * se * abs(scale))


def labelled(row, delta, direction, theta0=0.0):
    row.update(theta0=theta0, delta=delta, direction=direction,
               label=decide(row["est"], row["ci90_lo"], row["ci90_hi"], theta0, delta, direction))
    return row


# ================================================================ signals (WIOD)
def perms_for(c, n):
    rng = np.random.default_rng(11 + sum(map(ord, c)))
    return np.vstack([rng.permutation(n) for _ in range(NPERM)])


def signals():
    countries = wiod.eligible()
    s = wiod.sea()
    sea = s[s.variable.isin(["EMP", "EMPE", "GO", "GO_QI", "H_EMPE", "CAP", "K"])].pivot_table(
        index=["country", "year", "code"], columns="variable", values="v")
    rows, prow, skipped = [], [], []
    P = {c: perms_for(c, wiod.N) for c in countries}
    for y in YEARS:
        for c in countries:
            e = wiod.economy(c, y)
            e.hours = np.where(np.isfinite(e.hours), e.hours, 0.0)   # EMPE = 0 in SEA -> hours missing (journal 2)
            try:
                S = Setup(e)
                r = e.actual_profit_rate(True)
                ppA = e.prices_of_production(r, True, "price", "actual")
                ppU = e.prices_of_production(r, True, "price", "uniform")
            except ValueError as exc:                             # unproductive system: country-year dropped
                skipped.append(dict(country=c, year=y, reason=str(exc)[:100]))
                print("skip", c, y, exc, flush=True)
                continue
            l = e.l()
            zL = S.z(l[None])[0]
            if y == 2000 and c == countries[0]:                    # consistency with Economy.basis_ratios
                assert np.allclose(zL, e.basis_ratios("L", True, False, "price", l=l))
            zF = S.z(np.ones((1, e.n)))[0]
            zP = S.z(l[P[c]])
            K, CAP = e.meta["K"], e.meta["CAP"]
            dl = wiod.delta(c, y)
            sv = sea.loc[(c, y)].reindex(e.labels)
            rows.append(pd.DataFrame(dict(
                country=c, year=y, ind=e.labels, x=e.x, hours=e.hours, zL=zL, ppA=ppA, ppU=ppU, zF=zF, r_act=r,
                K=K, CAP=CAP, delta=dl,
                EMP=sv.EMP.to_numpy(), EMPE=sv.EMPE.to_numpy(), GO=sv.GO.to_numpy(), GO_QI=sv.GO_QI.to_numpy())))
            prow.append(pd.DataFrame(zP.T.astype(np.float32), columns=[f"p{k:03d}" for k in range(NPERM)]))
            print("sig", y, c, flush=True)
        wiod.wiot.cache_clear()
    sig = pd.concat(rows, ignore_index=True)
    perm = pd.concat(prow, ignore_index=True)
    sig.to_parquet(OUT / "s1_signals.parquet")
    perm.to_parquet(OUT / "s1_perm.parquet")
    pd.DataFrame(skipped).to_csv(OUT / "s1_skipped.csv", index=False)
    return sig, perm


# ================================================================ panel
def sigma(sig, keys=("country", "ind")):
    b = sig[sig.year.between(2000, 2002)]
    g = b.groupby(list(keys))[["EMP", "EMPE"]].sum(min_count=1)
    out = (g.EMP - g.EMPE) / g.EMP
    return out.where(g.EMP > 0).rename("sigma")


def prepare(sig, perm=None):
    d = sig.copy()
    ok = (d.zL > 0) & (d.ppA > 0) & (d.ppU > 0) & (d.zF > 0) & (d.x > 0) & (d.hours > 0) & (d.EMP > 0)
    d["Sv"] = np.where(ok, -np.log(d.zL.where(d.zL > 0)), np.nan)
    d["Spp"] = np.where(ok, -np.log(d.ppA.where(d.ppA > 0)), np.nan)
    d["SppU"] = np.where(ok, -np.log(d.ppU.where(d.ppU > 0)), np.nan)
    d["Sflat"] = np.where(ok, -np.log(d.zF.where(d.zF > 0)), np.nan)
    d["rj"] = np.where(d.K > 0, (d.CAP - d.delta * d.K) / d.K.where(d.K > 0), np.nan)
    d["SE"] = d.EMP - d.EMPE
    d["lint"] = np.log((d.hours / d.GO).where((d.hours > 0) & (d.GO > 0)))
    for c in ("Sv", "Spp", "SppU", "Sflat", "rj", "lint"):                  # journal 2: non-finite -> missing
        d[c] = d[c].where(np.isfinite(d[c]))
    d = d.join(sigma(sig), on=["country", "ind"])
    if perm is not None:
        Z = perm.to_numpy(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            SP = np.where((Z > 0) & ok.to_numpy()[:, None], -np.log(np.where(Z > 0, Z, np.nan)), np.nan)
        return d, SP
    return d, None


def build_panel(d, h, excl=EXCL, s_years=None):
    """Rows: signal year s; response from t = s+1 to t+h; controls at s and s-1."""
    d = d[~d.ind.isin(excl)].copy()
    d["_row"] = d.index.to_numpy()                          # row in s1_perm (prepare keeps the order)
    k = d.set_index(["country", "ind", "year"])
    lag = lambda col, j: k[col].reindex(pd.MultiIndex.from_arrays([k.index.get_level_values(0),   # noqa: E731
                                                                   k.index.get_level_values(1),
                                                                   k.index.get_level_values(2) + j])).to_numpy()
    p = k.reset_index()
    for col in ("hours", "EMP", "GO_QI", "SE"):
        a, b = lag(col, 1), lag(col, 1 + h)
        with np.errstate(divide="ignore", invalid="ignore"):
            p[f"R_{col}"] = np.where((a > 0) & (b > 0), np.log(b) - np.log(np.where(a > 0, a, np.nan)), np.nan)
    p["lint_lag"] = p.lint
    p["dlint"] = p.lint - lag("lint", -1)
    # stock profit-rate signal relative to the country-year mean over included industries
    g = p.assign(num=(p.CAP - p.delta * p.K).where(p.K > 0), den=p.K.where(p.K > 0)).groupby(["country", "year"])
    rbar = g.num.transform("sum") / g.den.transform("sum")
    p["Sr"] = p.rj - rbar
    p["cy"] = p.country + "_" + p.year.astype(str)
    if s_years is not None:
        p = p[p.year.isin(list(s_years))]
    return p


def std(p, cols):
    p = p.copy()
    for c in cols:
        p[c] = p[c] / p[c].std()
    return p


# ================================================================ estimations
CTRL = ["lint_lag", "dlint"]


def joint(p, y, sv="Sv", spp="Spp", ctrl=CTRL, fe=("ind", "cy"), B=9999, boot=True):
    """b_v - b_pp as one coefficient: W = Sv - Spp, U = Sv + Spp; coefficient on W = (b_v - b_pp)/2."""
    p = p.dropna(subset=[y, sv, spp] + ctrl)
    p = std(p, [sv, spp])
    p = p.assign(U=p[sv] + p[spp], W=p[sv] - p[spp])
    return reg(p, y, ["W", "U"] + ctrl, 0, fe=fe, B=B, scale=2.0, boot=boot)


def coefs(p, y, xs, ctrl=CTRL, fe=("ind", "cy"), B=9999, boot=True):
    p = std(p.dropna(subset=[y] + xs + ctrl), xs)
    return {x: reg(p, y, xs + ctrl, i, fe=fe, B=B, boot=boot) for i, x in enumerate(xs)}


def groups(p, hi=SE_HI, lo=CORP_LO):
    g = pd.Series(np.where(p.sigma >= hi, "se", np.where(p.sigma < lo, "corp", "mid")), index=p.index)
    return g.where(p.sigma.notna(), "na")


def dd(p, y, sv="Sv", spp="Spp", ctrl=CTRL, fe=("ind", "cy"), hi=SE_HI, lo=CORP_LO, B=9999, boot=True,
       which="dd", w=None):
    """DD = (b_v - b_pp)_se - (b_v - b_pp)_corp = 2 * coef(W x D); corporate (b_pp - b_v) = -2 * coef(W)."""
    p = p.assign(grp=groups(p, hi, lo))
    p = p[p.grp.isin(["se", "corp"])].dropna(subset=[y, sv, spp] + ctrl)
    p = std(p, [sv, spp])
    D = (p.grp == "se").astype(float)
    p = p.assign(U=p[sv] + p[spp], W=p[sv] - p[spp], D=D)
    p = p.assign(WD=p.W * D, UD=p.U * D, **{f"{c}D": p[c] * D for c in ctrl})
    xs = ["WD", "W", "UD", "U"] + ([] if "ci" in fe else ["D"]) + ctrl + [f"{c}D" for c in ctrl]   # D is fixed
    if which == "dd":                                                                               # within ci
        r = reg(p, y, xs, 0, fe=fe, B=B, scale=2.0, boot=boot, w=w)
    else:                                                       # corporate b_pp - b_v
        r = reg(p, y, xs, 1, fe=fe, B=B, scale=-2.0, boot=boot, w=w)
    if boot:
        r.update(n_se=int((p.grp == "se").sum()), n_corp=int((p.grp == "corp").sum()),
                 ind_se=int(p[p.grp == "se"].ind.nunique()), ind_corp=int(p[p.grp == "corp"].ind.nunique()))
    return r


def dd_cont(p, y, sv="Sv", spp="Spp", ctrl=CTRL, fe=("ind", "cy"), B=9999):
    """continuous interaction: (b_v - b_pp) slope in sigma, scaled to the 0.10 -> 0.25 contrast"""
    p = p.dropna(subset=[y, sv, spp, "sigma"] + ctrl)
    p = std(p, [sv, spp])
    s = p.sigma - p.sigma.mean()
    p = p.assign(U=p[sv] + p[spp], W=p[sv] - p[spp], s=s)
    p = p.assign(Ws=p.W * s, Us=p.U * s, **{f"{c}s": p[c] * s for c in ctrl})
    xs = ["Ws", "W", "Us", "U", "s"] + ctrl + [f"{c}s" for c in ctrl]
    return reg(p, y, xs, 0, fe=fe, B=B, scale=2.0 * (SE_HI - CORP_LO))


def orth(p, y, first, second, ctrl=CTRL, fe=("ind", "cy"), B=9999):
    """second residualised on first within the fixed effects; both coefficients reported"""
    p = std(p.dropna(subset=[y, first, second] + ctrl), [first, second])
    M = demean(p[[first, second]].to_numpy(float), [p[f].to_numpy() for f in fe])
    beta = (M[:, 0] @ M[:, 1]) / (M[:, 0] @ M[:, 0])
    p = p.assign(**{second + "_perp": p[second] - beta * p[first]})
    xs = [first, second + "_perp"]
    return {x: reg(p, y, xs + ctrl, i, fe=fe, B=B) for i, x in enumerate(xs)}


# ================================================================ KLEMS (investment, hours)
def klems():
    ca = pd.read_csv(R9B / "klems2025_capital_accounts.csv", usecols=["nace_r2_code", "geo_code", "year", "I_GFCF"],
                     low_memory=False)
    na = pd.read_csv(R9B / "klems2025_national_accounts.csv", usecols=["nace_r2_code", "geo_code", "year", "H_EMP",
                                                                         "EMP"], low_memory=False)
    k = na.merge(ca, on=["nace_r2_code", "geo_code", "year"], how="outer")
    inv = {v: c for c, v in KLEMS_ISO.items()}
    k = k[k.geo_code.isin(inv)].assign(country=lambda x: x.geo_code.map(inv))
    return k.rename(columns={"nace_r2_code": "kind"})


def klems_panel(d, h=3, s_years=range(2001, 2014)):
    """WIOD signals aggregated to KLEMS industries (output-weighted), sigma and controls aggregated."""
    d = d[d.ind.isin(KLEMS_MAP) & d.country.isin(KLEMS_ISO)].copy()
    d["kind"] = d.ind.map(KLEMS_MAP)
    d["w"] = d.x.where(d.Sv.notna() & d.Spp.notna())
    for c in ("Sv", "Spp"):
        d[c + "_w"] = d[c] * d.w
    g = d.groupby(["country", "kind", "year"])
    a = pd.DataFrame({"Sv": g.Sv_w.sum(min_count=1) / g.w.sum(min_count=1),
                      "Spp": g.Spp_w.sum(min_count=1) / g.w.sum(min_count=1),
                      "hours": g.hours.sum(min_count=1), "GO": g.GO.sum(min_count=1),
                      "n_ok": g.Sv.count(), "n_all": g.ind.count()}).reset_index()
    a.loc[a.n_ok < a.n_all, ["Sv", "Spp"]] = np.nan              # all member industries must have signals
    b = d[d.year.between(2000, 2002)].groupby(["country", "kind"])[["EMP", "EMPE"]].sum(min_count=1)
    a = a.join(((b.EMP - b.EMPE) / b.EMP).where(b.EMP > 0).rename("sigma"), on=["country", "kind"])
    a["lint"] = np.log((a.hours / a.GO).where((a.hours > 0) & (a.GO > 0)))
    K = klems().set_index(["country", "kind", "year"])
    a = a.join(K[["I_GFCF", "H_EMP"]], on=["country", "kind", "year"])
    k = K.copy()

    def get(col, years):
        idx = pd.MultiIndex.from_arrays([a.country, a.kind, years])
        return k[col].reindex(idx).to_numpy()
    t = a.year + 1
    post = np.nanmean(np.vstack([get("I_GFCF", t + j) for j in range(1, h + 1)]), axis=0)
    pre = np.vstack([get("I_GFCF", t - j) for j in range(0, 3)])
    pre = np.where(np.isnan(pre).any(0), np.nan, pre.mean(0))
    postall = np.vstack([get("I_GFCF", t + j) for j in range(1, h + 1)])
    post = np.where(np.isnan(postall).any(0), np.nan, post)
    with np.errstate(divide="ignore", invalid="ignore"):
        a["R_I"] = np.where((post > 0) & (pre > 0), np.log(post) - np.log(pre), np.nan)
        h0, h1 = get("H_EMP", t), get("H_EMP", t + h)
        a["R_HK"] = np.where((h0 > 0) & (h1 > 0), np.log(h1) - np.log(h0), np.nan)
    ka = a.set_index(["country", "kind", "year"])
    a["dlint"] = a.lint - ka.lint.reindex(pd.MultiIndex.from_arrays([a.country, a.kind, a.year - 1])).to_numpy()
    a["lint_lag"] = a.lint
    a["ind"] = a.kind
    a["cy"] = a.country + "_" + a.year.astype(str)
    return a[a.year.isin(list(s_years))]


# ================================================================ FIGARO replication
FIG_COUNTRIES = ("USA", "DEU", "MEX", "FRA", "ITA", "ESP", "NLD", "AUT", "POL", "CZE", "KOR", "JPN", "GBR")
FIG_TO_WIOD = {"C10-12": "C10-C12", "C13-15": "C13-C15", "C31_32": "C31_C32", "E37-39": "E37-E39",
               "J59_60": "J59_J60", "J62_63": "J62_J63", "L": "L68", "L68A": "L68", "L68B": "L68",
               "M69_70": "M69_M70", "M74_75": "M74_M75", "N77": "N", "N78": "N", "N79": "N", "N80-82": "N",
               "Q86": "Q", "Q87_88": "Q", "R90-92": "R_S", "R93": "R_S", "S94": "R_S", "S95": "R_S", "S96": "R_S"}
FIG_EXCL = {"B", "K64", "K65", "K66", "L", "L68A", "L68B", "O84", "P85", "Q86", "Q87_88", "T", "U"}


def fig_wiod_code(lab):
    parts = lab.split("+")
    return [FIG_TO_WIOD.get(x, x) for x in parts]


def figaro_hours():
    from ..economy import build
    rows = []
    for c in FIG_COUNTRIES:
        for y in range(2010, 2023):
            try:
                e, _ = build(c, y)
            except Exception as exc:                             # noqa: BLE001
                print("fig skip", c, y, exc, flush=True)
                continue
            rows.append(pd.DataFrame(dict(country=c, year=y, ind=e.labels, x=e.x, hours=e.hours,
                                          persons=e.meta["persons"])))
            print("fig", c, y, flush=True)
    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(OUT / "s1_figaro_hours.parquet")
    return out


def figaro_panel(sig_wiod, h=3, s_years=range(2011, 2019)):
    H = pd.read_parquet(OUT / "s1_figaro_hours.parquet")
    parts = []
    for f in ("main", "robust"):
        r = pd.read_parquet(ROOT / "results" / "tables" / f"ratios_long_{f}.parquet")
        r = r[(r.capital.astype(str) == "True") & (r.closed.astype(str) == "False") & (r.imports == "price")]
        keep = r.basis.isin(["labour", "pp_actual_wage"]) | r.basis.str.startswith("placebo_perm_")
        parts.append(r[keep & ((r.labour == "hours") | (r.basis == "labour"))])
    r = pd.concat(parts)
    r = r[r.country.isin(FIG_COUNTRIES)]
    r["col"] = np.where(r.basis == "labour", "z_" + r.labour, r.basis)
    w = r.pivot_table(index=["country", "year", "industry"], columns="col", values="z").reset_index()
    w = w.rename(columns={"industry": "ind"}).merge(H, on=["country", "year", "ind"], how="left")
    sg = sigma(sig_wiod)
    emp = sig_wiod[sig_wiod.year.between(2000, 2002)].groupby(["country", "ind"]).EMP.sum()

    def sig_for(c, lab):
        codes = fig_wiod_code(lab)
        s = sg.reindex([(c, k) for k in codes])
        e = emp.reindex([(c, k) for k in codes])
        ok = s.notna().to_numpy() & e.notna().to_numpy()
        return float(np.average(s[ok], weights=e[ok])) if ok.any() and e[ok].sum() > 0 else np.nan
    w["sigma"] = [sig_for(c, lab) for c, lab in zip(w.country, w.ind)]
    excl = w.ind.map(lambda lab: any(x in FIG_EXCL for x in lab.split("+")))
    w = w[~excl]
    for col in [c for c in w.columns if c.startswith("z_") or c == "pp_actual_wage" or c.startswith("placebo_perm_")]:
        w["S" + col] = -np.log(w[col].where(w[col] > 0))
    w["lint"] = np.log((w.hours / w.x).where((w.hours > 0) & (w.x > 0)))
    k = w.set_index(["country", "ind", "year"])

    def lag(col, j):
        return k[col].reindex(pd.MultiIndex.from_arrays([w.country, w.ind, w.year + j])).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        for col in ("hours", "persons"):
            a, b = lag(col, 1), lag(col, 1 + h)
            w[f"R_{col}"] = np.where((a > 0) & (b > 0), np.log(b) - np.log(np.where(a > 0, a, np.nan)), np.nan)
    w["lint_lag"] = w.lint
    w["dlint"] = w.lint - lag("lint", -1)
    w["cy"] = w.country + "_" + w.year.astype(str)
    return w[w.year.isin(list(s_years))]


# ================================================================ run
def row(name, r, **extra):
    return dict(outcome=name, **r, **extra)


def estimate():
    sig = pd.read_parquet(OUT / "s1_signals.parquet")
    perm = pd.read_parquet(OUT / "s1_perm.parquet")
    d, SP = prepare(sig, perm)
    d = d.reset_index(drop=True)
    out, desc = [], []
    P3 = build_panel(d, 3, s_years=range(2001, 2011))
    print("panel h=3:", len(P3), "rows;", P3.dropna(subset=["R_hours", "Sv", "Spp"] + CTRL).shape[0], "usable",
          flush=True)
    gcount = groups(P3).value_counts()
    print(gcount.to_string(), flush=True)

    # ---- outcomes 64-67
    r64 = labelled(dd(P3, "R_hours"), 0.02, ">")
    out.append(row("64 DD (b_v - b_pp)_self-employed - (b_v - b_pp)_corporate, hours, h = 3", r64))
    r65 = labelled(dd(P3, "R_hours", which="corp"), 0.02, ">")
    out.append(row("65 corporate (b_pp - b_v), hours, h = 3", r65))
    pse = P3[P3.sigma >= CORP_LO]
    r66 = labelled(joint(pse, "R_SE"), 0.05, ">")
    out.append(row("66 self-employed numbers (b_v - b_pp), cells sigma >= 0.10, h = 3", r66))
    K3 = klems_panel(d, 3)
    kc = K3[K3.sigma < CORP_LO]
    r67 = joint(kc, "R_I")                                       # = b_v - b_pp; flipped to b_pp - b_v
    r67 = labelled(dict(r67, est=-r67["est"], ci90_lo=-r67["ci90_hi"], ci90_hi=-r67["ci90_lo"]), 0.05, ">")
    out.append(row("67 investment (KLEMS), corporate (b_pp - b_v), h = 3", r67, n_klems_cells=len(kc)))
    O = pd.DataFrame(out)
    O.to_csv(OUT / "s1_outcomes.csv", index=False)
    print(O[["outcome", "est", "ci90_lo", "ci90_hi", "n", "G", "label"]].round(4).to_string(), flush=True)

    # ---- group components (descriptive)
    for g in ("se", "corp"):
        sub = P3[groups(P3) == g]
        for k, v in coefs(sub, "R_hours", ["Sv", "Spp"]).items():
            desc.append(row(f"group {g}: b_{k} (joint), hours", v, kind="components"))
        desc.append(row(f"group {g}: b_v - b_pp, hours", joint(sub, "R_hours"), kind="components"))

    # ---- test 1: joint and both orthogonalisation orders (pooled)
    for k, v in coefs(P3, "R_hours", ["Sv", "Spp"]).items():
        desc.append(row(f"test 1 joint: b_{k}", v, kind="test1"))
    for k, v in orth(P3, "R_hours", "Sv", "Spp").items():
        desc.append(row(f"test 1 order (Sv first): {k}", v, kind="test1"))
    for k, v in orth(P3, "R_hours", "Spp", "Sv").items():
        desc.append(row(f"test 1 order (Spp first): {k}", v, kind="test1"))
    for k, v in coefs(P3, "R_hours", ["Sv"]).items():
        desc.append(row("test 1 alone: b_Sv", v, kind="test1"))
    for k, v in coefs(P3, "R_hours", ["Spp"]).items():
        desc.append(row("test 1 alone: b_Spp", v, kind="test1"))

    # ---- placebos: permutations (DD and pooled b_v) and flat vector
    pr = []
    for kk in range(NPERM):
        q = P3.assign(Sperm=SP[P3._row.to_numpy(), kk])
        a = dd(q, "R_hours", sv="Sperm", boot=False)["est"]
        qq = std(q.dropna(subset=["R_hours", "Sperm", "Spp"] + CTRL), ["Sperm", "Spp"])
        b = reg(qq, "R_hours", ["Sperm", "Spp"] + CTRL, 0, boot=False)["est"]
        pr.append(dict(k=kk, dd=a, b_v=b))
        if kk % 20 == 0:
            print("perm", kk, round(a, 4), round(b, 4), flush=True)
    PR = pd.DataFrame(pr)
    PR.to_csv(OUT / "s1_placebo_perm.csv", index=False)
    bv = [x for x in desc if x["outcome"] == "test 1 joint: b_Sv"][0]["est"]
    plac = dict(dd_main=r64["est"], dd_perm_p95=float(PR.dd.quantile(0.95)), dd_share_perm_ge=float((PR.dd >= r64["est"]).mean()),
                bv_main=bv, bv_perm_p95=float(PR.b_v.quantile(0.95)), bv_share_perm_ge=float((PR.b_v >= bv).mean()),
                dd_specific=bool(r64["est"] > PR.dd.quantile(0.95)), bv_specific=bool(bv > PR.b_v.quantile(0.95)))
    pd.Series(plac).to_csv(OUT / "s1_placebo_summary.csv")
    print(pd.Series(plac).to_string(), flush=True)
    desc.append(row("placebo flat vector: DD with S_flat instead of S_v", dd(P3, "R_hours", sv="Sflat"), kind="placebo"))
    for k, v in coefs(P3, "R_hours", ["Sflat", "Spp"]).items():
        desc.append(row(f"placebo flat vector: joint b_{k}", v, kind="placebo"))

    # ---- variants of the DD (64) and corporate difference (65)
    V = []

    def var(name, p, y="R_hours", **kw):
        a = labelled(dd(p, y, **kw), 0.02, ">")
        b = labelled(dd(p, y, which="corp", **kw), 0.02, ">")
        V.append(row(f"{name}: DD", a, variant=name))
        V.append(row(f"{name}: corporate b_pp - b_v", b, variant=name))
        print("variant", name, round(a["est"], 4), a["label"], "|", round(b["est"], 4), b["label"], flush=True)
    var("employment (R_E)", P3, y="R_EMP")
    var("real output (R_Q)", P3, y="R_GO_QI")
    var("S_ppU (uniform wage)", P3, spp="SppU")
    var("S_r (stock profit rate)", P3, spp="Sr")
    var("no mean-reversion controls", P3, ctrl=[])
    P3c = P3.assign(ci=P3.country + "_" + P3.ind)
    var("country x industry FE", P3c, fe=("ci", "cy"))
    var("without agriculture", P3[~P3.ind.isin(AGRI)])
    P3all = build_panel(d, 3, excl={"T", "U"}, s_years=range(2001, 2011))
    var("exclusions T, U only", P3all)
    var("thresholds 0.40 / 0.05", P3, hi=0.40, lo=0.05)
    lo_, hi_ = P3.R_hours.quantile([0.01, 0.99])
    var("trim 1% response tails", P3[P3.R_hours.between(lo_, hi_)])
    var("weights: hours at signal year", P3, w="hours")
    var("hours KLEMS (H_EMP), KLEMS industries", K3[K3.year <= 2010], y="R_HK")
    V.append(row("continuous sigma: (b_v - b_pp) slope x 0.15", labelled(dd_cont(P3, "R_hours"), 0.02, ">"),
                 variant="continuous sigma"))
    for hh in (1, 2, 4, 5):
        Ph = build_panel(d, hh, s_years=range(2001, 2014 - hh))
        var(f"h = {hh}", Ph)
    pd.DataFrame(V).to_csv(OUT / "s1_variants.csv", index=False)

    # ---- descriptive extras for 66 and 67
    desc.append(row("66 variant: self-employed, thresholds sigma >= 0.25 only",
                    joint(P3[P3.sigma >= SE_HI], "R_SE"), kind="extra"))
    for k, v in coefs(kc, "R_I", ["Sv", "Spp"]).items():
        desc.append(row(f"67 components: corporate b_{k}, investment", v, kind="extra"))
    ks = K3[K3.sigma >= SE_HI]
    if len(ks.dropna(subset=["R_I", "Sv", "Spp"])) > 50:
        desc.append(row("67 contrast: self-employed KLEMS cells (b_v - b_pp), investment", joint(ks, "R_I"),
                        kind="extra"))

    # ---- FIGARO replication and reduction of complex labour
    F = figaro_panel(sig)
    print("FIGARO panel", len(F), flush=True)
    for zcol, nm in (("Sz_hours", "hours"), ("Sz_edu_years", "edu_years"), ("Sz_edu_train", "edu_train")):
        if zcol not in F:
            continue
        Fi = F.assign(ind=F.ind)
        desc.append(row(f"FIGARO DD ({nm})", dd(Fi, "R_hours", sv=zcol, spp="Spp_actual_wage"), kind="figaro"))
        desc.append(row(f"FIGARO corporate b_pp - b_v ({nm})",
                        dd(Fi, "R_hours", sv=zcol, spp="Spp_actual_wage", which="corp"), kind="figaro"))
        for k, v in coefs(Fi, "R_hours", [zcol, "Spp_actual_wage"]).items():
            desc.append(row(f"FIGARO joint b_{k} ({nm})", v, kind="figaro"))
    desc.append(row("FIGARO DD, persons response", dd(F, "R_persons", sv="Sz_hours", spp="Spp_actual_wage"),
                    kind="figaro"))
    pc = [c for c in F.columns if c.startswith("Splacebo_perm_")]
    fp = [dd(F, "R_hours", sv=c, spp="Spp_actual_wage", boot=False)["est"] for c in pc]
    fpp = pd.Series(fp)
    fdd = [x for x in desc if x["outcome"] == "FIGARO DD (hours)"][0]["est"]
    desc.append(dict(outcome="FIGARO placebo perm: share of DD_perm >= DD", est=float((fpp >= fdd).mean()),
                     n=len(fpp), kind="figaro", p95=float(fpp.quantile(0.95))))
    D = pd.DataFrame(desc)
    D.to_csv(OUT / "s1_descriptive.csv", index=False)
    pd.set_option("display.width", 250)
    print(D[["outcome", "est", "ci90_lo", "ci90_hi", "n", "G"]].round(4).to_string(), flush=True)
    print(pd.DataFrame(V)[["outcome", "est", "ci90_lo", "ci90_hi", "n", "label"]].round(4).to_string(), flush=True)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("signals", "all"):
        signals()
    if what in ("figaro_hours", "all"):
        figaro_hours()
    if what in ("estimate", "all"):
        estimate()
