"""Iteration 13a, stage 1 (US): the law of the tendency (pre_registration_v13a.md 1.1-1.6, edition after the
pre-calculation review, journal item 3; outcomes 110-118 and 123-131).

Six switches from the 12b main measure to the main form r*_h (1.2):
  cap   - capital data: '12b' (FA 2.1 asset composition common to all industries) | 'detail' (FA Detailed, asset x
          industry; for historical valuation in the '12b' position the detailed historical stock is rescaled by the
          ratio of 12b to detailed current-cost stocks of the same year and scope);
  lab   - 'persons' (12b) | 'hours' (BEA-BLS hrs, stage 0);
  who   - 'all' (12b: all workers of productive industries in L and V) | 'pns' (production and nonsupervisory
          workers + self-employed in L; V = basket of PNS compensation);
  dep   - 'gross' (12b) | 'net' (depreciation in lambda);
  val   - 'curr' (12b) | 'hist' (hours of the purchase year);
  den   - 'P' (productive capital, 12b) | 'B' (all private business nonresidential capital; with cap='12b' - the
          12b 'all' scope, FA 2.1 private nonresidential lines 3, 36, 77).
Main form: cap=detail, lab=hours, who=pns, dep=net, val=hist, den=B.

Specification layer (journal item 3): spec = 'main' | 'no_se' (self-employed hours out of L) | 'omega_alt'
(omega* = PNS wages and salaries / (personal income - personal current transfer receipts)) | 'no_ipp' (capital
without software ENS*, R&D RD*, entertainment originals AE*).

Usage: PYTHONPATH=src python -P -m lts.v13a.stage1 [spec|series|tests|all]
Outputs: results/v13a/s1_*.csv
"""
from __future__ import annotations

import itertools
import math
import sys
from functools import lru_cache

import numpy as np
import pandas as pd
import statsmodels.api as sm

from ..v6.series import ROOT, nipa
from ..v12b import us_io as u
from . import est
from . import matrix as mx
from . import stage0 as s0

OUT = ROOT / "results" / "v13a"
CODES = u.CODES
NC = len(CODES)
YEARS = list(range(1948, 2024))
LAMBDA_YEARS = list(range(1947, 2024))
PROD = np.array([c not in u.UNPRODUCTIVE for c in CODES])
GOV = np.array([c.startswith("G") for c in CODES])
SWITCHES = ["cap", "lab", "who", "dep", "val", "den"]
B12 = dict(cap="12b", lab="persons", who="all", dep="gross", val="curr", den="P")
MAIN = dict(cap="detail", lab="hours", who="pns", dep="net", val="hist", den="B")
ALT = dict(MAIN)
SPECS = ["no_se", "omega_alt", "no_ipp"]
IPP_PREFIX = ("ENS", "RD", "AE")
HEDONIC = {"EP1A", "EP1B", "EP1C", "EP1D", "EP1E", "EP1F", "EP1G", "EP1H", "EP20", "ENS1", "ENS2", "ENS3"}

# phases (1.3)
PHASES = {"A": (1948, 1973), "B": (1983, 2007), "C": (2010, 2019)}
CRISES = [(1974, 1982), (2008, 2009), (2020, 2021)]
NBER = [1948, 1953, 1957, 1960, 1969, 1973, 1980, 1981, 1990, 2001, 2007, 2019, 2023]
NBER_TROUGH = [1949, 1954, 1958, 1961, 1970, 1975, 1980, 1982, 1991, 2001, 2009, 2020]
A_PRIME = (1964, 1973)
D = 0.002


# ================================================================ inputs
@lru_cache(None)
def hours():
    return pd.read_csv(OUT / "s0_hours_47.csv", index_col=0).reindex(columns=CODES)


@lru_cache(None)
def ces(variant="main"):
    Dd = pd.read_csv(OUT / "s0_ces_shares.csv", dtype={"code": str})
    Dd = Dd[Dd.variant == variant]
    return {k: Dd.pivot_table(index="year", columns="code", values=k).reindex(columns=CODES) for k in ("sH", "sW")}


@lru_cache(None)
def se_share():
    return pd.read_csv(OUT / "s0_selfemployed_share.csv", index_col=0).reindex(columns=CODES).fillna(0.0)


@lru_cache(None)
def wages():
    return pd.read_csv(OUT / "s0_wages_47.csv", index_col=0).reindex(columns=CODES)


@lru_cache(None)
def transfers():
    return nipa(["A577RC"]).A577RC


@lru_cache(None)
def vint():
    z = np.load(OUT / "s0_vintages.npz", allow_pickle=True)
    return {k: z[k] for k in z.files}


@lru_cache(None)
def producer_index():
    """For each vintage row (industry x asset): index of the producing industry among the 47 codes; scope flags."""
    v = vint()
    prod_code = np.array([CODES.index(s0.asset_producer(a)) for a in v["asset"]])
    unprod = np.array([i in s0.FA_UNPROD for i in v["ind"]])
    return prod_code, ~unprod


@lru_cache(None)
def row_mask(ipp=True):
    v = vint()
    if ipp:
        return np.ones(len(v["asset"]), bool)
    return ~np.array([str(a).startswith(IPP_PREFIX) for a in v["asset"]])


@lru_cache(None)
def lam_index():
    return pd.read_csv(OUT / "s0_lambda_index_pre1947.csv", index_col=0).iloc[:, 0]


def _y(year):
    return min(max(year, 1948), 2023)


# ================================================================ labour and lambda for one year
@lru_cache(None)
def labour(year, lab, who, ces_variant="main", se_in_L=True):
    """L_j (value-creating labour of productive industries) and PNS compensation share s_W by code."""
    yy = _y(year)
    if lab == "hours":
        base = hours().loc[year].to_numpy(float)
    else:
        base = u.persons(yy).to_numpy(float)
    se = se_share().loc[yy].to_numpy(float)
    if who == "pns":
        sh = ces(ces_variant)
        sH = sh["sH"].loc[yy].fillna(1.0).to_numpy(float)
        sW = sh["sW"].loc[yy].fillna(1.0).to_numpy(float)
        L = base * ((1 - se) * sH + (se if se_in_L else 0.0))
    else:
        sW = np.ones(NC)
        L = base * (1.0 if se_in_L else (1 - se))
    return np.where(PROD, L, 0.0), sW


@lru_cache(None)
def lam(year, lab, who, dep, ces_variant="main", se_in_L=True):
    """lambda_h (hours per dollar of output by industry)."""
    s = u.system(year)
    A, x = s["A"], s["x"]
    L, sW = labour(year, lab, who, ces_variant, se_in_L)
    M = A.copy()
    if dep == "net":
        d = np.divide(u.depreciation(_y(year)), x, out=np.zeros(NC), where=x > 0)
        g = s["inv"] / s["inv"].sum()
        M = M + np.outer(g, d)
    l = np.divide(L, x, out=np.zeros(NC), where=x > 0)
    return l @ np.linalg.inv(np.eye(NC) - M)


def omega(year, sW, who, mode="main"):
    """omega*: PNS compensation of productive industries / personal income (1.1); 'omega_alt': PNS wages and
    salaries / (personal income - personal current transfer receipts)."""
    PI = float(u.nipa_series().loc[year, "A065RC"])
    if mode == "omega_alt":
        ws = wages().loc[year].to_numpy(float)
        return float(np.nansum(ws * sW * PROD)) / (PI - float(transfers().loc[year]))
    comp = u.compensation(year).to_numpy(float)
    return float((comp * sW * PROD).sum()) / PI


# ================================================================ capital in hours
@lru_cache(None)
def detail_current_vectors(year, ipp=True):
    """Current-cost net stocks (stk1) by producing industry, for scope P (productive FA industries) and B (all)."""
    v = vint()
    j = v["years"].tolist().index(year)
    K1 = np.nan_to_num(v["K1"][:, j]) * row_mask(ipp)
    pc, prod_rows = producer_index()
    P = np.bincount(pc[prod_rows], weights=K1[prod_rows], minlength=NC)
    B = np.bincount(pc, weights=K1, minlength=NC)
    return P, B


@lru_cache(None)
def b12_vectors(year):
    Kp = sum(u.capital(year, "productive").values())
    Ka = sum(u.capital(year, "all").values())
    return Kp, Ka


@lru_cache(None)
def hist_hours(lab, who, dep, pre="index", ces_variant="main", se_in_L=True, ipp=True):
    """Historical-cost capital in hours of the purchase year, by year 1947-2023, scope P and B.
    pre: 'index' (lambda_1947 x uslt4x index), 'const' (lambda_1947), 'current' (pre-1947 vintages at current cost)."""
    v = vint()
    years = v["years"].tolist()
    I1, dd = v["I1"] * row_mask(ipp)[:, None], v["delta"]
    pc, prod_rows = producer_index()
    lam_by = {y: lam(y, lab, who, dep, ces_variant, se_in_L) for y in LAMBDA_YEARS}
    li = lam_index()
    W = np.zeros_like(I1)
    for j, y in enumerate(years):
        if y >= 1947:
            W[:, j] = lam_by[y][pc]
        elif pre == "index":
            W[:, j] = lam_by[1947][pc] * float(li.get(y, 1.0))
        elif pre == "const":
            W[:, j] = lam_by[1947][pc]
        else:
            W[:, j] = 0.0                       # handled below at current cost
    Z = s0.vintage_stock(I1, dd, W)
    if pre == "current":
        # surviving pre-1947 vintages at current cost, revalued with investment prices p = I1 / I2
        I2 = v["I2"]
        with np.errstate(divide="ignore", invalid="ignore"):
            p = np.where(I2 > 0, v["I1"] / I2, np.nan)
        p = pd.DataFrame(p).T.ffill().bfill().T.to_numpy()
        prev = np.zeros(I1.shape[0])
        for j, y in enumerate(years):
            ratio = np.divide(p[:, j], p[:, j - 1], out=np.ones(I1.shape[0]), where=p[:, j - 1] > 0) if j else 1.0
            inv = I1[:, j] if y < 1947 else 0.0
            prev = prev * ratio * (1 - dd[:, j]) + inv * (1 - dd[:, j] / 2)
            if y >= 1947:
                Z[:, j] = Z[:, j] + prev * lam_by[y][pc]
    return {y: (float(Z[prod_rows, j].sum()), float(Z[:, j].sum())) for j, y in enumerate(years) if y >= 1947}


def capital_hours(year, cfg, lam_t, pre="index", ces_variant="main", se_in_L=True, ipp=True):
    """C_h for one year and configuration."""
    Pd, Bd = detail_current_vectors(year, ipp)
    curr_detail = float(lam_t @ (Bd if cfg["den"] == "B" else Pd))
    if cfg["cap"] == "12b":
        Kp, Ka = b12_vectors(year)
        curr_12b = float(lam_t @ (Ka if cfg["den"] == "B" else Kp))
    if cfg["val"] == "curr":
        return curr_detail if cfg["cap"] == "detail" else curr_12b
    hp, hb = hist_hours(cfg["lab"], cfg["who"], cfg["dep"], pre, ces_variant, se_in_L, ipp)[year]
    hist_detail = hb if cfg["den"] == "B" else hp
    if cfg["cap"] == "detail":
        return hist_detail
    return hist_detail * curr_12b / curr_detail


# ================================================================ one year, one configuration
def year_values(year, cfg, pre="index", ces_variant="main", spec="main"):
    se_in_L = spec != "no_se"
    ipp = spec != "no_ipp"
    s = u.system(year)
    c = s["c"]
    L, sW = labour(year, cfg["lab"], cfg["who"], ces_variant, se_in_L)
    lt = lam(year, cfg["lab"], cfg["who"], cfg["dep"], ces_variant, se_in_L)
    om = omega(year, sW, cfg["who"], "omega_alt" if spec == "omega_alt" else "main")
    b = om * c
    V = float(lt @ b)
    Ltot = float(L.sum())
    S = Ltot - V
    kw = dict(pre=pre, ces_variant=ces_variant, se_in_L=se_in_L, ipp=ipp)
    C = capital_hours(year, cfg, lt, **kw)
    C_curr = capital_hours(year, {**cfg, "val": "curr"}, lt, **kw)
    C_P = capital_hours(year, {**cfg, "den": "P"}, lt, **kw)
    C_curr_P = capital_hours(year, {**cfg, "val": "curr", "den": "P"}, lt, **kw)
    return dict(year=year, L=Ltot, V=V, S=S, C=C, C_curr=C_curr, C_P=C_P, C_curr_P=C_curr_P, omega=om,
                basket_usd=float(b.sum()), e=S / V, k=C / V, r=S / (C + V), r_P=S / (C_P + V),
                r_curr=S / (C_curr + V), k_curr=C_curr / V, k_P=C_P / V, CL_P=C_P / Ltot, CL=C / Ltot)


def config_key(cfg):
    return "|".join(f"{k}={cfg[k]}" for k in SWITCHES)


def all_configs():
    for bits in itertools.product([0, 1], repeat=6):
        yield {k: (ALT[k] if b else B12[k]) for k, b in zip(SWITCHES, bits)}, bits


DESC = {"cap": ("FA 2.1: общая структура активов для всех отраслей (12б)", "FA Detailed: активы × отрасли"),
        "lab": ("лица, занятые в производстве (12б)", "часы BEA–BLS (этап 0.1)"),
        "who": ("все занятые производительных отраслей в L и V (12б)", "производственные и рядовые + самозанятые в L; "
                "V — корзина оплаты рядовых"),
        "dep": ("валовая λ (12б)", "чистая λ: M = A + g dᵀ"),
        "val": ("текущая оценка (12б)", "историческая: часы года покупки"),
        "den": ("(П): капитал производительных отраслей", "(В): весь частный деловой нежилой капитал")}


def shapley_spec():
    """Table of all 64 cells (review P20): data and formulas of every combination; written before running."""
    rows = []
    for cfg, bits in all_configs():
        r = {"bits": "".join(map(str, bits))}
        for k, b in zip(SWITCHES, bits):
            r[k] = DESC[k][b]
        if cfg["cap"] == "12b" and cfg["val"] == "hist":
            r["note"] = ("историческая при данных 12б: историческая сумма поколений FA Detailed × (текущий капитал 12б / "
                         "текущий капитал FA Detailed) того же года и охвата")
        elif cfg["cap"] == "12b" and cfg["den"] == "B":
            r["note"] = "(В) при данных 12б: капитал 12б в охвате «все отрасли» (FA 2.1, строки 3, 36, 77)"
        else:
            r["note"] = ""
        rows.append(r)
    pd.DataFrame(rows).to_csv(OUT / "s1_shapley_spec.csv", index=False)


# ================================================================ series
def series():
    rows = []
    for cfg, bits in all_configs():
        for y in YEARS:
            r = year_values(y, cfg)
            r.update(config=config_key(cfg), bits="".join(map(str, bits)))
            rows.append(r)
        print("config", "".join(map(str, bits)), flush=True)
    S_ = pd.DataFrame(rows)
    S_.to_csv(OUT / "s1_configs.csv", index=False)
    b12 = S_[S_.bits == "000000"].set_index("year").r
    ref = pd.read_csv(ROOT / "results" / "v12b" / "s1_us_series.csv")
    ref = ref[ref.variant == "main"].set_index("year").r_h
    main = S_[S_.bits == "111111"].set_index("year").r
    ms = main_series()
    chk = pd.DataFrame({"r_13a_12b_position": b12, "r_h_12b": ref.reindex(b12.index), "r_111111": main,
                        "r_main_series": ms.r})
    chk["absdiff_12b_end"] = (chk.r_13a_12b_position - chk.r_h_12b).abs()
    chk["absdiff_13a_end"] = (chk.r_111111 - chk.r_main_series).abs()
    chk.to_csv(OUT / "s1_check_12b.csv")
    print("max |r(000000) - r_h 12b| =", float(chk.absdiff_12b_end.max()),
          "; max |r(111111) - r*_h| =", float(chk.absdiff_13a_end.max()), flush=True)
    return S_


@lru_cache(None)
def main_series(pre="index", ces_variant="main", spec="main", cfg_key=None):
    cfg = MAIN if cfg_key is None else dict(kv.split("=") for kv in cfg_key.split("|"))
    return pd.DataFrame([year_values(y, cfg, pre, ces_variant, spec) for y in YEARS]).set_index("year")


# ================================================================ money analogues and TSSI with MELT
@lru_cache(None)
def _hist_dollars(ipp=True):
    v = vint()
    return s0.vintage_stock(v["I1"] * row_mask(ipp)[:, None], v["delta"])


def ndp(year):
    s = u.system(year)
    return float(s["n"].sum() - u.depreciation(year).sum())


@lru_cache(None)
def money_series():
    """lambda_$ = 1 on the same physical quantities (1.1): S_$ = NDP - V_$; C_$ historical (vintage sums in dollars
    of the purchase year) or current (stk1); (B) or (P)."""
    v = vint()
    years = v["years"].tolist()
    pc, prod_rows = producer_index()
    Hd = _hist_dollars()
    rows = []
    for y in YEARS:
        s = u.system(y)
        L, sW = labour(y, "hours", "pns")
        V = omega(y, sW, "pns") * float(s["c"].sum())
        Lall, _ = labour(y, "hours", "all")
        V_all = omega(y, np.ones(NC), "all") * float(s["c"].sum())
        N = ndp(y)
        j = years.index(y)
        K1 = np.nan_to_num(v["K1"][:, j])
        r = dict(year=y, NDP=N, V=V, V_all=V_all, S=N - V, L=float(L.sum()),
                 C_hist_B=float(Hd[:, j].sum()), C_hist_P=float(Hd[prod_rows, j].sum()),
                 C_curr_B=float(K1.sum()), C_curr_P=float(K1[prod_rows].sum()))
        r.update(r_hist=(N - V) / (r["C_hist_B"] + V), k_hist=r["C_hist_B"] / V,
                 r_curr=(N - V) / (r["C_curr_B"] + V), k_curr=r["C_curr_B"] / V,
                 CN_P=r["C_hist_P"] / N, e=(N - V) / V, e_all=(N - V_all) / V_all,
                 U=float(u.compensation(y).to_numpy(float)[~GOV].sum()) - float(
                     (u.compensation(y).to_numpy(float) * sW * PROD).sum()))
        rows.append(r)
    return pd.DataFrame(rows).set_index("year")


@lru_cache(None)
def tssi_series():
    """r_TSSI with MELT (1.1): MELT(t) = NDP_$ / L; before 1947 MELT(1947) / lambda-index; C_TSSI = sum of vintages
    in dollars / MELT of the purchase year (B)."""
    v = vint()
    years = v["years"].tolist()
    li = lam_index()
    melt = {}
    for y in LAMBDA_YEARS:
        L, _ = labour(y, "hours", "pns")
        melt[y] = ndp(y) / float(L.sum())
    W = np.zeros_like(v["I1"])
    for j, y in enumerate(years):
        W[:, j] = 1.0 / (melt[y] if y >= 1947 else melt[1947] / float(li.get(y, 1.0)))
    Z = s0.vintage_stock(v["I1"], v["delta"], W)
    M_ = money_series()
    rows = []
    for y in YEARS:
        j = years.index(y)
        C = float(Z[:, j].sum())
        Vt = M_.loc[y, "V"] / melt[y]
        Lh = M_.loc[y, "L"]
        rows.append(dict(year=y, MELT=melt[y], C_TSSI=C, V_TSSI=Vt, S_TSSI=Lh - Vt, r_TSSI=(Lh - Vt) / (C + Vt)))
    return pd.DataFrame(rows).set_index("year")


# ================================================================ Okishio counterfactual (114, 115)
def okishio(ms, phases, valuation, pcons=None):
    """Rates with the real basket per hour of rank-and-file labour fixed at each phase start (hours)."""
    pcons = u.nipa_series().DPCERG if pcons is None else pcons
    out = {}
    for nm, (a, b) in phases.items():
        if a not in ms.index:
            continue
        q = ms.loc[a, "basket_usd"] / (pcons[a] * ms.loc[a, "L"])
        for y in range(a, b + 1):
            if y not in ms.index:
                continue
            r = ms.loc[y]
            Vcf = r["V"] * (q * pcons[y] * r["L"]) / r["basket_usd"]
            C = r["C_curr"] if valuation == "curr" else r["C"]
            out[y] = (r["L"] - Vcf) / (C + Vcf)
    return pd.Series(out, dtype=float).sort_index()


def okishio_money(phases, valuation):
    pcons = u.nipa_series().DPCERG
    M_ = money_series()
    out = {}
    for nm, (a, b) in phases.items():
        if a not in M_.index:
            continue
        q = M_.loc[a, "V"] / (pcons[a] * M_.loc[a, "L"])
        for y in range(a, b + 1):
            if y not in M_.index:
                continue
            Vcf = q * pcons[y] * M_.loc[y, "L"]
            C = M_.loc[y, "C_curr_B"] if valuation == "curr" else M_.loc[y, "C_hist_B"]
            out[y] = (M_.loc[y, "NDP"] - Vcf) / (C + Vcf)
    return pd.Series(out, dtype=float).sort_index()


# ================================================================ moral depreciation (113)
@lru_cache(None)
def asset_prices():
    """p_k(t) = sum over industries of current-cost stock / chained stock, by asset (common to industries)."""
    v = vint()
    assets = v["asset"]
    K1, K2 = np.nan_to_num(v["K1"]), np.nan_to_num(v["K2"])
    df1 = pd.DataFrame(K1).groupby(assets).sum()
    df2 = pd.DataFrame(K2).groupby(assets).sum()
    P = (df1 / df2.replace(0, np.nan))
    P.columns = v["years"]
    return P


@lru_cache(None)
def moral_depreciation():
    """MD(t) = sum_k sum_{i in B} [h_k(t-1) - h_k(t)] Q_surv(t), h_k = lambda_{j(k)} p_k; m(t) = MD / C_curr^B(t-1).
    Split (descriptive, review P7): h = (lambda P)(p / P), P = GDP deflator; midpoint split into the productivity part
    (hours per real dollar of the producing industry) and the relative asset price part. Variant without assets
    with hedonic prices."""
    cfg = MAIN
    v = vint()
    years = v["years"].tolist()
    pc, _ = producer_index()
    pk = asset_prices().reindex(v["asset"]).to_numpy()
    K1, K2, D2 = np.nan_to_num(v["K1"]), np.nan_to_num(v["K2"]), np.nan_to_num(v["D2"])
    gd = u.nipa_series().A191RD
    hed = np.isin(v["asset"], list(HEDONIC))
    rows = []
    for y in YEARS:
        j = years.index(y)
        l1 = lam(y, cfg["lab"], cfg["who"], cfg["dep"])[pc]
        l0 = lam(y - 1, cfg["lab"], cfg["who"], cfg["dep"])[pc]
        qs = K2[:, j - 1] - D2[:, j]
        h1, h0 = l1 * pk[:, j], l0 * pk[:, j - 1]
        md_rows = np.nan_to_num((h0 - h1) * qs)
        a0, a1 = l0 * gd[y - 1], l1 * gd[y]
        b0, b1 = pk[:, j - 1] / gd[y - 1], pk[:, j] / gd[y]
        md_a = float(np.nansum((a0 - a1) * (b0 + b1) / 2 * qs))
        md_b = float(np.nansum((b0 - b1) * (a0 + a1) / 2 * qs))
        cp = np.nan_to_num(l0 * K1[:, j - 1])
        rows.append(dict(year=y, MD=float(md_rows.sum()), C_curr_prev=float(cp.sum()),
                         m=float(md_rows.sum()) / float(cp.sum()), m_productivity=md_a / float(cp.sum()),
                         m_relative_price=md_b / float(cp.sum()),
                         m_no_hedonic=float(md_rows[~hed].sum()) / float(cp[~hed].sum())))
    return pd.DataFrame(rows).set_index("year")


def runs_excluding(excl, a=1948, b=2023):
    """Contiguous runs of years in [a, b] not in excl, as a dict of windows."""
    out, cur = {}, None
    for y in range(a, b + 1):
        if y in excl:
            if cur:
                out[f"{cur[0]}-{cur[1]}"] = cur
            cur = None
        else:
            cur = (cur[0], y) if cur else (y, y)
    if cur:
        out[f"{cur[0]}-{cur[1]}"] = cur
    return out


def crises_between(phases):
    ph = sorted(phases.values())
    return [(a[1] + 1, b[0] - 1) for a, b in zip(ph[:-1], ph[1:]) if b[0] - 1 >= a[1] + 1] + [(2020, 2021)]


# ================================================================ phases and robustness
def boundary_variants():
    out = []
    for ae, bs, be, cs, ce in itertools.product(range(1971, 1976), range(1981, 1986), range(2005, 2010),
                                                 range(2008, 2013), range(2017, 2022)):
        if cs <= be:
            continue
        out.append({"A": (1948, ae), "B": (bs, be), "C": (cs, ce)})
    return out


def nber_phases():
    return {f"{a}-{b}": (a + (0 if i == 0 else 1), b) for i, (a, b) in enumerate(zip(NBER[:-1], NBER[1:]))}


def _sub(y, sub, window):
    a0, a1 = window
    if sub == "-first5":
        return y[y.index > a0 + 4]
    if sub == "-last5":
        return y[y.index < a1 - 4]
    return y


def assess_within(name, y_of, delta, direction, lam_ms=None, spec_ms=None, grid=True, nber=True):
    """Main estimate + robustness layers for a within-phase common slope. y_of(ms, phases) -> series."""
    y = y_of(None, PHASES)
    o = est.within_trend(y, PHASES, delta, direction)
    o.update(est.phase_slopes(y, PHASES))
    main = o["label_full"]
    nine = [est.within_trend(_sub(y, sub, (1948, 2019)), PHASES, delta, direction, lag=lag)["label_full"]
            for lag in (2, 4, 8) for sub in ("full", "-first5", "-last5")]
    lam_l = [est.within_trend(y_of(m, PHASES), PHASES, delta, direction)["label_full"] for m in (lam_ms or [])]
    g = []
    if grid:
        for k, ph in enumerate(boundary_variants()):
            g.append(est.within_trend(y_of(None, ph), ph, delta, direction, B=499, seed=1000 + k)["label_full"])
    nb = None
    if nber:
        ph = nber_phases()
        nb = est.within_trend(y_of(None, ph), ph, delta, direction)["label_full"]
    sp = [est.within_trend(y_of(m, PHASES), PHASES, delta, direction)["label_full"] for m in (spec_ms or [])]
    rob, why = est.robust_flag(main, nine, lam_l, g, nb, sp)
    o.update(outcome=name, nine=";".join(sorted(set(nine))), lambda_labels=";".join(lam_l),
             grid_labels=";".join(sorted(set(g))), grid_share_main=float(np.mean([x == main for x in g])) if g else np.nan,
             grid_n=len(g), nber_label=nb, spec_labels=";".join(sp),
             spec_share_main=float(np.mean([x == main for x in sp])) if sp else np.nan, robust=rob, robust_why=why)
    return o


def assess_diff(name, y_of, pa, pb, delta, direction, window, spec_ms=None, grid=True):
    """B - A difference of own phase slopes (116, 126-128): main, 9 combinations, boundary grid (A' = 1964 to the
    end of A where pa starts in 1964), specification layer."""
    y = y_of(None)
    o = est.two_phase_diff(y, pa, pb, delta, direction)
    main = o["label_full"]
    nine = [est.two_phase_diff(_sub(y, sub, window), pa, pb, delta, direction, lag=lag)["label_full"]
            for lag in (2, 4, 8) for sub in ("full", "-first5", "-last5")]
    g = []
    if grid:
        for k, ph in enumerate(boundary_variants()):
            a = (pa[0], ph["A"][1])
            g.append(est.two_phase_diff(y, a, ph["B"], delta, direction, B=499, seed=4000 + k)["label_full"])
    sp = [est.two_phase_diff(y_of(m), pa, pb, delta, direction)["label_full"] for m in (spec_ms or [])]
    rob, why = est.robust_flag(main, nine, [], g, None, sp)
    o.update(outcome=name, nine=";".join(sorted(set(nine))), grid_labels=";".join(sorted(set(g))),
             grid_share_main=float(np.mean([x == main for x in g])) if g else np.nan, grid_n=len(g),
             spec_labels=";".join(sp), spec_share_main=float(np.mean([x == main for x in sp])) if sp else np.nan,
             robust=rob, robust_why=why)
    return o


# ================================================================ Shapley
def shapley(values):
    """values: dict bits-string -> statistic; Shapley value per switch, fixed-order steps, pairwise Harsanyi
    dividends at the 12b base."""
    n = 6
    phi = np.zeros(n)
    for i in range(n):
        for bits in itertools.product([0, 1], repeat=n):
            if bits[i] == 1:
                continue
            k = sum(bits)
            w = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
            on = list(bits)
            on[i] = 1
            phi[i] += w * (values["".join(map(str, on))] - values["".join(map(str, bits))])
    steps, cur = [], [0] * n
    for i in range(n):
        prev = values["".join(map(str, cur))]
        cur[i] = 1
        steps.append(values["".join(map(str, cur))] - prev)
    pair = {}
    z = "0" * n
    for i, j in itertools.combinations(range(n), 2):
        b_i = ["0"] * n
        b_i[i] = "1"
        b_j = ["0"] * n
        b_j[j] = "1"
        b_ij = ["0"] * n
        b_ij[i] = b_ij[j] = "1"
        pair[(SWITCHES[i], SWITCHES[j])] = values["".join(b_ij)] - values["".join(b_i)] - values["".join(b_j)] + values[z]
    return phi, np.array(steps), pair


def shapley_series(lr_by_bits):
    """Series whose within-phase slope equals the Shapley contribution of each switch (linearity)."""
    n = 6
    out = {}
    for i in range(n):
        acc = 0.0
        for bits in itertools.product([0, 1], repeat=n):
            if bits[i] == 1:
                continue
            k = sum(bits)
            w = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
            on = list(bits)
            on[i] = 1
            acc = acc + w * (lr_by_bits["".join(map(str, on))] - lr_by_bits["".join(map(str, bits))])
        out[SWITCHES[i]] = acc
    return out


# ================================================================ descriptive: energy, manufacturing
def energy_year(year):
    """Energy (EIA primary energy, 1949-), current valuation only."""
    s = u.system(year)
    A, x = s["A"], s["x"]
    d = np.divide(u.depreciation(year), x, out=np.zeros(NC), where=x > 0)
    M = A + np.outer(s["inv"] / s["inv"].sum(), d)
    row, _ = u.energy_row(year, x)
    lE = row @ np.linalg.inv(np.eye(NC) - M)
    L, sW = labour(year, "hours", "pns")
    b = omega(year, sW, "pns") * s["c"]
    E = float(row @ x)
    V = float(lE @ b)
    Pd, Bd = detail_current_vectors(year)
    C = float(lE @ Bd)
    return dict(year=year, r_E=(E - V) / (C + V), k_E=C / V, e_E=(E - V) / V)


MFG_FA = {"3210", "3270", "3310", "3320", "3330", "3340", "3350", "336M", "336O", "3370", "3380", "3390", "3110",
          "3120", "313T", "315A", "3220", "3230", "3240", "3250", "3260"}


@lru_cache(None)
def _hist_hours_rows():
    v = vint()
    pc, _ = producer_index()
    li = lam_index()
    lam_by = {y: lam(y, "hours", "pns", "net") for y in LAMBDA_YEARS}
    W = np.zeros_like(v["I1"])
    for j, y in enumerate(v["years"].tolist()):
        W[:, j] = lam_by[y][pc] if y >= 1947 else lam_by[1947][pc] * float(li.get(y, 1.0))
    return s0.vintage_stock(v["I1"], v["delta"], W)


def manufacturing_year(year):
    """Main form for manufacturing alone: L, V (PNS basket of manufacturing), C (historical, manufacturing rows)."""
    s = u.system(year)
    L, sW = labour(year, "hours", "pns")
    lt = lam(year, "hours", "pns", "net")
    mfg = np.array([c in u.MFG for c in CODES])
    comp = u.compensation(year).to_numpy(float)
    om = float((comp * sW * mfg).sum()) / float(u.nipa_series().loc[year, "A065RC"])
    V = float(lt @ (om * s["c"]))
    Lm = float(L[mfg].sum())
    v = vint()
    rows = np.array([i in MFG_FA for i in v["ind"]])
    j = v["years"].tolist().index(year)
    C = float(_hist_hours_rows()[rows, j].sum())
    return dict(year=year, r_mfg=(Lm - V) / (C + V), k_mfg=C / V, e_mfg=(Lm - V) / V)


# ================================================================ reading tables (1.5, descriptive)
def reading_110_111(l110, l111):
    def k(x):
        return "п" if x == est.CONF else ("о" if x.startswith(est.REF) else "н")
    T = {("п", "п"): "строение растёт, норма падает",
         ("п", "о"): "тенденция строения есть, внутри фаз её перекрыли противодействующие причины",
         ("п", "н"): "норма не установлена",
         ("о", "п"): "норма падает не через рост строения (против механизма гл. 13)",
         ("о", "о"): "строение не растёт, норма не падает",
         ("о", "н"): "норма не установлена",
         ("н", "п"): "норма падает; механизм не установлен",
         ("н", "о"): "норма не падает; механизм не установлен",
         ("н", "н"): "неразличимо"}
    return T[(k(l110), k(l111))]


def reading_114_115(l114, l115):
    def k(x):
        return "п" if x == est.CONF else ("о" if x.startswith(est.REF) else "н")
    T = {("п", "п"): "норма падает при постоянной реальной корзине при обеих оценках",
         ("п", "о"): "падает только при текущей оценке",
         ("п", "н"): "падает при текущей; при исторической не установлено",
         ("о", "п"): "граница метода: закон выполняется в прочтении М-ист (близком к TSSI) и не выполняется в "
                     "воспроизводственном прочтении",
         ("о", "о"): "при постоянной корзине норма не падает ни при какой оценке",
         ("о", "н"): "при текущей не падает; при исторической не установлено",
         ("н", "п"): "падает при исторической; при текущей не установлено",
         ("н", "о"): "при исторической не падает; при текущей не установлено",
         ("н", "н"): "неразличимо"}
    return T[(k(l114), k(l115))]


# ================================================================ tests
def tests():
    S_ = pd.read_csv(OUT / "s1_configs.csv")
    S_["bits"] = S_.bits.astype(str).str.zfill(6)
    ms = main_series()
    lamv = {"index (main)": ms, "const": main_series(pre="const"), "current": main_series(pre="current")}
    pd.concat(lamv, names=["lambda_pre1947"]).to_csv(OUT / "s1_lambda_variants.csv")
    lam_ms = [lamv["const"], lamv["current"]]
    spec = {sp: main_series(spec=sp) for sp in SPECS}
    pd.concat(spec, names=["spec"]).to_csv(OUT / "s1_spec_variants.csv")
    spec_ms = [spec[s_] for s_ in SPECS]
    M_ = money_series()
    M_.to_csv(OUT / "s1_money.csv")
    T_ = tssi_series()
    T_.to_csv(OUT / "s1_tssi.csv")
    pcons = u.nipa_series().DPCERG

    def col(c):
        return lambda m, ph: np.log((ms if m is None else m)[c])

    out, money_est = [], {}
    # ---- 110, 111, 115, 117, 125: full layers; 110, 111, 115 with lambda variants
    out.append(assess_within("110: trend ln k*_h within phases", col("k"), D, ">", lam_ms, spec_ms))
    out.append(assess_within("111: trend ln r*_h within phases", col("r"), D, "<", lam_ms, spec_ms))
    out.append(assess_within("114: Okishio counterfactual, current valuation",
                             lambda m, ph: np.log(okishio(ms if m is None else m, ph, "curr", pcons)), D, "<",
                             None, spec_ms))
    out.append(assess_within("115: Okishio counterfactual, historical valuation",
                             lambda m, ph: np.log(okishio(ms if m is None else m, ph, "hist", pcons)), D, "<",
                             lam_ms, spec_ms))
    out.append(assess_within("117: trend ln r(B) - ln r(P) within phases",
                             lambda m, ph: np.log((ms if m is None else m).r / (ms if m is None else m).r_P), 0.001, "<",
                             None, spec_ms))
    out.append(assess_within("123: trend ln r_h,curr (B) within phases", col("r_curr"), D, "<", None, spec_ms))
    out.append(assess_within("124: trend ln k_h,curr (B) within phases", col("k_curr"), D, ">", None, spec_ms))
    out.append(assess_within("125: trend ln(C_h,hist(P) / L) within phases", col("CL_P"), D, ">", None, spec_ms))
    out.append(assess_within("129: trend ln r_TSSI (MELT) within phases", lambda m, ph: np.log(T_.r_TSSI), D, "<"))
    out.append(assess_within("130: trend ln(r*_h / r*_$) within phases",
                             lambda m, ph: np.log(ms.r) - np.log(M_.r_hist), D, "<"))
    # ---- differences of phase slopes
    out.append(assess_diff("116: trend ln e*_h in phase B minus in A' = 1964-1973",
                           lambda m: np.log((ms if m is None else m).e), A_PRIME, PHASES["B"], D, ">", (1964, 2007),
                           spec_ms))
    out.append(assess_diff("126: ln r*_h, slope B minus slope A", lambda m: np.log((ms if m is None else m).r),
                           PHASES["A"], PHASES["B"], D, ">", (1948, 2007), spec_ms))
    out.append(assess_diff("127: ln(C_h,hist(P) / L), slope B minus slope A",
                           lambda m: np.log((ms if m is None else m).CL_P), PHASES["A"], PHASES["B"], D, "<",
                           (1948, 2007), spec_ms))
    allh = main_series(cfg_key=config_key({**MAIN, "who": "all"}))
    out.append(assess_diff("128: ln e with managers in V and L (all employees), B minus A'",
                           lambda m: np.log(allh.e), A_PRIME, PHASES["B"], D, ">", (1964, 2007)))
    # ---- 112, 113, 118, 131
    lr = np.log(ms.r)
    rec = [float(lr.loc[1983] - lr.loc[1973]), float(lr.loc[2010] - lr.loc[2007])]
    out.append(dict(outcome="112: recovery between phases, mean d ln r*_h (1973->1983, 2007->2010)",
                    est=float(np.mean(rec)), theta0=0.0, delta=0.05, direction=">", label="описательно (2 события)",
                    label_full="описательно (2 события)", events=";".join(f"{x:.4f}" for x in rec)))
    MD = moral_depreciation()
    MD.to_csv(OUT / "s1_moral_depreciation.csv")
    o = est.ratio_R(MD.m, PHASES, CRISES)
    g = [est.ratio_R(MD.m, ph, crises_between(ph), B=499, seed=2000 + k)["label_full"]
         for k, ph in enumerate(boundary_variants())]
    nbp = runs_excluding(set(NBER_TROUGH))
    nb = est.ratio_R(MD.m, nbp, [(y, y) for y in NBER_TROUGH])
    rob, why = est.robust_flag(o["label_full"], grid=g, nber=nb["label_full"])
    o.update(outcome="113: R = mean moral depreciation rate in crisis windows / within phases",
             grid_labels=";".join(sorted(set(g))), grid_share_main=float(np.mean([x == o["label_full"] for x in g])),
             grid_n=len(g), nber_label=nb["label_full"], nber_est=nb["est"], robust=rob, robust_why=why,
             lambda_labels="мера — текущая оценка: λ до 1947 г. не входят")
    out.append(o)
    lr_secular = est.single_trend(lr, D, "<")
    out.append(dict(outcome="131: secular trend ln r*_h 1948-2023 (descriptive)", **lr_secular))
    # 118 Moseley (descriptive)
    v6 = pd.read_csv(ROOT / "results" / "v6" / "series.csv")
    nfc = v6[(v6.geo == "USA_NFC") & (v6.variant == "main")].set_index("year").r.reindex(M_.index)
    mos = np.log(1 - M_.U / M_.S)
    idx = [y for y in YEARS if np.isfinite(nfc.get(y, np.nan))]
    gap = (np.log(nfc) - lr).loc[idx]
    gtr = est.single_trend(gap, D, "<")
    a_ = float(np.polyfit(np.array(idx, float), mos.loc[idx].to_numpy(), 1)[0])
    rng = np.random.default_rng(118)
    bs = []
    n = len(idx)
    t = np.array(idx, float)
    for _ in range(2000):
        ii = est.block_idx(n, 1, 5, rng)[0]
        tt = t[ii]
        aa = np.polyfit(tt, mos.loc[idx].to_numpy()[ii], 1)[0]
        gg = np.polyfit(tt, gap.to_numpy()[ii], 1)[0]
        bs.append(aa / gg if gg != 0 else np.nan)
    lo, hi = np.nanquantile(bs, [0.05, 0.95])
    show = not (gtr["ci90_lo"] <= 0 <= gtr["ci90_hi"])
    out.append(dict(outcome=f"118: share of the trend gap ln r_NFC - ln r*_h due to ln(1 - U/S$) (Moseley), "
                            f"{idx[0]}-{idx[-1]}",
                    est=a_ / gtr["est"] if show else np.nan, ci90_lo=lo if show else np.nan,
                    ci90_hi=hi if show else np.nan, label="описательно" if show else
                    "описательно: ДИ наклона разрыва включает 0 — доля не приводится", moseley_slope=a_,
                    gap_slope=gtr["est"], gap_ci90_lo=gtr["ci90_lo"], gap_ci90_hi=gtr["ci90_hi"]))
    bw, _, _ = est.within_slope(mos, PHASES)
    gw = est.within_trend(np.log(nfc) - lr, PHASES, D, "<")
    show = not (gw["ci90_lo"] <= 0 <= gw["ci90_hi"])
    out.append(dict(outcome="118: same, within phases (common slopes)", est=bw / gw["est"] if show else np.nan,
                    label="описательно" if show else "описательно: ДИ наклона разрыва включает 0 — доля не приводится",
                    moseley_slope=bw, gap_slope=gw["est"], gap_ci90_lo=gw["ci90_lo"], gap_ci90_hi=gw["ci90_hi"]))
    O = pd.DataFrame(out)
    O["no"] = O.outcome.str.extract(r"^(\d+)").astype(int)
    O.to_csv(OUT / "s1_outcomes.csv", index=False)

    # ---- money analogues for the money-counted theories [$]
    mon = {110: ("within", np.log(M_.k_hist), ">"), 111: ("within", np.log(M_.r_hist), "<"),
           114: ("within_ph", lambda ph: np.log(okishio_money(ph, "curr")), "<"),
           115: ("within_ph", lambda ph: np.log(okishio_money(ph, "hist")), "<"),
           116: ("diff", np.log(M_.e), A_PRIME, ">"), 123: ("within", np.log(M_.r_curr), "<"),
           124: ("within", np.log(M_.k_curr), ">"), 125: ("within", np.log(M_.CN_P), ">"),
           126: ("diff", np.log(M_.r_hist), PHASES["A"], ">"), 127: ("diff", np.log(M_.CN_P), PHASES["A"], "<"),
           128: ("diff", np.log(M_.e_all), A_PRIME, ">")}
    mrows = []
    for no, spec_ in mon.items():
        if spec_[0] == "within":
            r_ = est.within_trend(spec_[1], PHASES, D, spec_[2])
            r_.update(est.phase_slopes(spec_[1], PHASES))
        elif spec_[0] == "within_ph":
            yy = spec_[1](PHASES)
            r_ = est.within_trend(yy, PHASES, D, spec_[2])
            r_.update(est.phase_slopes(yy, PHASES))
        else:
            r_ = est.two_phase_diff(spec_[1], spec_[2], PHASES["B"], D, spec_[3])
        r_["no"] = no
        mrows.append(r_)
        money_est[no] = r_
    pd.DataFrame(mrows).to_csv(OUT / "s1_money_outcomes.csv", index=False)

    # ---- matrix counting
    P0 = pd.read_csv(OUT / "p0_power.csv")
    def power_note(no):
        hit = P0[P0.outcome.str.contains(rf"(?:^|\D){no}(?:\D|$)", regex=True)]
        if len(hit) and np.isfinite(hit.half_width.iloc[0]) and hit.half_width.iloc[0] >= hit.delta.iloc[0]:
            return "≈ 0: «за» недостижимо при данной мощности"
        return ""
    crow = []
    for no, preds in mx.PRED.items():
        if no >= 120 and no <= 122:
            continue
        rh = O[O.no == no].iloc[0]
        for th, (pred, basis) in preds.items():
            if pred == "молчит":
                verdict, src = "нейтрально", "—"
                e_, l_, h_, p_ = np.nan, np.nan, np.nan, np.nan
            else:
                src_r = money_est[no] if basis == "$" else rh
                e_, l_, h_ = src_r["est"], src_r["ci90_lo"], src_r["ci90_hi"]
                p_ = src_r.get("p_equal_slopes", np.nan)
                hom = True
                if no in mx.WITHIN and np.isfinite(p_):
                    hom = p_ >= 0.10
                th0 = 1.0 if no == 113 else 0.0
                verdict = est.count(pred, e_, l_, h_, rh["delta"], th0, hom)
                src = "деньги [$]" if basis == "$" else "часы"
            crow.append(dict(no=no, theory=th, prediction=pred, basis=src, est=e_, ci90_lo=l_, ci90_hi=h_,
                             p_equal_slopes=p_, verdict=verdict, note=mx.NOTES.get((no, th), ""),
                             power_note=power_note(no) if pred == "≈0" else ""))
    C_ = pd.DataFrame(crow)
    C_.to_csv(OUT / "s1_matrix.csv", index=False)
    lines = []
    for no, g_ in C_.groupby("no"):
        za = ", ".join(g_[g_.verdict == "за"].theory)
        pr = ", ".join(g_[g_.verdict == "против"].theory)
        ne = ", ".join(g_[(g_.verdict != "за") & (g_.verdict != "против") & (g_.prediction != "молчит")].theory)
        si = ", ".join(g_[g_.prediction == "молчит"].theory)
        lines.append(dict(no=no, label=O[O.no == no].label_full.iloc[0], za=za, protiv=pr, neutral_with_prediction=ne,
                          silent=si))
    pd.DataFrame(lines).to_csv(OUT / "s1_matrix_lines.csv", index=False)

    # ---- stage rule (1.5) and descriptive readings
    def rule_label(no):
        r_ = O[O.no == no].iloc[0]
        return r_["label_full"] if bool(r_.get("robust", False)) else "неинформативно"
    lab_rule = {no: rule_label(no) for no in (110, 111, 113)}
    if all(lab_rule[k] == est.CONF for k in lab_rule):
        verdict = "поддержано"
    elif any(lab_rule[k].startswith(est.REF) for k in lab_rule):
        verdict = "против"
    else:
        verdict = "неразличимо"
    l114, l115 = O[O.no == 114].label_full.iloc[0], O[O.no == 115].label_full.iloc[0]
    boundary = l114.startswith(est.REF) and (l115 == est.CONF)
    l125 = rule_label(125)
    pd.DataFrame([dict(item="labels for the rule (robust or 'неинформативно')",
                       value=";".join(f"{k}:{v}" for k, v in lab_rule.items())),
                  dict(item="stage 1 verdict (task rule 1.5)", value=verdict),
                  dict(item="method boundary (114 not falling, 115 falling; main labels)", value=boundary),
                  dict(item="reading 110 x 111 (descriptive)", value=reading_110_111(lab_rule[110], lab_rule[111])),
                  dict(item="reading 125 x 111 (descriptive)", value=reading_110_111(l125, lab_rule[111])),
                  dict(item="reading 114 x 115 (descriptive, main labels)", value=reading_114_115(l114, l115)),
                  dict(item="labels 114; 115 (main)", value=f"{l114}; {l115}")]).to_csv(OUT / "s1_rule.csv", index=False)

    # ---- identity decomposition (descriptive): ln r = ln e - ln(1+k); ln k = ln(C/L) + ln(1+e)
    dec = []
    for nm, yy in (("ln r*_h", np.log(ms.r)), ("ln e*_h", np.log(ms.e)), ("ln(1 + k*_h)", np.log(1 + ms.k)),
                   ("ln k*_h", np.log(ms.k)), ("ln(C^(В) / L)", np.log(ms.CL)), ("ln(1 + e*_h)", np.log(1 + ms.e)),
                   ("ln k*_h at (P)", np.log(ms.k_P))):
        r_ = est.within_trend(yy, PHASES, D, "<")
        r_.update(est.phase_slopes(yy, PHASES))
        dec.append(dict(series=nm, **r_))
    pd.DataFrame(dec).to_csv(OUT / "s1_identity.csv", index=False)

    # ---- Shapley
    stats_, lr_bits = {}, {}
    for b_, g_ in S_.groupby("bits"):
        g_ = g_.set_index("year").sort_index()
        lrr = np.log(g_.r)
        lr_bits[b_] = lrr
        t = np.arange(len(lrr), dtype=float)
        stats_[b_] = dict(within=est.within_slope(lrr, PHASES)[0], linear=float(np.polyfit(t, lrr, 1)[0]),
                          level=float(lrr.mean()))
    sh, inter = [], []
    ser = shapley_series(lr_bits)
    for st in ("within", "linear", "level"):
        phi, steps, pair = shapley({k: v[st] for k, v in stats_.items()})
        for i, sw in enumerate(SWITCHES):
            r_ = dict(statistic=st, switch=sw, shapley=phi[i], fixed_order_step=steps[i])
            if st == "within":
                w_ = est.within_trend(ser[sw], PHASES, D, "<")
                r_.update(ci90_lo=w_["ci90_lo"], ci90_hi=w_["ci90_hi"], check_linear=w_["est"])
            sh.append(r_)
        sh.append(dict(statistic=st, switch="total (13a - 12b)", shapley=stats_["111111"][st] - stats_["000000"][st],
                       fixed_order_step=stats_["111111"][st] - stats_["000000"][st]))
        for (a, b), val in pair.items():
            inter.append(dict(statistic=st, pair=f"{a}×{b}", harsanyi_dividend=val))
    pd.DataFrame(sh).to_csv(OUT / "s1_shapley.csv", index=False)
    pd.DataFrame(inter).to_csv(OUT / "s1_shapley_pairs.csv", index=False)
    pd.DataFrame(stats_).T.to_csv(OUT / "s1_config_stats.csv")

    # ---- descriptive (1.6): money, energy, manufacturing, phase slopes, 113 extras, variants
    desc = []
    E_ = pd.DataFrame([energy_year(y) for y in YEARS if y >= 1949]).set_index("year")
    F_ = pd.DataFrame([manufacturing_year(y) for y in YEARS]).set_index("year")
    E_.to_csv(OUT / "s1_energy.csv")
    F_.to_csv(OUT / "s1_manufacturing.csv")
    for nm, ser_, d_ in (("money, historical: ln r*_$", np.log(M_.r_hist), "<"),
                         ("money, current: ln r_$", np.log(M_.r_curr), "<"),
                         ("money, historical: ln k*_$", np.log(M_.k_hist), ">"),
                         ("energy (current): ln r_E", np.log(E_.r_E), "<"), ("energy (current): ln k_E", np.log(E_.k_E), ">"),
                         ("manufacturing: ln r*", np.log(F_.r_mfg), "<"), ("manufacturing: ln k*", np.log(F_.k_mfg), ">"),
                         ("ln e*_h", np.log(ms.e), ">"), ("110 at (P): ln k*_h(P)", np.log(ms.k_P), ">")):
        r_ = est.within_trend(ser_, PHASES, D, d_)
        r_.update(est.phase_slopes(ser_, PHASES))
        desc.append(dict(item=f"within phases: {nm}", **r_))
    for nm, yy in (("116 slopes", np.log(ms.e)), ("128 slopes", np.log(allh.e))):
        r_ = est.two_phase_diff(yy, A_PRIME, PHASES["B"], D, ">")
        desc.append(dict(item=f"{nm}: beta_A' and beta_B", est=r_["est"], slope_A=r_["slope_A"], slope_B=r_["slope_B"]))
    for sp in SPECS:
        r_ = est.two_phase_diff(np.log(spec[sp].e), A_PRIME, PHASES["B"], D, ">")
        desc.append(dict(item=f"116 [{sp}]", **r_))
    # 113 extras
    for nm, mm in (("113 R without hedonic assets", MD.m_no_hedonic),):
        desc.append(dict(item=nm, **est.ratio_R(mm, PHASES, CRISES)))
    for part in ("m_productivity", "m_relative_price"):
        cr = np.any([(MD.index >= a) & (MD.index <= b) for a, b in CRISES], axis=0)
        ph_ = np.any([(MD.index >= a) & (MD.index <= b) for a, b in PHASES.values()], axis=0)
        desc.append(dict(item=f"113 split: mean {part} in crisis windows / within phases",
                         mean_crisis=float(MD[part][cr].mean()), mean_phase=float(MD[part][ph_].mean())))
    desc.append(dict(item="113 R by NBER trough years (all other years as non-crisis)", **nb))
    grp = pd.Series("2022-2023", index=MD.index)
    for k_, (a, b) in PHASES.items():
        grp[(MD.index >= a) & (MD.index <= b)] = k_
    for a, b in CRISES:
        grp[(MD.index >= a) & (MD.index <= b)] = f"{a}-{b}"
    X = pd.get_dummies(grp).astype(float)
    X["trough"] = MD.index.isin(NBER_TROUGH).astype(float)
    fit = sm.OLS(MD.m.to_numpy(float), X.to_numpy(float)).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
    gam, se_ = float(fit.params[-1]), float(fit.bse[-1])
    desc.append(dict(item="113 regression m_t = a_group + gamma 1[NBER trough year]", est=gam,
                     ci90_lo=gam - est.Z90 * se_, ci90_hi=gam + est.Z90 * se_))
    variants = {"all hours (who=all)": dict(cfg_key=config_key({**MAIN, "who": "all"})),
                "rho_h = 1": dict(ces_variant="rho=1"),
                "pre-1964 shares with goods trend": dict(ces_variant="pre-1964 goods trend"),
                "agriculture = private shares": dict(ces_variant="agriculture = private"),
                "CES unlinked": dict(ces_variant="unlinked"),
                "gross lambda": dict(cfg_key=config_key({**MAIN, "dep": "gross"}))}
    for vn, kw in variants.items():
        V_ = main_series(**kw)
        for c_, d_ in (("k", ">"), ("r", "<")):
            desc.append(dict(item=f"variant [{vn}]: within phases ln {c_}",
                             **est.within_trend(np.log(V_[c_]), PHASES, D, d_)))
        desc.append(dict(item=f"variant [{vn}]: 115 Okishio historical",
                         **est.within_trend(np.log(okishio(V_, PHASES, "hist", pcons)), PHASES, D, "<")))
        desc.append(dict(item=f"variant [{vn}]: 117 ln r(B)/r(P)",
                         **est.within_trend(np.log(V_.r / V_.r_P), PHASES, 0.001, "<")))
    pd.DataFrame(desc).to_csv(OUT / "s1_descriptive.csv", index=False)
    ms.to_csv(OUT / "s1_main_series.csv")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 70)
    print(O[["outcome", "est", "ci90_lo", "ci90_hi", "label_full", "robust"]].round(5).to_string())
    print(pd.read_csv(OUT / "s1_rule.csv").to_string())


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else "all"
    if part in ("spec", "all"):
        shapley_spec()
    if part in ("series", "all"):
        series()
    if part in ("tests", "all"):
        tests()
