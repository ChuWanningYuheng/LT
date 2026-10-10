"""Iteration 12, stage 6: rent with prices of production on the capital STOCK (pre_registration_v12.md, stage 6).

p = p M + mu + r (p K_d + 1 K_m) + w b, K_d = g_d (x) k, K_m = g_m (x) k; k_j = (K/VA)_KLEMS group * VA_j / x_j, split within
a KLEMS group by consumption of fixed capital; r = (VA - labour income - CFC) / sum K; w from sum p x = sum x.
d_j = -ln p_j. Outcomes 84 (A01 and B), 85 (L), 86 (elasticity of d(B) on real energy+metals index).

Usage: PYTHONPATH=src python -P -m lts.v12.stage6 [fallback]
Outputs: results/v12/s6_*.csv (variant with coarser KLEMS groups, journal 9: s6_*_fallback.csv)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import spectral_radius
from ..economy import build
from ..v6.series import ROOT
from ..v8.rule import decide, mean_row
from ..v10 import stage10 as s10

OUT = ROOT / "results" / "v12"
R9B = ROOT / "data" / "raw" / "v9b"
COUNTRIES = {"USA": "US", "DEU": "DE", "FRA": "FR", "ITA": "IT", "ESP": "ES", "NLD": "NL", "AUT": "AT", "POL": "PL",
             "CZE": "CZ", "JPN": "JP", "GBR": "UK"}
YEARS = range(2010, 2022)
FIG2KLEMS = {"A01": "A", "A02": "A", "A03": "A", "B": "B", "C10-12": "C10-C12", "C13-15": "C13-C15", "C16": "C16-C18",
             "C17": "C16-C18", "C18": "C16-C18", "C19": "C19", "C20": "C20", "C21": "C21", "C22": "C22-C23",
             "C23": "C22-C23", "C24": "C24-C25", "C25": "C24-C25", "C26": "C26", "C27": "C27", "C28": "C28",
             "C29": "C29-C30", "C30": "C29-C30", "C31_32": "C31-C33", "C33": "C31-C33", "D35": "D", "E36": "E",
             "E37-39": "E", "F": "F", "G45": "G45", "G46": "G46", "G47": "G47", "H49": "H49", "H50": "H50", "H51": "H51",
             "H52": "H52", "H53": "H53", "I": "I", "J58": "J58-J60", "J59_60": "J58-J60", "J61": "J61", "J62_63": "J62-J63",
             "K64": "K", "K65": "K", "K66": "K", "L": "L", "L68A": "L", "L68B": "L", "M69_70": "M", "M71": "M", "M72": "M",
             "M73": "M", "M74_75": "M", "N77": "N", "N78": "N", "N79": "N", "N80-82": "N", "O84": "O", "P85": "P",
             "Q86": "Q", "Q87_88": "Q", "R90-92": "R", "R93": "R", "S94": "S", "S95": "S", "S96": "S", "T": "T"}
RENT = ["A01", "B"]
# journal 9 (after the result): coarser KLEMS groups tried in order when the fine group has no data
INTER = {"C20": "C20-C21", "C21": "C20-C21", "C26": "C26-C27", "C27": "C26-C27", "Q86": "Q", "Q87_88": "Q"}
COMBINED = {"D": "D-E", "E": "D-E", "M": "M-N", "N": "M-N", "O": "O-Q", "P": "O-Q", "Q": "O-Q", "R": "R-S", "S": "R-S"}


def chain(code):
    out = [FIG2KLEMS.get(code), INTER.get(code), code[0], COMBINED.get(code[0])]
    return [g for i, g in enumerate(out) if g and g not in out[:i]]


def klems():
    na = pd.read_csv(R9B / "klems2025_national_accounts.csv", usecols=["nace_r2_code", "geo_code", "year", "VA_CP"],
                     low_memory=False)
    ca = pd.read_csv(R9B / "klems2025_capital_accounts.csv", usecols=["nace_r2_code", "geo_code", "year", "K_GFCF"],
                     low_memory=False)
    m = na.merge(ca, on=["nace_r2_code", "geo_code", "year"])
    m = m[m.geo_code.isin(COUNTRIES.values())]
    m["kva"] = m.K_GFCF / m.VA_CP
    return m.set_index(["geo_code", "nace_r2_code", "year"]).kva


def resolve(code, kva, g2, y):
    for g in chain(code):
        r = kva.get((g2, g, y), np.nan)
        if np.isfinite(r) and r > 0:
            return g
    return None


def stock_coeffs(e, kva, g2, y, fallback=False):
    """k_j (capital per unit of output, FIGARO units) for each FIGARO label; NaN where unmapped"""
    n = e.n
    groups = []
    for lab in e.labels:
        if fallback and lab not in ("T", "U"):
            gs = sorted({resolve(m, kva, g2, y) for m in lab.split("+")} - {None})
        else:
            gs = sorted({FIG2KLEMS.get(m) for m in lab.split("+")} - {None})
        groups.append(gs)
    K = np.zeros(n)
    ok = np.ones(n, bool)
    allg = sorted({g for gs in groups for g in gs})
    for g in allg:
        r = kva.get((g2, g, y), np.nan)
        mem = [i for i, gs in enumerate(groups) if g in gs]
        share = np.array([1.0 / len(groups[i]) for i in mem])
        va_g = float((e.va[mem] * share).sum())
        cfc_w = e.cfc[mem] * share
        if not np.isfinite(r) or r <= 0 or va_g <= 0:
            ok[mem] = False
            continue
        Kg = r * va_g
        w = cfc_w / cfc_w.sum() if cfc_w.sum() > 0 else share * e.va[mem] / (share * e.va[mem]).sum()
        K[mem] += Kg * w
    for i, gs in enumerate(groups):
        if not gs:
            ok[i] = False
    k = np.divide(K, e.x, out=np.zeros(n), where=e.x > 0)
    return k, ok


def prices_stock(e, k):
    M, Mm, _ = e.system(True, False, "price", e.l())
    n = e.n
    tot = e.gfcf_dom.sum() + e.gfcf_imp.sum()
    gd, gm = e.gfcf_dom / tot, e.gfcf_imp / tot
    Kd = np.outer(gd, k)
    km = gm.sum() * k
    li = e.labour_income if e.labour_income is not None else e.wages
    Ktot = (k * e.x).sum()
    r = float((e.va.sum() - li.sum() - e.cfc.sum()) / Ktot)
    T = M + r * Kd
    rho = spectral_radius(T)
    if rho >= 1:
        raise ValueError(f"spectral radius {rho:.3f}")
    Linv = np.linalg.inv(np.eye(n) - T)
    a = (Mm.sum(0) + r * km) @ Linv
    b = np.divide(li, e.x, out=np.zeros(n), where=e.x > 0) @ Linv
    w = (e.x.sum() - a @ e.x) / (b @ e.x)
    p = a + w * b
    return p, r, rho


def run(fallback=False):
    sfx = "_fallback" if fallback else ""
    kva = klems()
    rows, skipped = [], []
    for c, g2 in COUNTRIES.items():
        for y in YEARS:
            e, _ = build(c, y)
            k, ok = stock_coeffs(e, kva, g2, y, fallback)
            try:
                p, r, rho = prices_stock(e, k)
            except ValueError as exc:
                skipped.append(dict(country=c, year=y, reason=str(exc)))
                continue
            d = -np.log(np.where(p > 0, p, np.nan))
            rows.append(pd.DataFrame(dict(country=c, year=y, industry=e.labels, x=e.x, k=k, mapped=ok, p=p, d=d, r=r,
                                          rho=rho)))
            print("s6", c, y, round(r, 4), flush=True)
    D = pd.concat(rows, ignore_index=True)
    D.to_csv(OUT / f"s6_prices_stock{sfx}.csv", index=False)
    pd.DataFrame(skipped).to_csv(OUT / f"s6_skipped{sfx}.csv", index=False)
    D = D[D.mapped & ~D.industry.isin(["T", "U"]) & np.isfinite(D.d)]
    out, desc = [], []
    s84 = D[D.industry.isin(RENT)].groupby(["country", "year"]).d.mean().groupby(level=0).mean()
    r84 = mean_row("84: mean d = ln(p/pp_stock) of A01 and B (FIGARO x KLEMS, 2010-2021)", s84.to_numpy(), 0, 0.05, ">")
    # placebo: 1000 random pairs of non-rent industries present in every country-year
    common = set.intersection(*[set(g.industry) for _, g in D.groupby(["country", "year"])])
    pool = sorted(i for i in common if not i.startswith(("A", "B", "L", "T", "U")) and "+" not in i)
    rng = np.random.default_rng(612)
    pl = []
    for _ in range(1000):
        pick = list(rng.choice(pool, 2, replace=False))
        pl.append(D[D.industry.isin(pick)].groupby(["country", "year"]).d.mean().groupby(level=0).mean().mean())
    pl = np.array(pl)
    r84.update(placebo_percentile=float((pl < r84["est"]).mean()), placebo_median=float(np.median(pl)), pool=len(pool),
               d_A01=float(D[D.industry == "A01"].groupby("country").d.mean().mean()),
               d_B=float(D[D.industry == "B"].groupby("country").d.mean().mean()))
    out.append(r84)
    sL = D[D.industry == "L"].groupby("country").d.mean()
    out.append(mean_row("85: mean d of real estate L (stock-based prices of production)", sL.to_numpy(), 0, 0.05, ">"))
    ci = s10.commodity_index()
    Bd = D[D.industry == "B"][["country", "year", "d"]].merge(ci, left_on="year", right_index=True)
    r86 = s10.fe_wild(Bd, "d", "ln_em", 86)
    out.append(dict(outcome="86: elasticity of d(B) w.r.t. ln real energy+metals index (stock-based pp, country FE)",
                    theta0=0, delta=0.05, direction=">", **r86))
    # descriptive: d by industry, d(L) and real house prices
    by = D.groupby("industry").agg(d=("d", "mean"), k=("k", "mean"), n=("d", "size")).sort_values("d")
    by.to_csv(OUT / f"s6_d_by_industry{sfx}.csv")
    hp = s10.bis_real().reset_index()
    iso2 = {"USA": "US", "DEU": "DE", "FRA": "FR", "ITA": "IT", "ESP": "ES", "NLD": "NL", "AUT": "AT", "POL": "PL",
            "CZE": "CZ", "JPN": "JP", "GBR": "GB"}
    Ld = D[D.industry == "L"][["country", "year", "d"]].copy()
    Ld["geo"] = Ld.country.map(iso2)
    Ld = Ld.merge(hp, on=["geo", "year"], how="left")
    Ld["ln_hp"] = np.log(Ld.hp_real)
    try:
        rL = s10.fe_wild(Ld, "d", "ln_hp", 85)
        desc.append(dict(item="d(L) on ln real house price (BIS), country FE", **rL))
    except Exception as exc:                                          # noqa: BLE001
        desc.append(dict(item=f"d(L) on house prices failed: {exc}"))
    desc.append(dict(item="actual profit rate on the stock r: mean over country-years", est=float(D.groupby(["country", "year"]).r.first().mean())))
    O = pd.DataFrame(out)
    O.to_csv(OUT / f"s6_outcomes{sfx}.csv", index=False)
    pd.DataFrame(desc).to_csv(OUT / f"s6_descriptive{sfx}.csv", index=False)
    pd.set_option("display.width", 250)
    print(O.round(4).to_string())
    print(pd.DataFrame(desc).round(4).to_string())
    print(by.round(3).to_string())


if __name__ == "__main__":
    import sys
    run(fallback="fallback" in sys.argv[1:])
