"""Iteration 12b: the rate of profit in hours for other countries (stage 1.6: WIOD 2000-2014, FIGARO 2010-2022;
outcomes 95-96) and the world rate in hours on the WIOD world system (stage 2, outcome 100).

Same definitions as the US (stage1.py): total-use (competitive imports) open system; n = x - A_tot x;
r_X = (X_total - lambda_X b) / (lambda_X K + lambda_X b); basket (i) classical: productive compensation spent on the
household consumption composition; K composition by the GFCF column (WIOD) or by EU KLEMS asset types (FIGARO).

Usage: PYTHONPATH=src python -P -m lts.v12b.stage1_world [wiod|figaro|world|tests|all]
Outputs: results/v12b/s1w_*.csv, s2_world_hours*.csv
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from ..v6.series import ROOT
from ..v8.rule import decide

OUT = ROOT / "results" / "v12b"
WIOD_UNPROD = {"G45", "G46", "G47", "K64", "K65", "K66", "L68", "M69_M70", "N", "O84", "T", "U"}
FIG_UNPROD = {"G45", "G46", "G47", "K64", "K65", "K66", "L", "L68A", "L68B", "M69_70", "N77", "N78", "N79", "N80-82",
              "O84", "T", "U"}
FIG_COUNTRIES = {"USA": "US", "DEU": "DE", "FRA": "FR", "ITA": "IT", "ESP": "ES", "NLD": "NL", "AUT": "AT", "POL": "PL",
                 "CZE": "CZ", "JPN": "JP", "GBR": "UK"}
KLEMS_PROD = ["A", "B", "C", "D-E", "F", "H", "I", "J", "M", "P", "Q", "R-S"]
ASSET_TO_FIG = {"IT": ["C26"], "CT": ["C26"], "TraEq": ["C29", "C30"], "OMach": ["C28"], "Rstruc": ["F"], "OCon": ["F"],
                "Soft_DB": ["J62_63"], "RD": ["M72"], "OIPP": ["J58"], "Cult": ["A01"]}


def rates(A_tot, x, l, b, K, extra_rows=None):
    """Values in units: hours (row l), dollar (=1), and any extra rows; returns dict of S, V, C, e, k, r."""
    n = x - A_tot @ x
    Linv = np.linalg.inv(np.eye(len(x)) - A_tot)
    rows = {"h": l}
    rows.update(extra_rows or {})
    out = {}
    for X, row in rows.items():
        lam = row @ Linv
        tot = float(row @ x)
        V, C = float(lam @ b), float(lam @ K)
        out.update({f"V_{X}": V, f"C_{X}": C, f"S_{X}": tot - V, f"tot_{X}": tot, f"ident_{X}": float(lam @ n) / tot - 1})
    V, C, tot = float(b.sum()), float(K.sum()), float(n.sum())
    out.update({"V_$": V, "C_$": C, "S_$": tot - V, "tot_$": tot})
    for X in list(rows) + ["$"]:
        S, V, C = out[f"S_{X}"], out[f"V_{X}"], out[f"C_{X}"]
        out.update({f"e_{X}": S / V, f"k_{X}": C / V, f"r_{X}": S / (C + V)})
    return out


# ================================================================ WIOD national systems
def wiod_country_year(c, y):
    from ..revisit import wiod
    e = wiod.economy(c, y)
    hours = e.hours.copy()
    if c == "CHN":
        from .stage0 import avh_china
        s = wiod.sea()
        s = s[(s.country == c) & (s.year == y) & (s.variable == "EMP")].set_index("code").v.reindex(e.labels)
        hours = np.nan_to_num(s.to_numpy()) * avh_china(y) / 1000
    prod = np.array([lb not in WIOD_UNPROD for lb in e.labels])
    x = e.x
    if hours[prod].sum() <= 0:
        return None
    l = np.divide(hours * prod, x, out=np.zeros(len(x)), where=x > 0)
    hh = e.hh_dom + e.hh_imp
    b = float((e.wages * prod).sum()) * hh / hh.sum()
    g = e.gfcf_dom + e.gfcf_imp
    K = float((e.meta["K"] * prod).sum()) * g / g.sum()
    A_tot = e.A + e.Am
    r = rates(A_tot, x, l, b, K)
    return dict(country=c, year=y, **r)


def run_wiod():
    from ..revisit import wiod
    W = wiod.wiot(2010)
    countries = [c for c in W["countries"] if c != "ROW"]
    rows = []
    for y in wiod.YEARS:
        for c in countries:
            try:
                r = wiod_country_year(c, y)
            except Exception as exc:                                   # noqa: BLE001
                print("fail", c, y, exc, flush=True)
                continue
            if r:
                rows.append(r)
        print("wiod", y, flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "s1w_wiod.csv", index=False)
    return D


# ================================================================ FIGARO + KLEMS + PEFA
def klems_assets():
    k = pd.read_csv(ROOT / "data" / "raw" / "v9b" / "klems2025_capital_accounts.csv", low_memory=False)
    na = pd.read_csv(ROOT / "data" / "raw" / "v9b" / "klems2025_national_accounts.csv", low_memory=False,
                     usecols=["nace_r2_code", "geo_code", "year", "VA_CP"])
    return k, na


def pefa():
    d = json.load(open(ROOT / "data" / "raw" / "v12b" / "es_env_ac_pefa04.json"))
    dims, size = d["id"], d["size"]
    cats = {k: list(d["dimension"][k]["category"]["index"].keys()) for k in dims}
    idx = {k: d["dimension"][k]["category"]["index"] for k in dims}
    rows = []
    strides = np.cumprod([1] + size[::-1])[::-1][1:]
    for pos, v in d["value"].items():
        p = int(pos)
        coord = {}
        for k, st in zip(dims, strides):
            coord[k] = p // st
            p %= st
        lab = {k: cats[k][coord[k]] for k in dims}
        rows.append(dict(nace=lab["nace_r2"], indic=lab["indic_pefa"], geo=lab["geo"], year=int(lab["time"]), v=v))
    return pd.DataFrame(rows)


def figaro_country_year(c, y, K_assets, VAk, P):
    from ..economy import build
    e, _ = build(c, y)
    labs = e.labels
    prod = np.array([not (set(lb.split("+")) & FIG_UNPROD) for lb in labs])
    x = e.x
    l = np.divide(e.hours * prod, x, out=np.zeros(len(x)), where=x > 0)
    hh = e.hh_dom + e.hh_imp
    b = float((e.wages * prod).sum()) * hh / hh.sum()
    # capital by asset type (productive KLEMS sections), converted to FIGARO units by VA ratio
    K = np.zeros(len(x))
    conv = float(e.va.sum()) / VAk if VAk > 0 else np.nan
    for asset, tgt in ASSET_TO_FIG.items():
        val = K_assets.get(asset, np.nan)
        if not np.isfinite(val) or val <= 0:
            continue
        idx = [i for i, lb in enumerate(labs) if set(lb.split("+")) & set(tgt)]
        if not idx:
            continue
        w = x[idx] / x[idx].sum()
        K[idx] += val * conv * w
    A_tot = e.A + e.Am
    extra = {}
    if P is not None and len(P):
        en = np.zeros(len(x))
        ok = 0
        for i, lb in enumerate(labs):
            for part in lb.split("+"):
                key = {"M69_70": "M69_M70", "J59_60": "J59_J60", "J62_63": "J62_J63", "C31_32": "C31_C32",
                       "N80-82": "N80-N82", "E37-39": "E37-E39", "R90-92": "R90-R92", "C10-12": "C10-C12",
                       "C13-15": "C13-C15", "M74_75": "M74_M75", "Q87_88": "Q87_Q88", "L68A": "L68A", "L68B": "L68B"}.get(part, part)
                if key in P.index:
                    en[i] += P[key]
                    ok += 1
        if ok > len(labs) * 0.5:
            extra["E"] = np.divide(en, x, out=np.zeros(len(x)), where=x > 0)
    r = rates(A_tot, x, l, b, K, extra)
    return dict(country=c, year=y, K_fig=float(K.sum()), **r)


def run_figaro():
    k, na = klems_assets()
    pe = pefa()
    pe = pe[pe.indic == "NETDOM_EUSE"]
    rows = []
    for c, g2 in FIG_COUNTRIES.items():
        for y in range(2010, 2023):
            kk = k[(k.geo_code == g2) & (k.year == y) & k.nace_r2_code.isin(KLEMS_PROD)]
            if kk.empty:
                continue
            assets = {a: float(kk[f"K_{a}"].sum(min_count=1)) for a in ASSET_TO_FIG}
            vak = float(na[(na.geo_code == g2) & (na.year == y) & (na.nace_r2_code == "TOT")].VA_CP.sum())
            p2 = {"UK": "UK"}.get(g2, g2)
            P = pe[(pe.geo == p2) & (pe.year == y)].set_index("nace").v
            try:
                r = figaro_country_year(c, y, assets, vak, P if len(P) else None)
            except Exception as exc:                                   # noqa: BLE001
                print("fail", c, y, exc, flush=True)
                continue
            rows.append(r)
        print("figaro", c, flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "s1w_figaro.csv", index=False)
    return D


# ================================================================ world system (outcome 100)
def run_world():
    from ..v9 import stage34 as s34
    from ..revisit import wiod
    from .stage0 import make_year_system
    ys = make_year_system("a")
    rows = []
    for y in wiod.YEARS:
        A, cells = ys(y)
        W = wiod.wiot(y)
        countries = W["countries"]
        Nn = wiod.N
        x = cells.x.to_numpy()
        isrow = (cells.country == "ROW").to_numpy()
        prod = ~cells.code.isin(WIOD_UNPROD).to_numpy() & ~isrow
        l = cells.H_x.to_numpy() * prod
        s = s34.sea_wide()
        s = s[s.year == y].set_index(["country", "code"])
        key = pd.MultiIndex.from_arrays([cells.country, cells.code])
        comp = (s.COMP * s.xr).reindex(key).to_numpy()
        Kc = (s.K * s.xr).reindex(key).to_numpy()
        b = np.zeros(len(x))
        K = np.zeros(len(x))
        F57, F60 = W["F"][57], W["F"][60]
        for ci, c in enumerate(countries):
            if c == "ROW":
                continue
            m = (cells.country == c).to_numpy()
            wp = float(np.nansum(comp[m & prod]))
            kp = float(np.nansum(Kc[m & prod]))
            hh, gf = F57[:, ci], np.clip(F60[:, ci], 0, None)
            if hh.sum() > 0:
                b += wp * hh / hh.sum()
            if gf.sum() > 0:
                K += kp * gf / gf.sum()
        r = rates(A, x, l, b, K)
        rows.append(dict(year=y, **r))
        print("world", y, flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "s2_world_hours.csv", index=False)
    return D


# ================================================================ tests
def panel_trend(D, col, direction):
    from ..v11.stage1 import reg
    d = D[["country", "year", col]].dropna().copy()
    d = d[d[col] > 0]
    d["y"] = np.log(d[col])
    d["t"] = d.year - d.year.min()
    r = reg(d, "y", ["t"], 0, fe=("country",), cl="country")
    r.update(theta0=0.0, delta=0.002, direction=direction, label=decide(r["est"], r["ci90_lo"], r["ci90_hi"], 0, 0.002, direction))
    return r


def tests():
    from .stage1 import trend
    out, desc = [], []
    W = pd.read_csv(OUT / "s1w_wiod.csv")
    W = W[(W.r_h > 0) & (W["r_$"] > 0)]
    W["rh_rd"] = W.r_h / W["r_$"]
    out.append(dict(outcome="95: WIOD 2000-2014, within-country trend of ln r_h (country FE, clusters countries)",
                    **panel_trend(W, "r_h", "<")))
    out.append(dict(outcome="96: WIOD, within-country trend of ln(r_h / r_$)", **panel_trend(W, "rh_rd", "<")))
    for col, d_ in (("r_$", "<"), ("k_h", ">"), ("e_h", ">"), ("k_$", ">")):
        desc.append(dict(item=f"WIOD within-country trend of ln {col}", **panel_trend(W, col, d_)))
    for c, g in W.groupby("country"):
        g = g.set_index("year").sort_index()
        if len(g) >= 10:
            t = trend(np.log(g.r_h), 4, 0.002, "<")
            desc.append(dict(item=f"WIOD {c}: trend ln r_h", **t))
    F = pd.read_csv(OUT / "s1w_figaro.csv")
    F = F[(F.r_h > 0) & (F["r_$"] > 0)]
    F["rh_rd"] = F.r_h / F["r_$"]
    for col, d_ in (("r_h", "<"), ("r_$", "<"), ("rh_rd", "<"), ("k_h", ">")):
        desc.append(dict(item=f"FIGARO 2010-2022 within-country trend of ln {col}", **panel_trend(F, col, d_)))
    if "r_E" in F:
        FE = F[F.r_E.notna() & (F.r_E > 0)].copy()
        if FE.country.nunique() >= 3:
            FE["rh_rE"] = FE.r_h / FE.r_E
            desc.append(dict(item="FIGARO (PEFA countries) within-country trend of ln r_E", **panel_trend(FE, "r_E", "<")))
            desc.append(dict(item="FIGARO (PEFA countries) within-country trend of ln(r_h / r_E)", **panel_trend(FE, "rh_rE", "<")))
    Wd = pd.read_csv(OUT / "s2_world_hours.csv").set_index("year")
    o100 = trend(np.log(Wd.r_h), 4, 0.002, "<")
    out.append(dict(outcome="100: WIOD world system (China hours imputed), trend ln r_h, 2000-2014", **o100))
    desc.append(dict(item="WIOD world system: trend ln r_$", **trend(np.log(Wd["r_$"]), 4, 0.002, "<")))
    desc.append(dict(item="WIOD world system: trend ln(r_h / r_$)", **trend(np.log(Wd.r_h / Wd["r_$"]), 4, 0.002, "<")))
    O = pd.DataFrame(out)
    O.to_csv(OUT / "s1w_outcomes.csv", index=False)
    Dd = pd.DataFrame(desc)
    Dd.to_csv(OUT / "s1w_descriptive.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 80)
    print(O.round(5).to_string())
    print(Dd.round(5).to_string())


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else "all"
    if part in ("wiod", "all"):
        run_wiod()
    if part in ("figaro", "all"):
        run_figaro()
    if part in ("world", "all"):
        run_world()
    if part in ("tests", "all"):
        tests()
