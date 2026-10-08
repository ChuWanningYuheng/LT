"""Iteration 10, stage 4: uniform data checks of the labour blocks (pre_registration_v10.md, stage 4; journal 5).

Flags R1-R4 for the 13 FIGARO countries (OECD labour blocks, 2010-2023) and WIOD SEA (2000-2014); source check
(imputation log, second source on the overlap 2010-2014, time pattern of R4); recompute outcomes 1, 3, 15, 28, 44
(a) without flagged industry-years in the evaluation, (b) with corrections of identified data errors; outcome 55.

Usage: PYTHONPATH=src python -P -m lts.v10.stage4 [flags|figaro|wiod15|wiod28|summary ...]
Outputs: results/v10/s4_*.csv
"""
from __future__ import annotations

import copy
import sys

import numpy as np
import pandas as pd

from ..economy import EXCLUDE_BASE, build
from ..levels import mask_for
from ..metrics import ratio_metrics
from ..v6.series import ROOT
from ..v8.rule import mean_row, share_row

OUT = ROOT / "results" / "v10"
OUT.mkdir(parents=True, exist_ok=True)
TAB = ROOT / "results" / "tables"
V3 = ("USA", "DEU", "MEX", "FRA", "ITA", "ESP", "NLD", "AUT", "POL", "CZE", "KOR", "JPN", "GBR")
FIG_YEARS = range(2010, 2024)
O3_YEARS = range(2010, 2023)
W_YEARS = range(2000, 2015)
FIG2WIOD = {"C10-12": "C10-C12", "C13-15": "C13-C15", "C31_32": "C31_C32", "E37-39": "E37-E39", "J59_60": "J59_J60",
            "J62_63": "J62_J63", "L": "L68", "M69_70": "M69_M70", "M74_75": "M74_M75"}
NOT_1TO1 = {"N77", "N78", "N79", "N80-82", "Q86", "Q87_88", "R90-92", "R93", "S94", "S95", "S96"}
WIOD2FIG = {v: k for k, v in FIG2WIOD.items()}
NOT_1TO1_W = {"N", "Q", "R_S"}


def fig2wiod(lab):
    if "+" in lab or lab in NOT_1TO1:
        return None
    return FIG2WIOD.get(lab, lab)


# ================================================================ cells
def figaro_cells():
    p = OUT / "s4_figaro_cells.csv"
    if p.exists():
        return pd.read_csv(p)
    rows = []
    for c in V3:
        for y in FIG_YEARS:
            try:
                e, info = build(c, y)
            except Exception as ex:                                  # noqa: BLE001
                print("FIGARO skip", c, y, repr(ex)[:80], flush=True)
                continue
            lg = info["labour_log"]
            split = set(lg.get("split_kids", {})) | set(lg.get("split_fallback", []))
            hppfb = set(lg.get("hpp_fallback_list", []))
            for j, lab in enumerate(e.labels):
                mem = lab.split("+")
                rows.append(dict(source="FIGARO", country=c, year=y, code=lab, x=e.x[j], hours=e.hours[j],
                                 persons=e.meta["persons"][j], li=e.labour_income[j],
                                 eval=not any(m in EXCLUDE_BASE for m in mem),
                                 imputed_split=any(m in split for m in mem), imputed_hpp=any(m in hppfb for m in mem)))
            print("FIGARO cells", c, y, flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(p, index=False)
    return d


def wiod_cells():
    from ..v9.stage34 import sea_wide
    s = sea_wide()
    s = s[s.year.between(min(W_YEARS), max(W_YEARS))]
    H = s.H_EMPE * np.where(s.EMPE > 0, s.EMP / s.EMPE, 1.0)
    return pd.DataFrame(dict(source="WIOD", country=s.country, year=s.year, code=s.code, x=s.GO, hours=H,
                             persons=s.EMP, li=s.LAB, eval=~s.code.isin(["T", "U"]), imputed_split=False,
                             imputed_hpp=False)).reset_index(drop=True)


# ================================================================ flags R1-R4
def flags(df):
    df = df.copy()
    df["hours"] = df.hours.fillna(0.0)
    df["persons"] = df.persons.fillna(0.0)
    df["x"] = df.x.fillna(0.0)
    pos = df[df.hours > 0]
    tot = pos.groupby(["country", "year"]).agg(H=("hours", "sum"), LI=("li", "sum"))
    df = df.join(tot, on=["country", "year"])
    with np.errstate(divide="ignore", invalid="ignore"):
        df["w_rel"] = (df.li / df.hours) / (df.LI / df.H)
    df["h_share"] = df.hours / df.H
    ev = df["eval"].astype(bool)
    df["R1"] = ev & (df.hours > 0) & ((df.w_rel > 3) | (df.w_rel < 1 / 3))
    df["R2"] = ev & (df.x > 0) & (df.h_share < 0.002)
    df["R3"] = ev & (df.x > 0) & ((df.hours <= 0) | (df.persons <= 0))
    df = df.sort_values(["source", "country", "code", "year"]).reset_index(drop=True)
    ev = df["eval"].astype(bool)                                  # re-aligned after sorting (journal 9)
    df["hpp"] = (df.hours / df.persons).where((df.hours > 0) & (df.persons > 0))
    g = df.groupby(["source", "country", "code"])
    py, ny = g.year.shift(1), g.year.shift(-1)
    prev = g.hpp.shift(1).where(df.year - py == 1)
    nxt = g.hpp.shift(-1).where(ny - df.year == 1)
    df["chg"] = df.hpp / prev - 1
    df["chg_next"] = nxt / df.hpp - 1
    df["R4"] = ev & (df.chg.abs() > 0.3)
    df["spike"] = df.R4 & (df.chg_next.abs() > 0.3) & (np.sign(df.chg) != np.sign(df.chg_next)) & \
        ((nxt / prev - 1).abs() <= 0.3)
    df["hpp_prev"], df["hpp_next"] = prev, nxt
    df["after_spike"] = df.groupby(["source", "country", "code"]).spike.shift(1).fillna(False).astype(bool) & df.R4
    df["flag"] = df[["R1", "R2", "R3", "R4"]].any(axis=1)
    return df


def second_source(F, W):
    """same-rule check in the other source, 13 countries, 2010-2014, one-to-one codes"""
    Fw = F.copy()
    Fw["key"] = Fw.code.map(fig2wiod)
    Ww = W[W.country.isin(V3) & W.year.between(2010, 2014)].copy()
    Ww["key"] = Ww.code.where(~Ww.code.isin(NOT_1TO1_W))
    a = Fw.merge(Ww[["country", "year", "key", "R1", "R2", "R3", "R4", "w_rel"]].dropna(subset=["key"]),
                 on=["country", "year", "key"], how="left", suffixes=("", "_2"))   # NaN keys never match (journal 15)
    Wf = W.copy()
    Wf["key"] = Wf.code.where(~Wf.code.isin(NOT_1TO1_W))
    Fk = F[F.year.between(2010, 2014)].copy()
    Fk["key"] = Fk.code.map(fig2wiod)
    b = Wf.merge(Fk[["country", "year", "key", "R1", "R2", "R3", "R4", "w_rel"]].dropna(subset=["key"]),
                 on=["country", "year", "key"], how="left", suffixes=("", "_2"))
    out = []
    for d in (a, b):
        res = []
        for r in d.itertuples():
            if not r.flag:
                res.append("")
                continue
            if pd.isna(r.key) or pd.isna(r.R1_2):
                res.append("нет второго источника")
                continue
            same = (r.R1 and r.R1_2 and np.sign(np.log(r.w_rel)) == np.sign(np.log(r.w_rel_2))) or \
                   (r.R2 and r.R2_2) or (r.R3 and r.R3_2) or (r.R4 and r.R4_2)
            res.append("то же правило" if same else "нет флага")
        d["second"] = res
        out.append(d.drop(columns=[c for c in d.columns if c.endswith("_2")] + ["key"]))
    return out[0], out[1]


def classify(r):
    if not r.flag:
        return ""
    if r.R3:
        return "ошибка: пропуск"
    if r.R4 and r.spike:
        return "ошибка: выброс"
    if r.R4 and r.after_spike:
        return "следствие выброса"
    if r.R4:
        return "разрыв ряда"
    if r.imputed_split or r.imputed_hpp:
        return "артефакт заполнения"
    if r.second == "то же правило":
        return "подтверждено вторым источником"
    if r.second == "нет флага":
        return "расхождение источников"
    return "не проверяемо"


def corrections(df):
    df = df.copy()
    df["hours_corr"] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        hx = (df.hours / df.x).where((df.hours > 0) & (df.x > 0))
    med = hx.groupby([df.source, df.country, df.code]).transform("median")
    m = (df.cls == "ошибка: пропуск") & (df.hours <= 0) & med.notna()
    df.loc[m, "hours_corr"] = df.x[m] * med[m]
    m = df.cls == "ошибка: выброс"
    df.loc[m, "hours_corr"] = df.persons[m] * np.sqrt(df.hpp_prev[m] * df.hpp_next[m])
    return df


def run_flags():
    F = flags(figaro_cells())
    W = flags(wiod_cells())
    F, W = second_source(F, W)
    D = pd.concat([F, W], ignore_index=True)
    D["second"] = D.second.fillna("")
    D["cls"] = [classify(r) for r in D.itertuples()]
    D = corrections(D)
    D.to_csv(OUT / "s4_flags_all.csv.gz", index=False)
    fl = D[D.flag]
    fl.to_csv(OUT / "s4_flags.csv", index=False)
    ev = D[D["eval"].astype(bool)]
    summ = ev.groupby(["source", "country"]).agg(cells=("flag", "size"), flagged=("flag", "sum"), R1=("R1", "sum"),
                                                 R2=("R2", "sum"), R3=("R3", "sum"), R4=("R4", "sum"),
                                                 corrected=("hours_corr", lambda v: int(v.notna().sum())))
    cls = fl.groupby(["source", "country", "cls"]).size().unstack("cls", fill_value=0)
    summ = summ.join(cls).fillna(0)
    summ.to_csv(OUT / "s4_flags_by_country.csv")
    print(summ.to_string(), "\n", fl.groupby(["source", "cls"]).size().to_string(), flush=True)
    return D


def load_flags():
    return pd.read_csv(OUT / "s4_flags_all.csv.gz", low_memory=False)


def flag_set(D, source):
    d = D[(D.source == source) & D.flag]
    return set(zip(d.country, d.year.astype(int), d.code))


def corr_map(D, source):
    d = D[(D.source == source) & D.hours_corr.notna()]
    return {(c, int(y), k): v for c, y, k, v in zip(d.country, d.year, d.code, d.hours_corr)}


# ================================================================ outcome 1 (levels, FIGARO)
def o1_frames():
    cols = ["basis", "industry", "z", "x", "capital", "closed", "imports", "labour", "country", "year"]
    out = []
    for tag in ("main", "robust"):
        d = pd.read_parquet(TAB / f"ratios_long_{tag}.parquet", columns=cols)
        d = d[(d.capital == True) & (d.closed.astype(str) == "False") & (d.imports == "price")]   # noqa: E712
        d = d[((d.basis == "labour") & (d.labour == "hours")) | d.basis.str.startswith("k:")]
        out.append(d)
    return pd.concat(out, ignore_index=True)


def o1_share(g, excl, zlab=None):
    """share of commodity bases with MAWD below hours, one country-year"""
    c, y = g.country.iloc[0], int(g.year.iloc[0])
    res = {}
    for b, gb in g.groupby("basis", sort=False):
        labels = gb.industry.tolist()
        m = mask_for(labels, set())
        if excl:
            m &= np.array([(c, y, lab) not in excl for lab in labels])
        z = gb.z.to_numpy()
        if b == "labour" and zlab is not None:
            z = zlab
        res[b] = ratio_metrics(z, gb.x.to_numpy(), m)["mawd"]
    lab = res.pop("labour")
    v = np.array(list(res.values()))
    return float(np.mean(v < lab))


# ================================================================ FIGARO pass: outcome 1 (b) and outcome 3 (a, b)
def run_figaro(D):
    from ..v3.placebo_disp import Draws, Setup, shares
    excl = flag_set(D, "FIGARO")
    cm = corr_map(D, "FIGARO")
    R = o1_frames()
    o1rows = []
    o3rows = []
    for c in V3:
        Dr = None
        for y in FIG_YEARS:
            g = R[(R.country == c) & (R.year == y)]
            in3 = y in O3_YEARS
            if g.empty and not in3:
                continue
            e, _ = build(c, y)
            e2 = None
            fix = {j: cm[(c, y, lab)] for j, lab in enumerate(e.labels) if (c, y, lab) in cm}
            if fix:
                e2 = copy.deepcopy(e)
                h = e2.hours.copy()
                for j, v in fix.items():
                    h[j] = v
                e2.hours = h
            if not g.empty:
                zlab = None
                if e2 is not None:
                    gl = g[g.basis == "labour"]
                    assert gl.industry.tolist() == list(e.labels)
                    zlab = e2.basis_ratios("L", True, False, "price", l=e2.l())
                o1rows.append(dict(country=c, year=y, orig=o1_share(g, None), clean=o1_share(g, excl),
                                   corr=o1_share(g, None, zlab), n_fixed=len(fix)))
            if in3:
                S = Setup(e)
                if Dr is None:
                    Dr = Draws(c, e.n, S.base_mask)
                    labels0 = list(e.labels)
                assert list(e.labels) == labels0
                clean = S.base_mask & np.array([(c, y, lab) not in excl for lab in e.labels])
                for r in shares(S, Dr, e.l(), {"orig": S.base_mask, "clean": clean}, kinds=("perm", "lnorm")):
                    o3rows.append(dict(country=c, year=y, **r))
                S2 = Setup(e2) if e2 is not None else S
                l2 = e2.l() if e2 is not None else e.l()
                for r in shares(S2, Dr, l2, {"corr": S2.base_mask}, kinds=("perm", "lnorm")):
                    o3rows.append(dict(country=c, year=y, **r))
            print("figaro pass", c, y, len(fix), flush=True)
    O1 = pd.DataFrame(o1rows)
    O3 = pd.DataFrame(o3rows)
    O1.to_csv(OUT / "s4_o1_country_years.csv", index=False)
    O3.to_csv(OUT / "s4_o3_shares.csv", index=False)
    return O1, O3


def o1_rows(O1):
    out = []
    for v in ("orig", "clean", "corr"):
        s = O1.groupby("country")[v].mean()
        out.append(mean_row(f"1 [{v}]: share of commodity bases better than labour (MAWD)", s.to_numpy(), 0.5, 0.10, "<"))
    return out


def top5(d):
    a = d.groupby(["country", "kind", "metric"]).share_better.mean().unstack(["kind", "metric"])
    return (a[[("perm", "mawd"), ("perm", "d"), ("lnorm", "mawd"), ("lnorm", "d")]] <= 0.05).all(axis=1)


def o3_rows(O3):
    out = []
    for v in ("orig", "clean", "corr"):
        t = top5(O3[O3["mask"] == v])
        out.append(share_row(f"3 [{v}]: countries with hours in the top 5% of placebos (a) and (b)", t, 0.05, 0.15, ">",
                             not_top=",".join(sorted(t[~t].index))))
    return out


# ================================================================ WIOD: outcome 15 (R1 of the revisit)
def run_wiod15(D):
    from ..revisit import wiod as rw
    from ..v3.placebo_disp import Draws, Setup, shares
    excl = flag_set(D, "WIOD")
    cm = corr_map(D, "WIOD")
    r1 = pd.read_csv(ROOT / "results" / "revisit" / "r1_main.csv")
    skipped = set(r1[r1.kind == "skipped"].country)
    countries = [c for c in rw.eligible() if c not in skipped]
    rows = []
    for c in countries:
        Dr = None
        for y in W_YEARS:
            e = rw.economy(c, y)
            S = Setup(e)
            if Dr is None:
                Dr = Draws(c, e.n, S.base_mask)
            clean = S.base_mask & np.array([(c, y, lab) not in excl for lab in e.labels])
            for r in shares(S, Dr, e.l(), {"orig": S.base_mask, "clean": clean}, kinds=("perm", "lnorm")):
                rows.append(dict(country=c, year=y, **r))
            fix = {j: cm[(c, y, lab)] for j, lab in enumerate(e.labels) if (c, y, lab) in cm}
            if fix:
                e2 = copy.deepcopy(e)
                h = e2.hours.copy()
                for j, v in fix.items():
                    h[j] = v
                e2.hours = h
                S2 = Setup(e2)
                rr = shares(S2, Dr, e2.l(), {"corr": S2.base_mask}, kinds=("perm", "lnorm"))
            else:
                rr = [dict(r, mask="corr") for r in rows[-8:] if r["mask"] == "orig"]
                rr = [{k: v for k, v in r.items() if k not in ("country", "year")} for r in rr]
            for r in rr:
                rows.append(dict(country=c, year=y, **r))
        print("wiod15", c, flush=True)
    O = pd.DataFrame(rows)
    O.to_csv(OUT / "s4_o15_shares.csv", index=False)
    return O


def o15_rows(O):
    out = []
    for v in ("orig", "clean", "corr"):
        t = top5(O[O["mask"] == v])
        out.append(share_row(f"15 [{v}]: WIOD countries with hours in the top 5% of placebos", t, 0.05, 0.15, ">",
                             not_top=",".join(sorted(t[~t].index))))
    return out


# ================================================================ WIOD world system: outcomes 28 and 44
def corrected_sea_wide(D):
    from ..v9 import stage34 as s34
    w = s34.sea_wide().copy()
    cm = corr_map(D, "WIOD")
    key = list(zip(w.country, w.code, w.year.astype(int)))
    idx = [i for i, k in enumerate(key) if (k[0], k[2], k[1]) in cm]
    for i in idx:
        c, k, y = key[i]
        hc = cm[(c, y, k)]
        emp, empe = w.at[i, "EMP"], w.at[i, "EMPE"]
        if not (emp > 0):
            w.at[i, "EMP"] = w.at[i, "EMPE"] = 1.0
            w.at[i, "H_EMPE"] = hc
        elif not (empe > 0):
            w.at[i, "EMPE"] = emp
            w.at[i, "H_EMPE"] = hc
        else:
            w.at[i, "H_EMPE"] = hc * empe / emp
    return w, len(idx)


def national(y, cont, cells, W, k, excl, strict):
    from ..v9.stage34 import GOVL, mawd, qmap
    rows = []
    keep = ~cells.code.isin(GOVL) & (cells.x > 0) & (cells.country != "ROW")
    if excl is not None:
        keep &= ~pd.Series([(c, y, cd) in excl for c, cd in zip(cells.country, cells.code)], index=cells.index)
    for c, g in cells[keep].groupby("country"):
        idx = g.index.to_numpy()
        lam = cont[k][idx]
        wld = g.code.map(W[k]).to_numpy()
        if (lam <= 0).any() or len(idx) < 20 or (strict and not np.isfinite(wld).all()):
            continue
        x = g.x.to_numpy()
        mn, mw = mawd(lam, x)[0], mawd(qmap(wld, lam), x)[0]
        rows.append(dict(year=y, cand=k, country=c, mawd_national=mn, mawd_world_eq=mw, world_better=mw < mn))
    return rows


def run_wiod28(D):
    from ..v9 import stage34 as s34
    excl = flag_set(D, "WIOD")
    orig_sw = s34.sea_wide
    wc, nfix = corrected_sea_wide(D)
    print("WIOD corrected cells:", nfix, flush=True)
    rows = []
    for variant in ("orig", "corr"):
        if variant == "corr":
            s34.sea_wide = lambda: wc
        for y in W_YEARS:
            A, cells = s34.year_system(y)
            cont = s34.contents(A, cells)
            W, _ = s34.world_industries(cont, cells)
            masks = {"orig": None, "clean": excl} if variant == "orig" else {"corr": None}
            for mname, ex in masks.items():
                for r in national(y, cont, cells, W, "hours", ex, strict=False):
                    rows.append(dict(r, mask=mname, rule="v9"))
                for k in ("hours", "lab", "cap"):
                    for r in national(y, cont, cells, W, k, ex, strict=True):
                        rows.append(dict(r, mask=mname, rule="v9b"))
            print("wiod28", variant, y, flush=True)
            pd.DataFrame(rows).to_csv(OUT / "s4_o28_44_national.csv", index=False)
    s34.sea_wide = orig_sw
    return pd.DataFrame(rows)


def o28_44_rows(N):
    from ..v9b.stage23 import outcome44
    out = []
    for v in ("orig", "clean", "corr"):
        n9 = N[(N["mask"] == v) & (N.rule == "v9")]
        byc = n9.groupby("country").world_better.mean() > 0.5
        out.append(share_row(f"28 [{v}]: countries where world-average labour beats national (majority of years)", byc,
                             0.5, 0.15, ">"))
        o = outcome44(N[(N["mask"] == v) & (N.rule == "v9b")])
        r = o[o.outcome.str.contains("lab")].iloc[0].to_dict()
        r["outcome"] = f"44 [{v}]: share(hours) - share(lab), world beats national"
        r.update(theta0=0, delta=0.1, direction=">")
        out.append(r)
    return out


# ================================================================ summary: outcome 55
def summary():
    rows = []
    O1 = pd.read_csv(OUT / "s4_o1_country_years.csv")
    rows += o1_rows(O1)
    rows += o3_rows(pd.read_csv(OUT / "s4_o3_shares.csv"))
    rows += o15_rows(pd.read_csv(OUT / "s4_o15_shares.csv"))
    rows += o28_44_rows(pd.read_csv(OUT / "s4_o28_44_national.csv"))
    R = pd.DataFrame(rows)
    R["n_outcome"] = R.outcome.str.split(" ").str[0]
    R["variant"] = R.outcome.str.extract(r"\[(\w+)\]")[0]
    lab = R.pivot(index="n_outcome", columns="variant", values="label")
    est = R.pivot(index="n_outcome", columns="variant", values="est")
    T = pd.concat({"label": lab, "est": est}, axis=1)
    T["changed_a"] = lab["clean"] != lab["orig"]
    T["changed_b"] = lab["corr"] != lab["orig"]
    T["changed"] = T.changed_a | T.changed_b
    R.to_csv(OUT / "s4_outcomes_recomputed.csv", index=False)
    T.to_csv(OUT / "s4_labels.csv")
    o55 = dict(outcome="55: number of labels (of outcomes 1, 3, 15, 28, 44) changed by (a) or (b)",
               est=int(T.changed.sum()), n=len(T), label="описательно",
               changed=",".join(T.index[T.changed]))
    pd.DataFrame([o55]).to_csv(OUT / "s4_outcomes.csv", index=False)
    print(R.round(4).to_string(), "\n", T.to_string(), "\n", o55, flush=True)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    parts = sys.argv[1:] or ["flags", "figaro", "wiod15", "wiod28", "summary"]
    if "flags" in parts:
        D = run_flags()
    else:
        D = load_flags()
    if "figaro" in parts:
        run_figaro(D)
    if "wiod15" in parts:
        run_wiod15(D)
    if "wiod28" in parts:
        run_wiod28(D)
    if "summary" in parts:
        summary()
