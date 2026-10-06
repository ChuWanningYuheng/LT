"""Iteration 8, stage 2: degree of transformation b and the distribution of profit by ratios
(pre_registration_v8.md, stage 2).

b: ln(PI/W) = b ln(K/W) + country x year effects; industry clusters; PI > 0.  b = 0 equal exploitation (prices
proportional to values), b = 1 equal profit rates.  Panels: KLEMS (B3 of iteration 4), WIOD (R2 of the revisit),
Eurostat (nama_10_a64 + nama_10_nfa_st).  Ratios: ln(PI/GO) on ln(K/GO), ln((II+CFC)/GO), ln(W/GO).
Turnover (US, NIPA inventories) and concentration (US, Economic Census CR4).
Outputs: results/v8/s2_*.csv
"""
from __future__ import annotations

import json
import zipfile

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from ..codes import BEA_TO_NACE
from ..revisit.wiod import RENT_W, r2_panel, sea
from ..v4.b_data import divs
from ..v6.national import es
from ..v6.series import ROOT, nipa
from ..v7.part1 import FINE_ORDER, klems_panel, make_a, slope
from ..v7.part2 import naics_to_bea

OUT = ROOT / "results" / "v8"
OUT.mkdir(parents=True, exist_ok=True)
R8 = ROOT / "data" / "raw" / "v8"
PERIODS = {"2000-2007": (2000, 2007), "2008-2014": (2008, 2014), "2015-2021": (2015, 2021)}
EU_LEAF = ["A01", "A02", "A03", "B", "C10-C12", "C13-C15", "C16", "C17", "C18", "C19", "C20", "C21", "C22", "C23",
           "C24", "C25", "C26", "C27", "C28", "C29", "C30", "C31_C32", "C33", "D", "E36", "E37-E39", "F", "G45", "G46",
           "G47", "H49", "H50", "H51", "H52", "H53", "I", "J58", "J59_J60", "J61", "J62_J63", "K64", "K65", "K66",
           "L68A", "L68B", "M69_M70", "M71", "M72", "M73", "M74_M75", "N77", "N78", "N79", "N80-N82", "O", "P", "Q86",
           "Q87_Q88", "R90-R92", "R93", "S94", "S95", "S96", "T", "U"]
EU_RENT = {"B", "D", "E36", "E37-E39", "K64", "K65", "K66", "L68A", "L68B"}
EU_GOV = {"O", "P", "Q86", "Q87_Q88", "T", "U"}


def bfit(d, y, x, extra=(), fe="cy"):
    b, p, fit = slope(d.assign(_y=y, _x=x, **{f"_c{i}": c for i, c in enumerate(extra)}), "_y", "_x",
                      ctrl=[f"_c{i}" for i in range(len(extra))], fe=fe)
    se = float(fit.bse["_x"])
    return dict(b=b, se=se, ci90_lo=b - 1.645 * se, ci90_hi=b + 1.645 * se, p=p,
                tost_full_eq=float(max(stats.norm.sf((b - 0.9) / se), stats.norm.cdf((b - 1.1) / se))),
                tost_values=float(max(stats.norm.sf((b + 0.1) / se), stats.norm.cdf((b - 0.1) / se))),
                mde80=2.8 * se, n=len(d), clusters=int(d.ind.nunique()))


def label_b(r):
    """Unified rule for outcome 19: theta0 = 1, Delta = 0.1, 'for' = b < 1."""
    if r["ci90_hi"] < 1 and r["b"] < 0.9:
        return "подтверждено (неполное выравнивание)"
    if r["ci90_lo"] >= 0.9 and r["ci90_hi"] <= 1.1 or r["ci90_lo"] > 1:
        return "опровергнуто (полное выравнивание)"
    return "неинформативно"


# ---------------------------------------------------------------- panels
def klems():
    p = klems_panel()
    p = p[(p.PI > 0) & (p.W > 0) & (p.K > 0)].copy()
    return p


def wiod():
    w = r2_panel()
    w = w[~w.rent & (w.r.abs() <= 2)].copy()
    s = sea()
    ii = s[s.variable == "II"].rename(columns={"v": "II"})[["country", "code", "year", "II"]]
    w = w.merge(ii, on=["country", "code", "year"], how="left")
    w["W"], w["ind"], w["geo"] = w.LAB, w.code, w.country
    w["CFC"] = w.delta * w.K
    return w[(w.PI > 0) & (w.W > 0)].copy()


def eurostat(gross=False):
    it = {k: es(R8 / f"a64_{k}.json") for k in ("B1G", "D1", "P51C", "D29X39", "P1", "P2")}
    frames = []
    for k, d in it.items():
        frames.append(d[["geo", "nace_r2", "time", "value"]].rename(columns={"value": k}).set_index(["geo", "nace_r2", "time"]))
    a = pd.concat(frames, axis=1).reset_index()
    st = es(R8 / "eurostat_nama_10_nfa_st.json")
    st = st.pivot_table(index=["geo", "nace_r2", "time"], columns="asset10", values="value").reset_index()
    a = a.merge(st, on=["geo", "nace_r2", "time"], how="inner")
    a = a[a.nace_r2.isin(EU_LEAF) & ~a.nace_r2.isin(EU_RENT | EU_GOV)]
    a = a.rename(columns={"nace_r2": "ind", "time": "year", "D1": "W", "P1": "GO", "P2": "II", "P51C": "CFC"})
    a["year"] = a.year.astype(int)
    # self-employed correction from KLEMS (W/COMP of the matching KLEMS industry, same country-year), else none
    kp = pd.read_csv(ROOT / "results" / "v4" / "b_panel.csv.gz")[["geo", "ind", "year", "W", "COMP"]]
    kp = kp[kp.COMP > 0].assign(f=lambda x: x.W / x.COMP)
    kmap = {i: next((k for k in FINE_ORDER if divs(i.replace("_", "-") if "-" not in i else i) <= divs(k)), None)
            for i in EU_LEAF}
    a["kind"] = a.ind.map(kmap)
    a = a.merge(kp.rename(columns={"ind": "kind"})[["geo", "kind", "year", "f"]], on=["geo", "kind", "year"], how="left")
    a["f"] = a.f.fillna(1.0).clip(1.0, 3.0)
    a["COMP"] = a.W
    a["W"] = a.COMP * a.f
    a["PI"] = a.B1G - a.W - a.D29X39 - a.CFC
    a["K"] = a.N11G if gross else a.N11N
    a["cy"] = a.geo + a.year.astype(str)
    a = a[(a.K > 0) & (a.W > 0) & (a.GO > 0)]
    a = a[(a.PI / a.K).abs() <= 2]
    return a[a.PI > 0].copy()


# ---------------------------------------------------------------- 2.1
def b_table():
    rows = []
    K = klems()
    W = wiod()
    E = eurostat()
    EG = eurostat(gross=True)
    for name, d in (("KLEMS", K), ("WIOD", W), ("Eurostat, net stock", E), ("Eurostat, gross stock", EG)):
        rows.append(dict(panel=name, variant="main", **bfit(d, np.log(d.PI / d.W), np.log(d.K / d.W))))
        for per, (a, b) in PERIODS.items():
            s = d[d.year.between(a, b)]
            if len(s) > 100:
                rows.append(dict(panel=name, variant=f"period {per}", **bfit(s, np.log(s.PI / s.W), np.log(s.K / s.W))))
        for g, s in d.groupby("geo"):
            if s.groupby("cy").size().median() >= 10 and s.ind.nunique() >= 10:
                try:
                    rows.append(dict(panel=name, variant=f"country {g}", **bfit(s, np.log(s.PI / s.W), np.log(s.K / s.W))))
                except Exception:  # noqa: BLE001
                    pass
    a = K[K.U_lag.notna()].copy()
    r1, x1 = make_a()(a, 1.0)
    PI1 = r1 * (a.K + a.K_nonNA)
    ok = PI1 > 0
    rows.append(dict(panel="KLEMS", variant="with non-NA intangibles (m = 1)",
                     **bfit(a[ok], np.log(PI1[ok] / a.W[ok]), x1[ok])))
    rows.append(dict(panel="KLEMS", variant="same sample, without intangibles",
                     **bfit(a, np.log(a.PI / a.W), np.log(a.K / a.W))))
    t = pd.DataFrame(rows)
    t["label_unified"] = t.apply(label_b, axis=1)
    return t, K, W, E


# ---------------------------------------------------------------- 2.2 ratios
def ratios(d, name, n_perm=1000, seed=8):
    d = d[(d.II > 0) & (d.CFC > 0)].copy()
    F = d.II + d.CFC
    lp, lk, lf, lw = np.log(d.PI / d.GO), np.log(d.K / d.GO), np.log(F / d.GO), np.log(d.W / d.GO)
    e = d.assign(_y=lp, _k=lk, _f=lf, _w=lw)
    cols = ["_y", "_k", "_f", "_w"]
    X = e[cols] - e.groupby("cy")[cols].transform("mean")
    fit = sm.OLS(X._y, X[["_k", "_f", "_w"]]).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(e.ind)[0]})
    rows = [dict(panel=name, item=f"coef {k}", est=fit.params[c], se=fit.bse[c], ci90_lo=fit.params[c] - 1.645 * fit.bse[c],
                 ci90_hi=fit.params[c] + 1.645 * fit.bse[c], n=len(e)) for k, c in (("K/GO", "_k"), ("(II+CFC)/GO", "_f"), ("W/GO", "_w"))]
    rng = np.random.default_rng(seed)
    for k, X_ in (("K", d.K), ("II+CFC", F), ("W", d.W)):
        dev, share = [], []
        for cy, g in d.assign(X=X_).groupby("cy"):
            if len(g) < 8:
                continue
            rho = g.PI.sum() / g.X.sum()
            w = g.GO / g.GO.sum()
            act = float((w * np.abs(np.log(g.PI / (rho * g.X)))).sum())
            q = (g.X / g.GO).to_numpy()
            go, pi, ww = g.GO.to_numpy(), g.PI.to_numpy(), w.to_numpy()
            Q = np.array([rng.permutation(q) for _ in range(n_perm)]) * go[None, :]
            rp = pi.sum() / Q.sum(1)
            devp = (ww[None, :] * np.abs(np.log(pi[None, :] / (rp[:, None] * Q)))).sum(1)
            dev.append(act)
            share.append(float((devp <= act).mean()))
        rows.append(dict(panel=name, item=f"PI proportional to {k}", est=float(np.median(dev)),
                         share_perm_better=float(np.mean(share)), n=len(dev)))
    return rows


# ---------------------------------------------------------------- 2.3 turnover (US)
DUR = {"C16", "C22-C23", "C24-C25", "C26", "C27", "C28", "C29-C30", "C31-C33"}


def turnover():
    p = klems()
    us = p[p.geo == "US"].copy()
    inv = nipa(["B372RC", "B377RC", "B378RC", "A379RC", "A382RC", "B864RC", "A385RC"]) / 1  # millions? (NIPA units -6)
    inv = inv * 1.0
    us["F"] = us.II + us.W
    grp = np.where(us.ind == "A", "farm", np.where(us.ind.str.startswith("C") & us.ind.isin(DUR), "mdur",
                   np.where(us.ind.str.startswith("C"), "mnondur", np.where(us.ind == "G46", "whole",
                   np.where(us.ind == "G47", "retail", np.where(us.ind == "G45", "mvd", "other"))))))
    us["grp"] = grp
    m = {"farm": "B372RC", "mdur": "B377RC", "mnondur": "B378RC", "whole": "A379RC", "mvd": "B864RC", "other": "A385RC"}
    inv["retail_ex"] = inv.A382RC - inv.B864RC
    m["retail"] = "retail_ex"
    us["INV"] = np.nan
    for (g, y), s in us.groupby(["grp", "year"]):
        tot = inv.loc[y, m[g]] if y in inv.index else np.nan
        us.loc[s.index, "INV"] = tot * s.F / s.F.sum()
    us["tau"] = us.INV / us.F
    us["Kadv"] = us.K + us.INV
    rows = []
    for name, K_ in (("K", us.K), ("K + inventories", us.Kadv)):
        rows.append(dict(item=f"b, US KLEMS, {name}", **bfit(us, np.log(us.PI / us.W), np.log(K_ / us.W))))
    share = (us.INV / us.K).groupby(us.ind).median().sort_values(ascending=False)
    return pd.DataFrame(rows), share.rename("inventories / K (median)").reset_index(), us


# ---------------------------------------------------------------- 2.5 concentration
def cr4_klems():
    z = zipfile.ZipFile(R8 / "census_EC1700SIZECONCEN.zip")
    d = pd.read_csv(z.open("EC1700SIZECONCEN.dat"), sep="|", low_memory=False, dtype=str)
    d = d[(d["#GEOTYPE"] == "1") & (d.TYPOP == "0")]
    d["RCPTOT"] = pd.to_numeric(d.RCPTOT, errors="coerce")
    tot = d[d.CONCENFI == "1"].set_index("NAICS2017").RCPTOT
    c4 = d[d.CONCENFI == "604"].set_index("NAICS2017").RCPTOT
    cr = pd.DataFrame({"tot": tot, "c4": c4}).dropna()
    cr = cr[cr.index.str.len() == 6]
    cr["cr4"] = cr.c4 / cr.tot
    cr["bea"] = [naics_to_bea(n[:4]) for n in cr.index]
    out = []
    for k in FINE_ORDER:
        dk = divs(k)
        sel = cr[[b in BEA_TO_NACE and set(BEA_TO_NACE[b]) <= set(dk) if b else False for b in cr.bea]]
        if len(sel):
            out.append(dict(ind=k, cr4=float(np.average(sel.cr4, weights=sel.tot)), n_naics=len(sel)))
    return pd.DataFrame(out)


def concentration():
    from ..v6.stage5 import klems_panel_1995, mueller
    p = klems_panel_1995()
    p = p[(p.geo == "US") & p.cfc_sh.notna() & (p.K > 0) & ~p.gov.astype(bool) & ~p.rent.astype(bool)].copy()
    p = p.sort_values(["ind", "year"])
    p["r"] = p.PI / p.groupby("ind").K.shift()
    p["dev_r"] = p.r - p.groupby("year").r.transform("mean")
    m = mueller(p, "dev_r")
    cr = cr4_klems()
    j = m.merge(cr, on="ind")
    rows = []
    for y in ("lam", "lr"):
        Xc = sm.add_constant(j.cr4)
        f = sm.OLS(j[y], Xc).fit(cov_type="HC1")
        rows.append(dict(item=f"{y} on CR4", est=f.params.cr4, se=f.bse.cr4, ci90_lo=f.params.cr4 - 1.645 * f.bse.cr4,
                         ci90_hi=f.params.cr4 + 1.645 * f.bse.cr4, p=f.pvalues.cr4, mde80=2.8 * f.bse.cr4, n=len(j)))
    f = sm.OLS(j.lr.abs(), sm.add_constant(j.cr4)).fit(cov_type="HC1")
    rows.append(dict(item="|lr| on CR4", est=f.params.cr4, se=f.bse.cr4, ci90_lo=f.params.cr4 - 1.645 * f.bse.cr4,
                     ci90_hi=f.params.cr4 + 1.645 * f.bse.cr4, p=f.pvalues.cr4, mde80=2.8 * f.bse.cr4, n=len(j)))
    return pd.DataFrame(rows), j


if __name__ == "__main__":
    t, K, W, E = b_table()
    t.to_csv(OUT / "s2_b.csv", index=False)
    print(t[["panel", "variant", "b", "ci90_lo", "ci90_hi", "p", "n", "label_unified"]].round(3).to_string(), flush=True)
    rr = pd.DataFrame(ratios(K, "KLEMS") + ratios(W, "WIOD"))
    rr.to_csv(OUT / "s2_ratios.csv", index=False)
    print(rr.round(3).to_string(), flush=True)
    tt, sh, _ = turnover()
    tt.to_csv(OUT / "s2_turnover.csv", index=False)
    sh.to_csv(OUT / "s2_turnover_share.csv", index=False)
    print(tt.round(3).to_string(), sh.head(10).round(3).to_string(), flush=True)
    cc, j = concentration()
    cc.to_csv(OUT / "s2_concentration.csv", index=False)
    j.to_csv(OUT / "s2_concentration_detail.csv", index=False)
    print(cc.round(3).to_string(), flush=True)
