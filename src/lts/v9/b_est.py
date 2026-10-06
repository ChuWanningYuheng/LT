"""Iteration 9, stage 1.2: which b is right (pre_registration_v9.md, 1.2).

Model PI/W = A_cy (K/W)^b.  Estimators: (a) log OLS, PI > 0; (b) asinh; (c) NLS in levels; (d) PPML (gross; net with
losses set to 0); (e) grouped by K/W quintiles within country-year; (f1) IV with the same industry in other countries;
(f2) hours instead of W.  Panels: KLEMS (B3 without overlaps), WIOD (R2), Eurostat (net stock).
Outputs: results/v9/s1_b_estimators.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize

from ..v8.rule import decide
from ..v8.stage2 import EU_GOV, EU_LEAF, EU_RENT, FINE_ORDER, R8, bfit, divs, es
from .stage1 import OUT, use_noovl

B_BOOT = 999


# ---------------------------------------------------------------- panels with losses kept
def klems_all():
    use_noovl()
    from ..v7.part1 import klems_panel
    p = klems_panel()
    p = p[(p.W > 0) & (p.K > 0)].copy()
    p["G"] = p.PI + p.CFC
    return p


def wiod_all():
    from ..revisit.wiod import r2_panel
    w = r2_panel()
    w = w[~w.rent & (w.r.abs() <= 2)].copy()
    w["W"], w["ind"], w["geo"] = w.LAB, w.code, w.country
    w["CFC"] = w.delta * w.K
    w["G"] = w.PI + w.CFC
    w["H"] = w.H_EMPE
    return w[(w.W > 0) & (w.K > 0)].copy()


def eurostat_all():
    from ..v6.series import ROOT
    it = {k: es(R8 / f"a64_{k}.json") for k in ("B1G", "D1", "P51C", "D29X39", "P1", "P2")}
    a = pd.concat([d[["geo", "nace_r2", "time", "value"]].rename(columns={"value": k}).set_index(["geo", "nace_r2", "time"])
                   for k, d in it.items()], axis=1).reset_index()
    st = es(R8 / "eurostat_nama_10_nfa_st.json").pivot_table(index=["geo", "nace_r2", "time"], columns="asset10",
                                                             values="value").reset_index()
    a = a.merge(st, on=["geo", "nace_r2", "time"], how="inner")
    a = a[a.nace_r2.isin(EU_LEAF) & ~a.nace_r2.isin(EU_RENT | EU_GOV)]
    a = a.rename(columns={"nace_r2": "ind", "time": "year", "D1": "W", "P1": "GO", "P2": "II", "P51C": "CFC"})
    a["year"] = a.year.astype(int)
    kp = pd.read_csv(OUT / "b_panel.csv.gz")[["geo", "ind", "year", "W", "COMP"]]
    kp = kp[kp.COMP > 0].assign(f=lambda x: x.W / x.COMP)
    kmap = {i: next((k for k in FINE_ORDER if divs(i.replace("_", "-") if "-" not in i else i) <= divs(k)), None)
            for i in EU_LEAF}
    a["kind"] = a.ind.map(kmap)
    a = a.merge(kp.rename(columns={"ind": "kind"})[["geo", "kind", "year", "f"]], on=["geo", "kind", "year"], how="left")
    a["f"] = a.f.fillna(1.0).clip(1.0, 3.0)
    a["W"] = a.W * a.f
    a["PI"] = a.B1G - a.W - a.D29X39 - a.CFC
    a["G"] = a.PI + a.CFC
    a["K"] = a.N11N
    a["cy"] = a.geo + a.year.astype(str)
    a = a[(a.K > 0) & (a.W > 0) & (a.GO > 0)]
    return a[(a.PI / a.K).abs() <= 2].copy()


# ---------------------------------------------------------------- estimators
def _a_cy(y, xb, cy, kind):
    """Concentrated-out country-year intercepts (as multipliers) for NLS ('nls') or Poisson ('ppml')."""
    if kind == "nls":
        num = pd.Series(y * xb).groupby(cy).transform("sum").to_numpy()
        den = pd.Series(xb * xb).groupby(cy).transform("sum").to_numpy()
    else:
        num = pd.Series(y).groupby(cy).transform("sum").to_numpy()
        den = pd.Series(xb).groupby(cy).transform("sum").to_numpy()
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0)


def nls_b(y, x, cy):
    lx = np.log(x)

    def ssr(b):
        xb = np.exp(b * lx)
        return float(((y - _a_cy(y, xb, cy, "nls") * xb) ** 2).sum())
    return optimize.minimize_scalar(ssr, bounds=(-1, 2), method="bounded", options={"xatol": 1e-5}).x


def ppml_b(y, x, cy):
    lx = np.log(x)

    def foc(b):
        xb = np.exp(b * lx)
        return float(((y - _a_cy(y, xb, cy, "ppml") * xb) * lx).sum())
    try:
        return optimize.brentq(foc, -1, 2, xtol=1e-6)
    except ValueError:
        return np.nan


def grouped(d, yname):
    """Quintiles of K/W within country-year (terciles with < 10 industries); returns group aggregates."""
    d = d.copy()
    n = d.groupby("cy").ind.transform("size")
    q = np.where(n >= 10, 5, 3)
    d["rk"] = d.groupby("cy").kw.rank(pct=True, method="first")
    d["g"] = np.ceil(d.rk * q).astype(int)
    a = d.groupby(["cy", "g"]).agg(P=(yname, "sum"), W=("W", "sum"), K=("K", "sum")).reset_index()
    return a


def grouped_b(d, yname="PI"):
    a = grouped(d, yname)
    lost = (a.P <= 0).mean()
    a = a[a.P > 0]
    y = np.log(a.P / a.W)
    x = np.log(a.K / a.W)
    X = pd.DataFrame({"y": y, "x": x})
    X = X - X.groupby(a.cy).transform("mean")
    b = float((X.x * X.y).sum() / (X.x ** 2).sum())
    return b, lost, len(a)


def cluster_boot(d, fn, B=B_BOOT, seed=12):
    """Industry cluster bootstrap; fn(sample) -> b."""
    rng = np.random.default_rng(seed)
    inds = d.ind.unique()
    groups = {i: g for i, g in d.groupby("ind")}
    out = []
    for _ in range(B):
        pick = rng.choice(inds, len(inds), replace=True)
        s = pd.concat([groups[i].assign(cy=groups[i].cy) for i in pick], ignore_index=True)
        try:
            out.append(fn(s))
        except Exception:                                      # noqa: BLE001
            out.append(np.nan)
    out = np.array(out, float)
    return np.nanquantile(out, [0.05, 0.95]), np.nanstd(out)


def iv_b(d):
    """(f1) 2SLS of ln(PI/W) on ln(K/W), instrument = mean ln(K/W) of the same industry, other countries, same year."""
    d = d[d.PI > 0].copy()
    d["y"] = np.log(d.PI / d.W)
    d["x"] = np.log(d.K / d.W)
    s = d.groupby(["ind", "year"]).x.transform("sum")
    c = d.groupby(["ind", "year"]).x.transform("count")
    d["z"] = (s - d.x) / (c - 1)
    d = d[c > 1].dropna(subset=["z"])
    X = d[["y", "x", "z"]] - d.groupby("cy")[["y", "x", "z"]].transform("mean")
    zx = (X.z * X.x).sum()
    b = float((X.z * X.y).sum() / zx)
    e = X.y - b * X.x
    grp = d.ind.to_numpy()
    meat = sum(((X.z * e)[grp == g].sum()) ** 2 for g in np.unique(grp))
    G = len(np.unique(grp))
    se = float(np.sqrt(meat * G / (G - 1)) / abs(zx))
    fs_t = float(np.corrcoef(X.z, X.x)[0, 1] * np.sqrt(len(X)))
    return b, se, len(d), fs_t


def run_panel(name, d, has_h=True):
    rows = []
    d = d.copy()
    d["kw"] = d.K / d.W
    d["y"] = d.PI / d.W
    n_all = len(d)

    def add(est, b, lo, hi, n, lost, se=np.nan, note=""):
        rows.append(dict(panel=name, estimator=est, b=b, ci90_lo=lo, ci90_hi=hi, se=se, n=n, share_lost=lost,
                         label=decide(b, lo, hi, 1, 0.1, "<"),
                         values_regime=decide(b, lo, hi, 0, 0.1, ">") if np.isfinite(lo) else "", note=note))
    pos = d[d.PI > 0]
    f = bfit(pos, np.log(pos.PI / pos.W), np.log(pos.kw))
    add("(a) log, PI > 0", f["b"], f["ci90_lo"], f["ci90_hi"], f["n"], 1 - f["n"] / n_all, f["se"])
    f = bfit(d, np.arcsinh(d.y), np.log(d.kw))
    add("(b) asinh(PI/W)", f["b"], f["ci90_lo"], f["ci90_hi"], f["n"], 0.0, f["se"])
    print(name, "a, b done", flush=True)
    cy = d.cy.to_numpy()
    b_c = nls_b(d.y.to_numpy(), d.kw.to_numpy(), cy)
    (lo, hi), sd = cluster_boot(d, lambda s: nls_b(s.y.to_numpy(), s.kw.to_numpy(), s.cy.to_numpy()))
    add("(c) NLS in levels", b_c, lo, hi, n_all, 0.0, sd)
    print(name, "c done", flush=True)
    g = d[d.G > 0]
    b_dg = ppml_b((g.G / g.W).to_numpy(), g.kw.to_numpy(), g.cy.to_numpy())
    (lo, hi), sd = cluster_boot(g, lambda s: ppml_b((s.G / s.W).to_numpy(), s.kw.to_numpy(), s.cy.to_numpy()))
    add("(d) PPML, gross profit", b_dg, lo, hi, len(g), 1 - len(g) / n_all, sd)
    yn = d.y.clip(lower=0).to_numpy()
    b_dn = ppml_b(yn, d.kw.to_numpy(), cy)
    (lo, hi), sd = cluster_boot(d, lambda s: ppml_b(s.y.clip(lower=0).to_numpy(), s.kw.to_numpy(), s.cy.to_numpy()))
    add("(d') PPML, net profit, losses = 0", b_dn, lo, hi, n_all, 0.0, sd)
    print(name, "d done", flush=True)
    b_e, lost, ng = grouped_b(d)
    (lo, hi), sd = cluster_boot(d, lambda s: grouped_b(s)[0])
    add("(e) grouped (K/W quintiles)", b_e, lo, hi, ng, lost, sd, note="share_lost = share of groups with sum PI <= 0")
    print(name, "e done", flush=True)
    b_iv, se_iv, n_iv, fst = iv_b(d)
    add("(f1) IV: same industry, other countries", b_iv, b_iv - 1.645 * se_iv, b_iv + 1.645 * se_iv, n_iv,
        1 - n_iv / n_all, se_iv, note=f"first-stage t approx {fst:.1f}")
    if has_h:
        hh = d[(d.PI > 0) & (d.H > 0)]
        f = bfit(hh, np.log(hh.PI / hh.H), np.log(hh.K / hh.H))
        add("(f2) hours instead of W, PI > 0", f["b"], f["ci90_lo"], f["ci90_hi"], f["n"], 1 - f["n"] / n_all, f["se"])
    return rows


def choose(R):
    out = []
    for p, g in R.groupby("panel"):
        g = g.set_index("estimator")
        bc, be = g.loc["(c) NLS in levels", "b"], g.loc["(e) grouped (K/W quintiles)", "b"]
        if abs(bc - be) > 0.15:
            concl = f"interval between (c) and (e): {min(bc, be):.2f} to {max(bc, be):.2f}"
        else:
            concl = f"main (e) = {be:.2f}"
        out.append(dict(panel=p, b_c=bc, b_e=be, gap=abs(bc - be), conclusion=concl,
                        label_e=g.loc["(e) grouped (K/W quintiles)", "label"]))
    return pd.DataFrame(out)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    rows = []
    for name, loader, has_h in (("KLEMS", klems_all, True), ("WIOD", wiod_all, True), ("Eurostat", eurostat_all, False)):
        rows += run_panel(name, loader(), has_h)
        R = pd.DataFrame(rows)
        R.to_csv(OUT / "s1_b_estimators.csv", index=False)
    print(R.round(3).to_string())
    C = choose(R)
    C.to_csv(OUT / "s1_b_choice.csv", index=False)
    print(C.round(3).to_string())
