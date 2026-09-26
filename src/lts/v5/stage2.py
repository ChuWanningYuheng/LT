"""Iteration 5, stage 2: automation and aggregate profitability (pre_registration_v5.md, section 2).

Automation measures per country-year:
  A1     capital share of the bottom decile of industries by hours per unit of real VA (2015 prices), within
         the country-year; rent and regulated industries (B, D, E, K, L, O-U) are neither in the decile nor
         in the denominator.  Decile = round(n/10) industries (at least 1).
  A1fix  same with the decile list fixed in 2000 (first available year if later)
  A2     ICT + software share of capital (K_IT + K_CT + K_Soft_DB) / K_GFCF of the aggregate
Tests: T2.1 (levels with country and year FE; five-year differences), T2.2, T2.3; trap variants as in stage 1.
Industry level (descriptive): near-labourless industries and the transfer map T_j = PI_j - e_bar * W_j.

Outputs (results/v5/): s2_measures.csv, s2_tests.csv, s2_lowlab.csv, s2_transfer.csv, s2_transfer_reg.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .agg import EXCL_MAIN
from .stage1 import OUT, PERIODS, TRAPS, holm, load, ols_fwl

NONAUTO = {"B", "D", "E", "D-E", "K", "L", "L68A", "O", "P", "Q", "Q86", "Q87-Q88", "O-Q", "R", "S", "R-S", "T",
           "U"}
RENT = {"B", "D", "E", "D-E", "K"}          # resource rent, regulated utilities, finance (within the main set)


def industry():
    p = pd.read_csv(OUT / "agg_industry.csv.gz")
    p = p[~p.ind.isin(EXCL_MAIN) & p.cfc_sh.notna() & (p.geo != "BG")].copy()   # BG: no capital (A-V5-COV)
    p["Q_VA"] = p.VA_Q                     # KLEMS VA_Q: chain-linked volume, reference 2015, mn national currency
    p["hq"] = p.H / p.Q_VA
    return p


def n_dec(n):
    return max(1, int(round(n / 10)))


def measures(p):
    e = p[~p.ind.isin(NONAUTO) & np.isfinite(p.hq) & (p.hq > 0)]
    rows, lists = [], {}
    for (geo, y), g in e.groupby(["geo", "year"]):
        g = g.sort_values("hq")
        dec = g.head(n_dec(len(g)))
        rows.append(dict(geo=geo, year=y, A1=dec.K.sum() / g.K.sum(), n_elig=len(g),
                         dec_inds=",".join(dec.ind)))
    m = pd.DataFrame(rows)
    for geo, g in e.groupby("geo"):
        y0 = 2000 if (g.year == 2000).any() else g.year.min()
        g0 = g[g.year == y0].sort_values("hq")
        lists[geo] = set(g0.head(n_dec(len(g0))).ind)
    fx = e.assign(dec=[i in lists[g] for g, i in zip(e.geo, e.ind)]).groupby(["geo", "year"]).apply(
        lambda s: s[s.dec].K.sum() / s.K.sum(), include_groups=False).rename("A1fix").reset_index()
    return m.merge(fx, on=["geo", "year"], how="left"), lists


def five(d, cols):
    rows = []
    for (v, geo), g in d.groupby(["variant", "geo"]):
        for per, yrs in PERIODS.items():
            s = g[g.year.isin(yrs)]
            if len(s) < len(yrs):
                continue
            S = s[["PI", "K", "W", "GO", "H", "Q_K"]].sum()
            rows.append(dict(variant=v, geo=geo, period=per, rM=S.PI / (S.K + S.W), e=S.PI / S.W, pcm=S.PI / S.GO,
                             PI=S.PI / len(s), H=S.H / len(s), Q_K=s.Q_K.mean(), gap=s.gap.mean(),
                             **{c: s[c].mean() for c in cols}))
    f = pd.DataFrame(rows).sort_values(["variant", "geo", "period"])
    with np.errstate(divide="ignore", invalid="ignore"):
        for c in ("rM", "e", "PI"):
            f["ln" + c] = np.where(f[c] > 0, np.log(f[c]), np.nan)
    f["lnH"], f["lnQ_K"] = np.log(f.H), np.log(f.Q_K)
    g = f.groupby(["variant", "geo"])
    for c in ["lnrM", "lne", "lnPI", "lnH", "lnQ_K", "gap", "pcm"] + cols:
        f["d_" + c] = g[c].diff()
    return f[g.period.diff() == 1]


def tests(d):
    A = ["A1", "A2", "A1fix"]
    with np.errstate(divide="ignore", invalid="ignore"):
        d = d.assign(lnrM=np.where(d.PI > 0, np.log(d.rM), np.nan), lne=np.where(d.PI > 0, np.log(d.e), np.nan))
    f = five(d, A)
    rows = []
    for v in TRAPS:
        s, fv = d[d.variant == v], f[f.variant == v]
        for a in A:
            rows.append(dict(test="T2.1a_levels", variant=v, A=a, **ols_fwl(s, "rM", a, ["gap", "pcm"], ["geo", "year"])))
            rows.append(dict(test="T2.1b_levels", variant=v, A=a,
                             **ols_fwl(s, "rM", a, ["gap", "pcm", "lne"], ["geo", "year"])))
            rows.append(dict(test="T2.1a_5y", variant=v, A=a,
                             **ols_fwl(fv, "d_lnrM", "d_" + a, ["d_gap", "d_pcm"], ["geo", "period"])))
            rows.append(dict(test="T2.1b_5y", variant=v, A=a,
                             **ols_fwl(fv, "d_lnrM", "d_" + a, ["d_gap", "d_pcm", "d_lne"], ["geo", "period"])))
            rows.append(dict(test="T2.2_5y", variant=v, A=a, **ols_fwl(fv, "d_lne", "d_" + a, ["d_gap"], ["geo", "period"])))
        rows.append(dict(test="T2.3_H", variant=v, A="",
                         **ols_fwl(fv, "d_lnPI", "d_lnH", ["d_lnQ_K", "d_gap"], ["geo", "period"])))
        rows.append(dict(test="T2.3_QK", variant=v, A="",
                         **ols_fwl(fv, "d_lnPI", "d_lnQ_K", ["d_lnH", "d_gap"], ["geo", "period"])))
    r = pd.DataFrame(rows)
    for (t, v), g in r[r.A.isin(["A1", "A2"])].groupby(["test", "variant"]):   # Holm over the two measures
        h = holm(dict(zip(g.index, g.p_wild)))
        r.loc[g.index, "p_holm"] = [h[i] for i in g.index]
    return r, f


def lowlab(p, lists=None):
    e = p[~p.ind.isin(NONAUTO) & p.year.between(2015, 2019)]
    g = e.groupby(["geo", "ind"])[["H", "Q_VA", "PI", "K", "GO", "K_intNA", "VA", "W"]].sum(min_count=1).reset_index()
    g["hq"] = g.H / g.Q_VA
    rows = []
    for geo, s in g.dropna(subset=["hq"]).groupby("geo"):
        s = s.sort_values("hq")
        k = n_dec(len(s))
        for grp, t in (("bottom_decile", s.head(k)), ("rest", s.iloc[k:])):
            T = t[["PI", "K", "GO", "K_intNA", "VA", "W"]].sum()
            rows.append(dict(geo=geo, group=grp, inds=",".join(t.ind) if grp == "bottom_decile" else "",
                             r=T.PI / T.K, pcm=T.PI / T.GO, intang=T.K_intNA / T.K, e=T.PI / T.W, n=len(t)))
    return pd.DataFrame(rows)


def transfer(p):
    q = p.copy()
    tot = q.groupby(["geo", "year"])[["PI", "W"]].transform("sum")
    q["ebar"] = tot.PI / tot.W
    q["T"] = q.PI - q.ebar * q.W
    q["t_go"], q["pcm"] = q["T"] / q.GO, q.PI / q.GO
    q["intang"] = q.K_intNA / q.K
    q["rent"] = q.ind.isin(RENT).astype(float)
    q["ln_hq"] = np.log(q.hq)
    q["cy"] = q.geo + q.year.astype(str)
    q = q.replace([np.inf, -np.inf], np.nan)
    rows = []
    for x in ("pcm", "intang", "rent"):
        others = [c for c in ("pcm", "intang", "rent") if c != x]
        for spec, ctrl in (("all_controls", others + ["ln_hq"]), ("ln_hq_only", ["ln_hq"])):
            rows.append(dict(x=x, spec=spec, **ols_fwl(q, "t_go", x, ctrl, ["cy"], cluster="ind", B=999)))
    return q, pd.DataFrame(rows)


if __name__ == "__main__":
    p = industry()
    m, lists = measures(p)
    d = load()
    ict = d[d.variant == "main"][["geo", "year", "ict_share"]].rename(columns={"ict_share": "A2"})
    m = m.merge(ict, on=["geo", "year"], how="outer")
    m.to_csv(OUT / "s2_measures.csv", index=False)
    print(m.groupby("geo")[["A1", "A1fix", "A2"]].agg(["first", "last"]).round(3).to_string())
    print(pd.Series({g: ",".join(sorted(v)) for g, v in lists.items()}).to_string())
    d = d.merge(m[["geo", "year", "A1", "A1fix", "A2"]], on=["geo", "year"], how="left")
    r, f = tests(d)
    r.to_csv(OUT / "s2_tests.csv", index=False)
    print(r.round(3).to_string())
    ll = lowlab(p)
    ll.to_csv(OUT / "s2_lowlab.csv", index=False)
    print(ll.round(3).to_string())
    q, tr = transfer(p)
    q[["geo", "year", "ind", "T", "t_go", "pcm", "intang", "rent", "ln_hq", "ebar"]].to_csv(
        OUT / "s2_transfer.csv.gz", index=False)
    tr.to_csv(OUT / "s2_transfer_reg.csv", index=False)
    print(tr.round(3).to_string())
