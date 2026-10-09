"""Iteration 10: descriptive checks requested by the independent review (journal 16; results/v10/review_independent.md).

  sc51       donor placebo gaps for Hartz IV (R4)
  home       stage 6 with consistent normalisation: delta = d/(1+d) domestic hours and bundle/(1+d) per hour of any
             labour (R2); 2010, 2015, 2020
  rent       d by industry, all industries (R1)
  okishio    outcome 62 without the 2015-2020 window and with a country-cluster bootstrap (R6); real wage in the
             windows of outcome 63 (R7)
  selfemp    stage 9 r with labour income extended to the self-employed (R8)
  gamma      stage 5 reflected gamma vs shifted gamma and Laplace/Subbotin (R14)
  starbucks  share of one firm in the stage 2 estimation sample (R5)

Usage: PYTHONPATH=src python -P -m lts.v10.review [part ...]
Outputs: results/v10/rv_*.csv
"""
from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
from scipy import stats

from ..v6.series import ROOT
from ..v8.rule import decide, share_row

OUT = ROOT / "results" / "v10"


# ---------------------------------------------------------------- R4
def sc51():
    from .stage1 import DONORS, profit_share_ameco, sc_weights
    P = profit_share_ameco()
    pre, post = list(range(1991, 2005)), list(range(2005, 2013))
    D = P.reindex(pre + post)[["DEU"] + [d for d in DONORS if d != "DEU"]].dropna(axis=1)
    rows = []
    for t in D.columns:
        pool = [u for u in D.columns if u not in (t, "DEU")] if t != "DEU" else [u for u in D.columns if u != "DEU"]
        w = sc_weights(D.loc[pre, t].to_numpy(), D.loc[pre, pool].to_numpy())
        gap = D[t] - D[pool].to_numpy() @ w
        rows.append(dict(unit=t, post_gap=float(gap.loc[post].mean()), rmspe_pre=float(np.sqrt(np.mean(gap.loc[pre] ** 2))),
                         **{f"gap_{y}": float(gap.loc[y]) for y in post}))
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "rv_sc51_placebo_gaps.csv", index=False)
    de = R[R.unit == "DEU"].iloc[0]
    dn = R[R.unit != "DEU"]
    good = dn[dn.rmspe_pre <= 2 * de.rmspe_pre]
    S = pd.DataFrame([dict(item="donor post-gap SD (all donors)", value=dn.post_gap.std(), n=len(dn)),
                      dict(item="donor post-gap SD (pre-RMSPE <= 2x Germany)", value=good.post_gap.std(), n=len(good)),
                      dict(item="share of donors with |post gap| <= 0.5", value=(dn.post_gap.abs() <= 0.5).mean(), n=len(dn))])
    S.to_csv(OUT / "rv_sc51_summary.csv", index=False)
    print(S.round(3).to_string(), "\n", de.drop("unit").astype(float).round(2).to_string(), flush=True)


# ---------------------------------------------------------------- R2
def home():
    from ..core import Economy, spectral_radius
    from ..economy import build
    from .stage6 import ISO2, atus_d, hetus_d, shares
    A = pd.read_csv(OUT / "s6_atus_d.csv", index_col=0)
    H = pd.read_csv(OUT / "s6_hetus_d.csv", index_col=0)
    base = Economy.system

    def patch(e, d):
        def system(self, capital, closed, imports, l):
            M, Mm, lab = base(self, capital, closed, imports, l)
            if not closed or d <= 0:
                return M, Mm, lab
            n = self.n
            M, Mm = M.copy(), Mm.copy()
            delta = d / (1 + d)
            M[:n, n:] /= (1 + d)                                  # bundle per hour of any labour
            Mm[:n, n:] /= (1 + d)
            M[n:, n:] += delta * np.outer(self.hours / self.hours.sum(), np.ones(n))
            self.meta["_rho"] = spectral_radius(M + Mm if imports == "competitive" else M)
            return M, Mm, lab
        e.system = types.MethodType(system, e)
        return e
    rows = []
    for c in ("USA", "DEU", "FRA", "ITA", "ESP", "NLD", "AUT", "POL", "GBR"):
        d = float(A.loc[[y for y in A.index if y <= 2015][-1], "d"]) if c == "USA" else float(H.loc[ISO2[c], "d"])
        for y in (2010, 2015, 2020):
            dd = float(A.loc[y, "d"]) if c == "USA" else d
            e0, _ = build(c, y)
            s0, l0, _ = shares(e0)
            e1, _ = build(c, y)
            e1 = patch(e1, dd)
            s1, l1, _ = shares(e1)
            rows.append(dict(country=c, year=y, d=dd, share_d0=s0, share_consistent=s1, labour_part_d0=l0,
                             labour_part_consistent=l1, rho=e1.meta.get("_rho")))
            print("home", c, y, round(s0, 4), round(s1, 4), flush=True)
    R = pd.DataFrame(rows)
    R["identical"] = (R.share_d0 - R.share_consistent).abs() < 1e-9
    R.to_csv(OUT / "s6_consistent.csv", index=False)
    print(R.round(4).to_string(), flush=True)


# ---------------------------------------------------------------- R1
def rent():
    from .stage10 import figaro_d
    D = figaro_d("pp_actual_wage")
    m = D.groupby(["country", "industry"]).d.mean().groupby(level=1).agg(["mean", "count"])
    m = m.sort_values("mean", ascending=False)
    m["rank"] = np.arange(1, len(m) + 1)
    m.to_csv(OUT / "s10_d_by_industry.csv")
    print(m.head(12).round(3).to_string(), "\n", m.loc[["B", "A01", "L"]].round(3).to_string(), flush=True)


# ---------------------------------------------------------------- R6, R7
def okishio():
    from .stage11 import pwt
    W = pd.read_csv(OUT / "s11_windows.csv")
    w = W.dropna(subset=["marx_bias"]).copy()
    w["marx_bias"] = w.marx_bias.astype(bool)
    rows = [share_row("62 (descriptive): without the 2015-2020 window", w[w.start != 2015].marx_bias, 0.5, 0.1, ">")]
    rng = np.random.default_rng(62)
    cs = w.country.unique()
    g = {c: w[w.country == c].marx_bias.to_numpy() for c in cs}
    bs = []
    for _ in range(2000):
        pick = rng.choice(cs, len(cs))
        v = np.concatenate([g[c] for c in pick])
        bs.append(v.mean())
    lo, hi = np.quantile(bs, [0.05, 0.95])
    est = w.marx_bias.mean()
    rows.append(dict(outcome="62 (descriptive): country-cluster bootstrap", est=est, ci90_lo=lo, ci90_hi=hi, n=len(w),
                     label=decide(est, lo, hi, 0.5, 0.1, ">")))
    loo = [w[w.country != c].marx_bias.mean() for c in cs]
    rows.append(dict(outcome="62 (descriptive): leave-one-country-out estimates <= 0.60", est=float(np.mean(np.array(loo) <= 0.60)),
                     n=len(cs)))
    p = pwt()
    p["w_real_h"] = p.labsh * p.rgdpna / p.L
    f = W.dropna(subset=["dln_r"])
    f = f[f.dln_r < -0.05].copy()
    dw = []
    for r in f.itertuples():
        a, b = (r.country, r.start), (r.country, r.end)
        dw.append(np.log(p.loc[b, "w_real_h"] / p.loc[a, "w_real_h"]) if a in p.index and b in p.index else np.nan)
    f["dln_wreal"] = dw
    ykd = f.dln_yk_nom < 0.5 * f.dln_r
    rows += [dict(outcome="63 (descriptive): falling-r windows with non-rising real hourly wage",
                  est=float((f.dln_wreal <= 0).sum()), n=int(f.dln_wreal.notna().sum())),
             dict(outcome="63 (descriptive): 'Marx' windows (Y/K > half, omega not rising) with rising real wage",
                  est=float(((f.d_omega <= 0) & ykd & (f.dln_wreal > 0)).sum()), n=int(((f.d_omega <= 0) & ykd).sum())),
             dict(outcome="63 (descriptive): Y/K-dominant windows with rising omega", est=float((ykd & (f.d_omega > 0)).sum()),
                  n=int(ykd.sum())),
             dict(outcome="63 (descriptive): falling-r windows with rising omega (all)", est=float((f.d_omega > 0).sum()),
                  n=len(f))]
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "rv_okishio.csv", index=False)
    print(R.round(4).to_string(), flush=True)


# ---------------------------------------------------------------- R8
def selfemp():
    from . import stage9 as s9
    b = pd.read_stata(s9.R10 / "oecd_bsdb.dta")
    b["geo"] = b.cty.replace(s9.BSDB_ISO)
    a = b.groupby(["geo", "year"])[["gdpb", "wsse", "etb", "kbv", "pib"]].mean().reset_index()
    a["year"] = a.year.astype(int)
    a["comp"] = a.wsse * a.etb / 1e6
    a["r"] = (a.gdpb - a.comp) / (a.kbv * a.pib / 100)
    a["e"] = (a.gdpb - a.comp) / a.comp
    B = a.dropna(subset=["r"])[["geo", "year", "r", "e"]]
    K09 = s9.klems09()
    K09 = K09.assign(COMP=K09.LAB)
    M09 = s9.market09(K09)
    na = pd.read_csv(s9.R9B / "klems2025_national_accounts.csv", usecols=["nace_r2_code", "geo_code", "year", "VA_CP", "COMP",
                                                                         "EMP", "EMPE"], low_memory=False)
    ca = pd.read_csv(s9.R9B / "klems2025_capital_accounts.csv", usecols=["nace_r2_code", "geo_code", "year", "K_GFCF"],
                     low_memory=False)
    m = na.merge(ca, on=["nace_r2_code", "geo_code", "year"], how="left")
    m = m[m.geo_code.isin(s9.ISO3) & (m.nace_r2_code == "MARKT")].dropna(subset=["VA_CP", "COMP", "EMP", "EMPE", "K_GFCF"])
    m = m[(m.K_GFCF > 0) & (m.EMPE > 0)]
    lab = m.COMP * m.EMP / m.EMPE
    M25 = pd.DataFrame(dict(geo=m.geo_code.map(s9.ISO3), year=m.year.astype(int), r=(m.VA_CP - lab) / m.K_GFCF,
                            e=(m.VA_CP - lab) / lab))
    series, F, T = s9.run_r(B, M09, M25, "self-employed adjusted")
    T.to_csv(OUT / "rv_s9_selfemp_trends.csv", index=False)
    o = share_row("58 (descriptive): self-employed adjusted labour income", T.label == "подтверждено", 0.5, 0.15, ">",
                  countries=",".join(T[T.label == "подтверждено"].geo))
    pd.DataFrame([o]).to_csv(OUT / "rv_s9_selfemp_outcome.csv", index=False)
    print(T[["geo", "years", "est", "label"]].round(5).to_string(), "\n", o, flush=True)


# ---------------------------------------------------------------- R14
def gamma():
    from .stage5 import sample
    d = sample()
    rows = []
    for y in range(2011, 2026):
        x = d.loc[d.fy == y, "r_op"].dropna().to_numpy()
        x = x[np.abs(x) <= 1]
        n = len(x)
        out = dict(year=y, n=n)
        for name, dist, data in (("normal", stats.norm, x), ("laplace", stats.laplace, x)):
            p = dist.fit(data)
            out[f"ll_{name}"] = float(dist.logpdf(data, *p).sum())
        best = None
        for a0 in (1.5, 3.0, 10.0):                                 # reflected gamma: fit to -x
            try:
                p = stats.gamma.fit(-x, a0, loc=(-x).min() - 0.05, scale=np.std(x) / np.sqrt(a0))
                ll = float(stats.gamma.logpdf(-x, *p).sum())
                if np.isfinite(ll) and (best is None or ll > best[0]):
                    best = (ll, p[0])
            except Exception:                                        # noqa: BLE001
                continue
        out["ll_gamma_reflected"], out["shape_reflected"] = best if best else (np.nan, np.nan)
        rows.append(out)
    R = pd.DataFrame(rows)
    F = pd.read_csv(OUT / "s5_fits_r_op.csv")[["year", "bic_gamma_shift", "bic_normal", "bic_laplace", "bic_asym_subbotin"]]
    R = R.merge(F, on="year")
    R["bic_gamma_reflected"] = 3 * np.log(R.n) - 2 * R.ll_gamma_reflected
    R["gamma_shift_worse_than_normal"] = R.bic_gamma_shift > R.bic_normal
    R["reflected_worse_than_laplace"] = R.bic_gamma_reflected > R.bic_laplace
    R.to_csv(OUT / "rv_s5_gamma.csv", index=False)
    print(R.round(1).to_string(), flush=True)


# ---------------------------------------------------------------- R5
def starbucks():
    M = pd.read_csv(OUT / "s2_matched_elections.csv")
    w = M[((M.v - 0.5).abs() <= 0.15) & M.d_roa_2.notna()]
    sb = M.cik == 829224
    S = pd.DataFrame([dict(item="Starbucks among matched elections", k=int(sb.sum()), n=len(M)),
                      dict(item="Starbucks in the estimation sample (bw 0.15, d_roa_2)", k=int((w.cik == 829224).sum()), n=len(w)),
                      dict(item="estimation sample at v = 0.5 exactly", k=int((w.v == 0.5).sum()), n=len(w))])
    yrs = w[w.cik == 829224].year.value_counts().sort_index().to_dict()
    S["note"] = ["", f"years {yrs}", ""]
    S.to_csv(OUT / "rv_s2_starbucks.csv", index=False)
    print(S.to_string(), flush=True)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    parts = sys.argv[1:] or ["sc51", "rent", "okishio", "selfemp", "gamma", "starbucks", "home"]
    for part in parts:
        globals()[part]()
