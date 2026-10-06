"""Iteration 8, stage 3: price-test cleanup (pre_registration_v8.md, stage 3 and journal 5).

3.1 market sector, FISIM separately, factor-cost prices: outcomes 1-5 recomputed on FIGARO (13 countries, 2010-2022).
3.2 Eurostat gross (N11G) and net (N11N) capital stock as candidates in the equal-dispersion test (B0b).
3.3 disaggregation gradient on the BEA 2017 detail table: detail -> NAICS 3 -> summary.
3.4 dynamic test T1 without hedonically deflated industries.
Outputs: results/v8/s3_*.csv
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from ..core import leontief_inverse
from ..economy import build
from ..levels import mask_for
from ..v2.baskets import closed_bases
from ..v2.placebo_cost import metrics_mat
from ..v3.placebo_disp import COUNTRIES, YEARS, Draws, Setup
from ..v3.reduction import commodity_Z
from .. import oecd
from ..codes import divisions
from ..levels import SUBSETS
from ..v4 import b0b
from ..v4.b0b import label_rows
from ..v4.b_data import divs
from ..v6.national import es
from .rule import mean_row, share_row
from .stage2 import OUT, R8

MARKET = {"O84", "P85", "Q86", "Q87_88", "L"}
FISIM = {"K64", "K65", "K66"}
VARIANTS = {"all": (set(), False), "market": (MARKET, False), "market_nofisim": (MARKET | FISIM, False),
            "market_fc": (MARKET, True), "market_nofisim_fc": (MARKET | FISIM, True)}


# ---------------------------------------------------------------- 3.1
def vectors(e, c, y, S, D):
    """All ratio vectors of one country-year: dict name -> (P, n) array."""
    h = e.l()
    out = {"hours": S.z(h[None]), "flat": S.z(np.where(e.x > 0, 1.0, 0.0)[None])}
    r = min(e.actual_profit_rate(True), 0.99 * e.max_profit_rate(True, "price"))
    out["pp_actual"] = e.prices_of_production(r, True, "price", "actual")[None]
    out["bases_asym"] = commodity_Z(e, S)
    M, Mm, _ = e.system(True, False, "price", h)
    li = e.labour_income
    wbar = li.sum() / e.hours.sum()
    C = e.hh_dom.sum() + e.hh_imp.sum()
    zk, _ = closed_bases(M, Mm, wbar * np.outer(e.hh_dom / C, h), wbar * np.outer(e.hh_imp / C, h), e.x)
    out["bases_sym"] = np.array(list(zk.values()))
    out["perm"] = S.z(D.rows("perm", h))
    out["lnorm"] = S.z(D.rows("lnorm", h))
    return out


def market_rows(c, y, e, V, t):
    rows = []
    for vname, (excl, fc) in VARIANTS.items():
        m = mask_for(e.labels, excl)
        x = e.x[m] * ((1 - t[m]) if fc else 1)
        met = {k: metrics_mat(Z[:, m] / ((1 - t[m]) if fc else 1), x) for k, Z in V.items()}
        base = dict(country=c, year=y, variant=vname, n_eval=int(m.sum()))
        for mt in ("mawd", "d"):
            h = met["hours"][mt][0]
            rows.append(dict(base, metric=mt, hours=h, flat=met["flat"][mt][0], pp_actual=met["pp_actual"][mt][0],
                             bases_asym_better=float((met["bases_asym"][mt] < h).mean()),
                             bases_sym_better=float((met["bases_sym"][mt] < h).mean()),
                             perm_better=float((met["perm"][mt] < h).mean()),
                             lnorm_better=float((met["lnorm"][mt] < h).mean())))
    return rows


def run_market(countries=COUNTRIES):
    rows = []
    for c in countries:
        D = None
        for y in YEARS:
            e, _ = build(c, y)
            S = Setup(e)
            if D is None:
                D = Draws(c, e.n, S.base_mask)
            t = np.divide(label_rows(c, y, e.labels)["D29X39"], e.x, out=np.zeros(e.n), where=e.x > 0)
            t = np.clip(np.nan_to_num(t), -0.5, 0.5)
            rows += market_rows(c, y, e, vectors(e, c, y, S, D), t)
        print("s3 market", c, flush=True)
        pd.DataFrame(rows).to_csv(OUT / "s3_market_levels.csv", index=False)
    return pd.DataFrame(rows)


def market_outcomes(L=None):
    L = pd.read_csv(OUT / "s3_market_levels.csv") if L is None else L
    a = L.groupby(["variant", "country", "metric"]).mean(numeric_only=True).unstack("metric")
    res = []
    for v, g in a.groupby(level="variant"):
        g = g.droplevel("variant")
        top5 = (g[[("perm_better", "mawd"), ("perm_better", "d"), ("lnorm_better", "mawd"), ("lnorm_better", "d")]]
                <= 0.05).all(1)
        res += [mean_row("1 asym: share of commodity bases better than hours (MAWD)", g[("bases_asym_better", "mawd")],
                         0.5, 0.10, "<", variant=v),
                mean_row("2 sym: share of commodity bases better than hours (MAWD)", g[("bases_sym_better", "mawd")],
                         0.5, 0.10, "<", variant=v),
                share_row("3 stage P: hours in top 5% vs placebos (a), (b)", top5, 0.05, 0.15, ">", variant=v),
                share_row("4 flat better than hours (MAWD)", g[("flat", "mawd")] < g[("hours", "mawd")], 0.5, 0.15, "<",
                          variant=v),
                share_row("5 PP (actual wages) better than hours (MAWD)", g[("pp_actual", "mawd")] < g[("hours", "mawd")],
                          0.5, 0.15, "<", variant=v)]
    return pd.DataFrame(res)


# ---------------------------------------------------------------- 3.2
EU9 = ("AUT", "CZE", "DEU", "ESP", "FRA", "ITA", "NLD", "POL", "GBR")
_ST = None


def eu_stock():
    global _ST
    if _ST is None:
        st = es(R8 / "eurostat_nama_10_nfa_st.json")
        st["year"] = st.time.astype(int)
        _ST = st.dropna(subset=["value"]).set_index(["geo", "asset10", "year"]).sort_index()
    return _ST


def _divs(code):
    try:
        return frozenset(divs(code))
    except Exception:
        return frozenset()


def eu_capital(e, c3, y, asset):
    """Stock per FIGARO label: finest disjoint Eurostat codes available, connected groups split by CFC."""
    geo = "UK" if c3 == "GBR" else oecd.ISO3_TO_2[c3]
    st = eu_stock()
    if (geo, asset, y) not in st.index:
        return None
    g = st.loc[(geo, asset, y)]
    cand = sorted([(len(_divs(r.nace_r2)), r.nace_r2, _divs(r.nace_r2), r.value) for r in g.itertuples()
                   if r.nace_r2 != "TOTAL" and _divs(r.nace_r2)], key=lambda t: t[0])
    used, kind = set(), []
    for _, code, dv, v in cand:
        if not (dv & used):
            kind.append((code, dv, v))
            used |= dv
    ldiv = [set().union(*[divisions(m) or set() for m in lab.split("+")]) for lab in e.labels]
    n = e.n
    par = list(range(n + len(kind)))

    def f(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a
    for i, ld in enumerate(ldiv):
        for k, (_, kd, _) in enumerate(kind):
            if ld & kd:
                par[f(i)] = f(n + k)
    K = np.full(n, np.nan)
    groups = {}
    for i in range(n):
        groups.setdefault(f(i), []).append(i)
    for root, labs in groups.items():
        ks = [k for k in range(len(kind)) if f(n + k) == root]
        if not ks or not set().union(*[ldiv[i] for i in labs]) <= set().union(*[kind[k][1] for k in ks]):
            continue
        c = e.cfc[labs]
        K[labs] = sum(kind[k][2] for k in ks) * c / c.sum() if c.sum() > 0 else np.nan
    return K


_orig_candidates = b0b.candidates


def candidates_eu(e, c3, y):
    coef, nfix = _orig_candidates(e, c3, y)
    x = e.x
    for name, asset in (("eu_gross", "N11G"), ("eu_net", "N11N")):
        K = eu_capital(e, c3, y, asset)
        if K is None or np.isfinite(K).mean() <= 0.9:
            continue
        K = np.where(np.isfinite(K), K, np.nanmedian(K / x) * x)
        cc = np.divide(K, x, out=np.zeros(e.n), where=x > 0)
        pos = cc > 0
        nfix[name] = int((~pos & (x > 0)).sum())
        coef[name] = np.where(pos, cc, cc[pos].min())
    return coef, nfix


def run_gross(countries=EU9):
    b0b.candidates = candidates_eu
    b0b.CANDS = b0b.CANDS + ["eu_gross", "eu_net"]
    L, Bt = [], []
    for c in countries:
        lev, bt = b0b.summarise(b0b.run_country(c))
        L.append(lev)
        Bt.append(bt)
        pd.concat(L).to_csv(OUT / "s3_gross_levels.csv", index=False)
        pd.concat(Bt).to_csv(OUT / "s3_gross_boot.csv", index=False)


def gross_counts():
    b = pd.read_csv(OUT / "s3_gross_boot.csv")
    out = []
    for (mask, meth), g in b.groupby(["mask", "method"]):
        w = g.pivot_table(index=["country", "cand"], columns="metric", values=["hours_sig_better", "cand_sig_better", "delta"])
        hb = (w["hours_sig_better"]["mawd"] > 0) & (w["hours_sig_better"]["d"] > 0)
        nw = ~((w["hours_sig_better"]["mawd"] > 0) | (w["hours_sig_better"]["d"] > 0))
        pt = (w["delta"]["mawd"] > 0) & (w["delta"]["d"] > 0)
        s = pd.DataFrame({"n": 1, "hours_sig_better_both": hb, "not_worse": nw, "hours_point_better_both": pt}) \
            .groupby(level="cand").sum()
        s["mask"], s["method"] = mask, meth
        out.append(s.reset_index())
    return pd.concat(out)


# ---------------------------------------------------------------- 3.3
BEA_IO = R8.parent / "bea" / "AllTablesIO.zip"
SPECIAL = ["S00401", "S00402", "S00300", "S00900"]


def _sheet(name, sheet="2017"):
    import zipfile
    return pd.read_excel(zipfile.ZipFile(BEA_IO).open(name), sheet_name=sheet, header=None)


def _table(name):
    d = _sheet(name)
    cols = [str(c).strip() for c in d.iloc[5, 2:]]
    rows = [str(r).strip() for r in d.iloc[6:, 0]]
    v = d.iloc[6:, 2:].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()
    return pd.DataFrame(v, index=rows, columns=cols)


def concordance():
    d = _sheet("IOUse_Before_Redefinitions_PRO_2017_Detail.xlsx", "NAICS Codes")
    d = d.iloc[5:]
    summ = d[1].where(d[0].isna() & d[1].notna()).ffill()
    det = d[d[3].notna() & d[4].notna()]
    return pd.DataFrame({"detail": det[3].astype(str).str.strip(), "summary": summ[det.index].astype(str).str.strip(),
                         "naics": det[6].astype(str).str.strip()}).set_index("detail")


def _naics_tokens(s):
    out = []
    for tok in str(s).replace(" ", "").split(","):
        tok = tok.replace("*", "")
        if not tok or not tok[0].isdigit():
            continue
        if "-" in tok:
            a, b = tok.split("-")
            for k in range(int(a[-len(b):]), int(b) + 1):
                out.append(a[:-len(b)] + str(k).zfill(len(b)))
        else:
            out.append(tok)
    return out


def qcew_private():
    import zipfile
    z = zipfile.ZipFile(R8 / "qcew_2017_annual_by_industry.zip")
    rows = []
    for n in z.namelist():
        parts = n.split("/")[-1].split(" ")
        if len(parts) < 3 or parts[2] != "NAICS" or not (parts[1].isdigit() and len(parts[1]) == 6):
            continue
        d = pd.read_csv(z.open(n), dtype=str, usecols=["area_fips", "own_code", "industry_code", "annual_avg_emplvl"])
        d = d[(d.area_fips == "US000") & (d.own_code == "5")]
        if len(d):
            rows.append((parts[1], float(d.annual_avg_emplvl.iloc[0])))
    return pd.Series(dict(rows), name="emp")


def bea_detail_system():
    U = _table("IOUse_Before_Redefinitions_PRO_2017_Detail.xlsx")
    V = _table("IOMake_Before_Redefinitions_2017_Detail.xlsx")
    inds = [c for c in U.columns if c in V.index]
    comms = [c for c in V.columns if c in U.index and c not in SPECIAL and not c.startswith("T0")]
    Vm = V.loc[inds, comms].to_numpy()
    q = V.loc[:, comms].drop(index=[i for i in V.index if not i or i.startswith("T0")], errors="ignore").sum(0).to_numpy()
    imp = -U.loc[comms, "F05000"].to_numpy()
    s_imp = np.clip(np.divide(imp, q + imp, out=np.zeros(len(q)), where=(q + imp) > 0), 0, 1)
    Uc = U.loc[comms, inds].to_numpy()
    Dm = np.divide(Vm, q, out=np.zeros_like(Vm), where=q > 0)
    Z = np.clip(Dm @ ((1 - s_imp)[:, None] * Uc), 0, None)       # journal 6: rare small negative entries -> 0
    m = (s_imp[:, None] * Uc).sum(0) + U.loc[[r for r in SPECIAL if r in U.index], inds].to_numpy().sum(0)
    x = U.loc["T008", inds].to_numpy()
    comp = U.loc["V00100", inds].to_numpy()
    return inds, Z, m, x, comp


def detail_employment(inds, comp, coverage_fix=False):
    cc = concordance()
    emp = qcew_private()
    tok_owner = {}
    for det, r in cc.iterrows():
        if det in inds:
            for t in _naics_tokens(r.naics):
                tok_owner.setdefault(t, []).append(det)
    ci = dict(zip(inds, comp))
    E = dict.fromkeys(inds, 0.0)
    unmatched = 0.0
    for code, e in emp.items():
        best = next((code[:k] for k in range(6, 1, -1) if code[:k] in tok_owner), None)
        if best is None:
            unmatched += e
            continue
        own = tok_owner[best]
        w = np.array([max(ci[o], 0) for o in own])
        w = w / w.sum() if w.sum() > 0 else np.full(len(own), 1 / len(own))
        for o, wi in zip(own, w):
            E[o] += e * wi
    E = pd.Series(E)
    matched = E > 0
    if coverage_fix:                                                 # journal 7: QCEW coverage gaps
        wpe = pd.Series(comp, index=inds)[matched] / E[matched]
        bad = wpe.index[wpe > 5 * wpe.median()]
        E[bad] = 0.0
        matched = E > 0
    wage = comp[matched.to_numpy()].sum() / E[matched].sum()
    imputed = (~matched) & (pd.Series(comp, index=inds) > 0)
    E[imputed] = np.asarray(comp)[imputed.to_numpy()] / wage
    return E, matched, dict(qcew_total=float(emp.sum()), unmatched=float(unmatched), n_imputed=int(imputed.sum()))


def level_maps(inds):
    cc = concordance()
    summ = [cc.summary.get(i, i) for i in inds]
    return {"detail": list(inds), "naics3": [i[:3] for i in inds], "summary": summ}


def gradient(B=2000, n_p=1000, coverage_fix=False):
    tag = "_fix" if coverage_fix else ""
    inds, Z, m, x, comp = bea_detail_system()
    E, matched, info = detail_employment(inds, comp, coverage_fix)
    gov = np.array([i.startswith(("S00", "GSLG")) or i in ("531HSO", "531HST", "814000") for i in inds])
    ev_det = matched.to_numpy() & ~gov & (x > 0)
    res, store = [], {}
    for lev, keys in level_maps(inds).items():
        labs = list(dict.fromkeys(keys))
        G = np.zeros((len(labs), len(inds)))
        for j, k in enumerate(keys):
            G[labs.index(k), j] = 1
        xg, Zg, mg, Eg = G @ x, G @ Z @ G.T, G @ m, G @ E.to_numpy()
        ev = (G @ ev_det.astype(float)) == G.sum(1)                  # a group is evaluated if all members are
        ok = xg > 0
        A = np.divide(Zg, xg, out=np.zeros_like(Zg), where=xg > 0)
        L = leontief_inverse(A)
        mu = np.divide(mg, xg, out=np.zeros(len(xg)), where=ok) @ L
        def z(rows):
            lam = np.atleast_2d(rows) @ L
            sc = (xg.sum() - mu @ xg) / (lam @ xg)
            return sc[:, None] * lam + mu[None]
        l = np.divide(Eg, xg, out=np.zeros(len(xg)), where=ok)
        flat = ok.astype(float)
        rng = np.random.default_rng(sum(map(ord, f"BEA2017_{lev}")))
        idx = np.where(ev)[0]
        perm = np.tile(l, (n_p, 1))
        for p in range(n_p):
            perm[p, idx] = l[idx][rng.permutation(len(idx))]
        pos = l > 0
        lnorm = np.exp(rng.normal(0, 1, (n_p, len(l))) * np.log(l[pos & ev]).std())
        Zh, Zf, Zp, Zl = z(l)[:, ev], z(flat)[:, ev], z(perm)[:, ev], z(lnorm)[:, ev]
        xe = xg[ev]
        mh, mf = metrics_mat(Zh, xe), metrics_mat(Zf, xe)
        mp, ml = metrics_mat(Zp, xe), metrics_mat(Zl, xe)
        row = dict(level=lev, n=int(ev.sum()), n_all=len(labs), **info)
        for mt in ("mawd", "d"):
            row.update({f"hours_{mt}": mh[mt][0], f"flat_{mt}": mf[mt][0],
                        f"perm_better_{mt}": float((mp[mt] < mh[mt][0]).mean()),
                        f"lnorm_better_{mt}": float((ml[mt] < mh[mt][0]).mean())})
        # bootstrap over evaluated industries (placebo set fixed, re-evaluated on the drawn industries)
        J = len(xe)
        I = np.random.default_rng(7 + J).integers(0, J, (B, J))
        bd, bp = np.empty(B), np.empty(B)
        for b in range(B):
            ii = I[b]
            h = metrics_mat(Zh[:, ii], xe[ii])["mawd"][0]
            bd[b] = metrics_mat(Zf[:, ii], xe[ii])["mawd"][0] - h
            bp[b] = (metrics_mat(Zp[:, ii], xe[ii])["mawd"] < h).mean()
        store[lev] = (bd, bp)
        row.update(flat_minus_hours_mawd=mf["mawd"][0] - mh["mawd"][0],
                   fmh_ci90=(float(np.quantile(bd, 0.05)), float(np.quantile(bd, 0.95))),
                   perm_ci90=(float(np.quantile(bp, 0.05)), float(np.quantile(bp, 0.95))))
        res.append(row)
        print("s3 gradient", lev, ev.sum(), flush=True)
    R = pd.DataFrame(res)
    R.to_csv(OUT / f"s3_gradient{tag}.csv", index=False)
    out = []
    det, sm_ = R.set_index("level").loc["detail"], R.set_index("level").loc["summary"]
    for name, k, i, theta_d, direction in (("gradient: share of permutations better (MAWD), detail - summary",
                                            "perm_better_mawd", 1, 0.1, "<"),
                                           ("gradient: MAWD(flat) - MAWD(hours), detail - summary",
                                            "flat_minus_hours_mawd", 0, 0.02, ">")):
        est = det[k] - sm_[k]
        dd = store["detail"][i] - store["summary"][i]
        lo, hi = float(np.quantile(dd, 0.05)), float(np.quantile(dd, 0.95))
        from .rule import decide
        out.append(dict(outcome=name, est=est, ci90_lo=lo, ci90_hi=hi, theta0=0, delta=theta_d, direction=direction,
                        label=decide(est, lo, hi, 0, theta_d, direction)))
    pd.DataFrame(out).to_csv(OUT / f"s3_gradient{tag}_outcomes.csv", index=False)
    return R, pd.DataFrame(out)


# ---------------------------------------------------------------- 3.4
HED_F = {"C26", "J58", "J59_60", "J61", "J62_63"}
HED_B = {"334", "511", "512", "513", "514"}


def run_hedonic(countries=COUNTRIES):
    from .. import bea
    from ..dynamics import prepare, t1_constrained_r2
    from ..v4.b0 import frames_for
    SUBSETS["core_nohed"] = SUBSETS["core"] | HED_F | HED_B
    SUBSETS["all_nohed"] = SUBSETS["all"] | HED_F | HED_B
    res = []
    for c in list(countries) + ["USA_BEA"]:
        src = "bea" if c == "USA_BEA" else "figaro"
        frames, D, P0 = [], None, None
        for y in (range(1998, 2024) if src == "bea" else range(2010, 2023)):
            e, _ = bea.build(y) if src == "bea" else build(c, y)
            if not np.isfinite(e.meta["price_index"]).any():
                frames = None
                break
            S = Setup(e)
            if D is None:
                D = Draws(c, e.n, S.base_mask)
                P0 = e.meta["price_index"]
            frames.append(frames_for(e, S, D, c, y, P0))
        if frames is None:
            continue
        df = pd.concat(frames, ignore_index=True)
        for subset in ("core", "core_nohed", "all", "all_nohed"):
            t = t1_constrained_r2(prepare(df, subset, bea.BEA_ROLE if src == "bea" else None), horizons=(1, 5))
            res.append(t.assign(subset=subset))
        print("s3 hedonic", c, flush=True)
    T = pd.concat(res)
    T.to_csv(OUT / "s3_hedonic_t1.csv", index=False)
    rows = []
    for (c, s, h), g in T.groupby(["country", "subset", "h"]):
        Lb = g[g.basis == "labour"].r2_w.iloc[0]
        P = g[g.basis.str.startswith("perm")].r2_w
        rows.append(dict(country=c, subset=s, h=int(h), labour=Lb, flat_phys=g[g.basis == "flat_phys"].r2_w.iloc[0],
                         perm_share_ge_labour=(P >= Lb).mean()))
    r = pd.DataFrame(rows)
    r["informative_vs_phys"] = (r.perm_share_ge_labour <= 0.05) & (r.labour > r.flat_phys)
    r.to_csv(OUT / "s3_hedonic_summary.csv", index=False)
    out = [share_row("7 B0: labour informative vs physical flat and permutations, h = 5", g.informative_vs_phys,
                     0.5, 0.15, ">", subset=s) for s, g in r[r.h == 5].groupby("subset")]
    pd.DataFrame(out).to_csv(OUT / "s3_hedonic_outcomes.csv", index=False)
    return r


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else "market"
    if part == "market":
        run_market(sys.argv[2].split(",") if len(sys.argv) > 2 else COUNTRIES)
        O = market_outcomes()
        O.to_csv(OUT / "s3_market_outcomes.csv", index=False)
        print(O.round(3).to_string())
    elif part == "gross":
        run_gross()
        C = gross_counts()
        C.to_csv(OUT / "s3_gross_counts.csv", index=False)
        print(C[C.cand.isin(["capital", "eu_gross", "eu_net", "cfc", "nonlabour"])].to_string())
    elif part == "gradient":
        R, O = gradient(coverage_fix=len(sys.argv) > 2 and sys.argv[2] == "fix")
        print(R.round(4).to_string())
        print(O.round(4).to_string())
    elif part == "hedonic":
        print(run_hedonic().round(3).to_string())
        print(pd.read_csv(OUT / "s3_hedonic_outcomes.csv").round(3).to_string())
