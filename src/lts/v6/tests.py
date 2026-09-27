"""Iteration 6, stages 2-3: decompositions and tests on the long series (pre_registration_v6.md, sections 2-3).

Outputs (results/v6/): decomp_decades.csv, decomp_full.csv, trends.csv, breaks.csv, panel.csv, compare_v5.csv,
criteria.md
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ..v5.stage1 import nw_se, ols_fwl, sup_f
from .series import MAIN8, OUT

DECADES = {1: (1960, 1969), 2: (1970, 1979), 3: (1980, 1989), 4: (1990, 1999), 5: (2000, 2009), 6: (2010, 2019)}
HALF = {i + 1: (y, y + 4) for i, y in enumerate(range(1960, 2020, 5))}


def load():
    return pd.read_csv(OUT / "series.csv")


# ---------------------------------------------------------------- decompositions
def logs(g):
    pos = g.PI > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        return g.assign(lnr=np.where(pos, np.log(g.r), np.nan), lnrM=np.where(pos, np.log(g.rM), np.nan),
                        lne=np.where(pos, np.log(g.e), np.nan), ln1k=np.log1p(g.k),
                        lns=np.where(pos, np.log(g.share), np.nan), lnqyk=np.log(g.Q_Y / g.Q_K),
                        lnpyk=np.log(g.P_Y / g.P_K), lntech=np.log(g.tech), lnpkw=np.log(g.P_K / g.w_hour),
                        lnk=np.log(g.k))


def contrib(g, a, b):
    A, B = g.loc[a], g.loc[b]
    return dict(y0=a, y1=b, r0=A.r, r1=B.r, rM0=A.rM, rM1=B.rM, dlnr=B.lnr - A.lnr, c_share=B.lns - A.lns,
                c_qyk=B.lnqyk - A.lnqyk, c_pyk=B.lnpyk - A.lnpyk, dlnrM=B.lnrM - A.lnrM, c_e=B.lne - A.lne,
                c_1k=-(B.ln1k - A.ln1k), e0=A.e, e1=B.e, k0=A.k, k1=B.k, dlnk=B.lnk - A.lnk,
                c_tech=B.lntech - A.lntech, c_pkw=B.lnpkw - A.lnpkw)


def decompositions(s):
    rows, full = [], []
    for (geo, v), g in s.groupby(["geo", "variant"]):
        g = logs(g.set_index("year").sort_index())
        for dec, (a, b) in list(DECADES.items()) + [(7, (2020, 2024))]:
            if a in g.index and b in g.index:
                rows.append(dict(geo=geo, variant=v, decade=f"{a}-{b}", **contrib(g, a, b)))
        a, b = g.index.min(), g.index.max()
        row = dict(geo=geo, variant=v, **contrib(g, a, b))
        dd = g[["lnr", "lns", "lnqyk", "lnpyk", "lnrM", "lne", "ln1k"]].diff().dropna()
        if len(dd) > 10:
            vr, vm = dd.lnr.var(), dd.lnrM.var()
            row.update(v_share=dd.lnr.cov(dd.lns) / vr, v_qyk=dd.lnr.cov(dd.lnqyk) / vr, v_pyk=dd.lnr.cov(dd.lnpyk) / vr,
                       v_e=dd.lnrM.cov(dd.lne) / vm, v_1k=-dd.lnrM.cov(dd.ln1k) / vm)
        full.append(row)
    return pd.DataFrame(rows), pd.DataFrame(full)


# ---------------------------------------------------------------- T1 trends
def trend(y, lag):
    n = len(y)
    X = np.column_stack([np.ones(n), np.arange(n)])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    se = nw_se(X, y - X @ b, lag)[1]
    return b[1], se, 2 * stats.t.sf(abs(b[1] / se), n - 2)


def trends(s):
    rows = []
    for (geo, v), g in s.groupby(["geo", "variant"]):
        g = g.set_index("year").sort_index()
        s0, s1 = g.index.min(), g.index.max()
        starts = [s0, s0 + 5] + ([1946] if geo == "USA_NFC" else [])
        if geo == "USA_NFC":
            starts = [1951, 1956, 1946]
        ends = [min(s1, 2024), 2019]
        for series in ("r", "rM"):
            for a in starts:
                for b in ends:
                    y = g.loc[a:b, series].dropna()
                    if len(y) < 20:
                        continue
                    for lag in (2, 4, 8):
                        sl, se, p = trend(y.to_numpy(), lag)
                        rows.append(dict(geo=geo, variant=v, series=series, y0=a, y1=b, lag=lag, slope=sl, se=se, p=p,
                                         n=len(y), base=(a == starts[0] and b == ends[0])))
    return pd.DataFrame(rows)


def robust_neg(tr):
    out = []
    for (geo, v, s_), g in tr.groupby(["geo", "variant", "series"]):
        out.append(dict(geo=geo, variant=v, series=s_, robust_neg=bool(((g.slope < 0) & (g.p < 0.05)).all()),
                        robust_pos=bool(((g.slope > 0) & (g.p < 0.05)).all()), n_specs=len(g),
                        share_neg_sig=((g.slope < 0) & (g.p < 0.05)).mean()))
    return pd.DataFrame(out)


# ---------------------------------------------------------------- T2 breaks
def seg_ssr(y, i, j):
    t = np.arange(i, j, dtype=float)
    X = np.column_stack([np.ones(j - i), t])
    b = np.linalg.lstsq(X, y[i:j], rcond=None)[0]
    e = y[i:j] - X @ b
    return float(e @ e)


def bai_perron(y, mmax=3, trim=0.15):
    n = len(y)
    h = max(4, int(np.floor(trim * n)))
    S = np.full((n + 1, n + 1), np.inf)
    for i in range(n):
        for j in range(i + h, n + 1):
            S[i, j] = seg_ssr(y, i, j)
    best = {0: (S[0, n], [])}
    # dynamic programming: F[m][j] = min SSR of first j obs with m breaks
    F = {0: {j: (S[0, j], []) for j in range(h, n + 1)}}
    for m in range(1, mmax + 1):
        F[m] = {}
        for j in range((m + 1) * h, n + 1):
            cands = [(F[m - 1][i][0] + S[i, j], F[m - 1][i][1] + [i]) for i in range(m * h, j - h + 1) if i in F[m - 1]]
            if cands:
                F[m][j] = min(cands, key=lambda c: c[0])
        if n in F[m]:
            best[m] = F[m][n]
    out = []
    for m, (ssr, br) in best.items():
        p = 2 * (m + 1) + m
        lwz = np.log(ssr / (n - p)) + p * 0.299 * (np.log(n) ** 2.1) / n
        out.append(dict(m=m, ssr=ssr, lwz=lwz, breaks=br))
    return pd.DataFrame(out)


def window_supf(y, years, lo=1970, hi=1982, B=999, seed=7):
    n = len(y)
    t = np.arange(n, dtype=float)
    X0 = np.column_stack([np.ones(n), t])

    def supf(yy):
        e0 = yy - X0 @ np.linalg.lstsq(X0, yy, rcond=None)[0]
        s0 = e0 @ e0
        best, arg = -np.inf, None
        for k, yr in enumerate(years):
            if lo <= yr <= hi and 3 <= k <= n - 3:
                D = (t >= k).astype(float)
                X1 = np.column_stack([X0, D, D * (t - k)])
                e1 = yy - X1 @ np.linalg.lstsq(X1, yy, rcond=None)[0]
                F = ((s0 - e1 @ e1) / 2) / ((e1 @ e1) / (n - 4))
                if F > best:
                    best, arg = F, yr
        return best, arg
    F, yr = supf(y)
    b = np.linalg.lstsq(X0, y, rcond=None)[0]
    e = y - X0 @ b
    rho = float(np.clip((e[1:] @ e[:-1]) / (e[:-1] @ e[:-1]), -0.99, 0.99))
    u = e[1:] - rho * e[:-1]
    u -= u.mean()
    rng = np.random.default_rng(seed)
    Fs = np.empty(B)
    for i in range(B):
        us = rng.choice(u, n)
        es = np.empty(n)
        es[0] = us[0] / np.sqrt(1 - rho ** 2)
        for j in range(1, n):
            es[j] = rho * es[j - 1] + us[j]
        Fs[i] = supf(X0 @ b + es)[0]
    return F, yr, (1 + (Fs >= F).sum()) / (B + 1)


def breaks(s, variants=("main",)):
    rows = []
    for (geo, v), g in s[s.variant.isin(variants)].groupby(["geo", "variant"]):
        g = g.set_index("year").sort_index()
        if geo == "USA_NFC":
            g = g.loc[1951:]
        for series in ("r", "rM"):
            y = g[series].dropna()
            bp = bai_perron(y.to_numpy())
            m = bp.loc[bp.lwz.idxmin()]
            F, yr, p = window_supf(y.to_numpy(), y.index.to_numpy())
            rows.append(dict(geo=geo, variant=v, series=series, n=len(y), bp_m=int(m.m),
                             bp_dates=",".join(str(int(y.index[i])) for i in m.breaks),
                             bp3_dates=",".join(str(int(y.index[i])) for i in bp.loc[bp.m == bp.m.max(), "breaks"].iloc[0]),
                             win_F=F, win_year=yr, win_p=p))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- T3 panel
def hp_gap(y, lam=100):
    y = np.asarray(y, float)
    n = len(y)
    D = np.zeros((n - 2, n))
    for i in range(n - 2):
        D[i, i:i + 3] = [1, -2, 1]
    trend_ = np.linalg.solve(np.eye(n) + lam * D.T @ D, y)
    return 100 * (y - trend_)


def period_panel(s, geos, variant="main", periods=DECADES):
    rows = []
    for geo in geos:
        g = s[(s.geo == geo) & (s.variant == variant)].set_index("year").sort_index()
        if g.empty:
            continue
        ok = g.OVGD.notna()
        g.loc[ok, "hp"] = hp_gap(np.log(g.loc[ok, "OVGD"]))
        for per, (a, b) in periods.items():
            x = g.loc[a:b]
            if len(x) < b - a + 1:
                continue
            S = x[["PI", "K", "W"]].sum()
            rows.append(dict(geo=geo, period=per, rM=S.PI / (S.K + S.W), e=S.PI / S.W, k=S.K / S.W, r=S.PI / S.K,
                             hp=x.hp.mean(), gap=x.gap.mean() if x.gap.notna().all() else np.nan))
    f = pd.DataFrame(rows).sort_values(["geo", "period"])
    with np.errstate(divide="ignore", invalid="ignore"):
        f["lnrM"] = np.where(f.rM > 0, np.log(f.rM), np.nan)
    f["ln1k"] = np.log1p(f.k)
    g = f.groupby("geo")
    for c in ("lnrM", "ln1k", "hp", "gap"):
        f["d_" + c] = g[c].diff()
    return f[g.period.diff() == 1]


def panels(s):
    rows = []
    sets = {"main8": (MAIN8, "main"), "main8+JPN_nomi": (MAIN8 + ["JPN"], "no_mi"), "main8_nomi": (MAIN8, "no_mi"),
            "main8_hc": (MAIN8, "hc")}
    for name, (geos, var) in sets.items():
        for pname, per in (("decades", DECADES), ("halfdecades", HALF)):
            f = period_panel(s, geos, var, per)
            if f.empty:
                continue
            for fe, fe_name in ((["geo", "period"], "country+period"), (["geo"], "country")):
                for cyc in ([], ["d_hp"], ["d_gap"]):
                    try:
                        r = ols_fwl(f, "d_lnrM", "d_ln1k", cyc, fe, B=9999)
                    except Exception as exc:  # noqa: BLE001
                        r = dict(beta=np.nan, error=str(exc))
                    rows.append(dict(sample=name, periods=pname, fe=fe_name, cycle="+".join(cyc) or "none", **r))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- T4 comparison with iteration 5
def compare_v5(s):
    v5 = pd.read_csv(OUT.parent / "v5" / "agg_country_year.csv")
    v5 = v5[v5.variant == "main"]
    m = {"FRA": "FR", "DEU": "DE", "ITA": "IT", "NLD": "NL", "SWE": "SE", "GBR": "UK", "USA": "US"}
    rows = []
    for g3, g2 in m.items():
        a = s[(s.geo == g3) & (s.variant == "main")].set_index("year")
        b = v5[v5.geo == g2].set_index("year")
        yrs = a.index.intersection(b.index)
        yrs = yrs[(yrs >= 1995) & (yrs <= 2021)]
        sl_short, _, p_short = trend(a.loc[yrs, "rM"].to_numpy(), 4)
        sl_long, _, p_long = trend(a.loc[1960:2024, "rM"].dropna().to_numpy(), 4)
        rows.append(dict(geo=g3, n_overlap=len(yrs), corr_r_level=a.loc[yrs, "r"].corr(b.loc[yrs, "r"]),
                         corr_r_diff=a.loc[yrs, "r"].diff().corr(b.loc[yrs, "r"].diff()),
                         mean_r_ameco=a.loc[yrs, "r"].mean(), mean_r_v5=b.loc[yrs, "r"].mean(),
                         slope_rM_1995_2021=sl_short, p_1995_2021=p_short, slope_rM_1960_2024=sl_long, p_1960_2024=p_long))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- criteria
def criteria(full, rob, pan, tr):
    L = ["# Итерация 6, этап 3: предрегистрированные критерии (автоматически)\n"]
    m = rob[(rob.variant == "main") & rob.geo.isin(MAIN8) & (rob.series == "rM")]
    neg = m[m.robust_neg].geo.tolist()
    pos = m[m.robust_pos].geo.tolist()
    fm = full[(full.variant == "main") & full.geo.isin(MAIN8)].set_index("geo")
    kdriven = [g for g in neg if (abs(fm.loc[g, "c_1k"]) > abs(fm.loc[g, "c_e"])) and
               np.sign(fm.loc[g, "c_1k"]) == np.sign(fm.loc[g, "dlnrM"])]
    fell = fm[fm.dlnrM < 0]
    e_dominant = [g for g in fell.index if abs(fm.loc[g, "c_e"]) > abs(fm.loc[g, "c_1k"])]
    p = pan[(pan["sample"] == "main8") & (pan.periods == "decades") & (pan.cycle == "none")]
    main = p[p.fe == "country+period"].iloc[0]
    nofe = p[p.fe == "country"].iloc[0]
    pan_neg = (main.beta < 0) and (main.p_wild < 0.05) and (nofe.beta < 0) and (nofe.p_wild < 0.05)
    L.append(f"* Устойчиво отрицательный тренд r_M (все лаги и концы): {len(neg)} из 8 — {neg}")
    L.append(f"* Устойчиво положительный: {len(pos)} — {pos}")
    L.append(f"* Из них падение в основном за счёт k: {kdriven}")
    L.append(f"* Стран с падением r_M за весь период: {len(fell)}; из них вклад e по модулю больше вклада k: {e_dominant}")
    L.append(f"* Панель (десятилетия, без цикла): с эффектами периода β = {main.beta:.3f}, p_wild = {main.p_wild:.3f}; "
             f"без эффектов периода β = {nofe.beta:.3f}, p_wild = {nofe.p_wild:.3f}")
    pro = len(neg) >= 5 and len(kdriven) >= 5 and pan_neg
    against = (len(neg) <= 2 or len(e_dominant) > len(fell) / 2) and not pan_neg
    L.append(f"\n**Исход:** {'в пользу М' if pro else 'против М' if against else 'промежуточно'}")
    u = rob[(rob.geo == "USA_NFC") & (rob.series == "rM")]
    L.append("\nСША НФК, r_M, устойчиво отрицательный по вариантам: " +
             ", ".join(f"{r.variant}: {r.robust_neg} (доля спецификаций {r.share_neg_sig:.2f})" for r in u.itertuples()))
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    s = load()
    dd, full = decompositions(s)
    dd.to_csv(OUT / "decomp_decades.csv", index=False)
    full.to_csv(OUT / "decomp_full.csv", index=False)
    tr = trends(s)
    tr.to_csv(OUT / "trends.csv", index=False)
    rob = robust_neg(tr)
    rob.to_csv(OUT / "trends_robust.csv", index=False)
    print(rob[rob.variant.isin(["main", "no_mi", "hc", "no_ip"])].to_string(), flush=True)
    pan = panels(s)
    pan.to_csv(OUT / "panel.csv", index=False)
    print(pan.round(3).to_string(), flush=True)
    cv = compare_v5(s)
    cv.to_csv(OUT / "compare_v5.csv", index=False)
    print(cv.round(4).to_string(), flush=True)
    br = breaks(s, ("main", "no_mi", "hc"))
    br.to_csv(OUT / "breaks.csv", index=False)
    print(br.to_string(), flush=True)
    c = criteria(full, rob, pan, tr)
    (OUT / "criteria.md").write_text(c)
    print(c)
