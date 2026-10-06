"""Iteration 8, stage 4: the law of the tendency, tested closer to the theory (pre_registration_v8.md, stage 4).

4.1 r at normal capacity utilisation (US NFC: FRB CUMFNS; EU: Eurostat BS-ICU-PC)
4.2 phases between pre-named crises; devaluation in crises
4.3 cyclicality of e (profit squeeze): d ln e on lagged unemployment
4.4 external criterion: which profit-rate measure forecasts accumulation and recessions out of sample (US)
4.5 taxes and the social wage in e (US NFC)
4.6 one consistent capital measure: Eurostat whole-economy N11N / N11G (EU)
Outputs: results/v8/s4_*.csv
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from ..v6 import series as S6
from ..v6.national import es, z1
from ..v6.series import MAIN8, fa, nipa
from ..v6.tests import contrib, hp_gap, logs, trend
from .stage2 import OUT, R8

US_PEAKS = [1953, 1957, 1960, 1969, 1973, 1980, 1981, 1990, 2001, 2007, 2019]
EU_CRISES = [1974, 1975, 1980, 1981, 1982, 1991, 1992, 1993, 2001, 2002, 2008, 2009, 2020]
EU_ISO2 = {"DEU": "DE", "FRA": "FR", "ITA": "IT", "NLD": "NL", "SWE": "SE"}


def wilson(k, n, z=1.645):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return c - h, c + h


def label_share(k, n, theta=0.5, delta=0.15, up=True):
    lo, hi = wilson(k, n)
    p = k / n if n else np.nan
    if up and lo > theta and p > theta + delta or (not up) and hi < theta and p < theta - delta:
        return "подтверждено"
    if (lo >= theta - delta and hi <= theta + delta) or (up and hi < theta) or ((not up) and lo > theta):
        return "опровергнуто"
    return "неинформативно"


def series_all():
    return pd.read_csv(S6.OUT / "series.csv")


def util_us():
    u = pd.read_csv(R8 / "fred_CUMFNS.csv")
    u.columns = ["date", "u"]
    u["year"] = pd.to_datetime(u.date).dt.year
    return u.groupby("year").u.mean()


def util_eu():
    d = es(R8 / "eurostat_ei_bsin_q_r2.json")
    d["year"] = d.time.str[:4].astype(int)
    return d.groupby(["geo", "year"]).value.mean()


def normal(u, kind):
    if kind == "mean":
        return pd.Series(u.mean(), index=u.index)
    if kind == "hp":
        return u - hp_gap(u.to_numpy(), 100)
    return pd.Series(80.0, index=u.index)


# ---------------------------------------------------------------- 4.1
def r_normal():
    s = series_all()
    rows = []
    us = s[(s.geo == "USA_NFC") & (s.variant == "main")].set_index("year").sort_index()
    u = util_us().reindex(us.index).dropna()
    eu = util_eu()
    targets = [("USA_NFC", us.loc[u.index], u, [1951, 1956], [2024, 2019])]
    for g3, g2 in EU_ISO2.items():
        g = s[(s.geo == g3) & (s.variant == "main")].set_index("year").sort_index()
        if g2 in eu.index.get_level_values(0):
            ue = eu.loc[g2]
            ue = ue[ue.index.isin(g.index)]
            if len(ue) >= 20:
                targets.append((g3, g.loc[ue.index], ue, [int(ue.index.min()), int(ue.index.min()) + 5], [2024, 2019]))
    for geo, g, u, starts, ends in targets:
        for kind in ("mean", "hp", "80"):
            un = normal(u, kind)
            rn = g.r / (u / un)
            for name, y in (("r", g.r), (f"r_n ({kind})", rn)):
                if name == "r" and kind != "mean":
                    continue
                for a in starts:
                    for b in ends:
                        yy = y.loc[a:b].dropna()
                        if len(yy) < 15:
                            continue
                        for lag in (2, 4, 8):
                            sl, se, p = trend(yy.to_numpy(), lag)
                            rows.append(dict(geo=geo, series=name, y0=a, y1=b, lag=lag, slope=sl, p=p, n=len(yy)))
    t = pd.DataFrame(rows)
    rob = t.groupby(["geo", "series"]).apply(lambda x: pd.Series(dict(
        robust_neg=bool(((x.slope < 0) & (x.p < 0.05)).all()), robust_pos=bool(((x.slope > 0) & (x.p < 0.05)).all()),
        share_neg=((x.slope < 0) & (x.p < 0.05)).mean(), share_pos=((x.slope > 0) & (x.p < 0.05)).mean(),
        slope_base=x.slope.iloc[0], years=f"{x.y0.min()}-{x.y1.max()}")), include_groups=False).reset_index()
    return t, rob


# ---------------------------------------------------------------- 4.2
def phases_from(years, crises, last):
    out, start = [], None
    crisis = sorted(set(crises))
    y0 = min(years)
    bounds = [y0 - 1] + crisis + [last + 1]
    for a, b in zip(bounds[:-1], bounds[1:]):
        s, e = a + 1, b - 1
        if e - s + 1 >= 3:
            out.append((s, e))
    return out


def phases():
    s = series_all()
    rows = []
    sets = [("USA_NFC", US_PEAKS)] + [(g, EU_CRISES) for g in MAIN8]
    for geo, cr in sets:
        g = s[(s.geo == geo) & (s.variant == "main")].set_index("year").sort_index()
        if geo == "USA_NFC":
            g = g.loc[1951:]
            # US: phase = year after a peak .. the next peak
            bounds = [1950] + US_PEAKS + [2025]
            ph = [(a + 1, b) for a, b in zip(bounds[:-1], bounds[1:]) if b - a >= 3]
        else:
            ph = phases_from(list(g.index), cr, int(g.index.max()))
        lg = logs(g)
        for a, b in ph:
            x = g.loc[a:b]
            if len(x) < 3:
                continue
            a, b = int(x.index.min()), int(x.index.max())
            sl_rm = np.polyfit(x.index, x.rM, 1)[0]
            dk = np.log1p(x.k.iloc[-1]) - np.log1p(x.k.iloc[0])
            c = contrib(lg, a, b)
            rows.append(dict(geo=geo, start=a, end=b, years=b - a + 1, slope_rM=sl_rm, rM_falls=bool(sl_rm < 0),
                             dln1k=dk, k_rises=bool(dk > 0), dlnrM=c.get("dlnrM"), c_e=c.get("c_e"), c_1k=c.get("c_1k")))
    P = pd.DataFrame(rows)
    out = []
    for name, col in (("share of phases with falling r_M", "rM_falls"), ("share of phases with rising k", "k_rises")):
        k, n = int(P[col].sum()), len(P)
        lo, hi = wilson(k, n)
        out.append(dict(item=name, k=k, n=n, share=k / n, ci90_lo=lo, ci90_hi=hi, label=label_share(k, n, up=True)))
    return P, pd.DataFrame(out)


def devaluation():
    """Revaluation and other changes of the net stock, (dK - (I - D)) / K_{t-1}: crisis vs other years."""
    rows = []
    K = fa(4, "FAAt401-A", 37)
    I = fa(4, "FAAt407-A", 37)
    D = fa(4, "FAAt404-A", 37)
    us = pd.DataFrame({"K": K, "I": I, "D": D}).loc[1951:2024]
    us["reval"] = (us.K - us.K.shift() - (us.I - us.D)) / us.K.shift()
    crisis_us = {y + 1 for y in US_PEAKS} | set(US_PEAKS)
    s = series_all()
    data = [("USA_NFC", us.reval, crisis_us)]
    for geo in MAIN8:
        g = s[(s.geo == geo) & (s.variant == "main")].set_index("year").sort_index()
        rv = (g.K - g.K.shift() - (g.UIGT - g.UKCT)) / g.K.shift()
        data.append((geo, rv, set(EU_CRISES)))
    for geo, rv, cr in data:
        rv = rv.dropna()
        c = rv[rv.index.isin(cr)]
        o = rv[~rv.index.isin(cr)]
        t = stats.ttest_ind(c, o, equal_var=False)
        rows.append(dict(geo=geo, reval_crisis=c.mean(), reval_other=o.mean(), diff=c.mean() - o.mean(), p=t.pvalue,
                         n_crisis=len(c), n_other=len(o)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 4.3
def profit_squeeze():
    S6.CODES = list(dict.fromkeys(S6.CODES + ["ZUTN"]))
    raw = S6.ameco_raw()
    s = series_all()
    rows = []
    d = s[(s.variant == "main") & s.geo.isin(MAIN8)][["geo", "year", "e"]].copy()
    u = raw[["geo", "year", "ZUTN"]].copy()
    dw = u[u.geo == "D_W"].set_index("year").ZUTN
    de = u[u.geo == "DEU"].set_index("year").ZUTN
    deu = de.combine_first(dw[dw.index <= 1990]).rename("ZUTN").reset_index().assign(geo="DEU")
    u = pd.concat([u[u.geo != "DEU"], deu])
    d = d.merge(u, on=["geo", "year"], how="left").sort_values(["geo", "year"])
    eu = util_eu().rename("cu").reset_index()
    eu["geo"] = eu.geo.map({v: k for k, v in EU_ISO2.items()})
    d = d.merge(eu.dropna(), on=["geo", "year"], how="left")
    g = d.groupby("geo")
    d["dlne"] = g.e.transform(lambda x: np.log(x).diff())
    d["u_lag"] = g.ZUTN.shift()
    d["dcu"] = g.cu.diff()
    for name, cols in (("d ln e on u(t-1)", ["u_lag"]), ("d ln e on u(t-1) and d capacity utilisation", ["u_lag", "dcu"])):
        x = d.dropna(subset=["dlne"] + cols)
        X = x[["dlne"] + cols] - x.groupby("geo")[["dlne"] + cols].transform("mean")
        f = sm.OLS(X.dlne, X[cols]).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(x.geo)[0]})
        b, se = f.params.u_lag, f.bse.u_lag
        lab = "подтверждено" if (b - 1.645 * se > 0 and b > 0.005) else (
            "опровергнуто" if (b - 1.645 * se >= -0.005 and b + 1.645 * se <= 0.005) or b + 1.645 * se < 0 else "неинформативно")
        rows.append(dict(item=name, coef_u=b, se=se, ci90_lo=b - 1.645 * se, ci90_hi=b + 1.645 * se, p=f.pvalues.u_lag,
                         n=len(x), countries=x.geo.nunique(), label=lab,
                         coef_dcu=f.params.get("dcu", np.nan), p_dcu=f.pvalues.get("dcu", np.nan)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 4.4
def measures_us():
    s = series_all()
    us = s[(s.geo == "USA_NFC") & (s.variant == "main")].set_index("year").sort_index()
    u = util_us()
    un = normal(u, "hp")
    m = pd.DataFrame({"r": us.r, "rM": us.rM})
    m["r_n"] = us.r / (u / un).reindex(us.index)
    zt, sh = z1()
    # all-assets return: NOS + net interest + dividends + reinvested earnings over K + financial assets
    import zipfile
    from ..v6.series import RAW
    z = zipfile.ZipFile(RAW / "v6" / "z1" / "z1_csv_files.zip")
    ia = pd.read_csv(z.open("csv/S11_1_i_a.csv")).set_index("date")
    bs = pd.read_csv(z.open("csv/S11_1_b.csv")).set_index("date")
    num = lambda c: pd.to_numeric(c.replace("ND", np.nan), errors="coerce")  # noqa: E731
    ia.index = ia.index.astype(int)
    q4 = bs[bs.index.str.endswith("Q4")]
    q4.index = q4.index.str[:4].astype(int)
    K = fa(4, "FAAt401-A", 37)
    fin = num(ia["FU106402101.A"]) + num(ia["FU106130101.A"]) - num(ia["FU106130001.A"]) + \
        num(ia["FU106121101.A"]) + num(ia["FU103092201.A"]).fillna(0)
    m["all_assets"] = (fin / (K + num(q4["FL104090005.Q"]))).reindex(m.index)
    # target (a): real net stock growth (BEA FA 4.2 line 37 quantity index); (b) NBER recession indicator
    q = fa(4, "FAAt402-A", 37)
    rec = pd.read_csv(R8 / "fred_USREC.csv")
    rec.columns = ["date", "rec"]
    rec = rec.assign(year=pd.to_datetime(rec.date).dt.year).groupby("year").rec.max()
    return m, np.log(q), rec


def oos_measure(m, col, lnq, rec, start=1980):
    rows = []
    y = m[col].dropna()
    for h in (1, 3, 5):
        tgt = (lnq.shift(-h) - lnq).rename("t")
        own = (lnq - lnq.shift(1)).rename("own")
        e = pd.concat([tgt, own, y.rename("x0"), y.shift(1).rename("x1")], axis=1).dropna()
        fr, fu, act = [], [], []
        for t in e.index[e.index >= start]:
            tr = e[e.index <= t - h]
            if len(tr) < 15:
                continue
            Xr = sm.add_constant(tr[["own"]])
            Xu = sm.add_constant(tr[["own", "x0", "x1"]])
            br = np.linalg.lstsq(Xr, tr.t, rcond=None)[0]
            bu = np.linalg.lstsq(Xu, tr.t, rcond=None)[0]
            fr.append(float(np.r_[1, e.loc[t, ["own"]]] @ br))
            fu.append(float(np.r_[1, e.loc[t, ["own", "x0", "x1"]]] @ bu))
            act.append(float(e.loc[t, "t"]))
        fr, fu, act = map(np.array, (fr, fu, act))
        er, eu_ = act - fr, act - fu
        fadj = er ** 2 - (eu_ ** 2 - (fr - fu) ** 2)
        se = np.sqrt(sm.OLS(fadj, np.ones_like(fadj)).fit(cov_type="HAC", cov_kwds={"maxlags": max(h, 1)}).cov_params()[0, 0])
        cw = fadj.mean() / se
        rows.append(dict(measure=col, target="accumulation (real net stock growth)", h=h, n=len(act),
                         rmse_ratio=float(np.sqrt((eu_ ** 2).mean() / (er ** 2).mean())), cw_p=float(stats.norm.sf(cw)),
                         better=bool(stats.norm.sf(cw) < 0.05)))
    for h in (1, 2, 3):
        tgt = pd.concat([rec.shift(-k) for k in range(1, h + 1)], axis=1).max(axis=1).rename("t")
        e = pd.concat([tgt, y.rename("x0"), y.shift(1).rename("x1")], axis=1).dropna()
        pb, pu, act = [], [], []
        for t in e.index[e.index >= start]:
            tr = e[e.index <= t - h]
            if len(tr) < 15 or tr.t.nunique() < 2:
                continue
            base = tr.t.mean()
            try:
                f = sm.Logit(tr.t, sm.add_constant(tr[["x0", "x1"]])).fit(disp=0)
                pr = float(f.predict(np.r_[1, e.loc[t, ["x0", "x1"]]].reshape(1, -1))[0])
            except Exception:  # noqa: BLE001
                pr = base
            pb.append(base)
            pu.append(pr)
            act.append(float(e.loc[t, "t"]))
        pb, pu, act = map(np.array, (pb, pu, act))
        brier_b, brier_u = np.mean((pb - act) ** 2), np.mean((pu - act) ** 2)
        eps = 1e-6
        ls_b = -np.mean(act * np.log(pb + eps) + (1 - act) * np.log(1 - pb + eps))
        ls_u = -np.mean(act * np.log(pu + eps) + (1 - act) * np.log(1 - pu + eps))
        rows.append(dict(measure=col, target="recession within h years", h=h, n=len(act), brier_ratio=brier_u / brier_b,
                         logscore_model=ls_u, logscore_base=ls_b, better=bool(brier_u <= 0.95 * brier_b)))
    return rows


def external():
    m, lnq, rec = measures_us()
    rows = []
    for c in ("r", "r_n", "rM", "all_assets"):
        rows += oos_measure(m, c, lnq, rec)
    return pd.DataFrame(rows), m


# ---------------------------------------------------------------- 4.5
def taxes_social():
    n = nipa(["W326RC", "A460RC", "W325RC", "A065RC", "W055RC", "A061RC", "A063RC"])
    d = n.loc[1951:2024]
    e1 = d.W326RC / d.A460RC
    e2 = (d.W326RC + d.W325RC) / d.A460RC
    wnet = d.A460RC * (1 - (d.W055RC + d.A061RC) / d.A065RC + d.A063RC / d.A065RC)
    e3 = (d.W326RC + d.W325RC) / wnet
    rows, lv = [], pd.DataFrame({"e1": e1, "e2_indirect_taxes": e2, "e3_social_wage": e3})
    for c in lv:
        for a in (1951, 1956):
            for b in (2024, 2019):
                for lag in (2, 4, 8):
                    sl, se, p = trend(lv[c].loc[a:b].to_numpy(), lag)
                    rows.append(dict(series=c, y0=a, y1=b, lag=lag, slope=sl, p=p))
    return pd.DataFrame(rows), lv


# ---------------------------------------------------------------- 4.6
def eurostat_whole():
    it = {k: es(R8 / f"a64_{k}.json") for k in ("B1G", "D1", "P51C", "D29X39")}
    fr = [d[(d.nace_r2 == "TOTAL")][["geo", "time", "value"]].rename(columns={"value": k}).set_index(["geo", "time"])
          for k, d in it.items()]
    a = pd.concat(fr, axis=1).reset_index()
    st = es(R8 / "eurostat_nama_10_nfa_st.json")
    st = st[st.nace_r2 == "TOTAL"].pivot_table(index=["geo", "time"], columns="asset10", values="value").reset_index()
    a = a.merge(st, on=["geo", "time"])
    a["year"] = a.time.astype(int)
    a["PI"] = a.B1G - a.D1 - a.D29X39 - a.P51C
    rows = []
    for g2 in ("DE", "FR", "IT", "NL", "SE", "AT", "BE", "DK", "FI", "ES"):
        x = a[a.geo == g2].set_index("year").sort_index()
        for kname in ("N11N", "N11G"):
            r = (x.PI / x[kname]).dropna()
            if len(r) < 20:
                continue
            for lag in (2, 4, 8):
                sl, se, p = trend(r.to_numpy(), lag)
                rows.append(dict(geo=g2, K=kname, y0=int(r.index.min()), y1=int(r.index.max()), lag=lag, slope=sl, p=p,
                                 first=r.iloc[0], last=r.iloc[-1]))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    t, rob = r_normal()
    t.to_csv(OUT / "s4_rn_trends.csv", index=False)
    rob.to_csv(OUT / "s4_rn_robust.csv", index=False)
    print(rob.round(4).to_string(), flush=True)
    P, ps = phases()
    P.to_csv(OUT / "s4_phases.csv", index=False)
    ps.to_csv(OUT / "s4_phases_summary.csv", index=False)
    print(ps.round(3).to_string(), flush=True)
    dv = devaluation()
    dv.to_csv(OUT / "s4_devaluation.csv", index=False)
    print(dv.round(4).to_string(), flush=True)
    sq = profit_squeeze()
    sq.to_csv(OUT / "s4_squeeze.csv", index=False)
    print(sq.round(4).to_string(), flush=True)
    ex, m = external()
    ex.to_csv(OUT / "s4_external.csv", index=False)
    m.to_csv(OUT / "s4_measures_us.csv")
    print(ex.round(3).to_string(), flush=True)
    ts, lv = taxes_social()
    ts.to_csv(OUT / "s4_taxes_trends.csv", index=False)
    lv.to_csv(OUT / "s4_taxes_levels.csv")
    print(ts.groupby("series").apply(lambda x: pd.Series(dict(neg=((x.slope < 0) & (x.p < .05)).mean(),
                                                               pos=((x.slope > 0) & (x.p < .05)).mean())), include_groups=False))
    print(lv.loc[[1951, 1980, 2000, 2024]].round(3).to_string(), flush=True)
    ew = eurostat_whole()
    ew.to_csv(OUT / "s4_eurostat_whole.csv", index=False)
    print(ew[ew.lag == 4].round(4).to_string())
