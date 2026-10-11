"""Iteration 12b, stage 2: the world rate of profit (pre_registration_v12b.md, stage 2; outcomes 97-99; journal 3).

PWT 11.0: profit_c = (1 - labsh) Y - delta K; world r = sum profit / sum K.
  exchange-rate weights (main): Y = cgdpo x pl_gdpo, K = cn x pl_n;  PPP: Y = cgdpo, K = cn.
  main sample: balanced panel 1960-2019; variants: all countries available each year; GDP-weighted average of national
  rates (Maito-style weighting).
Shift-share (capital weights): dr_w = sum w_{c,t-1} dr_c (within) + sum dw_c r_{c,t} (weights); cumulative indices from
1960 divided by r_w(1960) (journal 3).
EPWT 7.0 (source of the country series of Basu et al.): same construction with PPP 2017 values.

Usage: PYTHONPATH=src python -P -m lts.v12b.stage2
Outputs: results/v12b/s2_*.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..v6.series import ROOT
from .stage1 import trend

OUT = ROOT / "results" / "v12b"
Y0, Y1 = 1960, 2019


def pwt():
    p = pd.read_excel(ROOT / "data" / "raw" / "v9" / "pwt110.xlsx", sheet_name="Data",
                      usecols=["countrycode", "year", "cgdpo", "cn", "labsh", "delta", "pl_gdpo", "pl_n"])
    p = p.rename(columns={"countrycode": "c"})
    p["Y_xr"], p["K_xr"] = p.cgdpo * p.pl_gdpo, p.cn * p.pl_n
    p["Y_ppp"], p["K_ppp"] = p.cgdpo, p.cn
    for w in ("xr", "ppp"):
        p[f"P_{w}"] = (1 - p.labsh) * p[f"Y_{w}"] - p.delta * p[f"K_{w}"]
    return p


def epwt():
    e = pd.read_excel(ROOT / "data" / "raw" / "v5" / "epwt70.xlsx", sheet_name="EPWT7.0")
    e = e.rename(columns={"Countrycode": "c", "Year": "year"})
    e["Y_ppp"], e["K_ppp"] = e.XGDPppp2017, e.Kppp2017
    e["P_ppp"] = (1 - e.LabShare) * e.Y_ppp - e.delta * e.K_ppp
    # journal 4 (after the result): current national prices converted at PWT 11 exchange rates
    xr = pd.read_excel(ROOT / "data" / "raw" / "v9" / "pwt110.xlsx", sheet_name="Data", usecols=["countrycode", "year", "xr"])
    e = e.merge(xr.rename(columns={"countrycode": "c"}), on=["c", "year"], how="left")
    e["Y_cur"], e["K_cur"] = e.XGDPnatcur / e.xr, e.Knatcur / e.xr
    e["P_cur"] = (1 - e.LabShare) * e.Y_cur - e.delta * e.K_cur
    return e


def balanced(d, w, y0=Y0, y1=Y1):
    cols = [f"Y_{w}", f"K_{w}", f"P_{w}"]
    s = d[d.year.between(y0, y1)].dropna(subset=cols)
    s = s[(s[f"K_{w}"] > 0)]
    n = s.groupby("c").year.nunique()
    keep = n[n == (y1 - y0 + 1)].index
    return s[s.c.isin(keep)]


def world(d, w):
    g = d.groupby("year")
    r = g[f"P_{w}"].sum() / g[f"K_{w}"].sum()
    d = d.assign(r_c=d[f"P_{w}"] / d[f"K_{w}"])
    gw = d.assign(wy=d[f"Y_{w}"] / d.groupby("year")[f"Y_{w}"].transform("sum"))
    r_gdpw = (gw.wy * gw.r_c).groupby(gw.year).sum()
    return r, r_gdpw, d


def shift_share(d, w):
    d = d.copy()
    d["r_c"] = d[f"P_{w}"] / d[f"K_{w}"]
    d["wk"] = d[f"K_{w}"] / d.groupby("year")[f"K_{w}"].transform("sum")
    P = d.pivot(index="year", columns="c", values="r_c")
    Wk = d.pivot(index="year", columns="c", values="wk")
    within = (Wk.shift(1) * P.diff()).sum(1, min_count=1)
    between = (Wk.diff() * P).sum(1, min_count=1)
    rw = (Wk * P).sum(1)
    base = float(rw.iloc[0])
    out = pd.DataFrame({"r_w": rw, "within": within.fillna(0), "between": between.fillna(0)})
    out["cum_within_rel"] = out.within.cumsum() / base
    out["cum_between_rel"] = out.between.cumsum() / base
    out["check"] = (base + out.within.cumsum() + out.between.cumsum()) / out.r_w - 1
    return out


def run():
    p = pwt()
    out, desc, series = [], [], {}
    b = balanced(p, "xr")
    r, rg, _ = world(b, "xr")
    series["PWT xr balanced"] = r
    series["PWT xr balanced, GDP-weighted mean (Maito-style)"] = rg
    o97 = trend(np.log(r), 4, 0.002, "<")
    out.append(dict(outcome=f"97: trend ln world r (PWT 11, exchange-rate weights, balanced {Y0}-{Y1}, "
                            f"{b.c.nunique()} countries)", **o97))
    ss = shift_share(b, "xr")
    ss.to_csv(OUT / "s2_shift_share_xr.csv")
    out.append(dict(outcome="98: trend of cumulative within-country index (/ r_w 1960)",
                    **trend(ss.cum_within_rel, 4, 0.002, "<")))
    out.append(dict(outcome="99: trend of cumulative country-weights index (/ r_w 1960)",
                    **trend(ss.cum_between_rel, 4, 0.002, ">")))
    # variants
    bp = balanced(p, "ppp")
    rp, rpg, _ = world(bp, "ppp")
    series["PWT ppp balanced"] = rp
    ssp = shift_share(bp, "ppp")
    ssp.to_csv(OUT / "s2_shift_share_ppp.csv")
    for w in ("xr", "ppp"):
        u = p[p.year.between(1950, 2023)].dropna(subset=[f"Y_{w}", f"K_{w}", f"P_{w}"])
        u = u[u[f"K_{w}"] > 0]
        ru, _, _ = world(u, w)
        series[f"PWT {w} unbalanced"] = ru
        series[f"number of countries, unbalanced {w}"] = u.groupby("year").c.nunique()
    e = epwt()
    be = balanced(e, "ppp")
    re_, reg_, _ = world(be, "ppp")
    series["EPWT 7.0 ppp2017 balanced"] = re_
    sse = shift_share(be, "ppp")
    sse.to_csv(OUT / "s2_shift_share_epwt.csv")
    for nm, s in series.items():
        if nm.startswith("number"):
            continue
        s = s.loc[Y0:Y1] if "unbalanced" not in nm else s
        desc.append(dict(item=f"trend ln {nm}", **trend(np.log(s.dropna()), 4, 0.002, "<")))
    bc = balanced(e, "cur")
    rc, _, _ = world(bc, "cur")
    series["EPWT 7.0 current prices, PWT xr, balanced"] = rc
    ssc = shift_share(bc, "cur")
    ssc.to_csv(OUT / "s2_shift_share_epwt_current.csv")
    desc.append(dict(item=f"trend ln EPWT 7.0 current prices, PWT xr, balanced ({bc.c.nunique()} countries)",
                     **trend(np.log(rc.loc[Y0:Y1].dropna()), 4, 0.002, "<")))
    for a, b_ in ((1960, 1980), (1980, 2019)):
        desc.append(dict(item=f"trend ln world r, EPWT 7.0 current prices, {a}-{b_}", **trend(np.log(rc.loc[a:b_].dropna()), 4, 0.002, "<")))
    for nm, ssx in (("PWT ppp", ssp), ("EPWT 7.0", sse), ("EPWT 7.0 current", ssc)):
        desc.append(dict(item=f"{nm}: trend cumulative within (/ r_w 1960)", **trend(ssx.cum_within_rel, 4, 0.002, "<")))
        desc.append(dict(item=f"{nm}: trend cumulative weights (/ r_w 1960)", **trend(ssx.cum_between_rel, 4, 0.002, ">")))
    for a, b_ in ((1960, 1980), (1980, 2019)):
        desc.append(dict(item=f"trend ln world r, PWT xr balanced, {a}-{b_}", **trend(np.log(r.loc[a:b_]), 4, 0.002, "<")))
        desc.append(dict(item=f"trend ln world r, EPWT 7.0 balanced, {a}-{b_}", **trend(np.log(re_.loc[a:b_].dropna()), 4, 0.002, "<")))
    # countries in the balanced panels; China's weight
    info = dict(pwt_xr_countries=",".join(sorted(b.c.unique())), epwt_countries=",".join(sorted(be.c.unique())))
    wchn = b[b.c == "CHN"].set_index("year").K_xr / b.groupby("year").K_xr.sum()
    S = pd.DataFrame(series)
    S["China K share (xr balanced)"] = wchn
    S.to_csv(OUT / "s2_world_series.csv")
    O = pd.DataFrame(out)
    O.to_csv(OUT / "s2_outcomes.csv", index=False)
    D = pd.DataFrame(desc)
    D.to_csv(OUT / "s2_descriptive.csv", index=False)
    pd.Series(info).to_csv(OUT / "s2_samples.csv")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 90)
    print(O.round(5).to_string())
    print(D.round(5).to_string())
    print(S.loc[[1960, 1970, 1980, 1990, 2000, 2010, 2019]].round(4).to_string())
    print(ss.loc[[1961, 1980, 2000, 2019]].round(4).to_string())
    print(info)


if __name__ == "__main__":
    run()
