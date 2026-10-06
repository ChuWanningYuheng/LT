"""Iteration 9, stage 5: unequal exchange (pre_registration_v9.md, stage 5).

5.1 replication of Hickel et al. (2024) on EXIOBASE 3.8.2 pxp (2017, 2021): embodied hours from producing region r in
    final demand of region s, H_rs = sum_{k in r} h_k [ (I - A)^-1 y_s ]_k; North = IMF advanced economies in EXIOBASE.
5.2 decomposition (2017): productivity (physical-productivity weights from PLD 2023, 2017 benchmark), wages at equal
    productivity (Emmanuel), profit-rate differences.
5.3 b between countries within industry vs between industries within country (WIOD SEA).
Outputs: results/v9/s5_*.csv
"""
from __future__ import annotations

import sys
import zipfile

import numpy as np
import pandas as pd
import scipy.linalg as sla

from ..v8.rule import decide
from .stage1 import OUT

R9 = OUT.parents[1] / "data" / "raw" / "v9"
NORTH = ["AT", "BE", "CY", "CZ", "DE", "DK", "EE", "ES", "FI", "FR", "GR", "IE", "IT", "LT", "LU", "LV", "MT", "NL", "PT",
         "SE", "SI", "SK", "GB", "US", "JP", "CA", "KR", "AU", "TW", "NO", "CH"]
NORTH_NARROW = [c for c in NORTH if c not in ("CZ", "SK", "SI", "EE", "LT", "LV")]
ISO2_3 = {"AT": "AUT", "BE": "BEL", "BG": "BGR", "CY": "CYP", "CZ": "CZE", "DE": "DEU", "DK": "DNK", "EE": "EST",
          "ES": "ESP", "FI": "FIN", "FR": "FRA", "GR": "GRC", "HR": "HRV", "HU": "HUN", "IE": "IRL", "IT": "ITA",
          "LT": "LTU", "LU": "LUX", "LV": "LVA", "MT": "MLT", "NL": "NLD", "PL": "POL", "PT": "PRT", "RO": "ROU",
          "SE": "SWE", "SI": "SVN", "SK": "SVK", "GB": "GBR", "US": "USA", "JP": "JPN", "CN": "CHN", "CA": "CAN",
          "KR": "KOR", "BR": "BRA", "IN": "IND", "MX": "MEX", "RU": "RUS", "AU": "AUS", "CH": "CHE", "TR": "TUR",
          "TW": "TWN", "NO": "NOR", "ID": "IDN", "ZA": "ZAF"}


def load(year):
    z = zipfile.ZipFile(R9 / f"exiobase_IOT_{year}_pxp.zip")
    p = f"IOT_{year}_pxp/"
    A = pd.read_csv(z.open(p + "A.txt"), sep="\t", header=[0, 1], index_col=[0, 1])
    idx = A.index
    A = A.to_numpy(np.float64)
    Y = pd.read_csv(z.open(p + "Y.txt"), sep="\t", header=[0, 1], index_col=[0, 1])
    x = pd.read_csv(z.open(p + "x.txt"), sep="\t", index_col=[0, 1]).iloc[:, 0].to_numpy()
    F = pd.read_csv(z.open(p + "satellite/F.txt"), sep="\t", header=[0, 1], index_col=0)
    return A, idx, Y, x, F


def satellite(F):
    rows = F.index.astype(str)
    hours = F[rows.str.startswith("Employment hours:") & ~rows.str.contains("Vulnerable")]
    skill = {}
    for lev in ("Low-skilled", "Medium-skilled", "High-skilled"):
        skill[lev] = F[rows.str.startswith("Employment hours: " + lev)].sum(0).to_numpy()
    comp = {lev: F[rows == f"Compensation of employees; wages, salaries, & employers' social contributions: {lev}"].sum(0).to_numpy()
            for lev in ("Low-skilled", "Medium-skilled", "High-skilled")}
    nos = F[rows == "Operating surplus: Remaining net operating surplus"].sum(0).to_numpy()
    cfc = F[rows == "Operating surplus: Consumption of fixed capital"].sum(0).to_numpy()
    return hours.sum(0).to_numpy(), skill, comp, nos, cfc


def flows(A, idx, Y, x, F, north=NORTH):
    regions = list(dict.fromkeys(idx.get_level_values(0)))
    reg = idx.get_level_values(0).to_numpy()
    cats = Y.columns.get_level_values(1)
    Yr = Y.loc[:, ~cats.str.startswith("Exports")].T.groupby(level=0).sum().T.reindex(columns=regions).to_numpy()
    n = A.shape[0]
    lu = sla.lu_factor(np.eye(n) - A)
    X = sla.lu_solve(lu, Yr)                                          # output needed for each region's final demand
    hrs, skill, comp, nos, cfc = satellite(F)
    h = np.divide(hrs, x, out=np.zeros(n), where=x > 0)              # M.hr per M.EUR
    isN = np.isin(reg, north)
    sN = np.isin(np.array(regions), north)
    H = np.array([[(h * X[:, s])[reg == r].sum() for s in range(len(regions))] for r in regions])   # producer x consumer
    imp = H[np.ix_(~sN, sN)].sum()                                     # South-produced hours in North final demand
    exp_ = H[np.ix_(sN, ~sN)].sum()
    by_skill = {}
    for lev, v in skill.items():
        hs = np.divide(v, x, out=np.zeros(n), where=x > 0)
        Hs = np.array([[(hs * X[:, s])[reg == r].sum() for s in range(len(regions))] for r in regions])
        by_skill[lev] = Hs[np.ix_(~sN, sN)].sum() - Hs[np.ix_(sN, ~sN)].sum()
    # northern wage per hour by skill (all northern production)
    wN = {lev: comp[lev][isN].sum() / skill[lev][isN].sum() for lev in skill}
    wage_value = sum(by_skill[lev] * wN[lev] for lev in skill)        # M.hr * M.EUR / M.hr = M.EUR
    return dict(total_hours=float(hrs.sum()), north_imports=float(imp), north_exports=float(exp_),
                net=float(imp - exp_), net_low=by_skill["Low-skilled"], net_med=by_skill["Medium-skilled"],
                net_high=by_skill["High-skilled"], wage_value_MEUR=float(wage_value), X=X, h=h, reg=reg,
                regions=regions, isN=isN, sN=sN, comp=sum(comp.values()), nos=nos, cfc=cfc, hrs=hrs, x=x, idx=idx)


# ---------------------------------------------------------------- 5.2 decomposition
def product_sector(idx):
    """Products -> PLD 2023 sector via Hickel et al. (2024) sector allocation (MOESM4)."""
    s = pd.read_excel(R9 / "hickel2024_MOESM4.xlsx", header=None)
    hdr = s.iloc[3].tolist()
    m = {}
    for j, grp in enumerate(hdr):
        if not isinstance(grp, str):
            continue
        for v in s.iloc[4:, j].dropna():
            m[str(v).strip()] = grp.strip()
    out = []
    for p in idx.get_level_values(1):
        g = m.get(str(p).strip(), "Services")
        if g == "Agriculture":
            out.append("agr")
        elif g == "Mining":
            out.append("min")
        elif g == "Manufacturing":
            out.append("man")
        elif g == "Other":
            pl = str(p).lower()
            out.append("con" if "construction" in pl else "tra" if "transport" in pl else "pu")
        else:
            out.append("svc")
    return np.array(out)


def pld_levels(year=2017):
    d = pd.read_excel(R9 / "pld2023_dataset.xlsx", sheet_name="Data")
    d = d[d.year == year].copy()
    d["pl"] = d.PPP_y / d.xr
    out = {(r.countrycode, r.sector): r.pl for r in d.itertuples() if np.isfinite(r.pl)}
    svc = d[d.sector.isin(["trd", "bus", "fin", "pub", "oth", "dwe"])]
    for c, g in svc.groupby("countrycode"):
        g = g[np.isfinite(g.pl)]
        if len(g):
            out[(c, "svc")] = float(np.average(g.pl, weights=g.VA.clip(lower=0) + 1e-9))
    return out


def decompose(f, year=2017, pl_year=2017, adjust_services=True):
    sec = product_sector(f["idx"])
    pls = pld_levels(pl_year)
    reg = f["reg"]
    x, hrs = f["x"], f["hrs"]
    pl = np.array([pls.get((ISO2_3.get(r, ""), s), np.nan) for r, s in zip(reg, sec)])
    if not adjust_services:
        pl[sec == "svc"] = np.nan
    q = x / pl
    ok = np.isfinite(q) & (hrs > 0) & (q > 0)
    prod = np.array(f["idx"].get_level_values(1))
    df = pd.DataFrame({"p": prod, "q": np.where(ok, q, 0), "H": np.where(ok, hrs, 0)})
    pib = (df.groupby("p").q.transform("sum") / df.groupby("p").H.transform("sum")).to_numpy()
    w = np.where(ok, (q / hrs) / pib, 1.0)
    w = np.where(np.isfinite(w), w, 1.0)
    hstar = f["h"] * w
    X, sN, regions = f["X"], f["sN"], f["regions"]

    def net(hv):
        Hm = np.array([[(hv * X[:, s])[reg == r].sum() for s in range(len(regions))] for r in regions])
        return Hm[np.ix_(~sN, sN)].sum() - Hm[np.ix_(sN, ~sN)].sum(), Hm
    Hn, _ = net(f["h"])
    Hs, _ = net(hstar)
    comp = f["comp"]
    isN = f["isN"]
    wN = comp[isN].sum() / (hrs * w)[isN].sum()                       # EUR per adjusted hour, North
    wS = comp[~isN].sum() / (hrs * w)[~isN].sum()
    va = comp + f["nos"] + f["cfc"]
    vrate = np.divide(va, x, out=np.zeros(len(x)), where=x > 0)
    nrate = np.divide(f["nos"], x, out=np.zeros(len(x)), where=x > 0)
    VSN = np.array([(vrate * X[:, s])[reg == r].sum() for r in regions for s in range(len(regions))]).reshape(len(regions), -1)
    NSN = np.array([(nrate * X[:, s])[reg == r].sum() for r in regions for s in range(len(regions))]).reshape(len(regions), -1)
    v_sn, v_ns = VSN[np.ix_(~sN, sN)].sum(), VSN[np.ix_(sN, ~sN)].sum()
    os_sn, os_ns = NSN[np.ix_(~sN, sN)].sum() / v_sn, NSN[np.ix_(sN, ~sN)].sum() / v_ns
    return dict(year=year, pl_year=pl_year, adjust_services=adjust_services, net_hours=Hn, net_adjusted_hours=Hs,
                a_productivity_hours=Hn - Hs, b_emmanuel_MEUR=Hs * (wN - wS), wN_per_adj_hour=wN, wS_per_adj_hour=wS,
                c_profit_MEUR=(os_sn - os_ns) * v_sn, os_share_SN=os_sn, os_share_NS=os_ns, va_SN_MEUR=v_sn,
                share_hours_weight1=float(1 - (ok * hrs).sum() / hrs.sum()))


# ---------------------------------------------------------------- 5.3 b between countries vs between industries
def b_between():
    from ..revisit.wiod import sea
    from .b_est import grouped_b, nls_b
    from ..v8.stage2 import bfit
    s = sea()
    s["year"] = s.year.astype(int)
    w = s.pivot_table(index=["country", "code", "year"], columns="variable", values="v").reset_index()
    w = w[~w.code.isin(["O84", "P85", "Q", "T", "U", "L68", "B", "D35", "K64", "K65", "K66"])]
    w = w[(w.CAP > 0) & (w.LAB > 0) & (w.K > 0)].copy()
    w["PI"], w["W"], w["ind"], w["geo"] = w.CAP, w.LAB, w.code, w.country
    rows = []
    for name, fe_col, cl in (("between industries (country x year)", w.country + w.year.astype(str), "code"),
                             ("between countries (industry x year)", w.code + w.year.astype(str), "country")):
        d = w.assign(cy=fe_col)
        d["kw"], d["y"] = d.K / d.W, d.PI / d.W
        d["ind"] = d[cl]                                              # clusters: the other dimension
        f = bfit(d, np.log(d.PI / d.W), np.log(d.kw), fe="cy")
        be, lost, ng = grouped_b(d)
        from .b_est import cluster_boot
        (lo, hi), sd = cluster_boot(d, lambda q: grouped_b(q)[0], B=499)
        rows.append(dict(spec=name, b_log=f["b"], b_log_lo=f["ci90_lo"], b_log_hi=f["ci90_hi"], b_grouped=be,
                         b_grouped_lo=lo, b_grouped_hi=hi, b_nls=nls_b(d.y.to_numpy(), d.kw.to_numpy(), d.cy.to_numpy()),
                         n=len(d)))
    R = pd.DataFrame(rows)
    a, b = R.iloc[1], R.iloc[0]
    est = a.b_grouped - b.b_grouped
    se = np.sqrt(((a.b_grouped_hi - a.b_grouped_lo) / 3.29) ** 2 + ((b.b_grouped_hi - b.b_grouped_lo) / 3.29) ** 2)
    out = dict(item="outcome 34: b between countries - b between industries (grouped)", est=est, ci90_lo=est - 1.645 * se,
               ci90_hi=est + 1.645 * se, label=decide(est, est - 1.645 * se, est + 1.645 * se, 0, 0.1, "<"))
    return R, out


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    part = sys.argv[1] if len(sys.argv) > 1 else "all"
    if part in ("replicate", "all"):
        rows, dec = [], []
        for y in (2021, 2017):
            A, idx, Y, x, F = load(y)
            for nm, north in (("IMF advanced", NORTH), ("narrow North", NORTH_NARROW)):
                f = flows(A, idx, Y, x, F, north)
                rows.append({k: v for k, v in f.items() if np.isscalar(v)} | dict(year=y, north=nm))
                print(y, nm, rows[-1], flush=True)
                if y == 2017 and nm == "IMF advanced":
                    dec.append(decompose(f))
                    dec.append(decompose(f, adjust_services=False))
                    dec.append(decompose(f, pl_year=2011))
                if y == 2017 and nm == "narrow North":
                    dec.append(decompose(f) | dict(north="narrow"))
            del A
            pd.DataFrame(rows).to_csv(OUT / "s5_hickel.csv", index=False)
            pd.DataFrame(dec).to_csv(OUT / "s5_decomposition.csv", index=False)
        print(pd.DataFrame(dec).round(3).to_string())
    if part in ("between", "all"):
        R, o = b_between()
        R.to_csv(OUT / "s5_b_between.csv", index=False)
        pd.DataFrame([o]).to_csv(OUT / "s5_b_between_outcome.csv", index=False)
        print(R.round(3).to_string(), "\n", o)
