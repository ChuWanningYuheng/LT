"""Revisit (pre_registration_revisit.md): WIOD 2016 national systems, 2000-2014.

R1  stage P of iteration 3 (placebos with the dispersion of labour) on WIOD: open system, capital included
    (CFC = delta_ct * K_j, delta from AMECO), imports at price; placebos (a) perm, (b) lnorm, (c) v2 mixtures;
    MAWD and d; flat vector comparison.
R2  B3.1 of iteration 4 on WIOD: r = (CAP - delta K)/K on ln(K/LAB), country x year FE, SE by industry.
Outputs: results/revisit/*.csv
"""
from __future__ import annotations

import glob
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ..core import Economy
from ..v2.placebo_cost import metrics_mat
from ..v3.placebo_disp import Draws, Setup, shares

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw" / "v6"
OUT = ROOT / "results" / "revisit"
YEARS = range(2000, 2015)
N = 56
PREREG_EXCL_WIOD = {"B", "K64", "K65", "K66", "L68", "J61", "J62_J63"}
V3_13 = ("USA", "DEU", "MEX", "FRA", "ITA", "ESP", "NLD", "AUT", "POL", "CZE", "KOR", "JPN", "GBR")


@lru_cache(None)
def wiot(year):
    import pyreadr
    f = glob.glob(str(RAW / "wiod" / "wiot" / f"WIOT{year}_*.RData"))[0]
    r = pyreadr.read_r(f)
    d = r[list(r)[0]]
    body = d.iloc[:44 * N]
    cols = list(d.columns)
    countries = list(dict.fromkeys(body.Country))
    codes = list(body.IndustryCode.iloc[:N])
    Zc = [c for c in cols if c[:3] in countries and c[3:].isdigit() and 1 <= int(c[3:]) <= N]
    Z = body[Zc].to_numpy(float).clip(min=0)       # a few tiny negative flows in WIOT (journal 7)
    F = {k: body[[f"{c}{k}" for c in countries]].to_numpy(float) for k in (57, 60)}
    go = d[d.IndustryCode == "GO"][Zc].to_numpy(float).ravel()
    va = d[d.IndustryCode == "VA"][Zc].to_numpy(float).ravel()
    return dict(Z=Z, F=F, x=go, va=va, countries=countries, codes=codes)


@lru_cache(None)
def sea():
    s = pd.read_excel(RAW / "wiod" / "Socio_Economic_Accounts.xlsx", sheet_name="DATA")
    return s.melt(id_vars=["country", "variable", "description", "code"], var_name="year", value_name="v")


@lru_cache(None)
def deltas():
    """delta_ct = UKCT / K (AMECO total economy); countries without AMECO: median of AMECO countries."""
    from ..v6.series import CODES, ameco_raw, derive
    raw = ameco_raw()
    out = {}
    for g in raw.geo.unique():
        d = raw[raw.geo == g].set_index("year")
        if not {"UKCT", "OKND"} <= set(d.columns):
            continue
        k = derive(d).K
        dl = (d.UKCT / k).loc[2000:2014].dropna()
        for y, v in dl.items():
            out[(g, int(y))] = float(v)
    s = pd.Series(out)
    med = s.groupby(level=1).median()
    return s, med


def delta(c, y, scale=1.0):
    s, med = deltas()
    return scale * s.get((c, y), med.loc[y])


def economy(c, y, dscale=1.0):
    W = wiot(y)
    C = W["countries"]
    ci = C.index(c)
    sl = slice(ci * N, (ci + 1) * N)
    x = W["x"][sl]
    Zc = W["Z"][:, sl]
    with np.errstate(divide="ignore", invalid="ignore"):
        A = np.where(x[None, :] > 0, Zc[sl] / x[None, :], 0.0)
        imp = sum(Zc[j * N:(j + 1) * N] for j in range(len(C)) if j != ci)
        Am = np.where(x[None, :] > 0, imp / x[None, :], 0.0)
    fd = {k: W["F"][k][:, ci] for k in (57, 60)}
    dom = {k: v[sl] for k, v in fd.items()}
    impf = {k: sum(v[j * N:(j + 1) * N] for j in range(len(C)) if j != ci) for k, v in fd.items()}
    s = sea()
    s = s[(s.country == c) & (s.year == y)].pivot_table(index="code", columns="variable", values="v")
    s = s.reindex(W["codes"])
    with np.errstate(divide="ignore", invalid="ignore"):
        hours = (s.H_EMPE * s.EMP / s.EMPE).fillna(0).to_numpy()
    # SEA values are in national currency, WIOT in USD: convert labour and capital via GO ratio per industry
    conv = np.divide(x, s.GO.to_numpy(), out=np.zeros(N), where=s.GO.to_numpy() > 0)
    wages = np.nan_to_num(s.COMP.to_numpy() * conv)
    lab = np.nan_to_num(s.LAB.to_numpy() * conv)
    K = np.nan_to_num(s.K.to_numpy() * conv)
    cfc = delta(c, y, dscale) * K
    e = Economy(labels=list(W["codes"]), A=A, Am=Am, x=x, hours=hours, wages=wages, cfc=cfc,
                gfcf_dom=dom[60].clip(min=0), gfcf_imp=impf[60].clip(min=0), hh_dom=dom[57], hh_imp=impf[57],
                va=W["va"][sl], labour_income=lab, meta=dict(country=c, year=y, K=K, CAP=np.nan_to_num(s.CAP.to_numpy() * conv)))
    return e


def eligible():
    s = sea()
    h = s[(s.variable == "H_EMPE") & s.year.isin(list(YEARS))]
    g = h.groupby(["country", "year"]).v.apply(lambda v: (v > 0).sum())
    ok = g.groupby(level=0).min()
    return sorted(ok[ok >= 30].index)


class NoCapital(Setup):
    def __init__(self, e):
        from ..core import leontief_inverse
        from ..levels import mask_for
        M, Mm, _ = e.system(False, False, "price", e.l())
        self.e, self.M = e, M
        self.L = leontief_inverse(M)
        self.mu = Mm.sum(0) @ self.L
        self.x = e.x
        self.base_mask = mask_for(e.labels, set())


def r1(countries=None, variant="main"):
    OUT.mkdir(parents=True, exist_ok=True)
    countries = countries or eligible()
    rows = []
    dscale = {"main": 1.0, "delta_lo": 0.5, "delta_hi": 1.5, "no_capital": 1.0}[variant]
    for c in countries:
        D = None
        for y in YEARS:
            e = economy(c, y, dscale)
            S = NoCapital(e) if variant == "no_capital" else Setup(e)
            if D is None:
                D = Draws(c, e.n, S.base_mask)
            masks = {"all": S.base_mask, "no_prereg": S.mask(PREREG_EXCL_WIOD)}
            kinds = ("perm", "lnorm", "mix") if variant == "main" else ("perm", "lnorm")
            for r in shares(S, D, e.l(), masks, kinds=kinds):
                rows.append(dict(country=c, year=y, variant=variant, **r))
            # flat vector (equal direct input per unit of output) vs hours
            m = S.base_mask
            zl, zf = S.z(e.l()[None]), S.z(np.ones((1, e.n)))
            mh, mf = metrics_mat(zl[:, m], S.x[m]), metrics_mat(zf[:, m], S.x[m])
            rows.append(dict(country=c, year=y, variant=variant, kind="flat", mask="all", metric="mawd",
                             labour=float(mh["mawd"][0]), share_better=float(mf["mawd"][0] < mh["mawd"][0])))
            rows.append(dict(country=c, year=y, variant=variant, kind="flat", mask="all", metric="d",
                             labour=float(mh["d"][0]), share_better=float(mf["d"][0] < mh["d"][0])))
        print("R1", variant, c, flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / f"r1_{variant}.csv", index=False)
    return out


def classify(df):
    m = df[(df["mask"] == "all") & df.kind.isin(["perm", "lnorm", "mix"])]
    s = m.groupby(["country", "kind", "metric"]).share_better.mean().unstack(["kind", "metric"])
    top = (s[("perm", "mawd")] <= 0.05) & (s[("perm", "d")] <= 0.05) & (s[("lnorm", "mawd")] <= 0.05) & \
          (s[("lnorm", "d")] <= 0.05)
    s.columns = [f"{a}_{b}" for a, b in s.columns]
    s["top5_ab"] = top
    f = df[df.kind == "flat"].groupby(["country", "metric"]).share_better.mean().unstack()
    s["flat_better_mawd_share_years"], s["flat_better_d_share_years"] = f["mawd"], f["d"]
    return s


def _main():
    what = sys.argv[1] if len(sys.argv) > 1 else "r1"
    if what == "r1":
        v = sys.argv[2] if len(sys.argv) > 2 else "main"
        df = r1(variant=v)
        cl = classify(df)
        cl.to_csv(OUT / f"r1_{v}_classified.csv")
        n = len(cl)
        k = int(cl.top5_ab.sum())
        k13 = int(cl.loc[cl.index.isin(V3_13), "top5_ab"].sum())
        print(cl.round(3).to_string())
        print(f"top5 (a)&(b): {k} of {n} ({k / n:.2f}); v3 13 countries: {k13} of {cl.index.isin(V3_13).sum()}")
    elif what == "r2":
        print(r2().round(3).to_string())
    elif what == "r2iv":
        print(r2_iv().round(3).to_string())


# ---------------------------------------------------------------- R2: B3.1 of iteration 4 on WIOD
RENT_W = {"B", "D35", "E36", "E37-E39", "K64", "K65", "K66", "L68"}
GOV_W = {"O84", "P85", "Q", "T", "U"}
NON_EUROPE = {"AUS", "BRA", "CAN", "IDN", "IND", "JPN", "KOR", "MEX", "RUS", "TUR", "TWN", "USA"}


def r2_panel():
    s = sea()
    s = s[s.year.isin(list(YEARS)) & s.variable.isin(["GO", "LAB", "COMP", "CAP", "K", "H_EMPE"])]
    p = s.pivot_table(index=["country", "code", "year"], columns="variable", values="v").reset_index()
    p["delta"] = [delta(c, int(y)) for c, y in zip(p.country, p.year)]
    p = p[~p.code.isin(GOV_W) & (p.K > 0) & (p.LAB > 0) & (p.GO > 0)].copy()
    p["rent"] = p.code.isin(RENT_W)
    p["PI"] = p.CAP - p.delta * p.K
    p["PI_nomi"] = p.CAP + (p.LAB - p.COMP) - p.delta * p.K
    p["r"], p["r_nomi"] = p.PI / p.K, p.PI_nomi / p.K
    p["kw"], p["kw_nomi"] = np.log(p.K / p.LAB), np.log(p.K / p.COMP.where(p.COMP > 0))
    p["pcm"], p["pcm_nomi"] = p.PI / p.GO, p.PI_nomi / p.GO
    p["cy"] = p.country + p.year.astype(str)
    return p


def r2():
    from ..v5.stage1 import ols_fwl
    p = r2_panel()
    rows = []
    samples = {"all": p, "klems_overlap": p[~p.country.isin(NON_EUROPE - {"USA", "JPN"})],
               "non_europe": p[p.country.isin(NON_EUROPE)]}
    for sname, d0 in samples.items():
        for rent in (False, True):
            for var, (y, x, pcm) in {"primary": ("r", "kw", "pcm"), "no_mi": ("r_nomi", "kw_nomi", "pcm_nomi")}.items():
                d = d0 if rent else d0[~d0.rent]
                d = d[d[y].abs() <= 2].replace([np.inf, -np.inf], np.nan).dropna(subset=[y, x, pcm])
                for ctrl in ([], [pcm]):
                    res = ols_fwl(d.rename(columns={"code": "ind"}), y, x, ctrl, ["cy"], cluster="ind", B=9999)
                    rows.append(dict(sample=sname, with_rent=rent, variant=var, controls="pcm" if ctrl else "none",
                                     countries=d.country.nunique(), **res))
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "r2_b31_wiod.csv", index=False)
    return out



def iv_fwl(d, y, x, z, controls, fe, cluster):
    """2SLS with one endogenous regressor and one instrument after FWL on FE + controls; CR1 SE."""
    from ..v5.stage1 import fe_resid
    from scipy import stats
    s = d.replace([np.inf, -np.inf], np.nan).dropna(subset=[y, x, z] + controls)
    R = fe_resid(s, [y, x, z] + controls, fe)
    Y, X, Z, C = R[:, 0], R[:, 1], R[:, 2], R[:, 3:]
    if C.shape[1]:
        P = C @ np.linalg.lstsq(C, np.column_stack([Y, X, Z]), rcond=None)[0]
        Y, X, Z = Y - P[:, 0], X - P[:, 1], Z - P[:, 2]
    b = (Z @ Y) / (Z @ X)
    u = Y - b * X
    g = s[cluster].to_numpy()
    cl, idx = np.unique(g, return_inverse=True)
    G = len(cl)
    sc = np.bincount(idx, weights=Z * u, minlength=G)
    se = np.sqrt((sc ** 2).sum() * G / (G - 1)) / abs(Z @ X)
    first = (Z @ X) / (Z @ Z)
    fs_sc = np.bincount(idx, weights=Z * (X - first * Z), minlength=G)
    fs_t = first / (np.sqrt((fs_sc ** 2).sum() * G / (G - 1)) / (Z @ Z))
    return dict(beta=b, se=se, t=b / se, p=2 * stats.t.sf(abs(b / se), G - 1), first_stage_t=fs_t, n=len(s), clusters=G)


def r2_iv():
    p = r2_panel()
    loo = {}                       # leave-one-out median of the same industry-year in other countries
    for (c, yy), g in p.groupby(["code", "year"]):
        for col in ("kw", "kw_nomi"):
            v = g[col].to_numpy()
            for i, ix in enumerate(g.index):
                loo[(ix, col)] = np.nanmedian(np.delete(v, i)) if len(v) > 1 else np.nan
    for col in ("kw", "kw_nomi"):
        p[col + "_iv"] = [loo[(ix, col)] for ix in p.index]
    rows = []
    samples = {"all": p, "klems_overlap": p[~p.country.isin(NON_EUROPE - {"USA", "JPN"})],
               "non_europe": p[p.country.isin(NON_EUROPE)]}
    for sname, d0 in samples.items():
        for var, (y, x, pcm) in {"primary": ("r", "kw", "pcm"), "no_mi": ("r_nomi", "kw_nomi", "pcm_nomi")}.items():
            d = d0[~d0.rent]
            d = d[d[y].abs() <= 2]
            for ctrl in ([], [pcm]):
                res = iv_fwl(d, y, x, x + "_iv", ctrl, ["cy"], "code")
                rows.append(dict(sample=sname, variant=var, controls="pcm" if ctrl else "none", **res))
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "r2_b31_wiod_iv.csv", index=False)
    return out


if __name__ == "__main__":
    _main()
