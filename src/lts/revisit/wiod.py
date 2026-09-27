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
    Z = body[Zc].to_numpy(float)
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


if __name__ == "__main__":
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
