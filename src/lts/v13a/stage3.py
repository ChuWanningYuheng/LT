"""Iteration 13a, stage 3: the world system in hours, 1995-2024 (EXIOBASE 3.11.2 ixi; pre_registration_v13a.md
'Этап 3', with the edits of journal items 3 and 5). Outcomes 120-122.

lambda_h = h (I - M)^-1 over 49 regions x 163 industries; h_j = hours / output for productive industries, 0 for
unproductive (results/v13a/s3_classification.csv). Main: net lambda, M = A + sum_r g_r d_r' (g_r - structure of the
region's gross fixed capital formation by supplier, d_r - consumption of fixed capital per unit of output of the
region's industries); gross lambda (M = A) - variant.
V: compensation of productive industries of region r (all skills) spent along the region's household final
consumption structure. L - all hours of productive industries (EXIOBASE hours include the self-employed and
peasants - journal item 5).
Capital of region r: K_r = (cn / cgdpo)_r [PWT 11] x value added at basic prices of r (EXIOBASE); for the rest-of-
world regions the ratio of sums over PWT countries outside EXIOBASE (results/v13a/s3_row_regions.csv); 2024 - PWT
2023. Composition by capital-good products: Soedersten et al. Kbar (EXIOBASE 3.8.2, cfc, pxi; 2021-2024 - 2020),
products mapped to producing industries (results/v13a/s3_product_industry.csv; ambiguous products split by output).
Current valuation only. Money analogue: lambda_$ = 1: S_$ = sum of net value added at basic prices - V_$, C_$ = sum K.
EXIOBASE licence (3.9+): no raw or derived coefficients are written - only world aggregates by year.

Usage: PYTHONPATH=src python -P -m lts.v13a.stage3 [years|tests|all]
"""
from __future__ import annotations

import io
import sys
import zipfile

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.linalg as sl
import scipy.sparse as sp

from ..v6.series import ROOT, RAW
from . import est
from . import matrix as mx

OUT = ROOT / "results" / "v13a"
EXIO = ROOT.parent / "LT_external" / "v13" / "exiobase"
KBAR = RAW / "v13" / "capital_matrices"
YEARS = list(range(1995, 2025))
PHASES_W = {"A": (1995, 2007), "B": (2010, 2019)}
D = 0.002
ISO2_3 = {"AT": "AUT", "BE": "BEL", "BG": "BGR", "CY": "CYP", "CZ": "CZE", "DE": "DEU", "DK": "DNK", "EE": "EST",
          "ES": "ESP", "FI": "FIN", "FR": "FRA", "GR": "GRC", "HR": "HRV", "HU": "HUN", "IE": "IRL", "IT": "ITA",
          "LT": "LTU", "LU": "LUX", "LV": "LVA", "MT": "MLT", "NL": "NLD", "PL": "POL", "PT": "PRT", "RO": "ROU",
          "SE": "SWE", "SI": "SVN", "SK": "SVK", "GB": "GBR", "US": "USA", "JP": "JPN", "CN": "CHN", "CA": "CAN",
          "KR": "KOR", "BR": "BRA", "IN": "IND", "MX": "MEX", "RU": "RUS", "AU": "AUS", "CH": "CHE", "TR": "TUR",
          "TW": "TWN", "NO": "NOR", "ID": "IDN", "ZA": "ZAF", "WA": "WA", "WL": "WL", "WE": "WE", "WF": "WF", "WM": "WM"}
KBAR_CODE = {"ROU": "ROM", "WA": "WWA", "WL": "WWL", "WE": "WWE", "WF": "WWF", "WM": "WWM"}


# ================================================================ readers
def _read(z, name, header_rows):
    """EXIOBASE text matrix: header_rows header lines (region, sector/category, ...), then rows with index columns."""
    with z.open(name) as f:
        head = [f.readline().decode().rstrip("\n").split("\t") for _ in range(header_rows)]
        df = pd.read_csv(f, sep="\t", header=None, engine="c")
    return head, df


def read_year(y):
    z = zipfile.ZipFile(EXIO / f"IOT_{y}_ixi.zip")
    head, Zd = _read(z, "Z.txt", 3)
    reg = head[0][2:]
    sec = head[1][2:]
    Z = Zd.iloc[:, 2:].to_numpy(float)
    _, xd = _read(z, "x.txt", 1)
    x = xd.iloc[:, 2].to_numpy(float)
    hy, Yd = _read(z, "Y.txt", 3)
    Y = Yd.iloc[:, 2:].to_numpy(float)
    yreg, ycat = hy[0][2:], hy[1][2:]
    def F(name):
        with z.open(name) as f:
            for _ in range(3):
                f.readline()
            dd = pd.read_csv(f, sep="\t", header=None, index_col=0, engine="c")
        return dd.apply(pd.to_numeric, errors="coerce")
    fi = F("factor_inputs/F.txt")
    em = F("employment/F.txt")
    return dict(Z=Z, x=x, Y=Y, yreg=np.array(yreg), ycat=np.array(ycat), reg=np.array(reg), sec=np.array(sec), fi=fi,
                em=em)


def pwt_ratio():
    """(cn / cgdpo) by EXIOBASE region and year 1995-2024."""
    d = pd.read_excel(RAW / "v13" / "pwt_data" / "pwt110.xlsx", sheet_name="Data",
                      usecols=["countrycode", "year", "cn", "cgdpo"])
    d = d[(d.year >= 1995)]
    row = pd.read_csv(OUT / "s3_row_regions.csv")
    out = {}
    for r2, r3 in ISO2_3.items():
        if r3 in ("WA", "WL", "WE", "WF", "WM"):
            cc = row[row.exiobase_row_region == r3].iso3
            g = d[d.countrycode.isin(cc)].dropna(subset=["cn", "cgdpo"]).groupby("year")[["cn", "cgdpo"]].sum()
        else:
            g = d[d.countrycode == r3].set_index("year")[["cn", "cgdpo"]]
        s = (g.cn / g.cgdpo)
        s = s.reindex(range(1995, 2025))
        s.loc[2024] = s.loc[2023] if not np.isfinite(s.loc[2024]) else s.loc[2024]
        out[r2] = s
    return pd.DataFrame(out)


def kbar(y):
    yy = min(y, 2020)
    m = sio.loadmat(KBAR / f"Kbar_exio_v3_8_2_{yy}_cfc_pxi.mat")
    countries = [str(c[0][0]) for c in m["countries"]]
    return m["Kbar"], countries


# ================================================================ one year
def year(y, ratio, conc, cls):
    d = read_year(y)
    Z, x, Y = d["Z"], d["x"], d["Y"]
    n = len(x)
    regions = list(dict.fromkeys(d["reg"]))
    assert len(regions) == 49 and n == 49 * 163
    sec163 = list(d["sec"][:163])
    assert sec163 == list(cls.industry), "industry labels of 3.11.2 differ from the classification table"
    ridx = {r: np.where(d["reg"] == r)[0] for r in regions}
    xs = np.where(x > 0, x, 1.0)
    A = Z / xs[None, :]
    A[:, x <= 0] = 0.0
    fi, em = d["fi"], d["em"]
    lab = fi.index.astype(str)
    comp_rows = [i for i in lab if i.startswith("Compensation of employees")]
    comp = fi.loc[comp_rows].to_numpy(float)                   # 3 skills x n
    cfc = fi.loc["Operating surplus: Consumption of fixed capital"].to_numpy(float)
    va = fi.loc[[i for i in lab if not i.startswith("Taxes less subsidies on products")]].to_numpy(float).sum(0)
    hrs_rows = [i for i in em.index.astype(str) if i.startswith("Employment hours")]
    H = em.loc[hrs_rows].to_numpy(float)
    H_all = H.sum(0)
    H_lm = em.loc[[i for i in hrs_rows if "High-skilled" not in i]].to_numpy(float).sum(0)
    comp_all = comp.sum(0)
    comp_lm = comp[[i for i, r in enumerate(comp_rows) if "High-skilled" not in r]].sum(0)
    prod_main = np.tile(cls.productive_main.to_numpy(bool), 49)
    prod_pq = np.tile(cls.productive_variant_PQ_out.to_numpy(bool), 49)
    # final demand blocks
    hh = {r: Y[:, np.where((d["yreg"] == r) & (np.char.find(d["ycat"].astype(str), "households") >= 0))[0]].sum(1)
          for r in regions}
    gf = {r: Y[:, np.where((d["yreg"] == r) & (np.char.startswith(d["ycat"].astype(str), "Gross fixed capital")))[0]].sum(1)
          for r in regions}
    # net M
    M = A.copy()
    dvec = cfc / xs
    dvec[x <= 0] = 0.0
    for r in regions:
        g = gf[r]
        if g.sum() > 0:
            M[:, ridx[r]] += np.outer(g / g.sum(), dvec[ridx[r]])
    rhs = {}
    for nm, Hh, pr in (("main", H_all, prod_main), ("PQ_out", H_all, prod_pq), ("lowmed", H_lm, prod_main)):
        rhs[nm] = np.where(pr, Hh / xs, 0.0) * (x > 0)
    names = list(rhs)
    R = np.column_stack([rhs[k] for k in names])
    lam = {}
    for kind, mat in (("net", M), ("gross", A)):
        lu = sl.lu_factor(np.eye(n) - mat, check_finite=False)
        sol = sl.lu_solve(lu, R, trans=1, check_finite=False)
        for i, k in enumerate(names):
            lam[(kind, k)] = sol[:, i]
    del M, A, Z
    # capital composition by producer industry: Kbar (3.8.2) region columns -> products -> producing industries
    Kb, kc = kbar(y)
    kreg = {r: kc.index(KBAR_CODE.get(ISO2_3[r], ISO2_3[r])) for r in regions}
    prod_to_ind = [list(map(int, str(v).split(";"))) for v in conc.industry_indices]
    ri, ci, vv = [], [], []
    for s_ in regions:
        base = ridx[s_][0]
        for p in range(200):
            inds = np.array(prod_to_ind[p])
            w = x[base + inds]
            w = w / w.sum() if w.sum() > 0 else np.full(len(inds), 1.0 / len(inds))
            for i_, w_ in zip(inds, w):
                ri.append(kreg[s_] * 200 + p)
                ci.append(base + i_)
                vv.append(w_)
    Pmap = sp.csr_matrix((vv, (ri, ci)), shape=(Kb.shape[0], n))
    def to_producers(pvec):
        """9800 product vector (Kbar region order) -> 7987 producer industries; ambiguous products split by output."""
        return np.asarray(Pmap.T @ pvec).ravel()
    re_col = list(cls.industry).index("Real estate activities (70)")
    rows = {}
    for r in regions:
        cols = slice(kreg[r] * 163, (kreg[r] + 1) * 163)
        Kr = Kb[:, cols]
        tot = Kr.sum()
        share_re = Kr[:, re_col].sum() / tot if tot > 0 else 0.0
        pv = Kr.sum(1)
        pv_nore = pv - Kr[:, re_col]
        k = to_producers(pv)
        k_nore = to_producers(pv_nore)
        g = gf[r]
        Kreg = float(ratio.loc[y, r]) * float(va[ridx[r]].sum())
        rows[r] = dict(K=Kreg, k=k / k.sum() if k.sum() > 0 else k, k_nore=k_nore / k_nore.sum() if k_nore.sum() > 0
                       else k_nore, share_re=share_re, g=g / g.sum() if g.sum() > 0 else g)
    del Kb
    out = []
    for kind in ("net", "gross"):
        for var, lk in (("main", "main"), ("PQ_out", "PQ_out"), ("lowmed", "lowmed")):
            lt = lam[(kind, lk)]
            prod = prod_pq if var == "PQ_out" else prod_main
            Hh = H_lm if var == "lowmed" else H_all
            cp = comp_lm if var == "lowmed" else comp_all
            L = float((Hh * prod).sum())
            V_h = V_d = 0.0
            for r in regions:
                W = float((cp * prod)[ridx[r]].sum())
                c = hh[r]
                if c.sum() > 0:
                    V_h += W * float(lt @ (c / c.sum()))
                V_d += W
            C_h = sum(rows[r]["K"] * float(lt @ rows[r]["k"]) for r in regions)
            C_h_nore = sum(rows[r]["K"] * (1 - rows[r]["share_re"]) * float(lt @ rows[r]["k_nore"]) for r in regions)
            C_h_gfcf = sum(rows[r]["K"] * float(lt @ rows[r]["g"]) for r in regions)
            C_d = sum(rows[r]["K"] for r in regions)
            C_d_nore = sum(rows[r]["K"] * (1 - rows[r]["share_re"]) for r in regions)
            nva = float(va.sum() - cfc.sum())
            S_h, S_d = L - V_h, nva - V_d
            out.append(dict(year=y, lam=kind, variant=var, L=L, V_h=V_h, S_h=S_h, C_h=C_h, C_h_nore=C_h_nore,
                            C_h_gfcf=C_h_gfcf, V_d=V_d, S_d=S_d, C_d=C_d, C_d_nore=C_d_nore,
                            r_h=S_h / (C_h + V_h), k_h=C_h / V_h, e_h=S_h / V_h, r_d=S_d / (C_d + V_d),
                            r_h_nore=S_h / (C_h_nore + V_h), r_d_nore=S_d / (C_d_nore + V_d),
                            r_h_gfcf=S_h / (C_h_gfcf + V_h), k_h_nore=C_h_nore / V_h, k_h_gfcf=C_h_gfcf / V_h))
    return out


def years(ys=None):
    ratio = pwt_ratio()
    conc = pd.read_csv(OUT / "s3_product_industry.csv")
    cls = pd.read_csv(OUT / "s3_classification.csv")
    path = OUT / "s3_world_years.csv"
    done = pd.read_csv(path) if path.exists() else pd.DataFrame()
    have = set(done.year) if len(done) else set()
    for y in (ys or YEARS):
        if y in have:
            continue
        rows = year(y, ratio, conc, cls)
        done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
        done.sort_values(["lam", "variant", "year"]).to_csv(path, index=False)
        print("year", y, flush=True)


# ================================================================ outcomes 120-122
def _sub(y, sub):
    if sub == "-first5":
        return y[y.index > 1999]
    if sub == "-last5":
        return y[y.index < 2015]
    return y


def assess(name, y, direction):
    o = est.within_trend(y, PHASES_W, D, direction, block=3)
    o.update(est.phase_slopes(y, PHASES_W))
    nine = [est.within_trend(_sub(y, s), PHASES_W, D, direction, lag=lag, block=3)["label_full"]
            for lag in (2, 4, 8) for s in ("full", "-first5", "-last5")]
    rob, why = est.robust_flag(o["label_full"], nine)
    o.update(outcome=name, nine=";".join(sorted(set(nine))), robust=rob, robust_why=why)
    return o


def tests():
    W = pd.read_csv(OUT / "s3_world_years.csv")
    main = W[(W.lam == "net") & (W.variant == "main")].set_index("year").sort_index()
    out = [assess("120: trend ln world r_h within phases 1995-2007, 2010-2019", np.log(main.r_h), "<"),
           assess("121: trend ln world k_h within phases", np.log(main.k_h), ">"),
           assess("122: trend ln(r_h / r_$) of the world system within phases", np.log(main.r_h / main.r_d), "<")]
    caveat = ("не толкуется как подтверждение закона: часы включают самозанятых и крестьян, сокращение крестьянской "
              "занятости механически снижает мировые e и r_h (журнал, п. 5)")
    for o in out[:2]:
        o["caveat"] = caveat
    O = pd.DataFrame(out)
    O["no"] = O.outcome.str.extract(r"^(\d+)").astype(int)
    O.to_csv(OUT / "s3_outcomes.csv", index=False)
    # matrix counting for 120-122 (all on hours)
    rows = []
    for no in (120, 121, 122):
        r_ = O[O.no == no].iloc[0]
        for th, (pred, basis) in mx.PRED[no].items():
            v = est.count(pred, r_["est"], r_["ci90_lo"], r_["ci90_hi"], D, 0.0,
                          (r_.get("p_equal_slopes", 1.0) >= 0.10) if np.isfinite(r_.get("p_equal_slopes", np.nan)) else True)
            rows.append(dict(no=no, theory=th, prediction=pred, basis="часы", est=r_["est"], ci90_lo=r_["ci90_lo"],
                             ci90_hi=r_["ci90_hi"], verdict="нейтрально" if pred == "молчит" else v))
    pd.DataFrame(rows).to_csv(OUT / "s3_matrix.csv", index=False)
    # descriptive variants and checks
    desc = []
    for (lamk, var), g in W.groupby(["lam", "variant"]):
        g = g.set_index("year").sort_index()
        for col, d_ in (("r_h", "<"), ("k_h", ">"), ("r_h_nore", "<"), ("k_h_nore", ">"), ("r_h_gfcf", "<"),
                        ("k_h_gfcf", ">"), ("r_d", "<")):
            desc.append(dict(item=f"lambda={lamk}, variant={var}: within phases ln {col}",
                             **est.within_trend(np.log(g[col]), PHASES_W, D, d_, block=3)))
    yy = np.log(main.r_h)
    yy = yy[(yy.index <= 2019) & ~yy.index.isin([2008, 2009])]
    desc.append(dict(item="single trend ln r_h 1995-2019 without 2008-2009 (gaps closed)",
                     **est.single_trend(yy.reset_index(drop=True).set_axis(yy.index), D, "<")))
    # WIOD check (12b outcome 100): sign of the trend over 2000-2014
    w12 = pd.read_csv(ROOT / "results" / "v12b" / "s2_world_hours.csv").set_index("year")
    a = est.single_trend(np.log(main.r_h.loc[2000:2014]), D, "<")
    b = est.single_trend(np.log(w12.r_h.loc[2000:2014]), D, "<")
    desc.append(dict(item="WIOD check 2000-2014: EXIOBASE ln r_h trend", **a))
    desc.append(dict(item="WIOD check 2000-2014: WIOD (12b) ln r_h trend", **b))
    desc.append(dict(item="WIOD check: same sign of the trend", est=float(np.sign(a["est"]) == np.sign(b["est"]))))
    pd.DataFrame(desc).to_csv(OUT / "s3_descriptive.csv", index=False)
    print(O[["outcome", "est", "ci90_lo", "ci90_hi", "label_full", "robust"]].round(5).to_string())


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else "all"
    if part in ("years", "all"):
        years()
    if part in ("tests", "all"):
        tests()
