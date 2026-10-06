"""Iteration 7, part 2: reduction of complex labour without wages, USA (pre_registration_v7.md, part 2).

Occupations: BLS EP table 5.4 (2025 categories) -> training years h_o; O*NET 31.0 alternative.
psi_o: Rowthorn/Hilferding (no interest), Shaikh exp(lambda h), observed hourly-wage ratio (OEWS May 2022).
Test 1: premiums vs training cost.  Tests 2-3: industry psi_j (OEWS 2022 in4, employment weights) in the
stage-P placebo test of iteration 3 on BEA 2022 and FIGARO USA 2022, against compression benchmarks (c1, c2).
Outputs: results/v7/p2_*.csv
"""
from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .. import bea
from ..codes import BEA_TO_NACE
from ..economy import build as build_figaro
from ..v2.placebo_cost import metrics_mat
from ..v3.placebo_disp import Draws, Setup, shares
from ..v3.weights import hilferding_E
from ..v4.b_data import divs
from ..v6.series import ROOT

OUT = ROOT / "results" / "v7"
R3, R7 = ROOT / "data" / "raw" / "v3", ROOT / "data" / "raw" / "v7"
H_S, T, HY = 1600, 40, 1800

EDU = {"No formal educational credential": 0, "High school diploma or equivalent": 2,
       "Postsecondary nondegree award": 3, "Some college, no degree": 3, "Associate's degree": 4,
       "Bachelor's degree": 6, "Master's degree": 8, "Doctoral or professional degree": 11}
OJT = {"None": 0, "Short-term on-the-job training": 0.04, "Moderate-term on-the-job training": 0.5,
       "Long-term on-the-job training": 1.5, "Apprenticeship": 4, "Internship/residency": 3}
EXP = {"None": 0, "Less than 5 years": 2.5, "5 years or more": 6}
RL_YEARS = {1: 0, 2: 2, 3: 3, 4: 3, 5: 4, 6: 6, 7: 7, 8: 8, 9: 9, 10: 11, 11: 11, 12: 13}
MID = {"PT": {1: 0, 2: 0.04, 3: 2 / 12, 4: 4.5 / 12, 5: 0.75, 6: 1.5, 7: 3, 8: 7, 9: 12},
       "OJ": {1: 0, 2: 0.04, 3: 2 / 12, 4: 4.5 / 12, 5: 0.75, 6: 1.5, 7: 3, 8: 7, 9: 12}}
BEA_NAICS = {"111CA": ["111", "112"], "113FF": ["113", "114", "115"], "211": ["211"], "212": ["212"], "213": ["213"],
             "22": ["22"], "23": ["23"], "321": ["321"], "327": ["327"], "331": ["331"], "332": ["332"], "333": ["333"],
             "334": ["334"], "335": ["335"], "3361MV": ["3361", "3362", "3363"], "3364OT": ["3364", "3365", "3366", "3369"],
             "337": ["337"], "339": ["339"], "311FT": ["311", "312"], "313TT": ["313", "314"], "315AL": ["315", "316"],
             "322": ["322"], "323": ["323"], "324": ["324"], "325": ["325"], "326": ["326"], "42": ["42"], "441": ["441"],
             "445": ["445"], "452": ["452", "455"], "4A0": ["442", "443", "444", "446", "447", "448", "449", "451", "453", "454",
                                                           "456", "457", "458", "459"],
             "481": ["481"], "482": ["482"], "483": ["483"], "484": ["484"], "485": ["485"], "486": ["486"],
             "487OS": ["487", "488", "492"], "493": ["493"], "511": ["511", "513"], "512": ["512"], "513": ["515", "516", "517"],
             "514": ["518", "519"], "521CI": ["521", "522"], "523": ["523"], "524": ["524"], "525": ["525"], "ORE": ["531"],
             "532RL": ["532", "533"], "5411": ["5411"], "5415": ["5415"],
             "5412OP": ["5412", "5413", "5414", "5416", "5417", "5418", "5419"], "55": ["55"], "561": ["561"],
             "562": ["562"], "61": ["61"], "621": ["621"], "622": ["622"], "623": ["623"], "624": ["624"],
             "711AS": ["711", "712"], "713": ["713"], "721": ["721"], "722": ["722"], "81": ["81"],
             "GFGN": ["9991"], "GSLG": ["9992", "9993"], "GFE": ["491"]}   # 513/516/491: NAICS 2022 codes in OEWS 2022 (review R5)


def naics_to_bea(code: str):
    best = None
    for b, prefs in BEA_NAICS.items():
        for p in prefs:
            if code.startswith(p) and (best is None or len(p) > best[1]):
                best = (b, len(p))
    return best[0] if best else None


# ---------------------------------------------------------------- occupations
def bls_categories():
    t = pd.read_excel(R7 / "bls" / "education.xlsx", "Table 5.4", header=1, keep_default_na=False)   # "None" is a category
    t.columns = ["title", "soc", "edu", "exp", "ojt", "ooh"]
    t = t[t.soc.astype(str).str.match(r"\d\d-\d{4}$")]
    return t.set_index("soc")


def onet_years():
    z = zipfile.ZipFile(R7 / "onet" / "db_31_0_excel.zip")
    ed = pd.concat([pd.read_excel(io.BytesIO(z.read(f"db_31_0_excel/{f}.xlsx")))
                    for f in ("Education", "Training and Experience")])
    ed = ed[ed["Scale ID"].isin(["RL", "PT", "OJ", "RW"])]
    ed["soc"] = ed["O*NET-SOC Code"].str[:7]
    out = {}
    for sc, mp in (("RL", RL_YEARS), ("PT", MID["PT"]), ("OJ", MID["OJ"])):
        s = ed[ed["Scale ID"] == sc].copy()
        s["y"] = s.Category.map(mp) * s["Data Value"] / 100
        out[sc] = s.groupby(["O*NET-SOC Code", "soc"]).y.sum().groupby("soc").mean()
    return pd.DataFrame(out)


def oews_national():
    z = zipfile.ZipFile(R3 / "oesm22nat.zip")
    d = pd.read_excel(io.BytesIO(z.read("oesm22nat/national_M2022_dl.xlsx")))
    d = d[d.O_GROUP == "detailed"]
    d["w"] = pd.to_numeric(d.H_MEAN, errors="coerce")
    d["emp"] = pd.to_numeric(d.TOT_EMP, errors="coerce")
    return d.set_index("OCC_CODE")[["OCC_TITLE", "w", "emp"]]


def occupations(E, Et):
    c = bls_categories()
    o = oews_national().join(c[["edu", "exp", "ojt"]], how="inner")
    o = o[o.w.notna() & o.emp.notna()]
    o["y_edu"], o["y_ojt"], o["y_exp"] = o.edu.map(EDU), o.ojt.map(OJT), o.exp.map(EXP)
    on = onet_years()
    o = o.join(on, how="left")
    o["simple"] = (o.edu == "No formal educational credential") & (o.exp == "None") & \
                  o.ojt.isin(["None", "Short-term on-the-job training"])
    ws = np.average(o.w[o.simple], weights=o.emp[o.simple])
    o["ln_obs"] = np.log(o.w / ws)
    variants = {
        "main": o.y_edu + 0.5 * o.y_ojt,
        "HS = simple": (o.y_edu - 2).clip(lower=0) + 0.5 * o.y_ojt,
        "OJT x0": o.y_edu, "OJT x1": o.y_edu + o.y_ojt,
        "exp x0.25": o.y_edu + 0.5 * o.y_ojt + 0.25 * o.y_exp, "exp x0.5": o.y_edu + 0.5 * o.y_ojt + 0.5 * o.y_exp,
        "O*NET": o.RL + 0.5 * (o.PT + o.OJ),
    }
    for k, h in variants.items():
        o[f"h[{k}]"] = h
    o.attrs.update(w_simple=ws, E=E, Et=Et)
    return o


def psi_rowthorn(h, E, Et=None):
    if Et is None:
        return 1 + h * (H_S + E) / (T * HY)
    return (1 + h * (H_S + E - Et) / (T * HY)) / (1 - h * Et / (T * HY))


def test1(o):
    rows = []
    for k in [c[2:-1] for c in o.columns if c.startswith("h[")]:
        d = o[o[f"h[{k}]"].notna()]
        h = d[f"h[{k}]"]
        fit = sm.WLS(d.ln_obs, sm.add_constant(h), weights=d.emp).fit()
        lam = fit.params.iloc[1]
        lnR = np.log(psi_rowthorn(h, o.attrs["E"]))
        lnRt = np.log(psi_rowthorn(h, o.attrs["E"], o.attrs["Et"]))
        w = d.emp / d.emp.sum()
        mean_obs = float((w * d.ln_obs).sum())
        mean_R = float((w * lnR).sum())
        mean_S = float((w * lam * h).sum())
        rows.append(dict(variant=k, n_occ=len(d), lambda_hat=lam, intercept=fit.params.iloc[0], R2_shaikh=fit.rsquared,
                         mean_ln_premium=mean_obs, rowthorn_part=mean_R, rowthorn_teachers_part=float((w * lnRt).sum()),
                         interest_part=mean_S - mean_R, residual_part=mean_obs - mean_S,
                         share_rowthorn=mean_R / mean_obs, share_shaikh=mean_S / mean_obs,
                         mean_h=float((w * h).sum()), corr_psiR_obs=np.corrcoef(lnR, d.ln_obs)[0, 1]))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- industries
def oews_industry():
    z = zipfile.ZipFile(R3 / "oesm22in4.zip")
    d = pd.read_excel(io.BytesIO(z.read("oesm22in4/nat4d_M2022_dl.xlsx")))
    d = d[d.O_GROUP == "detailed"].copy()
    d["emp"] = pd.to_numeric(d.TOT_EMP, errors="coerce")
    d["naics"] = d.NAICS.astype(str).str[:4]
    d["bea"] = d.naics.map(naics_to_bea)
    return d[d.emp.notna() & d.bea.notna()][["bea", "naics", "OCC_CODE", "emp"]]


def industry_psi(o, ind, col):
    m = ind.merge(o[[col]], left_on="OCC_CODE", right_index=True)
    m = m[m[col].notna()]
    g = m.groupby("bea").apply(lambda s: np.average(s[col], weights=s.emp), include_groups=False)
    emp = m.groupby("bea").emp.sum()
    return g, emp


def label_psi(labels, psi_bea, emp_bea, kind):
    out = np.ones(len(labels))
    for i, lab in enumerate(labels):
        if kind == "bea":
            parts = [p for p in lab.split("+") if p in psi_bea.index]
        else:
            dv = divs(lab)
            parts = [b for b in psi_bea.index if b in BEA_TO_NACE and set(BEA_TO_NACE[b]) & dv]
        if parts:
            out[i] = np.average(psi_bea[parts], weights=emp_bea[parts])
    return out


def compress(l, target, mask):
    """c1: power transform matching SD ln; c2: rank-preserving quantile map onto target's distribution."""
    pos = mask & (l > 0) & (target > 0)
    a = np.log(target[pos]).std() / np.log(l[pos]).std()
    c1 = l.copy()
    c1[pos] = l[pos] ** a * np.exp(np.log(target[pos]).mean() - a * np.log(l[pos]).mean())
    c2 = target.copy()
    c2[pos] = np.sort(target[pos])[np.argsort(np.argsort(l[pos]))]
    return c1, c2, a


def price_tests(o, ind, econ):
    E = o.attrs["E"]
    cols = {}
    h = o["h[main]"]
    lam = test1(o).set_index("variant").loc["main", "lambda_hat"]
    cols["Rowthorn"] = psi_rowthorn(h, E)
    cols["Rowthorn, teachers skilled"] = psi_rowthorn(h, E, o.attrs["Et"])
    cols["Shaikh lambda_hat"] = np.exp(lam * h)
    for L in (0.05, 0.08, 0.10):
        cols[f"Shaikh {L}"] = np.exp(L * h)
    cols["Rowthorn, O*NET"] = psi_rowthorn(o["h[O*NET]"], E)
    cols["Rowthorn, exp x0.5"] = psi_rowthorn(o["h[exp x0.5]"], E)
    cols["observed wage ratio"] = np.exp(o.ln_obs)
    for k, v in cols.items():
        o[f"psi[{k}]"] = v
    rows, wts = [], []
    for tname, (e, kind) in econ.items():
        S = Setup(e)
        D = Draws("USA_v7_" + tname, e.n, S.base_mask)
        m = S.base_mask
        l = e.l()
        base = metrics_mat(S.z(l[None])[:, m], S.x[m])
        flat = metrics_mat(S.z(np.ones_like(l)[None] * (l > 0))[:, m], S.x[m])
        rows.append(dict(table=tname, vector="hours (no reduction)", mawd=base["mawd"][0], d=base["d"][0]))
        rows.append(dict(table=tname, vector="flat", mawd=flat["mawd"][0], d=flat["d"][0]))
        for k in cols:
            pb, eb = industry_psi(o, ind, f"psi[{k}]")
            psi = label_psi(e.labels, pb, eb, kind)
            lr = l * psi
            c1, c2, a = compress(l, lr, m)
            mm = {nm: metrics_mat(S.z(v[None])[:, m], S.x[m]) for nm, v in (("reduced", lr), ("c1", c1), ("c2", c2))}
            sh = {(r["kind"], r["metric"]): r["share_better"] for r in shares(S, D, lr, {"all": m}, kinds=("perm", "lnorm"))}
            rows.append(dict(table=tname, vector=k, mawd=mm["reduced"]["mawd"][0], d=mm["reduced"]["d"][0],
                             c1_mawd=mm["c1"]["mawd"][0], c1_d=mm["c1"]["d"][0], c2_mawd=mm["c2"]["mawd"][0],
                             c2_d=mm["c2"]["d"][0], alpha_c1=a, sd_ln_psi=np.log(psi[m]).std(),
                             perm_mawd=sh[("perm", "mawd")], perm_d=sh[("perm", "d")],
                             lnorm_mawd=sh[("lnorm", "mawd")], lnorm_d=sh[("lnorm", "d")],
                             beats_c2_both=bool(mm["reduced"]["mawd"][0] < mm["c2"]["mawd"][0] and
                                                mm["reduced"]["d"][0] < mm["c2"]["d"][0]),
                             loses_c2_both=bool(mm["reduced"]["mawd"][0] > mm["c2"]["mawd"][0] and
                                                mm["reduced"]["d"][0] > mm["c2"]["d"][0])))
            wts.append(pd.DataFrame({"table": tname, "label": e.labels, "psi_" + k: psi}).set_index(["table", "label"]))
    return pd.DataFrame(rows), pd.concat(wts, axis=1).T.groupby(level=0).first().T


def main():
    ef, _ = build_figaro("USA", 2022)
    E, src = hilferding_E(ef, "USA", 2022)
    j = [i for i, lab in enumerate(ef.labels) if "P85" in lab.split("+")][0]
    from ..core import leontief_inverse
    M, _, _ = ef.system(True, False, "price", ef.l())
    v = ef.l() @ leontief_inverse(M)
    Et = E * ef.l()[j] / v[j]                                   # direct education hours within E
    o = occupations(E, Et)
    t1 = test1(o)
    ind = oews_industry()
    eb, _ = bea.build(2022)
    pt, w = price_tests(o, ind, {"BEA 2022": (eb, "bea"), "FIGARO USA 2022": (ef, "figaro")})
    info = pd.Series(dict(E=E, E_source=src, Et=Et, w_simple=o.attrs["w_simple"], n_simple=int(o.simple.sum()),
                          emp_simple=float(o.emp[o.simple].sum()), n_occ=len(o)))
    return o, t1, pt, w, info


if __name__ == "__main__":
    o, t1, pt, w, info = main()
    print(info.to_string(), flush=True)
    print(t1.round(3).to_string(), flush=True)
    print(pt.round(4).to_string(), flush=True)
    o.drop(columns=["OCC_TITLE"]).to_csv(OUT / "p2_occupations.csv")
    t1.to_csv(OUT / "p2_test1.csv", index=False)
    pt.to_csv(OUT / "p2_prices.csv", index=False)
    w.to_csv(OUT / "p2_industry_psi.csv")
    info.to_csv(OUT / "p2_info.csv")


def joint(o, ind):
    """Test 4 (exploratory): Marx sign on KLEMS US with K + non-NA intangibles and W* = H psi_j wbar."""
    from .part1 import klems_panel, make_a, slope
    p = klems_panel()
    us = p[(p.geo == "US") & p.U_lag.notna()].copy()
    rows = []
    for k in ("Rowthorn", "Shaikh lambda_hat"):
        pb, eb = industry_psi(o, ind, f"psi[{k}]")
        psi = {}
        for lab in us.ind.unique():
            dv = divs(lab)
            parts = [b for b in pb.index if b in BEA_TO_NACE and set(BEA_TO_NACE[b]) & dv]
            psi[lab] = np.average(pb[parts], weights=eb[parts]) if parts else 1.0
        us["psi"] = us.ind.map(psi)
        wbar = us.groupby("year").apply(lambda s: s.W.sum() / (s.H * s.psi).sum(), include_groups=False)
        us["Wstar"] = us.H * us.psi * us.year.map(wbar)
        r1, x1 = make_a()(us, 1.0)
        K1 = us.K + us.K_nonNA
        for name, r, x in (("base", us.r, np.log(us.K / us.W)),
                           ("W* only", us.r, np.log(us.K / us.Wstar)),
                           ("K + intangibles only", r1, x1),
                           ("both", r1, np.log(K1 / us.Wstar))):
            b, pv, _ = slope(us.assign(_r=r, _x=x), "_r", "_x")
            rows.append(dict(psi=k, spec=name, beta=b, p=pv, n=len(us)))
    return pd.DataFrame(rows)
