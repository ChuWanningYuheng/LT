"""Iteration 9, stage 8: disaggregation gradient with BLS hours instead of QCEW employment (journal 10).

Outputs: results/v9/s8_*.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..v8.rule import decide
from .stage1 import OUT, gradient_paired

R9 = OUT.parents[1] / "data" / "raw" / "v9"


def ip_2017():
    s = pd.read_csv(R9 / "bls_ip.series", sep="\t", dtype=str)
    s.columns = [c.strip() for c in s.columns]
    s = s.apply(lambda c: c.str.strip())
    s = s[(s.measure_code.isin(["L20", "W20"])) & (s.duration_code == "0") & s.industry_code.str.startswith("N")]
    d = pd.read_csv(R9 / "bls_ip.data.1.AllData", sep="\t", dtype=str)
    d.columns = [c.strip() for c in d.columns]
    d = d.apply(lambda c: c.str.strip())
    d = d[(d.year == "2017") & d.series_id.isin(s.series_id)]
    d["value"] = pd.to_numeric(d.value, errors="coerce")
    m = d.merge(s[["series_id", "industry_code", "measure_code"]], on="series_id")
    m["naics"] = m.industry_code.str[1:].str.rstrip("_")
    w = m.pivot_table(index="naics", columns="measure_code", values="value", aggfunc="first")
    return w


def bea_hours(coverage_fix=False):
    from ..v8.stage3 import _naics_tokens, bea_detail_system, concordance, detail_employment
    inds, Z, m, x, comp = bea_detail_system()
    E, matched, info = detail_employment(inds, comp, coverage_fix)
    ip = ip_2017()
    hrs, hpw = ip.L20.dropna(), (ip.L20 / ip.W20).dropna()
    cc = concordance()
    ci = dict(zip(inds, comp))
    owners = {}
    toks = {}
    for det in inds:
        t = _naics_tokens(cc.naics.get(det, "")) if det in cc.index else []
        toks[det] = t
        for k in t:
            owners.setdefault(k, []).append(det)
    H, how = {}, {}
    for det in inds:
        t = toks[det]
        if t and all(k in hrs.index for k in t):
            h = 0.0
            for k in t:
                own = owners[k]
                wts = np.array([max(ci[o], 0) for o in own], float)
                share = wts[own.index(det)] / wts.sum() if wts.sum() > 0 else 1 / len(own)
                h += hrs[k] * share
            H[det], how[det] = h, "ip"
        else:
            k3 = t[0][:3] if t else ""
            r = hpw.get(k3, hpw.get(k3[:2], np.nan))
            if not np.isfinite(r):
                r = float(hrs.sum() / ip.W20.dropna().sum())
            H[det], how[det] = E[det] * r, "qcew x ip hours/job"
    H = pd.Series(H)
    how = pd.Series(how)
    cov = dict(share_industries_ip=float((how == "ip").mean()),
               share_hours_ip=float(H[how == "ip"].sum() / H.sum()))
    return H, how, cov


def run():
    out, covs = [], []
    for fix in (False, True):
        H, how, cov = bea_hours(fix)
        covs.append(cov | dict(coverage_fix=fix))
        g = gradient_paired(coverage_fix=fix, E_override=H, tag=("coverage fix" if fix else "main") + ", BLS hours",
                            indep=True)
        out.append(g)
    G = pd.concat(out)
    G.to_csv(OUT / "s8_gradient_hours.csv", index=False)
    pd.DataFrame(covs).to_csv(OUT / "s8_coverage.csv", index=False)
    return G, pd.DataFrame(covs)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    G, C = run()
    print(C.round(3).to_string(), "\n", G.round(4).to_string())
