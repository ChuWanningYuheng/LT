"""Iteration 2, stage 1c step 3: labour's advantage vs labour's cost share (pre-registered).

For each country-year: sigma = sum labour income / sum gross output; adv_rank = 1 - share of commodity bases
better than labour (MAWD); adv_med / adv_best = MAWD(median / best basis) - MAWD(labour).
Regressions: adv ~ sigma + country FE + year FE (SE clustered by country) and between-country regression.
Industry groups: industries split at the country-year median of labour income per unit of output;
labour's rank computed within each half. Settings: open (primary) and closed with uniform basket.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from ..economy import build
from ..levels import mask_for
from ..metrics import ratio_metrics

ROOT = Path(__file__).resolve().parents[3]
TAB = ROOT / "results" / "tables"
OUT = ROOT / "results" / "v2"


def load_z():
    fr = []
    for tag in ("main", "robust"):
        d = pd.read_parquet(TAB / f"ratios_long_{tag}.parquet",
                            filters=[("capital", "==", True), ("imports", "==", "price"), ("labour", "==", "hours")])
        d = d[d.closed.isin(["False", "uniform"]) & (d.basis.str.startswith("k:") | (d.basis == "labour"))]
        fr.append(d[d.year <= 2022])
    return pd.concat(fr, ignore_index=True)


def run():
    z = load_z()
    rows = []
    for (c, y), g in z.groupby(["country", "year"]):
        e, _ = build(c, y)
        sigma = e.labour_income.sum() / e.x.sum()
        li_share = pd.Series(e.labour_income / e.x, index=e.labels)
        for closed, gs in g.groupby("closed"):
            labs = gs[gs.basis == "labour"].industry.tolist()
            base_mask = mask_for(labs, set())
            med = np.median(li_share[np.array(labs)[base_mask]])
            halves = {"all": base_mask,
                      "labour_intensive": base_mask & (li_share.reindex(labs).to_numpy() >= med),
                      "capital_intensive": base_mask & (li_share.reindex(labs).to_numpy() < med)}
            for hname, m in halves.items():
                res = {}
                for b, gb in gs.groupby("basis"):
                    gb = gb.set_index("industry").reindex(labs)
                    res[b] = ratio_metrics(gb.z.to_numpy(), gb.x.to_numpy(), m)["mawd"]
                L = res.pop("labour")
                vals = np.array(list(res.values()))
                rows.append(dict(country=c, year=y, setting="open" if closed == "False" else "closed_uniform",
                                 group=hname, sigma=sigma, labour_mawd=L,
                                 adv_rank=float(1 - (vals < L).mean()),
                                 adv_med=float(np.median(vals) - L), adv_best=float(vals.min() - L)))
        print("gradient", c, y, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "gradient_panel.csv", index=False)
    reg = []
    for setting in ("open", "closed_uniform"):
        d = df[(df.setting == setting) & (df.group == "all")]
        for yv in ("adv_rank", "adv_med", "adv_best"):
            fe = smf.ols(f"{yv} ~ sigma + C(country) + C(year)", d).fit(cov_type="cluster",
                                                                      cov_kwds={"groups": pd.factorize(d.country)[0]})
            bm = d.groupby("country")[[yv, "sigma"]].mean()
            be = smf.ols(f"{yv} ~ sigma", bm).fit(cov_type="HC1")
            iqr = d.sigma.quantile(0.75) - d.sigma.quantile(0.25)
            reg.append(dict(setting=setting, y=yv, beta_fe=fe.params["sigma"], se_fe=fe.bse["sigma"],
                            p_fe=fe.pvalues["sigma"], beta_between=be.params["sigma"], se_between=be.bse["sigma"],
                            p_between=be.pvalues["sigma"], iqr_sigma=iqr, effect_iqr_fe=fe.params["sigma"] * iqr,
                            effect_iqr_between=be.params["sigma"] * iqr, n=len(d)))
    pd.DataFrame(reg).to_csv(OUT / "gradient_regressions.csv", index=False)
    return df


if __name__ == "__main__":
    run()
