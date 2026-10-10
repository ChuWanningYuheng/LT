"""Iteration 12, stage 5 (exploratory): does policy respond to falling profitability? (pre_registration_v12.md, stage 5)

Profit rate r: AMECO (iteration 6 construction, lts.v6.series.derive/finish, variant main) for every AMECO country
where it can be built. Policies: combined statutory CIT rate (OECD CIT_C, standard), EPL_OV (version 1),
collective bargaining coverage (ERB) and union density (TUD). Model: d policy (t -> t+3) on d r (t-5 -> t, pp),
country and year effects, clusters by country. Labels descriptive (delta = 0.1 SD of the policy change per pp of r).

Usage: PYTHONPATH=src python -P -m lts.v12.stage5
Outputs: results/v12/s5_*.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..v6.series import ROOT, ameco_country, ameco_raw, derive, finish
from ..v8.rule import decide
from ..v11.stage1 import reg

OUT = ROOT / "results" / "v12"
R12 = ROOT / "data" / "raw" / "v12"
R9 = ROOT / "data" / "raw" / "v9"


def profit_rates():
    raw = ameco_raw()
    rows = []
    for geo in sorted(raw.geo.unique()):
        if len(geo) != 3 or not geo.isalpha():
            continue
        try:
            d = finish(derive(ameco_country(raw, geo), 1.0))
        except Exception:                                             # noqa: BLE001
            continue
        d = d[d.PI.notna() & d.K.notna() & (d.K > 0)]
        if len(d) < 15:
            continue
        rows.append(pd.DataFrame({"country": geo, "year": d.index.astype(int), "r": (100 * d.r).to_numpy()}))
    return pd.concat(rows, ignore_index=True)


def policies():
    c = pd.read_csv(R12 / "oecd_cit.csv", low_memory=False)
    c = c[(c.MEASURE == "CIT_C") & (c.TARGETING == "ST")]
    cit = c.groupby(["REF_AREA", "TIME_PERIOD"]).OBS_VALUE.mean().rename("cit")
    e = pd.read_csv(R12 / "oecd_epl.csv", low_memory=False)
    e = e[(e.MEASURE == "EPL_OV") & (e.VERSION == "VERSION1")]
    epl = e.groupby(["REF_AREA", "TIME_PERIOD"]).OBS_VALUE.mean().rename("epl")
    out = [cit, epl]
    for f, nm in (("oecd_cbc.csv", "cbc"), ("oecd_tud.csv", "tud")):
        d = pd.read_csv(R9 / f, low_memory=False)
        out.append(d.groupby(["REF_AREA", "TIME_PERIOD"]).OBS_VALUE.mean().rename(nm))
    P = pd.concat(out, axis=1)
    P.index.names = ["country", "year"]
    return P.reset_index()


def run():
    R = profit_rates()
    P = policies()
    d = R.merge(P, on=["country", "year"], how="outer").sort_values(["country", "year"])
    k = d.set_index(["country", "year"])
    d["dr5"] = d.r - k.r.reindex(pd.MultiIndex.from_arrays([d.country, d.year - 5])).to_numpy()
    rows = []
    for pol in ("cit", "epl", "cbc", "tud"):
        d[f"d3_{pol}"] = k[pol].reindex(pd.MultiIndex.from_arrays([d.country, d.year + 3])).to_numpy() - d[pol]
        s = d.dropna(subset=[f"d3_{pol}", "dr5"]).copy()
        s["yr"] = s.year.astype(str)
        if s.country.nunique() < 5:
            continue
        r = reg(s, f"d3_{pol}", ["dr5"], 0, fe=("country", "yr"), cl="country")
        delta = 0.1 * float(s[f"d3_{pol}"].std())
        rows.append(dict(policy=pol, **r, years=f"{s.year.min()}-{s.year.max()}", countries=int(s.country.nunique()),
                         delta=delta, direction=">", label=decide(r["est"], r["ci90_lo"], r["ci90_hi"], 0, delta, ">")))
    O = pd.DataFrame(rows)
    O.to_csv(OUT / "s5_exploratory.csv", index=False)
    d.to_csv(OUT / "s5_panel.csv", index=False)
    pd.set_option("display.width", 250)
    print(O.round(4).to_string())


if __name__ == "__main__":
    run()
