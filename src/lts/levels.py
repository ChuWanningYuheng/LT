"""Cross-section computations for every country-year and specification.

Produces
  results/tables/ratios_long.parquet  : z_j (value/price ratio) for every basis/spec/industry/year
  results/tables/metrics_long.csv     : deviation metrics by basis/spec/subset/year
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from .core import Economy
from .economy import (BASES, EXCLUDE_BASE, EXCLUDE_GOV, EXCLUDE_RENT_FIN_MINING, build,
                      education_weights)
from .metrics import ratio_metrics

ROOT = Path(__file__).resolve().parents[2]
TAB = ROOT / "results" / "tables"

SUBSETS = {
    "all": set(),
    "no_rent_fin_mining": EXCLUDE_RENT_FIN_MINING,
    "no_gov": EXCLUDE_GOV,
    "core": EXCLUDE_RENT_FIN_MINING | EXCLUDE_GOV,
}
N_PLACEBO = 200


def roles(label: str, role_map: dict | None) -> list[str]:
    parts = label.split("+")
    if role_map is None:
        return parts
    return [role_map.get(label, role_map.get(p, p)) for p in parts]


def mask_for(labels: list[str], excl: set[str], role_map: dict | None = None) -> np.ndarray:
    bad = EXCLUDE_BASE | excl
    return np.array([not any(m in bad for m in roles(lab, role_map)) for lab in labels])


def labour_vectors(e: Economy, country3: str, year: int, label_divs=None) -> dict[str, np.ndarray]:
    """Direct labour coefficients under the three reduction schemes (A-RED)."""
    h = e.l()
    li = e.labour_income
    wbar = li.sum() / e.hours.sum()
    wage_rel = np.divide(li / e.hours, wbar, out=np.ones(e.n), where=e.hours > 0)
    out = {"hours": h, "wage_weighted": h * wage_rel}
    for sch in ("train", "years"):
        out[f"edu_{sch}"] = h * education_weights(country3, year, e.labels, sch, label_divs)
    return out


def ratios_for(e: Economy, country3: str, year: int, rng: np.random.Generator, label_divs=None) -> pd.DataFrame:
    rows, capped = [], []
    lv = labour_vectors(e, country3, year, label_divs)
    idx = {lab: i for i, lab in enumerate(e.labels)}

    def add(basis, spec, z):
        rows.append(pd.DataFrame({"basis": basis, "industry": e.labels, "z": z, "x": e.x,
                                  **spec}))

    for capital, closed, imports, lname in itertools.product((False, True), (False, True, "uniform"),
                                                             ("price", "competitive"), lv):
        if closed == "uniform" and lname != "hours":
            continue  # labour basis identical to the open one; commodity bases independent of l
        l = lv[lname]
        spec = dict(capital=capital, closed=closed, imports=imports, labour=lname)
        add("labour", spec, e.basis_ratios("L", capital, closed, imports, l=l))
        # Commodity bases: every industry k in turn ("k:<code>"; named bases in BASES are aliases).
        # They depend on the labour vector in neither system (closed system: labour goods are
        # reproduced at actual wages), so they are computed once, under labour='hours'.
        if lname == "hours":
            for k, lab in enumerate(e.labels):
                if e.A[k].sum() + (e.dep_coeffs()[0][k].sum() if capital else 0) <= 0:
                    continue  # not used as an input (e.g. households as employers): no content
                add(f"k:{lab}", spec, e.basis_ratios(k, capital, closed, imports, l=l))
        if not closed:
            # A-RMAX: if the observed profit rate exceeds the maximum rate of the chosen system,
            # prices of production do not exist at r_act; use 0.99 R and record it.
            r_act = e.actual_profit_rate(capital)
            R = e.max_profit_rate(capital, imports)
            if r_act >= 0.99 * R:
                capped.append(dict(capital=capital, imports=imports, r_act=r_act, R=R))
                r_act = 0.99 * R
            add("pp_uniform_wage", spec, e.prices_of_production(r_act, capital, imports, "uniform", l=l))
            if lname == "hours":
                add("pp_actual_wage", spec, e.prices_of_production(r_act, capital, imports, "actual", l=l))
    # Marx/Shaikh eigen prices (wages advanced, competitive imports)
    for capital in (False, True):
        r_star, p = e.closed_eigen_prices(capital)
        add("pp_eigen", dict(capital=capital, closed=True, imports="competitive", labour="hours"), p)
    # placebo primary inputs (open system, imports at price): random coefficients with the same
    # log-dispersion as direct hours per unit of output, and permuted hours coefficients (A-PLAC)
    # Placebo draws are FIXED over time for a country (seeded by country), so that their dynamics are
    # driven only by changes in the IO structure (random coefficient per currency unit of output) or by
    # the hours coefficients of the wrong industries (fixed permutation) -- a valid dynamic placebo.
    l = lv["hours"]
    pos = l > 0
    sd = np.log(l[pos]).std()
    prng = np.random.default_rng(sum(map(ord, country3)) * 7919)
    P = np.exp(prng.normal(0, 1, (N_PLACEBO, e.n)) * sd)
    perms = [prng.permutation(e.n) for _ in range(50)]
    for i in range(N_PLACEBO):
        for capital in (False, True):
            spec = dict(capital=capital, closed=False, imports="price", labour="hours")
            add(f"placebo_random_{i:03d}", spec, e.basis_ratios("L", capital, False, "price", l=P[i]))
            if i < 50:
                add(f"placebo_perm_{i:03d}", spec, e.basis_ratios("L", capital, False, "price", l=l[perms[i]]))
    df = pd.concat(rows, ignore_index=True)
    df["country"], df["year"] = country3, year
    df.attrs["r_capped"] = capped
    return df


def metrics_from_ratios(df: pd.DataFrame, sizes: dict, role_map: dict | None = None) -> pd.DataFrame:
    out = []
    keys = ["country", "year", "basis", "capital", "closed", "imports", "labour"]
    for k, g in df.groupby(keys, sort=False):
        labels = g["industry"].tolist()
        size = sizes[(k[0], k[1])]
        for sname, excl in SUBSETS.items():
            m = ratio_metrics(g["z"].to_numpy(), g["x"].to_numpy(), mask_for(labels, excl, role_map), size=size)
            out.append({**dict(zip(keys, k)), "subset": sname, **m})
    return pd.DataFrame(out)


def run(countries=("USA", "DEU", "MEX"), years=range(2010, 2024), seed=12345, tag="main", source="figaro"):
    TAB.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    frames, infos, sizes, extra = [], [], {}, []
    role_map, label_divs = None, None
    if source == "bea":
        from . import bea
        from .codes import BEA_TO_NACE
        role_map = bea.BEA_ROLE
    for c in countries:
        for y in years:
            try:
                if source == "bea":
                    e, info = bea.build(y)
                    label_divs = {lab: set().union(*[set(BEA_TO_NACE[m]) for m in lab.split("+")]) for lab in e.labels}
                else:
                    e, info = build(c, y)
            except Exception as ex:  # missing year for a country -> documented, skipped
                infos.append(dict(country=c, year=y, error=repr(ex)))
                continue
            infos.append(info)
            fr = ratios_for(e, "USA" if source == "bea" else c, y, rng, label_divs)
            fr["country"] = c
            info["r_capped"] = fr.attrs.get("r_capped")
            frames.append(fr)
            M, Mm, _ = e.system(True, False, "price", e.l())
            sizes[(c, y)] = (M.sum(0) + Mm.sum(0)) * e.x   # non-labour input cost (Kliman-type size)
            rates = e.k_exploitation_rates(capital=False)
            r_star, _ = e.closed_eigen_prices(capital=True)
            extra.append(dict(country=c, year=y, r_actual_capital=e.actual_profit_rate(True),
                              r_actual_nocap=e.actual_profit_rate(False), R_max_capital=e.max_profit_rate(True),
                              r_eigen_capital=r_star,
                              uniform_wage_scale=e.uniform_wage_scale(*[m if i == 0 else m for i, m in enumerate(e.system(True, False, "competitive", e.l())[:1])], None),
                              **{f"exploit_{k}": v for k, v in rates.items()},
                              price_index=list(e.meta["price_index"]), labels=list(e.labels)))
            print(c, y, "ok", flush=True)
    df = pd.concat(frames, ignore_index=True)
    df["closed"] = df["closed"].astype(str)   # False / True / uniform
    df.to_parquet(TAB / f"ratios_long_{tag}.parquet")
    met = metrics_from_ratios(df, sizes, role_map)
    met.to_csv(TAB / f"metrics_long_{tag}.csv", index=False)
    pd.DataFrame(infos).to_json(TAB / f"build_info_{tag}.json", orient="records", indent=1)
    pd.DataFrame(extra).to_json(TAB / f"economy_extra_{tag}.json", orient="records", indent=1)
    return df, met


if __name__ == "__main__":
    import sys
    tag = sys.argv[1] if len(sys.argv) > 1 else "main"
    if tag == "bea":
        run(countries=("USA_BEA",), years=range(1998, 2024), tag="bea", source="bea")
    elif tag == "robust":
        run(countries=("FRA", "ITA", "ESP", "NLD", "AUT", "POL", "CZE", "KOR", "JPN", "GBR"),
            years=range(2010, 2023), tag="robust")
    else:
        run(tag=tag)
