"""Iteration 10, stage 10: land rent - deviation of prices from prices of production in rent industries
(pre_registration_v10.md, stage 10; journal 10).

Outcome 60: mean d = ln(p / pp) of A01, B, L (FIGARO 13 countries 2010-2022), cluster bootstrap by country; placebo of
3 random non-rent industries. Outcome 61: elasticity of d(B) w.r.t. the real energy + metals price index (country FE).
Descriptive: BEA (111CA, 211, 212; HS+ORE merged), agriculture vs agricultural prices, L vs BIS real house prices,
mining d vs World Bank natural resource rents.

Usage: PYTHONPATH=src python -P -m lts.v10.stage10
Outputs: results/v10/s10_*.csv
"""
from __future__ import annotations

import json
import zipfile

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from ..v6.series import ROOT
from ..v8.rule import decide
from ..v9b.stage45 import wild_t, widest

OUT = ROOT / "results" / "v10"
TAB = ROOT / "results" / "tables"
R10 = ROOT / "data" / "raw" / "v10"
RENT = ["A01", "B", "L"]
RENT_BEA = ["111CA", "211", "212"]
ISO2 = {"USA": "US", "DEU": "DE", "MEX": "MX", "FRA": "FR", "ITA": "IT", "ESP": "ES", "NLD": "NL", "AUT": "AT",
        "POL": "PL", "CZE": "CZ", "KOR": "KR", "JPN": "JP", "GBR": "GB"}
YEARS = range(2010, 2023)


def d_frame(tag, basis):
    cols = ["basis", "industry", "z", "x", "capital", "closed", "imports", "labour", "country", "year"]
    d = pd.read_parquet(TAB / f"ratios_long_{tag}.parquet", columns=cols)
    d = d[(d.basis == basis) & (d.capital == True) & (d.closed.astype(str) == "False") &   # noqa: E712
          (d.imports == "price") & (d.labour == "hours")].copy()
    d = d[~d.industry.isin(["T", "U"]) & (d.x > 0) & (d.z > 0)]
    s = d.groupby(["country", "year"]).apply(lambda g: (g.z * g.x).sum() / g.x.sum(), include_groups=False)
    d["zn"] = d.z / d.set_index(["country", "year"]).index.map(s)
    d["d"] = -np.log(d.zn)
    return d[["country", "year", "industry", "x", "d"]]


def figaro_d(basis):
    d = pd.concat([d_frame("main", basis), d_frame("robust", basis)], ignore_index=True)
    return d[d.year.isin(list(YEARS))]


def stat(D, inds):
    m = D[D.industry.isin(inds)].groupby(["country", "year"]).d.mean()
    return m.groupby(level=0).mean()


def boot_countries(v, B=2000, seed=60):
    v = np.asarray(v, float)
    rng = np.random.default_rng(seed)
    bs = v[rng.integers(0, len(v), (B, len(v)))].mean(1)
    return np.quantile(bs, [0.05, 0.95])


def commodity_index():
    x = pd.read_excel(R10 / "wb_CMO_Historical_Data_Monthly.xlsx", sheet_name="Monthly Indices", header=None)
    x = x.iloc[9:, [0, 2, 4, 14]]
    x.columns = ["m", "energy", "agriculture", "metals"]
    x = x[x.m.astype(str).str.match(r"^\d{4}M\d{2}$")]
    x["year"] = x.m.str[:4].astype(int)
    for c in ("energy", "agriculture", "metals"):
        x[c] = pd.to_numeric(x[c], errors="coerce")
    a = x.groupby("year")[["energy", "agriculture", "metals"]].mean()
    n = x.groupby("year").m.count()
    a = a[n == 12]
    c = pd.read_csv(ROOT / "data" / "raw" / "v9" / "fred_CPIAUCSL.csv")
    c["year"] = pd.to_datetime(c.iloc[:, 0]).dt.year
    cpi = c.groupby("year").CPIAUCSL.mean()
    real = a.div(cpi, axis=0).dropna()
    out = pd.DataFrame({"ln_em": 0.5 * (np.log(real.energy) + np.log(real.metals)), "ln_agr": np.log(real.agriculture)})
    return out


def bis_real():
    z = zipfile.ZipFile(R10 / "bis_WS_SPP_csv_flat.zip")
    d = pd.read_csv(z.open(z.namelist()[0]), low_memory=False)
    d = d[d["VALUE:Value"].str.startswith("R") & d["UNIT_MEASURE:Unit of measure"].str.startswith("628")]
    d["geo"] = d["REF_AREA:Reference area"].str[:2]
    d["year"] = d["TIME_PERIOD:Time period or range"].str[:4].astype(int)
    return d.groupby(["geo", "year"])["OBS_VALUE:Observation Value"].mean().rename("hp_real")


def wb_rents():
    w = json.load(open(R10 / "wb_NY.GDP.TOTL.RT.ZS.json"))[1]
    d = pd.DataFrame([dict(iso3=r["countryiso3code"], year=int(r["date"]), v=r["value"]) for r in w if r["value"] is not None])
    return d


def fe_wild(df, y, x, seed):
    df = df.dropna(subset=[y, x])
    X = (df[x] - df.groupby("country")[x].transform("mean")).to_numpy()[:, None]
    Y = (df[y] - df.groupby("country")[y].transform("mean")).to_numpy()
    b, se, lo_b, hi_b, p, G = wild_t(X, Y, df.country.to_numpy(), 0, B=9999, seed=seed)
    lo, hi, lab = widest(b, se, G, lo_b, hi_b, 0.05, ">")
    return dict(est=b, se=se, ci90_lo=lo, ci90_hi=hi, n=len(df), G=G, label=lab)


def run():
    rows, out = [], []
    for basis in ("pp_actual_wage", "pp_uniform_wage"):
        D = figaro_d(basis)
        s = stat(D, RENT)
        lo, hi = boot_countries(s.to_numpy())
        est = float(s.mean())
        per = D[D.industry.isin(RENT)].groupby(["country", "industry"]).d.mean().unstack()
        per.to_csv(OUT / f"s10_rent_d_{basis}.csv")
        # placebo: 1000 random sets of 3 non-rent industries, labels present in every country-year
        common = set.intersection(*[set(g.industry) for _, g in D.groupby(["country", "year"])])
        pool = sorted(common - set(RENT))
        rng = np.random.default_rng(600)
        pl = np.array([stat(D, list(rng.choice(pool, 3, replace=False))).mean() for _ in range(1000)])
        pct = float(np.mean(pl < est))
        r = dict(outcome="60: mean d = ln(p/pp) of rent industries A01, B, L (FIGARO, 2010-2022)" +
                 ("" if basis == "pp_actual_wage" else " [variant pp_uniform_wage]"),
                 est=est, ci90_lo=lo, ci90_hi=hi, n=len(s), theta0=0, delta=0.05, direction=">",
                 label=decide(est, lo, hi, 0, 0.05, ">"), placebo_percentile=pct, placebo_median=float(np.median(pl)),
                 placebo_p95=float(np.quantile(pl, 0.95)), pool=len(pool),
                 d_A01=float(per["A01"].mean()), d_B=float(per["B"].mean()), d_L=float(per["L"].mean()))
        out.append(r)
        if basis == "pp_actual_wage":
            Dmain = D
    # ---- outcome 61
    ci = commodity_index()
    ci.to_csv(OUT / "s10_commodity_index.csv")
    B = Dmain[Dmain.industry == "B"][["country", "year", "d"]].merge(ci, left_on="year", right_index=True)
    r61 = fe_wild(B, "d", "ln_em", 61)
    out.append(dict(outcome="61: elasticity of d(B) w.r.t. ln real energy+metals index (FIGARO, country FE)", theta0=0,
                    delta=0.05, direction=">", **r61))
    A = Dmain[Dmain.industry == "A01"][["country", "year", "d"]].merge(ci, left_on="year", right_index=True)
    ra = fe_wild(A, "d", "ln_agr", 62)
    rows.append(dict(item="d(A01) on ln real agriculture index, country FE (descriptive)", **ra))
    hp = bis_real().reset_index()
    L = Dmain[Dmain.industry == "L"][["country", "year", "d"]].copy()
    L["geo"] = L.country.map(ISO2)
    L = L.merge(hp, on=["geo", "year"], how="left")
    L["ln_hp"] = np.log(L.hp_real)
    rl = fe_wild(L, "d", "ln_hp", 63)
    rows.append(dict(item="d(L) on ln BIS real house price index, country FE (descriptive)", **rl))
    wr = wb_rents()
    wr = wr[wr.year.isin(list(YEARS))].groupby("iso3").v.mean()
    dB = B.groupby("country").d.mean()
    j = pd.concat([dB, wr.reindex(dB.index)], axis=1, keys=["d_B", "rents"]).dropna()
    rho, p = stats.spearmanr(j.d_B, j.rents)
    rows.append(dict(item="Spearman: country mean d(B) vs WB natural resource rents % GDP (descriptive)", est=rho, p=p,
                     n=len(j)))
    # ---- BEA (descriptive)
    Db = d_frame("bea", "pp_actual_wage")
    pb = Db[Db.industry.isin(RENT_BEA + ["HS+ORE"])].pivot_table(index="year", columns="industry", values="d")
    pb.to_csv(OUT / "s10_bea_d.csv")
    rows.append(dict(item="BEA 1998-2023: mean d of 111CA, 211, 212 (descriptive)", est=float(pb[RENT_BEA].mean(axis=1).mean()),
                     n=len(pb)))
    rows.append(dict(item="BEA 1998-2023: mean d of HS+ORE (housing merged with other real estate)",
                     est=float(pb["HS+ORE"].mean()), n=len(pb)))
    m = pb[["211", "212"]].mean(axis=1).rename("d").to_frame().join(ci, how="inner")
    f = sm.OLS(m.d, sm.add_constant(m.ln_em)).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
    fo = sm.OLS(m.d, sm.add_constant(m.ln_em)).fit()
    b, se = f.params.ln_em, f.bse.ln_em
    w = max(1.645 * se, stats.t.ppf(0.95, len(m) - 2) * fo.bse.ln_em)          # wider of HAC and t(n-2)
    rows.append(dict(item="61 (BEA variant): mean d(211, 212) on ln real energy+metals index, HAC", est=b, se=se,
                     ci90_lo=b - w, ci90_hi=b + w, n=len(m), label=decide(b, b - w, b + w, 0, 0.05, ">")))
    O = pd.DataFrame(out)
    O.to_csv(OUT / "s10_outcomes.csv", index=False)
    Rd = pd.DataFrame(rows)
    Rd.to_csv(OUT / "s10_descriptive.csv", index=False)
    pd.set_option("display.width", 250)
    print(O.round(4).to_string(), "\n", Rd.round(4).to_string(), flush=True)


if __name__ == "__main__":
    run()
