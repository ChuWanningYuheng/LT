"""Multi-regional variant: imports valued at the labour (or commodity) content of the exporting
country, using the full FIGARO world table (A-MRIO).

Labour input for every country-industry = persons employed (OECD TiMBC, 17 ICIO aggregates split
to FIGARO industries by the country's gross output). Persons, not hours, are used everywhere so that
the world labour vector is in one unit. Countries without TiMBC data (and the FIGARO rest of the world)
get the output-weighted average persons-per-output coefficients of the non-OECD FIGARO countries with
data (BR, CN, IN, ID, AR, RU, SA), industry by industry.

Outputs results/tables/mrio_ratios.parquet with, for each focus country and year, the ratios z for
  labour_mrio            persons, world Leontief inverse
  labour_nat_price       persons, national table, imports at price     (comparison)
  labour_nat_comp        persons, national table, competitive imports  (comparison)
and commodity bases (electricity, refined oil, mining, basic metals) computed in the same world system.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import figaro, oecd
from .core import Economy, content, leontief_inverse
from .economy import MERGES, _group_matrix, national_blocks
from .figaro import INDUSTRIES

ROOT = Path(__file__).resolve().parents[2]
TAB = ROOT / "results" / "tables"
PROXY_POOL = ["BR", "CN", "IN", "ID", "AR", "RU", "SA"]
MRIO_BASES = {"electricity": "D35", "oil_refined": "C19", "mining_energy": "B", "metals": "C24"}


def world_labour(year: int, d: dict) -> np.ndarray:
    C = list(d["countries"])
    N = len(INDUSTRIES)
    x = d["Z"].sum(1) + d["F"].sum(1)
    persons = np.full(len(x), np.nan)
    for i, c2 in enumerate(C):
        c3 = oecd.ISO2_TO_3.get(c2)
        if c3 is None:
            continue
        sl = slice(i * N, (i + 1) * N)
        p = oecd.employment_persons_timbc(c3, min(max(year, 2008), 2022), pd.Series(x[sl], index=INDUSTRIES))
        if p.notna().sum() > 0:
            persons[sl] = p.fillna(0).to_numpy() * 1000.0     # TiMBC in thousands
    lcoef = np.divide(persons, x, out=np.full(len(x), np.nan), where=x > 0)
    pool = [C.index(c) for c in PROXY_POOL if c in C]
    L = np.array([lcoef[p * N:(p + 1) * N] for p in pool])
    X = np.array([x[p * N:(p + 1) * N] for p in pool])
    avg = np.nansum(L * X, 0) / np.nansum(np.where(np.isnan(L), 0, X), 0)
    for i, c2 in enumerate(C):
        sl = slice(i * N, (i + 1) * N)
        if np.isnan(lcoef[sl]).all():
            lcoef[sl] = avg
    return np.nan_to_num(lcoef), x


def run(focus=("USA", "DEU", "MEX"), years=range(2010, 2023)):
    rows = []
    for y in years:
        d = figaro.load(y)
        C = list(d["countries"])
        N = len(INDUSTRIES)
        lw, x = world_labour(y, d)
        with np.errstate(divide="ignore", invalid="ignore"):
            A = np.where(x[None, :] > 0, d["Z"] / x[None, :], 0.0)
        Lw = leontief_inverse(A, check=False)
        assert (Lw > -1e-8).all()
        v_world = lw @ Lw
        kvals = {}
        for bname, code in MRIO_BASES.items():
            k = INDUSTRIES.index(code)
            idx = np.arange(len(C)) * N + k
            Ak = A.copy()
            Ak[idx, :] = 0.0
            kvals[bname] = A[idx, :].sum(0) @ leontief_inverse(Ak, check=False)
        for c3 in focus:
            c2 = oecd.ISO3_TO_2[c3]
            ci = C.index(c2)
            sl = slice(ci * N, (ci + 1) * N)
            b = national_blocks(c2, y)
            names, G = _group_matrix(INDUSTRIES, MERGES.get(c2, []))
            xg = G @ b["x"]
            keep = xg > 0
            agg = lambda v: (G @ (v * b["x"]))[keep] / xg[keep]   # output-weighted per-unit aggregation
            labs = [n for n, k_ in zip(names, keep) if k_]
            def emit(basis, zraw):
                z = zraw * xg[keep].sum() / (zraw @ xg[keep])
                rows.append(pd.DataFrame(dict(country=c3, year=y, basis=basis, industry=labs, z=z, x=xg[keep])))
            emit("labour_mrio", agg(v_world[sl]))
            for bname, kv in kvals.items():
                emit(f"{bname}_mrio", agg(kv[sl]))
            # national comparison with the SAME persons vector
            with np.errstate(divide="ignore", invalid="ignore"):
                Ad = np.where(xg > 0, (G @ b["Zdom"] @ G.T) / xg, 0)[np.ix_(keep, keep)]
                Am = np.where(xg > 0, (G @ b["Zimp"] @ G.T) / xg, 0)[np.ix_(keep, keep)]
            lnat = (G @ (lw[sl] * b["x"]))[keep] / xg[keep]
            vd = lnat @ leontief_inverse(Ad)
            mu = Am.sum(0) @ leontief_inverse(Ad)
            X = xg[keep]
            e = (X.sum() - mu @ X) / (vd @ X)
            emit("labour_nat_price", e * vd + mu)
            emit("labour_nat_comp", lnat @ leontief_inverse(Ad + Am))
            for bname, code in MRIO_BASES.items():
                kk = next(i for i, n in enumerate(labs) if code in n.split("+"))
                M = Ad + Am
                emit(f"{bname}_nat_comp", content(M, M[kk], zero_row=kk))
        print("mrio", y, flush=True)
    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(TAB / "mrio_ratios.parquet")
    return out


if __name__ == "__main__":
    run()
