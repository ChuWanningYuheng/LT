"""Independent re-implementation of the symmetric value-basis test (SPEC.md).

Run:  OMP_NUM_THREADS=2 python independent/symtest.py
Writes independent/results/{independent_shares,checks,figaro_rebuild_check}.csv
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PRIM = ROOT / "data" / "primitives"
FIG = ROOT / "data" / "processed" / "figaro"
OUT = Path(__file__).resolve().parent / "results"

COUNTRIES = ["USA", "DEU", "MEX"]
YEARS = range(2010, 2024)
SYSTEMS = ["open", "closed_actual", "closed_uniform"]


def load(country, year):
    d = np.load(PRIM / f"{country}_{year}.npz", allow_pickle=True)
    return {k: d[k] for k in d.files}


def build(p, capital, system):
    """Return (M, Mm) of the system whose commodity bases are tested."""
    x = p["x"].astype(float)
    A = p["Zdom"] / x[None, :]
    Am = p["Zimp"] / x[None, :]
    if capital:
        dvec = p["cfc"] / x
        G = p["gfcf_dom"].sum() + p["gfcf_imp"].sum()
        M = A + np.outer(p["gfcf_dom"] / G, dvec)
        Mm = Am + np.outer(p["gfcf_imp"] / G, dvec)
    else:
        M, Mm = A.copy(), Am.copy()
    if system == "open":
        return M, Mm, M, Mm
    C = p["hh_dom"].sum() + p["hh_imp"].sum()
    h = p["hours"] / x
    if system == "closed_actual":
        omega = p["labour_income"] / x
    else:
        omega = (p["labour_income"].sum() / p["hours"].sum()) * h
    Mc = M + np.outer(p["hh_dom"] / C, omega)
    Mcm = Mm + np.outer(p["hh_imp"] / C, omega)
    return Mc, Mcm, M, Mm


def z_of(a, S, Sm, x):
    """z = e*lambda + mu with sum(z x) = sum(x). Returns z, Leontief inverse."""
    n = len(x)
    L = np.linalg.inv(np.eye(n) - S)
    lam = a @ L
    mu = Sm.sum(axis=0) @ L
    e = (x.sum() - mu @ x) / (lam @ x)
    return e * lam + mu, L


def metrics(z, x, evalmask):
    m = evalmask & (x > 0) & np.isfinite(z) & (z > 0)
    zz, xx = z[m], x[m]
    zz = zz * xx.sum() / (zz * xx).sum()
    w = xx / xx.sum()
    q = 1.0 / zz
    mawd = float((w * np.abs(q - 1)).sum())
    cos = q.sum() / (np.sqrt(len(q)) * np.linalg.norm(q))
    dd = float(np.sqrt(max(2.0 * (1.0 - cos), 0.0)))
    return mawd, dd


def expanded_labour_z(p, M, Mm, omega, x):
    """Labour basis in the 'unfolded' 2n system: n goods + n labour-power
    commodities; labour rows primary (zeroed in S). Returns z for goods."""
    n = len(x)
    h = p["hours"] / x
    C = p["hh_dom"].sum() + p["hh_imp"].sum()
    # labour-power j per hour needs wage goods beta * (omega_j / h_j)
    B = np.outer(p["hh_dom"] / C, omega / h)
    Bm = np.outer(p["hh_imp"] / C, omega / h)
    S = np.zeros((2 * n, 2 * n))
    S[:n, :n] = M
    S[:n, n:] = B
    # rows n: (labour inputs, H = diag(h)) are primary -> zeroed in S
    Sm = np.zeros((2 * n, 2 * n))
    Sm[:n, :n] = Mm
    Sm[:n, n:] = Bm
    a = np.concatenate([h, np.zeros(n)])  # hours of labour used directly
    L = np.linalg.inv(np.eye(2 * n) - S)
    lam = (a @ L)[:n]
    mu = (Sm.sum(axis=0) @ L)[:n]
    e = (x.sum() - mu @ x) / (lam @ x)
    return e * lam + mu


def run_one(country, year):
    p = load(country, year)
    x = p["x"].astype(float)
    labels = [str(s) for s in p["labels"]]
    evalmask = np.array([not ({"T", "U"} & set(l.split("+"))) for l in labels])
    h = p["hours"] / x
    n = len(x)
    rows, checks = [], []
    for capital in (True, False):
        for system in SYSTEMS:
            S0, S0m, M, Mm = build(p, capital, system)
            # labour basis: a = h, S = M, imports by Mm (same in all systems)
            zl, Ll = z_of(h, M, Mm, x)
            lab_m, lab_d = metrics(zl, x, evalmask)
            radii = [np.max(np.abs(np.linalg.eigvals(M)))]
            minL = [Ll.min()]
            mindiag = [np.diag(Ll).min()]
            maxrow = 0.0
            if system == "open":
                labour_diff = np.nan
            else:
                omega = (p["labour_income"] / x if system == "closed_actual"
                         else (p["labour_income"].sum() / p["hours"].sum()) * h)
                ze = expanded_labour_z(p, M, Mm, omega, x)
                labour_diff = float(np.max(np.abs(ze - zl)))
            bm, bd = [], []
            # BASESET=open: same basis set in every system (rows positive in the OPEN matrix M),
            # added after comparison with iteration 1 (households-as-employers T becomes a basis only
            # in closed systems otherwise).
            ks = np.where((M if os.environ.get("BASESET") == "open" else S0).sum(axis=1) > 0)[0]
            for k in ks:
                a = S0[k, :].copy()
                S = S0.copy()
                S[k, :] = 0.0
                maxrow = max(maxrow, float(np.abs(S[k, :]).max()))
                z, L = z_of(a, S, S0m, x)
                radii.append(np.max(np.abs(np.linalg.eigvals(S))))
                minL.append(L.min())
                mindiag.append(np.diag(L).min())
                mk, dk = metrics(z, x, evalmask)
                bm.append(mk)
                bd.append(dk)
            bm, bd = np.array(bm), np.array(bd)
            rows.append(dict(country=country, year=year, capital=capital, system=system,
                             labour_mawd=lab_m, labour_d=lab_d,
                             share_better_mawd=float((bm < lab_m).mean()),
                             share_better_d=float((bd < lab_d).mean()),
                             n_bases=len(ks)))
            checks.append(dict(country=country, year=year, capital=capital, system=system,
                               n=n, n_bases=len(ks),
                               max_spectral_radius=float(max(radii)),
                               min_leontief_elem=float(min(minL)),
                               min_leontief_diag=float(min(mindiag)),
                               labour_closed_vs_open_max_abs_z=labour_diff,
                               max_abs_own_row_in_S=maxrow,
                               ok=bool(max(radii) < 1 and min(minL) >= 0 and min(mindiag) >= 1
                                       and maxrow == 0 and not (labour_diff > 1e-10))))
    return rows, checks


def figaro_rebuild(year=2015):
    g = np.load(FIG / f"figaro_{year}.npz", allow_pickle=True)
    Z, F = g["Z"], g["F"]
    ctry = [str(c) for c in g["countries"]]
    ind = [str(i) for i in g["industries"]]
    ni = len(ind)
    out = []
    for iso3, iso2 in (("DEU", "DE"), ("USA", "US")):
        p = load(iso3, year)
        labels = [str(s) for s in p["labels"]]
        c = ctry.index(iso2)
        cols = slice(c * ni, (c + 1) * ni)
        Zdom = Z[c * ni:(c + 1) * ni, cols]
        Ztot = Z[:, cols].reshape(len(ctry), ni, ni).sum(axis=0)
        Zimp = Ztot - Zdom
        x = Z[c * ni:(c + 1) * ni, :].sum(1) + F[c * ni:(c + 1) * ni, :].sum(1)
        # aggregation matrix industries -> primitive labels
        Agg = np.zeros((len(labels), ni))
        for r, l in enumerate(labels):
            for part in l.split("+"):
                Agg[r, ind.index(part)] = 1.0
        used = Agg.sum(0) > 0
        dropped = [ind[i] for i in np.where(~used)[0]]
        Zd = Agg @ Zdom @ Agg.T
        Zi = Agg @ Zimp @ Agg.T
        xa = Agg @ x
        for name, mine, theirs in (("Zdom", Zd, p["Zdom"]), ("Zimp", Zi, p["Zimp"]), ("x", xa, p["x"])):
            diff = np.abs(mine - theirs)
            out.append(dict(country=iso3, year=year, item=name, n=len(labels),
                            dropped_industries="+".join(dropped),
                            dropped_output=float(x[~used].sum()),
                            max_abs_diff=float(diff.max()),
                            max_rel_diff=float((diff / np.maximum(np.abs(theirs), 1e-9)).max()),
                            total_mine=float(mine.sum()), total_primitives=float(theirs.sum())))
    return pd.DataFrame(out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    allrows, allchecks = [], []
    for c in COUNTRIES:
        for y in YEARS:
            r, ch = run_one(c, y)
            allrows += r
            allchecks += ch
    shares = pd.DataFrame(allrows)
    checks = pd.DataFrame(allchecks)
    shares.to_csv(OUT / os.environ.get("OUTNAME", "independent_shares.csv"), index=False)
    checks.to_csv(OUT / "checks.csv", index=False)
    fr = figaro_rebuild(2015)
    fr.to_csv(OUT / "figaro_rebuild_check.csv", index=False)

    pd.set_option("display.width", 200)
    print(shares.groupby(["country", "capital", "system"])[
        ["labour_mawd", "labour_d", "share_better_mawd", "share_better_d", "n_bases"]].mean().round(4))
    print("\nWorst checks:")
    print(" max spectral radius:", checks.max_spectral_radius.max())
    print(" min Leontief element:", checks.min_leontief_elem.min())
    print(" min Leontief diagonal:", checks.min_leontief_diag.min())
    print(" max |z_closed_labour - z_open_labour|:", checks.labour_closed_vs_open_max_abs_z.max())
    print(" max |own row in S|:", checks.max_abs_own_row_in_S.max())
    print(" all ok:", checks.ok.all())
    print("\nFIGARO rebuild:")
    print(fr.to_string(index=False))


if __name__ == "__main__":
    main()
