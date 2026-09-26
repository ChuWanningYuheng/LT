"""Iteration 5: checks added in response to the independent review (results/v5/s5_review_checks.csv).
R1: prices of production at r vs labour / energy / flat benchmarks without simulation (random technologies).
R3: annual T2 with country effects only (no year effects).
R9: US NFC trend of r with Newey-West lags 2, 4, 8.
"""
import numpy as np
import pandas as pd

from .abm import balanced, energy_values, mawd, prod_prices, technology
from .stage1 import OUT, load, logs, nw_se, ols_fwl


def main():
    rows = []
    for s in (0.8, 0.6, 0.4, 0.2, 0.05):
        rng = np.random.default_rng(1)
        ne = nf = 0
        for _ in range(200):
            A, l, b = technology(rng, s)
            x, v = balanced(A, l, b)
            R = 1 / max(abs(np.linalg.eigvals(A))) - 1
            p = prod_prices(A, l, min(0.3, 0.9 * R))
            ne += mawd(p, v, x) < mawd(p, energy_values(A), x)
            nf += mawd(p, v, x) < mawd(p, np.ones(5), x)
        rows.append(dict(check="R1_pp_closer_to_labour_than_energy", spec=f"share {s}", value=ne, n=200))
        rows.append(dict(check="R1_pp_closer_to_labour_than_flat", spec=f"share {s}", value=nf, n=200))
    d = logs(load()).sort_values(["variant", "geo", "year"])
    g = d.groupby(["variant", "geo"])
    for c in ("lnrM", "ln1k"):
        d["d_" + c] = g[c].diff()
    d = d[g.year.diff() == 1]
    for v in ("main", "no_mi"):
        for fe in (["geo"], ["geo", "year"]):
            r = ols_fwl(d[d.variant == v], "d_lnrM", "d_ln1k", ["gap"], fe)
            rows.append(dict(check="R3_T2_annual", spec=f"{v} FE={'+'.join(fe)}", value=r["beta"], p=r["p_wild"], n=r["n"]))
    u = pd.read_csv(OUT / "s1_us_long.csv")
    y = u.r.to_numpy()
    X = np.column_stack([np.ones(len(y)), np.arange(len(y))])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    from scipy import stats
    for lag in (2, 4, 8):
        se = nw_se(X, y - X @ b, lag)[1]
        rows.append(dict(check="R9_US_trend_r", spec=f"NW lag {lag}", value=b[1], p=2 * stats.t.sf(abs(b[1] / se), len(y) - 2), n=len(y)))
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "s5_review_checks.csv", index=False)
    print(out.to_string())


if __name__ == "__main__":
    main()
