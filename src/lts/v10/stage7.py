"""Iteration 10, stage 7: absolute vs relative surplus value (descriptive; pre_registration_v10.md, stage 7; journal 12).

omega = W / (P * y_h * H): W nominal compensation per employee (HWCDW), P = UVGD / OVGD, y_h = OVGD / NLHT,
H = NLHA; e = (1 - omega) / omega. Contributions to d ln e ~ -d ln omega / (1 - omega_bar):
hours (absolute), productivity (relative), real wage; residual = exact - sum. Periods: start-1973, 1973-1997,
1997-2024. US variant: Marxian e_m (9b series). Intensity (EWCS) not available.

Usage: PYTHONPATH=src python -P -m lts.v10.stage7
Outputs: results/v10/s7_*.csv
"""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd

from ..v6.series import ROOT

OUT = ROOT / "results" / "v10"
CODES = ["HWCDW", "UVGD", "OVGD", "NLHT", "NLHA"]
PERIODS = [(None, 1973), (1973, 1997), (1997, 2024)]


def ameco():
    rows = []
    for f in sorted(glob.glob(str(ROOT / "data" / "raw" / "v6" / "ameco" / "AMECO*.TXT"))):
        d = pd.read_csv(f, sep=";", dtype=str, encoding="latin1")
        p = d.CODE.str.split(".", expand=True)
        # real GDP (OVGD) is stored as XXX.1.1.0.0.OVGD (national currency, reference-year prices)
        keep = p[5].isin(CODES) & (p[1] == "1") & (p[3] == "0") & (p[4] == "0") & \
            ((p[2] == "0") | ((p[5] == "OVGD") & (p[2] == "1")))
        for _, r in d[keep].iterrows():
            c = r.CODE.split(".")
            for y in [k for k in d.columns if k.strip().isdigit()]:
                v = pd.to_numeric(r[y], errors="coerce")
                if np.isfinite(v):
                    rows.append((c[0], c[5], int(y), v))
    a = pd.DataFrame(rows, columns=["geo", "var", "year", "v"]).drop_duplicates(["geo", "var", "year"])
    w = a.pivot_table(index=["geo", "year"], columns="var", values="v").reset_index()
    return w[w.year <= 2024]


def run():
    A = ameco().dropna(subset=CODES)
    A = A[~A.geo.str.match(r"^(EU|EA|DU|DA)")]
    A["P"] = A.UVGD / A.OVGD
    A["w_real"] = A.HWCDW / A.P
    A["y_h"] = A.OVGD / A.NLHT
    A["H"] = A.NLHA
    A["omega"] = A.w_real / (A.y_h * A.H)
    A["e"] = (1 - A.omega) / A.omega
    rows = []
    for g, x in A.groupby("geo"):
        x = x.set_index("year").sort_index()
        for a, b in PERIODS:
            a_ = int(x.index.min()) if a is None else a
            if a_ not in x.index or b not in x.index or a_ >= b:
                continue
            s, t = x.loc[a_], x.loc[b]
            wbar = 0.5 * (s.omega + t.omega)
            k = 1 / (1 - wbar)
            dle = np.log(t.e / s.e)
            c_h = k * np.log(t.H / s.H)
            c_y = k * np.log(t.y_h / s.y_h)
            c_w = -k * np.log(t.w_real / s.w_real)
            rows.append(dict(geo=g, period=f"{a_}-{b}", omega_start=s.omega, omega_end=t.omega, dln_e=dle,
                             contrib_hours=c_h, contrib_productivity=c_y, contrib_real_wage=c_w,
                             residual=dle - c_h - c_y - c_w, years=b - a_))
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "s7_decomposition.csv", index=False)
    R["per"] = np.where(R.period.str.endswith("-1973"), "start-1973", R.period)
    S = R.groupby("per")[["dln_e", "contrib_hours", "contrib_productivity", "contrib_real_wage", "residual"]].median()
    S["countries"] = R.groupby("per").size()
    S["share_hours_positive"] = R.groupby("per").contrib_hours.apply(lambda v: (v > 0).mean())
    S.to_csv(OUT / "s7_summary.csv")
    # US Marxian e_m (9b long series)
    L = pd.read_csv(ROOT / "results" / "v9b" / "s1_long_series.csv")
    u = L[L.variant == "main"].set_index("year")
    em = []
    for a, b in ((1947, 1973), (1973, 1997), (1997, 2024)):
        em.append(dict(period=f"{a}-{b}", dln_e_m=float(np.log(u.e_m.loc[b] / u.e_m.loc[a]))))
    pd.DataFrame(em).to_csv(OUT / "s7_us_em.csv", index=False)
    pd.set_option("display.width", 250)
    print(S.round(4).to_string(), "\n", pd.DataFrame(em).round(4).to_string(), flush=True)


if __name__ == "__main__":
    run()
