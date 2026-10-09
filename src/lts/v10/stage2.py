"""Iteration 10, stage 2: union elections (NLRB) and firm outcomes (SEC FSDS), regression discontinuity at 50%
(pre_registration_v10.md, stage 2; journal 6).

Usage: PYTHONPATH=src python -P -m lts.v10.stage2
Outputs: results/v10/s2_*.csv
"""
from __future__ import annotations

import re
import sqlite3

import numpy as np
import pandas as pd
from scipy import stats

from ..v6.series import ROOT
from ..v8.rule import decide
from ..v9b.stage45 import wild_t

OUT = ROOT / "results" / "v10"
DB = ROOT / "data" / "raw" / "v10" / "nlrb.db"
SEC = ROOT / "data" / "raw" / "v10" / "sec" / "sec_firm_fy_all.csv"
BW = 0.15
MIN_N = 200
SUFFIX = {"inc", "incorporated", "corp", "corporation", "llc", "co", "company", "holdings", "holding", "group", "the",
          "ltd", "lp", "plc"}


def norm(s):
    s = str(s).lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    w = [t for t in s.split() if t not in SUFFIX]
    return " ".join(w)


def elections():
    c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    d = pd.read_sql("""select e.election_id, e.case_number, e.date, f.name as case_name,
        sum(case when t.option = 'No union' then t.votes end) as no_votes,
        sum(case when t.option != 'No union' then t.votes end) as yes_votes, count(t.option) as n_opt
        from election e join filing f using(case_number) left join tally t using(election_id)
        where e.case_number like '%-RC-%' and e.tally_type = 'Initial' and e.ballot_type = 'Single Labor Organization'
        group by e.election_id""", c)
    emp = pd.read_sql("select case_number, participant from participant where type = 'Employer' or subtype = 'Employer'", c)
    d = d[(d.n_opt == 2) & d.no_votes.notna() & d.yes_votes.notna() & ((d.no_votes + d.yes_votes) > 0)].copy()
    d["v"] = d.yes_votes / (d.yes_votes + d.no_votes)
    d["year"] = pd.to_numeric(d.date.str[:4], errors="coerce")
    names = emp.groupby("case_number").participant.apply(lambda s: {norm(x) for x in s if isinstance(x, str)})
    d["names"] = [({norm(n)} | names.get(k, set())) - {""} for n, k in zip(d.case_name, d.case_number)]
    return d


def firms():
    f = pd.read_csv(SEC, low_memory=False, usecols=["cik", "fy", "name", "countryba", "OperatingIncomeLoss", "Assets",
                                                     "PaymentsToAcquirePropertyPlantAndEquipment"])
    f = f[f.Assets > 0].copy()
    f["roa"] = f.OperatingIncomeLoss / f.Assets
    f["inv"] = f.PaymentsToAcquirePropertyPlantAndEquipment / f.Assets
    f["nn"] = f.name.map(norm)
    m = f.groupby("nn").cik.nunique()
    amb = set(m[m > 1].index)
    lookup = f[~f.nn.isin(amb) & (f.nn != "")].drop_duplicates("nn").set_index("nn").cik.to_dict()
    panel = f.drop_duplicates(["cik", "fy"]).set_index(["cik", "fy"])[["roa", "inv"]]
    return lookup, panel, len(amb)


def rd(df, y, bw=BW):
    d = df[(df.v - 0.5).abs() <= bw].dropna(subset=[y])
    if len(d) < 10:
        return dict(n=len(d))
    r = d.v - 0.5
    D = (d.v > 0.5).astype(float)
    w = 1 - r.abs() / bw
    X = np.column_stack([np.ones(len(d)), D, r, D * r])
    sw = np.sqrt(w.to_numpy())
    Xw, yw = X * sw[:, None], d[y].to_numpy() * sw
    G = d.cik.nunique()
    if G < 30:
        b, se, lo_b, hi_b, p, G = wild_t(Xw, yw, d.cik.to_numpy(), 1, B=9999, seed=53)
        tq = stats.t.ppf(0.95, G - 1)
        lo, hi = min(b - tq * se, lo_b), max(b + tq * se, hi_b)
    else:
        XtX = np.linalg.inv(Xw.T @ Xw)
        bb = XtX @ Xw.T @ yw
        e = yw - Xw @ bb
        gi = pd.factorize(d.cik)[0]
        meat = sum(np.outer(Xw[gi == g].T @ e[gi == g], Xw[gi == g].T @ e[gi == g]) for g in range(G))
        V = XtX @ meat @ XtX * G / (G - 1)
        b, se = float(bb[1]), float(np.sqrt(V[1, 1]))
        tq = stats.t.ppf(0.95, G - 1)
        lo, hi = b - tq * se, b + tq * se
    return dict(est=b, se=se, ci90_lo=lo, ci90_hi=hi, n=len(d), G=G, n_win=int(D.sum()), n_lose=int((1 - D).sum()))


def run():
    E = elections()
    lookup, P, n_amb = firms()
    # journal 16 (b): an election whose names lead to different CIKs is ambiguous and dropped (was order-dependent)
    hits = [{lookup[n] for n in ns if n in lookup} for ns in E.names]
    E["n_cik"] = [len(h) for h in hits]
    E["cik"] = [next(iter(h)) if len(h) == 1 else np.nan for h in hits]
    M = E.dropna(subset=["cik", "year"]).copy()
    M["cik"] = M.cik.astype(int)
    M["year"] = M.year.astype(int)

    def get(col, k):
        return [P[col].get((c, y + k), np.nan) for c, y in zip(M.cik, M.year)]
    for col in ("roa", "inv"):
        base = np.array(get(col, -1))
        for k in (1, 2, 3):
            M[f"d_{col}_{k}"] = np.array(get(col, k)) - base
    M.drop(columns=["names"]).to_csv(OUT / "s2_matched_elections.csv", index=False)
    inwin = M[(M.v - 0.5).abs() <= BW]
    stats_ = dict(elections_with_tally=len(E), ambiguous_elections_dropped=int((E.n_cik > 1).sum()), matched=len(M),
                  matched_firms=M.cik.nunique(),
                  matched_in_window=len(inwin), in_window_with_droa2=int(inwin.d_roa_2.notna().sum()),
                  ambiguous_sec_names=n_amb)
    print(stats_, flush=True)
    rows = []
    for y in [f"d_roa_{k}" for k in (1, 2, 3)] + [f"d_inv_{k}" for k in (1, 2, 3)]:
        for bw in (BW, 0.10, 0.25):
            r = rd(M, y, bw)
            rows.append(dict(outcome=y, bw=bw, **r))
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "s2_rd.csv", index=False)
    main = R[(R.outcome == "d_roa_2") & (R.bw == BW)].iloc[0]
    n_ok = stats_["in_window_with_droa2"]
    if n_ok < MIN_N:
        label = f"недостаточно данных ({n_ok} < {MIN_N} сопоставленных выборов в окне с ΔROA)"
    else:
        label = decide(main.est, main.ci90_lo, main.ci90_hi, 0, 0.005, "<")
    o = dict(outcome="53: RD jump in delta ROA (t-1 -> t+2) at 50% union vote share", est=main.get("est"),
             ci90_lo=main.get("ci90_lo"), ci90_hi=main.get("ci90_hi"), n=main.n, theta0=0, delta=0.005, direction="<",
             label=label, **{k: v for k, v in stats_.items()})
    pd.DataFrame([o]).to_csv(OUT / "s2_outcomes.csv", index=False)
    pd.DataFrame([stats_]).to_csv(OUT / "s2_match_stats.csv", index=False)
    # density check around the threshold (descriptive): counts in bins of 0.05
    h = pd.cut(M.v, np.arange(0, 1.0001, 0.05), include_lowest=True).value_counts().sort_index()
    h.rename("n").to_csv(OUT / "s2_vote_share_hist.csv")
    print(R.round(4).to_string(), "\n", o, flush=True)


def robust():
    """post hoc (journal 13): one election per firm-year (vote share of the largest unit), and without Starbucks"""
    M = pd.read_csv(OUT / "s2_matched_elections.csv")
    M["votes"] = M.yes_votes + M.no_votes
    one = M.sort_values("votes").groupby(["cik", "year"]).tail(1)
    sb = M[M.cik != 829224]
    rows = []
    donut = M[M.v != 0.5]
    for name, D in (("one election per firm-year", one), ("without Starbucks (CIK 829224)", sb),
                    ("without ties at v = 0.5 (donut)", donut)):
        for y in ("d_roa_1", "d_roa_2", "d_roa_3"):
            rows.append(dict(sample=name, outcome=y, **rd(D, y)))
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "s2_rd_robust.csv", index=False)
    print(R.round(4).to_string(), flush=True)


if __name__ == "__main__":
    import sys
    pd.set_option("display.width", 250)
    if "robust" in sys.argv[1:]:
        robust()
    else:
        run()
