"""Iteration 6, exploratory (pre_registration_v6.md, journal 10): official nonfinancial-corporation (S11)
profit rates outside the US, and Z.1 financial-income variants for US NFC.

Sources (data/raw/v6/national, data/raw/v6/z1): Eurostat nasa_10_nf_tr / nama_10_nfa_bs, ONS Blue Book,
ABS 5204 tables 17 and 57, ESRI 2024 annual accounts, StatCan 36-10-0116 / 36-10-0580, Fed Z.1 (S11.1).
Outputs: results/v6/nat_series.csv, nat_trends.csv, nat_summary.csv, z1_trends.csv
"""
from __future__ import annotations

import json
import zipfile

import numpy as np
import pandas as pd

from .series import OUT, RAW, fa
from .tests import trend

NAT = RAW / "v6" / "national"
EUROSTAT = {"FRA": "FR", "DEU": "DE", "ITA": "IT", "NLD": "NL", "SWE": "SE", "NOR": "NO", "AUT": "AT", "BEL": "BE",
            "DNK": "DK", "FIN": "FI", "CZE": "CZ", "PRT": "PT"}


def es(path):
    d = json.load(open(path))
    ids, size = d["id"], d["size"]
    cats = [sorted(d["dimension"][k]["category"]["index"].items(), key=lambda x: x[1]) for k in ids]
    mult = [1] * len(size)
    for i in range(len(size) - 2, -1, -1):
        mult[i] = mult[i + 1] * size[i + 1]
    rows = []
    for k, v in d["value"].items():
        k = int(k)
        r = {idn: cats[i][(k // mult[i]) % size[i]][0] for i, idn in enumerate(ids)}
        r["value"] = v
        rows.append(r)
    f = pd.DataFrame(rows)
    if f.time.str.fullmatch(r"\d{4}").all():
        f["time"] = f.time.astype(int)
    return f


def eurostat():
    t = es(NAT / "eurostat_nasa_10_nf_tr_S11.json")
    a = es(NAT / "eurostat_nama_10_nfa_bs_S11.json")
    out = []
    for g3, g2 in EUROSTAT.items():
        nos = t[(t.geo == g2) & (t.na_item == "B2A3N") & (t.direct == "RECV")].set_index("time").value
        if nos.empty:
            nos = t[(t.geo == g2) & (t.na_item == "B2A3N")].groupby("time").value.first()
        w = t[(t.geo == g2) & (t.na_item == "D1") & (t.direct == "PAID")].set_index("time").value
        k = a[(a.geo == g2) & (a.asset10 == "N11N")].set_index("time").value
        d = pd.DataFrame({"PI": nos, "W": w, "K": k})
        out.append(d.assign(geo=g3, source="Eurostat S11"))
    return out


def ons():
    u = pd.read_csv(NAT / "ons_bb.csv", header=None, low_memory=False)
    cd = u.iloc[1].astype(str)
    body = u.iloc[7:]
    body = body[body[0].astype(str).str.fullmatch(r"\d{4}")]
    get = lambda c: pd.to_numeric(body[cd[cd == c].index[0]], errors="coerce").set_axis(body[0].astype(int))  # noqa: E731
    d = pd.DataFrame({"PI": get("FAIR"), "W": get("FAKT") + get("FAKU"), "K": get("NG2D")})
    return d.assign(geo="GBR", source="ONS Blue Book S11")


def abs_aus():
    def series(f, label):
        d = pd.read_excel(NAT / f, "Data1", header=None)
        j = [i for i in range(1, d.shape[1]) if str(d.iloc[0, i]).strip().startswith(label)][0]
        dates = pd.to_datetime(d.iloc[10:, 0], errors="coerce")
        v = pd.to_numeric(d.iloc[10:, j], errors="coerce")
        return pd.Series(v.to_numpy(), index=dates.dt.year.to_numpy()).dropna()
    gos = series("abs_5204017_NonFin_Corp_Income_Account.xlsx", "Gross operating surplus")
    cfc = series("abs_5204017_NonFin_Corp_Income_Account.xlsx", "Consumption of fixed capital")
    k = series("abs_5204057_Capital_Stock_By_Sector.xlsx", "Non-financial corporations ;  End-year net capital stock: Current prices")
    return pd.DataFrame({"PI": gos - cfc, "K": k, "W": np.nan}).assign(geo="AUS", source="ABS 5204 t17, t57 (FY to June)")


def esri():
    def row(f, sheet, yrow, label):
        d = pd.read_excel(NAT / f, sheet, header=None)
        years = pd.to_numeric(d.iloc[yrow, 1:], errors="coerce")
        i = [r for r in range(len(d)) if "".join(str(d.iloc[r, 0]).split()).startswith(label)][0]
        v = pd.to_numeric(d.iloc[i, 1:], errors="coerce")
        return pd.Series(v.to_numpy(), index=years.to_numpy()).dropna()
    nos = row("esri_2024i2_jp.xlsx", "暦年（１）", 6, "1.3営業余剰（純）")
    k = row("esri_2024si11_jp.xlsx", "期末貸借対照表", 8, "ａ．固定資産")
    s = pd.DataFrame({"PI": nos, "K": k, "W": np.nan})
    s.index = s.index.astype(int)
    return s.assign(geo="JPN", source="ESRI 2024 (calendar years)")


def statcan():
    z = zipfile.ZipFile(NAT / "statcan_36100116.zip")
    d = pd.read_csv(z.open("36100116.csv"), low_memory=False)
    # unadjusted quarters are zero before the 2000s; SAAR quarters averaged to an annual value (journal 10)
    d = d[(d.Corporations == "Non-financial corporations") & (d["Seasonal adjustment"] == "Seasonally adjusted at annual rates") &
          (d.Estimates == "Net operating surplus")]
    d["year"] = d.REF_DATE.str[:4].astype(int)
    g = d.groupby("year").VALUE.agg(["mean", "size"])
    nos = g["mean"][(g["size"] == 4) & (g["mean"] != 0)]           # zeros before 1997 = not available
    z = zipfile.ZipFile(NAT / "statcan_36100580.zip")
    b = pd.read_csv(z.open("36100580.csv"), low_memory=False)
    b = b[(b.Sectors == "Non-financial corporations") & (b.Categories == "Fixed assets") & (b.Valuation == "Market value") &
          b.REF_DATE.str.endswith("-10")]
    k = b.set_index(b.REF_DATE.str[:4].astype(int)).VALUE
    return pd.DataFrame({"PI": nos, "K": k, "W": np.nan}).assign(geo="CAN", source="StatCan 36-10-0116, 36-10-0580")


def build():
    parts = eurostat() + [ons(), abs_aus(), esri(), statcan()]
    rows = []
    for d in parts:
        d = d.copy()
        d.index.name = "year"
        d = d[d.PI.notna() & d.K.notna() & (d.K > 0)]
        d["r"] = d.PI / d.K
        d["rM"] = d.PI / (d.K + d.W)
        rows.append(d.reset_index())
    return pd.concat(rows, ignore_index=True)


def trends(nat, ameco):
    rows = []
    for geo, g in nat.groupby("geo"):
        g = g.set_index("year").sort_index()
        s0, s1 = int(g.index.min()), int(g.index.max())
        if s1 - s0 + 1 < 20:
            continue
        am = ameco[ameco.geo == geo].set_index("year").sort_index() if geo in set(ameco.geo) else None
        for series in ("r", "rM"):
            for label, src in (("official S11", g), ("AMECO same years", am)):
                if src is None or series not in src or src[series].isna().all():
                    continue
                for a in (s0, s0 + 5):
                    for b in (s1, 2019):
                        y = src.loc[a:b, series].dropna()
                        if len(y) < 15:
                            continue
                        for lag in (2, 4, 8):
                            sl, se, p = trend(y.to_numpy(), lag)
                            rows.append(dict(geo=geo, source=label, series=series, y0=a, y1=b, lag=lag, slope=sl, p=p,
                                             n=len(y), first=y.iloc[0], last=y.iloc[-1]))
    return pd.DataFrame(rows)


def summary(tr):
    out = []
    for (geo, src, s), g in tr.groupby(["geo", "source", "series"]):
        out.append(dict(geo=geo, source=src, series=s, n_specs=len(g), years=f"{g.y0.min()}-{g.y1.max()}",
                        robust_neg=bool(((g.slope < 0) & (g.p < 0.05)).all()),
                        robust_pos=bool(((g.slope > 0) & (g.p < 0.05)).all()),
                        share_neg_sig=((g.slope < 0) & (g.p < 0.05)).mean(),
                        share_pos_sig=((g.slope > 0) & (g.p < 0.05)).mean(),
                        first=g.loc[(g.y0 == g.y0.min()) & (g.y1 == g.y1.max()), "first"].iloc[0],
                        last=g.loc[(g.y0 == g.y0.min()) & (g.y1 == g.y1.max()), "last"].iloc[0]))
    return pd.DataFrame(out)


# ---------------------------------------------------------------- Z.1 (US NFC financial income)
def z1():
    z = zipfile.ZipFile(RAW / "v6" / "z1" / "z1_csv_files.zip")
    ia = pd.read_csv(z.open("csv/S11_1_i_a.csv")).set_index("date")
    bs = pd.read_csv(z.open("csv/S11_1_b.csv")).set_index("date")
    num = lambda s: pd.to_numeric(s.replace("ND", np.nan), errors="coerce")   # noqa: E731  millions, as BEA FA
    ia.index = ia.index.astype(int)
    q4 = bs[bs.index.str.endswith("Q4")]
    q4.index = q4.index.str[:4].astype(int)
    d = pd.DataFrame({
        "NOS": num(ia["FU106402101.A"]), "int_rec": num(ia["FU106130101.A"]), "int_paid": num(ia["FU106130001.A"]),
        "div_rec": num(ia["FU106121101.A"]), "re_rec": num(ia["FU103092201.A"]),
        "W": num(ia["FU106025005.A"])})
    d["FA"] = num(q4["FL104090005.Q"])
    d["K"] = fa(4, "FAAt401-A", 37)
    d = d.loc[1946:2024]
    d = d.fillna({"re_rec": 0.0})
    specs = {
        "NOS": d.NOS,
        "F1 gross": d.NOS + d.int_rec + d.div_rec + d.re_rec,
        "F2 net interest + dividends": d.NOS + d.int_rec - d.int_paid + d.div_rec + d.re_rec,
        "F3 net interest only": d.NOS + d.int_rec - d.int_paid,
    }
    rows = []
    for name, pi in specs.items():
        for den_name, den in (("K", d.K), ("K+FA", d.K + d.FA)):
            r = (pi / den).dropna()
            for a in (1946, 1951, 1958, 1963):
                for b in (2024, 2019):
                    y = r.loc[a:b]
                    for lag in (2, 4, 8):
                        sl, se, p = trend(y.to_numpy(), lag)
                        rows.append(dict(spec=name, denom=den_name, y0=a, y1=b, lag=lag, slope=sl, p=p, n=len(y),
                                         first=y.iloc[0], last=y.iloc[-1], mean=y.mean()))
    shares = pd.DataFrame({"fin_income_to_NOS_F1": (specs["F1 gross"] - d.NOS) / d.NOS,
                           "net_interest_to_NOS": (d.int_rec - d.int_paid) / d.NOS, "FA_to_K": d.FA / d.K})
    return pd.DataFrame(rows), shares


if __name__ == "__main__":
    nat = build()
    nat.to_csv(OUT / "nat_series.csv", index=False)
    print(nat.groupby(["geo", "source"]).year.agg(["min", "max", "size"]).to_string(), flush=True)
    s = pd.read_csv(OUT / "series.csv")
    am = s[s.variant == "main"][["geo", "year", "r", "rM"]]
    tr = trends(nat, am)
    tr.to_csv(OUT / "nat_trends.csv", index=False)
    sm = summary(tr)
    sm.to_csv(OUT / "nat_summary.csv", index=False)
    print(sm.round(3).to_string(), flush=True)
    zt, sh = z1()
    zt.to_csv(OUT / "z1_trends.csv", index=False)
    sh.to_csv(OUT / "z1_shares.csv")
    base = zt[(zt.y1 == 2024)]
    print(base.pivot_table(index=["spec", "denom"], columns=["y0", "lag"], values="p").round(3).to_string())
    print(base[base.lag == 4].pivot_table(index=["spec", "denom"], columns="y0", values="slope").round(5).to_string())
    print(sh.loc[[1958, 1970, 1990, 2010, 2024]].round(3).to_string())
