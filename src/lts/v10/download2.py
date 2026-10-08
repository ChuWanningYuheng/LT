"""Iteration 10, stage 0 (part 2): UN Comtrade public preview (HS 847950 imports, partner = world) and conversion of
Känzig (2021) / Kilian proxy series from the authors' replication repository (CC0) to CSV.

Usage: PYTHONPATH=src python -m lts.v10.download2 [comtrade|kaenzig]
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

from .download import RAW, UA
from ..v6.series import ROOT


def manifest(name, url, licence, note):
    data = (RAW / name).read_bytes()
    with open(ROOT / "data" / "raw" / "MANIFEST.csv", "a", newline="") as f:
        csv.writer(f).writerow(["v10", name, url, licence, f"iteration 10, {note}", len(data),
                                hashlib.sha256(data).hexdigest(), dt.datetime.now().isoformat(timespec="seconds")])


def comtrade():
    rows = []
    base = ("https://comtradeapi.un.org/public/v1/preview/C/A/HS?cmdCode=847950&flowCode={f}&period={y}"
            "&partnerCode=0&partner2Code=0&customsCode=C00&motCode=0")
    for flow in ("M", "X"):
        for y in range(1996, 2025):
            url = base.format(f=flow, y=y)
            for k in range(4):
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": UA})
                    d = json.load(urllib.request.urlopen(req, timeout=120))
                    break
                except Exception as e:                       # noqa: BLE001
                    print("retry", flow, y, e, flush=True)
                    time.sleep(2 ** (k + 2))
            else:
                print("FAILED", flow, y, flush=True)
                continue
            for r in d.get("data", []):
                rows.append({k: r.get(k) for k in ("refYear", "reporterCode", "flowCode", "partnerCode", "classificationCode",
                                                   "cmdCode", "primaryValue", "cifvalue", "fobvalue", "qty", "qtyUnitCode",
                                                   "netWgt", "isReported", "isAggregate")})
            print(flow, y, len(d.get("data", [])), d.get("count"), flush=True)
            time.sleep(1.5)
    out = pd.DataFrame(rows)
    out.to_csv(RAW / "comtrade_847950.csv", index=False)
    manifest("comtrade_847950.csv", base.format(f="{M,X}", y="{1996..2024}"), "UN Comtrade terms (public preview API)",
             "HS 847950 (industrial robots n.e.s.) imports and exports, partner = world, all reporters")


def kaenzig():
    import h5py
    src = "/home/user/dkaenzig/replicationoilsupplynews/data/OilSurprisesMLog.mat"
    h = h5py.File(src, "r")

    def s(ref):
        return "".join(chr(c) for c in np.array(h[ref]).ravel())
    d = [s(r) for r in np.array(h["sampleDatesProxy"]).ravel()]
    p = np.array(h["oilProxiesWTIM"])
    pd.DataFrame({"month": d, "oil_supply_surprise_contract15": p[14], "oil_supply_surprise_contract2": p[1]}).to_csv(
        RAW / "kaenzig2021_oil_supply_surprises.csv", index=False)
    k = [s(r) for r in np.array(h["sampleDatesProxyKilian"]).ravel()]
    kk = np.array(h["oilProxyKilian"])
    pd.DataFrame({"month": k, "kilian_proxy_1": kk[0], "kilian_proxy_2": kk[1]}).to_csv(RAW / "kilian_proxy_from_kaenzig.csv",
                                                                                        index=False)
    url = "https://github.com/dkaenzig/replicationoilsupplynews (data/OilSurprisesMLog.mat)"
    manifest("kaenzig2021_oil_supply_surprises.csv", url, "CC0 1.0 (author's repository)",
             "Känzig (2021) monthly oil supply surprise series (WTI futures, contracts 15 and 2), 1974M01-2017M12")
    manifest("kilian_proxy_from_kaenzig.csv", url, "CC0 1.0 (author's repository)",
             "Kilian oil supply shock proxy as distributed in Känzig's replication files, 1973M01-2017M12")


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else "all"
    if part in ("kaenzig", "all"):
        kaenzig()
    if part in ("comtrade", "all"):
        comtrade()
