"""Parse Eurostat FIGARO inter-country industry-by-industry tables into dense yearly arrays.

Output: data/processed/figaro/figaro_<year>.npz with
    Z  : (C*N, C*N)  intermediate use; row = (country of origin, supplying industry),
                      column = (country of destination, using industry)
    F  : (C*N, C*K)  final use by destination country and final-demand category
    V  : (R, C*N)    value-added rows of each using industry (D1, B2A3G, D29X39, D21X31, OP_RES, OP_NRES)
    countries, industries, fd, varows : label arrays
Units: million EUR, current prices.
"""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "figaro"
OUT = ROOT / "data" / "processed" / "figaro"

INDUSTRIES = ['A01', 'A02', 'A03', 'B', 'C10-12', 'C13-15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22',
              'C23', 'C24', 'C25', 'C26', 'C27', 'C28', 'C29', 'C30', 'C31_32', 'C33', 'D35', 'E36', 'E37-39',
              'F', 'G45', 'G46', 'G47', 'H49', 'H50', 'H51', 'H52', 'H53', 'I', 'J58', 'J59_60', 'J61',
              'J62_63', 'K64', 'K65', 'K66', 'L', 'M69_70', 'M71', 'M72', 'M73', 'M74_75', 'N77', 'N78', 'N79',
              'N80-82', 'O84', 'P85', 'Q86', 'Q87_88', 'R90-92', 'R93', 'S94', 'S95', 'S96', 'T', 'U']
FD = ['P3_S13', 'P3_S14', 'P3_S15', 'P51G', 'P5M']
VAROWS = ['D1', 'B2A3G', 'D29X39', 'D21X31', 'OP_RES', 'OP_NRES']


def _iter_file(path: Path):
    with gzip.open(path, "rt") as f:
        header = f.readline().rstrip("\n").split("\t")
        years = [int(h.strip()) for h in header[1:]]
        for chunk in pd.read_csv(f, sep="\t", header=None, names=["key"] + [str(y) for y in years],
                                 dtype=str, chunksize=2_000_000):
            yield years, chunk


def build(force: bool = False) -> list[int]:
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(RAW.glob("naio_10_fcp_ii*.tsv.gz"))
    done = []
    for path in files:
        data = {}  # year -> dict of arrays
        countries = None
        for years, ch in _iter_file(path):
            k = ch["key"].str.split(",", expand=True)
            k.columns = ["freq", "ind_use", "ind_ava", "c_dest", "unit", "c_orig"]
            if countries is None:
                # FIGARO country list (fixed order; RoW last)
                countries = sorted(set(k["c_dest"].unique()) - {"WRL_REST"}) + ["WRL_REST"]
                cidx = {c: i for i, c in enumerate(countries)}
                iidx = {c: i for i, c in enumerate(INDUSTRIES)}
                fidx = {c: i for i, c in enumerate(FD)}
                vidx = {c: i for i, c in enumerate(VAROWS)}
                C, N, K, R = len(countries), len(INDUSTRIES), len(FD), len(VAROWS)
                for y in years:
                    data[y] = dict(Z=np.zeros((C * N, C * N)), F=np.zeros((C * N, C * K)), V=np.zeros((R, C * N)))
            else:
                new = set(k["c_dest"].unique()) - set(countries)
                assert not new, new
            cd = k["c_dest"].map(cidx).to_numpy()
            iu = k["ind_use"]
            use_ind = iu.map(iidx)
            use_fd = iu.map(fidx)
            ia = k["ind_ava"]
            ava_ind = ia.map(iidx)
            ava_va = ia.map(vidx)
            co = k["c_orig"].map(cidx)
            for y in years:
                vals = pd.to_numeric(ch[str(y)].str.strip().str.split(" ").str[0], errors="coerce").to_numpy()
                vals = np.nan_to_num(vals)
                d = data[y]
                # intermediate: origin industry row, destination industry column
                m = (ava_ind.notna() & use_ind.notna() & co.notna()).to_numpy()
                r = co[m].astype(int).to_numpy() * N + ava_ind[m].astype(int).to_numpy()
                c = cd[m] * N + use_ind[m].astype(int).to_numpy()
                np.add.at(d["Z"], (r, c), vals[m])
                m = (ava_ind.notna() & use_fd.notna() & co.notna()).to_numpy()
                r = co[m].astype(int).to_numpy() * N + ava_ind[m].astype(int).to_numpy()
                c = cd[m] * K + use_fd[m].astype(int).to_numpy()
                np.add.at(d["F"], (r, c), vals[m])
                m = (ava_va.notna() & use_ind.notna()).to_numpy()
                r = ava_va[m].astype(int).to_numpy()
                c = cd[m] * N + use_ind[m].astype(int).to_numpy()
                np.add.at(d["V"], (r, c), vals[m])
        for y, d in data.items():
            if np.abs(d["Z"]).sum() == 0:
                continue
            np.savez_compressed(OUT / f"figaro_{y}.npz", Z=d["Z"], F=d["F"], V=d["V"],
                                countries=np.array(countries), industries=np.array(INDUSTRIES),
                                fd=np.array(FD), varows=np.array(VAROWS))
            done.append(y)
            print(f"figaro {y}: saved", flush=True)
    return done


def load(year: int) -> dict:
    z = np.load(OUT / f"figaro_{year}.npz", allow_pickle=False)
    return {k: z[k] for k in z.files}


def check(year: int) -> dict:
    """Accounting identities: output by rows (Z+F) equals output by columns (Z+V)."""
    d = load(year)
    xr = d["Z"].sum(1) + d["F"].sum(1)
    xc = d["Z"].sum(0) + d["V"].sum(0)
    rel = np.abs(xr - xc) / np.maximum(np.abs(xr), 1.0)
    return dict(year=year, max_rel_gap=float(rel.max()), median_rel_gap=float(np.median(rel)),
                total_output=float(xr.sum()))


if __name__ == "__main__":
    build()
    for y in sorted(int(p.stem.split("_")[1]) for p in OUT.glob("figaro_*.npz")):
        print(check(y))
