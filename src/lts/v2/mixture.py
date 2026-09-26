"""Iteration 2, stage 1b step 4: does X add information to labour (and labour to X)?

Mixture z_mix(w) = (1-w) z_L + w z_O  (z_L: standard labour values, z_O: asymmetric X-values; both
normalised on the evaluated industries). Criterion: MAWD averaged over years.
Out-of-bag bootstrap over industries (pre-registered): for each of B resamples, w is fitted on the
in-bag industries (grid of 21 values) and the gain Delta = MAWD_oob(L) - MAWD_oob(mix) is measured on
out-of-bag industries (the same industry resample is used for all years). p = share(Delta <= 0).
Reverse test: Delta_O = MAWD_oob(O) - MAWD_oob(mix). Benjamini-Hochberg across bases within country.
Also reported: full-sample w*, MAWD of L, O, mix (linear and log mixtures).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "v2"
GRID = np.linspace(0, 1, 21)


def bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted q-values."""
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank, i in reversed(list(enumerate(order, start=1))):
        prev = min(prev, p[i] * n / rank)
        q[i] = prev
    return q


def mawd_w(Z: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Z: (..., Y, J) ratios, W: (..., Y, J) weights (broadcastable). Returns MAWD over J, (..., Y)."""
    sw = W.sum(-1)
    k = sw / (W * Z).sum(-1)
    return (W * np.abs(1.0 / (Z * k[..., None]) - 1)).sum(-1) / sw


def run_country(c: str, B: int = 1000, seed: int = 2026, batch: int = 100) -> list[dict]:
    f = np.load(OUT / f"decomp_z_{c}.npz", allow_pickle=False)
    idx = np.where(f["mask"])[0]
    zL, x = f["zL"][:, idx], f["x"][:, idx]
    Y, J = zL.shape
    norm = lambda z: z * x.sum(-1, keepdims=True) / (z * x).sum(-1, keepdims=True)
    zL = norm(zL)
    rng = np.random.default_rng(seed + sum(map(ord, c)))
    counts = rng.multinomial(J, np.full(J, 1 / J), size=B)          # (B, J), same for all bases
    rows = []
    for setting, key in (("asym", "zO"), ("sym", "zS")):
        ZX = f[key][:, :, idx]
        prow = []
        for bi, bname in enumerate(f["bases"]):
            zo = norm(ZX[bi])
            if not np.isfinite(zo).all() or (zo <= 0).any():
                continue
            Zmix = (1 - GRID)[:, None, None] * zL[None] + GRID[:, None, None] * zo[None]      # (G, Y, J)
            full = mawd_w(Zmix, x[None]).mean(-1)                                              # (G,)
            g_full = int(np.argmin(full))
            logmix = np.exp((1 - GRID)[:, None, None] * np.log(zL)[None] + GRID[:, None, None] * np.log(zo)[None])
            lfull = mawd_w(logmix, x[None]).mean(-1)
            eL, eX, eM, wstar = [], [], [], []
            for s0 in range(0, B, batch):
                C = counts[s0:s0 + batch]
                Win = C[:, None, :] * x[None]
                Wout = (C == 0)[:, None, :] * x[None]
                min_ = mawd_w(Zmix[:, None], Win[None]).mean(-1)                               # (G, b)
                mout = mawd_w(Zmix[:, None], Wout[None]).mean(-1)
                g = np.argmin(min_, 0)
                ar = np.arange(len(g))
                eL.append(mout[0]); eX.append(mout[-1]); eM.append(mout[g, ar]); wstar.append(GRID[g])
            eL, eX, eM, wstar = map(np.concatenate, (eL, eX, eM, wstar))
            dL, dX = eL - eM, eX - eM
            prow.append(dict(country=c, setting=setting, basis=str(bname), w_full=float(GRID[g_full]),
                             mawd_L=float(full[0]), mawd_X=float(full[-1]), mawd_mix=float(full[g_full]),
                             w_full_log=float(GRID[int(np.argmin(lfull))]), mawd_mix_log=float(lfull.min()),
                             err_oob_L=float(eL.mean()), err_oob_X=float(eX.mean()), err_oob_mix=float(eM.mean()),
                             red_L_by_X=float(1 - eM.mean() / eL.mean()),      # share of labour's error closed by X
                             red_X_by_L=float(1 - eM.mean() / eX.mean()),      # share of X's error closed by labour
                             p_mix_vs_L=float((dL <= 0).mean()), p_mix_vs_X=float((dX <= 0).mean()),
                             w_boot_median=float(np.median(wstar)), share_boot_w0=float((wstar == 0).mean())))
        df = pd.DataFrame(prow)
        df["q_mix_vs_L"] = bh(df.p_mix_vs_L.to_numpy())
        df["q_mix_vs_X"] = bh(df.p_mix_vs_X.to_numpy())
        rows += df.to_dict("records")
    return rows


def run(countries=None):
    from .decomposition import COUNTRIES
    rows = []
    for c in countries or COUNTRIES:
        rows += run_country(c)
        print("mixture", c, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "decomp_mixture.csv", index=False)


if __name__ == "__main__":
    run()
