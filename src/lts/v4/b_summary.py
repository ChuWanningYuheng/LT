"""Tables for REPORT_v4 part B (results/v4/tables_b.md)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..v3.summarize import md
from .b_data import OUT


def b0():
    t = pd.read_csv(OUT / "b0_t1.csv")
    rows = []
    for (c, s, h), g in t.groupby(["country", "subset", "h"]):
        L = g[g.basis == "labour"].r2_w.iloc[0]
        P = g[g.basis.str.startswith("perm")].r2_w
        rows.append(dict(country=c, subset=s, h=int(h), labour=L, flat_money=g[g.basis == "flat"].r2_w.iloc[0],
                         flat_phys=g[g.basis == "flat_phys"].r2_w.iloc[0], perm_median=P.median(),
                         perm_share_ge_labour=(P >= L).mean()))
    r = pd.DataFrame(rows)
    r["beats_perm95"] = r.perm_share_ge_labour <= 0.05
    r["beats_flat_money"] = r.labour > r.flat_money
    r["beats_flat_phys"] = r.labour > r.flat_phys
    r["informative_vs_phys"] = r.beats_perm95 & r.beats_flat_phys
    return r


def b0b_counts():
    b = pd.read_csv(OUT / "b0b_boot.csv")
    out = []
    for (mask, meth), g in b.groupby(["mask", "method"]):
        w = g.pivot_table(index=["country", "cand"], columns="metric",
                          values=["hours_sig_better", "cand_sig_better", "delta"])
        hb = (w["hours_sig_better"]["mawd"] > 0) & (w["hours_sig_better"]["d"] > 0)
        nw = ~((w["hours_sig_better"]["mawd"] > 0) | (w["hours_sig_better"]["d"] > 0))
        cb = (w["cand_sig_better"]["mawd"] > 0) | (w["cand_sig_better"]["d"] > 0)
        pt = (w["delta"]["mawd"] > 0) & (w["delta"]["d"] > 0)
        s = pd.DataFrame({"n": 1, "hours_sig_better_both": hb, "not_worse": nw, "cand_sig_better_any": cb,
                          "hours_point_better_both": pt}).groupby(level="cand").sum()
        s["mask"], s["method"] = mask, meth
        out.append(s.reset_index())
    return pd.concat(out)


def main():
    parts = ["# Таблицы части Б итерации 4\n"]
    r = b0()
    r.to_csv(OUT / "b0_summary.csv", index=False)
    parts += ["## Б0: T1 (R² изменений относительных цен)\n", md(r.set_index("country"), "{:.3f}"), ""]
    c = b0b_counts()
    c.to_csv(OUT / "b0b_counts.csv", index=False)
    parts += ["## Б0б: число стран (часы против кандидата)\n", md(c.set_index("cand")), ""]
    b = pd.read_csv(OUT / "b0b_boot.csv")
    g = b[(b["mask"] == "all") & (b.method == "a_quantile")]
    for met in ("mawd", "d"):
        x = g[g.metric == met].pivot_table(index="country", columns="cand", values="mean")
        parts += [f"## Б0б: {met}, квантильное выравнивание, основная выборка\n", md(x, "{:.3f}"), ""]
    pr = b[(b.method == "a_quantile")].groupby(["mask", "cand", "metric"]).perm_better.median().unstack(["mask", "metric"])
    parts += ["## Б0б: медианная доля собственных перестановок лучше кандидата\n", md(pr, "{:.3f}"), ""]
    for f, title in (("b31.csv", "Б3.1"), ("b31_no_pcm_exploratory.csv", "Б3.1 без PCM (разведочно)"),
                     ("b33.csv", "Б3.3 (разведочно)"), ("b32.csv", "Б3.2"), ("b32_country.csv", "Б3.2 по странам")):
        try:
            x = pd.read_csv(OUT / f)
            parts += [f"## {title}\n", md(x.set_index(x.columns[0]), "{:.4f}"), ""]
        except FileNotFoundError:
            pass
    (OUT / "tables_b.md").write_text("\n".join(parts))


if __name__ == "__main__":
    main()
