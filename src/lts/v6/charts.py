"""Iteration 6: figures for REPORT_v6.md (static PNG, small multiples, light surface).

Palette: first three categorical slots of the reference palette (validated all-pairs for <= 3 series).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .series import MAIN8, OUT  # noqa: E402

FIG = OUT.parent.parent / "report" / "fig_v6"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
NAMES = dict(USA="США", GBR="Великобритания", FRA="Франция", DEU="Германия (ФРГ до 1990)", ITA="Италия",
             NLD="Нидерланды", SWE="Швеция", AUS="Австралия", USA_NFC="США, нефин. корпорации")


def small_multiples(s, series, fname, title, ylabel):
    geos = MAIN8 + ["USA_NFC"]
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), facecolor=SURFACE)
    for ax, geo in zip(axes.ravel(), geos):
        ax.set_facecolor(SURFACE)
        for (variant, col, label, color, ls) in series:
            v = variant if not (geo == "USA_NFC" and variant == "kdrift") else "no_ip"
            g = s[(s.geo == geo) & (s.variant == v)].sort_values("year")
            if g.empty:
                continue
            lab = label
            ax.plot(g.year, g[col], color=color, lw=2, ls=ls, label=lab)
        ax.set_title(NAMES[geo], fontsize=10, color=INK, loc="left")
        ax.grid(True, color=GRID, lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=8)
        ax.set_xlim(1945, 2025)
    handles, labels = [], []
    for ax in axes.ravel():
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 2), frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.955), labelcolor=INK)
    fig.suptitle(title, color=INK, fontsize=12, y=0.99)
    fig.supylabel(ylabel, color=INK2, fontsize=9)
    fig.tight_layout(rect=(0.01, 0, 1, 0.93))
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / fname, dpi=130, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    s = pd.read_csv(OUT / "series.csv")
    small_multiples(s, [("main", "rM", "r_M, основной вариант", S1, "-"),
                        ("kdrift", "rM", "r_M, K с поправкой на дрейф (разведочно; США НФК — K без ИС)", S3, "--")],
                    "rM.png", "Марксова норма прибыли r_M = Π/(K + W)", "доля")
    small_multiples(s, [("main", "r", "r, основной вариант", S1, "-"),
                        ("no_mi", "r", "r без поправки на смешанный доход", S2, "-"),
                        ("kdrift", "r", "r, K с поправкой на дрейф (разведочно; США НФК — K без ИС)", S3, "--")],
                    "r.png", "Норма прибыли r = Π/K", "доля")
    small_multiples(s, [("main", "e", "e, основной вариант", S1, "-"),
                        ("no_mi", "e", "e без поправки на смешанный доход", S2, "-")],
                    "e.png", "Норма эксплуатации e = Π/W", "отношение")
    small_multiples(s, [("main", "k", "k, основной вариант", S1, "-"),
                        ("kdrift", "k", "k, K с поправкой на дрейф (разведочно; США НФК — K без ИС)", S3, "--")],
                    "k.png", "Органическое строение (денежное) k = K/W", "отношение")
    print("ok")
