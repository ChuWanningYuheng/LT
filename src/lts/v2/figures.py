"""Figures for report v2 (results/figures/v2_*.png)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
V2 = ROOT / "results" / "v2"
FIG = ROOT / "results" / "figures"
BLUE, ORANGE, AQUA, GRAY, INK, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#b9b8b1", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "figure.dpi": 130})
MAIN = {"DEU": BLUE, "MEX": ORANGE, "USA": AQUA}


def theta_curve():
    t = pd.read_csv(V2 / "decomp_theta.csv")
    t = t[t.metric == "mawd"]
    feas = t.groupby(["country", "s"]).feasible.mean().unstack(1)
    c = t[t.feasible.astype(bool)].groupby(["country", "s"]).share_better.mean().unstack(1)
    c = c.where(feas.reindex_like(c) >= 0.5)          # keep points feasible in at least half of the years
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for cc, row in c.iterrows():
        col = MAIN.get(cc, GRAY)
        ax.plot(row.index, row.values, color=col, lw=2 if cc in MAIN else 1, zorder=3 if cc in MAIN else 2)
        if cc in MAIN:
            last = row.dropna()
            ax.text(last.index[-1] + 0.02, last.values[-1], cc, color=INK, fontsize=8, va="center")
    ax.axhline(0.2, color=INK2, lw=0.8, ls="--")
    ax.text(0.01, 0.215, "труд выпадает из верхних 20%", fontsize=7, color=INK2)
    ax.axvline(1.0, color=INK2, lw=0.8, ls=":")
    ax.text(1.01, 0.02, "корзина = средняя\nоплата часа", fontsize=7, color=INK2)
    ax.set_xlabel("доля корзины рабочих s (0 = асимметричная постановка)")
    ax.set_ylabel("доля товарных базисов лучше труда (MAWD)")
    ax.set_title("Процентиль труда как функция доли корзины (13 стран; серые — остальные 10)", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "v2_theta_curve.png")
    plt.close(fig)


def decomposition():
    d = pd.read_csv(V2 / "decomp_basis.csv")
    a = d.groupby(["country", "basis", "type"]).mean(numeric_only=True).reset_index()
    a["rel_sym"] = a.mawd_sym / a.mawd_labour
    a["rel_O"] = a.mawd_O / a.mawd_labour
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    cols = {"goods": ORANGE, "construction": AQUA, "services": BLUE}
    names = {"goods": "товары (A–E)", "construction": "строительство (F)", "services": "услуги (G–U)"}
    for t, g in a.groupby("type"):
        axes[0].scatter(g.share_L_w, g.rel_sym, s=8, color=cols[t], label=names[t], alpha=0.7)
        axes[1].scatter(g.rel_O, g.rel_sym, s=8, color=cols[t], alpha=0.7)
    for ax in axes:
        ax.axhline(1, color=INK2, lw=0.8, ls="--")
    axes[0].set_xlabel("доля трудовой части L_X в v_X^sym (взвеш. по выпуску)")
    axes[0].set_ylabel("MAWD(v_X^sym) / MAWD(труд)")
    axes[1].set_xscale("log")
    axes[1].axvline(1, color=INK2, lw=0.8, ls="--")
    axes[1].set_xlabel("MAWD(O_X в одиночку) / MAWD(труд), лог")
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Разложение симметричных стоимостей: ниже 1 = базис лучше труда (все 13 стран × базисы)",
                 fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "v2_decomposition.png")
    plt.close(fig)


def mixture():
    m = pd.read_csv(V2 / "decomp_mixture.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=False)
    bins = np.linspace(-0.3, 0.8, 45)
    for ax, st, title in ((axes[0], "asym", "X асимметричный (O_X)"), (axes[1], "sym", "X симметричный (v_X^sym)")):
        g = m[m.setting == st]
        ax.hist(g.red_L_by_X.clip(-0.3, 0.8), bins=bins, color=BLUE, alpha=0.7, label="добавка X к труду")
        ax.hist(g.red_X_by_L.clip(-0.3, 0.8), bins=bins, color=ORANGE, alpha=0.6, label="добавка труда к X")
        for v in (0.05, 0.2):
            ax.axvline(v, color=INK2, lw=0.8, ls="--")
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_xlabel("доля ошибки (MAWD вне выборки), закрываемая добавкой")
        ax.set_yticks([])
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Тест охвата: 13 стран × 51–63 базиса; пунктир — пороги 5% и 20%", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "v2_encompassing.png")
    plt.close(fig)


def gradient():
    p = pd.read_csv(V2 / "gradient_panel.csv")
    p = p[(p.setting == "open") & (p.group == "all")]
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    for c, g in p.groupby("country"):
        col = MAIN.get(c, GRAY)
        ax.scatter(g.sigma, g.adv_rank, s=10, color=col, zorder=3 if c in MAIN else 2)
        if c in MAIN or g.adv_rank.mean() < 0.95:
            ax.text(g.sigma.mean(), g.adv_rank.min() - 0.012, c, fontsize=7, color=INK, ha="center")
    ax.set_xlabel("доля трудового дохода в выпуске σ")
    ax.set_ylabel("доля товарных базисов, которые труд обгоняет")
    ax.set_title("Градиент: преимущество труда (открытая постановка) и доля труда", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "v2_gradient.png")
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    theta_curve()
    decomposition()
    mixture()
    gradient()


if __name__ == "__main__":
    main()
