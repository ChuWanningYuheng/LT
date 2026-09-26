"""Tables (results/v3/tables_v3.md) and figures (results/figures/v3_*.png) for report v3."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..v2.figures import AQUA, BLUE, FIG, GRAY, INK, INK2, ORANGE
from .placebo_disp import COUNTRIES, OUT

VARS = ["1.1a_edu_price", "1.1b_occ_price", "1.2a_jz_time", "1.2b_jz_price", "1.3a_hilf", "1.3b_hilf_subs",
        "1.5a_frozen2010", "1.5b_foreign_median"]


def md(df, fmt="{:.2f}"):
    df = df.copy()
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join([df.index.name or ""] + cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for i, r in df.iterrows():
        cells = [fmt.format(v) if isinstance(v, (float, np.floating)) and np.isfinite(v) else
                 ("н/д" if isinstance(v, (float, np.floating)) else str(v)) for v in r.values]
        lines.append("| " + " | ".join([str(i)] + cells) + " |")
    return "\n".join(lines)


def cat(g):
    if not np.isfinite(g):
        return "н/о"
    return "почти полностью" if g > 0.8 else "существенно" if g > 0.5 else "частично" if g >= 0.25 else "мало"


def stage_p():
    out = []
    for f, tag in (("placebo_disp.csv", "после исправления JPN"), ("placebo_disp_prefix_jpn.csv", "до исправления JPN")):
        d = pd.read_csv(OUT / f)
        a = d[d["mask"] == "all"].groupby(["country", "kind", "metric"]).share_better.mean().unstack(["kind", "metric"])
        a.columns = [f"{k}_{m}" for k, m in a.columns]
        a = a[["perm_mawd", "perm_d", "lnorm_mawd", "lnorm_d", "mix_mawd", "mix_d"]]
        a["top5_ab"] = ((a[["perm_mawd", "perm_d", "lnorm_mawd", "lnorm_d"]] <= 0.05).all(1)).map({True: "да", False: "нет"})
        a["top5_c"] = ((a[["mix_mawd", "mix_d"]] <= 0.05).all(1)).map({True: "да", False: "нет"})
        a.index.name = "страна"
        out.append((tag, a))
    return out


def stage_p_masks():
    d = pd.read_csv(OUT / "placebo_disp.csv")
    a = d.groupby(["country", "mask", "kind", "metric"]).share_better.mean().unstack(["kind", "metric"])
    a.columns = [f"{k}_{m}" for k, m in a.columns]
    return a


def contrib_table(countries):
    c = pd.read_csv(OUT / "placebo_disp_contrib.csv")
    rows = []
    for k in countries:
        g = c[c.country == k].groupby("industry")[["contrib", "hours_share", "va_share", "z"]].mean()
        tot = g.contrib.sum()
        for ind, r in g.sort_values("contrib", ascending=False).head(5).iterrows():
            rows.append(dict(страна=k, отрасль=ind, доля_в_MAWD=r.contrib / tot, доля_часов=r.hours_share,
                             доля_ДС=r.va_share, z=r.z))
    return pd.DataFrame(rows).set_index("страна")


def gap_table():
    d = pd.read_csv(OUT / "reduction_levels.csv")
    d = d[d.kind == "err"]
    rows = []
    for (c, met), g in d.groupby(["country", "metric"]):
        p = g.pivot_table(index="year", columns="variant", values="value")
        for v in VARS:
            if v not in p:
                continue
            yrs = p[v].dropna().index
            eh, ew, ev = p.loc[yrs, "hours"].mean(), p.loc[yrs, "wagebill"].mean(), p.loc[yrs, v].mean()
            gap = eh - ew
            rows.append(dict(country=c, metric=met, variant=v, G=(eh - ev) / gap if gap > 0.01 * eh else np.nan))
    sb = pd.read_csv(OUT / "section_bound.csv").groupby(["country", "metric", "variant"]).value.mean().unstack()
    for (c, met), r in sb.iterrows():
        gap = r.hours - r.wagebill
        rows.append(dict(country=c, metric=met, variant="диагн.: зарплата, средняя по секции",
                         G=(r.hours - r.wage_section_mean) / gap if gap > 0.01 * r.hours else np.nan))
    return pd.DataFrame(rows)


def variant_summary(G):
    rows = []
    for v, g in G.groupby("variant"):
        r = {"вариант": v}
        cats = {}
        for met in ("mawd", "d"):
            s = g[g.metric == met].G.dropna()
            r[f"медиана G ({met})"] = s.median()
            c = s.map(cat).value_counts()
            cats[met] = c.idxmax() if len(c) else "н/о"
            r[f"стран: мало/част./сущ./почти ({met})"] = "/".join(str(int(((s < .25)).sum())) if k == 0 else
                                                              str(int(((s >= .25) & (s <= .5)).sum())) if k == 1 else
                                                              str(int(((s > .5) & (s <= .8)).sum())) if k == 2 else
                                                              str(int((s > .8).sum())) for k in range(4))
        r["итог"] = cats["mawd"] if cats["mawd"] == cats["d"] else "зависит от метрики"
        rows.append(r)
    return pd.DataFrame(rows).set_index("вариант")


def losers_table():
    d = pd.read_csv(OUT / "reduction_levels.csv")
    d = d[d.kind.isin(["pct_bases", "plac_perm_own", "plac_lnorm_own", "plac_mix_own"])]
    a = d.groupby(["country", "variant", "kind", "metric"]).value.mean().unstack(["kind", "metric"])
    a.columns = [f"{k.replace('plac_', '').replace('_own', '')}_{m}" for k, m in a.columns]
    keep = ["hours", "1.1a_edu_price", "1.1b_occ_price", "1.2b_jz_price", "1.3b_hilf_subs", "1.5b_foreign_median", "wagebill"]
    a = a.reset_index()
    a = a[a.variant.isin(keep)]
    return a


def figures():
    FIG.mkdir(parents=True, exist_ok=True)
    # 1. stage P
    d = pd.read_csv(OUT / "placebo_disp.csv")
    a = d[d["mask"] == "all"].groupby(["country", "kind", "metric"]).share_better.mean().unstack(["kind", "metric"])
    order = a[("mix", "d")].add(a[("mix", "mawd")]).sort_values().index
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    cols = {"perm": BLUE, "lnorm": AQUA, "mix": ORANGE}
    names = {"perm": "(а) перестановки часов", "lnorm": "(б) лог-нормальные", "mix": "(в) смеси товаров v2"}
    yy = np.arange(len(order))
    for ax, met in zip(axes, ("mawd", "d")):
        for i, (k, col) in enumerate(cols.items()):
            ax.scatter(a.loc[order, (k, met)], yy + (i - 1) * 0.22, color=col, s=16, label=names[k], zorder=3)
        ax.axvline(0.05, color=INK2, lw=0.8, ls="--")
        ax.set_title(met.upper() if met == "mawd" else "d (Steedman–Tomkins)", fontsize=9, color=INK)
        ax.set_xlabel("доля плацебо, которые лучше труда")
        ax.set_xlim(-0.03, 1.03)
    axes[0].set_yticks(yy)
    axes[0].set_yticklabels(order)
    axes[0].legend(frameon=False, fontsize=7, loc="lower right")
    fig.suptitle("Этап П: труд против плацебо разной неравномерности (13 стран, 2010–2022; пунктир 5%)",
                 fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "v3_placebo_types.png")
    plt.close(fig)
    # 2. gap closure heatmap (MAWD)
    G = gap_table()
    g = G[G.metric == "mawd"].pivot_table(index="variant", columns="country", values="G")
    g = g.reindex([v for v in VARS] + ["диагн.: зарплата, средняя по секции"])
    fig, ax = plt.subplots(figsize=(9, 3.8))
    im = ax.imshow(g.values.clip(-0.2, 1.2), cmap="Blues", vmin=-0.1, vmax=1.1, aspect="auto")
    ax.set_xticks(range(g.shape[1]))
    ax.set_xticklabels(g.columns, fontsize=8)
    ax.set_yticks(range(g.shape[0]))
    ax.set_yticklabels(g.index, fontsize=8)
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            v = g.values[i, j]
            ax.text(j, i, "н/о" if not np.isfinite(v) else f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if np.isfinite(v) and v > 0.6 else INK)
    ax.grid(False)
    ax.set_title("Этап 1: доля разрыва «часы → фонд оплаты», закрываемая редукцией (MAWD)", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "v3_gap_closure.png")
    plt.close(fig)
    # 3. premium vs PCM
    ind = pd.read_csv(OUT / "premium_industry.csv")
    ind["cy"] = ind.country + ind.year.astype(str)
    for col in ("premium", "pcm"):
        ind[col + "_dm"] = ind[col] - ind.groupby("cy")[col].transform("mean")
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.scatter(ind.pcm_dm, ind.premium_dm, s=2, color=GRAY, alpha=0.4)
    b = pd.qcut(ind.pcm_dm, 20)
    m = ind.groupby(b, observed=True)[["pcm_dm", "premium_dm"]].mean()
    ax.plot(m.pcm_dm, m.premium_dm, color=BLUE, lw=2)
    ax.set_xlim(-0.4, 0.5)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel("PCM отрасли − среднее по стране-году")
    ax.set_ylabel("ln надбавки − среднее")
    ax.set_title("Этап 2: отраслевая надбавка и валовая маржа (линия — 20 квантилей)", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "v3_premium_pcm.png")
    plt.close(fig)
    # 4. coverage vs DV
    p = pd.read_csv(OUT / "cc_panel_data.csv")
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for c, g in p.groupby("country"):
        g = g.dropna(subset=["cbc"])
        col = ORANGE if g.dv.mean() > 0.05 else BLUE
        ax.plot(g.cbc, g.dv, "-o", ms=2, lw=1, color=col)
        if len(g):
            ax.text(g.cbc.mean(), g.dv.mean() + 0.03, c, fontsize=7, color=INK, ha="center")
    ax.set_xlabel("охват коллективными договорами, % (OECD)")
    ax.set_ylabel("доля плацебо (в) лучше труда, ср. MAWD и d")
    ax.set_title("Этап 3: охват и проигрыш труда (траектории стран 2010–2022)", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "v3_coverage.png")
    plt.close(fig)


def main():
    parts = ["# Таблицы итерации 3 (генерируются `python -m lts.v3.summarize`)\n"]
    for tag, a in stage_p():
        parts += [f"## Этап П: доля плацебо лучше труда ({tag})\n", md(a), ""]
    m = stage_p_masks()
    for mask in ("no_prereg", "no_top3"):
        x = m.xs(mask, level="mask")
        x.index.name = "страна"
        parts += [f"## Этап П, исключение отраслей: {mask}\n", md(x), ""]
    parts += ["## П4: вклад отраслей в MAWD труда (5 наибольших)\n",
              md(contrib_table(["GBR", "KOR", "NLD", "POL", "USA"])), ""]
    G = gap_table()
    gm = G.pivot_table(index="variant", columns=["metric", "country"], values="G")
    for met in ("mawd", "d"):
        x = gm[met].copy()
        x.index.name = f"G ({met})"
        parts += [f"## Этап 1: доля закрытого разрыва G, {met}\n", md(x), ""]
    parts += ["## Этап 1: итог по вариантам\n", md(variant_summary(G)), ""]
    lt = losers_table()
    for c in ("USA", "GBR", "JPN", "KOR", "NLD", "POL"):
        x = lt[lt.country == c].set_index("variant").drop(columns="country")
        x.index.name = c
        parts += [f"## 3.4 / П5: процентили, {c}\n", md(x), ""]
    parts += ["## Этап 2a (объединённая панель)\n", md(pd.read_csv(OUT / "premium_2a_pooled.csv").set_index("x"), "{:.3f}"), ""]
    c = pd.read_csv(OUT / "premium_2a_corr.csv").groupby("country")[["corr_pcm", "corr_pcm_li", "corr_nos_cfc"]].mean()
    c.index.name = "страна"
    parts += ["## Этап 2a по странам (средняя корреляция)\n", md(c), ""]
    parts += ["## Этап 2b\n", md(pd.read_csv(OUT / "premium_2b.csv").set_index("country"), "{:.3f}"), ""]
    parts += ["## 3.1 Признаки построения данных (2015)\n", md(pd.read_csv(OUT / "cc_artefacts.csv").set_index("country"), "{:.3f}"), ""]
    h = pd.read_csv(OUT / "cc_harmonised.csv").groupby(["country", "vector", "kind", "metric"]).share_better.mean().unstack(["kind", "metric"])
    h.columns = [f"{k}_{m}" for k, m in h.columns]
    h = h.reset_index().set_index("country")
    parts += ["## 3.1 Гармонизация: занятые вместо часов\n", md(h), ""]
    r = pd.read_csv(OUT / "cc_panel_tests.csv")
    parts += ["## 3.3 Панель (FE страны и года), дикий кластерный бутстреп, Холм\n",
              md(r.set_index("candidate"), "{:.3f}"), ""]
    x = pd.read_csv(OUT / "cc_cross_country.csv")
    parts += ["## 3.3 Межстрановая корреляция Спирмена (13 стран, вспомогательно)\n", md(x.set_index("candidate"), "{:.3f}"), ""]
    cy = pd.read_csv(OUT / "cc_panel_data.csv").groupby("country")[["dv", "cbc", "sd_logwage", "import_share", "pcm_mean",
                                                                   "sd_premium", "share_BKL"]].mean()
    cy.index.name = "страна"
    parts += ["## Кандидаты и DV, средние по стране\n", md(cy, "{:.3f}"), ""]
    (OUT / "tables_v3.md").write_text("\n".join(parts))
    figures()


if __name__ == "__main__":
    main()
