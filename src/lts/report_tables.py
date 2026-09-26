"""Build the report tables (markdown) and figures (PNG) from results/tables/*.

python -m lts.report_tables
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .levels import SUBSETS, mask_for
from .metrics import ratio_metrics

ROOT = Path(__file__).resolve().parents[2]
TAB = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"
OUT = ROOT / "results" / "report_tables.md"

BLUE, ORANGE, AQUA, GRAY, INK, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#b9b8b1", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "figure.dpi": 130})

NAMED_FIG = {"k:D35": "электроэнергия", "k:C19": "нефтепродукты", "k:B": "добыча", "k:C24": "металлы",
             "k:A01": "с/х", "k:C10-12": "пищевая", "k:C20": "химия", "k:H49": "транспорт", "k:K64": "финансы"}
NAMED_BEA = {"k:22": "электро/газ/вода", "k:324": "нефтепродукты", "k:211": "нефть и газ", "k:331": "металлы",
             "k:111CA": "с/х", "k:311FT": "пищевая", "k:325": "химия", "k:484": "автоперевозки", "k:521CI": "банки"}
SYS = {"False": "открытая", "True": "замкнутая (факт. з/п)", "uniform": "замкнутая (единая з/п)"}


def kind(b: pd.Series) -> np.ndarray:
    return np.select([b.str.startswith("k:"), b.str.startswith("placebo_random"), b.str.startswith("placebo_perm"),
                      b.str.startswith("pp_")], ["commodity", "plac_rand", "plac_perm", "pp"], "labour")


def load_metrics():
    ms = []
    for tag in ["main", "bea", "robust"]:
        p = TAB / f"metrics_long_{tag}.csv"
        if p.exists():
            m = pd.read_csv(p, low_memory=False)
            m["closed"] = m["closed"].astype(str)
            m["dataset"] = tag
            ms.append(m)
    m = pd.concat(ms, ignore_index=True)
    m["kind"] = kind(m.basis)
    return m


def md(df: pd.DataFrame, floatfmt=3) -> str:
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{v:.{floatfmt}f}")
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)) + "\n"


# ------------------------------------------------------------------------------------------------
def table_levels(m: pd.DataFrame, metric="mawd", subset="all", capital=True, imports="price") -> pd.DataFrame:
    g = m[(m.subset == subset) & (m.capital == capital) & (m.imports == imports)]
    rows = []
    for (c, closed), gc in g.groupby(["country", "closed"]):
        lab = gc[(gc.kind == "labour") & (gc.labour == "hours")]
        L = lab[metric].mean()
        com = gc[gc.kind == "commodity"].groupby("basis")[metric].mean()
        share = []
        for y, gy in gc.groupby("year"):
            ly = gy[(gy.kind == "labour") & (gy.labour == "hours")][metric]
            cy = gy[gy.kind == "commodity"][metric]
            if len(ly) and len(cy):
                share.append((cy < ly.iloc[0]).mean())
        pr = gc[gc.kind == "plac_rand"].groupby("basis")[metric].mean()
        pp = gc[gc.kind == "plac_perm"].groupby("basis")[metric].mean()
        rows.append({"страна": c, "система": SYS[closed],
                     "труд (часы)": L,
                     "труд (з/п)": gc[(gc.kind == "labour") & (gc.labour == "wage_weighted")][metric].mean(),
                     "труд (образ.)": gc[(gc.kind == "labour") & (gc.labour == "edu_years")][metric].mean(),
                     "PP един. з/п": gc[gc.basis == "pp_uniform_wage"][metric].mean(),
                     "PP факт. з/п": gc[(gc.basis == "pp_actual_wage")][metric].mean(),
                     "товары: лучший": com.min() if len(com) else np.nan,
                     "товары: медиана": com.median() if len(com) else np.nan,
                     "доля товаров лучше труда": np.mean(share) if share else np.nan,
                     "плацебо случ.: 5-й перц.": pr.quantile(0.05) if len(pr) else np.nan,
                     "плацебо перест.: 5-й перц.": pp.quantile(0.05) if len(pp) else np.nan})
    return pd.DataFrame(rows)


def table_named(m: pd.DataFrame, metric="mawd", subset="all", capital=True) -> pd.DataFrame:
    g = m[(m.subset == subset) & (m.capital == capital) & (m.imports == "price")]
    rows = []
    for (c, closed), gc in g.groupby(["country", "closed"]):
        names = NAMED_BEA if c == "USA_BEA" else NAMED_FIG
        r = {"страна": c, "система": SYS[closed],
             "труд": gc[(gc.kind == "labour") & (gc.labour == "hours")][metric].mean()}
        for b, n in names.items():
            r[n] = gc[gc.basis == b][metric].mean()
        rows.append(r)
    return pd.DataFrame(rows)


def table_size_bias(m: pd.DataFrame) -> pd.DataFrame:
    g = m[(m.subset == "all") & (m.capital == True) & (m.imports == "price") & (m.closed == "False")]
    rows = []
    for c, gc in g.groupby("country"):
        lab = gc[(gc.kind == "labour") & (gc.labour == "hours")]
        pr = gc[gc.kind == "plac_rand"]
        pe = gc[gc.kind == "plac_perm"]
        com = gc[gc.kind == "commodity"]
        rows.append({"страна": c,
                     "ρ совок. труд": lab.corr_totals.mean(),
                     "ρ совок. плацебо (медиана)": pr.corr_totals.median(),
                     "ρ совок. перестановки (медиана)": pe.corr_totals.median(),
                     "ρ совок. товары (медиана)": com.corr_totals.median(),
                     "ρ(лог) совок. труд": lab.corr_log_totals.mean(),
                     "ρ(лог) плацебо (медиана)": pr.corr_log_totals.median(),
                     "ρ без масштаба, труд": lab.corr_desized.mean(),
                     "ρ без масштаба, плацебо (медиана)": pr.corr_desized.median(),
                     "CV труд": lab.cv.mean(), "CV плацебо (медиана)": pr.cv.median(),
                     "d труд": lab.d.mean(), "d плацебо (медиана)": pr.d.median()})
    return pd.DataFrame(rows)


def table_robust_subsets(m: pd.DataFrame, metric="mawd") -> pd.DataFrame:
    g = m[(m.capital == True) & (m.imports == "price") & (m.closed.isin(["False", "uniform"]))]
    rows = []
    for (c, sub, closed), gc in g.groupby(["country", "subset", "closed"]):
        L = gc[(gc.kind == "labour") & (gc.labour == "hours")].groupby("year")[metric].mean()
        share = []
        for y, gy in gc.groupby("year"):
            cy = gy[gy.kind == "commodity"][metric]
            if y in L.index and len(cy):
                share.append((cy < L[y]).mean())
        rows.append({"страна": c, "подвыборка": sub, "система": SYS[closed], "труд (часы)": L.mean(),
                     "PP факт. з/п": gc[gc.basis == "pp_actual_wage"][metric].mean(),
                     "доля товаров лучше труда": np.mean(share) if share else np.nan})
    return pd.DataFrame(rows)


def table_imports(m: pd.DataFrame) -> pd.DataFrame:
    g = m[(m.subset == "all") & (m.kind == "labour") & (m.labour == "hours") & (m.closed == "False")]
    t = g.groupby(["country", "capital", "imports"])[["mawd", "d"]].mean().unstack("imports")
    t.columns = [f"{a} ({b})" for a, b in t.columns]
    t = t.reset_index()
    mr = TAB / "mrio_ratios.parquet"
    if mr.exists():
        r = pd.read_parquet(mr)
        rows = []
        for (c, b, y), gg in r.groupby(["country", "basis", "year"]):
            mm = ratio_metrics(gg.z.to_numpy(), gg.x.to_numpy(), mask_for(gg.industry.tolist(), set()))
            rows.append(dict(country=c, basis=b, year=y, mawd=mm["mawd"], d=mm["d"]))
        mt = pd.DataFrame(rows).groupby(["country", "basis"])[["mawd", "d"]].mean().unstack("basis")
        mt.columns = [f"{a} {b}" for a, b in mt.columns]
        return t, mt.reset_index()
    return t, None


def table_exploitation() -> pd.DataFrame:
    rows = []
    for tag in ["main", "bea"]:
        p = TAB / f"economy_extra_{tag}.json"
        if not p.exists():
            continue
        ex = pd.read_json(p)
        for c, g in ex.groupby("country"):
            cols = [k for k in g.columns if k.startswith("exploit_")]
            vals = g[cols].mean()
            L = vals.get("exploit_L", np.nan)
            other = vals.drop("exploit_L", errors="ignore").dropna()
            rows.append({"страна": c, "e(труд) = норма приб. стоимости": L,
                         "мин e(k)": other.min(), "медиана e(k)": other.median(), "макс e(k)": other.max(),
                         "доля товаров с e(k)>0": (other > 0).mean(),
                         "r факт. (с капиталом)": g.r_actual_capital.mean(), "R макс.": g.R_max_capital.mean(),
                         "r собств. (Маркс/Шейх)": g.r_eigen_capital.mean(),
                         "s (единая з/п)": g.uniform_wage_scale.mean() if "uniform_wage_scale" in g else np.nan})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------
def fig_levels(m: pd.DataFrame, metric="mawd"):
    g = m[(m.subset == "all") & (m.capital == True) & (m.imports == "price") & (m.dataset.isin(["main", "bea"]))]
    countries = [c for c in ["DEU", "MEX", "USA", "USA_BEA"] if c in set(g.country)]
    systems = ["False", "True", "uniform"]
    fig, axes = plt.subplots(1, len(countries), figsize=(3.1 * len(countries), 3.6), sharey=True)
    for ax, c in zip(np.atleast_1d(axes), countries):
        gc = g[g.country == c]
        for i, s in enumerate(systems):
            gs = gc[gc.closed == s]
            com = gs[gs.kind == "commodity"].groupby("basis")[metric].mean()
            jit = np.random.default_rng(0).uniform(-0.18, 0.18, len(com))
            ax.scatter(i + jit, com.values, s=10, color=GRAY, zorder=2, label="товарные базисы" if i == 0 else None)
            pr = gs[gs.kind == "plac_rand"].groupby("basis")[metric].mean()
            if len(pr):
                ax.plot([i - 0.28, i + 0.28], [pr.median()] * 2, color=ORANGE, lw=2, zorder=3,
                        label="плацебо (медиана)" if i == 0 else None)
            L = gs[(gs.kind == "labour") & (gs.labour == "hours")][metric].mean()
            ax.scatter([i], [L], s=60, color=BLUE, edgecolor="white", linewidth=1.5, zorder=4,
                       label="труд (часы)" if i == 0 else None)
        ax.set_yscale("log")
        ax.set_xticks(range(3), ["откр.", "замкн.\nфакт.", "замкн.\nединая"])
        ax.set_title(c, color=INK, fontsize=10)
    np.atleast_1d(axes)[0].set_ylabel("MAWD (лог. шкала), меньше = ближе к ценам")
    np.atleast_1d(axes)[0].legend(loc="upper right", frameon=False, fontsize=7)
    fig.suptitle("Труд среди всех товарных базисов: асимметричная и симметричная постановки", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_levels_bases.png")
    plt.close(fig)


def fig_size_bias(m: pd.DataFrame):
    g = m[(m.subset == "all") & (m.capital == True) & (m.imports == "price") & (m.closed == "False")
          & (m.dataset.isin(["main", "bea"]))]
    countries = [c for c in ["DEU", "MEX", "USA", "USA_BEA"] if c in set(g.country)]
    fig, axes = plt.subplots(2, len(countries), figsize=(3.0 * len(countries), 4.6))
    for j, c in enumerate(countries):
        gc = g[g.country == c]
        for i, (col, lab) in enumerate([("corr_totals", "ρ совокупных величин (смещ. масштабом)"),
                                        ("mawd", "MAWD (масштабно-инвариантная)")]):
            ax = axes[i, j]
            pr = gc[gc.kind == "plac_rand"].groupby("basis")[col].mean()
            ax.hist(pr.values, bins=25, color=GRAY, edgecolor="white", linewidth=0.5, label="случайные базисы")
            L = gc[(gc.kind == "labour") & (gc.labour == "hours")][col].mean()
            ax.axvline(L, color=BLUE, lw=2, label="труд (часы)")
            ax.set_title(f"{c}: {lab}" if j == 0 else c, fontsize=8, color=INK)
            ax.set_yticks([])
    axes[0, 0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.suptitle("Корреляция совокупных величин почти не отличает труд от случайного базиса", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_size_bias.png")
    plt.close(fig)


def dyn_tables():
    out = {}
    for tag in ["main", "bea"]:
        f1 = TAB / f"dyn_T1_{tag}.csv"
        if not f1.exists():
            continue
        out[tag] = {k: pd.read_csv(TAB / f"dyn_{k}_{tag}.csv") for k in ["T1", "T2", "T3", "T4"]}
    return out


def summarise_dyn(T: pd.DataFrame, value: str, keys: list[str]) -> pd.DataFrame:
    T = T.copy()
    T["kind"] = kind(T.basis)
    rows = []
    for k, g in T.groupby(keys):
        r = dict(zip(keys, k if isinstance(k, tuple) else (k,)))
        for b in ["labour", "pp_uniform_wage", "pp_actual_wage", "pp_eigen"]:
            v = g.loc[g.basis == b, value]
            r[b] = v.mean() if len(v) else np.nan
        for kd in ["commodity", "plac_rand", "plac_perm"]:
            v = g.loc[g.kind == kd, value]
            if len(v):
                r[f"{kd} медиана"] = v.median()
                r[f"{kd} 95-й перц."] = v.quantile(0.95)
                if np.isfinite(r["labour"]):
                    r[f"доля {kd} выше труда"] = (v > r["labour"]).mean()
        rows.append(r)
    return pd.DataFrame(rows)


def fig_dynamics(D: dict):
    items = [(tag, c) for tag, t in D.items() for c in sorted(t["T1"].country.unique())]
    if not items:
        return
    fig, axes = plt.subplots(1, len(items), figsize=(2.9 * len(items), 3.4), sharey=True)
    for ax, (tag, c) in zip(np.atleast_1d(axes), items):
        T = D[tag]["T1"]
        T = T[(T.country == c) & (T.spec == "closedU_cap") & (T.subset == "all")].copy()
        T0 = D[tag]["T1"]
        T0 = T0[(T0.country == c) & (T0.spec == "open_cap") & (T0.subset == "all")].copy()
        for i, h in enumerate([1, 3, 5]):
            for off, TT, col in [(-0.17, T0, GRAY), (0.17, T, AQUA)]:
                g = TT[TT.h == h]
                com = g[g.basis.str.startswith("k:")].r2_w
                ax.scatter(i + off + np.random.default_rng(1).uniform(-0.07, 0.07, len(com)), com, s=7, color=col, zorder=2)
            g = T0[T0.h == h]
            L = g[g.basis == "labour"].r2_w
            if len(L):
                ax.scatter([i], [L.iloc[0]], s=55, color=BLUE, edgecolor="white", linewidth=1.5, zorder=4)
            pa = g[g.basis == "pp_actual_wage"].r2_w
            if len(pa):
                ax.scatter([i], [pa.iloc[0]], s=45, marker="D", color=ORANGE, edgecolor="white", zorder=4)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_ylim(-3, 1)
        ax.set_xticks(range(3), ["h=1", "h=3", "h=5"])
        ax.set_title(c, fontsize=10, color=INK)
    ax0 = np.atleast_1d(axes)[0]
    ax0.set_ylabel("R² при β=1 (доля дисперсии отн. цен)")
    from matplotlib.lines import Line2D
    ax0.legend(handles=[Line2D([], [], marker="o", ls="", color=BLUE, label="труд"),
                        Line2D([], [], marker="D", ls="", color=ORANGE, label="PP факт. з/п"),
                        Line2D([], [], marker="o", ls="", color=GRAY, label="товары, откр."),
                        Line2D([], [], marker="o", ls="", color=AQUA, label="товары, замкн. единая")],
               frameon=False, fontsize=7, loc="lower left")
    fig.suptitle("Изменения относительных цен и изменения содержания базиса на реальную единицу (T1)", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_dynamics_T1.png")
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    m = load_metrics()
    parts = ["# Автоматически сгенерированные таблицы\n\n(python -m lts.report_tables; средние по годам)\n"]
    for sub in ["all", "core"]:
        for cap in (True, False):
            parts.append(f"\n## Уровни: MAWD, подвыборка `{sub}`, капитал={cap}, импорт по цене\n\n")
            parts.append(md(table_levels(m, "mawd", sub, cap)))
    parts.append("\n## Уровни: d-метрика Steedman–Tomkins, `all`, капитал=True\n\n" + md(table_levels(m, "d", "all", True)))
    parts.append("\n## Уровни: взвешенное log-SD отношений z, `all`, капитал=True\n\n" + md(table_levels(m, "log_sd_w", "all", True)))
    parts.append("\n## Именованные базисы (MAWD, `all`, капитал=True)\n\n" + md(table_named(m)))
    parts.append("\n## Эффект масштаба (открытая система, капитал)\n\n" + md(table_size_bias(m)))
    parts.append("\n## Устойчивость к исключению секторов (MAWD, капитал)\n\n" + md(table_robust_subsets(m)))
    ti, tm = table_imports(m)
    parts.append("\n## Импорт: национальные варианты (труд, часы, открытая)\n\n" + md(ti))
    if tm is not None:
        parts.append("\n## Импорт: MRIO (занятые, без капитала)\n\n" + md(tm))
    parts.append("\n## Обобщённая «эксплуатация» товаров (замкнутая система, конкурентный импорт, без капитала)\n\n"
                 + md(table_exploitation()))
    val = TAB / "validation_zachariah_like.csv"
    if val.exists():
        Z = pd.read_csv(val).groupby("country")[["n", "mawd_ltv", "mawd_tpp", "mawd_ltv_hours", "rho_totals_ltv"]].mean()
        parts.append("\n## Валидация: методика Zachariah (2006)\n\n" + md(Z.reset_index()))
    D = dyn_tables()
    for tag, t in D.items():
        parts.append(f"\n## Динамика ({tag}): T1 — R² при β=1\n\n"
                     + md(summarise_dyn(t["T1"], "r2_w", ["country", "spec", "subset", "h"])))
        parts.append(f"\n## Динамика ({tag}): T2 — регрессии, R² within\n\n"
                     + md(summarise_dyn(t["T2"], "r2_within", ["country", "spec", "subset", "lag"])))
        t2 = t["T2"]
        parts.append(f"\n## Динамика ({tag}): T2 — β для труда и цен производства\n\n"
                     + md(t2[t2.basis.isin(["labour", "pp_uniform_wage", "pp_actual_wage"])][
                         ["country", "spec", "subset", "basis", "lag", "n", "beta", "se", "r2_within"]]))
        t3 = t["T3"]
        wcols = [c for c in t3.columns if c.startswith("w_")]
        parts.append(f"\n## Динамика ({tag}): T3 — охватывающие веса (сумма = 1)\n\n"
                     + md(t3[["country", "spec", "subset", "set", "on", "n"] + wcols + ["var_opt"]]))
        parts.append(f"\n## Вне выборки ({tag}): T4 — RMSE лог-отклонения относительной цены\n\n"
                     + md(summarise_dyn(t["T4"].assign(neg=-t["T4"].rmse_anchor_mean), "rmse_anchor_mean",
                                        ["country", "spec", "subset", "test"]).merge(
                         t["T4"].groupby(["country", "spec", "subset", "test"]).rmse_naive_frozen_prices.first()
                         .reset_index(), on=["country", "spec", "subset", "test"])))
    OUT.write_text("".join(parts))
    fig_levels(m)
    fig_size_bias(m)
    fig_dynamics(D)
    print("written", OUT)


if __name__ == "__main__":
    main()
