"""Iteration 13a: estimators and the robustness / counting rules (pre_registration_v13a.md, 'Правило вывода' and
journal item 3 - answers to the pre-calculation review).

Within-phase trend: ln y_t = a_phase + b t + e over the years of all phases (common slope, own phase levels).
CI for b: the wider of
  * HAC (Newey-West, given lag) by observation order (years sorted; across a crisis gap the neighbouring phase
    years count as neighbours);
  * moving-block PAIRS bootstrap within phases: blocks of consecutive pairs (t, ln y_t), block length min(5, n_f),
    each phase rebuilt from its own blocks to its own length, the slope re-estimated by OLS with phase effects on
    the resampled sample; phases with fewer than 3 years are not resampled; percentile 90% interval.
(The draft code placed resampled values on the original time axis; that destroys the trend - review P23.1.)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from ..v8.rule import decide

Z90 = 1.6448536269514722
CONF, REF, UNINF = "подтверждено", "опровергнуто", "неинформативно"


def phase_mask(index, phases):
    lab = pd.Series(None, index=index, dtype=object)
    for nm, (a, b) in phases.items():
        lab[(index >= a) & (index <= b)] = nm
    return lab


def _prep(y, phases):
    y = pd.Series(y).dropna().sort_index()
    y = y[np.isfinite(y.to_numpy(float))]
    ph = phase_mask(y.index, phases)
    keep = ph.notna()
    return y[keep], ph[keep]


def within_slope(y, phases):
    """OLS of y on t with phase fixed effects (years outside phases dropped): slope, data, phase labels."""
    y, ph = _prep(y, phases)
    t = y.index.to_numpy(float)
    yd = y.to_numpy(float).copy()
    td = t.copy()
    for g in ph.unique():
        m = (ph == g).to_numpy()
        yd[m] -= yd[m].mean()
        td[m] -= td[m].mean()
    beta = float((td * yd).sum() / (td * td).sum())
    return beta, y, ph


def hac_se(y, ph, lag):
    X = pd.get_dummies(ph).astype(float)
    X["t"] = y.index.to_numpy(float)
    fit = sm.OLS(y.to_numpy(float), X.to_numpy(float)).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
    return float(fit.params[-1]), float(fit.bse[-1])


def block_idx(n, B, block, rng):
    bl = max(1, min(block, n))
    k = int(np.ceil(n / bl))
    starts = rng.integers(0, n - bl + 1, size=(B, k))
    return (starts[:, :, None] + np.arange(bl)[None, None, :]).reshape(B, -1)[:, :n]


def _phase_boot(tt, yy, B, rng, block=5):
    """Resampled (num, den) of the within-phase slope for one phase: pairs bootstrap; n < 3 kept fixed."""
    n = len(yy)
    if n < 3:
        td = tt - tt.mean()
        return np.full(B, float((td * (yy - yy.mean())).sum())), np.full(B, float((td * td).sum()))
    idx = block_idx(n, B, min(block, n), rng)
    tv, yv = tt[idx], yy[idx]
    td = tv - tv.mean(1, keepdims=True)
    num = (td * (yv - yv.mean(1, keepdims=True))).sum(1)
    den = (td * td).sum(1)
    return num, den


def block_boot(y, ph, B=2000, block=5, seed=113):
    rng = np.random.default_rng(seed)
    num = np.zeros(B)
    den = np.zeros(B)
    t = y.index.to_numpy(float)
    v = y.to_numpy(float)
    for g in pd.unique(ph):
        m = (ph == g).to_numpy()
        a, b = _phase_boot(t[m], v[m], B, rng, block)
        num += a
        den += b
    sl = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    return np.nanquantile(sl, [0.05, 0.95])


def form_of(label, lo, hi, theta0, delta):
    if label != REF:
        return ""
    return "экв." if (lo >= theta0 - delta and hi <= theta0 + delta) else "обр."


def full_label(label, form):
    return f"{label} ({form})" if form else label


def within_trend(y, phases, delta, direction, lag=4, B=2000, seed=113, boot=True, block=5):
    beta, yy, ph = within_slope(y, phases)
    _, se = hac_se(yy, ph, lag)
    lo, hi = beta - Z90 * se, beta + Z90 * se
    if boot:
        bl, bh = block_boot(yy, ph, B, block, seed)
        lo, hi = min(lo, bl), max(hi, bh)
    lab = decide(beta, lo, hi, 0.0, delta, direction)
    form = form_of(lab, lo, hi, 0.0, delta)
    return dict(est=beta, ci90_lo=lo, ci90_hi=hi, half_width=(hi - lo) / 2, se_hac=se, mde=2.8 * se, n=len(yy),
                delta=delta, direction=direction, label=lab, form=form, label_full=full_label(lab, form))


def single_trend(y, delta, direction, lag=4):
    """Trend of one series over a window without gaps: CI = wider of HAC (lag 4, normal) and t(n-2) OLS."""
    y = pd.Series(y).dropna().sort_index()
    t = y.index.to_numpy(float)
    X = sm.add_constant(t)
    ols = sm.OLS(y.to_numpy(float), X).fit()
    hac = sm.OLS(y.to_numpy(float), X).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
    b = float(ols.params[1])
    w = max(Z90 * float(hac.bse[1]), stats.t.ppf(0.95, len(y) - 2) * float(ols.bse[1]))
    lo, hi = b - w, b + w
    lab = decide(b, lo, hi, 0.0, delta, direction)
    form = form_of(lab, lo, hi, 0.0, delta)
    return dict(est=b, ci90_lo=lo, ci90_hi=hi, half_width=w, se_hac=float(hac.bse[1]), mde=2.8 * float(hac.bse[1]),
                n=len(y), delta=delta, direction=direction, label=lab, form=form, label_full=full_label(lab, form))


def phase_slopes(y, phases, lag=4):
    """Own slope of every phase (HAC SE) and the Wald test of equal slopes (HAC, chi2 with k-1 df)."""
    y, ph = _prep(y, phases)
    t = y.index.to_numpy(float)
    names = [g for g in phases if (ph == g).sum() >= 3]
    keep = ph.isin(names).to_numpy()
    y, ph, t = y[keep], ph[keep], t[keep]
    X = pd.DataFrame({f"c_{g}": (ph == g).astype(float).to_numpy() for g in names})
    for g in names:
        X[f"t_{g}"] = X[f"c_{g}"] * (t - t[(ph == g).to_numpy()].mean())
    fit = sm.OLS(y.to_numpy(float), X.to_numpy(float)).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
    k = len(names)
    b = fit.params[k:]
    V = fit.cov_params()[k:, k:]
    out = {f"slope_{g}": float(b[i]) for i, g in enumerate(names)}
    out.update({f"se_{g}": float(np.sqrt(V[i, i])) for i, g in enumerate(names)})
    if k >= 2:
        R = np.zeros((k - 1, k))
        for i in range(k - 1):
            R[i, i], R[i, i + 1] = 1.0, -1.0
        d = R @ b
        W = float(d @ np.linalg.pinv(R @ V @ R.T) @ d)
        out["wald_equal_slopes"] = W
        out["p_equal_slopes"] = float(stats.chi2.sf(W, k - 1))
    return out


def two_phase_diff(y, pa, pb, delta=0.002, direction=">", lag=4, B=2000, seed=116, block=5):
    """ln y = a_phase + b_A t 1[A] + b_B t 1[B]; estimate b_B - b_A. CI: wider of HAC and the pairs block bootstrap
    (each phase resampled from its own blocks; slopes re-estimated)."""
    y, ph = _prep(y, {"A": pa, "B": pb})
    t = y.index.to_numpy(float)
    X = pd.DataFrame({"cA": (ph == "A").astype(float).to_numpy(), "cB": (ph == "B").astype(float).to_numpy()})
    X["tA"] = X.cA * t
    X["tB"] = X.cB * t
    fit = sm.OLS(y.to_numpy(float), X.to_numpy()).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
    p = np.asarray(fit.params)
    cov = np.asarray(fit.cov_params())
    est = float(p[3] - p[2])
    se = float(np.sqrt(cov[3, 3] + cov[2, 2] - 2 * cov[2, 3]))
    lo, hi = est - Z90 * se, est + Z90 * se
    rng = np.random.default_rng(seed)
    sl = {}
    for g in ("A", "B"):
        m = (ph == g).to_numpy()
        a, b = _phase_boot(t[m], y.to_numpy(float)[m], B, rng, block)
        sl[g] = a / b
    bl, bh = np.nanquantile(sl["B"] - sl["A"], [0.05, 0.95])
    lo, hi = min(lo, bl), max(hi, bh)
    lab = decide(est, lo, hi, 0.0, delta, direction)
    form = form_of(lab, lo, hi, 0.0, delta)
    return dict(est=est, ci90_lo=lo, ci90_hi=hi, half_width=(hi - lo) / 2, se_hac=se, mde=2.8 * se, n=len(y),
                delta=delta, direction=direction, label=lab, form=form, label_full=full_label(lab, form),
                slope_A=float(p[2]), slope_B=float(p[3]))


def ratio_R(m, phases, crises, B=2000, seed=113, theta0=1.0, delta=0.25):
    """R = mean m in crisis windows / mean m within phases. CI = wider of (i) stratified block bootstrap of years
    (blocks min(3, window) inside every phase and window) and (ii) cluster bootstrap: crisis windows resampled as
    whole units with replacement, phase years by blocks of 5 (review P7: three events, not 13 years)."""
    m = pd.Series(m).dropna()
    def groups(wins):
        return [m[(m.index >= a) & (m.index <= b)].to_numpy() for a, b in wins]
    gc = [g for g in groups(crises) if len(g)]
    gp = [g for g in groups(phases.values()) if len(g)]
    num = np.concatenate(gc).mean()
    den = np.concatenate(gp).mean()
    base = dict(mean_crisis=num, mean_phase=den, n_crisis_windows=len(gc), theta0=theta0, delta=delta, direction=">")
    if den <= 0:
        return dict(est=np.nan, ci90_lo=np.nan, ci90_hi=np.nan, label="неинформативно (среднее внутри фаз ≤ 0)",
                    form="", label_full="неинформативно (среднее внутри фаз ≤ 0)", **base)
    rng = np.random.default_rng(seed)
    def boot_years(gs, bl):
        tot = np.zeros(B)
        cnt = 0
        for g in gs:
            idx = block_idx(len(g), B, min(bl, len(g)), rng)
            tot += g[idx].sum(1)
            cnt += len(g)
        return tot / cnt
    r1 = boot_years(gc, 3) / boot_years(gp, 3)
    # cluster bootstrap over crisis windows
    pick = rng.integers(0, len(gc), size=(B, len(gc)))
    sums = np.array([g.sum() for g in gc])
    lens = np.array([len(g) for g in gc])
    cm = sums[pick].sum(1) / lens[pick].sum(1)
    r2 = cm / boot_years(gp, 5)
    q1 = np.nanquantile(r1, [0.05, 0.95])
    q2 = np.nanquantile(r2, [0.05, 0.95])
    lo, hi = min(q1[0], q2[0]), max(q1[1], q2[1])
    est = num / den
    lab = decide(est, lo, hi, theta0, delta, ">")
    form = form_of(lab, lo, hi, theta0, delta)
    return dict(est=est, ci90_lo=lo, ci90_hi=hi, ci_years_lo=q1[0], ci_years_hi=q1[1], ci_windows_lo=q2[0],
                ci_windows_hi=q2[1], label=lab, form=form, label_full=full_label(lab, form), **base)


# ---------------------------------------------------------------- robustness (journal item 3, answer to P18)
def opposite(main, other):
    """Decisive opposite label: for 'confirmed' - 'refuted (reverse)'; for 'refuted (reverse)' - 'confirmed';
    for 'refuted (equivalence)' - 'confirmed' or 'refuted (reverse)'; for 'uninformative' - none."""
    if main == CONF:
        return other == f"{REF} (обр.)"
    if main == f"{REF} (обр.)":
        return other == CONF
    if main == f"{REF} (экв.)":
        return other in (CONF, f"{REF} (обр.)")
    return False


def robust_flag(main, nine=(), lam=(), grid=(), nber=None, spec=()):
    """Robust if: all 9 statistical combinations give the main label; all lambda-before-1947 variants give it
    (task rule); over the full boundary grid no decisive opposite label and the main label in >= 2/3; the NBER
    variant is not decisive-opposite (veto); over the specification layer no decisive opposite and >= 2/3."""
    if main.startswith(UNINF):
        return False, "основная метка «неинформативно»"
    why = []
    if any(x != main for x in nine):
        why.append("9 сочетаний: метка меняется")
    if any(x != main for x in lam):
        why.append("λ до 1947 г.: метка меняется")
    if len(grid):
        if any(opposite(main, x) for x in grid):
            why.append("сетка границ: есть противоположная метка")
        if np.mean([x == main for x in grid]) < 2 / 3:
            why.append("сетка границ: доля основной метки < 2/3")
    if nber is not None and opposite(main, nber):
        why.append("NBER: противоположная метка")
    if len(spec):
        if any(opposite(main, x) for x in spec):
            why.append("спецификационный слой: есть противоположная метка")
        if np.mean([x == main for x in spec]) < 2 / 3:
            why.append("спецификационный слой: доля основной метки < 2/3")
    return (not why), "; ".join(why)


# ---------------------------------------------------------------- pairwise counting (journal item 3, P17, P23.10)
def count(pred, est, lo, hi, delta, theta0=0.0, homogeneous=True):
    """'за' / 'против' / 'нейтрально' for one theory's prediction.
    pred: '>' | '<' | '≈0' | '>=0' (forbids < 0) | '<=0' (forbids > 0) | 'молчит'.
    Sign predictions: for - CI excludes theta0 and the estimate is beyond the zone on the predicted side
    ('confirmed'); against - CI entirely on the other side or inside the zone.
    '≈0': for - CI inside the zone (and, for within-phase common slopes, the equal-slopes test not rejected at 10%);
    against - CI entirely outside the zone. '>=0': against - confirmed fall; for - CI entirely above theta0;
    '(экв.)' - neutral."""
    if pred in ("молчит", "", None) or not (np.isfinite(lo) and np.isfinite(hi)):
        return "нейтрально"
    e, l, h = est - theta0, lo - theta0, hi - theta0
    conf_up = l > 0 and e > delta
    conf_dn = h < 0 and e < -delta
    equiv = l >= -delta and h <= delta
    outside = l > delta or h < -delta
    if pred == ">":
        return "за" if conf_up else ("против" if (h < 0 or equiv) else "нейтрально")
    if pred == "<":
        return "за" if conf_dn else ("против" if (l > 0 or equiv) else "нейтрально")
    if pred == "≈0":
        if equiv:
            return "за" if homogeneous else "нейтрально (≈ 0 при разнонаправленных фазах)"
        return "против" if outside else "нейтрально"
    if pred == ">=0":
        return "против" if conf_dn else ("за" if l > 0 else "нейтрально")
    if pred == "<=0":
        return "против" if conf_up else ("за" if h < 0 else "нейтрально")
    raise ValueError(pred)
