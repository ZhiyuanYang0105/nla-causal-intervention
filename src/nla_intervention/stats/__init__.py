"""Paired statistical analysis (see docs/statistical_analysis.md).

Operates on the long metrics table: one row per (input_id, condition) with columns
fve, sim_zz_prime, len_ratio, and the token-shift metrics. Implemented with
scipy (Friedman/Wilcoxon) + statsmodels (Holm, mixed-effects, power).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


# ---- 0. manipulation check ------------------------------------------------

def manipulation_check(
    df: pd.DataFrame,
    tau_keep: float = 0.85,
    tau_drift: float = 0.60,
    keep_conditions=("C1", "C2", "C3"),
    drift_conditions=("C4",),
    sim_col: str = "sim_zz_prime",
) -> dict[str, Any]:
    """Verify semantic-preserving conditions kept meaning (sim>=tau_keep) and drift
    conditions changed it (sim<=tau_drift). Returns per-condition pass rates and the
    boolean mask of rows that PASS (use to filter the primary analysis)."""
    if sim_col not in df.columns:
        raise KeyError(f"{sim_col!r} not in df; run with an embedder to populate it")
    keep_set, drift_set = set(keep_conditions), set(drift_conditions)

    def _passes(row) -> bool:
        c = row["condition"]
        if c in keep_set:
            return row[sim_col] >= tau_keep
        if c in drift_set:
            return row[sim_col] <= tau_drift
        return True  # baseline / model-free conditions are not checked

    passed = df.apply(_passes, axis=1)
    rates = (
        df.assign(_pass=passed)
        .groupby("condition")["_pass"]
        .mean()
        .to_dict()
    )
    return {
        "pass_mask": passed,
        "pass_rate_by_condition": rates,
        "n_failed": int((~passed).sum()),
        "tau_keep": tau_keep,
        "tau_drift": tau_drift,
    }


# ---- helpers: wide pivot, effect sizes, bootstrap -------------------------

def _wide(df: pd.DataFrame, value: str) -> pd.DataFrame:
    """input_id x condition matrix of `value`, complete cases only (paired)."""
    w = df.pivot_table(index="input_id", columns="condition", values=value)
    return w.dropna(axis=0, how="any")


def cohen_dz(diff: np.ndarray) -> float:
    """Paired effect size: mean(diff)/sd(diff)."""
    diff = np.asarray(diff, dtype=np.float64)
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else float("nan")


def rank_biserial(diff: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation from signed ranks of `diff`.
    +1 => all pairs decreased toward the contrast direction."""
    diff = np.asarray(diff, dtype=np.float64)
    nz = diff[diff != 0]
    if nz.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nz))
    rp = ranks[nz > 0].sum()
    rm = ranks[nz < 0].sum()
    total = rp + rm
    return float((rp - rm) / total) if total > 0 else 0.0


def bootstrap_ci(diff: np.ndarray, n_boot: int = 5000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float]:
    """Bootstrap CI for the mean of paired differences (resample pairs)."""
    diff = np.asarray(diff, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, diff.size, size=(n_boot, diff.size))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


# ---- 1. omnibus -----------------------------------------------------------

def omnibus(df: pd.DataFrame, value: str = "fve", method: str = "friedman") -> dict[str, Any]:
    """Repeated-measures omnibus across conditions. method in {friedman, rm_anova}."""
    w = _wide(df, value)
    if w.shape[1] < 2:
        raise ValueError("need >=2 conditions for an omnibus test")
    if method == "friedman":
        stat, p = stats.friedmanchisquare(*[w[c].values for c in w.columns])
        return {"method": "friedman", "statistic": float(stat), "p_value": float(p),
                "n": int(w.shape[0]), "k_conditions": int(w.shape[1])}
    if method == "rm_anova":
        from statsmodels.stats.anova import AnovaRM

        long = w.reset_index().melt(id_vars="input_id", var_name="condition",
                                    value_name=value)
        res = AnovaRM(long, depvar=value, subject="input_id", within=["condition"]).fit()
        tbl = res.anova_table
        return {"method": "rm_anova", "F": float(tbl["F Value"].iloc[0]),
                "p_value": float(tbl["Pr > F"].iloc[0]), "n": int(w.shape[0])}
    raise ValueError(f"unknown method {method!r}")


# ---- 2. pairwise vs baseline ----------------------------------------------

def pairwise_vs_baseline(
    df: pd.DataFrame, value: str = "fve", baseline: str = "C0",
    test: str = "wilcoxon", correction: str = "holm",
    alternative: str = "greater",   # baseline > condition (i.e. FVE drops)
    n_boot: int = 5000, alpha: float = 0.05,
) -> pd.DataFrame:
    """Each condition vs baseline (paired) on `value`. Returns a tidy table with
    statistic, raw/corrected p, effect sizes, and bootstrap CI of mean ΔFVE."""
    from statsmodels.stats.multitest import multipletests

    w = _wide(df, value)
    if baseline not in w.columns:
        raise ValueError(f"baseline {baseline!r} not present")
    rows = []
    for cond in [c for c in w.columns if c != baseline]:
        b, x = w[baseline].values, w[cond].values
        diff = b - x  # positive => FVE dropped vs baseline
        if test == "wilcoxon":
            if np.allclose(diff, 0):
                stat, p = np.nan, 1.0
            else:
                stat, p = stats.wilcoxon(b, x, alternative=alternative)
        elif test == "ttest":
            stat, p = stats.ttest_rel(b, x)
            if alternative == "greater":
                p = p / 2 if stat > 0 else 1 - p / 2
        else:
            raise ValueError(f"unknown test {test!r}")
        lo, hi = bootstrap_ci(diff, n_boot=n_boot, alpha=alpha)
        rows.append({
            "condition": cond, "baseline": baseline, "n": len(diff),
            "mean_delta": float(diff.mean()), "statistic": float(stat),
            "p_raw": float(p), "cohen_dz": cohen_dz(diff),
            "rank_biserial": rank_biserial(diff), "ci_lo": lo, "ci_hi": hi,
        })
    out = pd.DataFrame(rows)
    if len(out):
        out["p_corrected"] = multipletests(out["p_raw"], alpha=alpha, method=correction)[1]
        out["significant"] = out["p_corrected"] < alpha
    return out


# ---- 3. surface_shift composite (dataset-level) ---------------------------

def compute_surface_shift(df: pd.DataFrame,
                          cols=("jaccard_tokens", "ngram_overlap",
                                "edit_distance_norm", "js_divergence")) -> pd.Series:
    """Standardized composite of token-shift metrics -> single surface-distortion axis.

    Overlap metrics (jaccard, ngram_overlap) are inverted so that higher = more shift.
    Returns first principal component (sign-aligned so higher = more surface change)."""
    X = df[list(cols)].copy()
    for c in ("jaccard_tokens", "ngram_overlap"):
        if c in X:
            X[c] = 1.0 - X[c]
    Xz = (X - X.mean()) / X.std(ddof=0).replace(0, 1.0)
    # first PC via SVD
    U, S, Vt = np.linalg.svd(Xz.values - Xz.values.mean(0), full_matrices=False)
    pc1 = (Xz.values @ Vt[0])
    # align sign so higher = more surface shift (guard against degenerate/constant inputs
    # where corrcoef is NaN — fall back to aligning with the mean of the shift features)
    ref = Xz.mean(axis=1).values
    if pc1.std() > 0 and ref.std() > 0:
        if np.corrcoef(pc1, ref)[0, 1] < 0:
            pc1 = -pc1
    elif float(np.dot(pc1, ref)) < 0:
        pc1 = -pc1
    return pd.Series(pc1, index=df.index, name="surface_shift")


# ---- 4. mixed-effects models ----------------------------------------------

def _parse_random_group(formula: str) -> tuple[str, str]:
    """Split 'y ~ a + b + (1|g)' into ('y ~ a + b', 'g')."""
    import re

    m = re.search(r"\(\s*1\s*\|\s*(\w+)\s*\)", formula)
    if not m:
        raise ValueError(f"no random intercept (1|group) in formula: {formula}")
    group = m.group(1)
    fixed = re.sub(r"\+?\s*\(\s*1\s*\|\s*\w+\s*\)", "", formula).strip().rstrip("+").strip()
    return fixed, group


def _fit_mixedlm(df: pd.DataFrame, formula: str):
    import statsmodels.formula.api as smf

    fixed, group = _parse_random_group(formula)
    return smf.mixedlm(fixed, df, groups=df[group]).fit()


def mixed_model(df: pd.DataFrame, formula: str = "fve ~ condition + len_ratio + (1|input_id)"):
    """Fit fve ~ condition + covariates + (1|input_id) (statsmodels MixedLM)."""
    return _fit_mixedlm(df, formula)


def mechanism_regression(
    df: pd.DataFrame,
    formula: str = "delta_fve ~ surface_shift + sim_zz_prime + len_ratio + (1|input_id)",
    keep_conditions=("C1", "C2", "C3"),
    baseline: str = "C0", value: str = "fve",
) -> Any:
    """Core semantic-vs-surface test on the semantic-preserving subset.

    Builds delta_fve (= fve_baseline - fve) and surface_shift if absent, restricts to
    keep_conditions, then fits `delta_fve ~ surface_shift + sim_zz_prime + len_ratio` with a
    random intercept per input. A significant POSITIVE surface_shift coefficient (sim
    controlled) => more surface change predicts a larger reconstruction drop = surface/
    steganographic channel (H2).

    Returns a fitted statsmodels result with `.params`/`.pvalues`. With paired within-input
    deltas the random intercept routinely collapses to the boundary (group var ~ 0); when
    that happens the fixed-effect inference is still valid, so we FALL BACK to OLS with
    cluster-robust SE by input — avoiding a fragile/boundary mixed-model p-value. The
    returned object carries `._nla_method` = "mixedlm" | "ols_clustered"."""
    import warnings
    import statsmodels.formula.api as smf

    d = df.copy()
    if "surface_shift" not in d.columns:
        d["surface_shift"] = compute_surface_shift(d)
    if "delta_fve" not in d.columns:
        base = d[d["condition"] == baseline].set_index("input_id")[value]
        d["delta_fve"] = d.apply(
            lambda r: base.get(r["input_id"], np.nan) - r[value], axis=1)
    sub = d[d["condition"].isin(keep_conditions)].dropna(
        subset=["delta_fve", "surface_shift"]).copy()
    fixed, group = _parse_random_group(formula)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = smf.mixedlm(fixed, sub, groups=sub[group]).fit()
    re_var = float(np.asarray(res.cov_re).ravel()[0]) if getattr(res, "cov_re", None) is not None else 0.0
    collapsed = re_var < 1e-3 or any(
        "boundary" in str(x.message).lower() or "converg" in str(x.message).lower() for x in w)
    if collapsed:                                          # RE collapsed -> OLS + cluster-robust SE
        res = smf.ols(fixed, sub).fit(cov_type="cluster", cov_kwds={"groups": sub[group]})
        res._nla_method = "ols_clustered"
    else:
        res._nla_method = "mixedlm"
    return res


# ---- 5. power -------------------------------------------------------------

@dataclass
class PowerResult:
    power: float
    required_n: float
    effect_dz: float
    alpha: float


def power_analysis(observed_dz: float, n: int, alpha: float = 0.05,
                   target_power: float = 0.80) -> PowerResult:
    """Paired-test power at the observed effect, and N required for target_power."""
    from statsmodels.stats.power import TTestPower

    tp = TTestPower()
    power = tp.power(effect_size=abs(observed_dz), nobs=n, alpha=alpha, alternative="larger")
    req_n = tp.solve_power(effect_size=abs(observed_dz), power=target_power, alpha=alpha,
                           alternative="larger")
    return PowerResult(power=float(power), required_n=float(req_n),
                       effect_dz=float(observed_dz), alpha=alpha)
