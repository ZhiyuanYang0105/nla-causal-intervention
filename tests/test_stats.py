"""Tests for the stats module on a synthetic dataset with KNOWN signal:

    C0 (baseline) high FVE;  C1/C2/C3 (semantic-preserving) ~no drop;  C4 (drift) big drop.
This is an H1-like ground truth, so the analysis should flag C4 (not C1) as significant.
"""
import numpy as np
import pandas as pd
import pytest

from nla_intervention import stats as S

N = 60


def _synth(seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    base = rng.normal(0.80, 0.05, size=N)            # per-input baseline FVE
    # condition: (fve drop mean, sim mean, surface shift mean)
    # H1 ground truth: semantic-preserving conditions have NO systematic FVE drop
    # (centered at 0), only the drift condition drops. Significance should track that.
    spec = {
        "C0": (0.00, 1.00, 0.00),
        "C1": (0.00, 0.92, 0.30),   # light paraphrase: no drop, semantics kept
        "C2": (0.00, 0.88, 0.60),   # strong paraphrase
        "C3": (0.00, 0.90, 0.40),   # format-preserving
        "C4": (0.30, 0.45, 0.50),   # semantic drift: big drop, semantics broken
    }
    for i in range(N):
        for cond, (drop_mu, sim_mu, surf_mu) in spec.items():
            fve = base[i] - rng.normal(drop_mu, 0.02)
            rows.append({
                "input_id": f"a{i}", "condition": cond, "domain": "syn",
                "fve": fve,
                "sim_zz_prime": float(np.clip(rng.normal(sim_mu, 0.03), 0, 1)),
                "len_ratio": float(rng.normal(1.0, 0.05)),
                "jaccard_tokens": float(np.clip(1 - surf_mu + rng.normal(0, 0.05), 0, 1)),
                "ngram_overlap": float(np.clip(1 - surf_mu + rng.normal(0, 0.05), 0, 1)),
                "edit_distance_norm": float(np.clip(surf_mu + rng.normal(0, 0.05), 0, 1)),
                "js_divergence": float(np.clip(surf_mu + rng.normal(0, 0.05), 0, 1)),
            })
    return pd.DataFrame(rows)


def test_manipulation_check_pass_rates():
    df = _synth()
    res = S.manipulation_check(df, tau_keep=0.85, tau_drift=0.60)
    # C1 mostly passes the keep threshold; C4 mostly passes the drift threshold
    assert res["pass_rate_by_condition"]["C1"] > 0.8
    assert res["pass_rate_by_condition"]["C4"] > 0.8


def test_omnibus_friedman_detects_difference():
    df = _synth()
    res = S.omnibus(df, value="fve", method="friedman")
    assert res["method"] == "friedman"
    assert res["p_value"] < 1e-6          # conditions clearly differ
    assert res["n"] == N and res["k_conditions"] == 5


def test_rm_anova_runs():
    df = _synth()
    res = S.omnibus(df, value="fve", method="rm_anova")
    assert res["p_value"] < 1e-6


def test_pairwise_flags_drift_not_light_paraphrase():
    df = _synth()
    tbl = S.pairwise_vs_baseline(df, value="fve", baseline="C0").set_index("condition")
    # drift (C4) is a large, significant drop; light paraphrase (C1) is not
    assert tbl.loc["C4", "significant"]
    assert tbl.loc["C4", "mean_delta"] > 0.2
    assert not tbl.loc["C1", "significant"]
    # effect size ordering: C4 >> C1
    assert tbl.loc["C4", "cohen_dz"] > tbl.loc["C1", "cohen_dz"]


def test_effect_sizes_and_ci_present():
    df = _synth()
    tbl = S.pairwise_vs_baseline(df, value="fve", baseline="C0")
    for col in ("cohen_dz", "rank_biserial", "ci_lo", "ci_hi", "p_corrected"):
        assert col in tbl.columns
    assert (tbl["ci_lo"] <= tbl["ci_hi"]).all()


def test_surface_shift_composite():
    df = _synth()
    ss = S.compute_surface_shift(df)
    assert len(ss) == len(df)
    # higher composite for higher-distortion conditions (C2) than near-zero (C0)
    d = df.assign(surface_shift=ss).groupby("condition")["surface_shift"].mean()
    assert d["C2"] > d["C0"]


def test_mixed_model_fits():
    df = _synth()
    res = S.mixed_model(df, "fve ~ condition + len_ratio + (1|input_id)")
    assert hasattr(res, "params")
    assert any("condition" in str(k) for k in res.params.index)


def test_mechanism_regression_runs():
    df = _synth()
    res = S.mechanism_regression(df)
    # surface_shift and sim_zz_prime are estimated on the semantic-preserving subset
    assert "surface_shift" in res.params.index
    assert "sim_zz_prime" in res.params.index


def test_power_analysis():
    pr = S.power_analysis(observed_dz=0.5, n=60, alpha=0.05)
    assert 0.0 <= pr.power <= 1.0
    assert pr.required_n > 0
