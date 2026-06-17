"""Guards the gated intervention analysis in scripts/steg_intervention.py.

Loads the script module and exercises analyze() on synthetic metrics tables — verifying
the manipulation gate (invalid drift control is caught) and the verdict logic, with no
models involved.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

_SPEC = importlib.util.spec_from_file_location(
    "steg", Path(__file__).resolve().parents[1] / "scripts/steg_intervention.py")
steg = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(steg)


def _synth(drift_works=True, n=80, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(0.6, 0.1, n)
    spec = {"C0": (0.0, 1.0, 0.0), "C1": (0.0, 0.90, 0.3), "C2": (0.0, 0.88, 0.55),
            "C3": (0.0, 0.91, 0.4), "C4": (0.25, 0.45 if drift_works else 0.80, 0.5)}
    rows = []
    for i in range(n):
        for c, (d, sm, su) in spec.items():
            cos = base[i] - rng.normal(d, 0.02)
            sq_base = rng.uniform(0.8, 1.2)
            rows.append(dict(input_id=f"a{i}", condition=c, z="x", z_prime="y",
                             cosine=cos, sq_err=sq_base * (1 - cos), sq_base=sq_base, fve_ps=cos,
                             sim_zz_prime=float(np.clip(rng.normal(sm, 0.03), 0, 1)),
                             jaccard_tokens=float(np.clip(1 - su, 0, 1)),
                             ngram_overlap=float(np.clip(1 - su, 0, 1)),
                             edit_distance_norm=float(np.clip(su, 0, 1)),
                             js_divergence=float(np.clip(su, 0, 1)),
                             len_ratio=float(rng.normal(1, 0.05))))
    return pd.DataFrame(rows)


def test_broken_drift_control_is_caught():
    rep = steg.analyze(_synth(drift_works=False))
    assert rep["drift_control_ok"] is False
    assert rep["verdict"].startswith("INVALID-CONTROL")


def test_working_control_runs_full_analysis():
    rep = steg.analyze(_synth(drift_works=True))
    assert rep["drift_control_ok"] is True
    # set-level FVE reported per condition; drift (C4) reconstructs worse than C0
    assert rep["set_level_fve"]["C4"] < rep["set_level_fve"]["C0"]
    assert "INVALID-CONTROL" not in rep["verdict"]
    assert "friedman_cosine" in rep and "pairwise_cosine" in rep


def test_manipulation_filter_drops_failed_rows():
    # keep-condition rows below tau_keep should be filtered out of the primary analysis
    df = _synth(drift_works=True)
    rep = steg.analyze(df)
    assert rep["n_after_filter"] <= rep["n_rows"]
    assert 0.0 <= rep["manipulation_pass_rate"]["C1"] <= 1.0
