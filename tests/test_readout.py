"""Tests for the local-budget lightweight AR: bow features + closed-form ridge."""
import numpy as np
import pytest

from nla_intervention import metrics as M
from nla_intervention.conditions.rewriter import FakeRewriter
from nla_intervention.pipeline.readout import (
    RidgeReconstructor,
    hashing_ngram_features,
    make_bow_features,
)
from nla_intervention.pipeline.runner import run


def test_bow_features_surface_sensitive():
    a = hashing_ngram_features("the brown fox")
    assert np.array_equal(a, hashing_ngram_features("the brown fox"))   # deterministic
    b = hashing_ngram_features("the brown cat")                          # different token
    assert not np.array_equal(a, b)                                      # surface-sensitive


def test_bow_features_normalized():
    v = hashing_ngram_features("alpha beta gamma alpha")
    assert np.linalg.norm(v) == pytest.approx(1.0)


def test_ridge_requires_fit():
    ar = RidgeReconstructor(make_bow_features())
    with pytest.raises(RuntimeError):
        ar.reconstruct("anything")


def test_ridge_recovers_linear_signal():
    # build a learnable mapping: h = phi(z) @ W_true, ridge should reconstruct well
    rng = np.random.default_rng(0)
    zs = [f"sample number {i} with token tok{i % 7} and tok{i % 13}" for i in range(120)]
    phi = make_bow_features(dim=256)
    X = np.vstack([phi(z) for z in zs])
    W_true = rng.normal(size=(256, 32))
    H = X @ W_true                                   # exact linear signal
    ar = RidgeReconstructor(phi, alpha=1e-3).fit(zs, H)
    h_hat = np.vstack([ar.reconstruct(z) for z in zs])
    h_mean = H.mean(axis=0)
    assert M.fve(H, h_hat, h_mean) > 0.95            # high variance explained


def test_end_to_end_with_ridge_ar():
    # fit ridge on (z, h) then run the intervention grid; identity should reconstruct best
    rng = np.random.default_rng(1)
    zs = [f"explanation of activation {i} emphasizes topic{i % 5} and feature{i % 3}"
          for i in range(80)]
    phi = make_bow_features(dim=256)
    X = np.vstack([phi(z) for z in zs])
    H = X @ rng.normal(size=(256, 16))
    ar = RidgeReconstructor(phi, alpha=1e-2).fit(zs, H)

    activations = [{"input_id": f"a{i}", "h_l": H[i], "source_text": "", "z": zs[i],
                    "domain": "syn"} for i in range(80)]
    conditions = [
        {"code": "C0", "transform": "identity"},
        {"code": "C8", "transform": "stopword_strip"},
        {"code": "C4", "transform": "semantic_drift"},
    ]
    obs = run(activations, av=None, ar=ar, conditions=conditions,
              h_mean=H.mean(axis=0), rewriter=FakeRewriter())
    by_cond = {}
    for o in obs:
        by_cond.setdefault(o.condition, []).append(o.metrics["fve"])
    mean = {c: float(np.mean(v)) for c, v in by_cond.items()}
    # identity (z'==z) reconstructs the trained signal best; perturbations degrade it
    assert mean["C0"] == max(mean.values())
    assert mean["C4"] < mean["C0"]


def test_run_uses_precomputed_z_without_av():
    # av=None is allowed when records carry "z" (summary-proxy AV path)
    rng = np.random.default_rng(2)
    zs = [f"doc {i} token{i%4}" for i in range(20)]
    phi = make_bow_features(dim=64)
    H = np.vstack([phi(z) for z in zs]) @ rng.normal(size=(64, 8))
    ar = RidgeReconstructor(phi).fit(zs, H)
    acts = [{"input_id": f"a{i}", "h_l": H[i], "z": zs[i], "source_text": "", "domain": "d"}
            for i in range(20)]
    obs = run(acts, av=None, ar=ar, conditions=[{"code": "C0", "transform": "identity"}])
    assert len(obs) == 20 and all(o.z == zs[i] for i, o in enumerate(obs))
