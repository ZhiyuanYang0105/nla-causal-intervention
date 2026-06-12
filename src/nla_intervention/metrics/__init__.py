"""Evaluation metrics — four families (see docs/metrics.md).

    reconstruction : closeness of h_hat_l to h_l in ACTIVATION space  (primary outcome)
    semantic       : sim(z, z') on explanation TEXT                   (manipulation check)
    length         : delta_len, len_ratio on z text
    token_shift    : surface/statistical distance z->z' on text       (mechanism regressor)

Reconstruction (FVE/cosine/MSE, activation space) and text/length/token-shift metrics
are IMPLEMENTED and model-free. Model-dependent metrics (semantic embedding similarity,
perplexity shift, composite surface_shift) remain deferred.
"""
from __future__ import annotations

# implemented — activation space (numpy)
from nla_intervention.metrics.reconstruction import (
    activation_cosine,
    activation_mse,
    fve,
    fve_per_sample,
)

# implemented — text/surface (pure python)
from nla_intervention.metrics.text_shift import (
    delta_len,
    edit_distance_norm,
    jaccard_tokens,
    js_divergence,
    len_ratio,
    ngram_overlap,
    tokenize,
)

# implemented — semantic similarity (needs an embedder, fake or real)
from nla_intervention.metrics.semantic import (
    Embedder,
    FakeEmbedder,
    SentenceTransformerEmbedder,
    sim_zz_prime,
)

__all__ = [
    "fve", "fve_per_sample", "activation_cosine", "activation_mse",
    "delta_len", "len_ratio", "jaccard_tokens", "ngram_overlap",
    "edit_distance_norm", "js_divergence", "tokenize",
    "sim_zz_prime", "Embedder", "FakeEmbedder", "SentenceTransformerEmbedder",
    "ppl_shift", "surface_shift",
]


# ---- model-dependent metrics (deferred) -----------------------------------

def ppl_shift(z: str, z_prime: str, ref_lm=None) -> float:
    """ppl(z') - ppl(z) under a reference LM. Deferred (needs an LM)."""
    raise NotImplementedError("M3")


def surface_shift(z: str, z_prime: str) -> float:
    """Composite surface-distortion score: standardized combo of the implemented
    token-shift metrics (jaccard, ngram_overlap, edit_distance_norm, js_divergence).

    Standardization/PCA weights are fit over the run's distribution, so this is a
    dataset-level transform applied in the stats stage, not a per-pair pure function.
    Deferred (M4: fit on the metrics table)."""
    raise NotImplementedError("M4: standardize/PCA over token-shift columns")
