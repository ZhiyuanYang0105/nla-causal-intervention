"""Lightweight AR (Activation Reconstructor) for the local-budget setup.

Replaces the paper's "truncated LLM + affine map" with a FROZEN feature extractor +
a closed-form ridge regression head:  h_hat = phi(z) @ W.  Only W is "trained", in
closed form (no GPU, no backprop). See docs/local_budget_plan.md §3.

CRITICAL: the feature extractor phi must be SURFACE-SENSITIVE (read the actual tokens),
otherwise a paraphrase-invariant phi makes the experiment trivially conclude "semantic
channel". `hashing_ngram_features` (bag-of-ngrams) and a frozen-M readout are valid;
a semantic sentence-embedder is NOT (use it only for sim_zz').
"""
from __future__ import annotations

import zlib
from typing import Callable

import numpy as np

from nla_intervention.metrics.text_shift import tokenize


# ---- surface feature extractor (AR-bow): hashing n-gram TF ------------------

def hashing_ngram_features(z: str, dim: int = 4096, ngram=(1, 2)) -> np.ndarray:
    """Surface-sensitive bag-of-ngrams term-frequency vector (hashed to `dim`).

    Deterministic (stable hash). Different tokens -> different features, so paraphrase
    that changes surface form moves the vector — exactly what lets AR detect a surface
    channel."""
    toks = tokenize(z)
    vec = np.zeros(dim, dtype=np.float64)
    lo, hi = (ngram[0], ngram[-1]) if isinstance(ngram, (list, tuple)) else (1, ngram)
    for n in range(lo, hi + 1):
        for i in range(len(toks) - n + 1):
            g = " ".join(toks[i : i + n])
            vec[zlib.crc32(g.encode("utf-8")) % dim] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def make_bow_features(dim: int = 4096, ngram=(1, 2)) -> Callable[[str], np.ndarray]:
    return lambda z: hashing_ngram_features(z, dim=dim, ngram=ngram)


# ---- closed-form ridge AR --------------------------------------------------

class RidgeReconstructor:
    """AR: z -> h_hat via ridge regression on (optionally PCA-reduced) features.

    Closed form, no backprop. Satisfies the ActivationReconstructor protocol after .fit().

    feat_pca: if set, reduce features to this many PCA components (fit on train). Keeps
        surface sensitivity (the features still read tokens) while making the regression
        fittable when n_train < feature_dim — essential at small N.
    alpha: float, or a list of candidates → selected by a held-out validation split
        (CV) inside fit, maximizing reconstruction FVE.
    """

    def __init__(self, feature_fn: Callable[[str], np.ndarray],
                 alpha=10.0, feat_pca: int | None = None, val_frac: float = 0.2,
                 seed: int = 0):
        self.feature_fn = feature_fn
        self.alpha = alpha
        self.feat_pca = feat_pca
        self.val_frac = val_frac
        self.seed = seed
        self.W = self._x_mean = self._h_mean = None
        self._pca_mu = self._pca_P = None
        self.alpha_ = None

    def _features(self, zs: list[str]) -> np.ndarray:
        return np.vstack([self.feature_fn(z) for z in zs])

    def _project(self, X: np.ndarray) -> np.ndarray:
        if self._pca_P is None:
            return X
        return (X - self._pca_mu) @ self._pca_P.T

    @staticmethod
    def _solve(Xc: np.ndarray, Hc: np.ndarray, alpha: float) -> np.ndarray:
        return np.linalg.solve(Xc.T @ Xc + alpha * np.eye(Xc.shape[1]), Xc.T @ Hc)

    def fit(self, zs: list[str], H: np.ndarray) -> "RidgeReconstructor":
        X = self._features(zs)
        H = np.asarray(H, dtype=np.float64)
        if self.feat_pca:                            # fit PCA basis on training features
            self._pca_mu = X.mean(axis=0)
            _, _, Vt = np.linalg.svd(X - self._pca_mu, full_matrices=False)
            self._pca_P = Vt[: self.feat_pca]
        Xp = self._project(X)

        alphas = self.alpha if isinstance(self.alpha, (list, tuple)) else [self.alpha]
        if len(alphas) > 1:                          # CV-select alpha on a val split
            rng = np.random.default_rng(self.seed)
            idx = rng.permutation(len(zs)); nval = max(1, int(len(zs) * self.val_frac))
            vi, ti = idx[:nval], idx[nval:]
            xm, hm = Xp[ti].mean(0), H[ti].mean(0)
            best, best_a = -np.inf, alphas[0]
            for a in alphas:
                W = self._solve(Xp[ti] - xm, H[ti] - hm, a)
                pred = (Xp[vi] - xm) @ W + hm
                fve = 1 - ((H[vi] - pred) ** 2).sum() / ((H[vi] - H[vi].mean(0)) ** 2).sum()
                if fve > best:
                    best, best_a = fve, a
            self.alpha_ = best_a
        else:
            self.alpha_ = alphas[0]

        self._x_mean, self._h_mean = Xp.mean(0), H.mean(0)
        self.W = self._solve(Xp - self._x_mean, H - self._h_mean, self.alpha_)
        return self

    def reconstruct(self, z: str) -> np.ndarray:
        if self.W is None:
            raise RuntimeError("RidgeReconstructor.fit() must be called before reconstruct()")
        x = self._project(self.feature_fn(z).reshape(1, -1))[0] - self._x_mean
        return x @ self.W + self._h_mean


# ---- frozen-M readout feature extractor (AR-readout, recommended) ----------

class MReadoutFeatures:
    """phi(z) = last-token hidden state of FROZEN M at layer `layer`, read by running M
    on z. Surface-sensitive (the LLM reads the actual tokens). Inference only — M is
    never trained; only the downstream ridge W is. Lazy torch import (needs the model).

    Usage:  ar = RidgeReconstructor(MReadoutFeatures(model, tok, layer=8), alpha=10)
    """

    def __init__(self, model, tokenizer, layer: int = 8, device: str = "mps",
                 max_tokens: int = 128, pooling: str = "mean"):
        self.model, self.tok, self.layer = model, tokenizer, layer
        self.device, self.max_tokens, self.pooling = device, max_tokens, pooling

    def __call__(self, z: str) -> np.ndarray:
        import torch

        enc = self.tok(z, return_tensors="pt", truncation=True,
                       max_length=self.max_tokens).to(self.device)
        with torch.no_grad():
            out = self.model(**enc, output_hidden_states=True)
        h = out.hidden_states[self.layer][0]         # (T, d)
        v = h.mean(0) if self.pooling == "mean" else h[-1]   # align with target pooling
        return v.float().cpu().numpy()
