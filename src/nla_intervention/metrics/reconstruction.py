"""Reconstruction metrics in ACTIVATION space (the NLA's outcome).

h_l, h_hat_l are activation vectors (np.ndarray, shape (d,) or batched (n, d)).
See docs/metrics.md §1 and docs/paper_findings.md.
"""
from __future__ import annotations

import numpy as np


def _as2d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    return a[None, :] if a.ndim == 1 else a


def activation_mse(h: np.ndarray, h_hat: np.ndarray) -> np.ndarray:
    """Per-sample squared L2 error ||h - h_hat||^2. Returns shape (n,)."""
    H, Hh = _as2d(h), _as2d(h_hat)
    return np.sum((H - Hh) ** 2, axis=1)


def activation_cosine(h: np.ndarray, h_hat: np.ndarray) -> np.ndarray:
    """Per-sample cosine similarity cos(h, h_hat). Returns shape (n,)."""
    H, Hh = _as2d(h), _as2d(h_hat)
    num = np.sum(H * Hh, axis=1)
    den = np.linalg.norm(H, axis=1) * np.linalg.norm(Hh, axis=1)
    out = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    return out


def fve(h: np.ndarray, h_hat: np.ndarray, h_mean: np.ndarray) -> float:
    """Fraction of Variance Explained over a SET of activations (paper's metric):

        FVE = 1 - sum||h - h_hat||^2 / sum||h - h_mean||^2

    h, h_hat: (n, d). h_mean: (d,) dataset mean activation E[h_l].
    FVE=0 -> predicting the mean; FVE=1 -> perfect. Aggregate (not per-sample).
    """
    H, Hh = _as2d(h), _as2d(h_hat)
    h_mean = np.asarray(h_mean, dtype=np.float64).reshape(1, -1)
    resid = float(np.sum((H - Hh) ** 2))
    total = float(np.sum((H - h_mean) ** 2))
    if total == 0.0:
        return float("nan")
    return 1.0 - resid / total


def fve_per_sample(h: np.ndarray, h_hat: np.ndarray, h_mean: np.ndarray) -> np.ndarray:
    """Per-sample FVE = 1 - ||h-h_hat||^2 / ||h-h_mean||^2, shape (n,).

    Use for paired tests (one FVE value per activation per condition). The
    set-level `fve` is the recommended headline number; per-sample enables
    Wilcoxon/Friedman across conditions.
    """
    H, Hh = _as2d(h), _as2d(h_hat)
    h_mean = np.asarray(h_mean, dtype=np.float64).reshape(1, -1)
    resid = np.sum((H - Hh) ** 2, axis=1)
    total = np.sum((H - h_mean) ** 2, axis=1)
    return np.divide(resid, total, out=np.full_like(resid, np.nan), where=total > 0) * -1.0 + 1.0
