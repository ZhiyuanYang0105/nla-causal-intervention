"""Fake AV/AR for dry-runs and tests — NO trained models required.

These let the full pipeline (harvest -> AV -> intervene -> AR -> metrics) run
end-to-end before real Llama-based AV/AR exist (M2). They are deliberately crude:

- FakeVerbalizer: emits a deterministic explanation derived from the activation.
- FakeReconstructor: maps text back to an activation. Crucially it reconstructs from
  the TEXT's content, so semantic-preserving vs surface-destroying transforms produce
  visibly different FVE — a sanity check that the experiment can detect the effect.

Replace with Llama-3.1-8B AV/AR at M2; the runner code does not change.
"""
from __future__ import annotations

import hashlib

import numpy as np

from nla_intervention.metrics.text_shift import tokenize


class FakeVerbalizer:
    """h_l -> a short pseudo-explanation whose tokens encode the activation's sign bits.

    Deterministic and invertible-ish, so FakeReconstructor can partially recover h_l.
    """

    def __init__(self, vocab_dim: int = 16):
        self.vocab_dim = vocab_dim
        self._words = [f"feat{i}" for i in range(vocab_dim)]

    def verbalize(self, h_l: np.ndarray, n: int = 1) -> list[str]:
        h = np.asarray(h_l, dtype=np.float64)
        # pick the top-k coordinates by magnitude; name them with sign words
        k = min(self.vocab_dim, h.shape[0])
        top = np.argsort(-np.abs(h))[:k]
        words = []
        for idx in top:
            sign = "pos" if h[idx] >= 0 else "neg"
            words.append(f"{self._words[idx % self.vocab_dim]}_{sign}")
        text = "Explanation: this activation emphasizes " + ", ".join(words) + "."
        return [text] * n


class FakeReconstructor:
    """text -> activation via a fixed hashing embedding of the text's tokens.

    Because the mapping depends on which tokens are present, paraphrases that change
    surface tokens move h_hat (lower FVE), while identity keeps it (high FVE).
    """

    def __init__(self, dim: int, seed: int = 0):
        self.dim = dim
        self.seed = seed

    def _token_vec(self, tok: str) -> np.ndarray:
        h = hashlib.sha1(f"{self.seed}:{tok}".encode()).digest()
        # deterministic pseudo-random vector in [-1, 1]^dim
        rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
        return rng.uniform(-1.0, 1.0, size=self.dim)

    def reconstruct(self, z: str) -> np.ndarray:
        toks = tokenize(z)
        if not toks:
            return np.zeros(self.dim)
        return np.mean([self._token_vec(t) for t in toks], axis=0)
