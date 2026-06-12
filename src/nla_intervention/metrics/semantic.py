"""Semantic similarity on explanation TEXT — the manipulation check (docs/metrics.md §2).

sim_zz' = cos(emb(z), emb(z')). Needs a sentence embedder:

- SentenceTransformerEmbedder: wraps sentence-transformers (e.g. all-mpnet-base-v2).
  Real semantic embeddings — use for actual runs ([nlp] extra).
- FakeEmbedder: hashed bag-of-words vector. Deterministic, model-free, for plumbing/
  tests ONLY. NOT semantically aware (won't see synonyms as similar), so it cannot
  validate true meaning preservation — use a real embedder for the manipulation check.
"""
from __future__ import annotations

import zlib
from typing import Protocol

import numpy as np

from nla_intervention.metrics.text_shift import tokenize


def _stable_hash(s: str) -> int:
    return zlib.crc32(s.encode("utf-8"))


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...  # (n, d)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def sim_zz_prime(z: str, z_prime: str, embedder: Embedder) -> float:
    """Cosine similarity of explanation texts z and z' under `embedder`."""
    embs = embedder.embed([z, z_prime])
    return _cosine(np.asarray(embs[0], dtype=np.float64), np.asarray(embs[1], dtype=np.float64))


class SentenceTransformerEmbedder:
    """Real embeddings via sentence-transformers (lazy import; [nlp] extra)."""

    def __init__(self, model: str = "all-mpnet-base-v2"):
        from sentence_transformers import SentenceTransformer

        self._m = SentenceTransformer(model)

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._m.encode(texts, normalize_embeddings=False))


class FakeEmbedder:
    """Hashed bag-of-words embedding. Deterministic, model-free. Tests/plumbing only."""

    def __init__(self, dim: int = 128):
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim))
        for i, t in enumerate(texts):
            for tok in tokenize(t):
                out[i, _stable_hash(tok) % self.dim] += 1.0
        return out
