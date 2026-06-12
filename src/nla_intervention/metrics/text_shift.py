"""Surface / length / statistical shift between explanation texts z and z'.

All pure-Python (model-free), so fully testable. These quantify the surface
perturbation of the intervention and feed the mechanism regression
(docs/statistical_analysis.md §5). See docs/metrics.md §3-4.

Tokenization here is a simple whitespace/word split; swap in the target model's
tokenizer at integration time if token-exact counts matter.
"""
from __future__ import annotations

import math
import re
from collections import Counter

_WORD = re.compile(r"\w+", re.UNICODE)


def tokenize(s: str) -> list[str]:
    return _WORD.findall(s.lower())


# ---- length ----------------------------------------------------------------

def delta_len(z: str, z_prime: str) -> int:
    """len(z') - len(z) in tokens."""
    return len(tokenize(z_prime)) - len(tokenize(z))


def len_ratio(z: str, z_prime: str) -> float:
    """len(z') / len(z); 1.0 if z empty and z' empty, inf-safe."""
    nz = len(tokenize(z))
    return len(tokenize(z_prime)) / nz if nz else float("nan")


# ---- token / statistical shift --------------------------------------------

def jaccard_tokens(z: str, z_prime: str) -> float:
    """|Tz ∩ Tz'| / |Tz ∪ Tz'| over token SETS. 1.0 if both empty."""
    a, b = set(tokenize(z)), set(tokenize(z_prime))
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def ngram_overlap(z: str, z_prime: str, n: int = 2) -> float:
    """Containment of z' n-grams in z (multiset overlap / |z' n-grams|)."""
    def grams(toks):
        return Counter(tuple(toks[i : i + n]) for i in range(len(toks) - n + 1))
    ga, gb = grams(tokenize(z)), grams(tokenize(z_prime))
    tot = sum(gb.values())
    if tot == 0:
        return 1.0 if sum(ga.values()) == 0 else 0.0
    inter = sum(min(gb[g], ga.get(g, 0)) for g in gb)
    return inter / tot


def edit_distance_norm(z: str, z_prime: str) -> float:
    """Levenshtein distance over tokens, normalized by max length. 0=identical."""
    a, b = tokenize(z), tokenize(z_prime)
    if not a and not b:
        return 0.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / max(len(a), len(b))


def js_divergence(z: str, z_prime: str, base: float = 2.0) -> float:
    """Jensen-Shannon divergence between token-frequency distributions. 0=identical.

    Bounded in [0, 1] with base=2. Model-free statistical-shift signal.
    """
    ca, cb = Counter(tokenize(z)), Counter(tokenize(z_prime))
    vocab = set(ca) | set(cb)
    if not vocab:
        return 0.0
    na, nb = sum(ca.values()), sum(cb.values())
    p = {w: ca.get(w, 0) / na for w in vocab} if na else {w: 0.0 for w in vocab}
    q = {w: cb.get(w, 0) / nb for w in vocab} if nb else {w: 0.0 for w in vocab}

    def _kl(x, y):
        s = 0.0
        for w in vocab:
            if x[w] > 0 and y[w] > 0:
                s += x[w] * math.log(x[w] / y[w], base)
        return s

    m = {w: 0.5 * (p[w] + q[w]) for w in vocab}
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
