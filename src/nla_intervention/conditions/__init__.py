"""Intervention conditions (treatment levels) applied to the mediator z.

Each transform: z (str) -> z' (str). They define the (semantic x surface/format)
design grid from docs/research_plan.md §4. A registry maps condition codes -> callables
so the runner can apply them by name from configs/conditions.yaml.

NOTE: scaffold only. Transforms are declared with signatures + docstrings; LLM-backed
rewriting (paraphrase / drift) is deferred to M2.
"""
from __future__ import annotations

import re
from typing import Callable, Dict

# Transform signature: (z, **params) -> z'
Transform = Callable[..., str]

_REGISTRY: Dict[str, Transform] = {}


def register(name: str) -> Callable[[Transform], Transform]:
    """Decorator to register a transform under a name used in conditions.yaml."""
    def deco(fn: Transform) -> Transform:
        _REGISTRY[name] = fn
        return fn
    return deco


def get_transform(name: str) -> Transform:
    if name not in _REGISTRY:
        raise KeyError(f"unknown transform '{name}'. registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def available() -> list[str]:
    return sorted(_REGISTRY)


# --- transform stubs (implement at M2) -------------------------------------

@register("identity")
def identity(z: str, **_) -> str:
    """C0: no-op. Upper-bound baseline."""
    return z


def _require_rewriter(rewriter, transform_name: str):
    if rewriter is None:
        raise ValueError(
            f"transform '{transform_name}' needs a rewriter; pass rewriter= to "
            f"apply_condition/run (LLMRewriter or FakeRewriter, see conditions.rewriter)"
        )
    return rewriter


@register("paraphrase")
def paraphrase(z: str, *, strength: str = "light", rewriter=None, **_) -> str:
    """C1/C2: semantic-preserving paraphrase. strength in {light, strong}.

    Must preserve meaning (verified post-hoc via sim_zz' manipulation check),
    while changing lexicon/syntax. 'strong' maximizes surface change.
    """
    rw = _require_rewriter(rewriter, "paraphrase")
    mode = "paraphrase_strong" if strength == "strong" else "paraphrase"
    return rw.rewrite(z, mode=mode)


@register("format_preserving_paraphrase")
def format_preserving_paraphrase(z: str, *, preserve_length: bool = True,
                                 preserve_structure: bool = True, rewriter=None, **_) -> str:
    """C3: paraphrase that preserves format/length/structure (isolates format channel)."""
    rw = _require_rewriter(rewriter, "format_preserving_paraphrase")
    return rw.rewrite(z, mode="format_preserving")


@register("semantic_drift")
def semantic_drift(z: str, *, preserve_length: bool = True,
                   preserve_format: bool = True, rewriter=None, **_) -> str:
    """C4: negative control. Change MEANING while keeping format/length.

    sim_zz' should fall below tau_drift. If AR still reconstructs well here,
    reconstruction is not using semantics -> evidence for surface/stego channel.
    """
    rw = _require_rewriter(rewriter, "semantic_drift")
    return rw.rewrite(z, mode="semantic_drift")


@register("synonym_substitution")
def synonym_substitution(z: str, *, rewriter=None, **_) -> str:
    """C5: lexical-only substitution, syntax preserved."""
    rw = _require_rewriter(rewriter, "synonym_substitution")
    return rw.rewrite(z, mode="synonym")


@register("back_translation")
def back_translation(z: str, *, pivot_lang: str = "de", rewriter=None, **_) -> str:
    """C6: round-trip translation paraphrase."""
    rw = _require_rewriter(rewriter, "back_translation")
    return rw.rewrite(z, mode="back_translation", pivot_lang=pivot_lang)


# --- model-free transforms (implemented; usable in dry-runs / pilots) ------

_WORD = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_STOPWORDS = frozenset(
    "the a an and or but of to in on at for with as is are was were be been being "
    "this that these those it its by from into over under then than so such".split()
)


@register("token_shuffle")
def token_shuffle(z: str, *, seed: int = 0, **_) -> str:
    """C7: shuffle word order, preserve bag-of-words (syntax vs statistics probe).

    Deterministic given seed (no global RNG); model-free.
    """
    import random as _r

    toks = _WORD.findall(z)
    rng = _r.Random(seed)
    rng.shuffle(toks)
    return " ".join(toks)


@register("stopword_strip")
def stopword_strip(z: str, **_) -> str:
    """C8: remove stopwords + punctuation (surface-redundancy probe). Model-free."""
    toks = [t for t in _WORD.findall(z) if t.isalnum() and t.lower() not in _STOPWORDS]
    return " ".join(toks)


@register("random_text")
def random_text(z: str, *, match_length: bool = True, seed: int = 0, **_) -> str:
    """C9: unrelated text, optionally token-length-matched. Floor / lower bound.

    Draws from a fixed filler vocabulary; deterministic given seed. Model-free.
    """
    import random as _r

    filler = ("lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod "
              "tempor incididunt ut labore et dolore magna aliqua").split()
    rng = _r.Random(seed)
    n = len(_WORD.findall(z)) if match_length else len(filler)
    return " ".join(rng.choice(filler) for _ in range(max(1, n)))
