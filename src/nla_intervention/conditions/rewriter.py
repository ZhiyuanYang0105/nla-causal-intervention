"""Text rewriters that back the LLM-driven intervention conditions.

A `Rewriter` turns explanation text z into z' under a given MODE (paraphrase,
strong paraphrase, format-preserving, semantic-drift, ...). Two implementations:

- LLMRewriter: adapter over any `complete(prompt) -> str` callable (wire to
  Llama-3.1-8B-Instruct or an API). Fully implemented — just supply the callable.
- FakeRewriter: deterministic, model-free word ops approximating each mode, so the
  full pipeline and tests run without an LLM. NOT semantically faithful — fakes are
  for plumbing, not validity (a bag-of-words embedder won't see synonyms as similar).

The per-mode INSTRUCTIONS are the prompts an LLM actually receives.
"""
from __future__ import annotations

import re
from typing import Callable, Protocol

MODES = (
    "paraphrase",
    "paraphrase_strong",
    "format_preserving",
    "semantic_drift",
    "synonym",
    "back_translation",
)

INSTRUCTIONS: dict[str, str] = {
    "paraphrase": (
        "Paraphrase the following text. Preserve its full meaning but use different "
        "words and sentence structure. Output ONLY the paraphrase.\n\n{z}"
    ),
    "paraphrase_strong": (
        "Restate the following text in completely different words and sentence structure. "
        "IMPORTANT: keep the exact same meaning and all the same facts — do not add, remove, "
        "or change any information. Output ONLY the restated text.\n\n{z}"
    ),
    "format_preserving": (
        "Paraphrase the following text but PRESERVE its formatting, structure, and "
        "approximate length: keep the same number of sentences/lines, any headings, and "
        "the punctuation pattern. Only change the wording. Output ONLY the rewrite.\n\n{z}"
    ),
    "semantic_drift": (
        "Write a NEW sentence on a COMPLETELY DIFFERENT topic than the text below. Change "
        "the subject entirely (different people, places, things, and facts). Keep a similar "
        "length and writing style, but the content must be unrelated. Do NOT paraphrase or "
        "mention anything from the original. Output ONLY the new text.\n\n{z}"
    ),
    "synonym": (
        "Replace content words in the following text with synonyms where possible. Keep "
        "the syntax and structure unchanged. Output ONLY the result.\n\n{z}"
    ),
    "back_translation": (
        "Translate the following text to {pivot_lang} and then back to English, producing "
        "a natural paraphrase. Output ONLY the final English text.\n\n{z}"
    ),
}


class Rewriter(Protocol):
    def rewrite(self, z: str, mode: str, **params) -> str: ...


class LLMRewriter:
    """Adapter over a text-completion callable. `complete(prompt) -> str`.

    Wire to Llama-3.1-8B-Instruct (transformers) or an API by passing the callable.
    """

    def __init__(self, complete: Callable[[str], str]):
        self._complete = complete

    def rewrite(self, z: str, mode: str, **params) -> str:
        if mode not in INSTRUCTIONS:
            raise ValueError(f"unknown rewrite mode {mode!r}; valid: {MODES}")
        prompt = INSTRUCTIONS[mode].format(z=z, **params)
        return self._complete(prompt).strip()


# --- deterministic fake (model-free) ---------------------------------------

_TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_STOP = frozenset("the a an and or but of to in on at for with as is are was were be "
                  "this that it its by from into".split())

# tiny synonym table (lowercase). Surface changes, rough meaning preservation.
_SYN = {
    "activation": "signal", "emphasizes": "highlights", "explanation": "description",
    "text": "passage", "model": "network", "feature": "attribute", "topic": "subject",
    "describes": "characterizes", "content": "substance", "short": "brief",
    "quick": "fast", "brown": "tan", "fox": "canine", "jumps": "leaps",
    "important": "significant", "represents": "denotes", "concept": "notion",
}
# unrelated content words for semantic drift, indexed deterministically
_UNRELATED = "banana orbit concrete velvet thunder ledger cactus piano gravel harbor".split()


class FakeRewriter:
    """Deterministic per-mode word ops. Same input -> same output (no global RNG)."""

    def rewrite(self, z: str, mode: str, **params) -> str:
        toks = _TOKEN.findall(z)
        if mode == "paraphrase":
            return self._map_content(toks, frac=2)          # every 2nd content word
        if mode == "paraphrase_strong":
            return self._reverse_sentences(self._map_content(toks, frac=1))
        if mode == "synonym":
            return self._map_content(toks, frac=1)
        if mode == "back_translation":
            return self._map_content(_TOKEN.findall(self._map_content(toks, frac=1)), frac=1)
        if mode == "format_preserving":
            return self._map_content_inplace(toks)          # 1:1, keep length & punctuation
        if mode == "semantic_drift":
            return self._drift(toks)                         # change content, keep length
        raise ValueError(f"unknown mode {mode!r}")

    @staticmethod
    def _is_content(t: str) -> bool:
        return t.isalnum() and t.lower() not in _STOP

    def _map_content(self, toks, frac: int) -> str:
        out, c = [], 0
        for t in toks:
            if self._is_content(t):
                c += 1
                if c % frac == 0:
                    out.append(_SYN.get(t.lower(), t + "ish"))  # ensure surface changes
                    continue
            out.append(t)
        return self._detok(out)

    def _map_content_inplace(self, toks) -> str:
        # 1:1 replacement only (preserves token count & punctuation positions)
        out = [(_SYN.get(t.lower(), t) if self._is_content(t) else t) for t in toks]
        return self._detok(out)

    def _drift(self, toks) -> str:
        out, c = [], 0
        for t in toks:
            if self._is_content(t):
                out.append(_UNRELATED[c % len(_UNRELATED)])
                c += 1
            else:
                out.append(t)
        return self._detok(out)

    def _reverse_sentences(self, s: str) -> str:
        parts = re.split(r"([.!?])", s)
        rebuilt = []
        for seg in parts:
            if seg in ".!?" or not seg.strip():
                rebuilt.append(seg)
            else:
                words = seg.split()
                rebuilt.append(" " + " ".join(reversed(words)) if seg.startswith(" ") else " ".join(reversed(words)))
        return "".join(rebuilt)

    @staticmethod
    def _detok(toks) -> str:
        out = ""
        for t in toks:
            if re.match(r"[^\w\s]", t) and out:
                out += t
            else:
                out += (" " if out else "") + t
        return out
