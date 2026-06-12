"""Tests for rewriter-backed conditions (FakeRewriter) and the rewriter adapter."""
import pytest

from nla_intervention.conditions import get_transform
from nla_intervention.conditions.rewriter import INSTRUCTIONS, FakeRewriter, LLMRewriter
from nla_intervention.pipeline.runner import apply_condition

Z = "Explanation: this activation emphasizes the brown fox topic."


def test_llm_conditions_require_rewriter():
    # without a rewriter, LLM-backed transforms raise a helpful error
    with pytest.raises(ValueError, match="needs a rewriter"):
        apply_condition(Z, {"transform": "paraphrase"})


def test_fake_paraphrase_changes_surface():
    rw = FakeRewriter()
    out = apply_condition(Z, {"transform": "paraphrase"}, rewriter=rw)
    assert out != Z and len(out) > 0


def test_strength_strong_differs_from_light():
    rw = FakeRewriter()
    light = apply_condition(Z, {"transform": "paraphrase", "params": {"strength": "light"}}, rewriter=rw)
    strong = apply_condition(Z, {"transform": "paraphrase", "params": {"strength": "strong"}}, rewriter=rw)
    assert light != strong


def test_format_preserving_keeps_token_count():
    rw = FakeRewriter()
    out = apply_condition(Z, {"transform": "format_preserving_paraphrase"}, rewriter=rw)
    # 1:1 replacement keeps the token count identical
    import re
    n_in = len(re.findall(r"\w+|[^\w\s]", Z))
    n_out = len(re.findall(r"\w+|[^\w\s]", out))
    assert n_in == n_out


def test_semantic_drift_changes_content_keeps_length():
    rw = FakeRewriter()
    out = apply_condition(Z, {"transform": "semantic_drift"}, rewriter=rw)
    assert out != Z
    # content words replaced by unrelated ones -> "fox" gone
    assert "fox" not in out.lower()


def test_fake_rewriter_is_deterministic():
    a = FakeRewriter().rewrite(Z, mode="paraphrase")
    b = FakeRewriter().rewrite(Z, mode="paraphrase")
    assert a == b


def test_llm_rewriter_uses_instruction_prompt():
    seen = {}

    def fake_complete(prompt: str) -> str:
        seen["prompt"] = prompt
        return "REWRITTEN"

    rw = LLMRewriter(fake_complete)
    out = rw.rewrite(Z, mode="paraphrase")
    assert out == "REWRITTEN"
    assert Z in seen["prompt"]                  # z injected into the prompt
    assert INSTRUCTIONS["paraphrase"].split("{")[0] in seen["prompt"]


def test_llm_rewriter_rejects_unknown_mode():
    with pytest.raises(ValueError):
        LLMRewriter(lambda p: p).rewrite(Z, mode="bogus")
