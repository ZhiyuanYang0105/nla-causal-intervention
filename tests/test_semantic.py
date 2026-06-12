"""Tests for the semantic-similarity manipulation-check metric."""
import pytest

from nla_intervention.metrics import FakeEmbedder, sim_zz_prime


def test_identical_text_sim_is_one():
    emb = FakeEmbedder()
    assert sim_zz_prime("the brown fox", "the brown fox", emb) == pytest.approx(1.0)


def test_disjoint_text_sim_is_zero():
    emb = FakeEmbedder(dim=4096)  # large dim -> no hash collisions for these tokens
    assert sim_zz_prime("alpha beta gamma", "delta epsilon zeta", emb) == pytest.approx(0.0)


def test_partial_overlap_between_zero_and_one():
    emb = FakeEmbedder(dim=4096)
    s = sim_zz_prime("the brown fox runs", "the brown cat sleeps", emb)
    assert 0.0 < s < 1.0


def test_fake_embedder_is_deterministic():
    emb = FakeEmbedder()
    a = sim_zz_prime("hello world foo", "hello world bar", emb)
    b = sim_zz_prime("hello world foo", "hello world bar", emb)
    assert a == b
