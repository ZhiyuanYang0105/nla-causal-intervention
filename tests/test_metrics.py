"""Unit tests for the implemented metrics (activation-space + text-shift)."""
import numpy as np
import pytest

from nla_intervention import metrics as M


# ---- reconstruction (activation space) ------------------------------------

def test_fve_perfect_is_one():
    h = np.array([[1.0, 2.0], [3.0, 4.0]])
    mean = h.mean(axis=0)
    assert M.fve(h, h, mean) == pytest.approx(1.0)


def test_fve_mean_prediction_is_zero():
    h = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    mean = h.mean(axis=0)
    h_hat = np.tile(mean, (3, 1))  # predict the mean for every sample
    assert M.fve(h, h_hat, mean) == pytest.approx(0.0)


def test_fve_can_go_negative():
    h = np.array([[1.0, 0.0], [3.0, 0.0]])
    mean = h.mean(axis=0)
    h_hat = np.array([[100.0, 0.0], [-100.0, 0.0]])  # worse than mean
    assert M.fve(h, h_hat, mean) < 0.0


def test_activation_cosine_aligned_and_opposite():
    h = np.array([[1.0, 0.0]])
    assert M.activation_cosine(h, h)[0] == pytest.approx(1.0)
    assert M.activation_cosine(h, -h)[0] == pytest.approx(-1.0)


def test_activation_mse_matches_manual():
    h = np.array([[0.0, 0.0]])
    h_hat = np.array([[3.0, 4.0]])
    assert M.activation_mse(h, h_hat)[0] == pytest.approx(25.0)


def test_fve_per_sample_shape_and_values():
    h = np.array([[1.0, 2.0], [3.0, 4.0]])
    mean = h.mean(axis=0)
    ps = M.fve_per_sample(h, h, mean)
    assert ps.shape == (2,)
    assert np.allclose(ps, 1.0)


# ---- text / surface shift --------------------------------------------------

def test_identity_text_has_no_shift():
    z = "the quick brown fox"
    assert M.jaccard_tokens(z, z) == 1.0
    assert M.edit_distance_norm(z, z) == 0.0
    assert M.js_divergence(z, z) == pytest.approx(0.0)
    assert M.len_ratio(z, z) == 1.0
    assert M.delta_len(z, z) == 0


def test_jaccard_partial_overlap():
    # {a,b,c} vs {b,c,d} -> intersection 2, union 4
    assert M.jaccard_tokens("a b c", "b c d") == pytest.approx(0.5)


def test_delta_len_and_ratio():
    assert M.delta_len("a b", "a b c d") == 2
    assert M.len_ratio("a b", "a b c d") == pytest.approx(2.0)


def test_edit_distance_norm_one_substitution():
    # one token differs out of 3 -> 1/3
    assert M.edit_distance_norm("a b c", "a b x") == pytest.approx(1 / 3)


def test_js_divergence_disjoint_is_max():
    # disjoint vocabularies -> JS divergence = 1 (base 2)
    assert M.js_divergence("a a", "b b") == pytest.approx(1.0)


def test_js_divergence_bounded():
    assert 0.0 <= M.js_divergence("a b c a", "a b d") <= 1.0
