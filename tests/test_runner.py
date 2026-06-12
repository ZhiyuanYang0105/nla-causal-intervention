"""End-to-end pipeline dry-run with fake AV/AR (no trained models)."""
import numpy as np

from nla_intervention.conditions import get_transform
from nla_intervention.pipeline.fakes import FakeReconstructor, FakeVerbalizer
from nla_intervention.pipeline.runner import run, to_records

DIM = 32


def _fake_activations(n=8, seed=0):
    rng = np.random.default_rng(seed)
    return [
        {"input_id": f"a{i}", "h_l": rng.normal(size=DIM), "source_text": f"doc {i}", "domain": "com"}
        for i in range(n)
    ]


def _conditions():
    # only model-free conditions (no LLM needed for the dry-run)
    return [
        {"code": "C0", "transform": "identity"},
        {"code": "C7", "transform": "token_shuffle", "params": {"seed": 1}},
        {"code": "C8", "transform": "stopword_strip"},
        {"code": "C9", "transform": "random_text", "params": {"seed": 1}},
    ]


def test_pipeline_runs_end_to_end():
    acts = _fake_activations()
    obs = run(acts, FakeVerbalizer(), FakeReconstructor(DIM), _conditions())
    assert len(obs) == len(acts) * 4
    rows = to_records(obs)
    # every row has the implemented metrics
    for r in rows:
        for key in ("fve", "activation_cosine", "jaccard_tokens", "js_divergence"):
            assert key in r


def test_identity_is_best_random_is_worst():
    acts = _fake_activations()
    obs = run(acts, FakeVerbalizer(), FakeReconstructor(DIM), _conditions())
    by_cond = {}
    for o in obs:
        by_cond.setdefault(o.condition, []).append(o.metrics["fve"])
    mean_fve = {c: float(np.mean(v)) for c, v in by_cond.items()}
    # identity reconstructs perfectly (z'==z); random text is the floor
    assert mean_fve["C0"] == max(mean_fve.values())
    assert mean_fve["C9"] < mean_fve["C0"]


def test_identity_leaves_text_unperturbed():
    # under identity, z' == z, so surface-shift metrics show no change (the C0 baseline).
    # (FVE=1 is a property of a perfect AR, covered in test_metrics, not of the fake AR.)
    acts = _fake_activations()
    obs = run(acts, FakeVerbalizer(), FakeReconstructor(DIM),
              [{"code": "C0", "transform": "identity"}])
    for o in obs:
        assert o.z_prime == o.z
        assert o.metrics["jaccard_tokens"] == 1.0
        assert o.metrics["js_divergence"] == 0.0
        assert o.metrics["delta_len"] == 0


def test_model_free_transforms_change_text():
    z = "the quick brown fox jumps"
    assert get_transform("identity")(z) == z
    assert set(get_transform("token_shuffle")(z, seed=1).split()) == set(z.split())
    assert "the" not in get_transform("stopword_strip")(z).split()
    assert len(get_transform("random_text")(z, seed=1).split()) == len(z.split())
