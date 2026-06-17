"""Smoke tests for the scaffold — verify wiring works before models exist."""
from pathlib import Path

from nla_intervention import conditions, metrics
from nla_intervention.conditions import apply_condition
from nla_intervention.utils import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_conditions_registry_has_core_transforms():
    for t in ["identity", "paraphrase", "format_preserving_paraphrase", "semantic_drift"]:
        assert t in conditions.available()


def test_identity_is_noop():
    assert conditions.get_transform("identity")("hello") == "hello"


def test_apply_condition_identity():
    # conditions operate on the explanation TEXT z (model-free, testable)
    cond = {"transform": "identity", "params": {}}
    assert apply_condition("z text", cond) == "z text"


def test_reconstruction_metric_is_activation_space():
    # FVE is the primary outcome (activation space), not a text metric
    assert hasattr(metrics, "fve")
    assert hasattr(metrics, "activation_cosine")


def test_config_extends_merge():
    cfg = load_config(ROOT / "experiments/exp04_nla/config.yaml")
    assert cfg["experiment"]["id"] == "exp04_nla"          # overridden in experiment
    assert cfg["target_model"]["hidden_size"] == 896       # experiment-specific
    assert cfg["stats"]["omnibus"] == "friedman"          # inherited from default
    assert cfg["metrics"]["reconstruction"]["primary"] == "fve"  # inherited (activation-space)
