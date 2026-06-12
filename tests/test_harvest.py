"""Tests for the model-free parts of the activation harvester."""
import numpy as np

from nla_intervention.data import HarvestConfig, dataset_mean_activation
from nla_intervention.data.harvest import _domain_of, _split_of


def test_split_is_deterministic():
    cfg = HarvestConfig()
    assert _split_of("doc-123", cfg) == _split_of("doc-123", cfg)


def test_split_partitions_into_eval_and_train():
    cfg = HarvestConfig(eval_holdout_docs=500_000)  # ~50% to eval
    labels = {_split_of(f"doc-{i}", cfg) for i in range(2000)}
    assert labels == {"eval", "train"}  # both buckets populated, disjoint by construction


def test_eval_fraction_tracks_holdout():
    cfg = HarvestConfig(eval_holdout_docs=100_000)  # ~10%
    n = 20000
    frac = sum(_split_of(f"d{i}", cfg) == "eval" for i in range(n)) / n
    assert 0.08 < frac < 0.12


def test_domain_from_url():
    assert _domain_of({"url": "https://example.com/page"}) == "com"
    assert _domain_of({"url": "http://mit.edu/x"}) == "edu"
    assert _domain_of({}) == "unknown"


def test_dataset_mean_activation():
    recs = [{"h_l": np.array([0.0, 0.0])}, {"h_l": np.array([2.0, 4.0])}]
    mean = dataset_mean_activation(iter(recs))
    assert np.allclose(mean, [1.0, 2.0])
