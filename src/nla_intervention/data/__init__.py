"""Activation harvesting — the experiment's "dataset" is a set of activations h_l,
collected by running the target model M (Llama-3.1-8B) over FineWeb text.

Recipe (docs/paper_findings.md): pretraining-like snippets, randomly truncated, run M,
read the residual-stream activation at layer l of the FINAL token. The eval split is
made disjoint from training docs (hash-based) to avoid leakage into AV/AR.

Each record: {input_id, h_l: np.ndarray, source_text: str, domain: str}.

Implementation in harvest.py (lazy torch/transformers/datasets imports — needs GPU + HF).
Pure helpers (_split_of, _domain_of, dataset_mean_activation) are unit-testable.
"""
from __future__ import annotations

from nla_intervention.data.harvest import (
    HarvestConfig,
    dataset_mean_activation,
    harvest_activations,
)

__all__ = ["HarvestConfig", "harvest_activations", "dataset_mean_activation"]
