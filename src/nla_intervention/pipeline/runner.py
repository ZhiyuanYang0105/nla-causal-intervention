"""Experiment runner: AV -> intervene -> AR -> metrics, over activations.

For each activation h_l (z held fixed across conditions -> paired design) and each
condition: z' = T(z); h_hat = AR(z'); then attach reconstruction (FVE etc.) + text-shift
metrics. Model-agnostic given AV/AR objects, so it runs end-to-end with fakes for dry-runs.

See docs/research_plan.md §7 and docs/paper_findings.md.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from nla_intervention import metrics as M
from nla_intervention.conditions import get_transform
from nla_intervention.pipeline import (
    ActivationReconstructor,
    ActivationVerbalizer,
    Observation,
)


def apply_condition(z: str, condition: dict, rewriter=None) -> str:
    """Apply a single condition's transform to explanation text z. Model-free transforms
    ignore `rewriter`; LLM-backed ones (paraphrase/drift/...) require it."""
    fn = get_transform(condition["transform"])
    return fn(z, rewriter=rewriter, **condition.get("params", {}))


def run(
    activations: Iterable[dict],          # [{input_id, h_l, source_text, domain, z?}, ...]
    av: ActivationVerbalizer | None,      # may be None if records carry a precomputed "z"
    ar: ActivationReconstructor,
    conditions: list[dict],               # parsed from conditions.yaml
    h_mean: np.ndarray | None = None,     # FVE baseline; computed from activations if None
    rewriter=None,                        # for LLM-backed conditions (LLMRewriter/FakeRewriter)
    embedder=None,                        # for sim_zz' manipulation check (optional)
) -> list[Observation]:
    """Run the full intervention grid. One Observation per (activation, condition).

    The explanation z is taken from the record's precomputed "z" (e.g. the local-budget
    training-free summary-proxy AV) when present; otherwise av.verbalize(h_l) is called.
    """
    acts = list(activations)
    if not acts:
        return []
    if h_mean is None:
        h_mean = np.mean([np.asarray(a["h_l"], dtype=np.float64) for a in acts], axis=0)

    out: list[Observation] = []
    for a in acts:
        h_l = np.asarray(a["h_l"], dtype=np.float64)
        z = a.get("z") or av.verbalize(h_l, n=1)[0]    # precomputed z, else call AV
        for cond in conditions:
            z_prime = apply_condition(z, cond, rewriter=rewriter)
            h_hat = np.asarray(ar.reconstruct(z_prime), dtype=np.float64)
            out.append(
                Observation(
                    input_id=a["input_id"],
                    condition=cond.get("code", cond["transform"]),
                    domain=a.get("domain"),
                    source_text=a.get("source_text", ""),
                    h_l=h_l,
                    z=z,
                    z_prime=z_prime,
                    h_hat_l=h_hat,
                    metrics=_metrics_row(h_l, h_hat, z, z_prime, h_mean, embedder),
                )
            )
    return out


def _metrics_row(h_l, h_hat, z, z_prime, h_mean, embedder=None) -> dict:
    """Implemented metrics for one observation. ppl_shift/surface_shift (model-dependent /
    dataset-level) are added later in the metrics/stats stage."""
    row = {
        "fve": float(M.fve_per_sample(h_l, h_hat, h_mean)[0]),
        "activation_cosine": float(M.activation_cosine(h_l, h_hat)[0]),
        "activation_mse": float(M.activation_mse(h_l, h_hat)[0]),
        "delta_len": M.delta_len(z, z_prime),
        "len_ratio": M.len_ratio(z, z_prime),
        "jaccard_tokens": M.jaccard_tokens(z, z_prime),
        "ngram_overlap": M.ngram_overlap(z, z_prime),
        "edit_distance_norm": M.edit_distance_norm(z, z_prime),
        "js_divergence": M.js_divergence(z, z_prime),
    }
    if embedder is not None:
        row["sim_zz_prime"] = M.sim_zz_prime(z, z_prime, embedder)
    return row


def to_records(observations: list[Observation]) -> list[dict]:
    """Flatten Observations into table rows (scalar metrics; activations dropped).
    Ready for pandas.DataFrame(...) -> results/<run_id>/metrics.parquet."""
    rows = []
    for o in observations:
        rows.append({
            "input_id": o.input_id, "condition": o.condition, "domain": o.domain,
            "source_text": o.source_text, "z": o.z, "z_prime": o.z_prime, **o.metrics,
        })
    return rows
