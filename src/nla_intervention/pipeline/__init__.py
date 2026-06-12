"""AV/AR model interfaces and the experiment runner — ACTIVATION space.

NLA autoencodes residual-stream activations (see docs/paper_findings.md):

    h_l  --AV(Activation Verbalizer)-->  explanation text z
    z'   --AR(Activation Reconstructor)--> reconstructed activation h_hat_l

AV/AR are fine-tuned copies of the target model M (not public). These are
protocols + a runner skeleton so conditions/metrics/stats build against stable types.
Concrete bindings deferred (M1).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import numpy as np


class ActivationVerbalizer(Protocol):
    """AV: maps a residual-stream activation h_l to a natural-language explanation z.

    Paper samples a GROUP of explanations per activation: z ~ AV(.|h_l).
    """
    def verbalize(self, h_l: "np.ndarray", n: int = 1) -> list[str]: ...


class ActivationReconstructor(Protocol):
    """AR: reconstructs an activation h_hat_l from (possibly intervened) text z'."""
    def reconstruct(self, z: str) -> "np.ndarray": ...


@dataclass
class Observation:
    """One (activation, condition) paired observation — a row in the metrics table.

    The autoencoded unit is a single activation vector (one token position), NOT text.
    `source_text` is kept only for reference/stratification.
    """
    input_id: str            # id of the source activation
    condition: str
    domain: str | None       # corpus domain/source, for stratification
    source_text: str         # snippet that produced h_l (reference only)
    h_l: "np.ndarray"        # target activation (ground truth)
    z: str                   # AV(h_l) explanation
    z_prime: str             # T(z) intervened explanation
    h_hat_l: "np.ndarray"    # AR(z') reconstructed activation
    metrics: dict = field(default_factory=dict)


__all__ = ["ActivationVerbalizer", "ActivationReconstructor", "Observation"]
