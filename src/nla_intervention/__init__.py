"""NLA causal intervention: AV -> intermediate text (z) -> AR.

Mediator-intervention experiment testing whether AR reconstruction depends on the
*semantic* content of z or on *surface/statistical/steganographic* features of z.

Package layout:
    conditions/  intervention transforms applied to z  (treatment levels)
    pipeline/    AV encoder, AR decoder, experiment runner
    metrics/     reconstruction, semantic similarity, length, token/statistical shift
    stats/       paired significance tests + mechanism regression
    data/        dataset loading
    utils/       io, seeding, logging

Status: scaffold. Most functions raise NotImplementedError (see docs/research_plan.md).
"""

__version__ = "0.0.1"
