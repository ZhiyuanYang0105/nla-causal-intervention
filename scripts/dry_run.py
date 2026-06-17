#!/usr/bin/env python
# ⚠️ PHASE 1 (text-proxy) plumbing demo with fake AV/AR. Not the faithful NLA
#    (scripts/train_nla.py + steg_intervention.py). See docs/nla_faithful_findings.md.
"""End-to-end DRY RUN with fake AV/AR/rewriter/embedder — proves the FULL pipeline
(AV -> intervene -> AR -> metrics -> stats) before real models exist.

Runs all 5 core conditions (C0-C4), writes results/dryrun/metrics.csv, then runs the
paired statistical analysis (manipulation check, Friedman, Wilcoxon vs C0). No GPU /
torch / trained models. The FVE numbers are meaningless (fake AR) — the point is that
every stage is wired and the analysis produces a verdict-shaped report.

    python scripts/dry_run.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nla_intervention import stats as S
from nla_intervention.conditions.rewriter import FakeRewriter
from nla_intervention.metrics import FakeEmbedder
from nla_intervention.pipeline.fakes import FakeReconstructor, FakeVerbalizer
from nla_intervention.pipeline.runner import run, to_records

DIM, N = 64, 50

CONDITIONS = [
    {"code": "C0", "transform": "identity"},
    {"code": "C1", "transform": "paraphrase", "params": {"strength": "light"}},
    {"code": "C2", "transform": "paraphrase", "params": {"strength": "strong"}},
    {"code": "C3", "transform": "format_preserving_paraphrase"},
    {"code": "C4", "transform": "semantic_drift"},
]


def main() -> None:
    rng = np.random.default_rng(0)
    activations = [
        {"input_id": f"a{i}", "h_l": rng.normal(size=DIM),
         "source_text": f"synthetic doc {i}", "domain": "synthetic"}
        for i in range(N)
    ]

    obs = run(activations, FakeVerbalizer(), FakeReconstructor(DIM), CONDITIONS,
              rewriter=FakeRewriter(), embedder=FakeEmbedder())
    df = pd.DataFrame(to_records(obs))

    out = Path("results/dryrun")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "metrics.csv", index=False)
    print(f"wrote {out/'metrics.csv'}  ({len(df)} rows = {N} x {len(CONDITIONS)} conditions)\n")

    # --- statistical analysis (part 3) ---
    mc = S.manipulation_check(df, tau_keep=0.85, tau_drift=0.60)
    print("manipulation check — sim_zz' pass rates (FAKE embedder, not semantically aware):")
    for c, r in mc["pass_rate_by_condition"].items():
        print(f"  {c}: {r:.2f}")

    omni = S.omnibus(df, value="fve", method="friedman")
    print(f"\nFriedman omnibus: chi2={omni['statistic']:.2f}  p={omni['p_value']:.3g}  "
          f"(n={omni['n']}, k={omni['k_conditions']})")

    tbl = S.pairwise_vs_baseline(df, value="fve", baseline="C0")
    print("\npairwise vs C0 (paired Wilcoxon, Holm-corrected):")
    cols = ["condition", "mean_delta", "cohen_dz", "p_corrected", "significant"]
    print(tbl[cols].to_string(index=False,
          formatters={"mean_delta": "{:+.3f}".format, "cohen_dz": "{:+.2f}".format,
                      "p_corrected": "{:.3g}".format}))

    print("\n(With a REAL semantic-preserving paraphraser + embedder: H1 => C1-C3 ΔFVE≈0 & "
          "non-sig; H2 => C1-C3 significant and rising with surface_shift.)")


if __name__ == "__main__":
    main()
