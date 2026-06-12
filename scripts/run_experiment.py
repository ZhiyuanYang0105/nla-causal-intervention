#!/usr/bin/env python
"""Run an intervention experiment: load config -> data -> AV/AR -> conditions -> metrics.

Usage:
    python scripts/run_experiment.py --config experiments/exp01_pilot/config.yaml

Skeleton (M3): wires utils.load_config + data.load_inputs + pipeline.runner.run
+ metrics, then writes results/<run_id>/metrics.parquet and run_metadata.json.
"""
from __future__ import annotations

import argparse

from nla_intervention.utils import load_config, set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("data", {}).get("seed", 0))
    print(f"[run_experiment] loaded config: {cfg.get('experiment', {}).get('id', '<default>')}")
    raise NotImplementedError("M3: run pipeline + compute metrics + persist results")


if __name__ == "__main__":
    main()
