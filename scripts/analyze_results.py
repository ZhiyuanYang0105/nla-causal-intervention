#!/usr/bin/env python
"""Statistical analysis over a finished run's metrics table.

Loads results/<run_id>/metrics.{parquet,csv}, runs the paired analysis
(manipulation check -> Friedman -> Wilcoxon vs baseline -> mechanism regression ->
power), and writes stats_report.json + a human-readable summary.

    python scripts/analyze_results.py --run results/dryrun
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from nla_intervention import stats as S


def _load(run: Path) -> pd.DataFrame:
    for name in ("metrics.parquet", "metrics.csv"):
        p = run / name
        if p.exists():
            return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    raise FileNotFoundError(f"no metrics.parquet/csv in {run}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="results/<run_id> directory")
    ap.add_argument("--baseline", default="C0")
    ap.add_argument("--value", default="fve")
    args = ap.parse_args()

    run = Path(args.run)
    df = _load(run)
    report: dict = {"run": str(run), "n_rows": len(df),
                    "conditions": sorted(df["condition"].unique())}

    if "sim_zz_prime" in df.columns:
        mc = S.manipulation_check(df)
        report["manipulation_check"] = {
            "pass_rate_by_condition": mc["pass_rate_by_condition"],
            "n_failed": mc["n_failed"],
        }

    report["omnibus"] = S.omnibus(df, value=args.value, method="friedman")

    tbl = S.pairwise_vs_baseline(df, value=args.value, baseline=args.baseline)
    report["pairwise_vs_baseline"] = tbl.to_dict(orient="records")

    # mechanism regression (needs the semantic-preserving subset + token-shift cols)
    try:
        res = S.mechanism_regression(df, baseline=args.baseline, value=args.value)
        report["mechanism_regression"] = {
            "params": res.params.to_dict(),
            "pvalues": res.pvalues.to_dict(),
            "surface_shift_p": float(res.pvalues.get("surface_shift", float("nan"))),
        }
    except Exception as e:  # subset/cols may be absent in a partial run
        report["mechanism_regression"] = {"error": str(e)}

    # power at the largest observed effect
    if len(tbl):
        top = tbl.loc[tbl["cohen_dz"].abs().idxmax()]
        pr = S.power_analysis(observed_dz=float(top["cohen_dz"]), n=int(top["n"]))
        report["power_at_max_effect"] = {
            "condition": top["condition"], "dz": pr.effect_dz,
            "power": pr.power, "required_n_for_0.8": pr.required_n,
        }

    (run / "stats_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"wrote {run/'stats_report.json'}\n")
    print(f"Friedman: p={report['omnibus']['p_value']:.3g}")
    print("pairwise vs", args.baseline, "(Holm):")
    print(tbl[["condition", "mean_delta", "cohen_dz", "p_corrected", "significant"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
