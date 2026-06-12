#!/usr/bin/env python
"""Channel-localization: run THREE ARs of differing surface-sensitivity on the SAME
cached (activation, z, z') data and compare the ΔFVE spectrum per condition.

    AR-bow       pure surface (hashing n-grams)         -> upper bound on surface channel
    AR-readout   frozen-M reads tokens (the real AR)    -> the quantity of interest
    AR-semantic  paraphrase-invariant sentence embedder -> ~0 drop reference (H1 floor)

Reading: if AR-readout's ΔFVE under semantic-preserving paraphrase tracks AR-bow ->
surface/steganographic channel (H2). If it tracks AR-semantic (~0) -> semantic channel (H1).

Requires a finished run's checkpoints (run scripts/run_local_pilot.py first).
    python scripts/compare_ar.py --config experiments/exp02_open/config.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from nla_intervention import metrics as M
from nla_intervention.pipeline.readout import RidgeReconstructor, make_bow_features
from nla_intervention.utils import load_config

ROOT = Path(__file__).resolve().parents[1]


def _eval_ar(feature_fn, zmap, ids, H, train_idx, eval_idx, conditions, alpha):
    ar = RidgeReconstructor(feature_fn, alpha=alpha).fit(
        [zmap[ids[i]]["z"] for i in train_idx], H[train_idx])
    h_mean = H[eval_idx].mean(axis=0)
    per_cond = {c["code"]: [] for c in conditions}
    for i in eval_idx:
        for c in conditions:
            zp = zmap[ids[i]]["z_prime"][c["code"]]
            h_hat = np.asarray(ar.reconstruct(zp), dtype=np.float64)
            per_cond[c["code"]].append(float(M.fve_per_sample(H[i], h_hat, h_mean)[0]))
    mean_fve = {c: float(np.mean(v)) for c, v in per_cond.items()}
    base = mean_fve.get("C0", 0.0)
    return mean_fve, {c: base - v for c, v in mean_fve.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--train-frac", type=float, default=0.7)
    args = ap.parse_args()

    cfg = load_config(args.config)
    run_id = cfg["experiment"]["id"]
    interim = ROOT / "data/interim" / run_id
    acts = np.load(interim / "acts.npz", allow_pickle=True)
    zrows = [json.loads(l) for l in (interim / "z_zprime.jsonl").read_text().splitlines() if l.strip()]

    H = np.asarray(acts["h_l"], dtype=np.float64)
    ids = [str(x) for x in acts["input_id"]]
    zmap = {r["input_id"]: r for r in zrows}
    conds = load_config(ROOT / "configs/conditions.yaml")["conditions"]
    conditions = [c for c in conds if c["code"] in set(cfg["conditions"]["enabled"])]

    rng = np.random.default_rng(cfg["data"].get("seed", 0))
    perm = rng.permutation(len(ids))
    n_train = int(len(ids) * args.train_frac)
    train_idx, eval_idx = perm[:n_train], perm[n_train:]
    alpha = cfg.get("ar", {}).get("ridge_alpha", 10.0)

    # --- build the three feature extractors ---
    feats = {"bow": make_bow_features(
        dim=cfg["ar"]["bow"]["n_features"], ngram=tuple(cfg["ar"]["bow"]["ngram"]))}

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from nla_intervention.pipeline.readout import MReadoutFeatures
    rm = cfg["ar"]["readout"]["model"]
    rtok = AutoTokenizer.from_pretrained(rm)
    rmodel = AutoModelForCausalLM.from_pretrained(rm, dtype=torch.float16).to("mps").eval()
    feats["readout"] = MReadoutFeatures(rmodel, rtok, layer=cfg["ar"]["readout"]["layer"])

    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(cfg["embedding"]["model"])
    feats["semantic"] = lambda z: np.asarray(st.encode([z])[0], dtype=np.float64)

    # --- evaluate each AR ---
    rows = []
    for name, ff in feats.items():
        mean_fve, dfve = _eval_ar(ff, zmap, ids, H, train_idx, eval_idx, conditions, alpha)
        for c in conditions:
            rows.append({"ar": name, "condition": c["code"],
                         "fve": mean_fve[c["code"]], "delta_fve_vs_C0": dfve[c["code"]]})
    df = pd.DataFrame(rows)

    out = ROOT / "results" / run_id / "ar_comparison.csv"
    df.to_csv(out, index=False)

    print(f"=== ΔFVE spectrum (eval n={len(eval_idx)})  — wrote {out} ===\n")
    pivot = df.pivot(index="condition", columns="ar", values="delta_fve_vs_C0")
    pivot = pivot[["bow", "readout", "semantic"]]
    print("ΔFVE vs C0 (higher = more reconstruction lost under that condition):")
    print(pivot.to_string(float_format=lambda x: f"{x:+.3f}"))
    print("\nbow = pure-surface upper bound;  semantic = paraphrase-invariant floor.")
    print("Where does the REAL AR (readout) sit on semantic-preserving rows (C1-C3)?")
    print("  near bow  -> surface/steganographic channel (H2)")
    print("  near semantic(~0) -> semantic channel (H1)")


if __name__ == "__main__":
    main()
