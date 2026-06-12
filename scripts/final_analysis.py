#!/usr/bin/env python
"""Robust final analysis: k-fold CV reconstruction + paired stats on the cached run.

Single train/eval splits gave wildly split-dependent FVE (-0.18..+0.11, eval=92 too
small). This uses 5-fold CV so EVERY sample is held out once -> stable per-sample FVE,
then paired Wilcoxon (C0 vs each condition). Reuses cached acts + z/z' (no regeneration);
readout features for z and all z' are computed once and cached.

    python scripts/final_analysis.py --config experiments/exp02_open/config.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from nla_intervention import stats as S
from nla_intervention.utils import load_config

ROOT = Path(__file__).resolve().parents[1]


def _readout_features(texts, model_name, layer, batch=16, max_tokens=96):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).to("mps").eval()
    out = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], return_tensors="pt", padding=True,
                  truncation=True, max_length=max_tokens).to("mps")
        with torch.no_grad():
            o = m(**enc, output_hidden_states=True)
        h = o.hidden_states[layer]; mk = enc["attention_mask"].unsqueeze(-1).float()
        out.append(((h * mk).sum(1) / mk.sum(1).clamp(min=1)).float().cpu().numpy())
    del m
    return np.vstack(out).astype(np.float64)


def _ridge_fit(X, H, alpha):
    xm, hm = X.mean(0), H.mean(0)
    W = np.linalg.solve((X - xm).T @ (X - xm) + alpha * np.eye(X.shape[1]), (X - xm).T @ (H - hm))
    return W, xm, hm


def _fve_ps(Ht, Hh, mean):
    return 1 - ((Ht - Hh) ** 2).sum(1) / ((Ht - mean) ** 2).sum(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=100.0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    run_id = cfg["experiment"]["id"]
    interim = ROOT / "data/interim" / run_id
    results = ROOT / "results" / run_id

    acts = np.load(interim / "acts.npz", allow_pickle=True)
    H = np.asarray(acts["h_l"], dtype=np.float64)
    ids = [str(x) for x in acts["input_id"]]
    domain = {i: d for i, d in zip(ids, acts["domain"])}
    zrows = [json.loads(l) for l in (interim / "z_zprime.jsonl").read_text().splitlines() if l.strip()]
    zmap = {r["input_id"]: r for r in zrows}
    conditions = [c for c in load_config(ROOT / "configs/conditions.yaml")["conditions"]
                  if c["code"] in set(cfg["conditions"]["enabled"])]
    codes = [c["code"] for c in conditions]
    rm, layer = cfg["ar"]["readout"]["model"], cfg["ar"]["readout"]["layer"]

    # --- compute (or load cached) readout features for z and each condition's z' ---
    fcache = interim / "readout_feats.npz"
    if fcache.exists():
        d = np.load(fcache, allow_pickle=True)
        Xz = d["Xz"]; Xzp = {c: d[f"Xzp_{c}"] for c in codes}
    else:
        Xz = _readout_features([zmap[i]["z"] for i in ids], rm, layer)
        Xzp = {c: _readout_features([zmap[i]["z_prime"][c] for i in ids], rm, layer) for c in codes}
        np.savez(fcache, Xz=Xz, **{f"Xzp_{c}": Xzp[c] for c in codes})
        print(f"cached readout features -> {fcache}")

    # --- 5-fold CV: fit on train z, predict each condition's z' on the held-out fold ---
    n = len(ids)
    rng = np.random.default_rng(cfg["data"].get("seed", 0))
    folds = np.array_split(rng.permutation(n), args.folds)
    rows = []
    for k, test in enumerate(folds):
        train = np.setdiff1d(np.arange(n), test)
        W, xm, hm = _ridge_fit(Xz[train], H[train], args.alpha)
        h_mean = H[test].mean(0)                        # FVE baseline on held-out fold
        for c in codes:
            pred = (Xzp[c][test] - xm) @ W + hm
            fve = _fve_ps(H[test], pred, h_mean)
            for j, t in enumerate(test):
                zt, zpt = zmap[ids[t]]["z"], zmap[ids[t]]["z_prime"][c]
                rows.append({"input_id": ids[t], "condition": c, "domain": domain[ids[t]],
                             "fve": float(fve[j]),
                             "sim_zz_prime": float(np.nan), "z": zt, "z_prime": zpt})
    df = pd.DataFrame(rows)

    # fill sim_zz' from the original metrics (already computed there) if available
    mp = results / "metrics.csv"
    if mp.exists():
        m0 = pd.read_csv(mp)[["input_id", "condition", "sim_zz_prime",
                              "jaccard_tokens", "ngram_overlap", "edit_distance_norm",
                              "js_divergence", "len_ratio"]]
        df = df.drop(columns=["sim_zz_prime"]).merge(m0, on=["input_id", "condition"], how="left")

    df.to_csv(results / "metrics_kfold.csv", index=False)

    # --- robust stats on the CV-pooled per-sample FVE ---
    g = df.groupby("condition")["fve"].agg(["mean", "median", "std"]).round(3)
    print(f"\n=== {run_id}: {args.folds}-fold CV FVE (n={n}, alpha={args.alpha}) ===")
    print(g)
    base = float(df[df.condition == "C0"].fve.mean())
    print(f"\nC0 baseline FVE = {base:+.3f}  ({'>0 OK — AR reconstructs' if base > 0 else 'still <=0'})")

    omni = S.omnibus(df, value="fve", method="friedman")
    print(f"\nFriedman: chi2={omni['statistic']:.2f}  p={omni['p_value']:.4g}")
    tbl = S.pairwise_vs_baseline(df, value="fve", baseline="C0")
    print("\npairwise vs C0 (paired Wilcoxon, Holm):")
    print(tbl[["condition", "mean_delta", "cohen_dz", "p_corrected", "significant"]].to_string(index=False))

    report = {"run": run_id, "folds": args.folds, "alpha": args.alpha, "n": n,
              "fve_by_condition": g["mean"].to_dict(), "c0_baseline_fve": base,
              "friedman": omni, "pairwise": tbl.to_dict(orient="records")}
    if "sim_zz_prime" in df.columns:
        report["manipulation_check"] = S.manipulation_check(df)["pass_rate_by_condition"]
    (results / "stats_report_kfold.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {results/'stats_report_kfold.json'}  and  metrics_kfold.csv")


if __name__ == "__main__":
    main()
