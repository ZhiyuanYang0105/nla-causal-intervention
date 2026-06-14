#!/usr/bin/env python
"""Steganography intervention test on the CO-TRAINED NLA (the real experiment).

Loads the jointly-trained AV/AR, generates z = AV(h_l) for held-out activations, then
paraphrases z (semantic-preserving: C1/C2/C3) and drifts it (C4), reconstructs ĥ = AR(z'),
and compares FVE. Now meaningful because AV/AR were co-trained (steganography CAN exist):

  - C1-C3 significantly drop FVE while meaning is preserved -> AV hid info in SURFACE form
    that paraphrase destroys = steganographic/surface channel (H2).
  - only C4 (drift) drops FVE -> information rides the SEMANTIC channel (H1).

    python scripts/nla_intervention.py --config experiments/exp04_nla/config.yaml --n-eval 120
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from nla_intervention import metrics as M
from nla_intervention import stats as S
from nla_intervention.conditions.rewriter import LLMRewriter
from nla_intervention.nla.av import ActivationVerbalizer
from nla_intervention.nla.ar import ActivationReconstructor
from nla_intervention.pipeline.runner import apply_condition
from nla_intervention.utils import load_config
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONDS = [
    {"code": "C0", "transform": "identity"},
    {"code": "C1", "transform": "paraphrase", "params": {"strength": "light"}},
    {"code": "C2", "transform": "paraphrase", "params": {"strength": "strong"}},
    {"code": "C3", "transform": "format_preserving_paraphrase"},
    {"code": "C4", "transform": "semantic_drift"},
]


def _make_rewriter(model_name="Qwen/Qwen2.5-0.5B-Instruct"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    m = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).to("mps").eval()

    def complete(prompt: str) -> str:
        enc = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                      add_generation_prompt=True, return_tensors="pt",
                                      return_dict=True).to("mps")
        n = enc["input_ids"].shape[1]
        with torch.no_grad():
            out = m.generate(**enc, max_new_tokens=48, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0, n:], skip_special_tokens=True).strip()
    return LLMRewriter(complete)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n-eval", type=int, default=120)
    args = ap.parse_args()

    cfg = load_config(args.config)
    run_id = cfg["experiment"]["id"]
    ck = ROOT / "results" / run_id / "ckpt"
    mname = cfg["target_model"]["name"]; layer = cfg["target_model"]["layer_l"]
    hid = cfg["target_model"]["hidden_size"]

    acts = np.load(ROOT / "data/interim" / cfg["train"]["warmstart_from"] / "acts.npz", allow_pickle=True)
    H = np.asarray(acts["h_l"], dtype=np.float32)
    ev = np.load(ck / "eval_idx.npy")[: args.n_eval]
    print(f"=== 隐写干预测试 (co-trained NLA, {run_id}) | n_eval={len(ev)} ===")

    av = ActivationVerbalizer(mname, dtype=torch.float32, adapter_path=str(ck / "av_lora"))
    ar = ActivationReconstructor(mname, layer_l=layer, hidden_size=hid, dtype=torch.float32,
                                 adapter_path=str(ck / "ar_lora"), affine_path=str(ck / "ar_affine.pt"))
    rewriter = _make_rewriter()
    try:
        from nla_intervention.metrics import SentenceTransformerEmbedder
        embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
    except Exception:
        embedder = None

    h_mean = H[ev].mean(0)
    rows = []
    for k, i in enumerate(ev):
        h = torch.tensor(H[i], dtype=torch.float32, device=ar.device)
        z = av.generate(h, n=1, max_new_tokens=32, temperature=0.0)[0]
        for c in CONDS:
            zp = apply_condition(z, c, rewriter=rewriter)
            hhat = ar.reconstruct(zp).float().cpu().numpy()
            row = {"input_id": str(acts["input_id"][i]), "condition": c["code"],
                   "z": z, "z_prime": zp,
                   "fve": float(M.fve_per_sample(H[i], hhat, h_mean)[0])}
            if embedder is not None:
                row["sim_zz_prime"] = M.sim_zz_prime(z, zp, embedder)
            rows.append(row)
        if (k + 1) % 20 == 0:
            print(f"  intervened {k+1}/{len(ev)}")
    df = pd.DataFrame(rows)

    out = ROOT / "results" / run_id
    df.to_csv(out / "intervention_metrics.csv", index=False)

    g = df.groupby("condition")["fve"].median().round(3)
    print("\n每条件 FVE 中位数:"); print(g.to_string())
    omni = S.omnibus(df, value="fve", method="friedman")
    tbl = S.pairwise_vs_baseline(df, value="fve", baseline="C0")
    print(f"\nFriedman p={omni['p_value']:.4g}")
    print("pairwise vs C0 (paired Wilcoxon, Holm):")
    print(tbl[["condition", "mean_delta", "cohen_dz", "p_corrected", "significant"]].to_string(index=False))
    if "sim_zz_prime" in df.columns:
        sims = df.groupby("condition")["sim_zz_prime"].mean().round(3)
        print("\nsim(z,z') 均值:"); print(sims.to_string())

    # verdict — RIGOROUS: require a significant omnibus before reading pairwise.
    # (A single incoherent pairwise hit with a tiny effect size is NOT evidence.)
    sp = tbl.set_index("condition")
    def _sig(c):
        return c in sp.index and bool(sp.loc[c, "significant"])
    n_sp_sig = sum(_sig(c) for c in ["C1", "C2", "C3"])
    print("\n=== 判定 ===")
    if omni["p_value"] >= 0.05:
        print(f"omnibus 不显著 (Friedman p={omni['p_value']:.3f}) → UNDETERMINED:"
              " 信号/功效不足,不能区分通道(单个 pairwise 命中不算证据)")
    elif n_sp_sig >= 2 and not _sig("C4"):
        print("多数语义保持改写显著降 FVE、漂移不显著 → 倾向 H2(表面/隐写通道)")
    elif _sig("C4") and n_sp_sig == 0:
        print("仅语义漂移显著降 FVE → 倾向 H1(语义通道)")
    else:
        print("效应不一致 → UNDETERMINED")

    report = {"run": run_id, "n_eval": len(ev), "fve_median": g.to_dict(),
              "friedman_p": omni["p_value"], "verdict": "undetermined" if omni["p_value"] >= 0.05 else "see-pairwise",
              "pairwise": tbl.to_dict(orient="records")}
    (out / "intervention_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {out}/intervention_metrics.csv + intervention_report.json")


if __name__ == "__main__":
    main()
