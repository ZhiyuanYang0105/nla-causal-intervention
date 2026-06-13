#!/usr/bin/env python
"""Train a faithful local NLA: warm-start SFT [+ GRPO joint training].

    python scripts/train_nla.py --config experiments/exp03_nla/config.yaml            # warm-start
    python scripts/train_nla.py --config experiments/exp03_nla/config.yaml --grpo     # + GRPO

Reuses cached activations + summaries from a prior run's data/interim/<src>/ for warm-start.
Saves AV (LoRA) + AR (LoRA + affine) to results/<run>/ckpt/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from nla_intervention.nla.av import ActivationVerbalizer
from nla_intervention.nla.ar import ActivationReconstructor
from nla_intervention.nla import train as T
from nla_intervention.utils import load_config

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--grpo", action="store_true")
    ap.add_argument("--ws-epochs", type=int, default=3)
    ap.add_argument("--grpo-steps", type=int, default=150)
    args = ap.parse_args()

    cfg = load_config(args.config)
    run_id = cfg["experiment"]["id"]
    src = cfg["train"]["warmstart_from"]                       # interim dir with acts+summaries
    interim = ROOT / "data/interim" / src
    out = ROOT / "results" / run_id; (out / "ckpt").mkdir(parents=True, exist_ok=True)

    acts = np.load(interim / "acts.npz", allow_pickle=True)
    H = np.asarray(acts["h_l"], dtype=np.float32)
    ids = [str(x) for x in acts["input_id"]]
    if (interim / "summaries.json").exists():                  # new combined format
        summaries = json.loads((interim / "summaries.json").read_text())
    else:                                                      # legacy z_zprime.jsonl
        zrows = {r["input_id"]: r for r in
                 (json.loads(l) for l in (interim / "z_zprime.jsonl").read_text().splitlines() if l.strip())}
        summaries = [zrows[i]["z"] for i in ids]

    rng = np.random.default_rng(cfg["data"].get("seed", 0))
    perm = rng.permutation(len(ids)); ntr = int(len(ids) * 0.8)
    tr, ev = perm[:ntr], perm[ntr:]

    mname = cfg["target_model"]["name"]; layer = cfg["target_model"]["layer_l"]
    hid = cfg["target_model"]["hidden_size"]
    print(f"=== NLA train '{run_id}'  M={mname} layer={layer} n_train={len(tr)} ===")

    av = ActivationVerbalizer(mname, dtype=torch.float32, lora=True,
                              lora_r=cfg["train"].get("lora_r", 16))
    ar = ActivationReconstructor(mname, layer_l=layer, hidden_size=hid,
                                 dtype=torch.float32, lora=True,
                                 lora_r=cfg["train"].get("lora_r", 16))

    print(f"数据 {len(ids)} 对 | 初始 eval FVE:", round(T.eval_fve(av, ar, H[ev]), 3))
    T.warmstart(av, ar, H[tr], [summaries[i] for i in tr],
                epochs=args.ws_epochs, lr=cfg["train"].get("ws_lr", 1e-4),
                ar_lr=cfg["train"].get("ar_lr", 5e-4),
                weight_decay=cfg["train"].get("weight_decay", 0.01),
                eval_H=H[ev], eval_summaries=[summaries[i] for i in ev])
    fve_chain = T.eval_fve(av, ar, H[ev])               # AV->AR full chain (greedy)
    fve_ar = T.eval_ar_on_text(ar, [summaries[i] for i in ev], H[ev])  # AR alone on true summaries
    print(f"warm-start 后: chain FVE(AV->AR)={fve_chain:+.3f} | AR-only FVE(真摘要)={fve_ar:+.3f}")
    print("  (AR-only 高而 chain 低 => 瓶颈在 AV 生成,GRPO 会补;两者都低 => 摘要信息不足)")

    if args.grpo:
        tc = cfg["train"]
        T.grpo_train(av, ar, H[tr],
                     steps=args.grpo_steps, group=tc.get("group", 8),
                     batch=tc.get("grpo_batch", 4), beta=tc.get("kl_beta", 0.02),
                     lr_av=tc.get("grpo_lr_av", 1e-5), lr_ar=tc.get("grpo_lr_ar", 2e-4),
                     ar_steps=tc.get("ar_steps", 4), eval_H=H[ev])
        fve_g = T.eval_fve(av, ar, H[ev])
        print(f"GRPO 后 eval FVE: {fve_g:+.3f}")

    # save checkpoints
    av.model.save_pretrained(out / "ckpt/av_lora")
    ar.model.save_pretrained(out / "ckpt/ar_lora")
    torch.save(ar.affine.state_dict(), out / "ckpt/ar_affine.pt")
    np.save(out / "ckpt/eval_idx.npy", ev)
    print(f"saved checkpoints -> {out/'ckpt'}")


if __name__ == "__main__":
    main()
