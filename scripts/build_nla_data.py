#!/usr/bin/env python
"""Build the (h_l, summary) dataset for NLA warm-start: harvest activations from M over
FineWeb + generate summaries (batched). Optionally top up an existing set with --reuse.

    python scripts/build_nla_data.py --out exp04_nla --n 2500

Writes data/interim/<out>/{acts.npz, summaries.json}. Float32 mean-pool activations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from nla_intervention.data import HarvestConfig, harvest_activations

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PROMPT = ("Summarize the following text in 1-2 sentences, capturing its main topic "
                  "and key details. Output ONLY the summary.\n\n{text}")


def _summarizer(model_name="Qwen/Qwen2.5-0.5B-Instruct"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from nla_intervention.utils import resolve_device
    dev = resolve_device()
    dt = torch.bfloat16 if dev == "cuda" else torch.float32   # bf16 inference on GPU (saves VRAM)
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"                                  # left-pad for batched decode
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(model_name, dtype=dt).to(dev).eval()

    def summarize_batch(texts: list[str]) -> list[str]:
        prompts = [tok.apply_chat_template(
            [{"role": "user", "content": SUMMARY_PROMPT.format(text=t)}],
            add_generation_prompt=True, tokenize=False) for t in texts]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                  max_length=160).to(dev)
        n = enc["input_ids"].shape[1]
        with torch.no_grad():
            out = m.generate(**enc, max_new_tokens=48, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        return [tok.decode(o[n:], skip_special_tokens=True).strip() for o in out]
    return summarize_batch, m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="exp04_nla")
    ap.add_argument("--n", type=int, default=2500)
    ap.add_argument("--target-model", default="Qwen/Qwen2.5-0.5B")  # M (activations)
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--pooling", default="mean")              # "last" (paper) | "mean"
    ap.add_argument("--summarizer", default="Qwen/Qwen2.5-0.5B-Instruct")  # use 7B-Instruct on HPC
    ap.add_argument("--reuse", default=None,
                    help="optional: top up an existing data/interim/<reuse> instead of "
                         "harvesting all fresh")
    args = ap.parse_args()

    H, ids, src, dom, summaries = [], [], [], [], []
    if args.reuse:                                            # optional: top up an existing set
        reuse = ROOT / "data/interim" / args.reuse
        acts = np.load(reuse / "acts.npz", allow_pickle=True)
        H = list(np.asarray(acts["h_l"], dtype=np.float32))
        ids = [str(x) for x in acts["input_id"]]
        src = [str(s) for s in acts["source_text"]]
        dom = [str(d) for d in acts["domain"]]
        summaries = json.loads((reuse / "summaries.json").read_text())
        print(f"reuse {len(ids)} pairs from {args.reuse}; need {args.n - len(ids)} more")
    else:
        print(f"harvesting all {args.n} pairs fresh")
    have = set(ids)

    # harvest fresh activations (skip ids we already have). Use the "train" hash bucket (~98%
    # of docs) so we stream ~N docs, not ~50N: train_nla does its own 80/20 split of this set,
    # so the harvest-level eval bucket is unused here.
    hc = HarvestConfig(target_model=args.target_model, layer_l=args.layer, n_samples=args.n * 3,
                       max_snippet_tokens=96, pooling=args.pooling, split="train", seed=777)
    new = []
    for rec in harvest_activations(hc):
        if rec["input_id"] in have:
            continue
        have.add(rec["input_id"]); new.append(rec)
        if len(ids) + len(new) >= args.n:
            break
    print(f"harvested {len(new)} new activations; generating summaries...")

    summarize_batch, _ = _summarizer(args.summarizer)
    B = 16
    for k in range(0, len(new), B):
        chunk = new[k:k + B]
        sums = summarize_batch([r["source_text"] for r in chunk])
        for rec, s in zip(chunk, sums):
            H.append(np.asarray(rec["h_l"], dtype=np.float32))
            ids.append(rec["input_id"]); src.append(rec["source_text"]); dom.append(rec["domain"])
            summaries.append(s)
        print(f"  summarized {min(k+B, len(new))}/{len(new)}")

    out = ROOT / "data/interim" / args.out; out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "acts.npz", h_l=np.vstack(H).astype(np.float32),
             input_id=np.array(ids), source_text=np.array(src, dtype=object),
             domain=np.array(dom))
    (out / "summaries.json").write_text(json.dumps(summaries, ensure_ascii=False))
    print(f"wrote {out}/acts.npz + summaries.json  ({len(ids)} pairs)")


if __name__ == "__main__":
    main()
