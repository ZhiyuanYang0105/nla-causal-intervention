"""Harvest residual-stream activations h_l from M (Llama-3.1-8B) over FineWeb.

Recipe (docs/paper_findings.md): take pretraining-like snippets, randomly truncate,
run M, read the layer-l residual stream at the FINAL token.

Heavy deps (torch/transformers/datasets) are imported lazily inside functions so the
package stays importable without them. Requires GPU + HF access to actually run.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass
class HarvestConfig:
    # defaults match the faithful local setup (ungated Qwen2.5-0.5B on MPS)
    target_model: str = "Qwen/Qwen2.5-0.5B"
    layer_l: int = 12
    corpus: str = "HuggingFaceFW/fineweb"
    subset: str = "sample-10BT"
    streaming: bool = True
    text_file: str | None = None      # local jsonl ({"text":...}/line) -> OFFLINE, no HF streaming
    n_samples: int | None = 200
    min_snippet_tokens: int = 16
    max_snippet_tokens: int = 128     # seq len cap
    eval_holdout_docs: int = 20_000   # only docs assigned to the eval split are yielded
    split: str = "eval"               # "eval" (disjoint) | "train"
    pooling: str = "last"             # "last" (paper) | "mean" — ONE vector per sample
    normalize: bool = True            # unit-L2-normalize h_l (paper); keeps AV in / AR target / FVE consistent
    store_dtype: str = "float32"      # fp16 overflows on Qwen-style outlier activations
    batch_size: int = 16
    dtype: str = "float32"            # model compute dtype (fp16 can produce inf/nan)
    device: str | None = None         # None -> auto (cuda > mps > cpu)
    seed: int = 20260612


def _split_of(doc_id: str, cfg: HarvestConfig) -> str:
    """Deterministic, reproducible train/eval assignment by document id hash.

    Eval = first `eval_holdout_docs` ids by hash bucket; guarantees the eval set is
    disjoint from training docs regardless of stream order (no leakage into AV/AR).
    """
    h = int(hashlib.sha1(doc_id.encode()).hexdigest(), 16)
    return "eval" if (h % 1_000_000) < cfg.eval_holdout_docs else "train"


def _domain_of(record: dict) -> str:
    """Derive a coarse domain bucket from a FineWeb record's URL (for stratification)."""
    url = (record.get("url") or "").lower()
    if not url:
        return "unknown"
    host = url.split("//")[-1].split("/")[0]
    tld = host.rsplit(".", 1)[-1] if "." in host else "unknown"
    return tld  # e.g. com/org/edu/gov/... ; refine as needed


def harvest_activations(cfg: HarvestConfig | None = None, **overrides) -> Iterator[dict]:
    """Yield {input_id, h_l: np.ndarray(d,), source_text, domain} for the chosen split.

    One record == one activation (final token of one truncated snippet).
    """
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = cfg or HarvestConfig()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    from nla_intervention.utils import resolve_device
    cfg.device = resolve_device(cfg.device)               # cuda > mps > cpu
    rng = np.random.default_rng(cfg.seed)

    tok = AutoTokenizer.from_pretrained(cfg.target_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.target_model, dtype=getattr(torch, cfg.dtype)
    ).to(cfg.device).eval()

    # text source: a pre-fetched local jsonl ({"text": ...} per line, OFFLINE-capable) or
    # streamed FineWeb. Local files are used as-is (no hash split — the file IS the curated set).
    if cfg.text_file:
        import json

        def _local_docs():
            with open(cfg.text_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield json.loads(line)
        ds, use_split = _local_docs(), False
    else:
        ds = load_dataset(cfg.corpus, name=cfg.subset, split="train", streaming=cfg.streaming)
        use_split = True

    batch: list[dict] = []
    yielded = 0

    def flush(items):
        nonlocal yielded
        texts = [it["text"] for it in items]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=cfg.max_snippet_tokens).to(cfg.device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)   # portable across architectures
        # hidden_states[k]: residual stream entering block k (k=0 embeddings)
        h = out.hidden_states[cfg.layer_l]                  # (B, T, d)
        mask = enc["attention_mask"]                         # (B, T)
        if cfg.pooling == "mean":                           # masked mean over sequence
            m = mask.unsqueeze(-1).float()
            pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        else:                                               # "last": final real token
            # padding-side agnostic: largest position index where mask==1
            pos = torch.arange(mask.size(1), device=mask.device)
            last_idx = (mask * pos).argmax(dim=1)
            pooled = h[torch.arange(h.size(0)), last_idx]
        if cfg.normalize:                                   # paper: normalize h_l to unit L2.
            pooled = pooled / (pooled.norm(dim=1, keepdim=True) + 1e-6)  # keeps AV input,
            #                                                    AR target, and FVE consistent.
        pooled = pooled.float().cpu().numpy().astype(cfg.store_dtype)
        for it, vec in zip(items, pooled):
            yielded += 1
            yield {
                "input_id": it["input_id"],
                "h_l": vec,
                "source_text": it["text"],
                "domain": it["domain"],
            }

    try:
        for i, record in enumerate(ds):
            doc_id = record.get("id") or f"doc-{i}"
            if use_split and _split_of(doc_id, cfg) != cfg.split:
                continue
            ids = tok(record["text"], truncation=False)["input_ids"]
            if len(ids) < cfg.min_snippet_tokens:
                continue
            # random truncation length in [min, min(max, len)]
            hi = min(cfg.max_snippet_tokens, len(ids))
            cut = int(rng.integers(cfg.min_snippet_tokens, hi + 1))
            snippet = tok.decode(ids[:cut], skip_special_tokens=True)
            batch.append({"input_id": doc_id, "text": snippet, "domain": _domain_of(record)})
            if len(batch) >= cfg.batch_size:
                yield from flush(batch)
                batch = []
            if cfg.n_samples is not None and yielded >= cfg.n_samples:
                return
        if batch:
            yield from flush(batch)
    finally:
        del model


def dataset_mean_activation(records: Iterator[dict]) -> np.ndarray:
    """Streaming mean E[h_l] over harvested activations — the FVE baseline h̄_l.

    Accepts an iterator of harvest records; returns mean vector (d,). Compute once on
    the eval split and cache (e.g. data/processed/<run>/h_mean.npy).
    """
    total = None
    n = 0
    for rec in records:
        v = np.asarray(rec["h_l"], dtype=np.float64)
        total = v.copy() if total is None else total + v
        n += 1
    if n == 0:
        raise ValueError("no activations provided")
    return (total / n).astype(np.float32)
