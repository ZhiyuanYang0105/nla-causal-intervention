#!/usr/bin/env python
# ⚠️ PHASE 1 (text-proxy). Its AV/AR are NOT co-trained -> cannot test steganography;
#    the exp02 "H1" conclusion is RETRACTED. Faithful NLA is scripts/train_nla.py +
#    steg_intervention.py. See docs/nla_faithful_findings.md.
"""Staged LOCAL pilot orchestrator (MacBook M5 / 16GB). See docs/local_budget_plan.md.

Six stages, each checkpointed to disk so the run resumes; only ONE LLM is resident at a
time (each phase frees its model before the next).

    Stage 1  harvest    M(1B) -> activations h_l (selected layer, pooled, fp16)
    Stage 2  verbalize  Instruct -> z = summary(source)   (training-free AV)
    Stage 3  intervene  Instruct -> z'_k for each condition
    Stage 4  reconstruct ridge AR (bow | frozen-M readout): fit on train, predict eval
    Stage 5  metrics    FVE + sim_zz' + token-shift -> results/<run>/metrics.csv
    Stage 6  stats      Friedman / Wilcoxon / mechanism -> stats_report.json

Run for real (after `pip install -e ".[models,mlx,nlp]"`):
    python scripts/run_local_pilot.py --config experiments/exp01_pilot/config.yaml

Validate the orchestration with no models (uses fakes, runs anywhere):
    python scripts/run_local_pilot.py --config experiments/exp01_pilot/config.yaml --smoke

Flags: --force re-runs all stages; --force-stage N re-runs from stage N onward.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

from nla_intervention import stats as S
from nla_intervention.conditions import apply_condition
from nla_intervention.utils import load_config

SUMMARY_PROMPT = (
    "Summarize the following text in 1-2 sentences, capturing its main topic and key "
    "details. Output ONLY the summary.\n\n{text}"
)
ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- utils

def _free(*objs):
    for o in objs:
        del o
    gc.collect()
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def _load_conditions(cfg) -> list[dict]:
    conds = load_config(ROOT / "configs/conditions.yaml")["conditions"]
    enabled = set(cfg.get("conditions", {}).get("enabled", ["C0", "C1", "C2", "C3", "C4"]))
    return [c for c in conds if c["code"] in enabled]


def _read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def _write_jsonl(p: Path, rows: list[dict]) -> None:
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))


# --------------------------------------------------------------------- model loaders

def build_completer(model_cfg: dict):
    """Return complete(prompt)->str for an instruct model. transformers (MPS) or mlx."""
    runtime = model_cfg.get("runtime", "transformers")
    dec = model_cfg.get("decoding", {})
    if runtime == "mlx":
        from mlx_lm import generate, load
        model, tok = load(model_cfg["model"])

        def complete(prompt: str) -> str:
            msgs = [{"role": "user", "content": prompt}]
            text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            return generate(model, tok, prompt=text, max_tokens=dec.get("max_tokens", 128),
                            verbose=False).strip()
        return complete, (model, tok)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_cfg["model"])
    # float32: fp16 sampling on MPS is numerically unstable (multinomial gets nan/inf).
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["model"], dtype=torch.float32).to("mps").eval()

    def complete(prompt: str) -> str:
        msgs = [{"role": "user", "content": prompt}]
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True).to("mps")
        n_in = enc["input_ids"].shape[1]
        with torch.no_grad():
            # greedy decoding: deterministic + avoids the multinomial nan path entirely
            out = model.generate(**enc, max_new_tokens=dec.get("max_tokens", 128),
                                 do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0, n_in:], skip_special_tokens=True).strip()
    return complete, (model, tok)


def build_ar_feature_fn(cfg, smoke: bool = False):
    """Return (feature_fn, resources_to_free). bow = CPU; readout = reload frozen M."""
    from nla_intervention.pipeline.readout import MReadoutFeatures, make_bow_features

    ar = cfg.get("ar", {})
    if not smoke and ar.get("type", "bow") == "readout":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        m = ar["readout"]["model"]
        tok = AutoTokenizer.from_pretrained(m)
        # float32: Qwen residual-stream has massive outlier activations that overflow fp16
        model = AutoModelForCausalLM.from_pretrained(m, dtype=torch.float32).to("mps").eval()
        return MReadoutFeatures(model, tok, layer=ar["readout"]["layer"]), (model, tok)
    bow = ar.get("bow", {})
    return make_bow_features(dim=bow.get("n_features", 4096),
                             ngram=tuple(bow.get("ngram", [1, 2]))), None


def build_embedder(cfg, smoke: bool):
    from nla_intervention.metrics import FakeEmbedder
    if smoke:
        return FakeEmbedder()
    from nla_intervention.metrics import SentenceTransformerEmbedder
    return SentenceTransformerEmbedder(cfg["embedding"]["model"].split("/")[-1])


# ------------------------------------------------------------------------- stages

def stage1_harvest(cfg, interim: Path, smoke: bool) -> dict:
    f = interim / "acts.npz"
    if f.exists():
        d = np.load(f, allow_pickle=True)
        return {k: d[k] for k in d.files}
    n = cfg["data"].get("n_samples", 200)
    if smoke:
        rng = np.random.default_rng(0)
        dim = 64
        vocab = [f"tok{i}" for i in range(40)]
        source = [" ".join(rng.choice(vocab, size=rng.integers(20, 40))) for _ in range(n)]
        # activation = bow(source) @ W_true  -> learnable signal for the ridge AR
        from nla_intervention.pipeline.readout import hashing_ngram_features
        X = np.vstack([hashing_ngram_features(s, dim=128) for s in source])
        W = rng.normal(size=(128, dim))
        H = (X @ W).astype(np.float16)
        ids = np.array([f"s{i}" for i in range(n)])
        dom = np.array(["smoke"] * n)
    else:
        from nla_intervention.data import HarvestConfig, harvest_activations
        hc = HarvestConfig(
            target_model=cfg["target_model"]["name"], layer_l=cfg["target_model"]["layer_l"],
            corpus=cfg["data"]["corpus"], subset=cfg["data"]["subset"],
            n_samples=n, max_snippet_tokens=cfg["data"]["max_snippet_tokens"],
            pooling=cfg["data"].get("pooling", "last"), seed=cfg["data"].get("seed", 0))
        recs = list(harvest_activations(hc))
        H = np.vstack([r["h_l"] for r in recs]).astype(np.float16)
        ids = np.array([r["input_id"] for r in recs])
        source = [r["source_text"] for r in recs]
        dom = np.array([r["domain"] for r in recs])
    np.savez(f, h_l=H, input_id=ids, source_text=np.array(source, dtype=object), domain=dom)
    print(f"[stage1] harvested {len(ids)} activations dim={H.shape[1]} -> {f}")
    return {"h_l": H, "input_id": ids, "source_text": np.array(source, dtype=object), "domain": dom}


def stage2_3_verbalize_intervene(cfg, acts, interim: Path, conditions, smoke: bool) -> list[dict]:
    f = interim / "z_zprime.jsonl"
    if f.exists():
        return _read_jsonl(f)
    if smoke:
        from nla_intervention.conditions.rewriter import FakeRewriter
        rewriter = FakeRewriter()

        def summarize(text: str) -> str:                 # deterministic: keep ~60% of words
            w = text.split()
            return " ".join(w[: max(1, int(len(w) * 0.6))])
        res = None
    else:
        from nla_intervention.conditions.rewriter import LLMRewriter
        complete, res = build_completer(cfg["paraphraser"])
        av_complete = complete  # AV shares the instruct model
        rewriter = LLMRewriter(complete)

        def summarize(text: str) -> str:
            return av_complete(SUMMARY_PROMPT.format(text=text))

    rows = []
    for i, (iid, src) in enumerate(zip(acts["input_id"], acts["source_text"])):
        z = summarize(str(src))
        zprime = {c["code"]: apply_condition(z, c, rewriter=rewriter) for c in conditions}
        rows.append({"input_id": str(iid), "z": z, "z_prime": zprime})
        if (i + 1) % 25 == 0:
            print(f"[stage2-3] verbalized+intervened {i + 1}/{len(acts['input_id'])}")
    if not smoke:
        _free(*res)
    _write_jsonl(f, rows)
    print(f"[stage2-3] wrote {f} ({len(rows)} z + {len(conditions)} conditions each)")
    return rows


def stage4_5_reconstruct_metrics(cfg, acts, zrows, conditions, results: Path, smoke: bool,
                                 train_frac=0.7, seed=0) -> pd.DataFrame:
    f = results / "metrics.csv"
    if f.exists():
        return pd.read_csv(f)
    from nla_intervention import metrics as M
    from nla_intervention.pipeline.readout import RidgeReconstructor

    H = np.asarray(acts["h_l"], dtype=np.float64)
    ids = [str(x) for x in acts["input_id"]]
    zmap = {r["input_id"]: r for r in zrows}
    domain = {str(i): d for i, d in zip(acts["input_id"], acts["domain"])}

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ids))
    n_train = int(len(ids) * train_frac)
    train_idx, eval_idx = perm[:n_train], perm[n_train:]

    feature_fn, ar_res = build_ar_feature_fn(cfg, smoke)
    arc = cfg.get("ar", {})
    z_train = [zmap[ids[i]]["z"] for i in train_idx]
    ar = RidgeReconstructor(feature_fn, alpha=arc.get("ridge_alpha", 10.0),
                            feat_pca=arc.get("feat_pca")).fit(z_train, H[train_idx])
    print(f"[stage4] AR fitted (alpha={ar.alpha_}, feat_pca={arc.get('feat_pca')}, "
          f"n_train={len(z_train)})")

    h_mean = H[eval_idx].mean(axis=0)                    # FVE baseline = eval mean E[h_l]
    embedder = build_embedder(cfg, smoke)

    rows = []
    for i in eval_idx:
        iid = ids[i]
        z = zmap[iid]["z"]
        for c in conditions:
            zp = zmap[iid]["z_prime"][c["code"]]
            h_hat = np.asarray(ar.reconstruct(zp), dtype=np.float64)
            row = {"input_id": iid, "condition": c["code"], "domain": domain[iid],
                   "z": z, "z_prime": zp,
                   "fve": float(M.fve_per_sample(H[i], h_hat, h_mean)[0]),
                   "activation_cosine": float(M.activation_cosine(H[i], h_hat)[0]),
                   "delta_len": M.delta_len(z, zp), "len_ratio": M.len_ratio(z, zp),
                   "jaccard_tokens": M.jaccard_tokens(z, zp),
                   "ngram_overlap": M.ngram_overlap(z, zp),
                   "edit_distance_norm": M.edit_distance_norm(z, zp),
                   "js_divergence": M.js_divergence(z, zp),
                   "sim_zz_prime": M.sim_zz_prime(z, zp, embedder)}
            rows.append(row)
    if ar_res is not None:
        _free(*ar_res)
    df = pd.DataFrame(rows)
    df.to_csv(f, index=False)
    print(f"[stage4-5] wrote {f} ({len(df)} rows; train={n_train}, eval={len(eval_idx)})")
    return df


def stage6_stats(df, results: Path, baseline="C0") -> dict:
    report = {"n_rows": len(df), "conditions": sorted(df["condition"].unique())}
    report["manipulation_check"] = S.manipulation_check(df)["pass_rate_by_condition"]
    report["omnibus"] = S.omnibus(df, value="fve", method="friedman")
    tbl = S.pairwise_vs_baseline(df, value="fve", baseline=baseline)
    report["pairwise_vs_baseline"] = tbl.to_dict(orient="records")
    try:
        res = S.mechanism_regression(df, baseline=baseline, value="fve")
        report["mechanism_regression"] = {
            "surface_shift_coef": float(res.params.get("surface_shift", float("nan"))),
            "surface_shift_p": float(res.pvalues.get("surface_shift", float("nan"))),
            "sim_zz_prime_p": float(res.pvalues.get("sim_zz_prime", float("nan")))}
    except Exception as e:
        report["mechanism_regression"] = {"error": str(e)}
    (results / "stats_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\n[stage6] Friedman p={report['omnibus']['p_value']:.3g}")
    print(tbl[["condition", "mean_delta", "cohen_dz", "p_corrected", "significant"]]
          .to_string(index=False))
    print(f"\nwrote {results/'stats_report.json'}")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true", help="run with fakes (no models)")
    ap.add_argument("--force", action="store_true", help="re-run all stages")
    args = ap.parse_args()

    cfg = load_config(args.config)
    run_id = cfg.get("experiment", {}).get("id", "run") + ("_smoke" if args.smoke else "")
    interim = ROOT / "data/interim" / run_id
    results = ROOT / "results" / run_id
    interim.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    if args.force:
        for p in [interim / "acts.npz", interim / "z_zprime.jsonl", results / "metrics.csv"]:
            p.unlink(missing_ok=True)

    conditions = _load_conditions(cfg)
    print(f"=== local pilot '{run_id}'  conditions={[c['code'] for c in conditions]} "
          f"smoke={args.smoke} ===")

    acts = stage1_harvest(cfg, interim, args.smoke)
    zrows = stage2_3_verbalize_intervene(cfg, acts, interim, conditions, args.smoke)
    df = stage4_5_reconstruct_metrics(cfg, acts, zrows, conditions, results, args.smoke)
    stage6_stats(df, results)


if __name__ == "__main__":
    main()
