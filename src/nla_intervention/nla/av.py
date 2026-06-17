"""Activation Verbalizer (AV) — faithful to NLA (transformer-circuits 2026).

AV takes an ACTIVATION h_l (a vector) and produces a natural-language explanation z.
Mechanism (paper): normalize h_l to unit L2, scale by a constant, insert it IN PLACE OF a
placeholder token's embedding, then autoregressively sample z at temperature T.

LoRA-trainable. Exposes:
  - generate(h_l, n)                    : sample n explanations (GRPO group sampling / inference)
  - sequence_logprob(h_l, z, ref_model) : returns (sum log p(z|h_l), token-level KL to ref);
                                          the KL is the fluency penalty when ref_model is given

Only Opus-4.6 -> Qwen2.5-0.5B differs from the paper; the rest mirrors the method.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

# A fixed prompt; {ACT} marks the single position whose embedding is replaced by h_l.
PROMPT_PREFIX = "Describe the concept represented by this activation: "
PROMPT_SUFFIX = "\nDescription:"


class ActivationVerbalizer:
    def __init__(self, model_name: str, device: str = "mps", dtype=torch.float32,
                 act_scale: float | None = None, lora: bool = True, lora_r: int = 16,
                 adapter_path: str | None = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device)
        if adapter_path:                                   # load trained LoRA
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_path).to(device)
        elif lora:
            from peft import LoraConfig, get_peft_model
            cfg = LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.0,
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                             task_type="CAUSAL_LM")
            model = get_peft_model(model, cfg)
        self.model = model
        self.emb = model.get_input_embeddings()
        self.d = self.emb.weight.shape[1]

        # precompute prompt token embeddings (fixed); h_l is injected between prefix/suffix
        self.pre_ids = self.tok(PROMPT_PREFIX, return_tensors="pt").input_ids.to(device)
        self.suf_ids = self.tok(PROMPT_SUFFIX, add_special_tokens=False,
                                return_tensors="pt").input_ids.to(device)
        # scale h_l to roughly the token-embedding norm (paper uses a fixed constant)
        self.act_scale = act_scale or float(self.emb.weight.norm(dim=1).mean().item())

    # ---- input construction ------------------------------------------------

    def _prefix_embeds(self, h_l: torch.Tensor) -> torch.Tensor:
        """[prefix tokens] + [scaled h_l as one embedding] + [suffix tokens] -> (1, T, d)."""
        h = h_l.to(self.device).float()
        h = h / (h.norm() + 1e-6) * self.act_scale            # normalize then scale
        with torch.no_grad():
            pre = self.emb(self.pre_ids)                       # (1, P, d)
            suf = self.emb(self.suf_ids)                       # (1, S, d)
        act = h.to(pre.dtype).view(1, 1, self.d)
        return torch.cat([pre, act, suf], dim=1)               # (1, P+1+S, d)

    # ---- sampling (group sampling for GRPO / inference) --------------------

    @torch.no_grad()
    def generate(self, h_l: torch.Tensor, n: int = 1, max_new_tokens: int = 32,
                 temperature: float = 1.0) -> list[str]:
        emb = self._prefix_embeds(h_l).repeat(n, 1, 1)
        am = torch.ones(emb.shape[:2], device=self.device, dtype=torch.long)
        out = self.model.generate(
            inputs_embeds=emb, attention_mask=am, do_sample=temperature > 0,
            temperature=max(temperature, 1e-5), top_p=1.0, max_new_tokens=max_new_tokens,
            pad_token_id=self.tok.eos_token_id)
        return [self.tok.decode(o, skip_special_tokens=True).strip() for o in out]

    # ---- log-prob of a given z (teacher-forced; carries grad for GRPO) -----

    def sequence_logprob(self, h_l: torch.Tensor, z_text: str,
                         ref_model=None) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (sum_logprob, kl_to_ref) for z under AV. kl is token-mean KL to ref
        (the frozen init AV) over z's positions, used as the fluency penalty."""
        z_ids = self.tok(z_text, add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)
        pre = self._prefix_embeds(h_l)                         # (1, C, d)
        z_emb = self.emb(z_ids)                                # (1, L, d)
        full = torch.cat([pre, z_emb], dim=1)
        am = torch.ones(full.shape[:2], device=self.device, dtype=torch.long)
        logits = self.model(inputs_embeds=full, attention_mask=am).logits  # (1, C+L, V)
        C = pre.shape[1]
        # predict z_ids[t] from position C-1+t
        pred = logits[:, C - 1: C - 1 + z_ids.shape[1], :]
        logp = F.log_softmax(pred.float(), dim=-1)
        tok_lp = logp.gather(-1, z_ids.unsqueeze(-1)).squeeze(-1)          # (1, L)
        total = tok_lp.sum()

        kl = torch.tensor(0.0, device=self.device)
        if ref_model is not None:
            with torch.no_grad():
                rlogits = ref_model.model(inputs_embeds=full, attention_mask=am).logits
                rpred = rlogits[:, C - 1: C - 1 + z_ids.shape[1], :]
                rlogp = F.log_softmax(rpred.float(), dim=-1)
            kl = (logp.exp() * (logp - rlogp)).sum(-1).mean()
        return total, kl
