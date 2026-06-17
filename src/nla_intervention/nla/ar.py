"""Activation Reconstructor (AR) — faithful to NLA.

AR maps an explanation z back to an activation: z is wrapped in a fixed prompt, passed
through the model, and a LEARNED AFFINE MAP is applied to the layer-l activation at the
final token to obtain ĥ_l.

We read hidden_states[l] from the full model — numerically identical to "truncate to the
first l layers and take the output", which is how the target h_l was harvested (so AR's
representation lives at the same depth as the target). LoRA-trainable + the affine map.
"""
from __future__ import annotations

import torch
import torch.nn as nn

AR_PROMPT = "Concept description: {z}\nActivation:"


class ActivationReconstructor(nn.Module):
    def __init__(self, model_name: str, layer_l: int, hidden_size: int,
                 device: str = "mps", dtype=torch.float32, lora: bool = True, lora_r: int = 16,
                 adapter_path: str | None = None, affine_path: str | None = None):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device, self.layer_l = device, layer_l
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device)
        # Truncate to the first l transformer blocks (paper: "truncated to its first l
        # layers"). The AR's output is hidden_states[l], and blocks > l cannot affect it
        # (causal forward) — so this both saves compute AND ensures LoRA lands only on
        # layers that actually influence the output (otherwise upper-layer LoRA is dead).
        model.model.layers = model.model.layers[:layer_l]
        model.config.num_hidden_layers = layer_l
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
        # learned affine map: layer-l hidden (final token) -> activation space
        self.affine = nn.Linear(hidden_size, hidden_size).to(device).to(dtype)
        if affine_path:
            self.affine.load_state_dict(torch.load(affine_path, map_location=device))

    def _wrap(self, zs: list[str]):
        texts = [AR_PROMPT.format(z=z if z.strip() else ".") for z in zs]
        return self.tok(texts, return_tensors="pt", padding=True, truncation=True,
                        max_length=96).to(self.device)

    def forward(self, zs: list[str]) -> torch.Tensor:
        """z list -> ĥ_l  (B, d). Differentiable through LoRA + affine."""
        enc = self._wrap(zs)
        out = self.model(**enc, output_hidden_states=True)
        h = out.hidden_states[self.layer_l]                    # (B, T, d)
        mask = enc["attention_mask"]
        pos = torch.arange(mask.size(1), device=self.device)
        last = (mask * pos).argmax(dim=1)                      # final real token (pad-agnostic)
        h_last = h[torch.arange(h.size(0)), last]              # (B, d)
        return self.affine(h_last.to(self.affine.weight.dtype))

    @torch.no_grad()
    def reconstruct(self, z: str) -> torch.Tensor:
        return self.forward([z])[0]

    def trainable_parameters(self):
        ps = [p for p in self.model.parameters() if p.requires_grad]
        ps += list(self.affine.parameters())
        return ps
