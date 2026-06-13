"""Faithful NLA training: warm-start SFT + GRPO joint training (decoupled updates).

Mirrors transformer-circuits 2026 (only Opus-4.6 -> Qwen2.5-0.5B):
  warm-start : SFT AV on (h_l -> summary s), AR on (s -> h_l)        ~0.3-0.4 FVE
  joint      : AR via supervised MSE on sampled z;                    (z treated as input)
               AV via GRPO with reward r=-||h_l-AR(z)||^2,            (AR a fixed scorer)
               + KL penalty to the AV init (adapters disabled) for fluency.
Updates are decoupled: no gradient flows AV<->AR within a step.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _fve(H, Hh):
    H, Hh = np.asarray(H), np.asarray(Hh)
    return 1.0 - ((H - Hh) ** 2).sum() / ((H - H.mean(0)) ** 2).sum()


# --------------------------------------------------------------- warm-start

def _snapshot(av, ar):
    av_s = {n: p.detach().clone() for n, p in av.model.named_parameters() if p.requires_grad}
    ar_s = {n: p.detach().clone() for n, p in ar.model.named_parameters() if p.requires_grad}
    aff_s = {n: p.detach().clone() for n, p in ar.affine.named_parameters()}
    return av_s, ar_s, aff_s


def _restore(av, ar, snap):
    av_s, ar_s, aff_s = snap
    with torch.no_grad():
        for n, p in av.model.named_parameters():
            if n in av_s: p.copy_(av_s[n])
        for n, p in ar.model.named_parameters():
            if n in ar_s: p.copy_(ar_s[n])
        for n, p in ar.affine.named_parameters():
            p.copy_(aff_s[n])


def warmstart(av, ar, H, summaries, *, epochs=5, lr=1e-4, ar_lr=None, batch=8,
              weight_decay=0.01, eval_H=None, eval_summaries=None, log=print):
    """SFT: AV learns h_l->s (LM loss), AR learns s->h_l (MSE). Early-stops on the held-out
    AR-only FVE (keeps best trainable params) to curb overfitting on small data."""
    opt_av = torch.optim.AdamW([p for p in av.model.parameters() if p.requires_grad],
                               lr=lr, weight_decay=weight_decay)
    opt_ar = torch.optim.AdamW(ar.trainable_parameters(), lr=ar_lr or lr, weight_decay=weight_decay)
    n = len(summaries)
    Ht = torch.tensor(np.asarray(H), dtype=torch.float32, device=ar.device)
    best_fve, best_snap = -1e9, None
    for ep in range(epochs):
        order = np.random.default_rng(ep).permutation(n)
        av_loss = ar_loss = 0.0
        for i in order:
            lp, _ = av.sequence_logprob(Ht[i], summaries[i])
            loss_av = -lp / max(1, len(summaries[i].split()))
            opt_av.zero_grad(); loss_av.backward(); opt_av.step()
            av_loss += float(loss_av)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            hhat = ar([summaries[j] for j in idx])
            loss_ar = F.mse_loss(hhat, Ht[idx])
            opt_ar.zero_grad(); loss_ar.backward(); opt_ar.step()
            ar_loss += float(loss_ar) * len(idx)
        msg = f"[warmstart] ep{ep+1}/{epochs} av_nll={av_loss/n:.3f} ar_mse={ar_loss/n:.3f}"
        if eval_H is not None:                                # early stopping on eval AR-only FVE
            fve = eval_ar_on_text(ar, eval_summaries, eval_H)
            msg += f" eval_AR_FVE={fve:+.3f}"
            if fve > best_fve:
                best_fve, best_snap = fve, _snapshot(av, ar)
                msg += " *"
        log(msg)
    if best_snap is not None:
        _restore(av, ar, best_snap)
        log(f"[warmstart] restored best (eval AR-only FVE={best_fve:+.3f})")
    return best_fve


# --------------------------------------------------------------- GRPO joint

def grpo_train(av, ar, H, *, steps=200, group=8, batch=4, beta=0.02,
               lr_av=1e-5, lr_ar=2e-4, ar_steps=4, ar_mb=16, buffer_cap=1024,
               max_new_tokens=32, eval_H=None, eval_every=10, log=print):
    """Joint training, STABILIZED. Each step:
      1) sample a group of z per activation, reward = -MSE (AR fixed scorer)
      2) AV: GRPO with group-normalized advantage + PER-TOKEN KL to init (adapters off)
      3) AR: several minibatch MSE steps from a replay buffer of sampled (z, h)
    The replay buffer + multi-step AR cuts the high single-step variance that stalled
    the naive version; per-token KL keeps the penalty O(1) instead of O(seq_len)."""
    opt_av = torch.optim.AdamW([p for p in av.model.parameters() if p.requires_grad], lr=lr_av)
    opt_ar = torch.optim.AdamW(ar.trainable_parameters(), lr=lr_ar)
    Ht = torch.tensor(np.asarray(H), dtype=torch.float32, device=ar.device)
    n = len(H)
    rng = np.random.default_rng(0)
    buffer: list[tuple[str, torch.Tensor]] = []

    for step in range(steps):
        idx = rng.integers(0, n, size=batch)
        all_z, all_h, all_reward = [], [], []
        for bi in idx:                                         # group sampling + rewards
            h = Ht[bi]
            zs = av.generate(h, n=group, max_new_tokens=max_new_tokens, temperature=1.0)
            with torch.no_grad():
                r = -((ar(zs) - h.unsqueeze(0)) ** 2).mean(dim=1)
            all_z.append(zs); all_h.append(h); all_reward.append(r)
            buffer.extend((z, h.detach()) for z in zs)

        # ---- AV: GRPO + per-token KL to init ----
        opt_av.zero_grad()
        kl_acc = 0.0
        for zs, h, r in zip(all_z, all_h, all_reward):
            adv = (r - r.mean()) / (r.std() + 1e-6)
            for z, a in zip(zs, adv):
                lp = av.sequence_logprob(h, z)[0]
                with av.model.disable_adapter():
                    lp_ref = av.sequence_logprob(h, z)[0].detach()
                ntok = max(1, len(av.tok(z, add_special_tokens=False).input_ids))
                kl = (lp - lp_ref) / ntok                      # per-token KL surrogate
                ((-(a.detach() * lp) + beta * kl) / (group * batch)).backward()
                kl_acc += float(kl) / (group * batch)
        torch.nn.utils.clip_grad_norm_([p for p in av.model.parameters() if p.requires_grad], 1.0)
        opt_av.step()

        # ---- AR: replay-buffer minibatch MSE (decoupled, lower variance) ----
        buffer = buffer[-buffer_cap:]
        ar_loss_acc = 0.0
        for _ in range(ar_steps):
            mb = [buffer[j] for j in rng.integers(0, len(buffer), size=min(ar_mb, len(buffer)))]
            hhat = ar([z for z, _ in mb])
            loss = F.mse_loss(hhat, torch.stack([h for _, h in mb]))
            opt_ar.zero_grad(); loss.backward(); opt_ar.step()
            ar_loss_acc += float(loss)

        if step % eval_every == 0 or step == steps - 1:
            msg = (f"[grpo] step{step} reward={float(torch.cat(all_reward).mean()):.3f} "
                   f"ar_mse={ar_loss_acc/ar_steps:.3f} kl/tok={kl_acc:.3f}")
            if eval_H is not None:
                msg += f" eval_FVE={eval_fve(av, ar, eval_H):+.3f}"
            log(msg)


@torch.no_grad()
def eval_fve(av, ar, H, max_new_tokens=32):
    """Greedy z from each h (AV->AR chain), reconstruct, set-level FVE."""
    Ht = np.asarray(H)
    hhat = []
    for i in range(len(Ht)):
        h = torch.tensor(Ht[i], dtype=torch.float32, device=ar.device)
        z = av.generate(h, n=1, max_new_tokens=max_new_tokens, temperature=0.0)[0]
        hhat.append(ar.reconstruct(z).float().cpu().numpy())
    return float(_fve(Ht, np.vstack(hhat)))


@torch.no_grad()
def eval_ar_on_text(ar, texts, H):
    """FVE of AR alone, fed the given texts (e.g. true summaries). Isolates AR quality
    from AV generation quality."""
    hhat = np.vstack([ar.reconstruct(t).float().cpu().numpy() for t in texts])
    return float(_fve(np.asarray(H), hhat))
