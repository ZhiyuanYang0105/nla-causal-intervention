"""Utilities: config loading, device selection, seeding."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config, resolving a single `extends:` parent (deep merge).

    `extends:` is resolved RELATIVE TO THE CONFIG FILE (not the CWD), so a config loads
    correctly from any working directory."""
    import yaml  # lazy: keeps the package importable without the optional dep

    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    parent = cfg.pop("extends", None)
    if parent:
        ppath = Path(parent)
        if not ppath.is_absolute():
            ppath = path.parent / ppath            # relative to THIS file's directory
        cfg = _deep_merge(load_config(ppath), cfg)
    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        out[k] = _deep_merge(out[k], v) if isinstance(out.get(k), dict) and isinstance(v, dict) else v
    return out


def resolve_device(prefer: str | None = None) -> str:
    """Pick the compute device: explicit `prefer`, else cuda > mps > cpu. Lazy torch import
    so the package stays importable without torch."""
    if prefer:
        return prefer
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int) -> None:
    """Seed Python RNG (+ torch/numpy if available)."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
