"""Utilities: config loading, seeding, IO, run metadata."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config, resolving a single `extends:` parent (deep merge)."""
    import yaml  # lazy: keeps the package importable without the optional dep

    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    parent = cfg.pop("extends", None)
    if parent:
        base = load_config(parent)
        cfg = _deep_merge(base, cfg)
    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        out[k] = _deep_merge(out[k], v) if isinstance(out.get(k), dict) and isinstance(v, dict) else v
    return out


def set_seed(seed: int) -> None:
    """Seed Python RNG. Extend with numpy/torch when models are added (M1)."""
    random.seed(seed)
