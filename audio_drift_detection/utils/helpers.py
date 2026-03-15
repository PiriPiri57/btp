"""Misc helper utilities — config loading, path management."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path = "configs/config.yaml") -> dict[str, Any]:
    """Load YAML configuration file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Configuration dictionary.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if it doesn't exist.

    Args:
        path: Directory path.

    Returns:
        The resolved Path object.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_device() -> str:
    """Return 'cuda' if available, else 'cpu'."""
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"
