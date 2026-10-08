"""Shared utilities for the paper's experiment scripts."""
from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    """Load a YAML config file as a plain dict."""
    with open(path) as f:
        return yaml.safe_load(f)
