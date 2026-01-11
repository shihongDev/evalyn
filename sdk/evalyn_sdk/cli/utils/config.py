"""Configuration utilities for CLI: config loading, dataset path resolution."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

from ..constants import DEFAULT_CONFIG_PATHS


def _expand_env_vars(value: Any) -> Any:
    """Recursively expand environment variables in config values."""
    if isinstance(value, str):
        # Expand ${VAR} or $VAR patterns
        def replace_env(match):
            var_name = match.group(1) or match.group(2)
            return os.environ.get(var_name, match.group(0))

        return re.sub(r"\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)", replace_env, value)
    elif isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    return value


def load_config() -> Dict[str, Any]:
    """Load configuration from evalyn.yaml or .evalynrc if present."""
    for config_path in DEFAULT_CONFIG_PATHS:
        path = Path(config_path)
        if path.exists():
            try:
                import yaml  # Optional dependency

                with open(path) as f:
                    config = yaml.safe_load(f) or {}
                    # Expand environment variables
                    config = _expand_env_vars(config)
                    return config
            except ImportError:
                # Try JSON format if yaml not available
                try:
                    with open(path) as f:
                        config = json.load(f)
                        config = _expand_env_vars(config)
                        return config
                except Exception:
                    pass
            except Exception:
                pass
    return {}


def get_config_default(config: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Get nested config value with fallback."""
    value = config
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key)
        else:
            return default
        if value is None:
            return default
    return value


def find_latest_dataset(data_dir: str = "data") -> Optional[Path]:
    """Find the most recently modified dataset directory."""
    data_path = Path(data_dir)
    if not data_path.exists():
        return None

    # Find directories containing dataset.jsonl
    dataset_dirs = []
    for d in data_path.iterdir():
        if d.is_dir() and (d / "dataset.jsonl").exists():
            dataset_dirs.append(d)

    if not dataset_dirs:
        return None

    # Sort by modification time (most recent first)
    dataset_dirs.sort(key=lambda d: (d / "dataset.jsonl").stat().st_mtime, reverse=True)
    return dataset_dirs[0]


def resolve_dataset_path(
    dataset_arg: Optional[str], use_latest: bool = False, config: Optional[Dict] = None
) -> Optional[Path]:
    """Resolve dataset path from argument, --latest flag, or config."""
    if dataset_arg:
        path = Path(dataset_arg)
        if path.is_file():
            return path.parent
        return path

    if use_latest:
        return find_latest_dataset()

    if config:
        default_dataset = get_config_default(config, "defaults", "dataset")
        if default_dataset:
            return Path(default_dataset)

    return None


__all__ = [
    "_expand_env_vars",
    "load_config",
    "get_config_default",
    "find_latest_dataset",
    "resolve_dataset_path",
]
