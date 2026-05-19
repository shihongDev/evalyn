"""Configuration utilities for CLI: config loading, dataset path resolution."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

from ..constants import DEFAULT_CONFIG_PATHS

# Cache for project root to avoid repeated filesystem lookups
_project_root_cache: Path | None = None

# Cache for loaded config (read once per process - config doesn't change during a CLI session)
_UNSET = object()
_config_cache: Any = _UNSET


def find_project_root(cwd: Path | None = None) -> Path:
    """Find the project root directory.

    Searches upward from cwd for project markers:
    1. evalyn.yaml, .evalynrc (evalyn config files)
    2. pyproject.toml
    3. .git directory

    Args:
        cwd: Starting directory. If None, uses Path.cwd() and caches the result.

    Returns:
        Path to project root, or cwd if no markers found
    """
    global _project_root_cache
    # Only use cache when called without explicit cwd (production path)
    if cwd is None:
        if _project_root_cache is not None:
            return _project_root_cache
        start = Path.cwd()
    else:
        start = cwd

    markers = DEFAULT_CONFIG_PATHS + ["pyproject.toml", ".git"]

    current = start
    while current != current.parent:
        for marker in markers:
            if (current / marker).exists():
                if cwd is None:
                    _project_root_cache = current
                return current
        current = current.parent

    # No markers found, use start dir
    if cwd is None:
        _project_root_cache = start
    return start


def get_data_dir(subdir: str = "prod/datasets") -> Path:
    """Get the project's data directory (relative to project root).

    Args:
        subdir: Subdirectory within data/ (default: "prod/datasets")

    Returns:
        Path to data subdirectory
    """
    project_root = find_project_root()
    data_dir = project_root / "data" / subdir
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


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


def load_config() -> dict[str, Any]:
    """Load configuration from evalyn.yaml or .evalynrc if present.

    The result is cached for the lifetime of the process because the config
    file does not change during a single CLI session. This avoids redundant
    disk I/O and YAML parsing when multiple code-paths call load_config()
    (typically 3-5 times per command invocation).

    Raises:
        ValueError: If config file exists but cannot be parsed
    """
    global _config_cache
    if _config_cache is not _UNSET:
        return _config_cache

    for config_path in DEFAULT_CONFIG_PATHS:
        path = Path(config_path)
        if path.exists():
            try:
                import yaml  # Optional dependency

                with open(path, encoding="utf-8") as f:
                    config = yaml.safe_load(f) or {}
                    config = _expand_env_vars(config)
                    _config_cache = config
                    return config
            except ImportError:
                # YAML not available, try JSON
                with open(path, encoding="utf-8") as f:
                    config = json.load(f)
                    config = _expand_env_vars(config)
                    _config_cache = config
                    return config
            except Exception as e:
                raise ValueError(f"Failed to parse config file {path}: {e}") from e
    _config_cache = {}
    return {}


def clear_config_cache() -> None:
    """Reset the config cache so the next load_config() re-reads from disk.

    Useful in tests or when the config file is modified at runtime.
    """
    global _config_cache
    _config_cache = _UNSET


def get_config_default(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
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


def find_latest_dataset(data_dir: str = "data") -> Path | None:
    """Find the most recently modified dataset directory."""
    data_path = Path(data_dir)
    if not data_path.exists():
        return None

    # Find directories containing dataset.jsonl
    # Search in multiple locations:
    # 1. data/<name>/dataset.jsonl (legacy)
    # 2. data/prod/datasets/<name>/dataset.jsonl
    # 3. data/test/datasets/<name>/dataset.jsonl
    dataset_dirs = []

    # Legacy: direct subdirectories of data/
    for d in data_path.iterdir():
        if d.is_dir() and (d / "dataset.jsonl").exists():
            dataset_dirs.append(d)

    # New structure: data/prod/datasets/ and data/test/datasets/
    for env in ["prod", "test"]:
        datasets_path = data_path / env / "datasets"
        if datasets_path.exists():
            for d in datasets_path.iterdir():
                if d.is_dir() and (d / "dataset.jsonl").exists():
                    dataset_dirs.append(d)

    if not dataset_dirs:
        return None

    # Sort by modification time (most recent first)
    dataset_dirs.sort(key=lambda d: (d / "dataset.jsonl").stat().st_mtime, reverse=True)
    return dataset_dirs[0]


def resolve_dataset_path(
    dataset_arg: str | None, use_latest: bool = False, config: dict | None = None
) -> Path | None:
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


_MINIMAL_CONFIG = """\
# Evalyn Configuration
# See evalyn.yaml.example for all available options

# API Keys - only set what you need
api_keys:
  gemini: "your-gemini-api-key-here"  # Required for example agent
  # openai: "your-openai-key"         # Optional

llm:
  model: "gemini-2.5-flash-lite"

defaults:
  project: null
  version: null
"""


def create_evalyn_yaml(
    output_path: Path | None = None,
    force: bool = False,
) -> tuple[Path, bool]:
    """Create evalyn.yaml from the example template or a minimal fallback.

    Args:
        output_path: Where to write the config. Defaults to ./evalyn.yaml.
        force: Overwrite if the file already exists.

    Returns:
        (path, from_example) - the path written and whether the example was used.
        If the file already exists and force is False, returns (path, False) without writing.
    """
    import shutil

    if output_path is None:
        output_path = Path("evalyn.yaml")

    if output_path.exists() and not force:
        return output_path, False

    # Find the example file - check multiple locations
    example_paths = [
        Path("evalyn.yaml.example"),  # Current directory
        Path(__file__).parent.parent.parent / "evalyn.yaml.example",  # Project root
    ]

    for p in example_paths:
        if p.exists():
            shutil.copy(p, output_path)
            return output_path, True

    # Fallback: create minimal config if example not found
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(_MINIMAL_CONFIG)
    return output_path, False


__all__ = [
    "_expand_env_vars",
    "find_project_root",
    "get_data_dir",
    "load_config",
    "clear_config_cache",
    "get_config_default",
    "find_latest_dataset",
    "resolve_dataset_path",
    "create_evalyn_yaml",
    "_MINIMAL_CONFIG",
]
