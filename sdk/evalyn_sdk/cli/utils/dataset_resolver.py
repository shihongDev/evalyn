"""Dataset resolution: unified dataset path resolution and loading."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import find_latest_dataset as _config_find_latest_dataset
from .config import get_config_default, load_config
from .errors import fatal_error


@dataclass
class DatasetInfo:
    """Information about a resolved dataset."""

    path: Path
    dataset_file: Path
    meta: dict[str, Any]
    item_count: int

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def project(self) -> str | None:
        return self.meta.get("project")

    @property
    def version(self) -> str | None:
        return self.meta.get("version")

    @property
    def metrics_dir(self) -> Path:
        return self.path / "metrics"

    @property
    def eval_runs_dir(self) -> Path:
        return self.path / "eval_runs"

    def list_metrics_files(self) -> list[Path]:
        """List all metrics JSON files."""
        if not self.metrics_dir.exists():
            return []
        return sorted(self.metrics_dir.glob("*.json"))

    def list_eval_runs(self) -> list[Path]:
        """List all evaluation run directories."""
        if not self.eval_runs_dir.exists():
            return []
        return sorted(
            [d for d in self.eval_runs_dir.iterdir() if d.is_dir()],
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )


def _resolve_path(
    dataset_arg: str | None,
    use_latest: bool,
    config: dict[str, Any] | None = None,
) -> Path | None:
    """Resolve dataset argument to a Path without loading."""
    if dataset_arg:
        path = Path(dataset_arg)
        if path.is_file():
            return path.parent
        return path

    if use_latest:
        return find_latest_dataset()

    cfg = config or load_config()
    default = get_config_default(cfg, "defaults", "dataset")
    if default:
        return Path(default)

    return None


def _load_info(path: Path) -> DatasetInfo:
    """Load DatasetInfo from a resolved path."""
    # Normalize path: if file, use parent directory
    if path.is_file():
        dataset_file = path
        path = path.parent
    else:
        # Try dataset.jsonl first, then dataset.json
        dataset_file = path / "dataset.jsonl"
        if not dataset_file.exists():
            dataset_file = path / "dataset.json"

    if not dataset_file.exists():
        fatal_error(f"No dataset.jsonl found in {path}")

    # Count items efficiently
    with open(dataset_file, encoding="utf-8") as f:
        item_count = sum(1 for _ in f)

    # Load meta if available
    meta_file = path / "meta.json"
    meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}

    return DatasetInfo(
        path=path,
        dataset_file=dataset_file,
        meta=meta,
        item_count=item_count,
    )


def find_latest_dataset(data_dir: str = "data") -> Path | None:
    """Find most recently modified dataset directory.

    Delegates to config.find_latest_dataset which also checks
    data/prod/datasets/ and data/test/datasets/.
    """
    return _config_find_latest_dataset(data_dir)


def list_datasets(data_dir: str = "data", limit: int = 10) -> list[Path]:
    """List available dataset directories.

    Searches three locations (matching find_latest_dataset behavior):
    1. data/<name>/dataset.jsonl (direct children)
    2. data/prod/datasets/<name>/dataset.jsonl
    3. data/test/datasets/<name>/dataset.jsonl
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        return []

    search_dirs = [data_path]
    for sub in ("prod/datasets", "test/datasets"):
        candidate = data_path / sub
        if candidate.exists():
            search_dirs.append(candidate)

    seen: set = set()
    datasets: list[Path] = []
    for search_dir in search_dirs:
        for d in search_dir.iterdir():
            if d.is_dir() and (d / "dataset.jsonl").exists() and d not in seen:
                seen.add(d)
                datasets.append(d)

    datasets.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return datasets[:limit]


def _print_available_and_exit(data_dir: str = "data") -> None:
    """Print available datasets and exit."""
    print("No dataset specified. Available datasets:", file=sys.stderr)
    datasets = list_datasets(data_dir)
    if datasets:
        for d in datasets:
            mtime = datetime.fromtimestamp(d.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            print(f"  {d.name:<40} (modified: {mtime})", file=sys.stderr)
        total = len(list(Path(data_dir).iterdir())) if Path(data_dir).exists() else 0
        if total > len(datasets):
            print(f"  ... and {total - len(datasets)} more", file=sys.stderr)
    fatal_error("No dataset specified", "Use --dataset <path> or --latest")


def get_dataset(
    dataset_arg: str | None = None,
    use_latest: bool = False,
    require: bool = True,
    config: dict[str, Any] | None = None,
) -> DatasetInfo | None:
    """Resolve and load dataset info.

    Args:
        dataset_arg: Path to dataset file or directory
        use_latest: Use most recently modified dataset
        require: If True, exit with error if not found
        config: Optional config dict (loaded if not provided)

    Returns:
        DatasetInfo or None if not found and require=False
    """
    path = _resolve_path(dataset_arg, use_latest, config)

    if not path:
        if require:
            _print_available_and_exit()
        return None

    if not path.exists():
        if require:
            fatal_error(f"Dataset path does not exist: {path}")
        return None

    return _load_info(path)


__all__ = ["DatasetInfo", "get_dataset", "find_latest_dataset", "list_datasets"]
