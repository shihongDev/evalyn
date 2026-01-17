"""DatasetResolver: unified dataset path resolution and loading."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import load_config, get_config_default


@dataclass
class DatasetInfo:
    """Information about a resolved dataset."""

    path: Path
    dataset_file: Path
    meta: Dict[str, Any]
    item_count: int

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def project(self) -> Optional[str]:
        return self.meta.get("project")

    @property
    def version(self) -> Optional[str]:
        return self.meta.get("version")

    @property
    def metrics_dir(self) -> Path:
        return self.path / "metrics"

    @property
    def eval_runs_dir(self) -> Path:
        return self.path / "eval_runs"

    def get_active_metrics_file(self) -> Optional[Path]:
        """Get the active metrics file from meta.json."""
        active_set = self.meta.get("active_metric_set")
        if not active_set:
            return None
        metric_sets = self.meta.get("metric_sets", [])
        for ms in metric_sets:
            if ms.get("name") == active_set:
                rel_path = ms.get("file")
                if rel_path:
                    return self.path / rel_path
        return None

    def list_metrics_files(self) -> List[Path]:
        """List all metrics JSON files."""
        if not self.metrics_dir.exists():
            return []
        return sorted(self.metrics_dir.glob("*.json"))

    def list_eval_runs(self) -> List[Path]:
        """List all evaluation run directories."""
        if not self.eval_runs_dir.exists():
            return []
        return sorted(
            [d for d in self.eval_runs_dir.iterdir() if d.is_dir()],
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )


class DatasetResolver:
    """Resolve and load dataset information from paths."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or load_config()

    def resolve(
        self,
        dataset_arg: Optional[str] = None,
        use_latest: bool = False,
        require: bool = True,
    ) -> Optional[DatasetInfo]:
        """Resolve dataset path and load info.

        Args:
            dataset_arg: Path to dataset file or directory
            use_latest: Use most recently modified dataset
            require: If True, exit with error if not found

        Returns:
            DatasetInfo or None if not found and require=False
        """
        path = self._resolve_path(dataset_arg, use_latest)

        if not path:
            if require:
                self._print_available_and_exit()
            return None

        if not path.exists():
            if require:
                print(f"Error: Dataset path does not exist: {path}", file=sys.stderr)
                sys.exit(1)
            return None

        return self._load_info(path)

    def _resolve_path(
        self, dataset_arg: Optional[str], use_latest: bool
    ) -> Optional[Path]:
        """Resolve to a Path without loading."""
        if dataset_arg:
            path = Path(dataset_arg)
            if path.is_file():
                return path.parent
            return path

        if use_latest:
            return self.find_latest()

        default = get_config_default(self.config, "defaults", "dataset")
        if default:
            return Path(default)

        return None

    def _load_info(self, path: Path) -> DatasetInfo:
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
            print(f"Error: No dataset.jsonl found in {path}", file=sys.stderr)
            sys.exit(1)

        # Count items efficiently
        with open(dataset_file, encoding="utf-8") as f:
            item_count = sum(1 for _ in f)

        # Load meta if available
        meta_file = path / "meta.json"
        meta = (
            json.loads(meta_file.read_text(encoding="utf-8"))
            if meta_file.exists()
            else {}
        )

        return DatasetInfo(
            path=path,
            dataset_file=dataset_file,
            meta=meta,
            item_count=item_count,
        )

    def find_latest(self, data_dir: str = "data") -> Optional[Path]:
        """Find most recently modified dataset directory."""
        data_path = Path(data_dir)
        if not data_path.exists():
            return None

        dataset_dirs = [
            d
            for d in data_path.iterdir()
            if d.is_dir() and (d / "dataset.jsonl").exists()
        ]

        if not dataset_dirs:
            return None

        dataset_dirs.sort(
            key=lambda d: (d / "dataset.jsonl").stat().st_mtime, reverse=True
        )
        return dataset_dirs[0]

    def list_available(self, data_dir: str = "data", limit: int = 10) -> List[Path]:
        """List available dataset directories."""
        data_path = Path(data_dir)
        if not data_path.exists():
            return []

        datasets = [
            d
            for d in data_path.iterdir()
            if d.is_dir() and (d / "dataset.jsonl").exists()
        ]
        datasets.sort(key=lambda d: d.stat().st_mtime, reverse=True)
        return datasets[:limit]

    def _print_available_and_exit(self, data_dir: str = "data") -> None:
        """Print available datasets and exit."""
        print("No dataset specified. Available datasets:", file=sys.stderr)
        datasets = self.list_available(data_dir)
        if datasets:
            for d in datasets:
                mtime = datetime.fromtimestamp(d.stat().st_mtime).strftime(
                    "%Y-%m-%d %H:%M"
                )
                print(f"  {d.name:<40} (modified: {mtime})", file=sys.stderr)
            total = (
                len(list(Path(data_dir).iterdir())) if Path(data_dir).exists() else 0
            )
            if total > len(datasets):
                print(f"  ... and {total - len(datasets)} more", file=sys.stderr)
        print(
            "\nUse: --dataset <path> or --latest",
            file=sys.stderr,
        )
        sys.exit(1)


# Convenience function for simple cases
def get_dataset(
    dataset_arg: Optional[str] = None,
    use_latest: bool = False,
    require: bool = True,
) -> Optional[DatasetInfo]:
    """Resolve and load dataset info (convenience function)."""
    return DatasetResolver().resolve(dataset_arg, use_latest, require)


__all__ = ["DatasetResolver", "DatasetInfo", "get_dataset"]
