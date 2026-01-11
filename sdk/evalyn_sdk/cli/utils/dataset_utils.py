"""Dataset and metrics resolution utilities for CLI."""

from __future__ import annotations

import hashlib
import inspect
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ...trace.tracer import EvalTracer


def _extract_code_meta(tracer: "EvalTracer", fn: Callable[..., Any]) -> Optional[dict]:
    """
    Try to reuse cached code metadata from the tracer; fall back to inspect if not present.
    """
    meta = getattr(tracer, "_function_meta_cache", {}).get(id(fn))  # type: ignore[attr-defined]
    if meta:
        return meta
    try:
        source = inspect.getsource(fn)
        return {
            "module": getattr(fn, "__module__", None),
            "qualname": getattr(fn, "__qualname__", None),
            "doc": inspect.getdoc(fn),
            "signature": str(inspect.signature(fn)),
            "source": source,
            "source_hash": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "file_path": inspect.getsourcefile(fn),
        }
    except Exception:
        return None


def _resolve_dataset_and_metrics(
    dataset_arg: str, metrics_arg: Optional[str], metrics_all: bool = False
) -> tuple[Path, List[Path]]:
    """
    Resolve dataset file and metrics file paths.

    Args:
        dataset_arg: Path to dataset file or directory
        metrics_arg: Comma-separated paths to metrics files, or None for auto-detect
        metrics_all: If True, use all metrics files in the metrics/ folder

    Returns:
        Tuple of (dataset_file, list_of_metrics_paths)
    """
    dataset_path = Path(dataset_arg)

    # If dataset is a directory, look for dataset.jsonl inside
    if dataset_path.is_dir():
        dataset_dir = dataset_path
        dataset_file = dataset_dir / "dataset.jsonl"
        if not dataset_file.exists():
            dataset_file = dataset_dir / "dataset.json"
        if not dataset_file.exists():
            raise FileNotFoundError(
                f"No dataset.jsonl or dataset.json found in {dataset_dir}"
            )
    else:
        dataset_file = dataset_path
        dataset_dir = dataset_path.parent

    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")

    metrics_paths: List[Path] = []

    # Option 1: Use all metrics from metrics/ folder
    if metrics_all:
        metrics_dir = dataset_dir / "metrics"
        if metrics_dir.exists():
            for json_file in sorted(metrics_dir.glob("*.json")):
                metrics_paths.append(json_file)
        if not metrics_paths:
            raise FileNotFoundError(f"No metrics files found in {metrics_dir}")
        print(f"Using all metrics files: {len(metrics_paths)} files from {metrics_dir}")
        return dataset_file, metrics_paths

    # Option 2: Explicit metrics argument (supports comma-separated paths)
    if metrics_arg:
        for path_str in metrics_arg.split(","):
            path_str = path_str.strip()
            if path_str:
                metrics_path = Path(path_str)
                if not metrics_path.exists():
                    raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
                metrics_paths.append(metrics_path)
        if metrics_paths:
            return dataset_file, metrics_paths

    # Option 3: Auto-detect from meta.json
    meta_path = dataset_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"No --metrics specified and no meta.json found in {dataset_dir}.\n"
            "Either specify --metrics explicitly or run 'evalyn suggest-metrics' first."
        )
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise ValueError(f"Failed to parse meta.json: {e}")

    active_set = meta.get("active_metric_set")
    if not active_set:
        raise ValueError(
            f"No active_metric_set in meta.json. Run 'evalyn suggest-metrics' to select metrics."
        )

    # Find the metrics file from metric_sets
    metric_sets = meta.get("metric_sets", [])
    matching = [m for m in metric_sets if m.get("name") == active_set]
    if not matching:
        raise ValueError(
            f"Metric set '{active_set}' not found in meta.json metric_sets."
        )

    metrics_rel = matching[0].get("file")
    if not metrics_rel:
        raise ValueError(f"No file path for metric set '{active_set}' in meta.json.")

    metrics_path = dataset_dir / metrics_rel
    print(f"Auto-detected metrics: {metrics_path}")

    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    return dataset_file, [metrics_path]


def _dataset_has_reference(dataset_path: Optional[Path]) -> bool:
    """
    Check if a dataset has SEPARATE reference/golden-standard values for comparison.

    Reference-based metrics (ROUGE, BLEU, token_overlap) need TWO text values:
    1. output - what the model produced
    2. reference - what a human says is the correct answer (golden standard)

    This function checks for explicit reference fields that are SEPARATE from output:
    - human_label.reference (new format)
    - metadata.reference or metadata.golden (explicit reference)

    NOTE: The old 'expected' field often contains the model output (not a separate reference),
    so we don't count it unless there's also an 'output' field (indicating expected is actually
    the golden standard).

    Returns True if at least one item has a separate reference value, False otherwise.
    """
    from ...datasets import load_dataset

    if not dataset_path:
        return False

    # Find the actual dataset file
    if dataset_path.is_file():
        dataset_file = dataset_path
    else:
        if (dataset_path / "dataset.jsonl").exists():
            dataset_file = dataset_path / "dataset.jsonl"
        elif (dataset_path / "dataset.json").exists():
            dataset_file = dataset_path / "dataset.json"
        else:
            return False

    try:
        items = load_dataset(str(dataset_file))
        for item in items:
            # Check for human_label with reference (this is the clear signal)
            if hasattr(item, "human_label") and item.human_label:
                if isinstance(item.human_label, dict) and item.human_label.get(
                    "reference"
                ):
                    return True

            # Check metadata for explicit reference/golden fields
            if hasattr(item, "metadata") and item.metadata:
                if (
                    item.metadata.get("reference")
                    or item.metadata.get("golden")
                    or item.metadata.get("golden_answer")
                ):
                    return True

            # If BOTH output AND expected exist AND they are DIFFERENT, then expected is the golden standard
            has_output = hasattr(item, "output") and item.output is not None
            has_expected = hasattr(item, "expected") and item.expected is not None
            if has_output and has_expected:
                # Only count as reference if they're actually different values
                # (if same, expected is just a copy of output for backward compatibility)
                if item.output != item.expected:
                    return True

        return False
    except Exception:
        return False


class ProgressBar:
    """Progress bar with ETA for evaluation."""

    def __init__(self, total: int, width: int = 40):
        self.total = total
        self.width = width
        self._current = 0
        self._current_metric = ""
        self._current_type = ""
        self._start_time = time.time()
        self._last_render_time = 0
        self._errors: List[str] = []

    def update(self, current: int, total: int, metric: str, metric_type: str) -> None:
        self._current = current
        self._current_metric = metric
        self._current_type = metric_type
        self._render()

    def add_error(self, error: str) -> None:
        """Record an error to display."""
        self._errors.append(error)

    def _format_eta(self, seconds: float) -> str:
        """Format seconds as human-readable time."""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m"

    def _render(self) -> None:
        pct = self._current / self.total if self.total > 0 else 0
        filled = int(self.width * pct)
        bar = "=" * filled + "-" * (self.width - filled)
        type_label = "[obj]" if self._current_type == "objective" else "[llm]"

        # Calculate ETA
        elapsed = time.time() - self._start_time
        if self._current > 0 and pct < 1.0:
            rate = self._current / elapsed
            remaining = (self.total - self._current) / rate if rate > 0 else 0
            eta_str = f"ETA: {self._format_eta(remaining)}"
        elif pct >= 1.0:
            eta_str = f"Done in {self._format_eta(elapsed)}"
        else:
            eta_str = "ETA: --"

        line = f"\r[{bar}] {self._current}/{self.total} {type_label} {self._current_metric[:15]:<15} {eta_str}"
        sys.stderr.write(line)
        sys.stderr.flush()

    def finish(self) -> None:
        sys.stderr.write("\r" + " " * 100 + "\r")
        sys.stderr.flush()
        # Show errors if any
        if self._errors:
            sys.stderr.write(f"  ⚠ {len(self._errors)} error(s) during evaluation\n")
            for error in self._errors[:3]:  # Show first 3
                sys.stderr.write(f"    - {error}\n")
            if len(self._errors) > 3:
                sys.stderr.write(f"    ... and {len(self._errors) - 3} more\n")
            sys.stderr.flush()


__all__ = [
    "_extract_code_meta",
    "_resolve_dataset_and_metrics",
    "_dataset_has_reference",
    "ProgressBar",
]
