from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Union
from uuid import uuid4

from .decorators import get_default_tracer
from .datasets import hash_inputs
from .metrics.registry import Metric
from .models import DatasetItem, EvalRun, FunctionCall, MetricResult, now_utc
from .storage.base import StorageBackend
from .tracing import EvalTracer


# Progress callback type: (current_item, total_items, current_metric, metric_type)
ProgressCallback = Callable[[int, int, str, str], None]


def _synthetic_call_from_item(item: DatasetItem) -> FunctionCall:
    """
    Create a synthetic FunctionCall from a DatasetItem.

    This allows evaluation to work on datasets that don't have
    corresponding traces in storage (e.g., manually created datasets,
    or datasets where the traces have been deleted).

    Handles both old format (expected) and new format (output).
    """
    # Get output - prefer 'output' field, fall back to 'expected' for old datasets
    output = item.output
    if output is None:
        output = item.expected

    return FunctionCall(
        id=item.metadata.get("call_id") or f"synthetic-{item.id}",
        function_name=item.metadata.get("function", "unknown"),
        inputs=item.input or item.inputs or {},
        output=output,
        error=item.metadata.get("error"),
        started_at=now_utc(),
        ended_at=now_utc(),
        duration_ms=item.metadata.get("duration_ms", 0.0),
        session_id=item.metadata.get("session_id"),
        trace=[],  # No trace events for synthetic calls
        metadata=item.metadata,
    )


def _get_item_output(item: DatasetItem):
    """Get output from item, handling both old and new formats."""
    if item.output is not None:
        return item.output
    return item.expected


class EvalRunner:
    """
    Executes a dataset against a target function, applies metrics, and stores the run.
    If `instrument=True`, the runner will wrap the target function with the tracer automatically.
    """

    def __init__(
        self,
        target_fn: Callable,
        metrics: Iterable[Metric],
        tracer: Optional[EvalTracer] = None,
        storage: Optional[StorageBackend] = None,
        dataset_name: str = "dataset",
        instrument: bool = True,
        cache_enabled: bool = True,
        progress_callback: Optional[ProgressCallback] = None,
    ):
        self.tracer = tracer or get_default_tracer()
        if storage:
            self.tracer.attach_storage(storage)
        self.dataset_name = dataset_name
        self.metrics: List[Metric] = list(metrics)
        already_wrapped = getattr(target_fn, "_evalyn_instrumented", False)
        self.target_fn = (
            target_fn
            if not instrument or already_wrapped
            else self.tracer.instrument(target_fn)
        )
        self.cache_enabled = cache_enabled
        self._cache: Dict[str, str] = {}  # cache key -> call id
        self._progress_callback = progress_callback

    def run_dataset(self, dataset: Iterable[DatasetItem], use_synthetic: bool = True) -> EvalRun:
        """
        Run evaluation on a dataset.

        Args:
            dataset: Iterable of DatasetItem to evaluate
            use_synthetic: If True, create synthetic FunctionCall when trace not found.
                          This allows evaluation on datasets without original traces.
        """
        metric_results: List[MetricResult] = []
        failures: List[str] = []

        # Convert to list for progress tracking
        items = list(dataset)
        total_items = len(items)
        total_evals = total_items * len(self.metrics)
        current_eval = 0

        for item_idx, item in enumerate(items):
            call = None

            # First, try to load call from metadata (for pre-built datasets)
            if isinstance(item.metadata, dict) and "call_id" in item.metadata and self.tracer.storage:
                call_id = item.metadata["call_id"]
                call = self.tracer.storage.get_call(call_id)

            # Second, check cache by input hash
            if call is None and self.cache_enabled:
                cache_key = hash_inputs(item.inputs)
                if cache_key in self._cache and self.tracer.storage:
                    cached_id = self._cache[cache_key]
                    cached_matches = [c for c in self.tracer.storage.list_calls(limit=1_000) if c.id == cached_id]
                    call = cached_matches[0] if cached_matches else None

            # Third, create synthetic call from item data if enabled
            if call is None and use_synthetic and _get_item_output(item) is not None:
                call = _synthetic_call_from_item(item)

            # Fourth, try to re-run the function (only if not using synthetic and target_fn is real)
            if call is None and not use_synthetic:
                try:
                    self.target_fn(**item.inputs)
                except Exception:
                    failures.append(item.id)

                call = self.tracer.last_call
                cache_key = hash_inputs(item.inputs) if self.cache_enabled else None
                if cache_key and call:
                    self._cache[cache_key] = call.id

            if call is None:
                if use_synthetic:
                    raise RuntimeError(
                        f"Cannot evaluate item {item.id}: no trace found and no output data. "
                        "Dataset items must have 'output' or 'expected' field for evaluation."
                    )
                else:
                    raise RuntimeError(
                        "No trace was captured for the last call. Ensure the function is instrumented with @eval."
                    )

            for metric in self.metrics:
                current_eval += 1
                if self._progress_callback:
                    self._progress_callback(
                        current_eval,
                        total_evals,
                        metric.spec.id,
                        metric.spec.type,
                    )
                metric_results.append(metric.evaluate(call, item))

        summary = self._summarize(metric_results, failures)
        run = EvalRun(
            id=str(uuid4()),
            dataset_name=self.dataset_name,
            created_at=now_utc(),
            metric_results=metric_results,
            metrics=[m.spec for m in self.metrics],
            summary=summary,
        )

        if self.tracer.storage:
            self.tracer.storage.store_eval_run(run)

        return run

    @staticmethod
    def _summarize(results: List[MetricResult], failures: List[str]) -> dict:
        by_metric: defaultdict[str, List[MetricResult]] = defaultdict(list)
        for res in results:
            by_metric[res.metric_id].append(res)

        summary = {"metrics": {}, "failed_items": failures}
        for metric_id, metric_results in by_metric.items():
            scores = [r.score for r in metric_results if r.score is not None]
            passes = [r.passed for r in metric_results if r.passed is not None]
            summary["metrics"][metric_id] = {
                "count": len(metric_results),
                "avg_score": (sum(scores) / len(scores)) if scores else None,
                "pass_rate": (sum(1 for p in passes if p) / len(passes)) if passes else None,
            }
        return summary


def save_eval_run_json(
    run: EvalRun,
    dataset_dir: Union[str, Path],
    *,
    runs_subdir: str = "eval_runs",
) -> Path:
    """
    Save an EvalRun as JSON in a dedicated folder.

    Structure:
        <dataset_dir>/eval_runs/<timestamp>_<run_id>/
            results.json    # Eval results
            report.html     # Analysis report (generated separately)

    Args:
        run: The EvalRun to save
        dataset_dir: Path to the dataset directory
        runs_subdir: Subdirectory name for eval runs (default: "eval_runs")

    Returns:
        Path to the run folder (not the JSON file)
    """
    dataset_dir = Path(dataset_dir)
    runs_dir = dataset_dir / runs_subdir

    # Create folder with timestamp for sorting
    timestamp = run.created_at.strftime("%Y%m%d-%H%M%S") if run.created_at else "unknown"
    folder_name = f"{timestamp}_{run.id[:8]}"
    run_folder = runs_dir / folder_name
    run_folder.mkdir(parents=True, exist_ok=True)

    # Save as results.json
    results_path = run_folder / "results.json"
    results_path.write_text(
        json.dumps(run.as_dict(), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8"
    )

    return run_folder


def load_eval_run_json(path: Union[str, Path]) -> EvalRun:
    """Load an EvalRun from a JSON file or folder.

    Args:
        path: Path to results.json file, or folder containing results.json
    """
    path = Path(path)
    # Handle both folder path and direct JSON file path
    if path.is_dir():
        path = path / "results.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return EvalRun.from_dict(data)


def list_eval_runs_json(dataset_dir: Union[str, Path], runs_subdir: str = "eval_runs") -> List[EvalRun]:
    """List all eval runs from folders in a dataset directory."""
    dataset_dir = Path(dataset_dir)
    runs_dir = dataset_dir / runs_subdir
    if not runs_dir.exists():
        return []

    runs = []
    # Look for folders containing results.json
    for run_folder in sorted(runs_dir.iterdir(), reverse=True):
        if run_folder.is_dir():
            results_file = run_folder / "results.json"
            if results_file.exists():
                try:
                    runs.append(load_eval_run_json(results_file))
                except Exception:
                    continue
    # Also support legacy flat JSON files
    for json_file in sorted(runs_dir.glob("*.json"), reverse=True):
        try:
            runs.append(load_eval_run_json(json_file))
        except Exception:
            continue
    return runs
