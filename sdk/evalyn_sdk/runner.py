from __future__ import annotations

import json
import os
import tempfile
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

    Supports checkpointing for long-running evaluations:
    - checkpoint_path: Path to save progress (default: None, no checkpointing)
    - checkpoint_interval: Save checkpoint every N items (default: 5)
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
        checkpoint_path: Optional[Union[str, Path]] = None,
        checkpoint_interval: int = 5,
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
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.checkpoint_interval = checkpoint_interval

    def _load_checkpoint(self) -> Dict:
        """Load checkpoint if it exists. Returns dict with 'results' and 'completed_items'."""
        if not self.checkpoint_path or not self.checkpoint_path.exists():
            return {"results": [], "completed_items": set(), "run_id": str(uuid4())}

        try:
            with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Convert results back to MetricResult objects
            results = [MetricResult.from_dict(r) for r in data.get("results", [])]
            completed = set(data.get("completed_items", []))
            run_id = data.get("run_id", str(uuid4()))
            return {"results": results, "completed_items": completed, "run_id": run_id}
        except Exception:
            # If checkpoint is corrupted, start fresh
            return {"results": [], "completed_items": set(), "run_id": str(uuid4())}

    def _save_checkpoint(
        self, results: List[MetricResult], completed_items: set, run_id: str
    ) -> bool:
        """Save checkpoint atomically. Returns True on success."""
        if not self.checkpoint_path:
            return False

        try:
            # Prepare checkpoint data
            data = {
                "run_id": run_id,
                "completed_items": list(completed_items),
                "results": [r.as_dict() for r in results],
                "saved_at": now_utc().isoformat(),
            }

            # Write to temp file, then atomic rename
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            temp_fd, temp_path = tempfile.mkstemp(
                dir=self.checkpoint_path.parent,
                prefix=".checkpoint_",
                suffix=".tmp",
            )
            try:
                with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False, default=str)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(temp_path, self.checkpoint_path)
                return True
            except Exception:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                raise
        except Exception:
            return False

    def _cleanup_checkpoint(self) -> None:
        """Remove checkpoint file after successful completion."""
        if self.checkpoint_path and self.checkpoint_path.exists():
            try:
                self.checkpoint_path.unlink()
            except Exception:
                pass

    def run_dataset(
        self, dataset: Iterable[DatasetItem], use_synthetic: bool = True
    ) -> EvalRun:
        """
        Run evaluation on a dataset.

        Args:
            dataset: Iterable of DatasetItem to evaluate
            use_synthetic: If True, create synthetic FunctionCall when trace not found.
                          This allows evaluation on datasets without original traces.

        Supports checkpointing: if checkpoint_path is set, progress is saved every
        checkpoint_interval items. On resume, already-evaluated items are skipped.
        """
        # Load checkpoint if exists
        checkpoint = self._load_checkpoint()
        metric_results: List[MetricResult] = checkpoint["results"]
        completed_items: set = checkpoint["completed_items"]
        run_id = checkpoint["run_id"]
        failures: List[str] = []

        # Convert to list for progress tracking
        items = list(dataset)
        total_items = len(items)

        # Filter out already completed items
        pending_items = [
            (i, item) for i, item in enumerate(items) if item.id not in completed_items
        ]
        resumed = len(completed_items) > 0

        if resumed and self._progress_callback:
            # Log that we're resuming
            pass

        total_evals = len(pending_items) * len(self.metrics)
        current_eval = 0
        items_since_checkpoint = 0

        for item_idx, item in pending_items:
            call = None

            # First, try to load call from metadata (for pre-built datasets)
            if (
                isinstance(item.metadata, dict)
                and "call_id" in item.metadata
                and self.tracer.storage
            ):
                call_id = item.metadata["call_id"]
                call = self.tracer.storage.get_call(call_id)

            # Second, check cache by input hash
            if call is None and self.cache_enabled:
                cache_key = hash_inputs(item.inputs)
                if cache_key in self._cache and self.tracer.storage:
                    cached_id = self._cache[cache_key]
                    cached_matches = [
                        c
                        for c in self.tracer.storage.list_calls(limit=1_000)
                        if c.id == cached_id
                    ]
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
                # Per-metric error isolation - don't let one metric failure kill the run
                try:
                    metric_results.append(metric.evaluate(call, item))
                except Exception as e:
                    # Record the error as a failed result
                    metric_results.append(
                        MetricResult(
                            metric_id=metric.spec.id,
                            call_id=call.id,
                            score=None,
                            passed=False,
                            details={"error": str(e), "error_type": type(e).__name__},
                        )
                    )

            # Mark item as completed and save checkpoint
            completed_items.add(item.id)
            items_since_checkpoint += 1

            if (
                self.checkpoint_path
                and items_since_checkpoint >= self.checkpoint_interval
            ):
                self._save_checkpoint(metric_results, completed_items, run_id)
                items_since_checkpoint = 0

        # Save final checkpoint before creating run
        if self.checkpoint_path and items_since_checkpoint > 0:
            self._save_checkpoint(metric_results, completed_items, run_id)

        summary = self._summarize(metric_results, failures)
        run = EvalRun(
            id=run_id,  # Use consistent run_id from checkpoint
            dataset_name=self.dataset_name,
            created_at=now_utc(),
            metric_results=metric_results,
            metrics=[m.spec for m in self.metrics],
            summary=summary,
        )

        if self.tracer.storage:
            self.tracer.storage.store_eval_run(run)

        # Clean up checkpoint on successful completion
        self._cleanup_checkpoint()

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
                "pass_rate": (sum(1 for p in passes if p) / len(passes))
                if passes
                else None,
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
    timestamp = (
        run.created_at.strftime("%Y%m%d-%H%M%S") if run.created_at else "unknown"
    )
    folder_name = f"{timestamp}_{run.id[:8]}"
    run_folder = runs_dir / folder_name
    run_folder.mkdir(parents=True, exist_ok=True)

    # Save as results.json
    results_path = run_folder / "results.json"
    results_path.write_text(
        json.dumps(run.as_dict(), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
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


def list_eval_runs_json(
    dataset_dir: Union[str, Path], runs_subdir: str = "eval_runs"
) -> List[EvalRun]:
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
