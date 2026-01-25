from __future__ import annotations

import json
import logging
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
from uuid import uuid4

from .decorators import get_default_tracer
from .datasets import hash_inputs
from .eval_units import EvalUnitBuilder, get_default_builders, get_builders_for_types
from .eval_units.views import project_unit
from .execution import ProgressCallback, create_strategy
from .models import EvalUnit, EvalView, Metric
from .models import DatasetItem, EvalRun, FunctionCall, MetricResult, now_utc
from .storage.base import StorageBackend
from .trace.tracer import EvalTracer

logger = logging.getLogger(__name__)


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
        max_workers: int = 1,
        unit_builders: Optional[List[EvalUnitBuilder]] = None,
        unit_types: Optional[List[str]] = None,
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
        self.max_workers = max(1, min(max_workers, 16))  # Clamp 1-16

        # Unit-based evaluation configuration
        # Priority: explicit builders > unit_types > default (OutcomeBuilder)
        if unit_builders is not None:
            self.unit_builders = unit_builders
        elif unit_types is not None:
            self.unit_builders = get_builders_for_types(unit_types)
        else:
            self.unit_builders = get_default_builders()

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

    def _discover_units(self, call: FunctionCall) -> List[EvalUnit]:
        """Discover all evaluatable units from a call using configured builders."""
        units = []
        for builder in self.unit_builders:
            units.extend(builder.discover(call))
        return units

    def _evaluate_metric(
        self, metric: Metric, call: FunctionCall, item: DatasetItem
    ) -> MetricResult:
        """Evaluate a single metric. Thread-safe."""
        try:
            return metric.evaluate(call, item)
        except Exception as e:
            return MetricResult(
                metric_id=metric.spec.id,
                item_id=item.id,
                call_id=call.id,
                score=None,
                passed=False,
                details={"error": str(e), "error_type": type(e).__name__},
            )

    def _evaluate_metric_unit(
        self, metric: Metric, unit: EvalUnit, view: EvalView, item: DatasetItem
    ) -> MetricResult:
        """Evaluate a metric against a unit view. Thread-safe."""
        try:
            result = metric.evaluate_unit(view, item)
            # Enrich result with unit info
            result.unit_id = unit.id
            result.unit_type = unit.unit_type
            result.span_ids = unit.span_ids
            return result
        except Exception as e:
            return MetricResult(
                metric_id=metric.spec.id,
                item_id=item.id,
                call_id=unit.call_id,
                score=None,
                passed=False,
                details={"error": str(e), "error_type": type(e).__name__},
                unit_id=unit.id,
                unit_type=unit.unit_type,
                span_ids=unit.span_ids,
            )

    def _prepare_item_call(
        self, item: DatasetItem, use_synthetic: bool, failures: List[str]
    ) -> Optional[FunctionCall]:
        """Prepare FunctionCall for an item. Returns None if cannot be resolved."""
        storage = self.tracer.storage

        # Try to load call from metadata (for pre-built datasets)
        if isinstance(item.metadata, dict) and "call_id" in item.metadata and storage:
            call = storage.get_call(item.metadata["call_id"])
            if call:
                return call

        # Check cache by input hash
        if self.cache_enabled and storage:
            cache_key = hash_inputs(item.inputs)
            cached_id = self._cache.get(cache_key)
            if cached_id:
                call = storage.get_call(cached_id)
                if call:
                    return call

        # Create synthetic call from item data if enabled
        if use_synthetic and _get_item_output(item) is not None:
            return _synthetic_call_from_item(item)

        # Try to re-run the function (only if not using synthetic)
        if not use_synthetic:
            try:
                self.target_fn(**item.inputs)
            except Exception:
                failures.append(item.id)

            call = self.tracer.last_call
            if call and self.cache_enabled:
                self._cache[hash_inputs(item.inputs)] = call.id
            return call

        return None

    def _is_outcome_only(self) -> bool:
        """Check if using only OutcomeBuilder (default backward-compatible mode)."""
        from .eval_units import OutcomeBuilder

        return (
            len(self.unit_builders) == 1
            and isinstance(self.unit_builders[0], OutcomeBuilder)
        )

    def _run_unit_evaluation(
        self,
        prepared: List[Tuple[DatasetItem, FunctionCall]],
        run_id: str,
        completed_items: set,
    ) -> List[MetricResult]:
        """Run unit-based evaluation (for non-default unit types)."""
        results: List[MetricResult] = []

        # Collect all unit types we're evaluating
        unit_types_active = {b.unit_type for b in self.unit_builders}

        for item, call in prepared:
            # Discover units from this call
            units = self._discover_units(call)

            for unit in units:
                # Project unit to view
                view = project_unit(unit, call)

                # Find metrics that support this unit type
                for metric in self.metrics:
                    if metric.supports_unit_type(unit.unit_type):
                        result = self._evaluate_metric_unit(metric, unit, view, item)
                        results.append(result)

            completed_items.add(item.id)

            # Progress callback
            if self._progress_callback:
                self._progress_callback(
                    len(completed_items),
                    len(prepared),
                    "unit_eval",
                    "unit",
                )

        return results

    def run_dataset(
        self, dataset: Iterable[DatasetItem], use_synthetic: bool = True
    ) -> EvalRun:
        """
        Run evaluation on a dataset.

        Args:
            dataset: Iterable of DatasetItem to evaluate
            use_synthetic: If True, create synthetic FunctionCall when trace not found.
                          This allows evaluation on datasets without original traces.

        Supports checkpointing and parallel execution (max_workers > 1).
        When unit_types/unit_builders are configured (non-default), runs
        unit-based evaluation discovering spans from trace structure.
        """
        # Load checkpoint if exists
        checkpoint = self._load_checkpoint()
        metric_results: List[MetricResult] = checkpoint["results"]
        completed_items: set = checkpoint["completed_items"]
        run_id = checkpoint["run_id"]
        failures: List[str] = []

        # Convert to list for progress tracking
        items = list(dataset)

        # Filter out already completed items
        pending_items = [
            (i, item) for i, item in enumerate(items) if item.id not in completed_items
        ]

        # Prepare all items with their FunctionCalls first (sequential)
        prepared: List[Tuple[DatasetItem, FunctionCall]] = []
        for item_idx, item in pending_items:
            call = self._prepare_item_call(item, use_synthetic, failures)
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
            prepared.append((item, call))

        # Choose evaluation mode based on unit_builders configuration
        if self._is_outcome_only():
            # Default mode: use existing strategy-based execution (backward compat)
            checkpoint_fn = self._save_checkpoint if self.checkpoint_path else None
            strategy = create_strategy(
                max_workers=self.max_workers,
                evaluate_fn=self._evaluate_metric,
                checkpoint_fn=checkpoint_fn,
                checkpoint_interval=self.checkpoint_interval,
            )

            new_results = strategy.execute(
                prepared=prepared,
                metrics=self.metrics,
                progress_callback=self._progress_callback,
                run_id=run_id,
                completed_items=completed_items,
            )
        else:
            # Unit-based mode: discover units and evaluate per-unit
            new_results = self._run_unit_evaluation(
                prepared=prepared,
                run_id=run_id,
                completed_items=completed_items,
            )

        metric_results.extend(new_results)

        summary = self._summarize(metric_results, failures)
        usage_summary = self._compute_usage_summary(metric_results)
        run = EvalRun(
            id=run_id,  # Use consistent run_id from checkpoint
            dataset_name=self.dataset_name,
            created_at=now_utc(),
            metric_results=metric_results,
            metrics=[m.spec for m in self.metrics],
            summary=summary,
            usage_summary=usage_summary,
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

    @staticmethod
    def _compute_usage_summary(results: List[MetricResult]) -> dict:
        """Compute token usage and cost summary from metric results."""
        from .trace.instrumentation.providers._shared import (
            calculate_cost,
            is_model_pricing_known,
        )

        total_input_tokens = 0
        total_output_tokens = 0
        models_used = set()
        has_unknown_pricing = False

        # Track costs by model and metric
        cost_by_model: Dict[str, float] = defaultdict(float)
        cost_by_metric: Dict[str, float] = defaultdict(float)
        tokens_by_metric: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"input": 0, "output": 0}
        )

        for r in results:
            input_tok = r.input_tokens or 0
            output_tok = r.output_tokens or 0
            total_input_tokens += input_tok
            total_output_tokens += output_tok

            if r.model:
                models_used.add(r.model)
                if not is_model_pricing_known(r.model):
                    has_unknown_pricing = True

                # Calculate cost for this result
                cost = calculate_cost(r.model, input_tok, output_tok)
                cost_by_model[r.model] += cost
                cost_by_metric[r.metric_id] += cost

            # Track tokens by metric
            tokens_by_metric[r.metric_id]["input"] += input_tok
            tokens_by_metric[r.metric_id]["output"] += output_tok

        total_tokens = total_input_tokens + total_output_tokens
        total_cost_usd = sum(cost_by_model.values())

        return {
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_tokens,
            "models_used": sorted(models_used),
            "total_cost_usd": total_cost_usd,
            "cost_by_model": dict(cost_by_model),
            "cost_by_metric": dict(cost_by_metric),
            "tokens_by_metric": dict(tokens_by_metric),
            "has_unknown_pricing": has_unknown_pricing,
        }


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
