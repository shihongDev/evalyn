from __future__ import annotations

import json
import logging
import os
import tempfile
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path
from uuid import uuid4

from ..datasets import hash_inputs
from ..decorators import get_default_tracer
from ..models import (
    DatasetItem,
    EvalRun,
    EvalUnit,
    EvalView,
    FunctionCall,
    Metric,
    MetricResult,
    now_utc,
)
from ..storage.base import StorageBackend
from ..trace.tracer import EvalTracer
from .execution import ProgressCallback, ResultHook, create_strategy
from .guards import BudgetExceededError
from .units import EvalUnitBuilder, get_builders_for_types, get_default_builders
from .units.views import project_unit

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
    return item.output if item.output is not None else item.expected


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
        tracer: EvalTracer | None = None,
        storage: StorageBackend | None = None,
        dataset_name: str = "dataset",
        instrument: bool = True,
        cache_enabled: bool = True,
        progress_callback: ProgressCallback | None = None,
        checkpoint_path: str | Path | None = None,
        checkpoint_interval: int = 5,
        max_workers: int = 1,
        unit_builders: list[EvalUnitBuilder] | None = None,
        unit_types: list[str] | None = None,
        result_hook: ResultHook | None = None,
    ):
        self.tracer = tracer or get_default_tracer()
        if storage:
            self.tracer.attach_storage(storage)
        self.dataset_name = dataset_name
        self.metrics: list[Metric] = list(metrics)
        already_wrapped = getattr(target_fn, "_evalyn_instrumented", False)
        self.target_fn = (
            target_fn if not instrument or already_wrapped else self.tracer.instrument(target_fn)
        )
        self.cache_enabled = cache_enabled
        self._cache: dict[str, str] = {}  # cache key -> call id
        self._progress_callback = progress_callback
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.checkpoint_interval = checkpoint_interval
        self.max_workers = max(1, min(max_workers, 16))  # Clamp 1-16
        self._checkpoint_count: int = 0  # results already written to checkpoint JSONL
        self.result_hook = result_hook

        # Unit-based evaluation configuration
        # Priority: explicit builders > unit_types > default (OutcomeBuilder)
        if unit_builders is not None:
            self.unit_builders = unit_builders
        elif unit_types is not None:
            self.unit_builders = get_builders_for_types(unit_types)
        else:
            self.unit_builders = get_default_builders()

    @property
    def _checkpoint_results_path(self) -> Path | None:
        """JSONL file that stores metric results alongside the header."""
        if self.checkpoint_path is None:
            return None
        return self.checkpoint_path.with_suffix(".jsonl")

    def _load_checkpoint(self) -> dict:
        """Load checkpoint if it exists. Returns dict with 'results' and 'completed_items'.

        Supports two formats:
        - New: header JSON (.json) + results JSONL (.jsonl) - O(n) total writes
        - Legacy: single JSON file with embedded results - auto-migrated on resume
        """
        if not self.checkpoint_path or not self.checkpoint_path.exists():
            return {"results": [], "completed_items": set(), "run_id": str(uuid4())}

        try:
            with open(self.checkpoint_path, encoding="utf-8") as f:
                data = json.load(f)

            # Check for JSONL results file (new format)
            results_path = self._checkpoint_results_path
            if results_path and results_path.exists():
                raw_results = []
                with open(results_path, encoding="utf-8") as rf:
                    for line in rf:
                        line = line.strip()
                        if line:
                            raw_results.append(json.loads(line))
                results = [MetricResult.from_dict(r) for r in raw_results]
            else:
                # Legacy: results embedded in the header JSON
                raw_results = data.get("results", [])
                results = [MetricResult.from_dict(r) for r in raw_results]

            completed = set(data.get("completed_items", []))
            run_id = data.get("run_id", str(uuid4()))
            self._checkpoint_count = len(results)
            return {"results": results, "completed_items": completed, "run_id": run_id}
        except Exception as e:
            logger.warning("Checkpoint corrupted, starting fresh: %s", e)
            return {"results": [], "completed_items": set(), "run_id": str(uuid4())}

    def _save_checkpoint(
        self, results: list[MetricResult], completed_items: set, run_id: str
    ) -> bool:
        """Save checkpoint incrementally. Returns True on success.

        Uses a split format for O(n) total serialization:
        - Header file (.json): small dict with run_id and completed_items
          (rewritten each checkpoint, always tiny)
        - Results file (.jsonl): append-only, one JSON line per MetricResult
          (only new results are serialized and appended)

        Previously, the entire results list was re-serialized into a single
        JSON file on every checkpoint, giving O(n^2) total work over an
        evaluation run. This split approach keeps each checkpoint O(k)
        where k is the number of new results since last save.
        """
        if not self.checkpoint_path:
            return False

        try:
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            results_path = self._checkpoint_results_path

            # Append only new results to JSONL file
            new_start = self._checkpoint_count
            if new_start < len(results):
                with open(results_path, "a", encoding="utf-8") as rf:
                    for r in results[new_start:]:
                        rf.write(json.dumps(r.as_dict(), ensure_ascii=False, default=str))
                        rf.write("\n")
                    rf.flush()
                self._checkpoint_count = len(results)

            # Write header atomically (small file, always fast)
            header = {
                "run_id": run_id,
                "completed_items": list(completed_items),
                "results_file": results_path.name if results_path else None,
                "saved_at": now_utc().isoformat(),
            }
            temp_fd, temp_path = tempfile.mkstemp(
                dir=self.checkpoint_path.parent,
                prefix=".checkpoint_",
                suffix=".tmp",
            )
            try:
                with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
                    json.dump(header, f, ensure_ascii=False, default=str)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(temp_path, self.checkpoint_path)
                return True
            except Exception:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                raise
        except Exception as e:
            logger.warning("Failed to save checkpoint: %s", e)
            return False

    def _cleanup_checkpoint(self) -> None:
        """Remove checkpoint files after successful completion."""
        for path in (self.checkpoint_path, self._checkpoint_results_path):
            if path and path.exists():
                try:
                    path.unlink()
                except Exception:
                    pass

    def _discover_units(self, call: FunctionCall) -> list[EvalUnit]:
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
        self, item: DatasetItem, use_synthetic: bool, failures: list[str]
    ) -> FunctionCall | None:
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
        from .units import OutcomeBuilder

        return len(self.unit_builders) == 1 and isinstance(self.unit_builders[0], OutcomeBuilder)

    def _run_unit_evaluation(
        self,
        prepared: list[tuple[DatasetItem, FunctionCall]],
        run_id: str,
        completed_items: set,
    ) -> list[MetricResult]:
        """Run unit-based evaluation (for non-default unit types)."""
        from .guards import BudgetExceededError

        results: list[MetricResult] = []

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
                        # Invoke result_hook so the budget guard fires for
                        # unit-based runs too. Without this, --max-cost is
                        # silently inert when unit_types are configured.
                        if self.result_hook is not None:
                            try:
                                self.result_hook(result)
                            except BudgetExceededError as exc:
                                exc.partial_results = list(results)  # type: ignore[attr-defined]
                                raise

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

    def run_dataset(self, dataset: Iterable[DatasetItem], use_synthetic: bool = True) -> EvalRun:
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
        metric_results: list[MetricResult] = checkpoint["results"]
        completed_items: set = checkpoint["completed_items"]
        run_id = checkpoint["run_id"]
        failures: list[str] = []

        # Convert to list for progress tracking
        items = list(dataset)

        # Filter out already completed items
        pending_items = [
            (i, item) for i, item in enumerate(items) if item.id not in completed_items
        ]

        # Batch-fetch FunctionCalls from storage (1 query instead of N)
        storage = self.tracer.storage
        call_ids_to_fetch = []
        for _, item in pending_items:
            if isinstance(item.metadata, dict) and "call_id" in item.metadata:
                call_ids_to_fetch.append(item.metadata["call_id"])

        calls_by_id: dict[str, FunctionCall] = {}
        if call_ids_to_fetch and storage and hasattr(storage, "get_calls_batch"):
            calls_by_id = storage.get_calls_batch(call_ids_to_fetch)

        # Prepare all items with their FunctionCalls
        prepared: list[tuple[DatasetItem, FunctionCall]] = []
        for _item_idx, item in pending_items:
            # Try batch-fetched call first (O(1) dict lookup)
            call = None
            if isinstance(item.metadata, dict) and "call_id" in item.metadata:
                call = calls_by_id.get(item.metadata["call_id"])
            if call is None:
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

            # Objective metrics are CPU-bound microsecond operations (no I/O).
            # Threading adds ~3us overhead per task which dominates actual
            # compute for 100 items x 73 metrics (27ms overhead vs 1ms work).
            # Force sequential when all metrics are objective.
            effective_workers = self.max_workers
            if effective_workers > 1 and all(m.spec.type == "objective" for m in self.metrics):
                effective_workers = 1

            strategy = create_strategy(
                max_workers=effective_workers,
                evaluate_fn=self._evaluate_metric,
                checkpoint_fn=checkpoint_fn,
                checkpoint_interval=self.checkpoint_interval,
                result_hook=self.result_hook,
            )

            try:
                new_results = strategy.execute(
                    prepared=prepared,
                    metrics=self.metrics,
                    progress_callback=self._progress_callback,
                    run_id=run_id,
                    completed_items=completed_items,
                )
            except BudgetExceededError as exc:
                # Build a partial EvalRun from the in-memory results the
                # strategy attached, then re-raise so the CLI layer can
                # surface the abort with exit code 2.
                partial_new = list(exc.partial_results)
                full_results = list(metric_results) + partial_new
                summary = self._summarize(full_results, failures)
                usage_summary = self._compute_usage_summary(full_results)
                run = EvalRun(
                    id=run_id,
                    dataset_name=self.dataset_name,
                    created_at=now_utc(),
                    metric_results=full_results,
                    metrics=[m.spec for m in self.metrics],
                    summary=summary,
                    usage_summary=usage_summary,
                )
                if self.tracer.storage:
                    try:
                        self.tracer.storage.store_eval_run(run)
                    except Exception:
                        pass
                exc.partial_run = run  # type: ignore[attr-defined]
                raise
        else:
            # Unit-based mode: discover units and evaluate per-unit
            # Note: unit evaluation runs sequentially without checkpointing
            if self.max_workers > 1:
                logger.warning(
                    "Unit-based evaluation does not support parallel execution; "
                    "running sequentially (max_workers=%d ignored)",
                    self.max_workers,
                )
            if self.checkpoint_path:
                logger.warning(
                    "Unit-based evaluation does not support checkpointing; "
                    "checkpoint_path=%s ignored",
                    self.checkpoint_path,
                )
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
    def _summarize(results: list[MetricResult], failures: list[str]) -> dict:
        # Additive stats: `min_score`, `max_score`, `median_score` were
        # previously not exposed, forcing analysts to recompute them
        # from raw results in CODE_AUDIT EXT-034. Existing consumers
        # continue to read `count`, `avg_score`, and `pass_rate`
        # unchanged; new fields are optional.
        from statistics import median

        by_metric: defaultdict[str, list[MetricResult]] = defaultdict(list)
        for res in results:
            by_metric[res.metric_id].append(res)

        summary: dict = {"metrics": {}, "failed_items": failures}
        for metric_id, metric_results in by_metric.items():
            scores = [r.score for r in metric_results if r.score is not None]
            passes = [r.passed for r in metric_results if r.passed is not None]
            summary["metrics"][metric_id] = {
                "count": len(metric_results),
                "avg_score": (sum(scores) / len(scores)) if scores else None,
                "min_score": min(scores) if scores else None,
                "max_score": max(scores) if scores else None,
                "median_score": median(scores) if scores else None,
                "pass_rate": (sum(1 for p in passes if p) / len(passes)) if passes else None,
            }
        return summary

    @staticmethod
    def _compute_usage_summary(results: list[MetricResult]) -> dict:
        """Compute token usage and cost summary from metric results."""
        from ..trace.instrumentation.providers._shared import (
            calculate_cost,
            is_model_pricing_known,
        )

        total_input_tokens = 0
        total_output_tokens = 0
        models_used = set()
        has_unknown_pricing = False

        # Track costs by model and metric
        cost_by_model: dict[str, float] = defaultdict(float)
        cost_by_metric: dict[str, float] = defaultdict(float)
        tokens_by_metric: dict[str, dict[str, int]] = defaultdict(lambda: {"input": 0, "output": 0})

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
    dataset_dir: str | Path,
    *,
    runs_subdir: str = "eval_runs",
    _precomputed_dict: dict | None = None,
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
        _precomputed_dict: Optional pre-computed run.as_dict() to avoid
            redundant serialization when the caller also needs the dict.

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
    data = _precomputed_dict if _precomputed_dict is not None else run.as_dict()
    results_path = run_folder / "results.json"
    results_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    return run_folder


def load_eval_run_json(path: str | Path) -> EvalRun:
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


def list_eval_runs_json(dataset_dir: str | Path, runs_subdir: str = "eval_runs") -> list[EvalRun]:
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
