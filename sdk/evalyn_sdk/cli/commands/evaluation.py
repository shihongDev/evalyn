"""Evaluation commands: run-eval, suggest-metrics, select-metrics, list-metrics.

This module provides CLI commands for the core evaluation workflow:
suggesting metrics, running evaluations, and viewing available metric templates.

Commands:
- run-eval: Run evaluation on a dataset using metrics from JSON file(s)
- suggest-metrics: Suggest metrics for a project using various modes (basic, llm-registry, llm-brainstorm, bundle)
- select-metrics: LLM-guided selection from metric registry (advanced)
- list-metrics: List all available objective and subjective metric templates

Metric suggestion modes:
- basic: Fast heuristic-based suggestion (no LLM required)
- bundle: Preset metric bundles for common use cases (summarization, orchestrator, etc.)
- llm-registry: LLM picks from the metric registry based on function analysis
- llm-brainstorm: LLM generates custom metric ideas (most creative, requires API key)

Metric types:
- objective: Deterministic metrics (length, format checks, word count, etc.)
- subjective: LLM-as-judge metrics (quality, relevance, safety, etc.)

Typical workflow:
1. Build dataset: 'evalyn build-dataset --project <name>'
2. Suggest metrics: 'evalyn suggest-metrics --project <name> --dataset <path> --mode basic'
3. Run evaluation: 'evalyn run-eval --dataset <path>'
4. Analyze results: 'evalyn analyze --latest'
"""

from __future__ import annotations

# Dashboard catalog group (used by evalyn_dashboard.introspect.build_catalog).
GROUP = "Eval"

# Non-required params that should still appear in the default form. Read by
# evalyn_dashboard.introspect.build_catalog and merged with required params.
# `dataset` + `metrics` cover the two knobs every run-eval invocation tunes;
# other run-eval optionals stay behind the "Show all options" disclosure.
ESSENTIAL = {"dataset", "metrics"}

import argparse
import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..constants import BUNDLES
from ..utils.config import get_config_default, load_config, resolve_dataset_path
from ..utils.dataset_utils import (
    ProgressBar,
    _dataset_has_reference,
    _extract_code_meta,
    _resolve_dataset_and_metrics,
)
from ..utils.errors import fatal_error
from ..utils.formatters import print_token_usage_summary
from ..utils.hints import HintCollector
from ..utils.loaders import _load_callable
from ..utils.rich import banner, icon, kv, section
from ..utils.rich import table as rich_table
from ..utils.validation import check_llm_api_keys


def _save_suggested_metrics(
    specs: list[MetricSpec],
    dataset_path: Path | None,
    metrics_name: str | None,
    append: bool,
    selected_mode: str,
    output_format: str,
    quiet: bool,
) -> Path | None:
    """Save suggested metrics to dataset's metrics directory.

    Args:
        specs: List of metric specs to save
        dataset_path: Path to dataset directory or file
        metrics_name: Optional name for the metrics file
        append: Whether to append to existing metrics
        selected_mode: The suggestion mode used (for metadata)
        output_format: Output format ('json' or 'table')
        quiet: Whether to suppress hints

    Returns:
        Path to saved metrics file, or None if not saved
    """
    from ...metrics.objective import OBJECTIVE_REGISTRY

    if not dataset_path:
        return None

    dataset_dir = dataset_path.parent if dataset_path.is_file() else dataset_path
    metrics_dir = dataset_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Standardized naming
    if metrics_name:
        safe_name = re.sub(r"[^a-zA-Z0-9._-]+", "-", metrics_name).strip("-")
    else:
        safe_name = "default"
    metrics_file = metrics_dir / (
        f"metrics-{safe_name}.json" if metrics_name else "metrics.json"
    )

    # Handle --append: merge with existing metrics
    existing_metrics: list[dict] = []
    if append and metrics_file.exists():
        try:
            existing_metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
            existing_ids = {m["id"] for m in existing_metrics}
            new_specs = [s for s in specs if s.id not in existing_ids]
            if not new_specs:
                if output_format != "json":
                    print("All suggested metrics already exist. Nothing to add.")
                return metrics_file
            specs = new_specs
            if output_format != "json":
                print(
                    f"Appending {len(new_specs)} new metric(s) to {len(existing_metrics)} existing."
                )
        except Exception as e:
            if output_format != "json":
                print(f"Warning: Could not read existing metrics for append: {e}")

    # Validate objective metrics - filter out custom ones
    valid_objective_ids = {t["id"] for t in OBJECTIVE_REGISTRY}
    invalid_objectives = [
        s for s in specs if s.type == "objective" and s.id not in valid_objective_ids
    ]
    if invalid_objectives and output_format != "json":
        print(
            f"Removed {len(invalid_objectives)} unsupported custom objective metric(s):"
        )
        for s in invalid_objectives:
            print(
                f"  - {s.id}: Use 'evalyn list-metrics --type objective' to see valid IDs"
            )
    specs = [
        s
        for s in specs
        if not (s.type == "objective" and s.id not in valid_objective_ids)
    ]

    if not specs:
        if output_format != "json":
            print("No valid metrics to save.")
        return None

    payload = existing_metrics + [
        {
            "id": spec.id,
            "type": spec.type,
            "description": spec.description,
            "config": spec.config,
            "why": getattr(spec, "why", ""),
        }
        for spec in specs
    ]
    metrics_file.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8"
    )

    # Update meta.json
    meta_path = dataset_dir / "meta.json"
    try:
        meta = (
            json.loads(meta_path.read_text(encoding="utf-8"))
            if meta_path.exists()
            else {}
        )
    except Exception:
        meta = {}
    existing_sets = meta.get("metric_sets")
    metric_sets = existing_sets if isinstance(existing_sets, list) else []
    entry = {
        "name": safe_name,
        "file": f"metrics/{metrics_file.name}",
        "mode": selected_mode,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "num_metrics": len(payload),
    }
    metric_sets = [m for m in metric_sets if m.get("name") != safe_name]
    metric_sets.append(entry)
    meta["metric_sets"] = metric_sets
    meta["active_metric_set"] = safe_name
    meta_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=True), encoding="utf-8"
    )

    if output_format != "json":
        print(f"{icon('pass')} Saved metrics to {metrics_file}")
        hints = HintCollector(quiet=quiet, format=output_format)
        hints.add(
            f"evalyn run-eval --dataset {dataset_dir}",
            "Run evaluation with these metrics",
            options=[
                ("--provider gemini|openai|ollama", "LLM provider for judges (default: gemini)"),
                ("--workers <N>", "Parallel workers (default: 4)"),
                ("--use-calibrated", "Use calibrated prompts if available"),
                ("--confidence consistency", "Enable confidence scoring"),
                ("--verbose", "Show detailed cost breakdown by metric"),
            ],
        )
        hints.render()
    return metrics_file


def _build_llm_caller(args: argparse.Namespace) -> Callable:
    """Build an LLM caller from args."""
    from ...defaults import DEFAULT_EVAL_MODEL
    from ...utils.api_client import call_gemini_api

    # Custom caller if provided
    if args.llm_caller:
        return _load_callable(args.llm_caller)

    # Build default caller - check args, then config, then env
    config = load_config()
    api_key = (
        args.api_key
        or get_config_default(config, "api_keys", "gemini")
        or os.environ.get("GEMINI_API_KEY", "")
    )
    model = args.model or DEFAULT_EVAL_MODEL

    def default_caller(prompt: str) -> str:
        return call_gemini_api(prompt, model=model, api_key=api_key)

    return default_caller


def _resolve_run_eval_inputs(
    args: argparse.Namespace,
    output_format: str,
    config: dict,
) -> tuple[Path, list[Any], dict[str, dict], list[Path]]:
    """Resolve dataset and load merged metric specs for run-eval."""
    from ...datasets import load_dataset

    metrics_all = getattr(args, "metrics_all", False)

    dataset_arg = getattr(args, "dataset", None)
    use_latest = getattr(args, "latest", False)
    dataset_path = resolve_dataset_path(dataset_arg, use_latest, config)
    if not dataset_path:
        fatal_error("No dataset specified", "Use --dataset <path> or --latest")

    try:
        dataset_file, metrics_paths = _resolve_dataset_and_metrics(
            str(dataset_path), args.metrics, metrics_all=metrics_all
        )
    except (FileNotFoundError, ValueError) as e:
        fatal_error(str(e))

    dataset_list = list(load_dataset(str(dataset_file)))

    all_metrics_data: dict[str, dict] = {}
    duplicate_ids: list[str] = []
    for metrics_path in metrics_paths:
        try:
            file_data = json.loads(metrics_path.read_text(encoding="utf-8"))
            if not isinstance(file_data, list):
                if output_format != "json":
                    print(f"Warning: Skipping {metrics_path} - not a JSON array")
                continue
            for spec_data in file_data:
                metric_id = spec_data.get("id")
                if metric_id:
                    if metric_id in all_metrics_data:
                        duplicate_ids.append(metric_id)
                    else:
                        all_metrics_data[metric_id] = spec_data
        except Exception as e:
            if output_format != "json":
                print(f"Warning: Failed to load {metrics_path}: {e}")

    if duplicate_ids and output_format != "json":
        unique_dups = sorted(set(duplicate_ids))
        print(
            f"Warning: {len(unique_dups)} duplicate metric ID(s) found (first definition wins):"
        )
        for dup_id in unique_dups[:5]:
            print(f"  - {dup_id}")
        if len(unique_dups) > 5:
            print(f"  ... and {len(unique_dups) - 5} more")

    if not all_metrics_data:
        fatal_error("No valid metrics loaded from files")

    if output_format != "json" and len(metrics_paths) > 1:
        print(
            f"Merged {len(all_metrics_data)} unique metrics from {len(metrics_paths)} files"
        )

    return dataset_file, dataset_list, all_metrics_data, metrics_paths


def _parse_unit_types(args: argparse.Namespace) -> list[str] | None:
    """Parse and validate --unit-types."""
    unit_types_raw = getattr(args, "unit_types", None)
    if not unit_types_raw:
        return None

    unit_types = [t.strip() for t in unit_types_raw.split(",") if t.strip()]
    valid_types = {"outcome", "single_turn", "tool_use", "multi_turn", "custom"}
    invalid = [t for t in unit_types if t not in valid_types]
    if invalid:
        fatal_error(
            f"Invalid unit type(s): {', '.join(invalid)}",
            f"Valid types: {', '.join(sorted(valid_types))}",
        )
    return unit_types


@dataclass
class EvalMetricsConfig:
    """Configuration and metrics built for an evaluation run."""

    metrics: list[Any] = field(default_factory=list)
    objective_count: int = 0
    subjective_count: int = 0
    calibrated_count: int = 0
    provider: str = "gemini"
    confidence_method: str = "none"
    confidence_samples: int = 3
    unit_types: list[str] | None = None


def _build_run_eval_metrics(
    args: argparse.Namespace,
    output_format: str,
    config: dict,
    dataset_dir: Path,
    all_metrics_data: dict[str, dict],
) -> EvalMetricsConfig:
    """Build metric instances and return counters/settings."""
    from ...calibration import load_optimized_prompt
    from ...metrics.factory import build_objective_metric, build_subjective_metric
    from ...models import MetricSpec

    metrics: list[Any] = []
    objective_count = 0
    subjective_count = 0
    calibrated_count = 0

    provider = getattr(args, "provider", "gemini")
    confidence_method = getattr(args, "confidence", "none")
    confidence_samples = getattr(args, "confidence_samples", 3)
    unit_types = _parse_unit_types(args)

    if confidence_method in ("logprobs", "deepconf") and provider == "gemini":
        fatal_error(
            f"{confidence_method} confidence requires --provider openai or --provider ollama",
            "Gemini does not expose logprobs. Use --confidence consistency instead.",
        )

    if provider == "openai":
        api_key = get_config_default(config, "api_keys", "openai") or os.environ.get(
            "OPENAI_API_KEY"
        )
    else:
        api_key = get_config_default(config, "api_keys", "gemini")

    use_calibrated = getattr(args, "use_calibrated", False)
    skipped_metrics = []

    for spec_data in all_metrics_data.values():
        spec = MetricSpec.from_dict(spec_data)

        if use_calibrated and spec.type == "subjective":
            try:
                optimized_prompt = load_optimized_prompt(str(dataset_dir), spec.id)
                if optimized_prompt:
                    spec.config = dict(spec.config or {})
                    spec.config["prompt"] = optimized_prompt
                    calibrated_count += 1
                    if output_format != "json":
                        print(f"  Using calibrated prompt for {spec.id}")
            except Exception as e:
                if output_format != "json":
                    print(f"  Warning: Could not load calibrated prompt for {spec.id}: {e}")

        try:
            if spec.type == "objective":
                metric = build_objective_metric(spec.id, spec.config)
                objective_count += 1
            else:
                metric = build_subjective_metric(
                    spec.id,
                    spec.config,
                    description=spec.description,
                    api_key=api_key,
                    provider=provider,
                    confidence_method=confidence_method,
                    confidence_samples=confidence_samples,
                )
                subjective_count += 1
            if metric:
                metrics.append(metric)
        except KeyError as e:
            skipped_metrics.append((spec.id, spec.type, str(e)))
        except Exception as e:
            if output_format != "json":
                print(f"Warning: Failed to build metric '{spec.id}': {e}")

    if skipped_metrics and output_format != "json":
        print(f"Skipped {len(skipped_metrics)} unknown metrics:")
        for mid, mtype, reason in skipped_metrics:
            if mtype == "objective":
                print(
                    f"  - {mid} [objective]: Custom objective metrics not supported. Use 'evalyn list-metrics' to see available templates."
                )
            else:
                print(f"  - {mid}: {reason}")

    if not metrics:
        fatal_error("No valid metrics loaded from file")

    if output_format != "json":
        metrics_summary = (
            f"Loaded {len(metrics)} metrics ({objective_count} objective, {subjective_count} subjective"
        )
        if calibrated_count > 0:
            metrics_summary += f", {calibrated_count} calibrated"
        metrics_summary += ")"
        print(metrics_summary)
        if subjective_count > 0:
            judge_info = f"Judge: {provider}"
            if confidence_method != "none":
                if confidence_method == "consistency":
                    judge_info += (
                        f", confidence=consistency({confidence_samples} samples)"
                    )
                else:
                    judge_info += f", confidence={confidence_method}"
            print(judge_info)

    return EvalMetricsConfig(
        metrics=metrics,
        objective_count=objective_count,
        subjective_count=subjective_count,
        calibrated_count=calibrated_count,
        provider=provider,
        confidence_method=confidence_method,
        confidence_samples=confidence_samples,
        unit_types=unit_types,
    )


def _execute_run_eval(
    args: argparse.Namespace,
    output_format: str,
    dataset_file: Path,
    dataset_dir: Path,
    dataset_list: list[Any],
    metrics: list[Any],
    subjective_count: int,
    unit_types: list[str] | None,
):
    """Execute evaluation in standard or batch mode."""
    from ...decorators import get_default_tracer

    tracer = get_default_tracer()
    if not tracer.storage:
        fatal_error("No storage configured")

    total_evals = len(dataset_list) * len(metrics)
    progress = ProgressBar(total_evals) if output_format != "json" else None

    def progress_callback(
        current: int, total: int, metric: str, metric_type: str
    ) -> None:
        if progress:
            progress.update(current, total, metric, metric_type)

    from ...evaluation.runner import EvalRunner

    checkpoint_path = dataset_dir / ".eval_checkpoint.json"
    use_batch = getattr(args, "batch", False)

    if use_batch and subjective_count == 0 and output_format != "json":
        print("Note: --batch only applies to subjective metrics. No subjective metrics found; ignoring --batch.")

    if use_batch and subjective_count > 0:
        from ...evaluation.batch import BatchEvalProgress, BatchEvaluator
        from ...models import EvalRun

        batch_provider = getattr(args, "batch_provider", "gemini")
        if output_format != "json":
            print(
                f"\nUsing batch API ({batch_provider}) for {subjective_count} subjective metrics"
            )
            print("This may take several minutes. Progress will be shown below.\n")

        objective_metrics = [m for m in metrics if m.spec.type == "objective"]
        subjective_metrics = [m for m in metrics if m.spec.type == "subjective"]

        objective_results = []
        if objective_metrics:
            if output_format != "json":
                print(f"Running {len(objective_metrics)} objective metrics...")
            runner = EvalRunner(
                target_fn=lambda: None,
                metrics=objective_metrics,
                dataset_name=args.dataset_name or dataset_file.stem,
                tracer=tracer,
                instrument=False,
                progress_callback=progress_callback if output_format != "json" else None,
                max_workers=getattr(args, "workers", 1),
                unit_types=unit_types,
            )
            obj_run = runner.run_dataset(dataset_list, use_synthetic=True)
            objective_results = obj_run.metric_results

        subjective_results = []
        if subjective_metrics:
            if output_format != "json":
                print(
                    f"\nSubmitting {len(subjective_metrics)} subjective metrics to batch API..."
                )

            prepared = []
            call_ids = []
            item_call_id_pairs = []
            for item in dataset_list:
                call_id = item.metadata.get("call_id") if isinstance(item.metadata, dict) else None
                if call_id:
                    call_ids.append(call_id)
                    item_call_id_pairs.append((item, call_id))
            calls_by_id = tracer.storage.get_calls_batch(call_ids) if call_ids else {}
            for item, call_id in item_call_id_pairs:
                call = calls_by_id.get(call_id)
                if call:
                    prepared.append((item, call))

            def batch_progress(p: BatchEvalProgress):
                if output_format != "json":
                    if p.phase == "waiting":
                        pct = 100 * p.completed_requests / max(1, p.total_requests)
                        print(
                            f"\r  Batch progress: {p.completed_requests}/{p.total_requests} ({pct:.0f}%) - {p.elapsed_seconds:.0f}s elapsed",
                            end="",
                            flush=True,
                        )
                    elif p.phase == "complete":
                        print(
                            f"\n  Batch complete: {p.completed_requests} results in {p.elapsed_seconds:.1f}s"
                        )

            batch_evaluator = BatchEvaluator(provider=batch_provider, poll_interval=10.0)

            try:
                subjective_results = batch_evaluator.evaluate(
                    prepared=prepared,
                    metrics=subjective_metrics,
                    progress_callback=batch_progress,
                    checkpoint_path=checkpoint_path,
                )
            except KeyboardInterrupt:
                if output_format != "json":
                    print("\n\nBatch evaluation interrupted.")
                    if checkpoint_path.exists():
                        print(f"Batch job checkpoint saved to: {checkpoint_path}")
                return None
            except Exception as e:
                if output_format != "json":
                    print(f"\n\nBatch evaluation failed: {e}")
                return None

        run = EvalRun(
            id=f"batch-{int(datetime.now(timezone.utc).timestamp())}",
            dataset_name=args.dataset_name or dataset_file.stem,
            metric_results=objective_results + subjective_results,
            summary={},
            created_at=datetime.now(timezone.utc),
        )
        if progress:
            progress.finish()
        return run

    if checkpoint_path.exists() and output_format != "json":
        print("  Resuming from checkpoint...")

    runner = EvalRunner(
        target_fn=lambda: None,
        metrics=metrics,
        dataset_name=args.dataset_name or dataset_file.stem,
        tracer=tracer,
        instrument=False,
        progress_callback=progress_callback if output_format != "json" else None,
        checkpoint_path=checkpoint_path,
        checkpoint_interval=5,
        max_workers=getattr(args, "workers", 1),
        unit_types=unit_types,
    )

    try:
        run = runner.run_dataset(dataset_list, use_synthetic=True)
    except KeyboardInterrupt:
        if progress:
            progress.finish()
        if output_format != "json":
            print("\n\nEvaluation interrupted.")
            print(f"Progress saved to: {checkpoint_path}")
            print(f"Resume with: evalyn run-eval --dataset {dataset_dir}")
        return None

    if progress:
        progress.finish()
    return run


def _save_eval_run_and_report(
    run: Any,
    output_format: str,
    dataset_dir: Path,
    dataset_list: list[Any],
) -> tuple[Path, Path, Path | None, Any | None]:
    """Persist run and generate HTML report.

    Returns (run_folder, results_path, report_path, run_analysis).
    run_analysis is the RunAnalysis object if analysis succeeded, else None.
    """
    from ...analysis import (
        analyze_run as analyze_run_data,
    )
    from ...analysis import (
        generate_html_report,
    )
    from ...evaluation.runner import save_eval_run_json

    run_data = run.as_dict()  # serialize once, reuse for both file and analysis
    run_folder = save_eval_run_json(run, dataset_dir, _precomputed_dict=run_data)
    results_path = run_folder / "results.json"
    analysis = None

    try:
        analysis = analyze_run_data(run_data)
        item_details = {}
        for item in dataset_list:
            item_details[item.id] = {
                "input": item.input or item.inputs,
                "output": item.output or item.expected,
            }
        html_report = generate_html_report(analysis, item_details=item_details)
        report_path = run_folder / "report.html"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_report)
    except Exception as e:
        report_path = None
        if output_format != "json":
            print(f"Warning: Could not generate HTML report: {e}")

    return run_folder, results_path, report_path, analysis


def _render_run_eval_output(
    args: argparse.Namespace,
    output_format: str,
    run: Any,
    run_folder: Path,
    results_path: Path,
    report_path: Path | None,
    metrics: list[Any],
) -> None:
    """Render run-eval output in JSON or table format."""
    if output_format == "json":
        result = {
            "id": run.id,
            "dataset_name": run.dataset_name,
            "created_at": run.created_at.isoformat() if run.created_at else None,
            "summary": run.summary,
            "run_folder": str(run_folder),
            "results_file": str(results_path),
            "report_file": str(report_path) if report_path else None,
            "metric_results": [
                {
                    "metric_id": r.metric_id,
                    "item_id": r.item_id,
                    "call_id": r.call_id,
                    "score": r.score,
                    "passed": r.passed,
                    "details": r.details,
                }
                for r in run.metric_results
            ],
        }
        print(json.dumps(result, indent=2))
        return

    print(f"\n{icon('pass')} Eval run {run.id}")
    print(kv([
        ("Dataset", run.dataset_name),
        ("Run folder", str(run_folder)),
    ]))
    print("  results.json - evaluation data")
    if report_path:
        print("  report.html  - analysis report")
    print()

    metric_types = {m.spec.id: m.spec.type for m in metrics}
    api_errors_by_metric: dict[str, int] = {}
    for result in run.metric_results:
        if result.details and isinstance(result.details, dict):
            reason = result.details.get("reason") or ""
            if (
                reason
                and "API" in reason
                and ("error" in reason.lower() or "failed" in reason.lower())
            ):
                api_errors_by_metric[result.metric_id] = (
                    api_errors_by_metric.get(result.metric_id, 0) + 1
                )

    print("Results:")
    print("-" * 80)
    print(
        f"{'Metric':<25} {'Type':<10} {'Count':<6} {'Avg Score':<12} {'Pass Rate':<10} {'Errors':<8}"
    )
    print("-" * 80)

    for metric_id, stats in run.summary.get("metrics", {}).items():
        mtype = metric_types.get(metric_id, "?")
        type_label = "obj" if mtype == "objective" else "llm"
        avg_score = stats.get("avg_score")
        avg_score_str = f"{avg_score:.4f}" if avg_score is not None else "N/A"
        pass_rate = stats.get("pass_rate")
        pass_rate_str = f"{pass_rate * 100:.1f}%" if pass_rate is not None else "N/A"
        error_count = api_errors_by_metric.get(metric_id, 0)
        error_str = f"{error_count}" if error_count > 0 else "-"
        print(
            f"{metric_id:<25} [{type_label:<3}]    {stats['count']:<6} {avg_score_str:<12} {pass_rate_str:<10} {error_str:<8}"
        )

    print("-" * 80)

    total_api_errors = sum(api_errors_by_metric.values())
    if total_api_errors > 0:
        print(f"\n{total_api_errors} API error(s) detected in LLM judge results.")
        print("   Check that GEMINI_API_KEY or OPENAI_API_KEY is set and valid.")
        print("   Re-run with a valid API key to get accurate LLM judge scores.")

    print_token_usage_summary(
        run.usage_summary, verbose=getattr(args, "verbose", False)
    )

    failed_items = run.summary.get("failed_items", [])
    failed_count = len(failed_items) if isinstance(failed_items, list) else failed_items
    if failed_count:
        print(f"Failed items: {failed_count}")

    quiet = getattr(args, "quiet", False)
    hints = HintCollector(quiet=quiet, format=output_format)
    if failed_count >= 3:
        failed_metric = None
        for metric_id, stats in run.summary.get("metrics", {}).items():
            if stats.get("failed", 0) > 0:
                failed_metric = metric_id
                break
        if failed_metric:
            hints.add(
                f"evalyn cluster-failures --metric-id {failed_metric}",
                "Cluster failures by pattern",
                options=[
                    ("--top <N>", "Number of clusters to show (default: 5)"),
                ],
            )
    hints.add(
        f"evalyn analyze --run {run.id}",
        "Analyze results in detail",
        options=[
            ("--format json", "Machine-readable output"),
        ],
    )
    hints.render()


def _run_auto_insights(
    run_analysis: Any,
    dataset_dir: Path,
    dataset_list: list[Any],
    results_path: Path,
) -> None:
    """Run lightweight deterministic insights after an eval and print a summary.

    This runs after every run-eval to surface actionable intelligence without
    requiring an explicit insights command. No LLM calls are made.

    Two tiers:
    - Lightweight (always): regression detection, top recommendations
    - Deep (dataset <= 500 items): metric correlations, input feature analysis
    """
    from ...analysis import (
        analyze_run as analyze_run_data,
    )
    from ...analysis import (
        find_eval_runs,
        load_eval_run,
    )
    from ...analysis.insights import (
        InsightsReport,
        analyze_input_features,
        analyze_score_distributions,
        compute_metric_correlations,
        detect_regressions,
        generate_recommendations,
    )

    print()
    print("=" * 80)
    print("  AUTO-INSIGHTS")
    print("=" * 80)

    previous_analysis = None
    try:
        all_runs = find_eval_runs(dataset_dir)
        resolved_current = results_path.resolve()
        previous_runs = [r for r in all_runs if r.resolve() != resolved_current]
        if previous_runs:
            prev_data = load_eval_run(previous_runs[0])
            previous_analysis = analyze_run_data(prev_data)
    except Exception as e:
        print(f"  Warning: could not load previous run: {e}")
        previous_analysis = None

    regressions = []
    if previous_analysis is not None:
        try:
            regressions = detect_regressions(run_analysis, previous_analysis)
        except Exception as e:
            print(f"  Warning: regression detection failed: {e}")
            regressions = []

        if regressions:
            print()
            print("Regressions detected:")
            severity_icons = {"critical": "!!!", "warning": "! ", "info": "  "}
            for alert in regressions:
                icon = severity_icons.get(alert.severity, "  ")
                print(
                    f"  {icon} {alert.metric_id}: "
                    f"{alert.previous_pass_rate * 100:.0f}% -> {alert.current_pass_rate * 100:.0f}% "
                    f"({alert.delta * 100:+.0f}pp) [{alert.severity}]"
                )
        else:
            print()
            print("No regressions detected vs. previous run.")
    else:
        print()
        print("This run will be used as the baseline for future comparisons.")

    report = InsightsReport()
    report.regressions = regressions

    num_items = len(dataset_list)
    if num_items <= 500:
        try:
            correlations = compute_metric_correlations(run_analysis)
            report.correlations = correlations
            if correlations:
                print()
                print("Metric correlations:")
                for corr in correlations[:5]:
                    label = "redundant" if corr.relationship == "redundant" else "tradeoff"
                    print(f"  {corr.metric_a} <-> {corr.metric_b}: r={corr.pearson} ({label})")
        except Exception as e:
            print(f"  Warning: correlation analysis failed: {e}")

        try:
            dataset_dicts = []
            for item in dataset_list:
                d: dict[str, Any] = {"id": item.id}
                for field in ("input", "inputs", "output", "expected"):
                    val = getattr(item, field, None)
                    if val is not None:
                        d[field] = val
                dataset_dicts.append(d)
            feature_insights = analyze_input_features(dataset_dicts, run_analysis)
            report.feature_insights = feature_insights
            if feature_insights:
                print()
                print("Feature analysis:")
                for feat in feature_insights:
                    print(f"  {feat.finding}")
        except Exception as e:
            print(f"  Warning: feature analysis failed: {e}")

    try:
        dist_insights = analyze_score_distributions(run_analysis)
        report.distribution_insights = dist_insights
    except Exception as e:
        print(f"  Warning: score distribution analysis failed: {e}")

    try:
        recommendations = generate_recommendations(run_analysis, report)
        if recommendations:
            print()
            top_recs = recommendations[:3]
            print("Top recommendations:")
            for rec in top_recs:
                print(f"  {rec.priority}. [{rec.category}] {rec.message}")
                print(f"     -> {rec.action}")
    except Exception as e:
        print(f"  Warning: recommendation generation failed: {e}")

    print()
    print("Run `evalyn insights --deep` for full LLM expert analysis.")
    print("=" * 80)


def cmd_run_eval(args: argparse.Namespace) -> None:
    """Run evaluation using pre-computed traces from dataset and metrics from JSON file(s).

    This command evaluates a dataset against a set of metrics:
    1. Load dataset items (input/output pairs from traced calls)
    2. Load metric specs from JSON file(s)
    3. Build metric instances (objective or subjective/LLM-judge)
    4. Run each metric against each item
    5. Save results to eval_runs/<timestamp>/results.json
    6. Generate HTML analysis report

    For subjective metrics, an API key (GEMINI_API_KEY or OPENAI_API_KEY) is required.
    Use --use-calibrated to apply optimized prompts from previous calibration.
    """
    output_format = getattr(args, "format", "table")
    config = load_config()

    dataset_file, dataset_list, all_metrics_data, _ = _resolve_run_eval_inputs(
        args, output_format, config
    )
    dataset_dir = dataset_file.parent

    eval_config = _build_run_eval_metrics(
        args,
        output_format,
        config,
        dataset_dir,
        all_metrics_data,
    )

    if output_format != "json":
        print(f"Dataset: {len(dataset_list)} items")
        if eval_config.unit_types and eval_config.unit_types != ["outcome"]:
            print(f"Unit types: {', '.join(eval_config.unit_types)} (span-level evaluation)")
        if eval_config.subjective_count > 0:
            check_llm_api_keys(quiet=False)
        print()

    run = _execute_run_eval(
        args,
        output_format,
        dataset_file,
        dataset_dir,
        dataset_list,
        eval_config.metrics,
        eval_config.subjective_count,
        eval_config.unit_types,
    )
    if run is None:
        return

    run_folder, results_path, report_path, run_analysis = _save_eval_run_and_report(
        run, output_format, dataset_dir, dataset_list
    )
    _render_run_eval_output(
        args,
        output_format,
        run,
        run_folder,
        results_path,
        report_path,
        eval_config.metrics,
    )

    # Auto-insights: lightweight deterministic analysis after every eval
    if output_format != "json" and run_analysis is not None:
        _run_auto_insights(
            run_analysis=run_analysis,
            dataset_dir=dataset_dir,
            dataset_list=dataset_list,
            results_path=results_path,
        )


def cmd_suggest_metrics(args: argparse.Namespace) -> None:
    """Suggest metrics for a project or target function.

    Modes:
    - basic: Fast heuristic based on function signature and traces (no LLM)
    - bundle: Preset metrics for common patterns (summarization, orchestrator, etc.)
    - llm-registry: LLM picks from objective/subjective registry templates
    - llm-brainstorm: LLM generates custom metric ideas (most creative)

    Scope filters:
    - overall: Metrics that evaluate final output quality
    - llm_call: Metrics that evaluate individual LLM calls
    - tool_call: Metrics that evaluate tool usage
    - trace: Metrics that aggregate across the entire trace
    """
    from ...decorators import get_default_tracer
    from ...metrics.objective import OBJECTIVE_REGISTRY
    from ...metrics.subjective import SUBJECTIVE_REGISTRY
    from ...metrics.suggester import LLMSuggester, TemplateSelector

    output_format = getattr(args, "format", "table")
    tracer = get_default_tracer()

    _maybe_infer_project_from_dataset_meta(args)
    _validate_suggest_metrics_args(args)

    config = load_config()
    dataset_path_obj = _resolve_suggest_metrics_dataset_path(args, config, output_format)
    has_reference = _dataset_has_reference(dataset_path_obj)
    _print_reference_note_if_needed(dataset_path_obj, has_reference, output_format)

    context = _load_suggest_metrics_context(args, tracer, output_format)
    target_fn = context["target_fn"]
    metric_mode_hint = context["metric_mode_hint"]
    metric_bundle_hint = context["metric_bundle_hint"]
    traces = context["traces"]

    selected_mode = _select_metric_suggestion_mode(args.mode, metric_mode_hint)
    bundle_name = args.bundle or metric_bundle_hint
    max_metrics = _normalize_suggest_metrics_limit(selected_mode, args.num_metrics)
    scope_filter = _resolve_scope_filter(args, output_format)

    if selected_mode == "bundle":
        _run_suggest_metrics_bundle_mode(
            args=args,
            bundle_name=bundle_name,
            has_reference=has_reference,
            scope_filter=scope_filter,
            max_metrics=max_metrics,
            dataset_path_obj=dataset_path_obj,
            output_format=output_format,
        )
        return

    if selected_mode == "llm-registry":
        caller = _build_llm_caller(args)
        filtered_templates = _filter_metric_templates_by_scope(
            OBJECTIVE_REGISTRY + SUBJECTIVE_REGISTRY, scope_filter
        )
        selector = TemplateSelector(
            caller, filtered_templates, has_reference=has_reference
        )
        specs = selector.select(
            target_fn,
            traces=traces,
            code_meta=_extract_code_meta(tracer, target_fn),
            desired_count=max_metrics,
        )
        _finalize_suggest_metrics_output(
            specs=specs,
            max_metrics=max_metrics,
            dataset_path_obj=dataset_path_obj,
            args=args,
            selected_mode=selected_mode,
            output_format=output_format,
        )
        return

    if selected_mode == "llm-brainstorm":
        caller = _build_llm_caller(args)
        suggester = LLMSuggester(caller=caller)
        specs = suggester.suggest(
            target_fn, traces, desired_count=max_metrics, scope=scope_filter
        )
        if not specs:
            _print_suggest_metrics_empty_brainstorm(output_format)
            return
        _finalize_suggest_metrics_output(
            specs=specs,
            max_metrics=max_metrics,
            dataset_path_obj=dataset_path_obj,
            args=args,
            selected_mode=selected_mode,
            output_format=output_format,
        )
        return

    specs = _run_suggest_metrics_basic_mode(
        target_fn=target_fn,
        traces=traces,
        has_reference=has_reference,
        scope_filter=scope_filter,
    )
    _finalize_suggest_metrics_output(
        specs=specs,
        max_metrics=max_metrics,
        dataset_path_obj=dataset_path_obj,
        args=args,
        selected_mode=selected_mode,
        output_format=output_format,
    )


def _maybe_infer_project_from_dataset_meta(args: argparse.Namespace) -> None:
    """Infer --project from dataset meta.json when only --dataset is provided."""
    if args.project or args.target or not args.dataset:
        return
    dataset_path = Path(args.dataset)
    if dataset_path.is_file():
        dataset_path = dataset_path.parent
    meta_file = dataset_path / "meta.json"
    if not meta_file.exists():
        return
    try:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # meta.json is optional - missing / unreadable means we just
        # don't auto-fill args.project from it.
        return
    if meta.get("project"):
        args.project = meta["project"]


def _validate_suggest_metrics_args(args: argparse.Namespace) -> None:
    """Validate required args for metric suggestion."""
    if args.project or args.target:
        return
    fatal_error(
        "Either --project or --target is required",
        "Use --project <name>, --target <path>, or --dataset <path>",
    )


def _resolve_suggest_metrics_dataset_path(
    args: argparse.Namespace, config: dict, output_format: str
) -> Path | None:
    """Resolve and validate optional dataset path for metric suggestion."""
    dataset_path_obj: Path | None = None
    use_latest = getattr(args, "latest", False)

    if args.dataset or use_latest:
        resolved = resolve_dataset_path(args.dataset, use_latest, config)
        if resolved:
            dataset_path_obj = resolved
        elif args.dataset:
            dataset_path_obj = Path(args.dataset)

    if not dataset_path_obj:
        return None

    if dataset_path_obj.is_file():
        if not dataset_path_obj.exists():
            fatal_error(
                f"Dataset file not found: {dataset_path_obj}",
                "Run 'evalyn build-dataset' first",
            )
        return dataset_path_obj

    if not dataset_path_obj.exists():
        fatal_error(
            f"Dataset directory not found: {dataset_path_obj}",
            "Run 'evalyn build-dataset' first",
        )
    has_dataset = (dataset_path_obj / "dataset.jsonl").exists() or (
        dataset_path_obj / "dataset.json"
    ).exists()
    if not has_dataset and output_format != "json":
        print(
            f"Warning: Directory exists but no dataset.jsonl or dataset.json found in: {dataset_path_obj}"
        )
    return dataset_path_obj


def _print_reference_note_if_needed(
    dataset_path_obj: Path | None, has_reference: bool, output_format: str
) -> None:
    """Print dataset reference-value note when relevant."""
    if dataset_path_obj and not has_reference and output_format != "json":
        print(
            "Note: Dataset has no reference/expected values. Reference-based metrics (ROUGE, BLEU, etc.) excluded."
        )


def _load_suggest_metrics_context(
    args: argparse.Namespace, tracer, output_format: str
) -> dict[str, Any]:
    """Load traces and target function context from project or target input."""
    if args.project:
        return _load_suggest_metrics_project_context(args, tracer, output_format)
    return _load_suggest_metrics_target_context(args, tracer)


def _load_suggest_metrics_project_context(
    args: argparse.Namespace, tracer, output_format: str
) -> dict[str, Any]:
    """Load trace context for project-based suggestion."""
    if not tracer.storage:
        fatal_error("No storage configured", "Cannot load project traces")

    version = getattr(args, "version", None) or None
    project_traces = list(
        tracer.storage.list_calls(limit=500, project=args.project, version=version)
    )

    if not project_traces:
        version_hint = f" (version: {args.version})" if args.version else ""
        fatal_error(
            f"No traces found for project '{args.project}'{version_hint}",
            "Run 'evalyn show-projects' to see available projects",
        )

    traces = project_traces[: args.num_traces]
    if output_format != "json":
        print(f"Found {len(project_traces)} traces for project '{args.project}'")
        if args.version:
            print(f"  (version: {args.version})")

    first_call = project_traces[0]
    function_name = first_call.function_name
    meta = first_call.metadata if isinstance(first_call.metadata, dict) else {}
    function_docstring = meta.get("docstring", "")

    def _placeholder_fn(*_args, **_kwargs):
        pass

    _placeholder_fn.__name__ = function_name
    _placeholder_fn.__doc__ = function_docstring

    return {
        "target_fn": _placeholder_fn,
        "metric_mode_hint": None,
        "metric_bundle_hint": None,
        "traces": traces,
        "function_name": function_name,
    }


def _load_suggest_metrics_target_context(args: argparse.Namespace, tracer) -> dict[str, Any]:
    """Load direct callable and optional recent traces for target-based suggestion."""
    target_fn = _load_callable(args.target)
    return {
        "target_fn": target_fn,
        "metric_mode_hint": getattr(target_fn, "_evalyn_metric_mode", None),
        "metric_bundle_hint": getattr(target_fn, "_evalyn_metric_bundle", None),
        "traces": tracer.storage.list_calls(limit=args.num_traces) if tracer.storage else [],
        "function_name": target_fn.__name__,
    }


def _select_metric_suggestion_mode(
    requested_mode: str, metric_mode_hint: str | None
) -> str:
    """Resolve final metric suggestion mode including decorator hints."""
    if requested_mode != "auto":
        return requested_mode
    return metric_mode_hint or "llm-registry"


def _normalize_suggest_metrics_limit(
    selected_mode: str, requested_limit: int | None
) -> int | None:
    """Normalize metric limit defaults by mode."""
    max_metrics = requested_limit
    if selected_mode == "bundle" and max_metrics == 5:
        return None
    return max_metrics


def _resolve_scope_filter(args: argparse.Namespace, output_format: str) -> str | None:
    """Resolve scope filter string and print context note."""
    scope_raw = getattr(args, "scope", "all")
    scope_filter = None if scope_raw == "all" else scope_raw
    if scope_filter and output_format != "json":
        print(f"Filtering metrics by scope: {scope_filter}")
    return scope_filter


def _filter_metric_templates_by_scope(templates: list, scope_filter: str | None) -> list:
    """Filter metric templates by scope."""
    if not scope_filter:
        return templates
    return [t for t in templates if t.get("scope", "overall") == scope_filter]


def _print_suggested_metric_spec(spec: MetricSpec, output_format: str) -> None:
    """Print one suggested metric in table mode."""
    if output_format == "json":
        return
    why = getattr(spec, "why", "") or ""
    suffix = f" | why: {why}" if why else ""
    print(f"  {icon('info')} {spec.id} [{spec.type}] :: {spec.description}{suffix}")


def _print_suggest_metrics_json(
    specs: list[MetricSpec], saved_path: Path | None = None
) -> None:
    """Print JSON output for suggested metrics."""
    result = {
        "metrics": [
            {
                "id": s.id,
                "type": s.type,
                "name": s.name,
                "description": s.description,
                "config": s.config,
                "why": getattr(s, "why", ""),
            }
            for s in specs
        ],
        "count": len(specs),
        "saved_to": str(saved_path) if saved_path else None,
    }
    print(json.dumps(result, indent=2))


def _finalize_suggest_metrics_output(
    *,
    specs: list[MetricSpec],
    max_metrics: int | None,
    dataset_path_obj: Path | None,
    args: argparse.Namespace,
    selected_mode: str,
    output_format: str,
) -> None:
    """Apply limits, print specs, save, and emit JSON output when requested."""
    final_specs = specs[:max_metrics] if max_metrics else specs
    if output_format != "json":
        print(banner("SUGGESTED METRICS"))
    for spec in final_specs:
        _print_suggested_metric_spec(spec, output_format)
    saved_path = _save_suggested_metrics(
        final_specs,
        dataset_path_obj,
        args.metrics_name,
        getattr(args, "append", False),
        selected_mode,
        output_format,
        getattr(args, "quiet", False),
    )
    if output_format == "json":
        _print_suggest_metrics_json(final_specs, saved_path)


def _print_suggest_metrics_empty_brainstorm(output_format: str) -> None:
    """Print empty-result message for brainstorm mode."""
    if output_format == "json":
        print(json.dumps({"metrics": [], "count": 0, "saved_to": None}))
    else:
        print("No metrics were returned by the LLM (brainstorm mode).")


def _run_suggest_metrics_bundle_mode(
    *,
    args: argparse.Namespace,
    bundle_name: str | None,
    has_reference: bool,
    scope_filter: str | None,
    max_metrics: int | None,
    dataset_path_obj: Path | None,
    output_format: str,
) -> None:
    """Handle bundle-based metric suggestion mode."""
    from ...metrics.objective import OBJECTIVE_REGISTRY
    from ...metrics.subjective import SUBJECTIVE_REGISTRY
    from ...models import MetricSpec

    bundle = (bundle_name or "").lower()
    ids = BUNDLES.get(bundle)
    if not ids:
        available = ", ".join(sorted(BUNDLES.keys()))
        if output_format == "json":
            print(
                json.dumps(
                    {
                        "error": f"No bundle specified or unknown bundle '{bundle_name}'",
                        "available": sorted(BUNDLES.keys()),
                        "hint": "Use --bundle <name> to select a bundle",
                    }
                )
            )
        else:
            if not bundle_name or bundle_name.lower() == "none":
                print(f"No bundle specified. Use --bundle <name> to select one.\n\nAvailable bundles: {available}")
            else:
                print(f"Unknown bundle '{bundle_name}'. Available: {available}")
        return

    all_templates = _filter_metric_templates_by_scope(
        OBJECTIVE_REGISTRY + SUBJECTIVE_REGISTRY, scope_filter
    )
    tpl_map = {t["id"]: t for t in all_templates}
    specs: list[MetricSpec] = []
    skipped_ref_metrics = []

    for mid in ids:
        tpl = tpl_map.get(mid)
        if not tpl:
            continue
        if tpl.get("requires_reference", False) and not has_reference:
            skipped_ref_metrics.append(mid)
            continue
        specs.append(MetricSpec.from_dict(tpl))

    if skipped_ref_metrics and output_format != "json":
        print(
            f"Skipped reference-based metrics (no expected values): {', '.join(skipped_ref_metrics)}"
        )

    _finalize_suggest_metrics_output(
        specs=specs,
        max_metrics=max_metrics,
        dataset_path_obj=dataset_path_obj,
        args=args,
        selected_mode="bundle",
        output_format=output_format,
    )


def _run_suggest_metrics_basic_mode(
    *,
    target_fn: Callable,
    traces: list,
    has_reference: bool,
    scope_filter: str | None,
) -> list[MetricSpec]:
    """Handle heuristic/basic metric suggestion mode."""
    from ...metrics.objective import OBJECTIVE_REGISTRY
    from ...metrics.subjective import SUBJECTIVE_REGISTRY
    from ...metrics.suggester import HeuristicSuggester

    suggester = HeuristicSuggester(has_reference=has_reference)
    specs = suggester.suggest(target_fn, traces)
    if not scope_filter:
        return specs

    all_templates = OBJECTIVE_REGISTRY + SUBJECTIVE_REGISTRY
    template_scope = {t["id"]: t.get("scope", "overall") for t in all_templates}
    return [s for s in specs if template_scope.get(s.id, "overall") == scope_filter]


def cmd_select_metrics(args: argparse.Namespace) -> None:
    """LLM-guided selection from metric registry."""
    from ...decorators import get_default_tracer
    from ...metrics.objective import OBJECTIVE_REGISTRY, register_builtin_metrics
    from ...metrics.subjective import SUBJECTIVE_REGISTRY
    from ...metrics.suggester import LLMRegistrySelector, TemplateSelector
    from ...models import MetricRegistry

    target_fn = _load_callable(args.target)
    tracer = get_default_tracer()
    traces = tracer.storage.list_calls(limit=args.limit) if tracer.storage else []

    if args.llm_caller:
        caller = _load_callable(args.llm_caller)
        selector = TemplateSelector(caller, OBJECTIVE_REGISTRY + SUBJECTIVE_REGISTRY)
        selected = selector.select(
            target_fn, traces=traces, code_meta=_extract_code_meta(tracer, target_fn)
        )
    else:
        # fallback to heuristic/built-in registry selection
        registry = MetricRegistry()
        register_builtin_metrics(registry)
        selector = LLMRegistrySelector(lambda prompt: [])
        selected = registry.list()

    print("Selected metrics:")
    for spec in selected:
        print(f"- {spec.id}: [{spec.type}] config={getattr(spec, 'config', {})}")


def cmd_list_metrics(args: argparse.Namespace) -> None:
    """List available metric templates (objective + subjective)."""
    from ...metrics.objective import OBJECTIVE_REGISTRY
    from ...metrics.subjective import SUBJECTIVE_REGISTRY

    output_format = getattr(args, "format", "table")

    # JSON output mode
    if output_format == "json":
        result = {
            "objective": OBJECTIVE_REGISTRY,
            "subjective": SUBJECTIVE_REGISTRY,
            "objective_count": len(OBJECTIVE_REGISTRY),
            "subjective_count": len(SUBJECTIVE_REGISTRY),
        }
        print(json.dumps(result, indent=2))
        return

    def _compact_text(value: Any, max_len: int = 55) -> str:
        text = str(value or "")
        text = re.sub(r"\s+", " ", text).strip()
        return text if len(text) <= max_len else text[: max_len - 3] + "..."

    def _build_rows(templates: list[dict]) -> list[list[str]]:
        rows = []
        for tpl in templates:
            rows.append(
                [
                    tpl.get("id", ""),
                    tpl.get("scope", "overall"),
                    tpl.get("category", ""),
                    _compact_text(tpl.get("description", ""), 50),
                ]
            )
        return rows

    headers = ["ID", "Scope", "Category", "Description"]
    align = ["left", "left", "left", "left"]

    print(banner(f"OBJECTIVE METRICS ({len(OBJECTIVE_REGISTRY)})"))
    print(rich_table(headers, _build_rows(OBJECTIVE_REGISTRY), align))

    print(section(f"SUBJECTIVE METRICS ({len(SUBJECTIVE_REGISTRY)})"))
    print(rich_table(headers, _build_rows(SUBJECTIVE_REGISTRY), align))


def register_commands(subparsers) -> None:
    """Register evaluation commands."""
    from ...defaults import DEFAULT_EVAL_MODEL

    # run-eval
    run_parser = subparsers.add_parser(
        "run-eval", help="Run evaluation on dataset using specified metrics"
    )
    run_parser.add_argument(
        "--dataset",
        help="Path to JSON/JSONL dataset file or directory containing dataset.jsonl",
    )
    run_parser.add_argument(
        "--latest", action="store_true", help="Use the most recently modified dataset"
    )
    run_parser.add_argument(
        "--metrics",
        help="Path to metrics JSON file(s), comma-separated for multiple (auto-detected from meta.json if omitted)",
    )
    run_parser.add_argument(
        "--metrics-all",
        action="store_true",
        help="Use all metrics files from the metrics/ folder",
    )
    run_parser.add_argument(
        "--use-calibrated",
        action="store_true",
        help="Use calibrated prompts for subjective metrics (if available)",
    )
    run_parser.add_argument(
        "--dataset-name", help="Name for the eval run (defaults to dataset filename)"
    )
    run_parser.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    run_parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=4,
        help="Parallel workers for LLM evaluation (default: 4, max: 16)",
    )
    run_parser.add_argument(
        "--batch",
        action="store_true",
        help="Use batch API for 50%% cost savings (async, may take minutes to hours)",
    )
    run_parser.add_argument(
        "--batch-provider",
        choices=["gemini", "openai", "anthropic"],
        default="gemini",
        help="Batch API provider (default: gemini)",
    )
    run_parser.add_argument(
        "--provider",
        choices=["gemini", "openai", "ollama"],
        default="gemini",
        help="LLM provider for judges: gemini (default), openai, ollama",
    )
    run_parser.add_argument(
        "--confidence",
        choices=["none", "consistency", "logprobs", "deepconf"],
        default="none",
        help="Confidence method: none (default), consistency (N samples), logprobs/deepconf (openai/ollama only)",
    )
    def _positive_int(value):
        ivalue = int(value)
        if ivalue < 1:
            raise argparse.ArgumentTypeError(f"--confidence-samples must be >= 1, got {value}")
        return ivalue

    run_parser.add_argument(
        "--confidence-samples",
        type=_positive_int,
        default=3,
        help="Number of samples for consistency method (default: 3, must be >= 1)",
    )
    run_parser.add_argument(
        "--unit-types",
        help="Evaluation unit types (comma-separated): outcome, single_turn, tool_use, multi_turn, custom. Default: outcome (full trace)",
    )
    run_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed cost breakdown by metric",
    )
    run_parser.set_defaults(func=cmd_run_eval)

    # suggest-metrics
    suggest_parser = subparsers.add_parser(
        "suggest-metrics", help="Suggest metrics for a project or target function"
    )
    suggest_parser.add_argument(
        "--project",
        help="Project name (use 'evalyn show-projects' to see available projects)",
    )
    suggest_parser.add_argument(
        "--version", help="Filter by version (optional, used with --project)"
    )
    suggest_parser.add_argument(
        "--target",
        help="Callable to analyze in the form module:function (alternative to --project)",
    )
    suggest_parser.add_argument(
        "--num-traces",
        type=int,
        default=5,
        help="How many recent traces to include as examples",
    )
    suggest_parser.add_argument(
        "-n",
        "--num-metrics",
        type=int,
        default=5,
        help="Maximum number of metrics to return (default: 5)",
    )
    suggest_parser.add_argument(
        "--mode",
        choices=["auto", "llm-registry", "llm-brainstorm", "bundle", "basic"],
        default="auto",
        help="auto (use function's metric_mode if set), llm-registry (LLM picks from registry), llm-brainstorm (free-form), bundle (preset), or basic heuristic.",
    )
    suggest_parser.add_argument(
        "--llm-mode",
        choices=["local", "api"],
        default="api",
        help="When using LLM modes: choose local (ollama) or api (OpenAI/Gemini-compatible).",
    )
    suggest_parser.add_argument(
        "--model",
        default=DEFAULT_EVAL_MODEL,
        help=f"Model name (default: {DEFAULT_EVAL_MODEL})",
    )
    suggest_parser.add_argument(
        "--api-base", help="Custom API base URL for --llm-mode api (optional)"
    )
    suggest_parser.add_argument(
        "--api-key", help="API key override for --llm-mode api (optional)"
    )
    suggest_parser.add_argument(
        "--llm-caller",
        help="Optional callable path that accepts a prompt string and returns a list of metric dicts",
    )
    suggest_parser.add_argument(
        "--bundle",
        help="Bundle name when --mode bundle (e.g., summarization, orchestrator, research-agent)",
    )
    suggest_parser.add_argument(
        "--dataset",
        help="Dataset directory (or dataset.jsonl/meta.json) to save metrics into",
    )
    suggest_parser.add_argument(
        "--latest",
        action="store_true",
        help="Use the most recently modified dataset",
    )
    suggest_parser.add_argument(
        "--metrics-name", help="Metrics set name when saving to a dataset"
    )
    suggest_parser.add_argument(
        "--scope",
        choices=["all", "overall", "llm_call", "tool_call", "trace"],
        default="all",
        help="Filter metrics by scope: overall (final output), llm_call (per LLM), tool_call (per tool), trace (aggregates), or all.",
    )
    suggest_parser.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    suggest_parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing metrics.json instead of overwriting. Duplicates by ID are skipped.",
    )
    suggest_parser.set_defaults(func=cmd_suggest_metrics)

    # select-metrics
    select_parser = subparsers.add_parser(
        "select-metrics", help="LLM-guided selection from metric registry"
    )
    select_parser.add_argument(
        "--target",
        required=True,
        help="Callable to analyze in the form module:function",
    )
    select_parser.add_argument(
        "--llm-caller",
        required=True,
        help="Callable that accepts a prompt and returns metric ids or dicts",
    )
    select_parser.add_argument(
        "--limit", type=int, default=5, help="Recent traces to include as examples"
    )
    select_parser.set_defaults(func=cmd_select_metrics)

    # list-metrics
    list_metrics = subparsers.add_parser(
        "list-metrics", help="List available metric templates (objective + subjective)"
    )
    list_metrics.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    list_metrics.set_defaults(func=cmd_list_metrics)


__all__ = [
    "cmd_run_eval",
    "cmd_suggest_metrics",
    "cmd_select_metrics",
    "cmd_list_metrics",
    "register_commands",
]
