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

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ...datasets import load_dataset
from ...decorators import get_default_tracer
from ...metrics.objective import OBJECTIVE_REGISTRY
from ...metrics.subjective import SUBJECTIVE_REGISTRY
from ..constants import BUNDLES
from ...metrics.suggester import (
    HeuristicSuggester,
    LLMSuggester,
    TemplateSelector,
    LLMRegistrySelector,
)
from ...models import MetricRegistry, MetricSpec
from ..utils.config import load_config, get_config_default, resolve_dataset_path
from ..utils.hints import print_hint
from ..utils.loaders import _load_callable
from ..utils.validation import check_llm_api_keys
from ..utils.dataset_utils import (
    ProgressBar,
    _resolve_dataset_and_metrics,
    _dataset_has_reference,
    _extract_code_meta,
)


def _build_llm_caller(args: argparse.Namespace) -> Callable:
    """Build an LLM caller from args."""
    from ...utils.api_client import call_gemini_api

    # Custom caller if provided
    if args.llm_caller:
        return _load_callable(args.llm_caller)

    # Build default caller
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY", "")
    model = args.model or "gemini-2.5-flash-lite"

    def default_caller(prompt: str) -> str:
        return call_gemini_api(prompt, model=model, api_key=api_key)

    return default_caller


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
    metrics_all = getattr(args, "metrics_all", False)
    config = load_config()

    # Resolve dataset path using --dataset, --latest, or config
    dataset_arg = getattr(args, "dataset", None)
    use_latest = getattr(args, "latest", False)
    dataset_path = resolve_dataset_path(dataset_arg, use_latest, config)

    if not dataset_path:
        print("Error: No dataset specified. Use --dataset <path> or --latest")
        sys.exit(1)

    # Resolve dataset and metrics paths
    try:
        dataset_file, metrics_paths = _resolve_dataset_and_metrics(
            str(dataset_path), args.metrics, metrics_all=metrics_all
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Load dataset
    dataset = load_dataset(str(dataset_file))
    dataset_list = list(dataset)

    # Load and merge metrics from all files (deduplicate by ID)
    all_metrics_data: Dict[str, dict] = {}  # id -> spec_data
    duplicate_ids: List[str] = []  # Track duplicates for warning
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

    # Warn about duplicate metric IDs
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
        print("Error: No valid metrics loaded from files")
        sys.exit(1)

    if output_format != "json" and len(metrics_paths) > 1:
        print(
            f"Merged {len(all_metrics_data)} unique metrics from {len(metrics_paths)} files"
        )

    # Build metrics from specs
    from ...metrics.factory import build_objective_metric, build_subjective_metric
    from ...annotation import load_optimized_prompt

    metrics = []
    objective_count = 0
    subjective_count = 0
    calibrated_count = 0

    # Get API key from config for LLM judges
    gemini_api_key = get_config_default(config, "api_keys", "gemini")

    # Check if we should use calibrated prompts
    use_calibrated = getattr(args, "use_calibrated", False)
    dataset_dir = dataset_file.parent

    skipped_metrics = []
    for spec_data in all_metrics_data.values():
        spec = MetricSpec(
            id=spec_data["id"],
            name=spec_data.get("name", spec_data["id"]),
            type=spec_data["type"],
            description=spec_data.get("description", ""),
            config=spec_data.get("config", {}),
        )

        # Load calibrated prompt for subjective metrics if --use-calibrated is set
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
                    print(
                        f"  Warning: Could not load calibrated prompt for {spec.id}: {e}"
                    )

        try:
            if spec.type == "objective":
                metric = build_objective_metric(spec.id, spec.config)
                objective_count += 1
            else:
                metric = build_subjective_metric(
                    spec.id,
                    spec.config,
                    description=spec.description,
                    api_key=gemini_api_key,
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
        print("Error: No valid metrics loaded from file")
        sys.exit(1)

    if output_format != "json":
        metrics_summary = f"Loaded {len(metrics)} metrics ({objective_count} objective, {subjective_count} subjective"
        if calibrated_count > 0:
            metrics_summary += f", {calibrated_count} calibrated"
        metrics_summary += ")"
        print(metrics_summary)
        print(f"Dataset: {len(dataset_list)} items")

        # Check for API key if subjective metrics are present
        if subjective_count > 0:
            check_llm_api_keys(quiet=False)

        print()

    # Get tracer with storage
    tracer = get_default_tracer()
    if not tracer.storage:
        print("Error: No storage configured")
        sys.exit(1)

    # Create progress bar
    total_evals = len(dataset_list) * len(metrics)
    progress = ProgressBar(total_evals) if output_format != "json" else None

    def progress_callback(
        current: int, total: int, metric: str, metric_type: str
    ) -> None:
        if progress:
            progress.update(current, total, metric, metric_type)

    # Run evaluation using cached traces
    from ...runner import EvalRunner, save_eval_run_json

    # Checkpoint path for long-running evaluations
    checkpoint_path = dataset_dir / ".eval_checkpoint.json"

    # Check for existing checkpoint
    if checkpoint_path.exists():
        if output_format != "json":
            print("  Resuming from checkpoint...")

    runner = EvalRunner(
        target_fn=lambda: None,  # Dummy function, won't be called
        metrics=metrics,
        dataset_name=args.dataset_name or dataset_file.stem,
        tracer=tracer,
        instrument=False,
        progress_callback=progress_callback if output_format != "json" else None,
        checkpoint_path=checkpoint_path,
        checkpoint_interval=5,
        max_workers=getattr(args, "workers", 1),
    )
    run = runner.run_dataset(dataset_list, use_synthetic=True)

    if progress:
        progress.finish()

    # Save eval run in dedicated folder
    run_folder = save_eval_run_json(run, dataset_dir)
    results_path = run_folder / "results.json"

    # Generate HTML analysis report
    from ...analysis import (
        analyze_run as analyze_run_data,
        generate_html_report,
        load_eval_run,
    )

    try:
        run_data = load_eval_run(results_path)
        analysis = analyze_run_data(run_data)

        # Build item_details from dataset
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

    # JSON output
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

    # Table output
    print(f"\nEval run {run.id}")
    print(f"Dataset: {run.dataset_name}")
    print(f"Run folder: {run_folder}")
    print("  results.json - evaluation data")
    if report_path:
        print("  report.html  - analysis report")
    print()

    # Build metric type lookup
    metric_types = {m.spec.id: m.spec.type for m in metrics}

    # Check for API errors in results
    api_errors_by_metric: Dict[str, int] = {}
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

    # Show API error summary if any
    total_api_errors = sum(api_errors_by_metric.values())
    if total_api_errors > 0:
        print(f"\n{total_api_errors} API error(s) detected in LLM judge results.")
        print("   Check that GEMINI_API_KEY or OPENAI_API_KEY is set and valid.")
        print("   Re-run with a valid API key to get accurate LLM judge scores.")

    if run.summary.get("failed_items"):
        print(f"Failed items: {run.summary['failed_items']}")

    # Show hint for next step
    print_hint(
        f"To analyze results, run: evalyn analyze --run {run.id}",
        quiet=getattr(args, "quiet", False),
        format=getattr(args, "format", "table"),
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
    output_format = getattr(args, "format", "table")
    tracer = get_default_tracer()

    # If --dataset provided but not --project/--target, try to infer project from meta.json
    if not args.project and not args.target and args.dataset:
        dataset_path = Path(args.dataset)
        if dataset_path.is_file():
            dataset_path = dataset_path.parent
        meta_file = dataset_path / "meta.json"
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                if meta.get("project"):
                    args.project = meta["project"]
            except Exception:
                pass

    # Validate: need either --project or --target
    if not args.project and not args.target:
        print("Error: Either --project or --target is required.", file=sys.stderr)
        print("\nUsage:", file=sys.stderr)
        print(
            "  evalyn suggest-metrics --project <name>    # Suggest based on project traces",
            file=sys.stderr,
        )
        print(
            "  evalyn suggest-metrics --target <path>     # Suggest based on function code",
            file=sys.stderr,
        )
        print(
            "  evalyn suggest-metrics --dataset <path>    # Infer project from dataset meta.json",
            file=sys.stderr,
        )
        print("\nTo see available projects:", file=sys.stderr)
        print("  evalyn show-projects", file=sys.stderr)
        sys.exit(1)

    # Resolve dataset path using --dataset or --latest
    config = load_config()
    dataset_path_obj: Optional[Path] = None
    use_latest = getattr(args, "latest", False)

    if args.dataset or use_latest:
        resolved = resolve_dataset_path(args.dataset, use_latest, config)
        if resolved:
            dataset_path_obj = resolved
        elif args.dataset:
            # Try as direct path if resolve failed
            dataset_path_obj = Path(args.dataset)

    # Validate dataset path if provided
    if dataset_path_obj:
        if dataset_path_obj.is_file():
            if not dataset_path_obj.exists():
                print(
                    f"Error: Dataset file not found: {dataset_path_obj}",
                    file=sys.stderr,
                )
                print(
                    "Please create the dataset first using 'evalyn build-dataset' or ensure the path is correct.",
                    file=sys.stderr,
                )
                sys.exit(1)
        else:
            if not dataset_path_obj.exists():
                print(
                    f"Error: Dataset directory not found: {dataset_path_obj}",
                    file=sys.stderr,
                )
                print(
                    "Please create the dataset first using 'evalyn build-dataset' or ensure the path is correct.",
                    file=sys.stderr,
                )
                sys.exit(1)
            has_dataset = (dataset_path_obj / "dataset.jsonl").exists() or (
                dataset_path_obj / "dataset.json"
            ).exists()
            if not has_dataset and output_format != "json":
                print(
                    f"Warning: Directory exists but no dataset.jsonl or dataset.json found in: {dataset_path_obj}"
                )

    # Check if dataset has reference values
    has_reference = _dataset_has_reference(dataset_path_obj)
    if dataset_path_obj and not has_reference and output_format != "json":
        print(
            "Note: Dataset has no reference/expected values. Reference-based metrics (ROUGE, BLEU, etc.) excluded."
        )

    # Load traces and function info based on --project or --target
    target_fn = None
    metric_mode_hint = None
    metric_bundle_hint = None
    traces = []
    function_name = "unknown"

    if args.project:
        # Project-based: load traces from storage
        if not tracer.storage:
            print(
                "Error: No storage configured. Cannot load project traces.",
                file=sys.stderr,
            )
            sys.exit(1)

        # Load all calls for this project
        all_calls = tracer.storage.list_calls(limit=500)
        project_traces = []
        for call in all_calls:
            meta = call.metadata if isinstance(call.metadata, dict) else {}
            call_project = (
                meta.get("project_id") or meta.get("project_name") or call.function_name
            )
            call_version = meta.get("version") or ""

            if call_project == args.project:
                if args.version and call_version != args.version:
                    continue
                project_traces.append(call)

        if not project_traces:
            print(
                f"Error: No traces found for project '{args.project}'", file=sys.stderr
            )
            if args.version:
                print(f"  (filtered by version: {args.version})", file=sys.stderr)
            print("\nAvailable projects:", file=sys.stderr)
            print("  evalyn show-projects", file=sys.stderr)
            sys.exit(1)

        traces = project_traces[: args.num_traces]
        if output_format != "json":
            print(f"Found {len(project_traces)} traces for project '{args.project}'")
            if args.version:
                print(f"  (version: {args.version})")

        # Extract function info from the first trace
        first_call = project_traces[0]
        function_name = first_call.function_name
        meta = first_call.metadata if isinstance(first_call.metadata, dict) else {}
        function_docstring = meta.get("docstring", "")

        # Create a placeholder function for the suggester
        def _placeholder_fn(*a, **kw):
            pass

        _placeholder_fn.__name__ = function_name
        _placeholder_fn.__doc__ = function_docstring
        target_fn = _placeholder_fn

    else:
        # Target-based: load callable directly
        target_fn = _load_callable(args.target)
        metric_mode_hint = getattr(target_fn, "_evalyn_metric_mode", None)
        metric_bundle_hint = getattr(target_fn, "_evalyn_metric_bundle", None)
        traces = (
            tracer.storage.list_calls(limit=args.num_traces) if tracer.storage else []
        )
        function_name = target_fn.__name__

    selected_mode = (
        metric_mode_hint or "llm-registry" if args.mode == "auto" else args.mode
    )
    bundle_name = args.bundle or metric_bundle_hint
    max_metrics = args.num_metrics

    # Get scope filter (None means "all")
    scope_filter = getattr(args, "scope", "all")
    scope_filter = None if scope_filter == "all" else scope_filter

    def _filter_by_scope(templates: list) -> list:
        """Filter templates by scope."""
        if not scope_filter:
            return templates
        return [t for t in templates if t.get("scope", "overall") == scope_filter]

    if scope_filter and output_format != "json":
        print(f"Filtering metrics by scope: {scope_filter}")

    def _print_spec(spec: MetricSpec) -> None:
        if output_format == "json":
            return
        why = getattr(spec, "why", "") or ""
        suffix = f" | why: {why}" if why else ""
        print(f"- {spec.id} [{spec.type}] :: {spec.description}{suffix}")

    def _output_json(
        specs: List[MetricSpec], saved_path: Optional[Path] = None
    ) -> None:
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

    def _finalize_output(specs: List[MetricSpec]) -> None:
        """Apply limit, print, save, and output JSON if needed."""
        nonlocal max_metrics
        final_specs = specs[:max_metrics] if max_metrics else specs
        for spec in final_specs:
            _print_spec(spec)
        saved_path = _save_metrics(final_specs)
        if output_format == "json":
            _output_json(final_specs, saved_path)

    def _save_metrics(specs: List[MetricSpec]) -> Optional[Path]:
        if not dataset_path_obj:
            return None
        # Dataset path already validated and resolved (via --dataset or --latest)
        dataset_dir = (
            dataset_path_obj.parent if dataset_path_obj.is_file() else dataset_path_obj
        )

        metrics_dir = dataset_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)

        # Standardized naming: metrics.json (default) or metrics-<name>.json
        if args.metrics_name:
            safe_name = re.sub(r"[^a-zA-Z0-9._-]+", "-", args.metrics_name).strip("-")
        else:
            safe_name = "default"
        metrics_file = metrics_dir / (
            f"metrics-{safe_name}.json" if args.metrics_name else "metrics.json"
        )

        # Handle --append: merge with existing metrics, skip duplicates
        existing_metrics: List[dict] = []
        if getattr(args, "append", False) and metrics_file.exists():
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
            s
            for s in specs
            if s.type == "objective" and s.id not in valid_objective_ids
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
            print(f"Saved metrics to {metrics_file}")
            print_hint(
                f"To run evaluation, run: evalyn run-eval --dataset {dataset_dir}",
                quiet=getattr(args, "quiet", False),
            )
        return metrics_file

    if selected_mode == "bundle":
        bundle = (bundle_name or "").lower()
        ids = BUNDLES.get(bundle)
        if not ids:
            if output_format == "json":
                print(
                    json.dumps(
                        {
                            "error": f"Unknown bundle '{args.bundle}'",
                            "available": list(BUNDLES.keys()),
                        }
                    )
                )
            else:
                print(
                    f"Unknown bundle '{args.bundle}'. Available: {', '.join(BUNDLES.keys())}"
                )
            return
        all_templates = _filter_by_scope(OBJECTIVE_REGISTRY + SUBJECTIVE_REGISTRY)
        tpl_map = {t["id"]: t for t in all_templates}
        specs = []
        skipped_ref_metrics = []
        for mid in ids:
            tpl = tpl_map.get(mid)
            if tpl:
                # Skip reference-based metrics if no reference available
                if tpl.get("requires_reference", False) and not has_reference:
                    skipped_ref_metrics.append(mid)
                    continue
                specs.append(
                    MetricSpec(
                        id=tpl["id"],
                        name=tpl["id"],
                        type=tpl["type"],
                        description=tpl.get("description", ""),
                        config=tpl.get("config", {}),
                    )
                )
        if skipped_ref_metrics and output_format != "json":
            print(
                f"Skipped reference-based metrics (no expected values): {', '.join(skipped_ref_metrics)}"
            )
        _finalize_output(specs)
        return

    if selected_mode == "llm-registry":
        caller = _build_llm_caller(args)
        filtered_templates = _filter_by_scope(OBJECTIVE_REGISTRY + SUBJECTIVE_REGISTRY)
        selector = TemplateSelector(
            caller, filtered_templates, has_reference=has_reference
        )
        specs = selector.select(
            target_fn,
            traces=traces,
            code_meta=_extract_code_meta(tracer, target_fn),
            desired_count=max_metrics,
        )
        _finalize_output(specs)
        return

    if selected_mode == "llm-brainstorm":
        caller = _build_llm_caller(args)
        suggester = LLMSuggester(caller=caller)
        specs = suggester.suggest(
            target_fn, traces, desired_count=max_metrics, scope=scope_filter
        )
        if not specs:
            if output_format == "json":
                print(json.dumps({"metrics": [], "count": 0, "saved_to": None}))
            else:
                print("No metrics were returned by the LLM (brainstorm mode).")
            return
        _finalize_output(specs)
        return

    # Default: basic heuristic mode
    suggester = HeuristicSuggester(has_reference=has_reference)
    specs = suggester.suggest(target_fn, traces)

    # Filter by scope if specified
    if scope_filter:
        all_templates = OBJECTIVE_REGISTRY + SUBJECTIVE_REGISTRY
        template_scope = {t["id"]: t.get("scope", "overall") for t in all_templates}
        specs = [
            s for s in specs if template_scope.get(s.id, "overall") == scope_filter
        ]

    _finalize_output(specs)


def cmd_select_metrics(args: argparse.Namespace) -> None:
    """LLM-guided selection from metric registry."""
    from ...metrics.suggester import TemplateSelector
    from ...models import register_builtin_metrics

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

    def _compact(value, max_len: int = 60) -> str:
        try:
            text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            text = str(value)
        return text if len(text) <= max_len else text[: max_len - 3] + "..."

    def _compact_text(value: Any, max_len: int = 55) -> str:
        text = str(value or "")
        text = re.sub(r"\s+", " ", text).strip()
        return text if len(text) <= max_len else text[: max_len - 3] + "..."

    def _config_summary(cfg: Any) -> str:
        if not cfg:
            return "{}"
        if not isinstance(cfg, dict):
            return _compact(cfg, 55)
        parts: list[str] = []
        for key in sorted(cfg.keys()):
            val = cfg.get(key)
            if key == "rubric" and isinstance(val, list):
                parts.append(f"rubric[{len(val)}]")
                continue
            if key == "policy" and isinstance(val, str):
                parts.append(f"policy[{len(val)}]")
                continue
            if key == "schema" and isinstance(val, dict):
                parts.append(f"schema[{len(val)}]")
                continue
            if isinstance(val, (str, int, float, bool)) or val is None:
                parts.append(f"{key}={_compact_text(val, 18)}")
            else:
                parts.append(f"{key}=...")
        text = ", ".join(parts)
        return text if text else "{}"

    def _print_table(title: str, templates: list[dict]) -> None:
        print(title)
        headers = ["id", "scope", "category", "config", "desc"]
        rows = []
        for tpl in templates:
            rows.append(
                [
                    tpl.get("id", ""),
                    tpl.get("scope", "overall"),
                    tpl.get("category", ""),
                    _config_summary(tpl.get("config", {})),
                    _compact_text(tpl.get("description", ""), 50),
                ]
            )

        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(str(cell)))

        header_line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
        print(header_line)
        print("-" * len(header_line))
        for row in rows:
            print(" | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))

    _print_table("\nObjective metrics:", OBJECTIVE_REGISTRY)
    _print_table("\nSubjective metrics:", SUBJECTIVE_REGISTRY)


def register_commands(subparsers) -> None:
    """Register evaluation commands."""
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
        default="gemini-2.5-flash-lite",
        help="Model name (e.g., gemini-2.5-flash-lite, gpt-4, llama3.1 for Ollama)",
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
