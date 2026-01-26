"""Calibration commands: calibrate, list-calibrations.

This module provides CLI commands for calibrating LLM judges using human annotations.
Calibration analyzes disagreements between LLM judgments and human labels,
then optimizes the judge's prompt to improve alignment.

Commands:
- calibrate: Calibrate a subjective metric using human annotations
- list-calibrations: List calibration records for a dataset

Optimization methods:
- basic (default): Single-shot LLM analysis of disagreement patterns
- gepa: Uses GEPA evolutionary algorithm for systematic prompt optimization
- opro: Uses OPRO (Optimization by PROmpting) trajectory-based optimization
- ape: Uses APE (Automatic Prompt Engineer) search-based optimization with UCB selection

Calibration outputs:
- Alignment metrics: Accuracy, precision, recall, F1, Cohen's Kappa
- Confusion matrix: True/false positives/negatives
- Suggested threshold: Optimal score cutoff for pass/fail
- Optimized prompt: Improved preamble and rubric (saved to calibrations/ folder)

Typical workflow:
1. Annotate items: 'evalyn annotate --dataset <path> --per-metric'
2. Calibrate: 'evalyn calibrate --metric-id <id> --annotations <path> --dataset <path>'
3. Re-evaluate: 'evalyn run-eval --dataset <path> --use-calibrated'
4. Compare: 'evalyn compare --run1 <before> --run2 <after>'
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

from ...annotation import import_annotations
from ...annotation import (
    CalibrationEngine,
    GEPAConfig,
    GEPA_AVAILABLE,
    GEPANativeConfig,
    OPROConfig,
    APEConfig,
    save_calibration,
)
from ...datasets import load_dataset
from ...decorators import get_default_tracer
from ...models import DatasetItem
from ..utils.config import load_config, resolve_dataset_path, get_config_default
from ..utils.errors import fatal_error
from ..utils.formatters import print_token_usage_summary
from ..utils.hints import print_hint
from ..utils.ui import Spinner


def _apply_calibration_config_defaults(args: argparse.Namespace, config: dict) -> None:
    """Apply config file defaults to calibration args."""
    # Basic calibration settings
    if args.optimizer == "basic":  # Only override if still at default
        args.optimizer = get_config_default(
            config, "calibration", "optimizer", default="basic"
        )
    if args.threshold == 0.5:
        args.threshold = get_config_default(
            config, "calibration", "threshold", default=0.5
        )

    # GEPA settings
    if args.gepa_task_lm == "gemini/gemini-2.5-flash":
        args.gepa_task_lm = get_config_default(
            config, "calibration", "gepa", "task_lm", default="gemini/gemini-2.5-flash"
        )
    if args.gepa_reflection_lm == "gemini/gemini-2.5-flash":
        args.gepa_reflection_lm = get_config_default(
            config,
            "calibration",
            "gepa",
            "reflection_lm",
            default="gemini/gemini-2.5-flash",
        )
    if args.gepa_max_calls == 150:
        args.gepa_max_calls = get_config_default(
            config, "calibration", "gepa", "max_calls", default=150
        )

    # OPRO settings
    if args.opro_optimizer_model == "gemini-2.5-flash":
        args.opro_optimizer_model = get_config_default(
            config, "calibration", "opro", "optimizer_model", default="gemini-2.5-flash"
        )
    if args.opro_scorer_model == "gemini-2.5-flash-lite":
        args.opro_scorer_model = get_config_default(
            config,
            "calibration",
            "opro",
            "scorer_model",
            default="gemini-2.5-flash-lite",
        )
    if args.opro_iterations == 10:
        args.opro_iterations = get_config_default(
            config, "calibration", "opro", "iterations", default=10
        )
    if args.opro_candidates == 4:
        args.opro_candidates = get_config_default(
            config, "calibration", "opro", "candidates", default=4
        )

    # APE settings
    if args.ape_candidates == 10:
        args.ape_candidates = get_config_default(
            config, "calibration", "ape", "candidates", default=10
        )
    if args.ape_rounds == 5:
        args.ape_rounds = get_config_default(
            config, "calibration", "ape", "rounds", default=5
        )
    if args.ape_samples == 5:
        args.ape_samples = get_config_default(
            config, "calibration", "ape", "samples", default=5
        )

    # GEPA-Native settings
    if args.gepa_native_task_model == "gemini-2.5-flash":
        args.gepa_native_task_model = get_config_default(
            config, "calibration", "gepa_native", "task_model", default="gemini-2.5-flash"
        )
    if args.gepa_native_reflection_model == "gemini-2.5-flash":
        args.gepa_native_reflection_model = get_config_default(
            config, "calibration", "gepa_native", "reflection_model", default="gemini-2.5-flash"
        )
    if args.gepa_native_max_calls == 150:
        args.gepa_native_max_calls = get_config_default(
            config, "calibration", "gepa_native", "max_calls", default=150
        )
    if args.gepa_native_initial_candidates == 5:
        args.gepa_native_initial_candidates = get_config_default(
            config, "calibration", "gepa_native", "initial_candidates", default=5
        )
    if args.gepa_native_batch_size == 5:
        args.gepa_native_batch_size = get_config_default(
            config, "calibration", "gepa_native", "batch_size", default=5
        )


def cmd_calibrate(args: argparse.Namespace) -> None:
    """Calibrate a subjective metric using human annotations.

    Calibration process:
    1. Load metric results from an eval run (LLM judge verdicts)
    2. Load human annotations (ground truth labels)
    3. Calculate alignment metrics (accuracy, F1, Cohen's Kappa)
    4. Analyze disagreement patterns (false positives/negatives)
    5. Optimize the judge prompt to reduce disagreements
    6. Validate the optimized prompt against held-out samples
    7. Save calibration record and optimized prompt

    Optimization methods:
    - basic: Single-shot LLM analysis (fast, simple)
    - gepa: Evolutionary algorithm for systematic optimization (slower but thorough)
    """
    tracer = get_default_tracer()
    if not tracer.storage:
        fatal_error("No storage configured")

    run = tracer.storage.get_eval_run(args.run_id) if args.run_id else None
    if run is None:
        runs = tracer.storage.list_eval_runs(limit=1)
        run = runs[0] if runs else None
    if run is None:
        fatal_error("No eval runs available")

    metric_results = [r for r in run.metric_results if r.metric_id == args.metric_id]
    if not metric_results:
        fatal_error(
            f"No metric results found for metric_id={args.metric_id} in run {run.id}"
        )

    # Load annotations
    anns = import_annotations(args.annotations)

    # Get current rubric and preamble from metric config if available
    current_rubric: List[str] = []
    current_preamble: str = ""
    for metric_spec in run.metrics:
        if metric_spec.id == args.metric_id:
            cfg = metric_spec.config or {}
            # Extract rubric (evaluation criteria)
            rubric_val = cfg.get("rubric", [])
            if isinstance(rubric_val, list):
                current_rubric = [str(r) for r in rubric_val]
            # Extract preamble (base prompt before rubric)
            preamble_val = cfg.get("prompt", "")
            if isinstance(preamble_val, str):
                current_preamble = preamble_val
            break

    # Load config and apply defaults
    config = load_config()
    _apply_calibration_config_defaults(args, config)

    # Load dataset items for context (if dataset path provided)
    dataset_items: Optional[List[DatasetItem]] = None
    dataset_dir: Optional[Path] = None

    # Resolve dataset path using --dataset, --latest, or config
    dataset_arg = getattr(args, "dataset", None)
    use_latest = getattr(args, "latest", False)
    resolved_dataset = resolve_dataset_path(dataset_arg, use_latest, config)

    if resolved_dataset:
        dataset_dir = Path(resolved_dataset)
        dataset_file = (
            dataset_dir / "dataset.jsonl" if dataset_dir.is_dir() else dataset_dir
        )
        if dataset_file.exists():
            dataset_items = load_dataset(dataset_file)
            if dataset_dir.is_file():
                dataset_dir = dataset_dir.parent

    # Build GEPA config if using GEPA optimizer
    gepa_config = None
    if args.optimizer == "gepa":
        if not GEPA_AVAILABLE:
            fatal_error("GEPA is not installed", "Install with: pip install gepa")
        gepa_config = GEPAConfig(
            task_lm=args.gepa_task_lm,
            reflection_lm=args.gepa_reflection_lm,
            max_metric_calls=args.gepa_max_calls,
        )

    # Build OPRO config if using OPRO optimizer
    opro_config = None
    if args.optimizer == "opro":
        opro_config = OPROConfig(
            optimizer_model=args.opro_optimizer_model,
            scorer_model=args.opro_scorer_model,
            max_iterations=args.opro_iterations,
            candidates_per_step=args.opro_candidates,
        )

    # Build APE config if using APE optimizer
    ape_config = None
    if args.optimizer == "ape":
        ape_config = APEConfig(
            num_candidates=args.ape_candidates,
            eval_rounds=args.ape_rounds,
            eval_samples_per_round=args.ape_samples,
        )

    # Build GEPA-Native config if using native implementation
    gepa_native_config = None
    if args.optimizer == "gepa-native":
        gepa_native_config = GEPANativeConfig(
            task_model=args.gepa_native_task_model,
            reflection_model=args.gepa_native_reflection_model,
            max_metric_calls=args.gepa_native_max_calls,
            num_initial_candidates=args.gepa_native_initial_candidates,
            mini_batch_size=args.gepa_native_batch_size,
        )

    # Run enhanced calibration
    engine = CalibrationEngine(
        judge_name=args.metric_id,
        current_threshold=args.threshold,
        current_rubric=current_rubric,
        current_preamble=current_preamble,
        optimize_prompts=not args.no_optimize,
        optimizer_model=args.model,
        optimizer_type=args.optimizer,
        gepa_config=gepa_config,
        gepa_native_config=gepa_native_config,
        opro_config=opro_config,
        ape_config=ape_config,
    )

    # Use spinner for long-running operations (GEPA, OPRO, APE, or GEPA-Native)
    if args.optimizer == "gepa" and not args.no_optimize:
        spinner_msg = f"Running GEPA optimization (max {args.gepa_max_calls} calls)"
        with Spinner(spinner_msg):
            record = engine.calibrate(metric_results, anns, dataset_items)
    elif args.optimizer == "gepa-native" and not args.no_optimize:
        spinner_msg = f"Running GEPA-Native optimization (max {args.gepa_native_max_calls} calls)"
        with Spinner(spinner_msg):
            record = engine.calibrate(metric_results, anns, dataset_items)
    elif args.optimizer == "opro" and not args.no_optimize:
        spinner_msg = (
            f"Running OPRO optimization (max {args.opro_iterations} iterations)"
        )
        with Spinner(spinner_msg):
            record = engine.calibrate(metric_results, anns, dataset_items)
    elif args.optimizer == "ape" and not args.no_optimize:
        spinner_msg = f"Running APE optimization ({args.ape_candidates} candidates, {args.ape_rounds} UCB rounds)"
        with Spinner(spinner_msg):
            record = engine.calibrate(metric_results, anns, dataset_items)
    else:
        record = engine.calibrate(metric_results, anns, dataset_items)

    # Display results
    print(f"\n{'=' * 60}")
    print(f"CALIBRATION REPORT: {args.metric_id}")
    print(f"{'=' * 60}")
    print(f"Eval Run: {run.id}")
    print(
        f"Samples:  {record.adjustments.get('alignment_metrics', {}).get('total_samples', 0)}"
    )

    # Alignment metrics
    alignment = record.adjustments.get("alignment_metrics", {})
    if alignment:
        print("\n--- ALIGNMENT METRICS ---")
        print(f"Accuracy:       {alignment.get('accuracy', 0):.1%}")
        print(f"Precision:      {alignment.get('precision', 0):.1%}")
        print(f"Recall:         {alignment.get('recall', 0):.1%}")
        print(f"F1 Score:       {alignment.get('f1', 0):.1%}")
        print(f"Specificity:    {alignment.get('specificity', 0):.1%}")
        print(f"Cohen's Kappa:  {alignment.get('cohens_kappa', 0):.3f}")

        # Confusion matrix
        cm = alignment.get("confusion_matrix", {})
        if cm:
            print("\nConfusion Matrix:")
            print("                   Human PASS  Human FAIL")
            print(
                f"  Judge PASS       {cm.get('true_positive', 0):^10}  {cm.get('false_positive', 0):^10}"
            )
            print(
                f"  Judge FAIL       {cm.get('false_negative', 0):^10}  {cm.get('true_negative', 0):^10}"
            )

    # Disagreement patterns
    disagreements = record.adjustments.get("disagreement_patterns", {})
    if disagreements:
        fp_count = disagreements.get("false_positive_count", 0)
        fn_count = disagreements.get("false_negative_count", 0)
        if fp_count > 0 or fn_count > 0:
            print("\n--- DISAGREEMENT PATTERNS ---")
            print(f"False Positives (judge too lenient): {fp_count}")
            print(f"False Negatives (judge too strict):  {fn_count}")

            if args.show_examples:
                fp_examples = disagreements.get("false_positive_examples", [])[:3]
                fn_examples = disagreements.get("false_negative_examples", [])[:3]

                if fp_examples:
                    print("\nFalse Positive Examples:")
                    for i, ex in enumerate(fp_examples, 1):
                        print(f"  {i}. call_id={ex.get('call_id', '')[:8]}...")
                        print(
                            f"     Judge reason: {ex.get('judge_reason', '')[:80]}..."
                        )
                        if ex.get("human_notes"):
                            print(
                                f"     Human notes:  {ex.get('human_notes', '')[:80]}..."
                            )

                if fn_examples:
                    print("\nFalse Negative Examples:")
                    for i, ex in enumerate(fn_examples, 1):
                        print(f"  {i}. call_id={ex.get('call_id', '')[:8]}...")
                        print(
                            f"     Judge reason: {ex.get('judge_reason', '')[:80]}..."
                        )
                        if ex.get("human_notes"):
                            print(
                                f"     Human notes:  {ex.get('human_notes', '')[:80]}..."
                            )

    # Threshold suggestion
    print("\n--- THRESHOLD ---")
    print(f"Current:   {record.adjustments.get('current_threshold', 0.5):.3f}")
    print(f"Suggested: {record.adjustments.get('suggested_threshold', 0.5):.3f}")

    # Prompt optimization results
    optimizer_type = record.adjustments.get("optimizer_type", "basic")
    optimization = record.adjustments.get("prompt_optimization", {})
    if optimization:
        print("\n--- PROMPT OPTIMIZATION ---")
        print(f"Optimizer:             {optimizer_type.upper()}")
        print(
            f"Estimated improvement: {optimization.get('estimated_improvement', 'unknown')}"
        )

        reasoning = optimization.get("improvement_reasoning", "")
        if reasoning:
            # Word-wrap the reasoning
            words = reasoning.split()
            lines = []
            current_line = ""
            for word in words:
                if len(current_line) + len(word) + 1 <= 70:
                    current_line += (" " if current_line else "") + word
                else:
                    lines.append(current_line)
                    current_line = word
            if current_line:
                lines.append(current_line)
            print("\nReasoning:")
            for line in lines:
                print(f"  {line}")

        additions = optimization.get("suggested_additions", [])
        if additions:
            print("\nSuggested ADDITIONS to rubric:")
            for a in additions:
                print(f"  + {a}")

        removals = optimization.get("suggested_removals", [])
        if removals:
            print("\nSuggested REMOVALS from rubric:")
            for r in removals:
                print(f"  - {r}")

        # Show optimized preamble (for GEPA)
        optimized_preamble = optimization.get("optimized_preamble", "")
        if optimized_preamble:
            print("\nOPTIMIZED PREAMBLE:")
            # Show first 200 chars
            preview = (
                optimized_preamble[:200] + "..."
                if len(optimized_preamble) > 200
                else optimized_preamble
            )
            for line in preview.split("\n"):
                print(f"  {line}")

        improved = optimization.get("improved_rubric", [])
        if improved:
            print("\nRUBRIC (unchanged):")
            for i, criterion in enumerate(improved, 1):
                print(f"  {i}. {criterion}")

    # Validation results
    validation = record.adjustments.get("validation", {})
    if validation:
        print("\n--- VALIDATION RESULTS ---")

        is_better = validation.get("is_better", False)
        original_f1 = validation.get("original_f1", 0.0)
        optimized_f1 = validation.get("optimized_f1", 0.0)
        improvement_delta = validation.get("improvement_delta", 0.0)
        confidence = validation.get("confidence", "unknown")
        recommendation = validation.get("recommendation", "uncertain")
        val_samples = validation.get("validation_samples", 0)

        # Status indicator
        if is_better and improvement_delta > 0.05:
            status_icon = "SUCCESS"
            status_msg = "Optimized prompt is SIGNIFICANTLY BETTER"
        elif is_better:
            status_icon = "SUCCESS"
            status_msg = "Optimized prompt is BETTER"
        elif improvement_delta < -0.05:
            status_icon = "DEGRADED"
            status_msg = "Optimized prompt is SIGNIFICANTLY WORSE"
        elif improvement_delta < 0:
            status_icon = "DEGRADED"
            status_msg = "Optimized prompt is WORSE"
        else:
            status_icon = "UNCERTAIN"
            status_msg = "No significant difference"

        print(f"{status_icon} - {status_msg}")
        print()
        print(f"Original F1:     {original_f1:.3f}")
        print(f"Optimized F1:    {optimized_f1:.3f}")

        if improvement_delta > 0:
            print(
                f"Improvement:     +{improvement_delta:.3f} (+{improvement_delta * 100:.1f}%)"
            )
        elif improvement_delta < 0:
            print(
                f"Degradation:     {improvement_delta:.3f} ({improvement_delta * 100:.1f}%)"
            )
        else:
            print(f"Change:          {improvement_delta:.3f}")

        print(f"Validation size: {val_samples} samples")
        print(f"Confidence:      {confidence.upper()}")
        print()

        # Recommendation
        if recommendation == "use_optimized":
            print("RECOMMENDATION: USE OPTIMIZED PROMPT")
            print("   Next: evalyn run-eval --latest --use-calibrated")
        elif recommendation == "keep_original":
            print("RECOMMENDATION: KEEP ORIGINAL PROMPT")
            print("   The optimized prompt did not improve performance.")
        else:
            print("RECOMMENDATION: UNCERTAIN")
            print("   Consider testing both prompts manually.")

    # Save calibration record
    saved_files = {}

    # Auto-save to dataset's calibrations folder if dataset was resolved
    if dataset_dir and dataset_dir.exists():
        try:
            saved_files = save_calibration(record, str(dataset_dir), args.metric_id)
            print("\n--- SAVED FILES ---")
            print(f"Calibration: {saved_files.get('calibration', 'N/A')}")
            if saved_files.get("preamble"):
                print(f"Preamble:    {saved_files.get('preamble')}")
            if saved_files.get("full_prompt"):
                print(f"Full prompt: {saved_files.get('full_prompt')}")
        except Exception as e:
            print(f"\nWarning: Could not save to calibrations folder: {e}")

    # Also save to explicit output path if specified
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(record.as_dict(), f, indent=2, default=str)
        print(f"\nCalibration record also saved to: {output_path}")

    # Display token usage summary
    print_token_usage_summary(record.usage_summary, verbose=getattr(args, "verbose", False))

    print(f"\n{'=' * 60}")

    # Show hint for next step
    if dataset_dir:
        print_hint(
            f"To re-run evaluation with calibrated prompts, run: evalyn run-eval --dataset {dataset_dir} --use-calibrated",
            quiet=getattr(args, "quiet", False),
        )


def cmd_list_calibrations(args: argparse.Namespace) -> None:
    """List calibration records for a dataset."""
    config = load_config()
    dataset_path = resolve_dataset_path(
        getattr(args, "dataset", None), getattr(args, "latest", False), config
    )

    if not dataset_path:
        fatal_error("No dataset specified", "Use --dataset or --latest")

    calibrations_dir = dataset_path / "calibrations"
    if not calibrations_dir.exists():
        print(f"No calibrations found in {dataset_path}")
        return

    # Collect all calibration records
    calibrations = []
    for metric_dir in calibrations_dir.iterdir():
        if not metric_dir.is_dir():
            continue
        metric_id = metric_dir.name
        for cal_file in metric_dir.glob("*.json"):
            if cal_file.name.startswith("."):
                continue
            try:
                with open(cal_file, encoding="utf-8") as f:
                    record = json.load(f)
                    # Parse timestamp from filename (e.g., 20250101_120000_gepa.json)
                    parts = cal_file.stem.split("_")
                    timestamp = (
                        f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else "unknown"
                    )
                    optimizer = parts[2] if len(parts) >= 3 else "unknown"

                    alignment = record.get("adjustments", {}).get(
                        "alignment_metrics", {}
                    )
                    calibrations.append(
                        {
                            "metric_id": metric_id,
                            "timestamp": timestamp,
                            "optimizer": optimizer,
                            "accuracy": alignment.get("accuracy", 0),
                            "f1": alignment.get("f1", 0),
                            "kappa": alignment.get("cohens_kappa", 0),
                            "samples": alignment.get("total_samples", 0),
                            "path": str(cal_file),
                        }
                    )
            except Exception:
                pass

    if not calibrations:
        print(f"No calibration records found in {calibrations_dir}")
        return

    # Sort by timestamp (most recent first)
    calibrations.sort(key=lambda x: x["timestamp"], reverse=True)

    # Output format
    output_format = getattr(args, "format", "table")
    if output_format == "json":
        print(json.dumps(calibrations, indent=2))
        return

    # Table format
    print(f"\nCalibrations in {dataset_path.name}:")
    print(f"{'=' * 80}")
    print(
        f"{'Metric':<25} {'Timestamp':<17} {'Optimizer':<8} {'Acc':<7} {'F1':<7} {'Kappa':<7} {'N':<5}"
    )
    print(f"{'-' * 80}")
    for cal in calibrations:
        print(
            f"{cal['metric_id']:<25} {cal['timestamp']:<17} {cal['optimizer']:<8} "
            f"{cal['accuracy']:.1%}   {cal['f1']:.1%}   {cal['kappa']:.3f}  {cal['samples']:<5}"
        )

    # Show prompt files if any
    print(f"\n{'=' * 80}")
    print("Optimized prompts:")
    for metric_dir in calibrations_dir.iterdir():
        if not metric_dir.is_dir():
            continue
        prompts_dir = metric_dir / "prompts"
        if prompts_dir.exists():
            full_prompts = list(prompts_dir.glob("*_full.txt"))
            if full_prompts:
                latest = sorted(full_prompts, reverse=True)[0]
                print(f"  {metric_dir.name}: {latest}")

    # Show hint for next step
    if calibrations:
        print_hint(
            f"To re-run evaluation with calibrated prompts, run: evalyn run-eval --dataset {dataset_path} --use-calibrated",
            quiet=getattr(args, "quiet", False),
        )


def register_commands(subparsers) -> None:
    """Register calibration commands."""
    # calibrate
    calibrate_parser = subparsers.add_parser(
        "calibrate", help="Calibrate a subjective metric using human annotations"
    )
    calibrate_parser.add_argument(
        "--metric-id",
        required=True,
        help="Metric ID to calibrate (usually the judge metric id)",
    )
    calibrate_parser.add_argument(
        "--annotations",
        required=True,
        help="Path to annotations JSONL (target_id must match call_id)",
    )
    calibrate_parser.add_argument(
        "--run-id", help="Eval run id to calibrate; defaults to latest run"
    )
    calibrate_parser.add_argument(
        "--threshold", type=float, default=0.5, help="Current threshold for pass/fail"
    )
    calibrate_parser.add_argument(
        "--dataset",
        help="Path to dataset folder (provides input/output context for optimization)",
    )
    calibrate_parser.add_argument(
        "--latest", action="store_true", help="Use the most recently modified dataset"
    )
    calibrate_parser.add_argument(
        "--no-optimize", action="store_true", help="Skip LLM-based prompt optimization"
    )
    calibrate_parser.add_argument(
        "--optimizer",
        choices=["basic", "gepa", "gepa-native", "opro", "ape"],
        default="basic",
        help="Optimization method: 'basic' (single-shot, default), 'gepa' (evolutionary), 'gepa-native' (evolutionary with token tracking), 'opro' (trajectory-based), or 'ape' (search-based)",
    )
    calibrate_parser.add_argument(
        "--model",
        default="gemini-2.5-flash-lite",
        help="LLM model for prompt optimization (llm mode)",
    )
    calibrate_parser.add_argument(
        "--gepa-task-lm",
        default="gemini/gemini-2.5-flash",
        help="Task model for GEPA (model being optimized)",
    )
    calibrate_parser.add_argument(
        "--gepa-reflection-lm",
        default="gemini/gemini-2.5-flash",
        help="Reflection model for GEPA (strong model for reflection)",
    )
    calibrate_parser.add_argument(
        "--gepa-max-calls",
        type=int,
        default=150,
        help="Max metric calls budget for GEPA optimization",
    )
    # OPRO-specific arguments
    calibrate_parser.add_argument(
        "--opro-iterations",
        type=int,
        default=10,
        help="Max iterations for OPRO optimization",
    )
    calibrate_parser.add_argument(
        "--opro-candidates",
        type=int,
        default=4,
        help="Number of candidate prompts per OPRO iteration",
    )
    calibrate_parser.add_argument(
        "--opro-optimizer-model",
        default="gemini-2.5-flash",
        help="Model for generating OPRO candidates",
    )
    calibrate_parser.add_argument(
        "--opro-scorer-model",
        default="gemini-2.5-flash-lite",
        help="Model for scoring OPRO candidates",
    )
    # APE-specific arguments
    calibrate_parser.add_argument(
        "--ape-candidates",
        type=int,
        default=10,
        help="Number of candidate prompts for APE (default: 10)",
    )
    calibrate_parser.add_argument(
        "--ape-rounds",
        type=int,
        default=5,
        help="UCB evaluation rounds for APE (default: 5)",
    )
    calibrate_parser.add_argument(
        "--ape-samples",
        type=int,
        default=5,
        help="Samples per candidate per UCB round (default: 5)",
    )
    # GEPA-Native specific arguments
    calibrate_parser.add_argument(
        "--gepa-native-task-model",
        default="gemini-2.5-flash",
        help="Task model for GEPA-Native evaluation (default: gemini-2.5-flash)",
    )
    calibrate_parser.add_argument(
        "--gepa-native-reflection-model",
        default="gemini-2.5-flash",
        help="Reflection model for GEPA-Native mutations (default: gemini-2.5-flash)",
    )
    calibrate_parser.add_argument(
        "--gepa-native-max-calls",
        type=int,
        default=150,
        help="Max metric calls budget for GEPA-Native (default: 150)",
    )
    calibrate_parser.add_argument(
        "--gepa-native-initial-candidates",
        type=int,
        default=5,
        help="Number of initial candidates for GEPA-Native (default: 5)",
    )
    calibrate_parser.add_argument(
        "--gepa-native-batch-size",
        type=int,
        default=5,
        help="Mini-batch size for GEPA-Native feedback (default: 5)",
    )
    calibrate_parser.add_argument(
        "--show-examples", action="store_true", help="Show example disagreement cases"
    )
    calibrate_parser.add_argument(
        "--output", help="Path to save calibration record JSON"
    )
    calibrate_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed cost breakdown"
    )
    calibrate_parser.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    calibrate_parser.set_defaults(func=cmd_calibrate)

    # list-calibrations
    list_cal_parser = subparsers.add_parser(
        "list-calibrations", help="List calibration records for a dataset"
    )
    list_cal_parser.add_argument("--dataset", help="Path to dataset directory")
    list_cal_parser.add_argument(
        "--latest", action="store_true", help="Use the most recently modified dataset"
    )
    list_cal_parser.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    list_cal_parser.set_defaults(func=cmd_list_calibrations)


__all__ = ["cmd_calibrate", "cmd_list_calibrations", "register_commands"]
