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

from ..utils.command_common import load_eval_run_for_command, try_resolve_dataset_dir_and_file
from ..utils.config import load_config, resolve_dataset_path, get_config_default
from ..utils.errors import fatal_error
from ..utils.formatters import print_token_usage_summary
from ..utils.hints import HintCollector
from ..utils.rich import banner, table as rich_table, footer, icon, status_icon
from ..utils.ui import Spinner

# Real defaults for calibration arguments.
# Stored as constants so help text stays accurate when argparse defaults are None.
_DEFAULT_GEPA_TASK_LM = "gemini/gemini-2.5-flash"
_DEFAULT_GEPA_REFLECTION_LM = "gemini/gemini-2.5-flash"
_DEFAULT_GEPA_MAX_CALLS = 150
_DEFAULT_OPRO_OPTIMIZER_MODEL = "gemini-2.5-flash"
_DEFAULT_OPRO_ITERATIONS = 10
_DEFAULT_OPRO_CANDIDATES = 4
_DEFAULT_APE_CANDIDATES = 10
_DEFAULT_APE_ROUNDS = 5
_DEFAULT_APE_SAMPLES = 5
_DEFAULT_GEPA_NATIVE_TASK_MODEL = "gemini-2.5-flash"
_DEFAULT_GEPA_NATIVE_REFLECTION_MODEL = "gemini-2.5-flash"
_DEFAULT_GEPA_NATIVE_MAX_CALLS = 150
_DEFAULT_GEPA_NATIVE_INITIAL_CANDIDATES = 5

# Significance threshold for improvement/degradation in validation delta
_IMPROVEMENT_SIGNIFICANCE = 0.05
_DEFAULT_GEPA_NATIVE_BATCH_SIZE = 5
_DEFAULT_EVO_POPULATION = 8
_DEFAULT_EVO_GENERATIONS = 5
_DEFAULT_EVO_MUTATION_RATE = 0.3
_DEFAULT_TEXTGRAD_ITERATIONS = 8
_DEFAULT_TEXTGRAD_THRESHOLD = 0.01
_DEFAULT_MIPRO_INSTRUCTIONS = 6
_DEFAULT_MIPRO_DEMOS = 3
_DEFAULT_MIPRO_EVAL_SAMPLES = 10
_DEFAULT_PB_POPULATION = 6
_DEFAULT_PB_GENERATIONS = 5


def _apply_calibration_config_defaults(args: argparse.Namespace, config: dict) -> None:
    """Apply config file defaults to calibration args."""
    from ...defaults import DEFAULT_EVAL_MODEL

    # Basic calibration settings
    if args.optimizer is None:
        args.optimizer = get_config_default(
            config, "calibration", "optimizer", default="basic"
        )
    if args.threshold is None:
        args.threshold = get_config_default(
            config, "calibration", "threshold", default=0.5
        )

    # GEPA settings
    if args.gepa_task_lm is None:
        args.gepa_task_lm = get_config_default(
            config, "calibration", "gepa", "task_lm", default=_DEFAULT_GEPA_TASK_LM
        )
    if args.gepa_reflection_lm is None:
        args.gepa_reflection_lm = get_config_default(
            config,
            "calibration",
            "gepa",
            "reflection_lm",
            default=_DEFAULT_GEPA_REFLECTION_LM,
        )
    if args.gepa_max_calls is None:
        args.gepa_max_calls = get_config_default(
            config, "calibration", "gepa", "max_calls", default=_DEFAULT_GEPA_MAX_CALLS
        )

    # OPRO settings
    if args.opro_optimizer_model is None:
        args.opro_optimizer_model = get_config_default(
            config, "calibration", "opro", "optimizer_model", default=_DEFAULT_OPRO_OPTIMIZER_MODEL
        )
    if args.opro_scorer_model is None:
        args.opro_scorer_model = get_config_default(
            config,
            "calibration",
            "opro",
            "scorer_model",
            default=DEFAULT_EVAL_MODEL,
        )
    if args.opro_iterations is None:
        args.opro_iterations = get_config_default(
            config, "calibration", "opro", "iterations", default=_DEFAULT_OPRO_ITERATIONS
        )
    if args.opro_candidates is None:
        args.opro_candidates = get_config_default(
            config, "calibration", "opro", "candidates", default=_DEFAULT_OPRO_CANDIDATES
        )

    # APE settings
    if args.ape_candidates is None:
        args.ape_candidates = get_config_default(
            config, "calibration", "ape", "candidates", default=_DEFAULT_APE_CANDIDATES
        )
    if args.ape_rounds is None:
        args.ape_rounds = get_config_default(
            config, "calibration", "ape", "rounds", default=_DEFAULT_APE_ROUNDS
        )
    if args.ape_samples is None:
        args.ape_samples = get_config_default(
            config, "calibration", "ape", "samples", default=_DEFAULT_APE_SAMPLES
        )

    # GEPA-Native settings
    if args.gepa_native_task_model is None:
        args.gepa_native_task_model = get_config_default(
            config,
            "calibration",
            "gepa_native",
            "task_model",
            default=_DEFAULT_GEPA_NATIVE_TASK_MODEL,
        )
    if args.gepa_native_reflection_model is None:
        args.gepa_native_reflection_model = get_config_default(
            config,
            "calibration",
            "gepa_native",
            "reflection_model",
            default=_DEFAULT_GEPA_NATIVE_REFLECTION_MODEL,
        )
    if args.gepa_native_max_calls is None:
        args.gepa_native_max_calls = get_config_default(
            config, "calibration", "gepa_native", "max_calls", default=_DEFAULT_GEPA_NATIVE_MAX_CALLS
        )
    if args.gepa_native_initial_candidates is None:
        args.gepa_native_initial_candidates = get_config_default(
            config, "calibration", "gepa_native", "initial_candidates", default=_DEFAULT_GEPA_NATIVE_INITIAL_CANDIDATES
        )
    if args.gepa_native_batch_size is None:
        args.gepa_native_batch_size = get_config_default(
            config, "calibration", "gepa_native", "batch_size", default=_DEFAULT_GEPA_NATIVE_BATCH_SIZE
        )


def _load_calibration_run(args: argparse.Namespace):
    """Load eval run for calibration (explicit run or latest with target metric)."""
    from ...decorators import get_default_tracer

    tracer = get_default_tracer()
    dataset_arg = getattr(args, "dataset", None)
    dataset_path = Path(dataset_arg) if dataset_arg else None

    loaded = load_eval_run_for_command(
        run_id=getattr(args, "run_id", None),
        dataset_path=dataset_path,
        metric_id=getattr(args, "metric_id", None),
        fallback_to_storage=True,
        storage=tracer.storage,
        error_message="No eval runs available",
    )
    return tracer, loaded.run


def _get_calibration_metric_results(run, metric_id: str):
    """Filter metric results for the selected metric."""
    metric_results = [r for r in run.metric_results if r.metric_id == metric_id]
    if not metric_results:
        raise ValueError(f"No metric results found for metric_id={metric_id} in run {run.id}")
    return metric_results


def _extract_calibration_metric_prompt_context(run, metric_id: str) -> tuple[List[str], str]:
    """Extract rubric and prompt preamble from metric config in the run."""
    current_rubric: List[str] = []
    current_preamble = ""
    for metric_spec in run.metrics:
        if metric_spec.id != metric_id:
            continue
        cfg = metric_spec.config or {}
        rubric_val = cfg.get("rubric", [])
        if isinstance(rubric_val, list):
            current_rubric = [str(r) for r in rubric_val]
        preamble_val = cfg.get("prompt", "")
        if isinstance(preamble_val, str):
            current_preamble = preamble_val
        break
    return current_rubric, current_preamble


def _load_optional_calibration_dataset_context(
    args: argparse.Namespace, config: dict
) -> tuple[Optional[Path], Optional[List[DatasetItem]]]:
    """Resolve optional dataset context for calibration examples/validation."""
    from ...datasets import load_dataset
    from ...models import DatasetItem

    dataset_items: Optional[List[DatasetItem]] = None
    dataset_dir: Optional[Path] = None

    dataset_arg = getattr(args, "dataset", None)
    use_latest = getattr(args, "latest", False)
    resolved_dataset = resolve_dataset_path(dataset_arg, use_latest, config)

    if resolved_dataset:
        resolved = try_resolve_dataset_dir_and_file(
            dataset_arg, use_latest, config=config
        )
        if resolved:
            dataset_dir, dataset_file = resolved
            dataset_items = load_dataset(dataset_file)

    return dataset_dir, dataset_items


def _build_calibration_optimizer_configs(args: argparse.Namespace) -> dict:
    """Build optimizer-specific config objects from CLI args."""
    from ...calibration import (
        GEPAConfig,
        GEPA_AVAILABLE,
        GEPANativeConfig,
        OPROConfig,
        APEConfig,
    )

    gepa_config = None
    if args.optimizer == "gepa":
        if not GEPA_AVAILABLE:
            raise ImportError("GEPA is not installed. Install with: pip install gepa")
        gepa_config = GEPAConfig(
            task_lm=args.gepa_task_lm,
            reflection_lm=args.gepa_reflection_lm,
            max_metric_calls=args.gepa_max_calls,
        )

    opro_config = None
    if args.optimizer == "opro":
        opro_config = OPROConfig(
            optimizer_model=args.opro_optimizer_model,
            scorer_model=args.opro_scorer_model,
            max_iterations=args.opro_iterations,
            candidates_per_step=args.opro_candidates,
        )

    ape_config = None
    if args.optimizer == "ape":
        ape_config = APEConfig(
            num_candidates=args.ape_candidates,
            eval_rounds=args.ape_rounds,
            eval_samples_per_round=args.ape_samples,
        )

    gepa_native_config = None
    if args.optimizer == "gepa-native":
        gepa_native_config = GEPANativeConfig(
            task_model=args.gepa_native_task_model,
            reflection_model=args.gepa_native_reflection_model,
            max_metric_calls=args.gepa_native_max_calls,
            num_initial_candidates=args.gepa_native_initial_candidates,
            mini_batch_size=args.gepa_native_batch_size,
        )

    # New optimizers use generic optimizer_config
    optimizer_config = None
    if args.optimizer == "evoprompt":
        from ...calibration.evoprompt import EvoPromptConfig

        optimizer_config = EvoPromptConfig(
            population_size=getattr(args, "evo_population", None) or _DEFAULT_EVO_POPULATION,
            generations=getattr(args, "evo_generations", None) or _DEFAULT_EVO_GENERATIONS,
            mutation_rate=getattr(args, "evo_mutation_rate", None) or _DEFAULT_EVO_MUTATION_RATE,
        )
    elif args.optimizer == "textgrad":
        from ...calibration.textgrad import TextGradConfig

        optimizer_config = TextGradConfig(
            max_iterations=getattr(args, "textgrad_iterations", None) or _DEFAULT_TEXTGRAD_ITERATIONS,
            improvement_threshold=getattr(args, "textgrad_threshold", None) or _DEFAULT_TEXTGRAD_THRESHOLD,
        )
    elif args.optimizer == "miprov2":
        from ...calibration.miprov2 import MIPROv2Config

        optimizer_config = MIPROv2Config(
            num_instructions=getattr(args, "mipro_instructions", None) or _DEFAULT_MIPRO_INSTRUCTIONS,
            num_demos=getattr(args, "mipro_demos", None) or _DEFAULT_MIPRO_DEMOS,
            eval_samples=getattr(args, "mipro_eval_samples", None) or _DEFAULT_MIPRO_EVAL_SAMPLES,
        )
    elif args.optimizer == "promptbreeder":
        from ...calibration.promptbreeder import PromptBreederConfig

        optimizer_config = PromptBreederConfig(
            population_size=getattr(args, "pb_population", None) or _DEFAULT_PB_POPULATION,
            generations=getattr(args, "pb_generations", None) or _DEFAULT_PB_GENERATIONS,
        )

    return {
        "gepa_config": gepa_config,
        "opro_config": opro_config,
        "ape_config": ape_config,
        "gepa_native_config": gepa_native_config,
        "optimizer_config": optimizer_config,
    }


def _build_calibration_engine(
    args: argparse.Namespace,
    *,
    current_rubric: List[str],
    current_preamble: str,
    optimizer_configs: dict,
) -> CalibrationEngine:
    """Construct CalibrationEngine from args and optimizer configs."""
    from ...calibration import CalibrationEngine

    return CalibrationEngine(
        judge_name=args.metric_id,
        current_threshold=args.threshold,
        current_rubric=current_rubric,
        current_preamble=current_preamble,
        optimize_prompts=not args.no_optimize,
        optimizer_model=args.model,
        optimizer_type=args.optimizer,
        optimizer_config=optimizer_configs.get("optimizer_config"),
        gepa_config=optimizer_configs.get("gepa_config"),
        gepa_native_config=optimizer_configs.get("gepa_native_config"),
        opro_config=optimizer_configs.get("opro_config"),
        ape_config=optimizer_configs.get("ape_config"),
    )


def _run_calibration_with_spinner(
    args: argparse.Namespace,
    engine: CalibrationEngine,
    metric_results,
    anns,
    dataset_items: Optional[List[DatasetItem]],
):
    """Run calibration, using a spinner for long-running optimizers."""
    if args.optimizer != "basic" and not args.no_optimize:
        spinner_msg = f"Running {args.optimizer.upper()} optimization..."
        with Spinner(spinner_msg):
            return engine.calibrate(metric_results, anns, dataset_items)
    return engine.calibrate(metric_results, anns, dataset_items)


def _print_calibration_alignment_section(record) -> None:
    """Print alignment metrics and confusion matrix."""
    alignment = record.adjustments.get("alignment_metrics", {})
    if not alignment:
        return

    print("\n--- ALIGNMENT METRICS ---")
    print(f"Accuracy:       {alignment.get('accuracy', 0):.1%}")
    print(f"Precision:      {alignment.get('precision', 0):.1%}")
    print(f"Recall:         {alignment.get('recall', 0):.1%}")
    print(f"F1 Score:       {alignment.get('f1', 0):.1%}")
    print(f"Specificity:    {alignment.get('specificity', 0):.1%}")
    print(f"Cohen's Kappa:  {alignment.get('cohens_kappa', 0):.3f}")

    cm = alignment.get("confusion_matrix", {})
    if not cm:
        return
    print("\nConfusion Matrix:")
    print("                   Human PASS  Human FAIL")
    print(
        f"  Judge PASS       {cm.get('true_positive', 0):^10}  {cm.get('false_positive', 0):^10}"
    )
    print(
        f"  Judge FAIL       {cm.get('false_negative', 0):^10}  {cm.get('true_negative', 0):^10}"
    )


def _print_calibration_disagreement_section(args: argparse.Namespace, record) -> None:
    """Print disagreement counts and optional examples."""
    disagreements = record.adjustments.get("disagreement_patterns", {})
    if not disagreements:
        return

    fp_count = disagreements.get("false_positive_count", 0)
    fn_count = disagreements.get("false_negative_count", 0)
    if fp_count <= 0 and fn_count <= 0:
        return

    print("\n--- DISAGREEMENT PATTERNS ---")
    print(f"False Positives (judge too lenient): {fp_count}")
    print(f"False Negatives (judge too strict):  {fn_count}")

    if not args.show_examples:
        return

    fp_examples = disagreements.get("false_positive_examples", [])[:3]
    fn_examples = disagreements.get("false_negative_examples", [])[:3]

    if fp_examples:
        print("\nFalse Positive Examples:")
        for i, ex in enumerate(fp_examples, 1):
            print(f"  {i}. call_id={ex.get('call_id', '')[:8]}...")
            print(f"     Judge reason: {ex.get('judge_reason', '')[:80]}...")
            if ex.get("human_notes"):
                print(f"     Human notes:  {ex.get('human_notes', '')[:80]}...")

    if fn_examples:
        print("\nFalse Negative Examples:")
        for i, ex in enumerate(fn_examples, 1):
            print(f"  {i}. call_id={ex.get('call_id', '')[:8]}...")
            print(f"     Judge reason: {ex.get('judge_reason', '')[:80]}...")
            if ex.get("human_notes"):
                print(f"     Human notes:  {ex.get('human_notes', '')[:80]}...")


def _print_calibration_threshold_section(record) -> None:
    """Print threshold suggestion summary."""
    print("\n--- THRESHOLD ---")
    print(f"Current:   {record.adjustments.get('current_threshold', 0.5):.3f}")
    print(f"Suggested: {record.adjustments.get('suggested_threshold', 0.5):.3f}")


def _wrap_calibration_text(text: str, width: int = 70) -> List[str]:
    """Simple word-wrap helper for calibration report text."""
    words = text.split()
    lines: List[str] = []
    current_line = ""
    for word in words:
        if len(current_line) + len(word) + 1 <= width:
            current_line += (" " if current_line else "") + word
        else:
            lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return lines


def _print_calibration_prompt_optimization_section(record) -> None:
    """Print prompt optimization details."""
    optimizer_type = record.adjustments.get("optimizer_type", "basic")
    optimization = record.adjustments.get("prompt_optimization", {})
    if not optimization:
        return

    print("\n--- PROMPT OPTIMIZATION ---")
    print(f"Optimizer:             {optimizer_type.upper()}")
    print(
        f"Estimated improvement: {optimization.get('estimated_improvement', 'unknown')}"
    )

    reasoning = optimization.get("improvement_reasoning", "")
    if reasoning:
        print("\nReasoning:")
        for line in _wrap_calibration_text(reasoning):
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

    optimized_preamble = optimization.get("optimized_preamble", "")
    if optimized_preamble:
        print("\nOPTIMIZED PREAMBLE:")
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


def _print_calibration_validation_section(record) -> None:
    """Print validation comparison and recommendation details."""
    validation = record.adjustments.get("validation", {})
    if not validation:
        return

    print("\n--- VALIDATION RESULTS ---")

    is_better = validation.get("is_better", False)
    original_f1 = validation.get("original_f1", 0.0)
    optimized_f1 = validation.get("optimized_f1", 0.0)
    improvement_delta = validation.get("improvement_delta", 0.0)
    confidence = validation.get("confidence", "unknown")
    recommendation = validation.get("recommendation", "uncertain")
    val_samples = validation.get("validation_samples", 0)

    if is_better and improvement_delta > _IMPROVEMENT_SIGNIFICANCE:
        status_icon = "SUCCESS"
        status_msg = "Optimized prompt is SIGNIFICANTLY BETTER"
    elif is_better:
        status_icon = "SUCCESS"
        status_msg = "Optimized prompt is BETTER"
    elif improvement_delta < -_IMPROVEMENT_SIGNIFICANCE:
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
        print(f"Improvement:     +{improvement_delta:.3f} (+{improvement_delta * 100:.1f}%)")
    elif improvement_delta < 0:
        print(f"Degradation:     {improvement_delta:.3f} ({improvement_delta * 100:.1f}%)")
    else:
        print(f"Change:          {improvement_delta:.3f}")

    print(f"Validation size: {val_samples} samples")
    print(f"Confidence:      {confidence.upper()}")
    print()

    if recommendation == "use_optimized":
        print("RECOMMENDATION: USE OPTIMIZED PROMPT")
        print("   Next: evalyn run-eval --latest --use-calibrated")
    elif recommendation == "keep_original":
        print("RECOMMENDATION: KEEP ORIGINAL PROMPT")
        print("   The optimized prompt did not improve performance.")
    else:
        print("RECOMMENDATION: UNCERTAIN")
        print("   Consider testing both prompts manually.")


def _print_calibration_report(args: argparse.Namespace, run, record) -> None:
    """Print full calibration report."""
    print(f"\n{'=' * 60}")
    print(f"CALIBRATION REPORT: {args.metric_id}")
    print(f"{'=' * 60}")
    print(f"Eval Run: {run.id}")
    print(
        f"Samples:  {record.adjustments.get('alignment_metrics', {}).get('total_samples', 0)}"
    )

    _print_calibration_alignment_section(record)
    _print_calibration_disagreement_section(args, record)
    _print_calibration_threshold_section(record)
    _print_calibration_prompt_optimization_section(record)
    _print_calibration_validation_section(record)


def _save_calibration_outputs(
    args: argparse.Namespace,
    record,
    dataset_dir: Optional[Path],
) -> None:
    """Save calibration record to dataset artifacts and/or explicit output path."""
    from ...calibration import save_calibration

    saved_files = {}

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

    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(record.as_dict(), f, indent=2, default=str)
        print(f"\nCalibration record also saved to: {output_path}")


def _print_calibration_postamble(
    args: argparse.Namespace,
    record,
    dataset_dir: Optional[Path],
) -> None:
    """Print final token usage, footer, and next-step hint."""
    print_token_usage_summary(
        record.usage_summary, verbose=getattr(args, "verbose", False)
    )

    print(f"\n{'=' * 60}")

    if dataset_dir:
        hints = HintCollector(quiet=getattr(args, "quiet", False))
        hints.add(
            f"evalyn run-eval --dataset {dataset_dir} --use-calibrated",
            "Re-run with calibrated prompts",
            options=[
                ("--provider gemini|openai|ollama", "LLM provider for judges (default: gemini)"),
                ("--workers <N>", "Parallel workers (default: 4)"),
                ("--verbose", "Show detailed cost breakdown by metric"),
            ],
        )
        hints.render()


def calibrate_metric(
    metric_id: str,
    annotations_path: str,
    dataset_dir: str,
    *,
    optimizer: str = "basic",
    model: Optional[str] = None,
    threshold: float = 0.5,
    no_optimize: bool = False,
    run_id: Optional[str] = None,
    verbose: bool = False,
) -> None:
    """Run calibration for a single metric - callable without argparse.

    This is the typed API for calibration used by both the CLI and pipeline.
    It loads the run, annotations, and dataset, runs calibration, and saves outputs.
    """
    from ...annotation import import_annotations
    from ...defaults import DEFAULT_EVAL_MODEL

    if model is None:
        model = DEFAULT_EVAL_MODEL

    # Build a namespace with all attributes that helpers expect
    args = argparse.Namespace(
        metric_id=metric_id,
        annotations=annotations_path,
        dataset=dataset_dir,
        latest=False,
        run_id=run_id,
        optimizer=optimizer,
        model=model,
        threshold=threshold,
        no_optimize=no_optimize,
        show_examples=False,
        output=None,
        format="table",
        verbose=verbose,
        quiet=False,
        # GEPA defaults
        gepa_task_lm=_DEFAULT_GEPA_TASK_LM,
        gepa_reflection_lm=_DEFAULT_GEPA_REFLECTION_LM,
        gepa_max_calls=_DEFAULT_GEPA_MAX_CALLS,
        # OPRO defaults
        opro_iterations=_DEFAULT_OPRO_ITERATIONS,
        opro_candidates=_DEFAULT_OPRO_CANDIDATES,
        opro_optimizer_model=_DEFAULT_OPRO_OPTIMIZER_MODEL,
        opro_scorer_model=DEFAULT_EVAL_MODEL,
        # APE defaults
        ape_candidates=_DEFAULT_APE_CANDIDATES,
        ape_rounds=_DEFAULT_APE_ROUNDS,
        ape_samples=_DEFAULT_APE_SAMPLES,
        # GEPA-Native defaults
        gepa_native_task_model=_DEFAULT_GEPA_NATIVE_TASK_MODEL,
        gepa_native_reflection_model=_DEFAULT_GEPA_NATIVE_REFLECTION_MODEL,
        gepa_native_max_calls=_DEFAULT_GEPA_NATIVE_MAX_CALLS,
        gepa_native_initial_candidates=_DEFAULT_GEPA_NATIVE_INITIAL_CANDIDATES,
        gepa_native_batch_size=_DEFAULT_GEPA_NATIVE_BATCH_SIZE,
        # EvoPrompt defaults
        evo_population=_DEFAULT_EVO_POPULATION,
        evo_generations=_DEFAULT_EVO_GENERATIONS,
        evo_mutation_rate=_DEFAULT_EVO_MUTATION_RATE,
        # TextGrad defaults
        textgrad_iterations=_DEFAULT_TEXTGRAD_ITERATIONS,
        textgrad_threshold=_DEFAULT_TEXTGRAD_THRESHOLD,
        # MIPROv2 defaults
        mipro_instructions=_DEFAULT_MIPRO_INSTRUCTIONS,
        mipro_demos=_DEFAULT_MIPRO_DEMOS,
        mipro_eval_samples=_DEFAULT_MIPRO_EVAL_SAMPLES,
        # PromptBreeder defaults
        pb_population=_DEFAULT_PB_POPULATION,
        pb_generations=_DEFAULT_PB_GENERATIONS,
    )

    _, run = _load_calibration_run(args)
    metric_results = _get_calibration_metric_results(run, metric_id)
    anns = import_annotations(annotations_path)
    current_rubric, current_preamble = _extract_calibration_metric_prompt_context(
        run, metric_id
    )

    config = load_config()
    _apply_calibration_config_defaults(args, config)
    ds_dir, dataset_items = _load_optional_calibration_dataset_context(args, config)
    optimizer_configs = _build_calibration_optimizer_configs(args)
    engine = _build_calibration_engine(
        args,
        current_rubric=current_rubric,
        current_preamble=current_preamble,
        optimizer_configs=optimizer_configs,
    )
    record = _run_calibration_with_spinner(args, engine, metric_results, anns, dataset_items)

    _print_calibration_report(args, run, record)
    _save_calibration_outputs(args, record, ds_dir)
    _print_calibration_postamble(args, record, ds_dir)


def cmd_calibrate(args: argparse.Namespace) -> None:
    """Calibrate a subjective metric using human annotations."""
    from ...defaults import DEFAULT_EVAL_MODEL

    try:
        calibrate_metric(
            metric_id=args.metric_id,
            annotations_path=args.annotations,
            dataset_dir=getattr(args, "dataset", None) or "",
            optimizer=getattr(args, "optimizer", "basic"),
            model=getattr(args, "model", DEFAULT_EVAL_MODEL),
            threshold=getattr(args, "threshold", 0.5),
            no_optimize=getattr(args, "no_optimize", False),
            run_id=getattr(args, "run_id", None),
            verbose=getattr(args, "verbose", False),
        )
    except FileNotFoundError as e:
        fatal_error(str(e))
    except ValueError as e:
        fatal_error(str(e))


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
    headers = ["Metric", "Timestamp", "Optimizer", "Acc", "F1", "Kappa", "N"]
    align = ["left", "left", "left", "right", "right", "right", "right"]
    rows = []
    for cal in calibrations:
        rows.append([
            cal["metric_id"],
            cal["timestamp"],
            cal["optimizer"],
            f"{cal['accuracy']:.1%}",
            f"{cal['f1']:.1%}",
            f"{cal['kappa']:.3f}",
            str(cal["samples"]),
        ])

    print(banner("CALIBRATIONS"))
    print(rich_table(headers, rows, align=align))

    # Show prompt files if any
    has_prompts = False
    for metric_dir in calibrations_dir.iterdir():
        if not metric_dir.is_dir():
            continue
        prompts_dir = metric_dir / "prompts"
        if prompts_dir.exists():
            full_prompts = list(prompts_dir.glob("*_full.txt"))
            if full_prompts:
                if not has_prompts:
                    print("\nOptimized prompts:")
                    has_prompts = True
                latest = sorted(full_prompts, reverse=True)[0]
                print(f"  {metric_dir.name}: {latest}")

    # Show hints
    if calibrations:
        hint_items = [(f"evalyn run-eval --dataset {dataset_path} --use-calibrated", "Re-run with calibrated prompts")]
        hint_text = footer(hint_items, quiet=getattr(args, "quiet", False))
        if hint_text:
            print(hint_text)


def register_commands(subparsers) -> None:
    """Register calibration commands."""
    from ...defaults import DEFAULT_EVAL_MODEL

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
        "--threshold", type=float, default=None, help="Current threshold for pass/fail"
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
        choices=["basic", "gepa", "gepa-native", "opro", "ape", "evoprompt", "textgrad", "miprov2", "promptbreeder"],
        default=None,
        help="Optimization method: 'basic' (single-shot), 'ape' (search), 'opro' (trajectory), 'evoprompt' (evolutionary), 'textgrad' (critique-revise), 'miprov2' (instruction+demo), 'promptbreeder' (self-referential)",
    )
    calibrate_parser.add_argument(
        "--model",
        default=DEFAULT_EVAL_MODEL,
        help=f"LLM model for prompt optimization (default: {DEFAULT_EVAL_MODEL})",
    )
    calibrate_parser.add_argument(
        "--gepa-task-lm",
        default=None,
        help=f"Task model for GEPA (model being optimized) (default: {_DEFAULT_GEPA_TASK_LM})",
    )
    calibrate_parser.add_argument(
        "--gepa-reflection-lm",
        default=None,
        help=f"Reflection model for GEPA (strong model for reflection) (default: {_DEFAULT_GEPA_REFLECTION_LM})",
    )
    calibrate_parser.add_argument(
        "--gepa-max-calls",
        type=int,
        default=None,
        help=f"Max metric calls budget for GEPA optimization (default: {_DEFAULT_GEPA_MAX_CALLS})",
    )
    # OPRO-specific arguments
    calibrate_parser.add_argument(
        "--opro-iterations",
        type=int,
        default=None,
        help=f"Max iterations for OPRO optimization (default: {_DEFAULT_OPRO_ITERATIONS})",
    )
    calibrate_parser.add_argument(
        "--opro-candidates",
        type=int,
        default=None,
        help=f"Number of candidate prompts per OPRO iteration (default: {_DEFAULT_OPRO_CANDIDATES})",
    )
    calibrate_parser.add_argument(
        "--opro-optimizer-model",
        default=None,
        help=f"Model for generating OPRO candidates (default: {_DEFAULT_OPRO_OPTIMIZER_MODEL})",
    )
    calibrate_parser.add_argument(
        "--opro-scorer-model",
        default=None,
        help=f"Model for scoring OPRO candidates (default: {DEFAULT_EVAL_MODEL})",
    )
    # APE-specific arguments
    calibrate_parser.add_argument(
        "--ape-candidates",
        type=int,
        default=None,
        help=f"Number of candidate prompts for APE (default: {_DEFAULT_APE_CANDIDATES})",
    )
    calibrate_parser.add_argument(
        "--ape-rounds",
        type=int,
        default=None,
        help=f"UCB evaluation rounds for APE (default: {_DEFAULT_APE_ROUNDS})",
    )
    calibrate_parser.add_argument(
        "--ape-samples",
        type=int,
        default=None,
        help=f"Samples per candidate per UCB round (default: {_DEFAULT_APE_SAMPLES})",
    )
    # GEPA-Native specific arguments
    calibrate_parser.add_argument(
        "--gepa-native-task-model",
        default=None,
        help=f"Task model for GEPA-Native evaluation (default: {_DEFAULT_GEPA_NATIVE_TASK_MODEL})",
    )
    calibrate_parser.add_argument(
        "--gepa-native-reflection-model",
        default=None,
        help=f"Reflection model for GEPA-Native mutations (default: {_DEFAULT_GEPA_NATIVE_REFLECTION_MODEL})",
    )
    calibrate_parser.add_argument(
        "--gepa-native-max-calls",
        type=int,
        default=None,
        help=f"Max metric calls budget for GEPA-Native (default: {_DEFAULT_GEPA_NATIVE_MAX_CALLS})",
    )
    calibrate_parser.add_argument(
        "--gepa-native-initial-candidates",
        type=int,
        default=None,
        help=f"Number of initial candidates for GEPA-Native (default: {_DEFAULT_GEPA_NATIVE_INITIAL_CANDIDATES})",
    )
    calibrate_parser.add_argument(
        "--gepa-native-batch-size",
        type=int,
        default=None,
        help=f"Mini-batch size for GEPA-Native feedback (default: {_DEFAULT_GEPA_NATIVE_BATCH_SIZE})",
    )
    # EvoPrompt-specific arguments
    calibrate_parser.add_argument(
        "--evo-population", type=int, default=None,
        help=f"Population size for EvoPrompt (default: {_DEFAULT_EVO_POPULATION})",
    )
    calibrate_parser.add_argument(
        "--evo-generations", type=int, default=None,
        help=f"Number of generations for EvoPrompt (default: {_DEFAULT_EVO_GENERATIONS})",
    )
    calibrate_parser.add_argument(
        "--evo-mutation-rate", type=float, default=None,
        help=f"Mutation rate for EvoPrompt (default: {_DEFAULT_EVO_MUTATION_RATE})",
    )
    # TextGrad-specific arguments
    calibrate_parser.add_argument(
        "--textgrad-iterations", type=int, default=None,
        help=f"Max iterations for TextGrad (default: {_DEFAULT_TEXTGRAD_ITERATIONS})",
    )
    calibrate_parser.add_argument(
        "--textgrad-threshold", type=float, default=None,
        help=f"Min F1 improvement threshold for TextGrad (default: {_DEFAULT_TEXTGRAD_THRESHOLD})",
    )
    # MIPROv2-specific arguments
    calibrate_parser.add_argument(
        "--mipro-instructions", type=int, default=None,
        help=f"Number of instruction candidates for MIPROv2 (default: {_DEFAULT_MIPRO_INSTRUCTIONS})",
    )
    calibrate_parser.add_argument(
        "--mipro-demos", type=int, default=None,
        help=f"Number of few-shot demos for MIPROv2 (default: {_DEFAULT_MIPRO_DEMOS})",
    )
    calibrate_parser.add_argument(
        "--mipro-eval-samples", type=int, default=None,
        help=f"Evaluation samples per candidate for MIPROv2 (default: {_DEFAULT_MIPRO_EVAL_SAMPLES})",
    )
    # PromptBreeder-specific arguments
    calibrate_parser.add_argument(
        "--pb-population", type=int, default=None,
        help=f"Population size for PromptBreeder (default: {_DEFAULT_PB_POPULATION})",
    )
    calibrate_parser.add_argument(
        "--pb-generations", type=int, default=None,
        help=f"Number of generations for PromptBreeder (default: {_DEFAULT_PB_GENERATIONS})",
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
