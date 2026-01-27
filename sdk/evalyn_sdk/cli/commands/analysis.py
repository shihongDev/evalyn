"""Analysis commands: status, validate, analyze, compare, trend.

This module provides CLI commands for analyzing evaluation results, comparing runs,
tracking trends, and validating datasets. These commands help you understand
your evaluation data and identify areas for improvement.

Commands:
- status: Show comprehensive status of a dataset (items, metrics, runs, annotations, calibrations)
- validate: Validate dataset format and detect potential issues (missing fields, duplicates)
- analyze: Analyze evaluation results and generate insights (pass rates, failure patterns)
- compare: Compare two evaluation runs side-by-side (see improvements/regressions)
- trend: Show evaluation trends over time for a project

Key insights from analysis:
- Identify metrics with low pass rates that need calibration
- Find items that fail multiple metrics (potential outliers)
- Track whether changes improve or regress overall quality
- See if calibration is improving alignment with human judgment

Typical workflow:
1. After run-eval: 'evalyn analyze --latest' to see insights
2. After calibration: 'evalyn compare --run1 <old> --run2 <new>' to verify improvement
3. Over time: 'evalyn trend --project <name>' to track progress
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..utils.config import load_config, resolve_dataset_path
from ..utils.dataset_resolver import get_dataset
from ..utils.errors import fatal_error
from ..utils.hints import print_hint


def cmd_status(args: argparse.Namespace) -> None:
    """Show status of a dataset including items, metrics, runs, annotations, and calibrations."""
    ds = get_dataset(
        getattr(args, "dataset", None),
        getattr(args, "latest", False),
        require=True,
    )

    print(f"\n{'=' * 60}")
    print(f"DATASET STATUS: {ds.name}")
    print(f"{'=' * 60}")
    print(f"Path: {ds.path}")

    # Dataset items
    print("\n--- DATASET ---")
    print(f"Items: {ds.item_count}")
    if ds.project:
        print(f"Project: {ds.project}")
    if ds.version:
        print(f"Version: {ds.version}")

    # Metrics
    print("\n--- METRICS ---")
    metric_files = ds.list_metrics_files()
    if metric_files:
        print(f"Metric sets: {len(metric_files)}")
        for mf in metric_files:
            try:
                with open(mf, encoding="utf-8") as f:
                    metrics = json.load(f)
                    count = len(metrics) if isinstance(metrics, list) else 0
                    print(f"  {mf.name}: {count} metrics")
            except Exception:
                print(f"  {mf.name}: (error reading)")
    else:
        print("No metrics defined yet")
        print(f"  -> Run: evalyn suggest-metrics --dataset {ds.path}")

    # Eval runs
    print("\n--- EVAL RUNS ---")
    eval_runs = ds.list_eval_runs()
    if eval_runs:
        # Count JSON files in each run dir
        run_count = sum(1 for d in eval_runs if list(d.glob("*.json")))
        print(f"Eval runs: {run_count}")
        # Show latest 3
        for rd in eval_runs[:3]:
            results_file = rd / "results.json"
            if results_file.exists():
                try:
                    with open(results_file, encoding="utf-8") as f:
                        run = json.load(f)
                        created = run.get("created_at", "")[:19]
                        results_count = len(run.get("metric_results", []))
                        print(f"  {rd.name}: {results_count} results ({created})")
                except Exception:
                    pass
    else:
        print("No eval runs yet")
        print(f"  -> Run: evalyn run-eval --dataset {ds.path}")

    # Annotations
    annotations_file = ds.path / "annotations.jsonl"
    print("\n--- ANNOTATIONS ---")
    if annotations_file.exists():
        with open(annotations_file, encoding="utf-8") as f:
            ann_count = sum(1 for _ in f)
        coverage = (
            f"{ann_count}/{ds.item_count}" if ds.item_count > 0 else str(ann_count)
        )
        pct = f" ({ann_count / ds.item_count:.0%})" if ds.item_count > 0 else ""
        print(f"Annotated: {coverage}{pct}")
    else:
        print("No annotations yet")
        print(f"  -> Run: evalyn annotate --dataset {ds.path}")

    # Calibrations
    calibrations_dir = ds.path / "calibrations"
    print("\n--- CALIBRATIONS ---")
    if calibrations_dir.exists():
        cal_count = 0
        metrics_with_cal = []
        for metric_dir in calibrations_dir.iterdir():
            if metric_dir.is_dir():
                cals = list(metric_dir.glob("*.json"))
                if cals:
                    cal_count += len(cals)
                    metrics_with_cal.append(metric_dir.name)
        if cal_count > 0:
            print(f"Calibrations: {cal_count} across {len(metrics_with_cal)} metrics")
            for m in metrics_with_cal[:5]:
                prompts_dir = calibrations_dir / m / "prompts"
                has_prompt = (
                    "(prompt)"
                    if prompts_dir.exists() and list(prompts_dir.glob("*_full.txt"))
                    else ""
                )
                print(f"  {m} {has_prompt}")
        else:
            print("No calibrations yet")
    else:
        print("No calibrations yet")
        print(f"  -> Run: evalyn calibrate --metric-id <metric> --annotations {ds.path / 'annotations.jsonl'} --dataset {ds.path}")

    # Suggested next step
    print(f"\n{'=' * 60}")
    print("SUGGESTED NEXT STEP:")
    if not metric_files:
        print(f"  evalyn suggest-metrics --dataset {ds.path}")
    elif not eval_runs:
        print(f"  evalyn run-eval --dataset {ds.path}")
    elif not annotations_file.exists():
        print(f"  evalyn annotate --dataset {ds.path}")
    elif not calibrations_dir.exists():
        print(f"  evalyn calibrate --metric-id <metric> --annotations {ds.path / 'annotations.jsonl'} --dataset {ds.path}")
    else:
        print("  All steps complete! Consider:")
        print("  - Re-run eval with optimized prompts")
        print(f"  - Generate synthetic data: evalyn simulate --dataset {ds.path}")


def cmd_validate(args: argparse.Namespace) -> None:
    """Validate dataset format and detect potential issues."""
    config = load_config()
    dataset_path = resolve_dataset_path(args.dataset, args.latest, config)

    if not dataset_path:
        fatal_error("No dataset specified", "Use --dataset <path> or --latest")

    if not dataset_path.exists():
        fatal_error(f"Dataset path not found: {dataset_path}")

    # Find dataset file
    if dataset_path.is_dir():
        dataset_file = dataset_path / "dataset.jsonl"
        if not dataset_file.exists():
            dataset_file = dataset_path / "dataset.json"
        dataset_dir = dataset_path
    else:
        dataset_file = dataset_path
        dataset_dir = dataset_path.parent

    if not dataset_file.exists():
        fatal_error(f"Dataset file not found: {dataset_file}")

    print(f"\nValidating: {dataset_file}\n")
    print("-" * 60)

    errors = []
    warnings = []
    stats = {
        "total_items": 0,
        "has_id": 0,
        "has_inputs": 0,
        "has_output": 0,
        "has_expected": 0,
        "has_metadata": 0,
        "unique_ids": set(),
        "duplicate_ids": [],
    }

    # Validate JSON format line by line
    with open(dataset_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: Invalid JSON - {e}")
                continue

            stats["total_items"] += 1

            # Check required fields
            if "id" in item:
                stats["has_id"] += 1
                item_id = item["id"]
                if item_id in stats["unique_ids"]:
                    stats["duplicate_ids"].append((line_num, item_id))
                else:
                    stats["unique_ids"].add(item_id)

            if "inputs" in item or "input" in item:
                stats["has_inputs"] += 1
            else:
                errors.append(f"Line {line_num}: Missing 'inputs' or 'input' field")

            if "output" in item:
                stats["has_output"] += 1

            if "expected" in item or "reference" in item:
                stats["has_expected"] += 1

            if "metadata" in item:
                stats["has_metadata"] += 1

    # Check for duplicates
    for line_num, dup_id in stats["duplicate_ids"]:
        errors.append(f"Line {line_num}: Duplicate ID '{dup_id}'")

    # Print results
    print("VALIDATION RESULTS")
    print("-" * 60)

    total = stats["total_items"]
    print(f"\n  Total items:        {total}")
    print(
        f"  With 'id':          {stats['has_id']} ({100 * stats['has_id'] // max(1, total)}%)"
    )
    print(
        f"  With 'inputs':      {stats['has_inputs']} ({100 * stats['has_inputs'] // max(1, total)}%)"
    )
    print(
        f"  With 'output':      {stats['has_output']} ({100 * stats['has_output'] // max(1, total)}%)"
    )
    print(
        f"  With 'expected':    {stats['has_expected']} ({100 * stats['has_expected'] // max(1, total)}%)"
    )
    print(
        f"  With 'metadata':    {stats['has_metadata']} ({100 * stats['has_metadata'] // max(1, total)}%)"
    )
    print(f"  Unique IDs:         {len(stats['unique_ids'])}")

    # Warnings
    if stats["has_expected"] == 0:
        warnings.append(
            "No 'expected' or 'reference' values found. "
            "Reference-based metrics (ROUGE, BLEU, etc.) will not work."
        )

    if stats["has_id"] < total:
        warnings.append(
            f"{total - stats['has_id']} items missing 'id' field. "
            "Auto-generated IDs will be used."
        )

    # Check for meta.json
    meta_file = dataset_dir / "meta.json"
    if meta_file.exists():
        try:
            with open(meta_file, encoding="utf-8") as f:
                meta = json.load(f)
            print("\n  meta.json:          Found")
            if "project" in meta:
                print(f"  Project:            {meta.get('project')}")
            if "version" in meta:
                print(f"  Version:            {meta.get('version')}")
        except Exception as e:
            errors.append(f"meta.json: Invalid JSON - {e}")
    else:
        warnings.append("No meta.json found. Consider adding metadata.")

    # Check for metrics directory
    metrics_dir = dataset_dir / "metrics"
    if metrics_dir.exists():
        metric_files = list(metrics_dir.glob("*.json"))
        print(f"\n  Metrics files:      {len(metric_files)}")
        for mf in metric_files[:5]:
            print(f"    - {mf.name}")
        if len(metric_files) > 5:
            print(f"    ... and {len(metric_files) - 5} more")
    else:
        warnings.append(
            "No metrics/ directory found. Run 'evalyn suggest-metrics' first."
        )

    # Print warnings and errors
    if warnings:
        print(f"\nWARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"  - {w}")

    if errors:
        print(f"\nERRORS ({len(errors)}):")
        for e in errors[:20]:
            print(f"  - {e}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more errors")
        fatal_error(f"Dataset has {len(errors)} error(s)")
    else:
        print("\nDataset is valid!")


def cmd_analyze(args: argparse.Namespace) -> None:
    """Analyze evaluation results and generate insights.

    Analysis includes:
    - Metric summary: Pass rates and average scores per metric
    - Problem metrics: Metrics with <80% pass rate
    - Multi-fail items: Items failing multiple metrics (likely outliers)
    - Perfect metrics: Metrics with 100% pass rate (may be too lenient)
    - Overall health: Aggregate assessment and recommendations
    """
    from ...storage import SQLiteStorage
    from ...models import EvalRun, Annotation
    from ...calibration import AlignmentMetrics

    output_format = getattr(args, "format", "table")
    config = load_config()
    dataset_path = resolve_dataset_path(args.dataset, args.latest, config)

    # Load eval run
    run = None
    run_id = args.run

    if run_id:
        # Load from storage by ID
        storage = SQLiteStorage()
        run = storage.get_eval_run(run_id)
        if not run:
            fatal_error(f"No eval run found with ID '{run_id}'")
    elif dataset_path:
        # Load latest run from dataset's eval_runs directory
        runs_dir = dataset_path / "eval_runs"
        if runs_dir.exists():
            # Look for results.json in timestamped subdirectories
            run_files = sorted(runs_dir.glob("*/results.json"), reverse=True)
            if not run_files:
                # Fallback to direct json files
                run_files = sorted(runs_dir.glob("*.json"), reverse=True)
            if run_files:
                with open(run_files[0], encoding="utf-8") as f:
                    run_data = json.load(f)
                    run = EvalRun.from_dict(run_data)
                if output_format != "json":
                    print(f"Analyzing latest run: {run_files[0].name}")
        if not run:
            fatal_error("No eval runs found", "Run 'evalyn run-eval' first")
    else:
        fatal_error("Specify --run <run_id> or --dataset <path>")

    # Load annotations if available (for alignment stats)
    # Build lookup: (item_id, metric_id) -> human_label (bool)
    human_labels: dict[tuple[str, str], bool] = {}
    annotations_path = None
    if dataset_path:
        annotations_path = dataset_path / "annotations.jsonl"
    if annotations_path and annotations_path.exists():
        try:
            with open(annotations_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    ann = Annotation.from_dict(data)
                    # Per-metric labels take precedence
                    if ann.metric_labels:
                        for metric_id, ml in ann.metric_labels.items():
                            human_labels[(ann.target_id, metric_id)] = ml.human_label
                    elif ann.label is not None:
                        # Fallback: apply overall label to all metrics for this item
                        human_labels[(ann.target_id, "__overall__")] = bool(ann.label)
        except Exception:
            pass  # Silently ignore annotation loading errors

    # Aggregate metrics
    metrics_stats = {}
    item_failures = {}  # item_id -> list of failed metrics
    alignment_stats: dict[str, AlignmentMetrics] = {}  # metric_id -> alignment

    for mr in run.metric_results:
        item_id = mr.item_id or "unknown"
        metric_id = mr.metric_id
        passed = mr.passed
        score = mr.score

        if metric_id not in metrics_stats:
            metrics_stats[metric_id] = {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "scores": [],
                "failed_items": [],
            }
            alignment_stats[metric_id] = AlignmentMetrics()

        stats = metrics_stats[metric_id]
        stats["total"] += 1
        if passed:
            stats["passed"] += 1
        else:
            stats["failed"] += 1
            stats["failed_items"].append(item_id)
            item_failures.setdefault(item_id, []).append(metric_id)

        if score is not None:
            stats["scores"].append(score)

        # Track alignment if we have human labels
        human_label = human_labels.get(
            (item_id, metric_id), human_labels.get((item_id, "__overall__"))
        )
        if human_label is not None:
            alignment_stats[metric_id].record(predicted=passed, actual=human_label)

    sorted_metrics = sorted(
        metrics_stats.items(),
        key=lambda x: x[1]["failed"] / max(1, x[1]["total"]),
        reverse=True,
    )

    # Build insights
    insights = []

    # Find problematic metrics (< 80% pass rate)
    problem_metrics = [
        (m, s) for m, s in sorted_metrics if s["failed"] / max(1, s["total"]) > 0.2
    ]
    if problem_metrics:
        worst = problem_metrics[0]
        insights.append(
            f"'{worst[0]}' has the highest failure rate "
            f"({worst[1]['failed']}/{worst[1]['total']} failed). "
            f"Consider reviewing the rubric or calibrating."
        )

    # Find items failing multiple metrics
    multi_fail_items = [
        (item, metrics) for item, metrics in item_failures.items() if len(metrics) >= 2
    ]
    if multi_fail_items:
        worst_item = max(multi_fail_items, key=lambda x: len(x[1]))
        insights.append(
            f"Item '{worst_item[0][:20]}...' failed {len(worst_item[1])} metrics: "
            f"{', '.join(worst_item[1][:3])}"
            + ("..." if len(worst_item[1]) > 3 else "")
        )

    # Check for perfect metrics
    perfect_metrics = [
        m for m, s in sorted_metrics if s["failed"] == 0 and s["total"] > 1
    ]
    if perfect_metrics:
        insights.append(
            f"{len(perfect_metrics)} metric(s) have 100% pass rate: "
            f"{', '.join(perfect_metrics[:3])}"
            + ("..." if len(perfect_metrics) > 3 else "")
        )

    # Overall health
    total_evals = sum(s["total"] for s in metrics_stats.values())
    total_passed = sum(s["passed"] for s in metrics_stats.values())
    overall_rate = 100 * total_passed / max(1, total_evals)

    health_status = (
        "GOOD"
        if overall_rate >= 90
        else "MODERATE"
        if overall_rate >= 70
        else "NEEDS_ATTENTION"
    )
    insights.append(
        f"Overall health is {health_status} ({overall_rate:.0f}% pass rate)"
    )

    # JSON output
    if output_format == "json":
        # Build alignment data for JSON
        alignment_data = {}
        for m, _ in sorted_metrics:
            a = alignment_stats[m]
            if a.total > 0:
                alignment_data[m] = {
                    "n": a.total,
                    "accuracy": a.accuracy,
                    "precision": a.precision,
                    "recall": a.recall,
                    "f1": a.f1,
                    "cohens_kappa": a.cohens_kappa,
                    "confusion": {
                        "tp": a.true_positive,
                        "tn": a.true_negative,
                        "fp": a.false_positive,
                        "fn": a.false_negative,
                    },
                }
        result = {
            "run_id": run.id,
            "dataset_name": run.dataset_name,
            "created_at": run.created_at.isoformat() if run.created_at else None,
            "total_results": len(run.metric_results),
            "overall_pass_rate": overall_rate,
            "health_status": health_status,
            "metrics": {
                m: {
                    "total": s["total"],
                    "passed": s["passed"],
                    "failed": s["failed"],
                    "pass_rate": 100 * s["passed"] / max(1, s["total"]),
                    "avg_score": sum(s["scores"]) / len(s["scores"])
                    if s["scores"]
                    else None,
                    "failed_items": s["failed_items"][:10],
                }
                for m, s in sorted_metrics
            },
            "alignment": alignment_data if alignment_data else None,
            "problem_metrics": [m for m, _ in problem_metrics],
            "perfect_metrics": perfect_metrics,
            "multi_fail_items": [
                {"item_id": item, "failed_metrics": metrics}
                for item, metrics in multi_fail_items[:10]
            ],
            "insights": insights,
        }
        print(json.dumps(result, indent=2, default=str))
        return

    # Table output
    print(f"\n{'=' * 70}")
    print("  EVALUATION ANALYSIS")
    print(f"{'=' * 70}")
    print(f"\nRun ID:      {run.id}")
    print(f"Dataset:     {run.dataset_name}")
    print(f"Items:       {len(run.metric_results)}")
    print(f"Created:     {run.created_at}")

    # Print metric summary
    print(f"\n{'=' * 70}")
    print("  METRIC SUMMARY")
    print(f"{'=' * 70}\n")

    for metric_id, stats in sorted_metrics:
        total = stats["total"]
        passed = stats["passed"]
        failed = stats["failed"]
        pass_rate = 100 * passed / max(1, total)
        status = "PASS" if failed == 0 else "FAIL"

        avg_score = ""
        if stats["scores"]:
            avg = sum(stats["scores"]) / len(stats["scores"])
            avg_score = f"  avg={avg:.2f}"

        print(
            f"  [{status}] {metric_id:<30} {passed}/{total} passed ({pass_rate:.0f}%){avg_score}"
        )

    # Alignment stats section (only if annotations exist)
    has_alignment = any(a.total > 0 for a in alignment_stats.values())
    if has_alignment:
        print(f"\n{'=' * 70}")
        print("  ALIGNMENT STATS (vs human annotations)")
        print(f"{'=' * 70}\n")
        print(
            f"  {'Metric':<25} {'N':>4} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Kappa':>6}"
        )
        print(f"  {'-' * 61}")
        for metric_id, _ in sorted_metrics:
            align = alignment_stats[metric_id]
            if align.total == 0:
                continue
            print(
                f"  {metric_id:<25} {align.total:>4} "
                f"{align.accuracy:>5.0%} {align.precision:>5.0%} "
                f"{align.recall:>5.0%} {align.f1:>5.0%} {align.cohens_kappa:>6.2f}"
            )
        # Summary row
        total_align = AlignmentMetrics(
            true_positive=sum(a.true_positive for a in alignment_stats.values()),
            true_negative=sum(a.true_negative for a in alignment_stats.values()),
            false_positive=sum(a.false_positive for a in alignment_stats.values()),
            false_negative=sum(a.false_negative for a in alignment_stats.values()),
        )
        if total_align.total > 0:
            print(f"  {'-' * 61}")
            print(
                f"  {'OVERALL':<25} {total_align.total:>4} "
                f"{total_align.accuracy:>5.0%} {total_align.precision:>5.0%} "
                f"{total_align.recall:>5.0%} {total_align.f1:>5.0%} {total_align.cohens_kappa:>6.2f}"
            )

    # Insights section
    print(f"\n{'=' * 70}")
    print("  INSIGHTS")
    print(f"{'=' * 70}\n")

    for insight in insights:
        print(f"  {insight}\n")

    # Recommendations
    print(f"{'=' * 70}")
    print("  RECOMMENDATIONS")
    print(f"{'=' * 70}\n")

    if problem_metrics:
        print("  1. Run 'evalyn annotate' to provide human labels for failed items")
        print("  2. Run 'evalyn calibrate' to improve metric alignment")

    if not item_failures:
        print("  All items passed! Consider adding more challenging test cases.")
        print("    Run 'evalyn simulate --modes outlier' to generate edge cases.")

    print()

    # Show hint for next step
    dataset_flag = f"--dataset {dataset_path}" if dataset_path else "--latest"

    # Filter problem metrics to only subjective ones (calibration only works for LLM judges)
    from ...metrics.subjective import SUBJECTIVE_REGISTRY

    subjective_ids = {m["id"] for m in SUBJECTIVE_REGISTRY}
    subjective_problem_metrics = [
        (m, s) for m, s in problem_metrics if m in subjective_ids
    ]

    if subjective_problem_metrics:
        # Subjective metrics can be calibrated - suggest annotation workflow
        worst_metric = subjective_problem_metrics[0][0]
        print_hint(
            f"To calibrate '{worst_metric}', first annotate: evalyn annotate {dataset_flag}",
            quiet=getattr(args, "quiet", False),
            format=output_format,
        )
    elif problem_metrics or multi_fail_items:
        # Objective metrics or items failing multiple metrics - suggest annotation
        print_hint(
            f"To annotate failing items, run: evalyn annotate {dataset_flag}",
            quiet=getattr(args, "quiet", False),
            format=output_format,
        )
    else:
        print_hint(
            f"To see trends over time, run: evalyn trend --project {run.dataset_name}",
            quiet=getattr(args, "quiet", False),
            format=output_format,
        )


def cmd_compare(args: argparse.Namespace) -> None:
    """Compare two evaluation runs side-by-side.

    Useful for:
    - Before/after calibration: Did the optimized prompt improve?
    - A/B testing: Which model version performs better?
    - Regression testing: Did changes break anything?

    Shows per-metric pass rate changes and overall delta.
    """
    from ...storage import SQLiteStorage
    from ...models import EvalRun

    storage = SQLiteStorage()

    # Validate args: need either (--run1 and --run2) or --latest
    use_latest = getattr(args, "latest", False)
    has_runs = args.run1 and args.run2

    if not use_latest and not has_runs:
        fatal_error(
            "Either provide --run1 and --run2, or use --latest",
            "evalyn compare --run1 <id> --run2 <id>",
        )

    # If --latest with --dataset, compare two most recent runs
    if use_latest:
        config = load_config()
        dataset_path = resolve_dataset_path(
            getattr(args, "dataset", None), True, config
        )
        if not dataset_path:
            fatal_error("No dataset found", "Use --dataset <path>")

        runs_dir = dataset_path / "eval_runs"
        if not runs_dir.exists():
            fatal_error(f"No eval_runs directory in {dataset_path}")

        # Find the two most recent runs
        run_files = sorted(runs_dir.glob("*/results.json"), reverse=True)
        if not run_files:
            run_files = sorted(runs_dir.glob("*.json"), reverse=True)

        if len(run_files) < 2:
            fatal_error(f"Need at least 2 runs to compare. Found: {len(run_files)}")

        with open(run_files[0], encoding="utf-8") as f:
            run2 = EvalRun.from_dict(json.load(f))
        with open(run_files[1], encoding="utf-8") as f:
            run1 = EvalRun.from_dict(json.load(f))

        print(f"Comparing two most recent runs from {dataset_path.name}")
    else:
        # Original behavior: load from --run1 and --run2
        # Load run 1
        run1 = None
        if args.run1:
            run1 = storage.get_eval_run(args.run1)
            if not run1:
                # Try loading from file
                run1_path = Path(args.run1)
                if run1_path.exists():
                    with open(run1_path, encoding="utf-8") as f:
                        data = json.load(f)
                        run1 = EvalRun(**data)

        if not run1:
            fatal_error(f"Could not load run1: {args.run1}")

        # Load run 2
        run2 = None
        if args.run2:
            run2 = storage.get_eval_run(args.run2)
            if not run2:
                run2_path = Path(args.run2)
                if run2_path.exists():
                    with open(run2_path, encoding="utf-8") as f:
                        data = json.load(f)
                        run2 = EvalRun(**data)

        if not run2:
            fatal_error(f"Could not load run2: {args.run2}")

    print(f"\n{'=' * 70}")
    print("  EVALUATION COMPARISON")
    print(f"{'=' * 70}")

    print(f"\n  Run 1: {run1.id[:12]}... ({run1.dataset_name})")
    print(f"  Run 2: {run2.id[:12]}... ({run2.dataset_name})")

    # Build metric stats for each run
    def get_metric_stats(run):
        stats = {}
        for mr in run.metric_results:
            metric_id = (
                mr.metric_id
                if hasattr(mr, "metric_id")
                else mr.get("metric_id", "unknown")
            )
            if metric_id not in stats:
                stats[metric_id] = {"total": 0, "passed": 0, "scores": []}
            stats[metric_id]["total"] += 1
            passed = mr.passed if hasattr(mr, "passed") else mr.get("passed", True)
            if passed:
                stats[metric_id]["passed"] += 1
            score = mr.score if hasattr(mr, "score") else mr.get("score")
            if score is not None:
                stats[metric_id]["scores"].append(score)
        return stats

    stats1 = get_metric_stats(run1)
    stats2 = get_metric_stats(run2)

    # Get all metric IDs
    all_metrics = set(stats1.keys()) | set(stats2.keys())

    print(f"\n{'=' * 70}")
    print("  METRIC COMPARISON")
    print(f"{'=' * 70}\n")

    print(f"  {'Metric':<25} {'Run 1':>12} {'Run 2':>12} {'Delta':>12}")
    print(f"  {'-' * 25} {'-' * 12} {'-' * 12} {'-' * 12}")

    improvements = 0
    regressions = 0

    for metric_id in sorted(all_metrics):
        s1 = stats1.get(metric_id, {"total": 0, "passed": 0})
        s2 = stats2.get(metric_id, {"total": 0, "passed": 0})

        rate1 = 100 * s1["passed"] / max(1, s1["total"]) if s1["total"] > 0 else None
        rate2 = 100 * s2["passed"] / max(1, s2["total"]) if s2["total"] > 0 else None

        r1_str = f"{rate1:.0f}%" if rate1 is not None else "N/A"
        r2_str = f"{rate2:.0f}%" if rate2 is not None else "N/A"

        delta = ""
        if rate1 is not None and rate2 is not None:
            diff = rate2 - rate1
            if diff > 0:
                delta = f"+{diff:.0f}%"
                improvements += 1
            elif diff < 0:
                delta = f"{diff:.0f}%"
                regressions += 1
            else:
                delta = "="

        print(f"  {metric_id:<25} {r1_str:>12} {r2_str:>12} {delta:>12}")

    # Summary
    print(f"\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}\n")

    total1 = sum(s["passed"] for s in stats1.values())
    total2 = sum(s["passed"] for s in stats2.values())
    all1 = sum(s["total"] for s in stats1.values())
    all2 = sum(s["total"] for s in stats2.values())

    overall1 = 100 * total1 / max(1, all1)
    overall2 = 100 * total2 / max(1, all2)
    overall_delta = overall2 - overall1

    print("  Overall pass rate:")
    print(f"    Run 1: {overall1:.1f}% ({total1}/{all1})")
    print(f"    Run 2: {overall2:.1f}% ({total2}/{all2})")

    if overall_delta > 0:
        print(f"    Change: +{overall_delta:.1f}% IMPROVED")
    elif overall_delta < 0:
        print(f"    Change: {overall_delta:.1f}% REGRESSED")
    else:
        print("    Change: No change")

    print(f"\n  Metrics improved:  {improvements}")
    print(f"  Metrics regressed: {regressions}")
    print(f"  Metrics unchanged: {len(all_metrics) - improvements - regressions}")
    print()

    # Show helpful hints based on results
    if regressions > improvements:
        print_hint(
            "Run 2 shows regression. Consider: evalyn analyze --run " + run2.id,
            quiet=getattr(args, "quiet", False),
        )
    elif improvements > 0:
        print_hint(
            f"Run 2 shows improvement. To see trends: evalyn trend --project {run2.dataset_name}",
            quiet=getattr(args, "quiet", False),
        )


def cmd_trend(args: argparse.Namespace) -> None:
    """Show evaluation trends over time for a project."""
    from ...storage import SQLiteStorage
    from ...analysis import analyze_trends, generate_trend_text_report

    storage = SQLiteStorage()

    # Resolve project name from --project or --latest
    project_name = args.project
    use_latest = getattr(args, "latest", False)

    if use_latest and not project_name:
        config = load_config()
        dataset_path = resolve_dataset_path(
            getattr(args, "dataset", None), True, config
        )
        if dataset_path:
            project_name = dataset_path.name
        else:
            fatal_error("No dataset found", "Use --project or --dataset with --latest")

    if not project_name:
        fatal_error("--project is required", "Or use --latest with --dataset")

    # Get runs for the project
    runs = storage.list_eval_runs_by_project(project_name, limit=args.limit)

    if not runs:
        fatal_error(
            f"No eval runs found for project: {project_name}",
            "Run 'evalyn show-projects' to see available projects",
        )

    # Analyze trends
    trend = analyze_trends(runs)

    # Output based on format
    output_format = getattr(args, "format", "table")

    if output_format == "json":
        result = {
            "project": trend.project_name,
            "runs_analyzed": len(trend.runs),
            "runs": [
                {
                    "id": run.run_id,
                    "created_at": run.created_at,
                    "total_items": run.total_items,
                    "overall_pass_rate": run.overall_pass_rate,
                    "metrics": {
                        m: {
                            "pass_rate": ms.pass_rate,
                            "avg_score": ms.avg_score,
                            "count": ms.count,
                        }
                        for m, ms in run.metric_stats.items()
                    },
                }
                for run in trend.runs
            ],
            "trends": {
                "overall_delta": trend.overall_delta,
                "improving_metrics": trend.improving_metrics,
                "regressing_metrics": trend.regressing_metrics,
                "stable_metrics": trend.stable_metrics,
            },
        }
        print(json.dumps(result, indent=2))
    else:
        # ASCII table output
        report = generate_trend_text_report(trend)
        print(report)

        # Show helpful hints based on trend results
        if trend.regressing_metrics:
            print_hint(
                f"Metrics regressing: {', '.join(trend.regressing_metrics[:3])}. Consider calibration.",
                quiet=getattr(args, "quiet", False),
            )
        elif trend.overall_delta and trend.overall_delta > 0:
            print_hint(
                "Overall trend is improving. Keep up the good work!",
                quiet=getattr(args, "quiet", False),
            )


def register_commands(subparsers) -> None:
    """Register analysis commands."""
    # status
    p = subparsers.add_parser(
        "status",
        help="Show status of a dataset (items, metrics, runs, annotations, calibrations)",
    )
    p.add_argument("--dataset", help="Path to dataset directory")
    p.add_argument(
        "--latest", action="store_true", help="Use the most recently modified dataset"
    )
    p.set_defaults(func=cmd_status)

    # validate
    p = subparsers.add_parser(
        "validate", help="Validate dataset format and detect potential issues"
    )
    p.add_argument("--dataset", help="Path to dataset directory or file")
    p.add_argument(
        "--latest", action="store_true", help="Use the most recently modified dataset"
    )
    p.set_defaults(func=cmd_validate)

    # analyze
    p = subparsers.add_parser(
        "analyze", help="Analyze evaluation results and generate insights"
    )
    p.add_argument("--run", help="Eval run ID to analyze")
    p.add_argument("--dataset", help="Dataset path (uses latest run from eval_runs/)")
    p.add_argument(
        "--latest", action="store_true", help="Use the most recently modified dataset"
    )
    p.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    p.set_defaults(func=cmd_analyze)

    # compare
    p = subparsers.add_parser(
        "compare", help="Compare two evaluation runs side-by-side"
    )
    p.add_argument("--run1", help="First eval run ID or path to run JSON file")
    p.add_argument("--run2", help="Second eval run ID or path to run JSON file")
    p.add_argument("--dataset", help="Dataset path (used with --latest)")
    p.add_argument(
        "--latest",
        action="store_true",
        help="Compare the two most recent runs from the dataset",
    )
    p.set_defaults(func=cmd_compare)

    # trend
    p = subparsers.add_parser(
        "trend", help="Show evaluation trends over time for a project"
    )
    p.add_argument(
        "--project",
        help="Project/dataset name to analyze trends for",
    )
    p.add_argument(
        "--dataset",
        help="Dataset path (used with --latest to infer project name)",
    )
    p.add_argument(
        "--latest",
        action="store_true",
        help="Use the most recently modified dataset as project name",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of runs to analyze (default: 20)",
    )
    p.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    p.set_defaults(func=cmd_trend)


__all__ = [
    "cmd_status",
    "cmd_validate",
    "cmd_analyze",
    "cmd_compare",
    "cmd_trend",
    "register_commands",
]
