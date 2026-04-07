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
import re
from pathlib import Path

from ..utils.command_common import load_eval_run_for_command, resolve_dataset_dir_and_file
from ..utils.config import load_config, resolve_dataset_path
from ..utils.dataset_resolver import get_dataset
from ..utils.errors import fatal_error
from ..utils.hints import HintCollector


def _print_run_summary(run_name: str, results_file: Path) -> None:
    """Print a one-line run summary without parsing the full JSON.

    Reads the file once as bytes. Extracts created_at from the header via
    regex and counts metric results by scanning for "metric_id" keys.
    This avoids deserializing multi-hundred-KB result files.
    """
    raw = results_file.read_bytes()
    m = re.search(rb'"created_at"\s*:\s*"([^"]{1,25})"', raw[:512])
    created = m.group(1).decode("utf-8", errors="replace")[:19] if m else ""
    results_count = raw.count(b'"metric_id"')
    print(f"  {run_name}: {results_count} results ({created})")


def cmd_status(args: argparse.Namespace) -> None:
    """Show status of a dataset including items, metrics, runs, annotations, and calibrations."""
    from ..utils.rich import banner, footer, kv, section, status_icon

    ds = get_dataset(
        getattr(args, "dataset", None),
        getattr(args, "latest", False),
        require=True,
    )

    print(banner("DATASET STATUS"))
    info_pairs = [
        ("Dataset", ds.name),
        ("Path", str(ds.path)),
        ("Items", str(ds.item_count)),
    ]
    if ds.project:
        info_pairs.append(("Project", ds.project))
    if ds.version:
        info_pairs.append(("Version", ds.version))
    print(kv(info_pairs))

    # Gather pipeline info for summary
    metric_files = ds.list_metrics_files()
    eval_runs = ds.list_eval_runs()
    annotations_file = ds.path / "annotations.jsonl"
    has_annotations = annotations_file.exists()
    calibrations_dir = ds.path / "calibrations"
    has_calibrations = calibrations_dir.exists() and any(
        calibrations_dir.iterdir()
    ) if calibrations_dir.exists() else False

    # Metrics detail
    if metric_files:
        metrics_info = f"{len(metric_files)} set(s)"
    else:
        metrics_info = "none"

    # Eval runs detail
    if eval_runs:
        run_count = sum(1 for d in eval_runs if list(d.glob("*.json")))
        runs_info = f"{run_count} run(s)"
    else:
        runs_info = "none"

    # Annotations detail
    if has_annotations:
        with open(annotations_file, encoding="utf-8") as f:
            ann_count = sum(1 for _ in f)
        coverage = (
            f"{ann_count}/{ds.item_count}" if ds.item_count > 0 else str(ann_count)
        )
        pct = f" ({ann_count / ds.item_count:.0%})" if ds.item_count > 0 else ""
        ann_info = f"{coverage}{pct}"
    else:
        ann_info = "none"

    # Calibrations detail
    cal_count = 0
    metrics_with_cal: list[str] = []
    if has_calibrations:
        for metric_dir in calibrations_dir.iterdir():
            if metric_dir.is_dir():
                cals = list(metric_dir.glob("*.json"))
                if cals:
                    cal_count += len(cals)
                    metrics_with_cal.append(metric_dir.name)
    if cal_count > 0:
        cal_info = f"{cal_count} across {len(metrics_with_cal)} metric(s)"
    else:
        cal_info = "none"
        has_calibrations = False

    # Pipeline summary
    print()
    print(section("PIPELINE"))
    print(f"  {status_icon(bool(metric_files))}  Metrics       {metrics_info}")
    print(f"  {status_icon(bool(eval_runs))}  Eval Runs     {runs_info}")
    print(f"  {status_icon(has_annotations)}  Annotations   {ann_info}")
    print(f"  {status_icon(has_calibrations)}  Calibrations  {cal_info}")

    # Metric details
    if metric_files:
        print()
        print(section("METRICS"))
        for mf in metric_files:
            try:
                with open(mf, encoding="utf-8") as f:
                    metrics = json.load(f)
                    count = len(metrics) if isinstance(metrics, list) else 0
                    print(f"  {mf.name}: {count} metrics")
            except Exception:
                print(f"  {mf.name}: (error reading)")

    # Eval run details
    if eval_runs:
        print()
        print(section("EVAL RUNS"))
        for rd in eval_runs[:3]:
            results_file = rd / "results.json"
            if results_file.exists():
                try:
                    _print_run_summary(rd.name, results_file)
                except Exception:
                    pass

    # Calibration details
    if metrics_with_cal:
        print()
        print(section("CALIBRATIONS"))
        for m in metrics_with_cal[:5]:
            prompts_dir = calibrations_dir / m / "prompts"
            has_prompt = (
                "(prompt)"
                if prompts_dir.exists() and list(prompts_dir.glob("*_full.txt"))
                else ""
            )
            print(f"  {m} {has_prompt}")

    # Suggested next step
    print()
    print(section("NEXT STEP"))
    if not metric_files:
        next_cmd = (f"evalyn suggest-metrics --dataset {ds.path}", "Define metrics")
    elif not eval_runs:
        next_cmd = (f"evalyn run-eval --dataset {ds.path}", "Run evaluation")
    elif not has_annotations:
        next_cmd = (f"evalyn annotate --dataset {ds.path}", "Add annotations")
    elif not has_calibrations:
        next_cmd = (
            f"evalyn calibrate --metric-id <metric> --annotations {ds.path / 'annotations.jsonl'} --dataset {ds.path}",
            "Calibrate metrics",
        )
    else:
        next_cmd = (f"evalyn run-eval --dataset {ds.path}", "All steps complete - re-run eval with optimized prompts")
    print(footer([next_cmd]))


def cmd_validate(args: argparse.Namespace) -> None:
    """Validate dataset format and detect potential issues."""
    from ..utils.rich import banner, icon, kv, section, status_icon

    config = load_config()
    dataset_dir, dataset_file = resolve_dataset_dir_and_file(
        args.dataset, args.latest, config=config
    )

    print(banner("VALIDATION"))
    print(kv([
        ("File", str(dataset_file)),
        ("Directory", str(dataset_dir)),
    ]))

    errors: list[str] = []
    warnings: list[str] = []
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

    # Print field summary with status icons
    total = stats["total_items"]

    def _field_icon(count: int, total_n: int) -> str:
        """Return pass for 100%, warn for 0%, info otherwise."""
        if total_n == 0:
            return icon("info")
        pct = count / total_n
        if pct >= 1.0:
            return icon("pass")
        if pct <= 0.0:
            return icon("warn")
        return icon("info")

    print()
    print(section("FIELDS"))
    print(f"  Total items:        {total}")
    fields = [
        ("id", stats["has_id"]),
        ("inputs", stats["has_inputs"]),
        ("output", stats["has_output"]),
        ("expected", stats["has_expected"]),
        ("metadata", stats["has_metadata"]),
    ]
    for field_name, count in fields:
        pct = 100 * count // max(1, total)
        print(f"  {_field_icon(count, total)}  {field_name:<16} {count}/{total} ({pct}%)")
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
            meta_pairs = [("meta.json", "Found")]
            if "project" in meta:
                meta_pairs.append(("Project", str(meta.get("project"))))
            if "version" in meta:
                meta_pairs.append(("Version", str(meta.get("version"))))
            print()
            print(kv(meta_pairs))
        except Exception as e:
            errors.append(f"meta.json: Invalid JSON - {e}")
    else:
        warnings.append("No meta.json found. Consider adding metadata.")

    # Check for metrics directory
    metrics_dir = dataset_dir / "metrics"
    if metrics_dir.exists():
        metric_files = list(metrics_dir.glob("*.json"))
        print()
        print(section("METRICS"))
        print(f"  Metrics files: {len(metric_files)}")
        for mf in metric_files[:5]:
            print(f"    - {mf.name}")
        if len(metric_files) > 5:
            print(f"    ... and {len(metric_files) - 5} more")
    else:
        warnings.append(
            "No metrics/ directory found. Run 'evalyn suggest-metrics' first."
        )

    # Print warnings
    if warnings:
        print()
        print(section(f"WARNINGS ({len(warnings)})"))
        for w in warnings:
            print(f"  {icon('warn')} {w}")

    # Print errors and final result
    if errors:
        print()
        print(section(f"ERRORS ({len(errors)})"))
        for e in errors[:20]:
            print(f"  {icon('fail')} {e}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more errors")

    print()
    print(section("RESULT"))
    if errors:
        print(f"  {icon('fail')} Dataset is INVALID - {len(errors)} error(s) found")
        fatal_error(f"Dataset has {len(errors)} error(s)")
    else:
        print(f"  {icon('pass')} Dataset is valid!")


def _load_analysis_run(
    args: argparse.Namespace,
    *,
    dataset_path,
    output_format: str,
):
    """Load the eval run to analyze from storage or dataset folder."""
    loaded = load_eval_run_for_command(
        run_id=getattr(args, "run", None),
        dataset_path=dataset_path,
        error_hint="Specify --run <run_id> or --dataset <path>",
    )
    if loaded.run_file_path and output_format != "json":
        print(f"Analyzing latest run: {loaded.run_file_path.name}")
    return loaded.run


def _load_analysis_human_labels(dataset_path) -> dict[tuple[str, str], bool]:
    """Load optional human labels for alignment calculations."""
    from ...models import Annotation

    human_labels: dict[tuple[str, str], bool] = {}
    annotations_path = dataset_path / "annotations.jsonl" if dataset_path else None
    if not annotations_path or not annotations_path.exists():
        return human_labels

    try:
        with open(annotations_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                ann = Annotation.from_dict(data)
                if ann.metric_labels:
                    for metric_id, ml in ann.metric_labels.items():
                        human_labels[(ann.target_id, metric_id)] = ml.human_label
                elif ann.label is not None:
                    human_labels[(ann.target_id, "__overall__")] = bool(ann.label)
    except Exception:
        pass
    return human_labels


def _aggregate_analysis_stats(run, human_labels: dict[tuple[str, str], bool]) -> dict:
    """Aggregate metric, failure, and alignment stats for analysis output.

    Delegates metric/item statistics to analysis.core.analyze_run and adds
    alignment computation against human labels.
    """
    from ...analysis.core import analyze_run
    from ...calibration import AlignmentMetrics

    # Use the canonical analysis engine for metric stats
    run_analysis = analyze_run(run.as_dict())

    # Convert RunAnalysis into the dict format used by downstream output code
    metrics_stats = {}
    item_failures = {}
    for metric_id, ms in run_analysis.metric_stats.items():
        metrics_stats[metric_id] = {
            "total": ms.count,
            "passed": ms.passed,
            "failed": ms.failed,
            "scores": list(ms.scores),
            "failed_items": [],
        }
    for item_id, ist in run_analysis.item_stats.items():
        for mid, mr_info in ist.metric_results.items():
            if mr_info.get("passed") is False:
                metrics_stats.get(mid, {}).setdefault("failed_items", []).append(item_id)
                item_failures.setdefault(item_id, []).append(mid)

    # Compute alignment against human annotations (unique to this function)
    alignment_stats: dict[str, AlignmentMetrics] = {}
    for mr in run.metric_results:
        item_id = mr.item_id or "unknown"
        metric_id = mr.metric_id
        if metric_id not in alignment_stats:
            alignment_stats[metric_id] = AlignmentMetrics()
        human_label = human_labels.get(
            (item_id, metric_id), human_labels.get((item_id, "__overall__"))
        )
        if human_label is not None:
            alignment_stats[metric_id].record(predicted=mr.passed, actual=human_label)

    sorted_metrics = sorted(
        metrics_stats.items(),
        key=lambda x: x[1]["failed"] / max(1, x[1]["total"]),
        reverse=True,
    )
    return {
        "metrics_stats": metrics_stats,
        "item_failures": item_failures,
        "alignment_stats": alignment_stats,
        "sorted_metrics": sorted_metrics,
        "run_analysis": run_analysis,
    }


def _build_analysis_insights(sorted_metrics, item_failures: dict[str, list[str]]) -> dict:
    """Build insights and high-level summary metrics for the analysis report."""
    insights = []

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

    perfect_metrics = [
        m for m, s in sorted_metrics if s["failed"] == 0 and s["total"] > 1
    ]
    if perfect_metrics:
        insights.append(
            f"{len(perfect_metrics)} metric(s) have 100% pass rate: "
            f"{', '.join(perfect_metrics[:3])}"
            + ("..." if len(perfect_metrics) > 3 else "")
        )

    total_evals = sum(s["total"] for _, s in sorted_metrics)
    total_passed = sum(s["passed"] for _, s in sorted_metrics)
    overall_rate = 100 * total_passed / max(1, total_evals)
    health_status = (
        "GOOD"
        if overall_rate >= 90
        else "MODERATE"
        if overall_rate >= 70
        else "NEEDS_ATTENTION"
    )
    insights.append(f"Overall health is {health_status} ({overall_rate:.0f}% pass rate)")

    return {
        "insights": insights,
        "problem_metrics": problem_metrics,
        "multi_fail_items": multi_fail_items,
        "perfect_metrics": perfect_metrics,
        "overall_rate": overall_rate,
        "health_status": health_status,
    }


def _build_analysis_alignment_json(sorted_metrics, alignment_stats: dict) -> dict:
    """Serialize alignment metrics for JSON output."""
    alignment_data = {}
    for m, _ in sorted_metrics:
        a = alignment_stats[m]
        if a.total <= 0:
            continue
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
    return alignment_data


def _print_analysis_table_output(
    *,
    run,
    sorted_metrics,
    alignment_stats: dict,
    insights: list[str],
    problem_metrics,
    item_failures: dict[str, list[str]],
    run_analysis=None,
) -> None:
    """Print table-formatted evaluation analysis."""
    from ...calibration import AlignmentMetrics

    from ..utils.rich import banner, section, kv, icon, status_icon, progress_bar, footer

    item_count = len(set(mr.item_id for mr in run.metric_results))

    print(banner("EVALUATION ANALYSIS"))
    print(kv([
        ("Run", f"{run.id[:8]}  ({run.dataset_name}, {item_count} items)"),
        ("Created", str(run.created_at)[:16]),
    ]))

    print(f"\n{section('METRIC SUMMARY')}\n")

    total_evals = 0
    total_passed = 0
    for metric_id, stats in sorted_metrics:
        total = stats["total"]
        passed = stats["passed"]
        total_evals += total
        total_passed += passed
        rate = 100 * passed / max(1, total)
        ic = status_icon(stats["failed"] == 0)
        avg = sum(stats["scores"]) / len(stats["scores"]) if stats["scores"] else 0
        print(f"  {ic} {metric_id:<25} {passed:>3}/{total:<3} {rate:>5.1f}%  avg {avg:.2f}")

    overall_rate = 100 * total_passed / max(1, total_evals)
    health_status = (
        "GOOD"
        if overall_rate >= 90
        else "MODERATE"
        if overall_rate >= 70
        else "NEEDS_ATTENTION"
    )
    print(f"\n  Overall: {progress_bar(total_passed, total_evals, label=health_status)}")

    has_alignment = any(a.total > 0 for a in alignment_stats.values())
    if has_alignment:
        print(f"\n{section('ALIGNMENT STATS (vs human annotations)')}\n")
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

    print(f"\n{section('INSIGHTS')}\n")
    for insight in insights:
        print(f"  {icon('info')} {insight}\n")

    print(section("NEXT STEPS"))

    if problem_metrics:
        print(f"  {icon('next')} Run 'evalyn annotate' to provide human labels for failed items")
        print(f"  {icon('next')} Run 'evalyn calibrate' to improve metric alignment")

    if not item_failures:
        print(f"  {icon('info')} All items passed! Consider adding more challenging test cases.")
        print(f"    Run 'evalyn simulate --modes outlier' to generate edge cases.")

    # Key findings from insights engine
    if run_analysis is not None:
        try:
            from ...analysis.insights import (
                compute_metric_correlations,
                analyze_score_distributions,
            )

            findings = []
            for c in compute_metric_correlations(run_analysis):
                findings.append(
                    f"{c.metric_a} <-> {c.metric_b}: {c.relationship} (r={c.pearson})"
                )
            for d in analyze_score_distributions(run_analysis):
                findings.append(f"{d.metric_id}: {d.shape} - {d.finding}")

            if findings:
                print(f"\n{section('KEY FINDINGS')}\n")
                for finding in findings[:5]:
                    print(f"  {icon('info')} {finding}")
                print(f"\n  Run 'evalyn insights' for full diagnostic report.")
        except (KeyError, ValueError, TypeError):
            pass

    print()


def _print_analysis_next_hint(
    *,
    run,
    dataset_path,
    output_format: str,
    problem_metrics,
    multi_fail_items,
    quiet: bool,
) -> None:
    """Print post-analysis next-step hints."""
    dataset_flag = f"--dataset {dataset_path}" if dataset_path else "--latest"
    hints = HintCollector(quiet=quiet, format=output_format)

    from ...metrics.subjective import SUBJECTIVE_REGISTRY

    subjective_ids = {m["id"] for m in SUBJECTIVE_REGISTRY}
    subjective_problem_metrics = [
        (m, s) for m, s in problem_metrics if m in subjective_ids
    ]

    if subjective_problem_metrics:
        worst_metric = subjective_problem_metrics[0][0]
        hints.add(f"evalyn annotate {dataset_flag}", f"Annotate to calibrate '{worst_metric}'")
    if problem_metrics or multi_fail_items:
        hints.add(f"evalyn cluster-failures {dataset_flag}", "Cluster failures by pattern")
    hints.add(f"evalyn trend --project {run.dataset_name}", "See trends over time")
    hints.render()


def cmd_analyze(args: argparse.Namespace) -> None:
    """Analyze evaluation results and generate insights.

    Analysis includes:
    - Metric summary: Pass rates and average scores per metric
    - Problem metrics: Metrics with <80% pass rate
    - Multi-fail items: Items failing multiple metrics (likely outliers)
    - Perfect metrics: Metrics with 100% pass rate (may be too lenient)
    - Overall health: Aggregate assessment and recommendations
    """
    output_format = getattr(args, "format", "table")
    config = load_config()
    dataset_path = resolve_dataset_path(args.dataset, args.latest, config)
    run = _load_analysis_run(args, dataset_path=dataset_path, output_format=output_format)
    human_labels = _load_analysis_human_labels(dataset_path)
    aggregated = _aggregate_analysis_stats(run, human_labels)
    sorted_metrics = aggregated["sorted_metrics"]
    item_failures = aggregated["item_failures"]
    alignment_stats = aggregated["alignment_stats"]

    insight_data = _build_analysis_insights(sorted_metrics, item_failures)
    insights = insight_data["insights"]
    problem_metrics = insight_data["problem_metrics"]
    multi_fail_items = insight_data["multi_fail_items"]
    perfect_metrics = insight_data["perfect_metrics"]
    overall_rate = insight_data["overall_rate"]
    health_status = insight_data["health_status"]

    # JSON output
    if output_format == "json":
        alignment_data = _build_analysis_alignment_json(sorted_metrics, alignment_stats)
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

    _print_analysis_table_output(
        run=run,
        sorted_metrics=sorted_metrics,
        alignment_stats=alignment_stats,
        insights=insights,
        problem_metrics=problem_metrics,
        item_failures=item_failures,
        run_analysis=aggregated.get("run_analysis"),
    )
    _print_analysis_next_hint(
        run=run,
        dataset_path=dataset_path,
        output_format=output_format,
        problem_metrics=problem_metrics,
        multi_fail_items=multi_fail_items,
        quiet=getattr(args, "quiet", False),
    )


def cmd_compare(args: argparse.Namespace) -> None:
    """Compare two evaluation runs side-by-side.

    Useful for:
    - Before/after calibration: Did the optimized prompt improve?
    - A/B testing: Which model version performs better?
    - Regression testing: Did changes break anything?

    Shows per-metric pass rate changes and overall delta.
    """
    from ...analysis.core import find_eval_runs
    from ...decorators import get_default_tracer
    from ...models import EvalRun

    storage = get_default_tracer().storage

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

        run_files = find_eval_runs(dataset_path)

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
                        run1 = EvalRun.from_dict(data)

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
                        run2 = EvalRun.from_dict(data)

        if not run2:
            fatal_error(f"Could not load run2: {args.run2}")

    from ..utils.rich import banner, section, kv, icon, status_icon, progress_bar
    from ..utils.colors import delta_color

    print(banner("RUN COMPARISON"))
    print(kv([
        ("Baseline", f"{run1.id[:12]}  ({run1.dataset_name})"),
        ("Current", f"{run2.id[:12]}  ({run2.dataset_name})"),
    ]))

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

    print(f"\n{section('METRIC CHANGES')}\n")

    print(f"  {'Metric':<25} {'Baseline':>12} {'Current':>12} {'Delta':>12}")
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

        delta_str = ""
        if rate1 is not None and rate2 is not None:
            diff = rate2 - rate1
            if diff > 0:
                improvements += 1
            elif diff < 0:
                regressions += 1
            # Use delta_color for the formatted delta value
            delta_str = delta_color(diff / 100)
        else:
            delta_str = "="

        print(f"  {metric_id:<25} {r1_str:>12} {r2_str:>12} {delta_str:>12}")

    # Summary
    print(f"\n{section('SUMMARY')}\n")

    total1 = sum(s["passed"] for s in stats1.values())
    total2 = sum(s["passed"] for s in stats2.values())
    all1 = sum(s["total"] for s in stats1.values())
    all2 = sum(s["total"] for s in stats2.values())

    overall1 = 100 * total1 / max(1, all1)
    overall2 = 100 * total2 / max(1, all2)
    overall_delta = overall2 - overall1

    print(kv([
        ("Baseline pass rate", f"{overall1:.1f}% ({total1}/{all1})"),
        ("Current pass rate", f"{overall2:.1f}% ({total2}/{all2})"),
        ("Change", delta_color(overall_delta / 100)),
    ]))

    print(f"\n  {status_icon(improvements > 0)} Metrics improved:  {improvements}")
    print(f"  {status_icon(regressions == 0)} Metrics regressed: {regressions}")
    print(f"  {icon('info')} Metrics unchanged: {len(all_metrics) - improvements - regressions}")

    # Regression severity alerts from insights engine
    if regressions > 0:
        try:
            from ...analysis.core import analyze_run as _analyze_run
            from ...analysis.insights import detect_regressions as _detect_regressions
        except ImportError:
            _analyze_run = None

        if _analyze_run is not None:
            try:
                a1 = _analyze_run(run1.as_dict())
                a2 = _analyze_run(run2.as_dict())
                alerts = _detect_regressions(a2, a1)
                critical = [a for a in alerts if a.severity == "critical"]
                if critical:
                    print(f"\n{section('REGRESSION ALERTS')}\n")
                    for alert in critical:
                        print(f"  {icon('fail')} [CRITICAL] {alert.metric_id}: {abs(alert.delta) * 100:.0f}% drop")
            except (KeyError, ValueError, TypeError):
                pass

    print()

    # Show hints based on results
    output_format = getattr(args, "format", "table")
    hints = HintCollector(quiet=getattr(args, "quiet", False), format=output_format)
    if regressions > improvements:
        hints.add(f"evalyn analyze --run {run2.id}", "Investigate regression in detail")
    if improvements > 0:
        hints.add(f"evalyn trend --project {run2.dataset_name}", "See trends over time")
    if not hints._hints:
        hints.add(f"evalyn analyze --run {run2.id}", "Analyze run in detail")
    hints.render()


def cmd_trend(args: argparse.Namespace) -> None:
    """Show evaluation trends over time for a project."""
    from ...decorators import get_default_tracer
    from ...analysis import analyze_trends, generate_trend_text_report

    storage = get_default_tracer().storage

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

        # Show hints based on trend results
        output_format = getattr(args, "format", "table")
        hints = HintCollector(quiet=getattr(args, "quiet", False), format=output_format)
        if trend.regressing_metrics:
            metrics_str = ", ".join(trend.regressing_metrics[:3])
            hints.add(f"evalyn calibrate --metric-id <metric>", f"Calibrate regressing metrics: {metrics_str}")
        hints.add(f"evalyn insights --project {project_name}", "Deep-dive into results")
        hints.render()


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
