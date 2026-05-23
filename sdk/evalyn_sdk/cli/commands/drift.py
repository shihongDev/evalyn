"""Drift command: report regression vs a pinned baseline.

Usage:
    evalyn drift [--baseline <name>] [--run <id_or_path>] [--latest]
                 [--threshold <pct>] [--format table|json]

Loads the named baseline pointer from .evalyn/baselines.json, resolves the
"current" run from --run or --latest (using the same dataset as the baseline),
and prints a per-metric drift table. Exit code 1 when any regression exceeds
``--threshold``.
"""

from __future__ import annotations

# Dashboard catalog group.
GROUP = "Analysis"
ESSENTIAL = {"baseline", "run", "threshold"}

import argparse
import json
import sys
from pathlib import Path

from ...analysis.baseline import (
    DEFAULT_BASELINE_NAME,
    DriftReport,
    compute_drift,
    load_baseline,
)
from ...analysis.core import RunAnalysis, analyze_run, find_eval_runs, load_eval_run
from ...models import EvalRun
from ..utils.command_common import load_eval_run_for_command
from ..utils.config import load_config, resolve_dataset_path
from ..utils.errors import fatal_error


def _load_run_from_path(path: Path) -> EvalRun:
    """Load an EvalRun from a results.json file or a folder containing one."""
    data = load_eval_run(path)
    return EvalRun.from_dict(data)


def _resolve_baseline_run(pointer) -> EvalRun:
    """Resolve the baseline pointer back to a concrete EvalRun.

    Prefers the recorded ``run_path`` (fast); falls back to storage lookup.
    """
    if pointer.run_path:
        candidate = Path(pointer.run_path)
        if candidate.exists():
            return _load_run_from_path(candidate)

    loaded = load_eval_run_for_command(
        run_id=pointer.run_id,
        error_message=f"Baseline run '{pointer.run_id}' could not be loaded",
        error_hint="The pinned run file may have been moved or deleted",
    )
    return loaded.run


def _resolve_current_run(args: argparse.Namespace, baseline_dataset: str) -> EvalRun:
    """Resolve the run being compared against the baseline.

    Precedence: explicit --run, then --latest against --dataset (or the
    baseline's dataset by default).
    """
    if args.run:
        candidate = Path(args.run)
        if candidate.exists():
            return _load_run_from_path(candidate)
        loaded = load_eval_run_for_command(
            run_id=args.run,
            error_message=f"Could not load run '{args.run}'",
            error_hint="Pass a valid run ID or path to results.json",
        )
        return loaded.run

    if args.latest:
        config = load_config()
        dataset_path = resolve_dataset_path(args.dataset, True, config)
        if not dataset_path:
            # Fall back to baseline's dataset name under data/.
            guess = Path("data") / baseline_dataset
            if guess.exists():
                dataset_path = guess
        if not dataset_path:
            fatal_error(
                "Could not locate a dataset to read --latest from",
                "Pass --dataset <path> or set --run explicitly",
            )
        runs = find_eval_runs(dataset_path)
        if not runs:
            fatal_error(f"No eval runs found under {dataset_path}")
        return _load_run_from_path(runs[0])

    fatal_error(
        "Must specify either --run <id_or_path> or --latest",
        "evalyn drift --latest  OR  evalyn drift --run <id>",
    )


def _fmt_score(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:+.1f}%"


def _print_table(report: DriftReport, baseline_name: str) -> None:
    """Pretty-print a drift report."""
    from ..utils.rich import banner, kv, section, table

    print(banner("DRIFT REPORT"))
    print(
        kv(
            [
                ("Baseline", f"{baseline_name}  ({report.baseline_run_id[:12]})"),
                ("Current", report.current_run_id[:12]),
                ("Threshold", f"{report.threshold * 100:.1f}%"),
            ]
        )
    )

    print(f"\n{section('METRIC DRIFT')}\n")
    rows = [
        [
            entry.metric_id,
            _fmt_score(entry.baseline_mean),
            _fmt_score(entry.current_mean),
            _fmt_score(entry.delta_abs),
            _fmt_pct(entry.delta_pct),
            entry.status,
            "!" if entry.flagged else "",
        ]
        for entry in report.entries
    ]
    print(
        table(
            headers=["METRIC", "BASELINE", "CURRENT", "DELTA", "DELTA%", "STATUS", "FLAG"],
            rows=rows,
            align=["left", "right", "right", "right", "right", "left", "center"],
        )
    )

    flagged = report.flagged_regressions
    print(f"\n{section('SUMMARY')}\n")
    print(
        kv(
            [
                ("Improved", str(len(report.improvements))),
                ("Regressed", str(len(report.regressions))),
                ("Flagged", str(len(flagged))),
            ]
        )
    )
    if flagged:
        worst = min(flagged, key=lambda e: (e.delta_pct if e.delta_pct is not None else 0.0))
        detail = (
            f"{worst.delta_pct * 100:+.1f}%"
            if worst.delta_pct is not None
            else f"delta={worst.delta_abs:+.3f}"
        )
        print(f"\n  Worst regression: {worst.metric_id} ({detail})")


def cmd_drift(args: argparse.Namespace) -> None:
    """Compare the current run against a pinned baseline."""
    baseline_name = args.baseline or DEFAULT_BASELINE_NAME
    pointer = load_baseline(baseline_name, state_dir=args.state_dir)
    if pointer is None:
        fatal_error(
            f"No baseline named '{baseline_name}'",
            "Pin one first: evalyn baseline set --run <id_or_path>",
        )

    baseline_run = _resolve_baseline_run(pointer)
    current_run = _resolve_current_run(args, baseline_run.dataset_name)

    baseline_analysis: RunAnalysis = analyze_run(baseline_run.as_dict())
    current_analysis: RunAnalysis = analyze_run(current_run.as_dict())

    report = compute_drift(baseline_analysis, current_analysis, threshold=args.threshold)

    if getattr(args, "format", "table") == "json":
        payload = {
            "baseline_name": baseline_name,
            "baseline_dataset": pointer.dataset_name,
            **report.as_dict(),
            "summary": {
                "improvements": len(report.improvements),
                "regressions": len(report.regressions),
                "flagged": len(report.flagged_regressions),
            },
        }
        print(json.dumps(payload, indent=2))
    else:
        _print_table(report, baseline_name)

    # Non-zero exit when any regression breaches the threshold so CI can gate.
    if report.flagged_regressions:
        sys.exit(1)


def register_commands(subparsers) -> None:
    """Register the drift command."""
    p = subparsers.add_parser(
        "drift",
        help="Report metric drift between the current run and a pinned baseline",
    )
    p.add_argument(
        "--baseline",
        default=DEFAULT_BASELINE_NAME,
        help=f"Baseline label to compare against (default: {DEFAULT_BASELINE_NAME})",
    )
    p.add_argument("--run", help="Run ID or path to results.json (current run)")
    p.add_argument(
        "--latest",
        action="store_true",
        help="Use the latest run from --dataset (or the baseline's dataset)",
    )
    p.add_argument("--dataset", help="Dataset path (used with --latest)")
    p.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Relative change threshold for flagging (e.g. 0.05 = 5%%). Default 0.",
    )
    p.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    p.add_argument(
        "--state-dir",
        help="Directory holding baselines.json (default: ./.evalyn/)",
    )
    p.set_defaults(func=cmd_drift)


__all__ = ["cmd_drift", "register_commands"]
