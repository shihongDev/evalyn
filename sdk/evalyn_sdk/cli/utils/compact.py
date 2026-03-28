"""Compact output mode for CI pipelines.

Produces minimal single-line summaries suitable for CI logs and scripting.
Format: "RUN <id> PASS 85% (17/20) COST $0.12 TIME 45s"
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def format_run_compact(
    run_id: str,
    pass_rate: float,
    passed_count: int,
    total_count: int,
    cost_usd: float = 0.0,
    duration_seconds: float = 0.0,
    name: Optional[str] = None,
) -> str:
    """Format a single-line run summary for CI output.

    Args:
        run_id: Run identifier (will be truncated to 8 chars).
        pass_rate: Overall pass rate (0.0 to 1.0).
        passed_count: Number of passing items.
        total_count: Total number of items.
        cost_usd: Total evaluation cost in USD.
        duration_seconds: Total evaluation time in seconds.
        name: Optional run name.

    Returns:
        Single-line summary string.
    """
    short_id = run_id[:8]
    status = "PASS" if pass_rate >= 0.5 else "FAIL"
    pct = f"{pass_rate * 100:.0f}%"

    parts = [f"RUN {short_id}"]
    if name:
        parts.append(f"({name})")
    parts.append(status)
    parts.append(f"{pct} ({passed_count}/{total_count})")

    if cost_usd > 0:
        parts.append(f"COST ${cost_usd:.4f}")

    if duration_seconds > 0:
        if duration_seconds < 60:
            parts.append(f"TIME {duration_seconds:.0f}s")
        else:
            minutes = duration_seconds / 60
            parts.append(f"TIME {minutes:.1f}m")

    return " ".join(parts)


def format_compare_compact(
    metric_id: str,
    baseline_rate: float,
    current_rate: float,
    delta: float,
) -> str:
    """Format a single-line metric comparison for CI output.

    Args:
        metric_id: Metric identifier.
        baseline_rate: Baseline pass rate.
        current_rate: Current pass rate.
        delta: Change in pass rate (current - baseline).

    Returns:
        Single-line comparison string.
    """
    direction = "+" if delta >= 0 else ""
    status = "OK" if delta >= -0.05 else "REGRESSION"
    return (
        f"{metric_id}: {baseline_rate*100:.0f}% -> {current_rate*100:.0f}% "
        f"({direction}{delta*100:.1f}%) [{status}]"
    )


def format_summary_compact(
    run_id: str,
    metrics_summary: Dict[str, Dict[str, Any]],
    name: Optional[str] = None,
) -> str:
    """Format a compact multi-line summary with one line per metric.

    Args:
        run_id: Run identifier.
        metrics_summary: Dict of metric_id -> {pass_rate, count, avg_score}.
        name: Optional run name.

    Returns:
        Multi-line compact summary.
    """
    lines = []
    short_id = run_id[:8]
    header = f"RUN {short_id}"
    if name:
        header += f" ({name})"
    lines.append(header)

    for mid, data in sorted(metrics_summary.items()):
        pr = data.get("pass_rate")
        count = data.get("count", 0)
        if pr is not None:
            status = "PASS" if pr >= 0.5 else "FAIL"
            lines.append(f"  {mid}: {pr*100:.0f}% (n={count}) [{status}]")
        else:
            avg = data.get("avg_score")
            if avg is not None:
                lines.append(f"  {mid}: avg={avg:.2f} (n={count})")

    return "\n".join(lines)
