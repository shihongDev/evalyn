"""``/api/v2/reports`` router.

Endpoint:
- ``GET /weekly`` -> ``WeeklyReport``: aggregates runs from the last 7
  days vs the prior 7 days. Source-of-truth shapes live in
  ``dashboard/frontend/src/v2/api/types.ts``.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ._shared import (
    datasets_root,
    fmt_cost,
    load_all_runs,
    parse_iso,
    project_meta,
    run_id,
    run_pass_rate,
    total_cost,
)

router = APIRouter()


def _split_windows(runs: list[dict], anchor: datetime) -> tuple[list[dict], list[dict]]:
    """Return ``(this_week, prior_week)`` runs partitioned by ``anchor``."""
    week_start = anchor - timedelta(days=7)
    prior_start = anchor - timedelta(days=14)
    this_week: list[dict] = []
    prior_week: list[dict] = []
    for r in runs:
        dt = parse_iso(r.get("created_at"))
        if dt is None:
            continue
        if week_start <= dt <= anchor:
            this_week.append(r)
        elif prior_start <= dt < week_start:
            prior_week.append(r)
    return this_week, prior_week


def _avg_pass_rate(runs: list[dict]) -> float | None:
    rates = [r for r in (run_pass_rate(x) for x in runs) if r is not None]
    return sum(rates) / len(rates) if rates else None


def _failed_runs(runs: list[dict]) -> list[dict]:
    return [r for r in runs if (r.get("summary") or {}).get("status") == "failed"]


def _sum_cost(runs: list[dict]) -> float:
    return sum(c for c in (total_cost(r) for r in runs) if c is not None)


def _delta_str(curr: float | None, prev: float | None) -> str:
    if curr is None or prev is None:
        return "0.0"
    diff = round(curr - prev, 1)
    return f"{'+' if diff >= 0 else ''}{diff}"


def _delta_kind(curr: float | None, prev: float | None, *, higher_is_better: bool = True) -> str:
    if curr is None or prev is None or curr == prev:
        return "info"
    better = (curr > prev) if higher_is_better else (curr < prev)
    return "pass" if better else "fail"


def _shipped_lines(runs: list[dict]) -> list[dict]:
    """One ``{text: ...}`` per completed run for the WeeklyReport.shipped field."""
    out: list[dict] = []
    for r in runs:
        if (r.get("summary") or {}).get("status") == "failed":
            continue
        pr = run_pass_rate(r)
        pct = f"{round(pr * 100, 1)}%" if pr is not None else "n/a"
        out.append({"text": f"{run_id(r)} - {pct}"})
    return out


def _blocking(failed: list[dict]) -> dict | None:
    """Return the failed run with the most failed items, or ``None``."""
    if not failed:
        return None

    def n_failed(r: dict) -> int:
        return len((r.get("summary") or {}).get("failed_items") or [])

    worst = max(failed, key=n_failed)
    return {
        "title": f"Run {run_id(worst)} failed",
        "body_md": (
            f"Run `{run_id(worst)}` reported failure status with "
            f"{n_failed(worst)} failed items."
        ),
        "owner": "You",
        "eta": "tbd",
    }


@router.get("/weekly")
async def get_weekly() -> JSONResponse:
    """Return the weekly report aggregated from runs on disk."""
    runs = load_all_runs(datasets_root())
    anchor = (
        parse_iso(runs[-1].get("created_at")) if runs else None
    ) or datetime.now(timezone.utc)

    this_week, prior_week = _split_windows(runs, anchor)
    this_quality = _avg_pass_rate(this_week)
    prior_quality = _avg_pass_rate(prior_week)
    quality_pct = round(this_quality * 100, 1) if this_quality is not None else None
    quality_prev_pct = round(prior_quality * 100, 1) if prior_quality is not None else None

    this_failed = _failed_runs(this_week)
    prior_failed = _failed_runs(prior_week)
    this_cost = _sum_cost(this_week)
    prior_cost = _sum_cost(prior_week)

    big_numbers = [
        {
            "label": "Quality",
            "value": f"{quality_pct}%" if quality_pct is not None else "n/a",
            "delta": _delta_str(quality_pct, quality_prev_pct),
            "delta_kind": _delta_kind(quality_pct, quality_prev_pct, higher_is_better=True),
            "sub": f"{len(this_week)} runs this week",
        },
        {
            "label": "Open issues",
            "value": str(len(this_failed)),
            "delta": _delta_str(float(len(this_failed)), float(len(prior_failed))),
            "delta_kind": _delta_kind(
                float(len(this_failed)), float(len(prior_failed)), higher_is_better=False
            ),
            "sub": "failed runs",
        },
        {
            "label": "Eval cost",
            "value": fmt_cost(this_cost if this_cost > 0 else None),
            "delta": _delta_str(this_cost, prior_cost),
            "delta_kind": _delta_kind(this_cost, prior_cost, higher_is_better=False),
            "sub": "USD this week",
        },
    ]

    tldr = (
        f"Quality is {quality_pct}% over {len(this_week)} runs this week."
        if quality_pct is not None
        else f"No runs in the last 7 days (anchor {anchor.date().isoformat()})."
    )

    return JSONResponse(
        {
            "week_label": f"Week of {(anchor - timedelta(days=7)).date().isoformat()}",
            "project_name": project_meta()[0],
            "generated_at_iso": datetime.now(timezone.utc).isoformat(),
            "tldr_md": tldr,
            "big_numbers": big_numbers,
            "shipped": _shipped_lines(this_week),
            "blocking": _blocking(this_failed),
            "up_next": [],
        }
    )
