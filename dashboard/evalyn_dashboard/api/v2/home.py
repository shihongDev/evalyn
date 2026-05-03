"""``/api/v2/home`` router.

Aggregates the landing-page snapshot (HomeSnapshot in
``dashboard/frontend/src/v2/api/types.ts``) from the on-disk run layout
under ``.evalyn/data/datasets/``. Returns the empty/null version of the
shape with HTTP 200 when no data is on disk.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ._shared import (
    cumulative_pass_series,
    dataset_roots,
    is_inverse_metric,
    load_all_runs,
    parse_iso,
    project_meta,
    run_id,
    run_pass_rate,
    run_status,
)

router = APIRouter()

SHIP_GATE = 90


def _short_label(run: dict) -> str:
    """Short timeline x-label (``MM-DD HH:MM`` or run id fallback)."""
    dt = parse_iso(run.get("created_at"))
    if dt is None:
        return run_id(run)[:8]
    return dt.strftime("%m-%d %H:%M")


def _sub_metrics(latest: dict, previous: dict | None) -> list[dict]:
    """Per-metric pass rates from ``latest`` with delta vs ``previous``."""
    latest_metrics = (latest.get("summary") or {}).get("metrics") or {}
    prev_metrics = ((previous or {}).get("summary") or {}).get("metrics") or {}
    out: list[dict] = []
    for mid, m in latest_metrics.items():
        pr = m.get("pass_rate")
        if pr is None:
            pr = m.get("avg_score") or 0.0
        value = round(100.0 * float(pr), 1)
        prev_pr = (prev_metrics.get(mid, {}) or {}).get("pass_rate")
        if prev_pr is None:
            prev_pr = (prev_metrics.get(mid, {}) or {}).get("avg_score")
        delta = round(value - 100.0 * float(prev_pr), 1) if prev_pr is not None else 0.0
        out.append(
            {
                "label": mid,
                "value": value,
                "delta": delta,
                "inverse": is_inverse_metric(mid),
            }
        )
    return out


def _active_experiments(runs_newest_first: list[dict]) -> list[dict]:
    """Top 3 most recent runs as ``HomeSnapshot.active_experiments``."""
    out: list[dict] = []
    for r in runs_newest_first[:3]:
        _, spark = cumulative_pass_series(r, n_points=24)
        pr = run_pass_rate(r)
        status = run_status(r)
        progress = None
        if status == "running":
            n = len({
                mr.get("item_id")
                for mr in r.get("metric_results", [])
                if mr.get("item_id") is not None
            })
            total = (r.get("summary") or {}).get("total_items") or n
            progress = {"done": n, "total": total}
        out.append(
            {
                "id": run_id(r),
                "name": run_id(r),
                "status": status,
                "pass": round(pr * 100, 1) if pr is not None else None,
                "delta_pts": None,
                "progress": progress,
                "spark": spark,
            }
        )
    return out


def _recent_activity(runs_newest_first: list[dict]) -> list[dict]:
    """Up to four 'You ran <run>' events synthesized from runs."""
    return [
        {
            "who": "You",
            "what": "ran",
            "target": run_id(r),
            "when_iso": r.get("created_at") or "",
            "icon": "◆",  # solid diamond glyph
            "accent": False,
        }
        for r in runs_newest_first[:4]
    ]


def _brief(latest: dict, runs_30d: list[dict]) -> dict:
    """One-paragraph deterministic summary of the 30d window."""
    pr = run_pass_rate(latest)
    pr_pct = f"{pr * 100:.1f}%" if pr is not None else "n/a"
    n = len(runs_30d)
    body = (
        f"Quality is {pr_pct} over {n} run{'s' if n != 1 else ''} in the last 30 days. "
        f"Most recent run: {run_id(latest)} at {pr_pct}."
    )
    return {
        "generated_at_iso": datetime.now(timezone.utc).isoformat(),
        "body_md": body,
        "actions": [],
    }


@router.get("")
async def home() -> JSONResponse:
    """Return the aggregated landing snapshot."""
    project_name, project_version = project_meta()
    snap: dict = {
        "project": {"name": project_name, "version": project_version},
        "quality": {
            "current": None,
            "delta_30d": None,
            "weighted_across_metrics": 0,
            "graded_items": 0,
            "timeline": [],
            "ship_gate": SHIP_GATE,
        },
        "sub_metrics": [],
        "active_experiments": [],
        "recent_activity": [],
        "attention": [],
        "brief": None,
    }

    if not dataset_roots():
        return JSONResponse(snap)
    runs = load_all_runs()  # oldest first, walks every root
    if not runs:
        return JSONResponse(snap)

    runs_newest_first = list(reversed(runs))
    latest = runs_newest_first[0]

    # Anchor the 30-day window on the latest run so output is deterministic
    # relative to the data on disk.
    anchor = parse_iso(latest.get("created_at")) or datetime.now(timezone.utc)
    window_start = anchor - timedelta(days=30)
    runs_30d = [r for r in runs if (parse_iso(r.get("created_at")) or anchor) >= window_start]

    timeline: list[dict] = []
    cumulative_passes = 0.0
    for idx, r in enumerate(runs_30d, start=1):
        cumulative_passes += run_pass_rate(r) or 0.0
        timeline.append(
            {"x": _short_label(r), "y": round(100.0 * cumulative_passes / idx, 2)}
        )

    current = run_pass_rate(latest)
    snap["quality"]["timeline"] = timeline
    snap["quality"]["current"] = round(current * 100, 1) if current is not None else None
    if runs_30d:
        oldest_pr = run_pass_rate(runs_30d[0])
        if current is not None and oldest_pr is not None:
            snap["quality"]["delta_30d"] = round(100.0 * (current - oldest_pr), 1)
    snap["quality"]["weighted_across_metrics"] = len(
        (latest.get("summary") or {}).get("metrics") or {}
    )
    snap["quality"]["graded_items"] = len(
        {
            mr.get("item_id")
            for mr in latest.get("metric_results", [])
            if mr.get("item_id") is not None
        }
    )

    previous = runs_newest_first[1] if len(runs_newest_first) > 1 else None
    snap["sub_metrics"] = _sub_metrics(latest, previous)
    snap["active_experiments"] = _active_experiments(runs_newest_first)
    snap["recent_activity"] = _recent_activity(runs_newest_first)
    snap["brief"] = _brief(latest, runs_30d)
    return JSONResponse(snap)
