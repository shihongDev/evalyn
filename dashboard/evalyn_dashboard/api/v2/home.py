"""``/api/v2/home`` router.

Aggregates the landing-page snapshot (HomeSnapshot in
``dashboard/frontend/src/v2/api/types.ts``) from the on-disk run layout
under ``.evalyn/data/datasets/``. Returns the empty/null version of the
shape with HTTP 200 when no data is on disk.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ._shared import (
    calibration_suggestions,
    cost_or_zero,
    cumulative_pass_series,
    daily_cost_buckets,
    dataset_roots,
    fmt_cost,
    get_cached_snapshot,
    has_any_calibration,
    is_inverse_metric,
    load_all_runs,
    median,
    parse_iso,
    project_meta,
    run_id,
    run_pass_rate,
    run_status,
    set_cached_snapshot,
    total_cost,
)

router = APIRouter()
logger = logging.getLogger(__name__)

SHIP_GATE = 90
REGRESSION_PTS = 5.0  # severity threshold for "quality regression" attention.
STRUGGLING_METRIC_PASS = 0.60  # metric < 60% pass across recent N runs.
STRUGGLING_METRIC_WINDOW = 3  # runs to look at for struggling-metric signal.
COST_OUTLIER_MULT = 2.0  # last-run cost > 2x median over last 7 → warn.
COST_WINDOW = 7  # number of recent runs in the cost-median baseline.
STALE_DAYS = 7  # no-runs-in-X-days threshold.
ATTENTION_CAP = 5  # cap final attention list.
CALIBRATION_READY_CAP = 3  # cap calibration-ready attention items.
BRIEF_SENTENCE_CAP = 5  # max sentences in the composed brief body.
SEVERITY_RANK = {"fail": 0, "warn": 1, "info": 2}


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


def _latest_per_dataset(runs_newest_first: list[dict]) -> dict[str, dict]:
    """Return ``{dataset_name: most_recent_run_in_that_dataset}``.

    Inspects: each run's ``_dataset`` and ``created_at``.
    """
    out: dict[str, dict] = {}
    for r in runs_newest_first:
        ds = r.get("_dataset")
        if ds and ds not in out:
            out[ds] = r
    return out


def _runs_by_dataset(runs_newest_first: list[dict]) -> dict[str, list[dict]]:
    """Return ``{dataset_name: [runs newest first]}``.

    Inspects: each run's ``_dataset``.
    """
    out: dict[str, list[dict]] = {}
    for r in runs_newest_first:
        ds = r.get("_dataset")
        if not ds:
            continue
        out.setdefault(ds, []).append(r)
    return out


def _struggling_metric(runs_newest_first: list[dict]) -> tuple[str, float, int] | None:
    """Find a metric with avg pass_rate < 60% across recent runs.

    Inspects: ``summary.metrics[<id>].pass_rate`` (falls back to
    ``avg_score``) on the last :data:`STRUGGLING_METRIC_WINDOW` runs
    across all datasets.

    Returns ``(metric_id, avg_pass_rate, n_runs_seen)`` for the worst
    qualifying metric, or ``None`` when nothing crosses the threshold.
    """
    window = runs_newest_first[:STRUGGLING_METRIC_WINDOW]
    if not window:
        return None
    sums: dict[str, list[float]] = {}
    for r in window:
        metrics = (r.get("summary") or {}).get("metrics") or {}
        for mid, m in metrics.items():
            if not isinstance(m, dict):
                continue
            pr = m.get("pass_rate")
            if pr is None:
                pr = m.get("avg_score")
            if isinstance(pr, (int, float)):
                sums.setdefault(mid, []).append(float(pr))
    worst: tuple[str, float, int] | None = None
    for mid, rates in sums.items():
        if len(rates) < 1:
            continue
        avg = sum(rates) / len(rates)
        if avg < STRUGGLING_METRIC_PASS:
            if worst is None or avg < worst[1]:
                worst = (mid, avg, len(rates))
    return worst


def _quality_regression(
    latest: dict, runs_by_dataset: dict[str, list[dict]]
) -> tuple[float, float, float] | None:
    """Detect quality regression on the latest run vs previous in dataset.

    Inspects: ``run_pass_rate(latest)`` and the previous run within the
    same ``_dataset``. Returns ``(prev_pct, curr_pct, delta_pts)`` when
    the drop is >= :data:`REGRESSION_PTS`. ``delta_pts`` is positive
    (the magnitude of the drop).
    """
    ds = latest.get("_dataset")
    if not ds:
        return None
    same_ds = runs_by_dataset.get(ds, [])
    if len(same_ds) < 2:
        return None
    prev = same_ds[1]
    curr_pr = run_pass_rate(latest)
    prev_pr = run_pass_rate(prev)
    if curr_pr is None or prev_pr is None:
        return None
    curr_pct = curr_pr * 100.0
    prev_pct = prev_pr * 100.0
    delta = prev_pct - curr_pct
    if delta >= REGRESSION_PTS:
        return prev_pct, curr_pct, delta
    return None


def _cost_outlier(runs_newest_first: list[dict]) -> tuple[float, float] | None:
    """Detect last run > 2x median cost of the prior :data:`COST_WINDOW` runs.

    Inspects: ``usage_summary.total_cost``/``total_cost_usd`` via
    :func:`total_cost`. Returns ``(curr_cost, median_prior)`` when the
    outlier fires; ``None`` otherwise (including when the median is 0
    or fewer than 2 prior runs exist).
    """
    if not runs_newest_first:
        return None
    curr = total_cost(runs_newest_first[0])
    if curr is None or curr <= 0:
        return None
    prior = [
        c for c in (total_cost(r) for r in runs_newest_first[1 : 1 + COST_WINDOW])
        if c is not None and c > 0
    ]
    if len(prior) < 2:
        return None
    med = median(prior)
    if med is None or med <= 0:
        return None
    if curr > COST_OUTLIER_MULT * med:
        return curr, med
    return None


def _attention_calibration_ready(suggestions: list[dict]) -> list[dict]:
    """Emit one ATTENTION item per ready calibration.

    Severity ``info`` because a ready calibration is progress, not a
    problem. The CTA navigates to ``/review`` where the user can click
    "Run calibrate" (Lane I of iter 6 wired the actual deep-link).
    Capped at :data:`CALIBRATION_READY_CAP` so the attention queue isn't
    dominated by these.
    """
    out: list[dict] = []
    for s in suggestions[:CALIBRATION_READY_CAP]:
        out.append({
            "severity": "info",
            "title": f"Calibration ready for `{s['metric_id']}`",
            "subtitle": (
                f"{s['verdict_count']} verdicts collected "
                f"(threshold: {s['threshold']})"
            ),
            "cta": "Open Review",
            "cta_target": "/review",
        })
    return out


def _attention(
    runs_newest_first: list[dict],
    anchor: datetime,
    calibration_ready: list[dict] | None = None,
) -> list[dict]:
    """Build the ``HomeSnapshot.attention`` list.

    Six signals (each documented at its detector helper above):
      1. quality regression (fail) - drop >= 5pts vs previous run in dataset
      2. struggling metric (warn) - <60% pass across recent 3 runs
      3. no calibration (info) - any runs but no calibrations dirs anywhere
      4. cost outlier (warn) - last run > 2x median of last 7
      5. stale workspace (info) - last run > 7 days ago
      6. calibration-ready (info) - per-metric calibration suggestion fired

    Suppresses signal #3 when calibration-ready items are present (the
    brief would otherwise read contradictorily: "no calibration" plus
    "calibration ready").

    The list is sorted ``fail > warn > info`` then by recency (latest
    first as a stable tiebreak so freshly-fired signals win) and capped
    at :data:`ATTENTION_CAP`.
    """
    out: list[dict] = []
    if not runs_newest_first:
        return out
    latest = runs_newest_first[0]
    runs_by_ds = _runs_by_dataset(runs_newest_first)
    cal_ready = calibration_ready or []

    # 1. Quality regression
    regression = _quality_regression(latest, runs_by_ds)
    if regression is not None:
        prev_pct, curr_pct, delta = regression
        out.append({
            "severity": "fail",
            "title": f"Quality regression in {latest.get('_dataset', 'unknown')}",
            "subtitle": (
                f"{prev_pct:.1f}% → {curr_pct:.1f}% (-{delta:.1f} pts) in last run"
            ),
            "cta": "Investigate",
            "cta_target": f"/experiments/{run_id(latest)}",
        })

    # 2. Struggling metric
    struggling = _struggling_metric(runs_newest_first)
    if struggling is not None:
        mid, avg, n_seen = struggling
        out.append({
            "severity": "warn",
            "title": f"Metric `{mid}` failing >{int(round((1 - avg) * 100))}% of items",
            "subtitle": f"Across {n_seen} recent run{'s' if n_seen != 1 else ''}",
            "cta": "Open metric",
            "cta_target": "/metrics",
        })

    # 4. Cost outlier (number 4 in the spec but cheap to compute here)
    cost_outlier = _cost_outlier(runs_newest_first)
    if cost_outlier is not None:
        curr_cost, med_cost = cost_outlier
        out.append({
            "severity": "warn",
            "title": f"Last run cost {fmt_cost(curr_cost)} (2x median)",
            "subtitle": f"Median {fmt_cost(med_cost)}",
            "cta": "View run",
            "cta_target": f"/experiments/{run_id(latest)}",
        })

    # 5. Stale: last run older than STALE_DAYS
    last_dt = parse_iso(latest.get("created_at"))
    if last_dt is not None:
        age = anchor - last_dt
        if age > timedelta(days=STALE_DAYS):
            out.append({
                "severity": "info",
                "title": f"No new evaluations in {age.days} days",
                "subtitle": f"Last: {last_dt.strftime('%Y-%m-%d')}",
                "cta": "Run an eval",
                "cta_target": "/commands",
            })

    # 3. No calibration anywhere (only meaningful when runs exist - which
    #    is the path we're on). Suppressed when calibration-ready items
    #    exist - they're contradictory (calibrations ARE coming).
    if not has_any_calibration() and not cal_ready:
        out.append({
            "severity": "info",
            "title": "Judge has no human calibration",
            "subtitle": "Verdicts may drift from human judgment",
            "cta": "Calibrate",
            "cta_target": "/metrics",
        })

    # 6. Calibration-ready (per-metric).
    out.extend(_attention_calibration_ready(cal_ready))

    out.sort(key=lambda a: (
        SEVERITY_RANK.get(a["severity"], 99),
        # Stable: items appended earlier (= more recently fired) keep order.
    ))
    return out[:ATTENTION_CAP]


def _weekly_pass_rate(
    runs_newest_first: list[dict], anchor: datetime
) -> tuple[float | None, float | None]:
    """Return ``(this_week_avg, last_week_avg)`` pass-rate percentages.

    Inspects: ``run_pass_rate`` and ``created_at`` for runs falling
    inside the two 7-day windows immediately preceding ``anchor``.
    Each value is either a 0..100 percentage or ``None`` when the
    window is empty.
    """
    this_start = anchor - timedelta(days=7)
    last_start = anchor - timedelta(days=14)
    this_rates: list[float] = []
    last_rates: list[float] = []
    for r in runs_newest_first:
        dt = parse_iso(r.get("created_at"))
        if dt is None:
            continue
        pr = run_pass_rate(r)
        if pr is None:
            continue
        if dt >= this_start:
            this_rates.append(pr * 100.0)
        elif dt >= last_start:
            last_rates.append(pr * 100.0)
    this_avg = sum(this_rates) / len(this_rates) if this_rates else None
    last_avg = sum(last_rates) / len(last_rates) if last_rates else None
    return this_avg, last_avg


def _largest_dataset_delta(
    runs_by_dataset: dict[str, list[dict]], anchor: datetime
) -> tuple[str, float] | None:
    """Find the dataset whose week-over-week pass-rate moved the most.

    Inspects: per-dataset run lists, comparing avg pass over the last 7
    days vs the prior 7 days. Returns ``(dataset_name, delta_pts)``
    with delta SIGNED (negative = drop, positive = gain) for the
    dataset with the largest absolute delta. ``None`` when nothing
    qualifies (e.g. no dataset has runs in both windows).
    """
    this_start = anchor - timedelta(days=7)
    last_start = anchor - timedelta(days=14)
    best: tuple[str, float] | None = None
    for ds, runs in runs_by_dataset.items():
        this_rates: list[float] = []
        last_rates: list[float] = []
        for r in runs:
            dt = parse_iso(r.get("created_at"))
            pr = run_pass_rate(r)
            if dt is None or pr is None:
                continue
            if dt >= this_start:
                this_rates.append(pr * 100.0)
            elif dt >= last_start:
                last_rates.append(pr * 100.0)
        if not this_rates or not last_rates:
            continue
        delta = (sum(this_rates) / len(this_rates)) - (sum(last_rates) / len(last_rates))
        if best is None or abs(delta) > abs(best[1]):
            best = (ds, delta)
    return best


def _cost_summary(runs_30d: list[dict], anchor: datetime) -> dict | None:
    """Build the ``HomeSnapshot.cost`` payload from the 30d run window.

    Returns ``None`` when ``runs_30d`` is empty (no signal to surface).
    Otherwise:
      * ``total_30d`` / ``total_7d`` - sum of ``cost_or_zero`` per run.
      * ``runs_with_cost`` - count of runs in the 30d window where
        ``total_cost(run) > 0`` (None / 0 don't count).
      * ``runs_total`` - len(runs_30d).
      * ``daily_30d`` - 30-element list, one per calendar day (UTC),
        oldest day first; built by :func:`daily_cost_buckets`.
      * ``projected_monthly`` - simple linear projection from the 7d
        window: ``(total_7d / actual_days_with_runs) * 30``. Set to
        ``None`` when fewer than 3 distinct days had runs in 7d so the
        UI doesn't extrapolate from a tiny sample.
    """
    if not runs_30d:
        return None
    week_start = anchor - timedelta(days=7)
    total_30d = 0.0
    total_7d = 0.0
    runs_with_cost = 0
    days_with_runs_7d: set = set()
    for run in runs_30d:
        c = cost_or_zero(run)
        total_30d += c
        original = total_cost(run)
        if original is not None and original > 0:
            runs_with_cost += 1
        dt = parse_iso(run.get("created_at"))
        if dt is not None and dt >= week_start:
            total_7d += c
            days_with_runs_7d.add(dt.date())
    daily_30d = daily_cost_buckets(runs_30d, anchor, days=30)
    projected: float | None = None
    if len(days_with_runs_7d) >= 3:
        projected = round((total_7d / len(days_with_runs_7d)) * 30.0, 2)
    return {
        "total_30d": round(total_30d, 4),
        "total_7d": round(total_7d, 4),
        "runs_with_cost": runs_with_cost,
        "runs_total": len(runs_30d),
        "daily_30d": [round(v, 4) for v in daily_30d],
        "projected_monthly": projected,
    }


def _brief(
    latest: dict,
    runs_30d: list[dict],
    runs_newest_first: list[dict],
    attention: list[dict],
    anchor: datetime,
    cost: dict | None = None,
    calibration_ready_count: int = 0,
) -> dict:
    """Compose a 1-3 sentence brief grounded in disk data.

    Data sources (one per sentence so each line is falsifiable):
      * S1 (overall):         ``run_pass_rate`` over ``runs_30d`` + count.
      * S2 (week-over-week):  :func:`_weekly_pass_rate` and
                              :func:`_largest_dataset_delta`.
      * S3 (worst metric):    :func:`_struggling_metric` over the most
                              recent :data:`STRUGGLING_METRIC_WINDOW` runs.

    Special cases:
      * No runs in 30d window     → empty-state CTA prompt.
      * Exactly one run in window → "first run captured" message.

    When ``calibration_ready_count > 0`` an extra sentence is appended
    pointing the user at /review (subject to :data:`BRIEF_SENTENCE_CAP`).

    Returns the full ``brief`` dict including ``actions`` derived from
    ``attention`` (regression → primary "Open run"; struggling metric
    → secondary "Open metric"; always "Open co-pilot" bare).
    """
    now_iso = datetime.now(timezone.utc).isoformat()

    # Empty 30d window.
    if not runs_30d:
        body = "No runs in the last 30 days. Run `evalyn run-eval` to populate this brief."
        return {
            "generated_at_iso": now_iso,
            "body_md": body,
            "actions": [{"label": "Open co-pilot", "kind": "bare", "intent": "/copilot"}],
        }

    # Single-run path.
    if len(runs_30d) == 1:
        only = runs_30d[0]
        pr = run_pass_rate(only)
        pct = f"{pr * 100:.1f}%" if pr is not None else "n/a"
        body = (
            f"First run captured: {run_id(only)} at {pct}. "
            "Run another to start tracking trends."
        )
        if calibration_ready_count > 0:
            plural = "s" if calibration_ready_count != 1 else ""
            body += (
                f" {calibration_ready_count} rubric calibration{plural} "
                "ready - go to Review."
            )
        return {
            "generated_at_iso": now_iso,
            "body_md": body,
            "actions": _brief_actions(attention),
        }

    # Sentence 1: 30d window summary.
    pr_latest = run_pass_rate(latest)
    n_runs = len(runs_30d)
    overall_rates = [
        run_pass_rate(r) * 100.0 for r in runs_30d if run_pass_rate(r) is not None
    ]
    overall_avg = sum(overall_rates) / len(overall_rates) if overall_rates else None
    pr_pct = f"{overall_avg:.1f}%" if overall_avg is not None else (
        f"{pr_latest * 100:.1f}%" if pr_latest is not None else "n/a"
    )
    sentences = [
        f"Quality is {pr_pct} over {n_runs} run{'s' if n_runs != 1 else ''} in the last 30 days."
    ]

    # Sentence 2: week-over-week + dataset drift.
    this_avg, last_avg = _weekly_pass_rate(runs_newest_first, anchor)
    if this_avg is not None and last_avg is not None:
        wow = this_avg - last_avg
        arrow = "↓" if wow < 0 else ("↑" if wow > 0 else "→")
        ds_drift = _largest_dataset_delta(_runs_by_dataset(runs_newest_first), anchor)
        if ds_drift is not None:
            ds_name, ds_delta = ds_drift
            mover = (
                f", mostly from the {ds_name} dataset" if abs(ds_delta) >= 1.0 else ""
            )
        else:
            mover = ""
        sentences.append(
            f"{arrow} {abs(wow):.1f} pts vs last week{mover}."
        )

    # Sentence 3: worst metric across recent runs.
    struggling = _struggling_metric(runs_newest_first)
    if struggling is not None:
        mid, avg, _ = struggling
        sentences.append(
            f"`{mid}` is the largest failure mode at {avg * 100:.1f}% pass."
        )

    # Sentence 4: cost (only when meaningful and within 4-sentence cap).
    if cost is not None and cost.get("total_30d", 0) > 0 and len(sentences) < 4:
        spend = cost["total_30d"]
        proj = cost.get("projected_monthly")
        if proj is not None:
            sentences.append(
                f"Spend this month: ${spend:.2f} (projected ${proj:.2f})."
            )
        else:
            sentences.append(f"Spend this month: ${spend:.2f}.")

    # Sentence 5: calibration ready (only when we have headroom).
    if calibration_ready_count > 0 and len(sentences) < BRIEF_SENTENCE_CAP:
        plural = "s" if calibration_ready_count != 1 else ""
        sentences.append(
            f"{calibration_ready_count} rubric calibration{plural} ready - go to Review."
        )

    body = " ".join(sentences[:BRIEF_SENTENCE_CAP])
    return {
        "generated_at_iso": now_iso,
        "body_md": body,
        "actions": _brief_actions(attention),
    }


def _brief_actions(attention: list[dict]) -> list[dict]:
    """Derive brief CTAs from attention items.

    Rules:
      * fail-severity item with ``cta_target`` starting ``/experiments/`` →
        primary "Open run" pointing at that run.
      * struggling-metric warn (``cta_target == "/metrics"``) → secondary
        "Open metric".
      * Always trailing bare "Open co-pilot" → ``/copilot``.
    """
    actions: list[dict] = []
    for a in attention:
        if a["severity"] == "fail" and a["cta_target"].startswith("/experiments/"):
            actions.append({
                "label": "Open run",
                "kind": "primary",
                "intent": a["cta_target"],
            })
            break
    for a in attention:
        if a["severity"] == "warn" and a["cta_target"] == "/metrics":
            actions.append({
                "label": "Open metric",
                "kind": "secondary",
                "intent": "/metrics",
            })
            break
    actions.append({"label": "Open co-pilot", "kind": "bare", "intent": "/copilot"})
    return actions


@router.get("")
async def home() -> JSONResponse:
    """Return the aggregated landing snapshot.

    Cached on the union of dataset-root mtimes so warm requests skip the
    attention-detector walk (5 detectors + per-run scans add up to ~1s
    cold). Cache invalidates the moment any root mtime bumps.
    """
    cached = get_cached_snapshot("home")
    if cached is not None:
        return JSONResponse(cached)
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
        "cost": None,
    }

    if not dataset_roots():
        set_cached_snapshot("home", snap)
        return JSONResponse(snap)
    runs = load_all_runs()  # oldest first, walks every root
    if not runs:
        set_cached_snapshot("home", snap)
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
    cal_ready = calibration_suggestions(runs)
    attention = _attention(runs_newest_first, anchor, cal_ready)
    snap["attention"] = attention
    cost = _cost_summary(runs_30d, anchor)
    snap["cost"] = cost
    snap["brief"] = _brief(
        latest,
        runs_30d,
        runs_newest_first,
        attention,
        anchor,
        cost,
        calibration_ready_count=len(cal_ready),
    )
    set_cached_snapshot("home", snap)
    return JSONResponse(snap)
