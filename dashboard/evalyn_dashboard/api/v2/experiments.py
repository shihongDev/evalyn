"""``/api/v2/experiments`` router.

Endpoints:
- ``GET /``                          -> ExperimentList
- ``GET /lineage``                   -> Lineage (run-by-run config diffs)
- ``GET /{id}``                      -> ExperimentDetail
- ``GET /{id}/cluster/{cluster_id}`` -> ClusterDetail
- ``GET /{id}/clusters``             -> ClustersResponse (semantic, via SDK)

Source of truth for the JSON shapes is
``dashboard/frontend/src/v2/api/types.ts``.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from . import experiments_items
from ._shared import (
    cumulative_pass_series,
    dataset_roots,
    fmt_cost,
    fmt_duration,
    input_text,
    is_inverse_metric,
    load_all_runs,
    load_dataset_items,
    parse_iso,
    run_dataset_dir,
    run_id,
    run_pass_rate,
    run_status,
    total_cost,
)

logger = logging.getLogger(__name__)

router = APIRouter()
experiments_items.register(router)


# ---------------------------------------------------------------------------
# failure_mix: classify each item as pass/warn/fail based on metric_results.
#   pass: every metric for the item passed
#   warn: at least one soft (0.3 < score < 0.7) but no hard fail
#   fail: at least one metric failed
# Returns ``None`` when the run has zero items yet.
# ---------------------------------------------------------------------------


def _failure_mix(run: dict) -> dict | None:
    """Compute per-item pass/warn/fail counts for ``run``.

    Returns ``None`` when ``metric_results`` is empty (run has no items
    yet). Soft-warn is keyed off the existing soft-band thresholds used
    by the review queue.
    """
    item_state: dict[str, str] = {}
    for r in run.get("metric_results") or []:
        iid = r.get("item_id")
        if iid is None:
            continue
        passed = r.get("passed")
        score = r.get("score")
        prior = item_state.get(iid, "pass")
        if passed is False:
            item_state[iid] = "fail"
            continue
        if prior == "fail":
            continue
        if isinstance(score, (int, float)) and 0.3 < float(score) < 0.7:
            if prior != "warn":
                item_state[iid] = "warn"

    if not item_state:
        return None
    out = {"pass": 0, "warn": 0, "fail": 0}
    for state in item_state.values():
        out[state] = out.get(state, 0) + 1
    return out


def _items_string(run: dict) -> str:
    """Format ``items`` field per ExperimentRow."""
    item_ids = {mr.get("item_id") for mr in run.get("metric_results") or []}
    n = len(item_ids)
    summary = run.get("summary") or {}
    if summary.get("in_progress") or summary.get("status") == "running":
        total = summary.get("total_items") or n
        return f"{n}/{total}"
    return str(n)


def _spark_for_row(run: dict) -> list[float]:
    """Five evenly-spaced cumulative pass-rate samples for the list row."""
    _, series = cumulative_pass_series(run, n_points=5)
    return series


def _tags(run: dict) -> list[str]:
    """Best-effort tag extraction: dataset name + version tokens (e.g. 'v3')."""
    tags = [run.get("_dataset", "")]
    for source in (run.get("_dataset", ""), run_id(run)):
        for tok in source.split("-"):
            if tok.startswith("v") and tok[1:].isdigit():
                tags.append(tok)
    return [t for t in tags if t]


def _delta_str(curr: float | None, prev: float | None) -> str:
    """Format a pass-rate delta. Returns ``"-"`` for moves <=0.05pts so the
    Δ column doesn't read as "+0.0" noise on the 70%+ of consecutive runs
    that are statistically identical (UX walkthrough finding).
    """
    if prev is None:
        return "baseline"
    if curr is None:
        return "-"
    diff = round(100.0 * (curr - prev), 1)
    if abs(diff) < 0.05:
        return "-"
    sign = "+" if diff > 0 else ""
    return f"{sign}{diff}"


def _friendly_name(run: dict) -> str:
    """Use the dataset folder name + version tag as a human label, falling
    back to the raw run id. Avoids the "wall of identical timestamp ids"
    table problem."""
    ds = run.get("_dataset") or ""
    rid = run_id(run)
    if ds:
        # Strip the trailing '-vNN' or timestamp tail when present so two
        # runs on the same agent share a label.
        return ds
    return rid


def _serialize_row(run: dict, prev: dict | None) -> dict:
    pr = run_pass_rate(run)
    summary = run.get("summary") or {}
    return {
        "id": run_id(run),
        "name": _friendly_name(run),
        "author": "You",
        "when_iso": run.get("created_at") or "",
        "status": run_status(run),
        "pass": round(pr * 100, 1) if pr is not None else None,
        "delta": _delta_str(pr, run_pass_rate(prev) if prev else None),
        "items": _items_string(run),
        "duration": fmt_duration(summary.get("duration_s")),
        "cost": fmt_cost(total_cost(run)),
        "spark": _spark_for_row(run),
        "tags": _tags(run),
        "failure_mix": _failure_mix(run),
        **({"err": summary.get("error", "failed")} if summary.get("status") == "failed" else {}),
    }


@router.get("")
async def list_experiments() -> JSONResponse:
    """Return one row per run, newest first, with delta vs previous run."""
    if not dataset_roots():
        return JSONResponse([])
    runs = load_all_runs()  # oldest first, walks every root
    if not runs:
        return JSONResponse([])

    # Group by dataset to compute per-dataset deltas.
    by_dataset: dict[str, list[dict]] = defaultdict(list)
    for r in runs:
        by_dataset[r["_dataset"]].append(r)

    rows: list[dict] = []
    for drs in by_dataset.values():
        for idx, r in enumerate(drs):
            prev = drs[idx - 1] if idx > 0 else None
            rows.append(_serialize_row(r, prev))

    rows.sort(key=lambda x: x.get("when_iso") or "", reverse=True)
    return JSONResponse(rows)


# ---------- Lineage ----------


def _humanize_when(iso: str) -> str:
    """Return ``"2h"``, ``"3d"``-style string from an iso timestamp.

    Falls back to "" on parse failure. Reference time is "now" so freshly
    written runs read as "0m" / "1m" / etc.
    """
    if not iso:
        return ""
    dt = parse_iso(iso)
    if dt is None:
        return ""
    now = datetime.now(timezone.utc)
    delta = now - dt
    secs = int(delta.total_seconds())
    if secs < 60:
        return f"{max(0, secs)}s"
    mins = secs // 60
    if mins < 60:
        return f"{mins}m"
    hours = mins // 60
    if hours < 48:
        return f"{hours}h"
    days = hours // 24
    return f"{days}d"


def _config_signature(run: dict) -> dict:
    """Pull stored config bits worth diffing run-to-run.

    Looks at ``judge_configs[*].model``, ``metrics[*].config``, and the
    ``dataset_name``. Missing fields are dropped (not emitted as ``None``).
    """
    sig: dict[str, Any] = {}
    judges: list[str] = []
    for jc in run.get("judge_configs") or []:
        if isinstance(jc, dict):
            mdl = jc.get("model")
            if isinstance(mdl, str) and mdl:
                judges.append(mdl)
    if judges:
        sig["judge_model"] = sorted(set(judges))[0]
    # Metric thresholds + judge models seen on metrics.
    metric_models: list[str] = []
    thresholds: list[float] = []
    for m in run.get("metrics") or []:
        if not isinstance(m, dict):
            continue
        cfg = m.get("config") or {}
        if isinstance(cfg, dict):
            mdl = cfg.get("model")
            if isinstance(mdl, str) and mdl:
                metric_models.append(mdl)
            th = cfg.get("success_threshold")
            if isinstance(th, (int, float)):
                thresholds.append(float(th))
    if metric_models and "judge_model" not in sig:
        sig["judge_model"] = sorted(set(metric_models))[0]
    if thresholds:
        # Pick a stable representative; thresholds usually agree across metrics.
        sig["threshold"] = round(sum(thresholds) / len(thresholds), 3)
    ds = run.get("dataset_name") or run.get("_dataset")
    if isinstance(ds, str) and ds:
        sig["dataset_id"] = ds
    return sig


def _config_diff_str(curr: dict, prev: dict | None) -> str:
    """Render a one-line config diff.

    "no config change - baseline" when ``prev`` is None; otherwise lists
    each key whose value differs as ``key X -> Y``. When nothing differs
    returns ``"no config change"``.
    """
    if prev is None:
        return "no config change - baseline"
    curr_sig = _config_signature(curr)
    prev_sig = _config_signature(prev)
    diffs: list[str] = []
    for key in sorted(set(curr_sig) | set(prev_sig)):
        cv = curr_sig.get(key)
        pv = prev_sig.get(key)
        if cv == pv:
            continue
        diffs.append(f"{key} {pv} -> {cv}")
    if not diffs:
        return "no config change"
    return ", ".join(diffs)


def _items_count_or_progress(run: dict) -> Any:
    """Return ``int`` for completed runs, ``"done/total"`` for running."""
    item_ids = {mr.get("item_id") for mr in run.get("metric_results") or []}
    n = len(item_ids)
    summary = run.get("summary") or {}
    if summary.get("in_progress") or summary.get("status") == "running":
        total = summary.get("total_items") or n
        return f"{n}/{total}"
    return n


def _lineage_pulse_spark(runs_oldest_first: list[dict]) -> list[float]:
    """Return up to 12 recent pass-rate samples (oldest first)."""
    out: list[float] = []
    for r in runs_oldest_first[-12:]:
        pr = run_pass_rate(r)
        if pr is None:
            continue
        out.append(round(100.0 * pr, 1))
    return out


@router.get("/lineage")
async def get_lineage() -> JSONResponse:
    """Return runs-as-lineage with config diffs and pulse sparkline."""
    if not dataset_roots():
        return JSONResponse({"runs": [], "median_pass": None, "pulse_spark": []})
    runs = load_all_runs()  # oldest first
    if not runs:
        return JSONResponse({"runs": [], "median_pass": None, "pulse_spark": []})

    by_dataset: dict[str, list[dict]] = defaultdict(list)
    for r in runs:
        by_dataset[r["_dataset"]].append(r)

    out_runs: list[dict] = []
    pass_rates: list[float] = []
    for drs in by_dataset.values():
        for idx, r in enumerate(drs):
            prev = drs[idx - 1] if idx > 0 else None
            pr = run_pass_rate(r)
            if pr is not None:
                pass_rates.append(round(100.0 * pr, 1))
            out_runs.append(
                {
                    "id": run_id(r),
                    "name": _friendly_name(r),
                    "author": "You",
                    "when": _humanize_when(r.get("created_at") or ""),
                    "status": run_status(r),
                    "pass": round(pr * 100, 1) if pr is not None else None,
                    "items": _items_count_or_progress(r),
                    "failure_mix": _failure_mix(r),
                    "config_diff": _config_diff_str(r, prev),
                    "tags": _tags(r),
                    "pinned": False,
                    "live": run_status(r) == "running",
                }
            )

    # Newest-first ordering using created_at from the source runs.
    when_by_id: dict[str, str] = {run_id(r): (r.get("created_at") or "") for r in runs}
    out_runs.sort(key=lambda x: when_by_id.get(x["id"], ""), reverse=True)

    median_pass: float | None = None
    if pass_rates:
        s = sorted(pass_rates)
        n = len(s)
        median_pass = s[n // 2] if n % 2 else round((s[n // 2 - 1] + s[n // 2]) / 2.0, 1)

    pulse = _lineage_pulse_spark(runs)

    return JSONResponse(
        {
            "runs": out_runs,
            "median_pass": median_pass,
            "pulse_spark": pulse,
        }
    )


# ---------- Detail ----------


def _locate_run(run_id_query: str) -> tuple[dict | None, dict | None, list[dict]]:
    """Single-scan lookup. Returns ``(run, baseline, dataset_runs)``.

    ``baseline`` is the run immediately preceding ``run`` in the same
    dataset (chronological), or ``None`` if it's the oldest. ``dataset_runs``
    is every run in the same dataset, oldest first - reused by the cluster
    handler to compute its trend without a second disk scan.
    """
    all_runs = load_all_runs()  # oldest first, walks every root
    target = next(
        (r for r in all_runs if run_id(r) == run_id_query or r.get("id") == run_id_query),
        None,
    )
    if target is None:
        return None, None, []
    dataset_runs = [r for r in all_runs if r["_dataset"] == target["_dataset"]]
    idx = next(i for i, r in enumerate(dataset_runs) if run_id(r) == run_id(target))
    baseline = dataset_runs[idx - 1] if idx > 0 else None
    return target, baseline, dataset_runs


def _failed_metric_ids_per_item(run: dict) -> dict[str, list[str]]:
    """Return ``{item_id: [metric_id, ...]}`` for failed metrics."""
    out: dict[str, list[str]] = defaultdict(list)
    for r in run.get("metric_results") or []:
        if r.get("passed") is False:
            out[r.get("item_id", "")].append(r.get("metric_id", ""))
    return out


def _cluster_run(run: dict) -> dict[str, list[str]]:
    """Group failed item ids by their first failed metric.

    Returns ``{metric_id: [item_id, ...]}``. v2 first-cut bucketing; LLM
    clustering is wired later.
    """
    failures = _failed_metric_ids_per_item(run)
    clusters: dict[str, list[str]] = defaultdict(list)
    for item_id, metric_ids in failures.items():
        if metric_ids:
            clusters[metric_ids[0]].append(item_id)
    return clusters


def _headline(run: dict, baseline: dict | None) -> list[dict]:
    summary = run.get("summary") or {}
    base_summary = (baseline or {}).get("summary") or {}
    cards: list[dict] = []

    pr = run_pass_rate(run)
    base_pr = run_pass_rate(baseline) if baseline else None
    if pr is not None:
        delta = round(100.0 * (pr - base_pr), 1) if base_pr is not None else 0.0
        sign = "+" if delta >= 0 else ""
        cards.append(
            {
                "label": "Pass rate",
                "value": f"{round(pr * 100, 1)}%",
                "delta": f"{sign}{delta}" if base_pr is not None else "baseline",
                "delta_kind": "pass" if delta >= 0 else "fail",
                "sub": "weighted across metrics",
            }
        )

    metrics = summary.get("metrics") or {}
    base_metrics = base_summary.get("metrics") or {}

    # Hallucination card.
    hallucin_id = next((m for m in metrics if "hallucin" in m.lower()), None)
    if hallucin_id:
        avg = metrics[hallucin_id].get("avg_score") or 0.0
        rate = round(100.0 * (1.0 - avg), 1)
        base_avg = (base_metrics.get(hallucin_id, {}) or {}).get("avg_score")
        delta = round(100.0 * ((1.0 - avg) - (1.0 - base_avg)), 1) if base_avg is not None else 0.0
        sign = "+" if delta >= 0 else ""
        cards.append(
            {
                "label": "Hallucination",
                "value": f"{rate}%",
                "delta": f"{sign}{delta}" if base_avg is not None else "baseline",
                "delta_kind": "pass" if delta <= 0 else "fail",
                "sub": "lower is better",
            }
        )

    # Tool-call accuracy.
    tool_id = next((m for m in metrics if m.lower().startswith("tool")), None)
    if tool_id:
        pr_tool = metrics[tool_id].get("pass_rate")
        if pr_tool is not None:
            base_tool = (base_metrics.get(tool_id, {}) or {}).get("pass_rate")
            delta = round(100.0 * (pr_tool - base_tool), 1) if base_tool is not None else 0.0
            sign = "+" if delta >= 0 else ""
            cards.append(
                {
                    "label": "Tool-call accuracy",
                    "value": f"{round(100.0 * pr_tool, 1)}%",
                    "delta": f"{sign}{delta}" if base_tool is not None else "baseline",
                    "delta_kind": "pass" if delta >= 0 else "fail",
                    "sub": tool_id,
                }
            )

    # Median latency from details.latency_ms.
    latencies: list[float] = []
    for mr in run.get("metric_results") or []:
        d = mr.get("details") or {}
        if isinstance(d, dict) and isinstance(d.get("latency_ms"), (int, float)):
            latencies.append(float(d["latency_ms"]))
    if latencies:
        latencies.sort()
        median = latencies[len(latencies) // 2]
        cards.append(
            {
                "label": "Median latency",
                "value": f"{int(median)} ms",
                "delta": "info",
                "delta_kind": "info",
                "sub": f"n={len(latencies)}",
            }
        )

    return cards


def _pass_timeline(run: dict, baseline: dict | None) -> dict:
    """Cumulative pass-rate timeline for ``run`` (and ``baseline`` when given).

    The baseline series is padded or truncated to share an x-axis with
    the current run.
    """
    labels, this = cumulative_pass_series(run, n_points=50)
    series: list[dict] = [{"label": "this run", "data": this, "color_kind": "ember"}]
    if baseline is not None:
        _, b_data = cumulative_pass_series(baseline, n_points=50)
        if len(b_data) > len(this):
            b_data = b_data[: len(this)]
        elif len(b_data) < len(this):
            pad = b_data[-1] if b_data else 0.0
            b_data = b_data + [pad] * (len(this) - len(b_data))
        series.append({"label": "baseline", "data": b_data, "color_kind": "steel"})

    flat = [v for s in series for v in s["data"]]
    return {
        "y_min": round(min(flat), 2) if flat else 0.0,
        "y_max": round(max(flat), 2) if flat else 100.0,
        "ship_gate": 90,
        "x_labels": labels,
        "series": series,
    }


def _failure_clusters(run: dict, baseline: dict | None) -> dict:
    """Per-metric failure buckets (NOT semantic clusters).

    Each entry's ``label`` is the metric_id verbatim - the v2 first cut buckets
    failed items by their first failing metric. The frontend is expected to
    render this as a per-metric grouping; future LLM-driven clustering may
    repurpose the same shape.
    """
    clusters_now = _cluster_run(run)
    clusters_base = _cluster_run(baseline) if baseline else {}
    palette = ["fail", "warn", "violet", "steel", "text3"]
    cluster_list: list[dict] = []
    for idx, (metric_id, items) in enumerate(
        sorted(clusters_now.items(), key=lambda kv: -len(kv[1]))
    ):
        baseline_count = len(clusters_base.get(metric_id, []))
        cluster_list.append(
            {
                "id": f"metric-{metric_id}",
                "label": metric_id,
                "count": len(items),
                "color_kind": palette[idx % len(palette)],
                "regression": len(items) > baseline_count,
            }
        )

    item_ids = {mr.get("item_id") for mr in run.get("metric_results") or []}
    failed_ids = {iid for iid, mids in _failed_metric_ids_per_item(run).items() if mids}
    return {
        "total_failures": len(failed_ids),
        "total_items": len(item_ids),
        "clusters": cluster_list,
    }


def _sub_metrics_detail(run: dict, baseline: dict | None) -> list[dict]:
    metrics = (run.get("summary") or {}).get("metrics") or {}
    base_metrics = ((baseline or {}).get("summary") or {}).get("metrics") or {}
    out: list[dict] = []
    for mid, m in metrics.items():
        pr = m.get("pass_rate")
        if pr is None:
            pr = m.get("avg_score") or 0.0
        bpr = (base_metrics.get(mid, {}) or {}).get("pass_rate")
        if bpr is None:
            bpr = (base_metrics.get(mid, {}) or {}).get("avg_score")
        out.append(
            {
                "label": mid,
                "value": round(100.0 * float(pr), 1),
                "baseline": round(100.0 * float(bpr), 1) if bpr is not None else None,
                "inverse": is_inverse_metric(mid),
            }
        )
    return out


def _confusion(run: dict, baseline: dict) -> dict:
    """Compute 2x2 confusion of item-level pass/fail vs baseline."""

    def item_pass(d: dict) -> dict[str, bool]:
        out: dict[str, bool] = {}
        for r in d.get("metric_results") or []:
            iid = r.get("item_id")
            if iid is None:
                continue
            out.setdefault(iid, True)
            if r.get("passed") is False:
                out[iid] = False
        return out

    now = item_pass(run)
    base = item_pass(baseline)
    bp_p = bp_f = bf_p = bf_f = 0
    for iid, np_pass in now.items():
        bp = base.get(iid)
        if bp is None:
            continue
        if bp and np_pass:
            bp_p += 1
        elif bp and not np_pass:
            bp_f += 1
        elif (not bp) and np_pass:
            bf_p += 1
        else:
            bf_f += 1
    return {
        "base_pass_v_pass": bp_p,
        "base_pass_v_fail": bp_f,
        "base_fail_v_pass": bf_p,
        "base_fail_v_fail": bf_f,
        "net_delta": bf_p - bp_f,
    }


def _failed_items_preview(run: dict) -> list[dict]:
    """Top 3 lowest-scoring failed item rows.

    Pulls ``user`` and ``expected`` from the dataset.jsonl entry rather
    than fabricating empty strings (silent-failure hunter punch list).
    """
    items = load_dataset_items(run_dataset_dir(run))
    failures: dict[str, dict[str, Any]] = {}
    for r in run.get("metric_results") or []:
        if r.get("passed") is not False:
            continue
        iid = r.get("item_id", "")
        score = float(r.get("score") or 0.0)
        existing = failures.get(iid)
        if existing is not None and score >= existing["score"]:
            continue
        details = r.get("details") if isinstance(r.get("details"), dict) else {}
        item = items.get(iid)
        user = input_text(item) or (details.get("input") or "")
        expected = ""
        if isinstance(item, dict):
            exp_val = item.get("expected") or item.get("expected_output")
            if isinstance(exp_val, str):
                expected = exp_val
            elif exp_val is not None:
                expected = json.dumps(exp_val, ensure_ascii=False)
        failures[iid] = {
            "id": iid,
            "user": user[:80],
            "expected": expected[:120],
            "got": (details.get("output") or "")[:80],
            "cluster": r.get("metric_id", ""),
            "score": round(score, 3),
        }
    return sorted(failures.values(), key=lambda x: x["score"])[:3]


@router.get("/{exp_id}")
async def get_experiment(exp_id: str) -> JSONResponse:
    """Return ExperimentDetail for ``exp_id`` or 404."""
    run, baseline, _ = _locate_run(exp_id)
    if run is None:
        raise HTTPException(404, f"experiment {exp_id} not found")

    summary = run.get("summary") or {}
    item_count = len({mr.get("item_id") for mr in run.get("metric_results") or []})
    return JSONResponse(
        {
            "id": run_id(run),
            "name": run_id(run),
            "status": run_status(run),
            "finished_at_iso": run.get("created_at") or "",
            "duration": fmt_duration(summary.get("duration_s")),
            "cost": fmt_cost(total_cost(run)),
            "dataset": {"name": run.get("_dataset", ""), "n": item_count},
            # model/rubric metadata is not yet captured in results.json; ship
            # null instead of fake defaults so the UI renders "-" rather than
            # an empty model chip.
            "model": None,
            "rubric": None,
            "baseline_id": run_id(baseline) if baseline else None,
            "headline": _headline(run, baseline),
            "pass_timeline": _pass_timeline(run, baseline),
            "failure_clusters": _failure_clusters(run, baseline),
            "sub_metrics": _sub_metrics_detail(run, baseline),
            "confusion": _confusion(run, baseline) if baseline else None,
            "failed_items_preview": _failed_items_preview(run),
        }
    )


# ---------- Cluster ----------


def _trigrams(text: str) -> list[tuple[str, str, str]]:
    """Word-level trigrams (lowercased) for trigger phrase mining."""
    toks = [t.lower() for t in text.split() if any(c.isalpha() for c in t)]
    return [(toks[i], toks[i + 1], toks[i + 2]) for i in range(len(toks) - 2)]


def _short_label(run: dict) -> str:
    """Return ``MM-DD`` from ``created_at`` falling back to a short id."""
    ts = parse_iso(run.get("created_at"))
    if ts is not None:
        return ts.strftime("%m-%d")
    return run_id(run)[:8]


def _full_label(run: dict) -> str:
    """Return ``MM-DD HH:MM`` to disambiguate same-day runs."""
    ts = parse_iso(run.get("created_at"))
    if ts is not None:
        return ts.strftime("%m-%d %H:%M")
    return run_id(run)[:10]


def _seconds_label(run: dict) -> str:
    """Return ``MM-DD HH:MM:SS`` to disambiguate same-minute runs."""
    ts = parse_iso(run.get("created_at"))
    if ts is not None:
        return ts.strftime("%m-%d %H:%M:%S")
    return run_id(run)[:12]


def _build_cluster_trend(dataset_runs: list[dict], metric_id: str) -> tuple[list[int], list[str]]:
    """Build a cluster-size trend across recent runs.

    Returns ``([], [])`` when fewer than 3 runs exist in the dataset - a 1- or
    2-point flat line reads as broken to users, so we let the frontend render
    a "not enough history" sentence instead.

    Label escalation (least to most precise) so the chart x-axis is
    always unambiguous:
      1. ``MM-DD`` when every run lands on a distinct day
      2. ``MM-DD HH:MM`` when same-day duplicates appear
      3. ``MM-DD HH:MM:SS`` when two runs share an HH:MM (same minute)
    """
    if len(dataset_runs) < 3:
        return [], []
    recent = dataset_runs[-4:]
    data = [len(_cluster_run(r).get(metric_id, [])) for r in recent]
    short_labels = [_short_label(r) for r in recent]
    if len(set(short_labels)) == len(short_labels):
        return data, short_labels
    full_labels = [_full_label(r) for r in recent]
    if len(set(full_labels)) == len(full_labels):
        return data, full_labels
    return data, [_seconds_label(r) for r in recent]


@router.get("/{exp_id}/cluster/{cluster_id}")
async def get_cluster(exp_id: str, cluster_id: str) -> JSONResponse:
    """Return ClusterDetail for ``exp_id`` + ``cluster_id`` or 404."""
    run, _, dataset_runs = _locate_run(exp_id)
    if run is None:
        raise HTTPException(404, f"experiment {exp_id} not found")

    if not cluster_id.startswith("metric-"):
        raise HTTPException(404, f"cluster {cluster_id} not found")
    metric_id = cluster_id[len("metric-") :]

    clusters = _cluster_run(run)
    if metric_id not in clusters:
        raise HTTPException(404, f"cluster {cluster_id} not found")

    item_ids = clusters[metric_id]
    items_by_id = load_dataset_items(run_dataset_dir(run))

    # Per-item score lookup for the failing metric.
    score_by_item: dict[str, float] = {}
    for r in run.get("metric_results") or []:
        if r.get("metric_id") == metric_id:
            score_by_item[r.get("item_id", "")] = float(r.get("score") or 0.0)

    item_rows: list[dict] = []
    triggers_counter: Counter = Counter()
    for iid in item_ids:
        item = items_by_id.get(iid, {})
        user = input_text(item)
        out = item.get("output") if isinstance(item.get("output"), str) else ""
        item_rows.append(
            {
                "id": iid,
                "user": user[:80],
                "hallucinated": (out or "")[:60],
                "tier": metric_id,
                "score": round(score_by_item.get(iid, 0.0), 3),
            }
        )
        for tg in _trigrams(user):
            triggers_counter[tg] += 1

    triggers = [
        {"phrase": " ".join(tg), "count": cnt} for tg, cnt in triggers_counter.most_common(4)
    ]

    # Trend across last 4 runs in this dataset (dataset_runs is oldest first).
    # Skip the chart entirely when fewer than 3 runs exist; a 1- or 2-point
    # "trend" is misleading and the frontend renders an explanatory sentence
    # instead. Use date+time labels when bare-date duplicates would appear.
    trend_data, trend_labels = _build_cluster_trend(dataset_runs, metric_id)

    total_items_in_run = len({mr.get("item_id") for mr in run.get("metric_results") or []})
    total_failures = sum(1 for mids in _failed_metric_ids_per_item(run).values() if mids)

    return JSONResponse(
        {
            "cluster_id": cluster_id,
            "label": metric_id,
            "pattern": f"Items where metric `{metric_id}` failed.",
            "total_in_cluster": len(item_ids),
            "total_failures_in_run": total_failures,
            "total_items_in_run": total_items_in_run,
            "triggers": triggers,
            "trend": {
                "y_max": max(trend_data) if trend_data else 0,
                "x_labels": trend_labels,
                "data": trend_data,
            },
            "items": item_rows,
            "suggested_fix": None,
        }
    )


# ---------- Semantic clusters (SDK-backed) ----------


_CLUSTER_KIND_FAIL = "fail"
_CLUSTER_KIND_WARN = "warn"
_CLUSTER_KIND_INFO = "info"
_MAX_CLUSTERS = 6


def _semantic_cluster_kind(items_in_cluster: list[dict]) -> str:
    """Heuristic kind based on the cluster's worst score.

    All-failed -> fail; mixed soft -> warn; otherwise info. Keeps the
    classifier simple given the SDK only emits failure-derived clusters.
    """
    scores = [it.get("score", 0.0) for it in items_in_cluster]
    if not scores:
        return _CLUSTER_KIND_INFO
    if all(s == 0.0 or s is False for s in scores):
        return _CLUSTER_KIND_FAIL
    if any(0.3 < float(s or 0.0) < 0.7 for s in scores):
        return _CLUSTER_KIND_WARN
    return _CLUSTER_KIND_FAIL


def _truncate(text: str, n: int = 120) -> str:
    if not isinstance(text, str):
        return ""
    return text if len(text) <= n else text[: n - 1] + "…"


def _build_metric_clusters_fallback(run: dict) -> list[dict]:
    """Build clusters by metric grouping when the SDK is unavailable.

    Same shape as the semantic response so the FE doesn't need to branch.
    """
    clusters_now = _cluster_run(run)
    items_by_id = load_dataset_items(run_dataset_dir(run))
    out: list[dict] = []
    for metric_id, ids in sorted(clusters_now.items(), key=lambda kv: -len(kv[1])):
        sample_items: list[dict] = []
        for iid in ids[:25]:
            item = items_by_id.get(iid, {})
            sample_items.append({"id": iid, "text": _truncate(input_text(item), 120)})
        out.append(
            {
                "id": f"metric-{metric_id}",
                "label": metric_id,
                "n": len(ids),
                "confidence": 0.7,
                "kind": _CLUSTER_KIND_FAIL,
                "pattern_hint": None,
                "items": sample_items,
            }
        )
    return out[:_MAX_CLUSTERS]


def _try_sdk_clusters(run: dict) -> list[dict] | None:
    """Try the SDK's ``cluster_failures`` path. Returns ``None`` on failure.

    Gracefully degrades: any import or runtime issue (no API key, no
    network, etc.) returns ``None`` and the caller falls back to the
    metric-grouping clusters. We avoid hitting the LLM in tests by
    refusing to call when fewer than 3 failures or when the env signals
    a test (``EVALYN_DASHBOARD_TEST_NO_CACHE`` is the dashboard's test
    seam; reuse it here as a "no external calls" signal).
    """
    import os as _os

    if _os.environ.get("EVALYN_DASHBOARD_TEST_NO_CACHE") == "1":
        return None
    if _os.environ.get("EVALYN_DASHBOARD_DISABLE_LLM_CLUSTER") == "1":
        return None
    try:
        from evalyn_sdk.analysis.clustering import ReasonClusterer  # noqa: F401
        from evalyn_sdk.models import MetricResult  # noqa: F401
    except ImportError:
        return None

    failed_results: list[Any] = []
    try:
        for r in run.get("metric_results") or []:
            if r.get("passed") is not False:
                continue
            try:
                mr = MetricResult.from_dict(r)
            except Exception:  # noqa: BLE001
                continue
            failed_results.append(mr)
    except Exception:  # noqa: BLE001
        return None

    if len(failed_results) < 3:
        # SDK falls back to one-cluster-per-case; not useful for the UI.
        return None

    try:
        clusterer = ReasonClusterer()
        result = clusterer.cluster_failures(
            metric_results=failed_results,
            dataset_items=None,
            compute_embeddings=False,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("semantic clustering failed: %s", exc)
        return None

    items_by_id = load_dataset_items(run_dataset_dir(run))
    score_by_item: dict[str, float] = {}
    for r in run.get("metric_results") or []:
        iid = r.get("item_id")
        if iid is not None:
            score_by_item[iid] = float(r.get("score") or 0.0)

    out: list[dict] = []
    for cluster in (result.clusters or [])[:_MAX_CLUSTERS]:
        sample_items: list[dict] = []
        for reason in (cluster.reasons or [])[:25]:
            iid = ""
            text = ""
            for r in run.get("metric_results") or []:
                if r.get("passed") is not False:
                    continue
                details = r.get("details") if isinstance(r.get("details"), dict) else {}
                raw_judge = r.get("raw_judge") if isinstance(r.get("raw_judge"), dict) else {}
                rsn = (raw_judge or {}).get("reason") or (details or {}).get("reason") or ""
                if rsn == reason:
                    iid = r.get("item_id") or ""
                    text = input_text(items_by_id.get(iid, {})) or ""
                    break
            if not iid:
                continue
            sample_items.append({"id": iid, "text": _truncate(text, 120)})
        scores = [score_by_item.get(it["id"], 0.0) for it in sample_items]
        kind = _semantic_cluster_kind(
            [{"score": s} for s in scores] if scores else [{"score": 0.0}]
        )
        out.append(
            {
                "id": cluster.cluster_id,
                "label": cluster.label,
                "n": cluster.count,
                # Compactness proxy: shorter mean reason length -> tighter
                # cluster. Keep within 0..1; default to 0.7 when unknown.
                "confidence": 0.7,
                "kind": kind,
                "pattern_hint": cluster.label,
                "items": sample_items,
            }
        )
    return out or None


@router.get("/{exp_id}/clusters")
async def get_clusters(exp_id: str) -> JSONResponse:
    """Semantic-cluster view of a run's failed items.

    Tries the SDK's LLM-based clusterer first (``cluster_failures``); on
    any failure (no API key, transient network, very small failure count)
    falls back to the existing metric-grouping. Capped to 6 clusters,
    with anything beyond folded into an "Other" cluster.
    """
    run, _, _ = _locate_run(exp_id)
    if run is None:
        raise HTTPException(404, f"experiment {exp_id} not found")

    clusters = _try_sdk_clusters(run)
    if clusters is None:
        clusters = _build_metric_clusters_fallback(run)

    # Cap to _MAX_CLUSTERS, lump tail into a single "Other" entry.
    if len(clusters) > _MAX_CLUSTERS:
        head = clusters[: _MAX_CLUSTERS - 1]
        tail = clusters[_MAX_CLUSTERS - 1 :]
        other_n = sum(c["n"] for c in tail)
        other_items: list[dict] = []
        for c in tail:
            other_items.extend(c["items"][:5])
        head.append(
            {
                "id": "other",
                "label": "Other",
                "n": other_n,
                "confidence": 0.5,
                "kind": _CLUSTER_KIND_INFO,
                "pattern_hint": None,
                "items": other_items[:25],
            }
        )
        clusters = head

    return JSONResponse({"clusters": clusters})
