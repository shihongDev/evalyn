"""``/api/v2/experiments`` router.

Endpoints:
- ``GET /``                          -> ExperimentList
- ``GET /{id}``                      -> ExperimentDetail
- ``GET /{id}/cluster/{cluster_id}`` -> ClusterDetail

Source of truth for the JSON shapes is
``dashboard/frontend/src/v2/api/types.ts``.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from ._shared import (
    cumulative_pass_series,
    datasets_root,
    fmt_cost,
    fmt_duration,
    input_text,
    is_inverse_metric,
    load_all_runs,
    load_dataset_items,
    run_dataset_dir,
    run_id,
    run_pass_rate,
    run_status,
    total_cost,
)

router = APIRouter()


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
    if prev is None:
        return "baseline"
    if curr is None:
        return "-"
    diff = round(100.0 * (curr - prev), 1)
    sign = "+" if diff >= 0 else ""
    return f"{sign}{diff}"


def _serialize_row(run: dict, prev: dict | None) -> dict:
    pr = run_pass_rate(run)
    summary = run.get("summary") or {}
    return {
        "id": run_id(run),
        "name": run_id(run),
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
        **({"err": summary.get("error", "failed")} if summary.get("status") == "failed" else {}),
    }


@router.get("")
async def list_experiments() -> JSONResponse:
    """Return one row per run, newest first, with delta vs previous run."""
    root = datasets_root()
    if not root.exists():
        return JSONResponse([])
    runs = load_all_runs(root)  # oldest first
    if not runs:
        return JSONResponse([])

    # Group by dataset to compute per-dataset deltas.
    by_dataset: dict[str, list[dict]] = defaultdict(list)
    for r in runs:
        by_dataset[r["_dataset"]].append(r)

    rows: list[dict] = []
    for dataset, drs in by_dataset.items():
        for idx, r in enumerate(drs):
            prev = drs[idx - 1] if idx > 0 else None
            rows.append(_serialize_row(r, prev))

    rows.sort(key=lambda x: x.get("when_iso") or "", reverse=True)
    return JSONResponse(rows)


# ---------- Detail ----------


def _locate_run(run_id_query: str) -> tuple[dict | None, dict | None, list[dict]]:
    """Single-scan lookup. Returns ``(run, baseline, dataset_runs)``.

    ``baseline`` is the run immediately preceding ``run`` in the same
    dataset (chronological), or ``None`` if it's the oldest. ``dataset_runs``
    is every run in the same dataset, oldest first - reused by the cluster
    handler to compute its trend without a second disk scan.
    """
    all_runs = load_all_runs(datasets_root())  # oldest first
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
        cards.append({
            "label": "Pass rate",
            "value": f"{round(pr * 100, 1)}%",
            "delta": f"{sign}{delta}" if base_pr is not None else "baseline",
            "delta_kind": "pass" if delta >= 0 else "fail",
            "sub": "weighted across metrics",
        })

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
        cards.append({
            "label": "Hallucination",
            "value": f"{rate}%",
            "delta": f"{sign}{delta}" if base_avg is not None else "baseline",
            "delta_kind": "pass" if delta <= 0 else "fail",
            "sub": "lower is better",
        })

    # Tool-call accuracy.
    tool_id = next((m for m in metrics if m.lower().startswith("tool")), None)
    if tool_id:
        pr_tool = metrics[tool_id].get("pass_rate")
        if pr_tool is not None:
            base_tool = (base_metrics.get(tool_id, {}) or {}).get("pass_rate")
            delta = round(100.0 * (pr_tool - base_tool), 1) if base_tool is not None else 0.0
            sign = "+" if delta >= 0 else ""
            cards.append({
                "label": "Tool-call accuracy",
                "value": f"{round(100.0 * pr_tool, 1)}%",
                "delta": f"{sign}{delta}" if base_tool is not None else "baseline",
                "delta_kind": "pass" if delta >= 0 else "fail",
                "sub": tool_id,
            })

    # Median latency from details.latency_ms.
    latencies: list[float] = []
    for mr in run.get("metric_results") or []:
        d = mr.get("details") or {}
        if isinstance(d, dict) and isinstance(d.get("latency_ms"), (int, float)):
            latencies.append(float(d["latency_ms"]))
    if latencies:
        latencies.sort()
        median = latencies[len(latencies) // 2]
        cards.append({
            "label": "Median latency",
            "value": f"{int(median)} ms",
            "delta": "info",
            "delta_kind": "info",
            "sub": f"n={len(latencies)}",
        })

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
    clusters_now = _cluster_run(run)
    clusters_base = _cluster_run(baseline) if baseline else {}
    palette = ["fail", "warn", "violet", "steel", "text3"]
    cluster_list: list[dict] = []
    for idx, (metric_id, items) in enumerate(
        sorted(clusters_now.items(), key=lambda kv: -len(kv[1]))
    ):
        baseline_count = len(clusters_base.get(metric_id, []))
        cluster_list.append({
            "id": f"metric-{metric_id}",
            "label": metric_id,
            "count": len(items),
            "color_kind": palette[idx % len(palette)],
            "regression": len(items) > baseline_count,
        })

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
        out.append({
            "label": mid,
            "value": round(100.0 * float(pr), 1),
            "baseline": round(100.0 * float(bpr), 1) if bpr is not None else None,
            "inverse": is_inverse_metric(mid),
        })
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


@router.get("/{exp_id}/cluster/{cluster_id}")
async def get_cluster(exp_id: str, cluster_id: str) -> JSONResponse:
    """Return ClusterDetail for ``exp_id`` + ``cluster_id`` or 404."""
    run, _, dataset_runs = _locate_run(exp_id)
    if run is None:
        raise HTTPException(404, f"experiment {exp_id} not found")

    if not cluster_id.startswith("metric-"):
        raise HTTPException(404, f"cluster {cluster_id} not found")
    metric_id = cluster_id[len("metric-"):]

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
        {"phrase": " ".join(tg), "count": cnt}
        for tg, cnt in triggers_counter.most_common(4)
    ]

    # Trend across last 4 runs in this dataset (dataset_runs is oldest first).
    recent = dataset_runs[-4:]
    trend_data = [len(_cluster_run(r).get(metric_id, [])) for r in recent]
    trend_labels = [run_id(r)[:8] for r in recent]

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
