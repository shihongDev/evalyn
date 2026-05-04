"""``/api/v2/experiments/{run_id}/items`` sub-router.

Returns one row per item in the run with per-metric pass/score data,
plus a short input/expected/output preview so the Items tab on the
frontend can render its read-deep table without a second roundtrip per
row.

JSON shape (must match ``dashboard/frontend/src/v2/api/types.ts``):

    {
      "total": int,
      "offset": int,
      "limit": int,
      "items": [ItemRow, ...],
      "metric_ids": [str, ...]
    }

    ItemRow = {
      "id": str,
      "input_preview": str,         # first 120 chars of input
      "expected_preview": str|None, # first 120 chars or None
      "output_preview": str|None,   # first 120 chars or None
      "per_metric": [
        {"metric_id": str, "passed": bool|None, "score": float|None},
        ...
      ],
      "any_failed": bool,
      "passed_count": int,
      "failed_count": int,
    }

Filters apply BEFORE paginating; sort then page.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from ._shared import (
    input_text,
    load_all_runs,
    load_dataset_items,
    run_dataset_dir,
    run_id,
)

logger = logging.getLogger(__name__)

PREVIEW_CHARS = 120


def _locate_run(run_id_query: str) -> dict | None:
    """Return the run dict matching ``run_id_query`` or ``None``."""
    for r in load_all_runs():
        if run_id(r) == run_id_query or r.get("id") == run_id_query:
            return r
    return None


def _expected_preview(item: dict | None) -> str | None:
    """Best-effort short preview of the expected value."""
    if not isinstance(item, dict):
        return None
    exp = item.get("expected")
    if exp is None:
        exp = item.get("expected_output")
    if exp is None:
        return None
    if isinstance(exp, str):
        text = exp
    else:
        try:
            text = json.dumps(exp, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            logger.warning("expected serialization failed: %s", exc)
            return None
    return text[:PREVIEW_CHARS]


def _output_preview_from_item(item: dict | None) -> str | None:
    """Best-effort output preview pulled from the dataset entry."""
    if not isinstance(item, dict):
        return None
    out = item.get("output")
    if out is None:
        return None
    if isinstance(out, str):
        return out[:PREVIEW_CHARS]
    try:
        return json.dumps(out, ensure_ascii=False)[:PREVIEW_CHARS]
    except (TypeError, ValueError) as exc:
        logger.warning("output serialization failed: %s", exc)
        return None


def _output_preview_from_metric_details(details: Any) -> str | None:
    """Fallback output preview from a metric_result's ``details.output``."""
    if not isinstance(details, dict):
        return None
    out = details.get("output")
    if out is None:
        return None
    if isinstance(out, str):
        return out[:PREVIEW_CHARS]
    try:
        return json.dumps(out, ensure_ascii=False)[:PREVIEW_CHARS]
    except (TypeError, ValueError) as exc:
        logger.warning("metric output serialization failed: %s", exc)
        return None


def _build_rows(run: dict) -> tuple[list[dict], list[str]]:
    """Construct (rows, metric_ids) for the run.

    ``metric_ids`` preserves first-seen order across metric_results so
    the per-metric columns stay stable run-over-run.
    """
    metric_results = run.get("metric_results") or []
    if not metric_results:
        return [], []

    metric_ids: list[str] = []
    seen_metric: set[str] = set()
    by_item: dict[str, list[dict]] = defaultdict(list)
    item_order: list[str] = []
    for mr in metric_results:
        iid = mr.get("item_id")
        if iid is None:
            continue
        if iid not in by_item:
            item_order.append(iid)
        by_item[iid].append(mr)
        mid = mr.get("metric_id")
        if mid and mid not in seen_metric:
            seen_metric.add(mid)
            metric_ids.append(mid)

    items_by_id = load_dataset_items(run_dataset_dir(run))

    rows: list[dict] = []
    for iid in item_order:
        item = items_by_id.get(iid)
        results_for_item = by_item[iid]
        per_metric_lookup: dict[str, dict] = {}
        for mr in results_for_item:
            mid = mr.get("metric_id")
            if not mid:
                continue
            # Preserve last-write-wins for duplicate metric_id rows.
            per_metric_lookup[mid] = mr

        per_metric: list[dict] = []
        passed_count = 0
        failed_count = 0
        for mid in metric_ids:
            mr = per_metric_lookup.get(mid)
            if mr is None:
                per_metric.append({"metric_id": mid, "passed": None, "score": None})
                continue
            passed_val = mr.get("passed")
            score_val = mr.get("score")
            score_out: float | None
            if isinstance(score_val, (int, float)):
                score_out = float(score_val)
            else:
                score_out = None
            passed_out: bool | None
            if isinstance(passed_val, bool):
                passed_out = passed_val
                if passed_val:
                    passed_count += 1
                else:
                    failed_count += 1
            else:
                passed_out = None
            per_metric.append({"metric_id": mid, "passed": passed_out, "score": score_out})

        any_failed = failed_count > 0

        # Output: prefer dataset value, fall back to first metric result with details.output.
        output_preview = _output_preview_from_item(item)
        if output_preview is None:
            for mr in results_for_item:
                fallback = _output_preview_from_metric_details(mr.get("details"))
                if fallback is not None:
                    output_preview = fallback
                    break

        rows.append(
            {
                "id": iid,
                "input_preview": (input_text(item) or "")[:PREVIEW_CHARS],
                "expected_preview": _expected_preview(item),
                "output_preview": output_preview,
                "per_metric": per_metric,
                "any_failed": any_failed,
                "passed_count": passed_count,
                "failed_count": failed_count,
            }
        )

    return rows, metric_ids


def _apply_filter(rows: list[dict], filter_value: str) -> list[dict]:
    if filter_value == "passed":
        return [r for r in rows if not r["any_failed"] and r["passed_count"] > 0]
    if filter_value == "failed":
        return [r for r in rows if r["any_failed"]]
    return rows


def _apply_sort(rows: list[dict], sort_value: str) -> list[dict]:
    if sort_value == "score":
        # Sum of metric scores ascending (worst first) - useful pairing
        # with the failed filter for triage.
        def score_key(r: dict) -> tuple[float, str]:
            total = 0.0
            for pm in r["per_metric"]:
                s = pm.get("score")
                if isinstance(s, (int, float)):
                    total += float(s)
            return (total, r["id"])
        return sorted(rows, key=score_key)
    return sorted(rows, key=lambda r: r["id"])


def register(router: APIRouter) -> None:
    """Register the /{exp_id}/items route on the given router."""

    @router.get("/{exp_id}/items")
    async def get_experiment_items(
        exp_id: str,
        offset: int = Query(0, ge=0),
        limit: int = Query(50, ge=1, le=500),
        filter: str = Query("all", pattern="^(all|passed|failed)$"),
        sort: str = Query("item_id", pattern="^(item_id|score)$"),
    ) -> JSONResponse:
        run = _locate_run(exp_id)
        if run is None:
            raise HTTPException(404, f"experiment {exp_id} not found")

        rows, metric_ids = _build_rows(run)
        filtered = _apply_filter(rows, filter)
        sorted_rows = _apply_sort(filtered, sort)
        total = len(sorted_rows)
        page = sorted_rows[offset : offset + limit]
        return JSONResponse(
            {
                "total": total,
                "offset": offset,
                "limit": limit,
                "items": page,
                "metric_ids": metric_ids,
            }
        )
