"""``/api/v2/review`` router.

Endpoints:
- ``GET /queue``    -> ``ReviewQueue``: items needing human review. The
  selection cascade (primary then 3 fallbacks) keeps the queue useful
  even when runs are mostly programmatic; see :func:`_build_queue`.
- ``POST /verdict`` -> ``{ok: True}``: appends a verdict to
  ``.evalyn/data/datasets/<dataset>/reviews/<run_id>.jsonl``.

The queue response also carries ``calibration_suggestions`` - one
entry per (dataset, metric_id) that has accumulated enough verdicts
to make ``evalyn calibrate`` worthwhile. The frontend renders these
as a banner with a "Run calibrate" deep-link.

Source-of-truth shapes live in
``dashboard/frontend/src/v2/api/types.ts``.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ._shared import (
    PRIMARY_HI,
    PRIMARY_LO,
    SOFT_HI,
    SOFT_LO,
    _is_soft_score,
    calibration_suggestions,
    input_text,
    load_all_runs,
    load_dataset_items,
    run_dataset_dir,
    run_id,
)

logger = logging.getLogger(__name__)

router = APIRouter()

QUEUE_CAP = 50
FALLBACK3_CAP = 20


def _clear_review_caches_for_tests() -> None:
    """Backwards-compat shim. Caches now live in ``_shared.py``; the
    central :func:`_clear_caches_for_tests` clears them so this is a
    no-op kept only for tests that imported it directly.
    """
    return None


def _row_for_metric(
    mr: dict,
    item: dict,
    rid: str,
    confidence_value: float,
) -> dict:
    """Build the standard ReviewItem row from a metric_result + dataset item."""
    iid = mr.get("item_id", "")
    details = mr.get("details") if isinstance(mr.get("details"), dict) else {}
    user_text = input_text(item)
    agent_response = ""
    if isinstance(item.get("output"), str):
        agent_response = item["output"]
    elif isinstance(details.get("output"), str):
        agent_response = details["output"]
    expected = ""
    if isinstance(item.get("expected"), str):
        expected = item["expected"]
    judge_reasoning = ""
    if isinstance(details.get("reason"), str):
        judge_reasoning = details["reason"]
    elif isinstance(details.get("judge_reason"), str):
        judge_reasoning = details["judge_reason"]
    return {
        "item_id": iid,
        "category": item.get("category") or "uncategorized",
        "judge_confidence": round(confidence_value, 3),
        "user_text": user_text,
        "agent_response": agent_response,
        "expected": expected,
        "highlights": [],
        "source_run_id": rid,
        "source_run_label": rid,
        "judge_breakdown": [
            {
                "label": mr.get("metric_id", ""),
                "score": round(float(mr.get("score") or 0.0), 3),
                "kind": "warn",
            }
        ],
        "judge_reasoning": judge_reasoning,
    }


def _select_refs(
    refs: list[tuple[float, str, str, dict, float]],
    cap: int,
    items_cache: dict[Path, dict[str, dict]],
    run_ds_dir: dict[str, Path],
) -> list[dict]:
    """Dedupe by ``(rid, iid)``, sort by sortkey, cap, then materialise rows.

    Rows are only built for the surviving candidates so the hot path
    avoids a ``_row_for_metric`` call per ``(item, metric_result)``.
    """
    seen: set[tuple[str, str]] = set()
    deduped: list[tuple[float, str, str, dict, float]] = []
    for ref in refs:
        _, rid, iid, _, _ = ref
        key = (rid, iid)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ref)
    deduped.sort(key=lambda t: t[0])
    out: list[dict] = []
    for _, rid, iid, mr, conf in deduped[:cap]:
        ds_dir = run_ds_dir.get(rid)
        items_by_id = items_cache.get(ds_dir) if ds_dir is not None else None
        item = items_by_id.get(iid) if items_by_id else {}
        out.append(_row_for_metric(mr, item or {}, rid, conf))
    return out


def _build_queue(runs: list[dict]) -> tuple[list[dict], str]:
    """Apply the cascade in a single pass over runs+metric_results.

    The previous implementation walked ``load_all_runs()`` four times -
    once per strategy - and each walk re-iterated every run's
    ``metric_results``. With ~150 runs and ~33k metric_results that's
    ~130k visits warm. This single-pass version classifies each
    ``(item, metric_result)`` once into the highest-priority bucket it
    qualifies for, defers row construction until after sort+cap, and
    only touches the dataset items map for the surviving candidates.

    Returns ``(items, rationale)``.
    """
    # Lightweight per-classification refs:
    # ``(sortkey, rid, item_id, metric_result, confidence)``.
    primary: list[tuple[float, str, str, dict, float]] = []
    soft: list[tuple[float, str, str, dict, float]] = []
    failed: list[tuple[float, str, str, dict, float]] = []
    # Per-(rid, item_id) failing ref + fail-count so disagreement only
    # emits one ref per item even when several metrics fail.
    disagreement: dict[tuple[str, str], tuple[int, dict]] = {}

    run_ds_dir: dict[str, Path] = {}
    items_cache: dict[Path, dict[str, dict]] = {}

    for run in runs:
        ds_dir = run_dataset_dir(run)
        rid = run_id(run)
        run_ds_dir[rid] = ds_dir
        # Group metric_results per item so we can spot pass/fail
        # disagreement after the per-result classification.
        item_results: dict[str, list[dict]] = defaultdict(list)
        for mr in run.get("metric_results") or []:
            iid = mr.get("item_id", "")
            if not iid:
                continue
            item_results[iid].append(mr)
            details = mr.get("details") if isinstance(mr.get("details"), dict) else {}
            jc = details.get("judge_confidence")
            if isinstance(jc, (int, float)) and PRIMARY_LO <= jc <= PRIMARY_HI:
                jcf = float(jc)
                primary.append((abs(jcf - 0.5), rid, iid, mr, jcf))
                continue
            score = mr.get("score")
            if _is_soft_score(score) and SOFT_LO <= float(score) <= SOFT_HI:
                s = float(score)
                soft.append((abs(s - 0.5), rid, iid, mr, s))
                continue
            if mr.get("passed") is False:
                fs = float(score or 0.0)
                failed.append((fs, rid, iid, mr, fs))

        # After per-run walk, detect disagreement (some pass + some fail)
        # using the grouped results.
        for iid, results in item_results.items():
            verdicts = {r.get("passed") for r in results}
            if True not in verdicts or False not in verdicts:
                continue
            failing = next(
                (r for r in results if r.get("passed") is False),
                results[0],
            )
            n_fail = sum(1 for r in results if r.get("passed") is False)
            disagreement[(rid, iid)] = (n_fail, failing)

    def _materialise(refs, cap):
        # Lazy-load dataset items maps only for surviving rows. The
        # ``cap * 4`` slice covers the dedupe set with margin so we
        # don't load a map only to throw it away after the dedupe drops
        # most candidates from a single dataset.
        candidate_ds_dirs = {run_ds_dir.get(rid) for _, rid, _, _, _ in refs[: cap * 4]}
        for d in candidate_ds_dirs:
            if d is not None and d not in items_cache:
                items_cache[d] = load_dataset_items(d)
        return _select_refs(refs, cap, items_cache, run_ds_dir)

    if primary:
        items = _materialise(primary, QUEUE_CAP)
        return items, (
            f"Showing {len(items)} items where the judge gave a borderline score "
            f"({PRIMARY_LO}-{PRIMARY_HI})."
        )

    if soft:
        items = _materialise(soft, QUEUE_CAP)
        return items, (
            f"Showing {len(items)} items with soft judge scores between "
            f"{SOFT_LO} and {SOFT_HI} - no explicit confidence field, but the "
            "score itself is in the uncertain band."
        )

    if disagreement:
        # Sort key: more failed metrics wins (negative makes ascending
        # sort prioritise high-conflict items first).
        refs = [
            (-float(n_fail), rid, iid, failing, 0.5)
            for (rid, iid), (n_fail, failing) in disagreement.items()
        ]
        items = _materialise(refs, QUEUE_CAP)
        return items, (
            f"Showing {len(items)} items where multiple metrics disagreed - "
            "the judge may need calibration."
        )

    if failed:
        items = _materialise(failed, FALLBACK3_CAP)
        return items, (
            f"Showing the {len(items)} lowest-scoring failed items - your runs "
            "have no soft-scored metrics yet, so we picked the clearest failures "
            "for human review."
        )
    return [], "No items to review: every metric passed and no judge confidence band fired."


@router.get("/queue")
async def get_queue() -> JSONResponse:
    """Return up to 50 review items chosen by the cascade strategy."""
    runs = load_all_runs()
    items, rationale = _build_queue(runs)
    suggestions = calibration_suggestions(runs)
    return JSONResponse(
        {
            "items": items,
            "reviewers": [{"name": "You", "done": 0, "total": len(items), "you": True}],
            "rationale": rationale,
            "calibration_suggestions": suggestions,
        }
    )


# ---------- POST /verdict ----------


class VerdictBody(BaseModel):
    """Request body for ``POST /verdict``."""

    item_id: str = Field(..., min_length=1)
    source_run_id: str = Field(..., min_length=1)
    verdict: str = Field(..., pattern="^(pass|fail|skip)$")
    note: str | None = None


def _resolve_run_dataset_dir(source_run_id: str) -> Path | None:
    """Return the dataset dir containing ``source_run_id`` or ``None``."""
    for run in load_all_runs():
        if run_id(run) == source_run_id or run.get("id") == source_run_id:
            return run_dataset_dir(run)
    return None


@router.post("/verdict")
async def post_verdict(body: VerdictBody) -> JSONResponse:
    """Append a verdict line to the per-run reviews file."""
    dataset_dir = _resolve_run_dataset_dir(body.source_run_id)
    if dataset_dir is None:
        raise HTTPException(404, f"source run {body.source_run_id} not found")
    reviews_dir = dataset_dir / "reviews"
    reviews_dir.mkdir(parents=True, exist_ok=True)
    out_path = reviews_dir / f"{body.source_run_id}.jsonl"
    line = json.dumps(
        {
            "item_id": body.item_id,
            "source_run_id": body.source_run_id,
            "verdict": body.verdict,
            "note": body.note,
            "at": datetime.now(timezone.utc).isoformat(),
        }
    )
    with out_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    return JSONResponse({"ok": True})
