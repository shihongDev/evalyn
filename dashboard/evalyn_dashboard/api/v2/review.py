"""``/api/v2/review`` router.

Endpoints:
- ``GET /queue``    -> ``ReviewQueue``: items needing human review. The
  selection cascade (primary then 3 fallbacks) keeps the queue useful
  even when runs are mostly programmatic; see :func:`_build_queue`.
- ``POST /verdict`` -> ``{ok: True}``: appends a verdict to
  ``.evalyn/data/datasets/<dataset>/reviews/<run_id>.jsonl``.

Source-of-truth shapes live in
``dashboard/frontend/src/v2/api/types.ts``.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ._shared import (
    input_text,
    load_all_runs,
    load_dataset_items,
    run_dataset_dir,
    run_id,
)

router = APIRouter()

# Primary uncertainty band - explicit ``judge_confidence``.
PRIMARY_LO = 0.35
PRIMARY_HI = 0.65
# Fallback 1 band - any soft-scored metric in this range.
SOFT_LO = 0.3
SOFT_HI = 0.7
QUEUE_CAP = 50
FALLBACK3_CAP = 20


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


def _is_soft_score(score: float | int | None) -> bool:
    """A score is soft (LLM-judge-style) when strictly between 0 and 1."""
    if not isinstance(score, (int, float)):
        return False
    return 0.0 < float(score) < 1.0


def _iter_runs(runs: list[dict]):
    """Yield ``(rid, items_by_id, metric_results)`` per run, with a per-call dataset cache."""
    items_cache: dict[Path, dict[str, dict]] = {}
    for run in runs:
        dataset_dir = run_dataset_dir(run)
        items_by_id = items_cache.setdefault(
            dataset_dir, load_dataset_items(dataset_dir)
        )
        yield run_id(run), items_by_id, run.get("metric_results") or []


def _primary_candidates(runs: list[dict]) -> list[tuple[float, dict]]:
    """Strategy 1: explicit ``details.judge_confidence`` in [0.35, 0.65]."""
    out: list[tuple[float, dict]] = []
    for rid, items_by_id, mrs in _iter_runs(runs):
        for mr in mrs:
            details = mr.get("details") if isinstance(mr.get("details"), dict) else {}
            jc = details.get("judge_confidence")
            if not isinstance(jc, (int, float)) or not PRIMARY_LO <= jc <= PRIMARY_HI:
                continue
            row = _row_for_metric(mr, items_by_id.get(mr.get("item_id", ""), {}), rid, float(jc))
            out.append((abs(float(jc) - 0.5), row))
    return out


def _soft_score_candidates(runs: list[dict]) -> list[tuple[float, dict]]:
    """Strategy 2: any metric with a soft score in [0.3, 0.7]."""
    out: list[tuple[float, dict]] = []
    for rid, items_by_id, mrs in _iter_runs(runs):
        for mr in mrs:
            score = mr.get("score")
            if not _is_soft_score(score):
                continue
            s = float(score)
            if not SOFT_LO <= s <= SOFT_HI:
                continue
            row = _row_for_metric(mr, items_by_id.get(mr.get("item_id", ""), {}), rid, s)
            out.append((abs(s - 0.5), row))
    return out


def _disagreement_candidates(runs: list[dict]) -> list[tuple[float, dict]]:
    """Strategy 3: items where some metrics pass and some fail."""
    out: list[tuple[float, dict]] = []
    for rid, items_by_id, mrs in _iter_runs(runs):
        by_item: dict[str, list[dict]] = defaultdict(list)
        for mr in mrs:
            iid = mr.get("item_id", "")
            if iid:
                by_item[iid].append(mr)
        for iid, results in by_item.items():
            verdicts = {r.get("passed") for r in results}
            if True not in verdicts or False not in verdicts:
                continue
            failed = next(
                (r for r in results if r.get("passed") is False),
                results[0],
            )
            score = float(failed.get("score") or 0.0)
            row = _row_for_metric(failed, items_by_id.get(iid, {}), rid, 0.5)
            # Sort key: prefer items with more failed metrics (more conflict);
            # tiebreak by score so equally-conflicted items prioritize the worst.
            n_fail = sum(1 for r in results if r.get("passed") is False)
            out.append((-n_fail + score / 1000.0, row))
    return out


def _lowest_failed_candidates(runs: list[dict]) -> list[tuple[float, dict]]:
    """Strategy 4: lowest-scoring failed items overall."""
    out: list[tuple[float, dict]] = []
    for rid, items_by_id, mrs in _iter_runs(runs):
        for mr in mrs:
            if mr.get("passed") is not False:
                continue
            score = float(mr.get("score") or 0.0)
            row = _row_for_metric(mr, items_by_id.get(mr.get("item_id", ""), {}), rid, score)
            out.append((score, row))
    return out


def _select(
    candidates: list[tuple[float, dict]], cap: int
) -> list[dict]:
    """Dedupe by (source_run_id, item_id), sort by key ascending, and cap."""
    seen: set[tuple[str, str]] = set()
    deduped: list[tuple[float, dict]] = []
    for sortkey, row in candidates:
        key = (row.get("source_run_id", ""), row.get("item_id", ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append((sortkey, row))
    deduped.sort(key=lambda t: t[0])
    return [row for _, row in deduped[:cap]]


def _build_queue(runs: list[dict]) -> tuple[list[dict], str]:
    """Apply the cascade. Returns ``(items, rationale)``."""
    items = _select(_primary_candidates(runs), QUEUE_CAP)
    if items:
        return items, (
            f"Showing {len(items)} items where the judge gave a borderline score "
            f"({PRIMARY_LO}-{PRIMARY_HI})."
        )

    items = _select(_soft_score_candidates(runs), QUEUE_CAP)
    if items:
        return items, (
            f"Showing {len(items)} items with soft judge scores between "
            f"{SOFT_LO} and {SOFT_HI} - no explicit confidence field, but the "
            "score itself is in the uncertain band."
        )

    items = _select(_disagreement_candidates(runs), QUEUE_CAP)
    if items:
        return items, (
            f"Showing {len(items)} items where multiple metrics disagreed - "
            "the judge may need calibration."
        )

    items = _select(_lowest_failed_candidates(runs), FALLBACK3_CAP)
    if items:
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
    return JSONResponse(
        {
            "items": items,
            "reviewers": [{"name": "You", "done": 0, "total": len(items), "you": True}],
            "rationale": rationale,
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
