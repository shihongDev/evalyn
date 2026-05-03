"""``/api/v2/review`` router.

Endpoints:
- ``GET /queue``    -> ``ReviewQueue``: items needing human review (judge
  confidence in [0.4, 0.6] or score in [0.4, 0.6] for LLM-judge metrics).
- ``POST /verdict`` -> ``{ok: True}``: appends a verdict to
  ``.evalyn/data/datasets/<dataset>/reviews/<run_id>.jsonl``.

Source-of-truth shapes live in
``dashboard/frontend/src/v2/api/types.ts``.
"""

from __future__ import annotations

import json
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

UNCERTAIN_LO = 0.4
UNCERTAIN_HI = 0.6
QUEUE_CAP = 50


def _judge_metric_ids(run: dict) -> set[str]:
    """Metric ids that this run treats as LLM-judge based."""
    out: set[str] = set()
    for jc in run.get("judge_configs") or []:
        if isinstance(jc, dict):
            jid = jc.get("metric_id") or jc.get("id")
            if jid:
                out.add(jid)
    return out


def _confidence(mr: dict, judge_ids: set[str]) -> float | None:
    """Return judge confidence in ``[0,1]`` if uncertain, else ``None``."""
    details = mr.get("details") if isinstance(mr.get("details"), dict) else {}
    jc = details.get("judge_confidence")
    if isinstance(jc, (int, float)) and UNCERTAIN_LO <= jc <= UNCERTAIN_HI:
        return float(jc)
    score = mr.get("score")
    if (
        mr.get("metric_id") in judge_ids
        and isinstance(score, (int, float))
        and UNCERTAIN_LO <= score <= UNCERTAIN_HI
    ):
        return float(score)
    return None


@router.get("/queue")
async def get_queue() -> JSONResponse:
    """Return up to 50 uncertain items, closest to 0.5 first."""
    runs = load_all_runs()
    candidates: list[tuple[float, dict]] = []
    items_cache: dict[Path, dict[str, dict]] = {}
    for run in runs:
        judge_ids = _judge_metric_ids(run)
        dataset_dir = run_dataset_dir(run)
        items_by_id = items_cache.setdefault(dataset_dir, load_dataset_items(dataset_dir))
        rid = run_id(run)
        for mr in run.get("metric_results") or []:
            conf = _confidence(mr, judge_ids)
            if conf is None:
                continue
            iid = mr.get("item_id", "")
            item = items_by_id.get(iid, {})
            user_text = input_text(item)
            details = mr.get("details") if isinstance(mr.get("details"), dict) else {}
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
            row = {
                "item_id": iid,
                "category": item.get("category") or "uncategorized",
                "judge_confidence": round(conf, 3),
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
            candidates.append((abs(conf - 0.5), row))

    candidates.sort(key=lambda t: t[0])
    items = [row for _, row in candidates[:QUEUE_CAP]]

    rationale = (
        f"These {len(items)} items had judge confidence between "
        f"{UNCERTAIN_LO} and {UNCERTAIN_HI}, indicating low certainty about pass/fail."
    )
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
