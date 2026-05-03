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
    _caches_disabled,
    dataset_roots,
    input_text,
    list_dataset_dirs,
    load_all_runs,
    load_dataset_items,
    run_dataset_dir,
    run_id,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Primary uncertainty band - explicit ``judge_confidence``.
PRIMARY_LO = 0.35
PRIMARY_HI = 0.65
# Fallback 1 band - any soft-scored metric in this range.
SOFT_LO = 0.3
SOFT_HI = 0.7
QUEUE_CAP = 50
FALLBACK3_CAP = 20

# Calibration suggestion thresholds.
CALIBRATION_VERDICT_THRESHOLD = 10
CALIBRATION_SUGGESTIONS_CAP = 5

# Per-reviews-file cache. Walking 491 dataset dirs to find ``reviews/``
# subdirs costs ~750ms on WSL because ``glob`` does a ``scandir`` for
# each dir even when empty. Cache the parsed verdict tuples keyed on
# ``(path_str, mtime)`` so warm requests skip the read entirely.
_verdicts_file_cache: dict[tuple[str, float], list[dict]] = {}
# Per-root list of ``(reviews_jsonl_path, dataset_name)`` keyed on the
# root's mtime. Populated by :func:`_reviews_files_for_root`. Saves
# ~1500 ``scandir`` syscalls per warm request when only a handful of
# dataset dirs actually have a ``reviews/`` subdir.
_reviews_dirs_cache: dict[Path, tuple[float, list[tuple[Path, str]]]] = {}


def _clear_review_caches_for_tests() -> None:
    """Drop the per-file/per-root caches. Called from the test seam."""
    _verdicts_file_cache.clear()
    _reviews_dirs_cache.clear()


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


# ---------- Calibration suggestions ----------


def _verdict_metric_id(
    item_id: str,
    run_metric_results_by_item: dict[str, list[dict]],
) -> str | None:
    """Pick the most likely calibration metric_id for ``item_id`` in a run.

    A verdict line records ``(item_id, source_run_id)`` but not which
    metric the reviewer was judging. We attribute it to the same metric
    the cascade would have surfaced: judge_confidence-band first, then
    soft-scored band, then the failing metric, then any metric on the
    item. Returns ``None`` when the item is missing from the run.
    """
    results = run_metric_results_by_item.get(item_id) or []
    if not results:
        return None
    # 1. judge_confidence in the borderline band
    for mr in results:
        details = mr.get("details") if isinstance(mr.get("details"), dict) else {}
        jc = details.get("judge_confidence")
        if isinstance(jc, (int, float)) and PRIMARY_LO <= jc <= PRIMARY_HI:
            mid = mr.get("metric_id")
            if mid:
                return str(mid)
    # 2. soft-scored
    for mr in results:
        score = mr.get("score")
        if _is_soft_score(score) and SOFT_LO <= float(score) <= SOFT_HI:
            mid = mr.get("metric_id")
            if mid:
                return str(mid)
    # 3. first failing
    for mr in results:
        if mr.get("passed") is False:
            mid = mr.get("metric_id")
            if mid:
                return str(mid)
    # 4. anything
    mid = results[0].get("metric_id")
    return str(mid) if mid else None


def _read_verdicts(path: Path, mtime: float | None = None) -> list[dict]:
    """Parse a reviews jsonl file. Bad lines are logged and skipped.

    Cached by ``(path, mtime)`` - reviews files only grow (verdicts are
    append-only) so the mtime check catches every new line.
    """
    if mtime is not None and not _caches_disabled():
        cached = _verdicts_file_cache.get((str(path), mtime))
        if cached is not None:
            return cached
    out: list[dict] = []
    try:
        with path.open(encoding="utf-8") as f:
            for line_num, raw in enumerate(f, start=1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "verdicts jsonl line %d at %s: %s", line_num, path, exc
                    )
    except OSError as exc:
        logger.warning("verdicts jsonl read failed at %s: %s", path, exc)
    if mtime is not None and not _caches_disabled():
        _verdicts_file_cache[(str(path), mtime)] = out
    return out


def _walk_reviews_files(root: Path) -> list[tuple[Path, str]]:
    """Filesystem walk that ``_reviews_files_for_root`` memoizes."""
    out: list[tuple[Path, str]] = []
    for ds_dir in list_dataset_dirs(root):
        reviews_dir = ds_dir / "reviews"
        # ``glob`` returns ``[]`` silently when the dir is missing - one
        # syscall vs the explicit ``is_dir`` + ``iterdir`` pair.
        for path in sorted(reviews_dir.glob("*.jsonl")):
            out.append((path, ds_dir.name))
    return out


def _reviews_files_for_root(root: Path) -> list[tuple[Path, str]]:
    """Return ``[(jsonl_path, dataset_name), ...]`` under ``root``, mtime-cached.

    Walking every dataset dir to ``glob('*.jsonl')`` costs ~750ms on
    WSL/NTFS for 491 dataset folders even when only a handful have a
    ``reviews/`` subdir. The root's mtime bumps when a dataset is
    added/removed, so a single walk per root mtime is enough.
    """
    if not root.is_dir():
        return []
    if _caches_disabled():
        return _walk_reviews_files(root)
    try:
        mtime = root.stat().st_mtime
    except OSError:
        return _walk_reviews_files(root)
    cached = _reviews_dirs_cache.get(root)
    if cached and cached[0] == mtime:
        return cached[1]
    out = _walk_reviews_files(root)
    _reviews_dirs_cache[root] = (mtime, out)
    return out


def _latest_calibration_mtime(dataset_dir: Path, metric_id: str) -> float | None:
    """Return the mtime of the metric's calibration file, or ``None`` if absent."""
    cal = dataset_dir / "calibrations" / metric_id / "calibration.json"
    try:
        return cal.stat().st_mtime
    except OSError:
        return None


def _calibration_suggestions(runs: list[dict]) -> list[dict]:
    """Walk every dataset's reviews/*.jsonl and emit suggestions.

    Aggregates verdicts by (dataset, metric_id). A suggestion fires
    when the count crosses ``CALIBRATION_VERDICT_THRESHOLD`` AND no
    existing calibration file is newer than the most recent verdict
    for that metric. Capped at ``CALIBRATION_SUGGESTIONS_CAP``.
    """
    # Build a per-run lookup (rid -> {item_id -> [metric_results]}) once
    # so we don't re-walk metric_results per verdict.
    run_lookup: dict[str, tuple[Path, dict[str, list[dict]]]] = {}
    for run in runs:
        rid = run_id(run)
        ds_dir = run_dataset_dir(run)
        per_item: dict[str, list[dict]] = defaultdict(list)
        for mr in run.get("metric_results") or []:
            iid = mr.get("item_id")
            if iid:
                per_item[iid].append(mr)
        run_lookup[rid] = (ds_dir, per_item)
        # Some runs also identify themselves by ``id`` (legacy).
        legacy_id = run.get("id")
        if isinstance(legacy_id, str) and legacy_id and legacy_id != rid:
            run_lookup.setdefault(legacy_id, (ds_dir, per_item))

    counts: dict[tuple[Path, str, str], int] = defaultdict(int)
    latest_verdict_mtime: dict[tuple[Path, str], float] = {}

    for root in dataset_roots():
        for path, ds_name in _reviews_files_for_root(root):
            try:
                f_mtime = path.stat().st_mtime
            except OSError:
                f_mtime = 0.0
            ds_dir = path.parent.parent
            for verdict in _read_verdicts(path, mtime=f_mtime):
                rid = verdict.get("source_run_id") or ""
                iid = verdict.get("item_id") or ""
                if not rid or not iid:
                    continue
                lookup = run_lookup.get(rid)
                if lookup is None:
                    continue
                _, per_item = lookup
                metric_id = _verdict_metric_id(iid, per_item)
                if not metric_id:
                    continue
                counts[(ds_dir, ds_name, metric_id)] += 1
                key = (ds_dir, metric_id)
                if f_mtime > latest_verdict_mtime.get(key, 0.0):
                    latest_verdict_mtime[key] = f_mtime

    suggestions: list[dict] = []
    for (ds_dir, ds_name, metric_id), count in counts.items():
        if count < CALIBRATION_VERDICT_THRESHOLD:
            continue
        cal_mtime = _latest_calibration_mtime(ds_dir, metric_id)
        v_mtime = latest_verdict_mtime.get((ds_dir, metric_id), 0.0)
        if cal_mtime is not None and cal_mtime >= v_mtime:
            # Already calibrated since the latest verdict landed.
            continue
        annotations_path = str(ds_dir / "reviews")
        suggestions.append({
            "metric_id": metric_id,
            "dataset": ds_name,
            "verdict_count": count,
            "threshold": CALIBRATION_VERDICT_THRESHOLD,
            "cli_args": {
                "metric_id": metric_id,
                "annotations": annotations_path,
            },
        })
    # Stable ordering: highest count first, then alphabetical tiebreak.
    suggestions.sort(key=lambda s: (-s["verdict_count"], s["dataset"], s["metric_id"]))
    return suggestions[:CALIBRATION_SUGGESTIONS_CAP]


@router.get("/queue")
async def get_queue() -> JSONResponse:
    """Return up to 50 review items chosen by the cascade strategy."""
    runs = load_all_runs()
    items, rationale = _build_queue(runs)
    suggestions = _calibration_suggestions(runs)
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
