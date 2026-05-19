"""``/api/v2/annotation`` router - human annotation sessions.

Endpoints:
- ``POST   /sessions``                  -> create session (validates source + metric_ids)
- ``GET    /sessions``                  -> list sessions (in-progress first)
- ``GET    /sessions/{id}``             -> session metadata + progress
- ``GET    /sessions/{id}/items``       -> paginated batch with pre-labels
- ``POST   /sessions/{id}/verdict``     -> append verdict (idempotent on item_id)
- ``POST   /sessions/{id}/finalize``    -> merge to canonical annotations.jsonl
- ``DELETE /sessions/{id}``             -> abandon session
- ``GET    /smart-queue``               -> ranked items by information gain

Storage layout (under each dataset dir)::

    .evalyn/data/datasets/<ds>/
      reviews/<run>.jsonl              # existing per-run quick verdicts
      annotations.jsonl                # canonical merged set
      annotation_sessions/
        <session_id>.json              # session metadata + progress
        <session_id>.jsonl             # per-verdict event log (resumable)

The session jsonl is the source of truth during a session; ``session.json`` is
a derived progress snapshot updated after each verdict. On resume we replay
the jsonl to recompute progress (handles crashes mid-write).

Source-of-truth shapes live in ``dashboard/frontend/src/v2/api/types.ts``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Iterable

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from ._shared import (
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

# Filename safety: session ids and dataset names must match this regex
# before we touch the filesystem. Prevents path traversal via crafted ids.
_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9_.-]+$")

# Source kinds the picker can submit. Keep in lockstep with frontend types.
_SOURCE_KINDS = ("run", "dataset", "cluster", "custom")
_VALID_LABELS = ("pass", "fail", "skip")
_TERMINAL_STATUSES = {"completed", "abandoned"}

# Per-pane character cap on the preview fields. The user explicitly
# wanted to see the WHOLE output for verdict-making. 200000 (200KB)
# covers any realistic LLM output (research-agent style with dozens
# of citations runs ~30-80KB at the high end). At limit=200 worst-
# case payload is ~120MB but in practice items are nowhere near the
# cap; typical pages are <5MB. Increase further if real outputs
# routinely exceed this.
_PREVIEW_CHAR_CAP = 200000

# Bounds on per-item evidence (text snippets the annotator highlighted
# from the output as justification for their verdict). These caps exist
# to keep the per-item record bounded; they are well above any realistic
# annotation use - if a user hits them we'd want to know why.
_MAX_EVIDENCE_PER_ITEM = 50
_MAX_EVIDENCE_SNIPPET_CHARS = 2000
_MAX_EVIDENCE_NOTE_CHARS = 1000


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _safe_token(value: str) -> bool:
    """Return True if ``value`` is safe to use as a path segment."""
    return bool(value) and bool(_SAFE_TOKEN.match(value)) and ".." not in value


def _dataset_dir_for_name(name: str) -> Path | None:
    """Locate ``<dataset_root>/<name>`` across all roots, prod-first.

    Returns ``None`` if no matching directory exists. Path traversal in
    ``name`` is rejected up front.
    """
    if not _safe_token(name):
        return None
    for root in dataset_roots():
        candidate = root / name
        if candidate.is_dir():
            return candidate
    return None


def _sessions_dir(dataset_dir: Path) -> Path:
    return dataset_dir / "annotation_sessions"


def _session_meta_path(dataset_dir: Path, session_id: str) -> Path:
    return _sessions_dir(dataset_dir) / f"{session_id}.json"


def _session_log_path(dataset_dir: Path, session_id: str) -> Path:
    return _sessions_dir(dataset_dir) / f"{session_id}.jsonl"


def _annotations_canonical_path(dataset_dir: Path) -> Path:
    return dataset_dir / "annotations.jsonl"


def _reviews_path(dataset_dir: Path, run_id_value: str) -> Path:
    return dataset_dir / "reviews" / f"{run_id_value}.jsonl"


# ---------------------------------------------------------------------------
# Session id generation
# ---------------------------------------------------------------------------


def _new_session_id() -> str:
    """``ann-YYYYMMDD-HHMMSS_<6-hex>`` - sortable + unique within a session.

    Uses ``secrets.token_hex(3)`` (6 hex chars = 24 bits) for the suffix
    so two sessions started in the same second don't collide.
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    suffix = secrets.token_hex(3)
    return f"ann-{ts}_{suffix}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Source resolution: pick which item ids belong to a source
# ---------------------------------------------------------------------------


def _items_from_run(run_id_query: str) -> tuple[Path, list[str], list[str]] | None:
    """Locate a run by id and return ``(dataset_dir, item_ids, metric_ids)``.

    ``item_ids`` is the deduped list of items the run scored; ``metric_ids``
    is the deduped list of metrics seen across the run's metric_results.
    Order is preserved from first appearance for both.
    """
    for run in load_all_runs():
        if run_id(run) != run_id_query and run.get("id") != run_id_query:
            continue
        dataset_dir = run_dataset_dir(run)
        items: list[str] = []
        metrics: list[str] = []
        seen_items: set[str] = set()
        seen_metrics: set[str] = set()
        for mr in run.get("metric_results") or []:
            iid = mr.get("item_id")
            mid = mr.get("metric_id")
            if iid and iid not in seen_items:
                seen_items.add(iid)
                items.append(iid)
            if mid and mid not in seen_metrics:
                seen_metrics.add(mid)
                metrics.append(mid)
        return dataset_dir, items, metrics
    return None


def _items_from_dataset(dataset_name: str) -> tuple[Path, list[str], list[str]] | None:
    """Resolve a dataset by name -> all its items + observed metrics."""
    dataset_dir = _dataset_dir_for_name(dataset_name)
    if dataset_dir is None:
        return None
    items_by_id = load_dataset_items(dataset_dir)
    item_ids = list(items_by_id.keys())
    # Metrics observed across all runs of this dataset.
    metrics: list[str] = []
    seen: set[str] = set()
    for run in load_all_runs():
        if run.get("_dataset") != dataset_name:
            continue
        for mr in run.get("metric_results") or []:
            mid = mr.get("metric_id")
            if mid and mid not in seen:
                seen.add(mid)
                metrics.append(mid)
    return dataset_dir, item_ids, metrics


def _resolve_source(
    source_kind: str, source_id: str
) -> tuple[Path, list[str], list[str]] | None:
    """Dispatch source resolution. Returns (dataset_dir, item_ids, metric_ids)."""
    if source_kind == "run":
        return _items_from_run(source_id)
    if source_kind == "dataset":
        return _items_from_dataset(source_id)
    # cluster + custom not yet implemented; UI hides them in v1.
    return None


# ---------------------------------------------------------------------------
# Session IO
# ---------------------------------------------------------------------------


def _read_meta(meta_path: Path) -> dict | None:
    """Load session.json or return None on parse/IO failure (logged)."""
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("session meta read failed at %s: %s", meta_path, exc)
        return None


def _write_meta(meta_path: Path, meta: dict) -> None:
    """Atomic-ish meta write: write tmp + rename. Raises OSError on failure."""
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = meta_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    os.replace(tmp, meta_path)


def _replay_log(log_path: Path) -> dict[str, dict]:
    """Replay session.jsonl -> ``{item_id: latest_verdict_record}``.

    The session log is append-only and idempotent on item_id (latest
    wins). Lines that fail to parse are logged + skipped.
    """
    out: dict[str, dict] = {}
    if not log_path.exists():
        return out
    try:
        with log_path.open("r", encoding="utf-8") as f:
            for line_num, raw in enumerate(f, start=1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "session log line %d at %s: %s", line_num, log_path, exc
                    )
                    continue
                iid = rec.get("item_id")
                if iid:
                    out[iid] = rec
    except OSError as exc:
        logger.warning("session log read failed at %s: %s", log_path, exc)
    return out


def _append_log(log_path: Path, record: dict) -> None:
    """Append one JSON record to session.jsonl (raises OSError on failure)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _recompute_progress_from_log(
    log_path: Path,
) -> tuple[int, int, list[str], list[str]]:
    """Full replay of the session log. Source of truth for cold-start.

    Returns ``(items_done, items_skipped, items_done_ids, items_skipped_ids)``.
    Used to migrate pre-index sessions on first verdict POST and as a
    sanity-recovery path if the meta index ever falls out of sync with the
    log.
    """
    log_state = _replay_log(log_path)
    done_ids = sorted(iid for iid, r in log_state.items() if r.get("labels"))
    skipped_ids = sorted(
        iid
        for iid, r in log_state.items()
        if not r.get("labels") and r.get("skipped_metrics")
    )
    return len(done_ids), len(skipped_ids), done_ids, skipped_ids


def _meta_has_progress_index(meta: dict) -> bool:
    """True iff meta carries the incremental progress index.

    Sessions created before the incremental-tracking change (2026-05-12)
    lack ``items_done_ids`` / ``items_skipped_ids`` and need one replay
    to migrate.
    """
    return "items_done_ids" in meta and "items_skipped_ids" in meta


def _annotator_id_from_request(request: Request) -> str:
    """Best-effort annotator label. Today: hostname, tomorrow: auth user.

    Falls back to ``"local"`` so the field is never empty. Frontend can
    override by sending an explicit ``annotator_id`` in the create body.
    """
    return os.environ.get("USER") or os.environ.get("USERNAME") or "local"


# ---------------------------------------------------------------------------
# Pre-labeling: pull AI verdicts from the source run's metric_results
# ---------------------------------------------------------------------------


def _ai_labels_for_run(run_id_query: str) -> dict[tuple[str, str], dict]:
    """Return ``{(item_id, metric_id): {label, score}}`` from the run.

    Used to pre-fill the human's verdict with the AI's verdict.
    """
    for run in load_all_runs():
        if run_id(run) != run_id_query and run.get("id") != run_id_query:
            continue
        out: dict[tuple[str, str], dict] = {}
        for mr in run.get("metric_results") or []:
            iid = mr.get("item_id")
            mid = mr.get("metric_id")
            if not iid or not mid:
                continue
            passed = mr.get("passed")
            label = "pass" if passed is True else "fail" if passed is False else None
            score = mr.get("score")
            out[(iid, mid)] = {
                "label": label,
                "score": float(score) if isinstance(score, (int, float)) else None,
            }
        return out
    return {}


# ---------------------------------------------------------------------------
# Cache invalidation broadcast (best-effort)
# ---------------------------------------------------------------------------


async def _broadcast_invalidate(keys: Iterable[str]) -> None:
    """Tell connected v2 clients to refetch the given resource keys."""
    try:
        from .v2_ws import broadcast

        await broadcast({"type": "cache_invalidate", "keys": list(keys)})
    except Exception as exc:  # noqa: BLE001 - broadcast is best-effort
        logger.warning("annotation broadcast failed: %s", exc)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


async def _json_body(request: Request) -> dict:
    try:
        body = await request.json()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"invalid json: {exc}") from exc
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="body must be a json object")
    return body


@router.post("/sessions")
async def create_session(request: Request) -> JSONResponse:
    """Create a new annotation session.

    Body shape::

        {
          "source_kind": "run" | "dataset",
          "source_id": "<run_id or dataset_name>",
          "metric_ids": ["helpfulness", ...],   # optional; defaults to all observed
          "annotator_id": "alice"               # optional
        }
    """
    body = await _json_body(request)

    source_kind = body.get("source_kind")
    source_id = body.get("source_id")
    if source_kind not in _SOURCE_KINDS:
        raise HTTPException(
            status_code=422,
            detail=f"source_kind must be one of {list(_SOURCE_KINDS)}",
        )
    if not isinstance(source_id, str) or not source_id:
        raise HTTPException(status_code=422, detail="source_id must be a non-empty string")

    resolved = _resolve_source(source_kind, source_id)
    if resolved is None:
        raise HTTPException(
            status_code=404,
            detail=f"source {source_kind}:{source_id} not found",
        )
    dataset_dir, item_ids, observed_metrics = resolved

    requested_metric_ids = body.get("metric_ids")
    if requested_metric_ids is None:
        metric_ids = list(observed_metrics)
    else:
        if not isinstance(requested_metric_ids, list) or not all(
            isinstance(m, str) and m for m in requested_metric_ids
        ):
            raise HTTPException(
                status_code=422, detail="metric_ids must be a list of strings"
            )
        unknown = [m for m in requested_metric_ids if m not in observed_metrics]
        if unknown and observed_metrics:
            # If the source has known metrics, reject unknowns. If the
            # source has none yet (fresh dataset, no runs), accept any.
            raise HTTPException(
                status_code=422,
                detail=f"unknown metric_ids: {unknown}; observed: {observed_metrics}",
            )
        metric_ids = list(requested_metric_ids)

    if not item_ids:
        raise HTTPException(
            status_code=422, detail=f"source {source_kind}:{source_id} has no items"
        )

    annotator_id = body.get("annotator_id")
    if not isinstance(annotator_id, str) or not annotator_id:
        annotator_id = _annotator_id_from_request(request)

    session_id = _new_session_id()
    meta = {
        "id": session_id,
        "annotator_id": annotator_id,
        "source_kind": source_kind,
        "source_id": source_id,
        "metric_ids": metric_ids,
        "item_ids": item_ids,
        "items_total": len(item_ids),
        "items_done": 0,
        "items_skipped": 0,
        # Incremental progress index. New sessions ship empty so the
        # first verdict POST takes the fast path; legacy sessions
        # missing these fields get migrated via one replay on first POST.
        "items_done_ids": [],
        "items_skipped_ids": [],
        "started_at_iso": _now_iso(),
        "last_active_iso": _now_iso(),
        "status": "in_progress",
    }
    try:
        _write_meta(_session_meta_path(dataset_dir, session_id), meta)
    except OSError as exc:
        logger.warning("session create write failed: %s", exc)
        raise HTTPException(status_code=503, detail=f"could not create session: {exc}") from exc

    await _broadcast_invalidate(["annotation/sessions"])
    # Audit log: session creation is the entry point for annotation
    # work. Operators tracking "who annotated which dataset and
    # when" need annotator + source on record. items_total surfaces
    # silent zero-item edge cases (we 422 above but logging the
    # non-zero count makes the trail self-explanatory).
    logger.info(
        "annotation session created: id=%s annotator=%s "
        "source_kind=%s source_id=%s items_total=%d",
        session_id,
        annotator_id,
        source_kind,
        source_id,
        len(item_ids),
    )
    return JSONResponse({**meta, "_dataset": dataset_dir.name})


def _find_session(session_id: str) -> tuple[Path, dict] | None:
    """Locate a session across all dataset roots. Returns (dataset_dir, meta)."""
    if not _safe_token(session_id):
        return None
    for root in dataset_roots():
        for ds_dir in root.iterdir():
            if not ds_dir.is_dir():
                continue
            meta_path = _session_meta_path(ds_dir, session_id)
            if meta_path.exists():
                meta = _read_meta(meta_path)
                if meta is not None:
                    return ds_dir, meta
    return None


def _list_sessions() -> list[tuple[Path, dict]]:
    """All sessions on disk across all roots, newest-first."""
    out: list[tuple[Path, dict]] = []
    for root in dataset_roots():
        if not root.is_dir():
            continue
        for ds_dir in root.iterdir():
            sessions_dir = _sessions_dir(ds_dir)
            if not sessions_dir.is_dir():
                continue
            for meta_path in sessions_dir.glob("*.json"):
                meta = _read_meta(meta_path)
                if meta is not None:
                    out.append((ds_dir, meta))
    out.sort(
        key=lambda pair: pair[1].get("last_active_iso") or pair[1].get("started_at_iso") or "",
        reverse=True,
    )
    return out


@router.get("/sessions")
async def list_sessions() -> JSONResponse:
    """List all annotation sessions across all dataset roots, newest-first."""
    items = []
    for ds_dir, meta in _list_sessions():
        items.append({**meta, "_dataset": ds_dir.name})
    return JSONResponse({"sessions": items})


@router.get("/sessions/{session_id}")
async def get_session(session_id: str) -> JSONResponse:
    """Return one session's metadata."""
    found = _find_session(session_id)
    if found is None:
        raise HTTPException(status_code=404, detail=f"session {session_id} not found")
    ds_dir, meta = found
    return JSONResponse({**meta, "_dataset": ds_dir.name})


@router.get("/sessions/{session_id}/items")
async def get_session_items(
    session_id: str, offset: int = 0, limit: int = 50
) -> JSONResponse:
    """Return a paginated batch of items with pre-labels.

    Each item carries its current verdict (if already annotated in this
    session) plus AI pre-labels per metric (when the source is a run with
    metric_results).
    """
    found = _find_session(session_id)
    if found is None:
        raise HTTPException(status_code=404, detail=f"session {session_id} not found")
    ds_dir, meta = found

    if offset < 0 or limit < 1 or limit > 200:
        raise HTTPException(
            status_code=422, detail="offset >= 0 and 1 <= limit <= 200 required"
        )

    item_ids: list[str] = list(meta.get("item_ids") or [])
    total = len(item_ids)
    page_ids = item_ids[offset : offset + limit]

    items_by_id = load_dataset_items(ds_dir)
    log_state = _replay_log(_session_log_path(ds_dir, session_id))
    metric_ids: list[str] = list(meta.get("metric_ids") or [])

    # Pre-labels are only meaningful when source is a run (we have AI verdicts).
    ai_labels: dict[tuple[str, str], dict] = {}
    if meta.get("source_kind") == "run":
        ai_labels = _ai_labels_for_run(meta["source_id"])

    rows: list[dict] = []
    for iid in page_ids:
        item = items_by_id.get(iid) or {}
        record = log_state.get(iid)
        rows.append(
            {
                "item_id": iid,
                # Annotators need to read the actual content to make a
                # verdict - 240 chars was a teaser, not a usable preview.
                # 8000 chars covers ~99% of LLM outputs (around 2k tokens)
                # while keeping the page payload bounded (~2MB at limit=200).
                # For the rare ultra-long output, a future /full endpoint
                # can stream the rest on demand.
                "input_preview": input_text(item)[:_PREVIEW_CHAR_CAP],
                "expected_preview": (item.get("expected") or "")[:_PREVIEW_CHAR_CAP]
                if isinstance(item.get("expected"), str)
                else None,
                "output_preview": (item.get("output") or "")[:_PREVIEW_CHAR_CAP]
                if isinstance(item.get("output"), str)
                else None,
                # Per-metric pre-labels for the human to confirm/override.
                "ai_labels": [
                    {
                        "metric_id": mid,
                        "label": ai_labels.get((iid, mid), {}).get("label"),
                        "score": ai_labels.get((iid, mid), {}).get("score"),
                    }
                    for mid in metric_ids
                ],
                # The user's current verdict for this item (if any).
                "user_labels": (record or {}).get("labels") or [],
                "annotated": record is not None,
                "skipped_metrics": (record or {}).get("skipped_metrics") or [],
                "note": (record or {}).get("note"),
                # Saved evidence snippets - returned so the UI can rehydrate
                # the highlight list when the user revisits an item.
                "evidence": (record or {}).get("evidence") or [],
            }
        )
    return JSONResponse(
        {
            "session_id": session_id,
            "total": total,
            "offset": offset,
            "limit": limit,
            "metric_ids": metric_ids,
            "items": rows,
        }
    )


def _validate_evidence_record(rec: dict, allowed_metrics: set[str]) -> str | None:
    """Return None if rec is a valid evidence entry, else a reason string.

    Shape: ``{"snippet": str, "metric_id"?: str, "note"?: str}``. A snippet
    is mandatory; metric_id is optional (item-level evidence is allowed
    when the user highlights something not tied to any specific metric).
    """
    if not isinstance(rec, dict):
        return "evidence entry must be an object"
    snippet = rec.get("snippet")
    if not isinstance(snippet, str) or not snippet.strip():
        return "evidence.snippet must be a non-empty string"
    if len(snippet) > _MAX_EVIDENCE_SNIPPET_CHARS:
        return f"evidence.snippet must be <= {_MAX_EVIDENCE_SNIPPET_CHARS} chars"
    mid = rec.get("metric_id")
    if mid is not None:
        if not isinstance(mid, str) or mid not in allowed_metrics:
            return f"evidence.metric_id must be null or one of {sorted(allowed_metrics)}"
    note = rec.get("note")
    if note is not None:
        if not isinstance(note, str):
            return "evidence.note must be a string or null"
        if len(note) > _MAX_EVIDENCE_NOTE_CHARS:
            return f"evidence.note must be <= {_MAX_EVIDENCE_NOTE_CHARS} chars"
    return None


def _validate_label_record(rec: dict, allowed_metrics: set[str]) -> str | None:
    """Return None if rec is a valid label, else a user-facing reason string."""
    if not isinstance(rec, dict):
        return "label entry must be an object"
    mid = rec.get("metric_id")
    if not isinstance(mid, str) or mid not in allowed_metrics:
        return f"metric_id must be one of {sorted(allowed_metrics)}"
    label = rec.get("label")
    if label not in _VALID_LABELS:
        return f"label must be one of {list(_VALID_LABELS)}"
    used_ai = rec.get("used_ai_verdict", False)
    if not isinstance(used_ai, bool):
        return "used_ai_verdict must be a boolean"
    return None


@router.post("/sessions/{session_id}/verdict")
async def post_verdict(session_id: str, request: Request) -> JSONResponse:
    """Append a verdict to the session log + per-run reviews jsonl.

    Body shape::

        {
          "item_id": "abc-123",
          "labels": [
            {"metric_id": "helpfulness", "label": "pass", "used_ai_verdict": true,
             "confidence": 1.0, "note": "..."},
            ...
          ],
          "skipped_metrics": ["tone"],   # optional
          "note": "free-text item-level note"
        }
    """
    found = _find_session(session_id)
    if found is None:
        raise HTTPException(status_code=404, detail=f"session {session_id} not found")
    ds_dir, meta = found

    if meta.get("status") in _TERMINAL_STATUSES:
        raise HTTPException(
            status_code=409,
            detail=f"session {session_id} is {meta['status']}; start a new one",
        )

    body = await _json_body(request)
    item_id = body.get("item_id")
    labels = body.get("labels")
    if not isinstance(item_id, str) or not item_id:
        raise HTTPException(status_code=422, detail="item_id must be a non-empty string")
    if item_id not in (meta.get("item_ids") or []):
        raise HTTPException(
            status_code=422, detail=f"item_id {item_id!r} not in session"
        )
    if not isinstance(labels, list):
        raise HTTPException(status_code=422, detail="labels must be a list")

    metric_ids: list[str] = list(meta.get("metric_ids") or [])
    allowed_metrics = set(metric_ids)
    for entry in labels:
        err = _validate_label_record(entry, allowed_metrics)
        if err is not None:
            raise HTTPException(status_code=422, detail=err)

    skipped_metrics = body.get("skipped_metrics") or []
    if not isinstance(skipped_metrics, list) or not all(
        isinstance(m, str) for m in skipped_metrics
    ):
        raise HTTPException(
            status_code=422, detail="skipped_metrics must be a list of strings"
        )
    note = body.get("note")
    if note is not None and not isinstance(note, str):
        raise HTTPException(status_code=422, detail="note must be a string")

    evidence = body.get("evidence") or []
    if not isinstance(evidence, list):
        raise HTTPException(status_code=422, detail="evidence must be a list")
    if len(evidence) > _MAX_EVIDENCE_PER_ITEM:
        raise HTTPException(
            status_code=422,
            detail=f"evidence must be <= {_MAX_EVIDENCE_PER_ITEM} entries",
        )
    for entry in evidence:
        err = _validate_evidence_record(entry, allowed_metrics)
        if err is not None:
            raise HTTPException(status_code=422, detail=err)

    record = {
        "item_id": item_id,
        "labels": labels,
        "skipped_metrics": skipped_metrics,
        "note": note,
        "evidence": evidence,
        "annotator_id": meta.get("annotator_id"),
        "session_id": session_id,
        "ts_iso": _now_iso(),
    }

    log_path = _session_log_path(ds_dir, session_id)
    try:
        _append_log(log_path, record)
    except OSError as exc:
        logger.warning("session verdict append failed at %s: %s", log_path, exc)
        raise HTTPException(status_code=503, detail=f"could not save verdict: {exc}") from exc

    # Per-run reviews jsonl (existing surface). Only when source is a run.
    if meta.get("source_kind") == "run":
        try:
            run_id_value = meta["source_id"]
            if _safe_token(run_id_value):
                _append_log(_reviews_path(ds_dir, run_id_value), record)
        except OSError as exc:
            # Log but don't fail the verdict - session log is the source of truth.
            logger.warning("reviews jsonl append failed: %s", exc)

    # Progress tracking. Fast path: incremental update from the meta's id
    # index, O(log N) per verdict. Cold-start path: replay the log once
    # to migrate sessions written before the index existed.
    #
    # Before 2026-05-12, every verdict POST replayed the full log to
    # recompute progress - quadratic in session size and a wall-clock
    # tax on the hot path. The index makes the hot path constant-time
    # while preserving idempotency (drop-then-add per item_id).
    if _meta_has_progress_index(meta):
        done_ids: set[str] = set(meta.get("items_done_ids") or [])
        skipped_ids: set[str] = set(meta.get("items_skipped_ids") or [])
        # Idempotency: a re-post for the same item_id drops the prior
        # classification, then re-adds based on the fresh record. Items
        # with neither labels nor skipped_metrics fall out of both sets.
        done_ids.discard(item_id)
        skipped_ids.discard(item_id)
        if labels:
            done_ids.add(item_id)
        elif skipped_metrics:
            skipped_ids.add(item_id)
        items_done = len(done_ids)
        items_skipped = len(skipped_ids)
    else:
        items_done, items_skipped, done_ids_list, skipped_ids_list = (
            _recompute_progress_from_log(log_path)
        )
        done_ids = set(done_ids_list)
        skipped_ids = set(skipped_ids_list)

    meta["items_done"] = items_done
    meta["items_skipped"] = items_skipped
    meta["items_done_ids"] = sorted(done_ids)
    meta["items_skipped_ids"] = sorted(skipped_ids)
    meta["last_active_iso"] = _now_iso()
    if items_done + items_skipped >= meta["items_total"]:
        meta["status"] = "in_progress"  # finalize is explicit, not auto
    try:
        _write_meta(_session_meta_path(ds_dir, session_id), meta)
    except OSError as exc:
        logger.warning("session meta write failed: %s", exc)
        # Verdict is already appended; meta will self-heal on next call
        # via the cold-start path (index is missing -> replay).

    await _broadcast_invalidate(
        ["annotation/sessions", "review/queue", "home"]
    )

    return JSONResponse(
        {
            "ok": True,
            "items_done": items_done,
            "items_skipped": items_skipped,
            "items_total": meta["items_total"],
        }
    )


@router.post("/sessions/{session_id}/finalize")
async def finalize_session(session_id: str) -> JSONResponse:
    """Merge the session's verdicts into the dataset's canonical annotations.jsonl.

    Idempotent: a record already in the canonical set (matched by
    ``(annotator_id, item_id, session_id)``) is not duplicated. Sets the
    session status to ``completed``.
    """
    found = _find_session(session_id)
    if found is None:
        raise HTTPException(status_code=404, detail=f"session {session_id} not found")
    ds_dir, meta = found

    log_path = _session_log_path(ds_dir, session_id)
    log_state = _replay_log(log_path)
    if not log_state:
        # Nothing to merge; still mark complete so the session settles.
        meta["status"] = "completed"
        meta["last_active_iso"] = _now_iso()
        try:
            _write_meta(_session_meta_path(ds_dir, session_id), meta)
        except OSError as exc:
            raise HTTPException(status_code=503, detail=f"finalize write failed: {exc}") from exc
        await _broadcast_invalidate(["annotation/sessions", "home"])
        return JSONResponse({"ok": True, "merged": 0})

    # Read existing canonical to dedup.
    canonical_path = _annotations_canonical_path(ds_dir)
    seen: set[tuple[str, str, str]] = set()
    if canonical_path.exists():
        try:
            with canonical_path.open("r", encoding="utf-8") as f:
                for line_num, raw in enumerate(f, start=1):
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError as exc:
                        logger.warning(
                            "annotations.jsonl line %d at %s: %s",
                            line_num,
                            canonical_path,
                            exc,
                        )
                        continue
                    key = (
                        rec.get("annotator_id") or "",
                        rec.get("item_id") or "",
                        rec.get("session_id") or "",
                    )
                    seen.add(key)
        except OSError as exc:
            logger.warning("annotations canonical read failed: %s", exc)

    merged = 0
    try:
        canonical_path.parent.mkdir(parents=True, exist_ok=True)
        with canonical_path.open("a", encoding="utf-8") as f:
            for record in log_state.values():
                key = (
                    record.get("annotator_id") or "",
                    record.get("item_id") or "",
                    record.get("session_id") or "",
                )
                if key in seen:
                    continue
                f.write(json.dumps(record) + "\n")
                seen.add(key)
                merged += 1
    except OSError as exc:
        logger.warning("annotations canonical write failed: %s", exc)
        raise HTTPException(status_code=503, detail=f"finalize write failed: {exc}") from exc

    meta["status"] = "completed"
    meta["last_active_iso"] = _now_iso()
    try:
        _write_meta(_session_meta_path(ds_dir, session_id), meta)
    except OSError as exc:
        logger.warning("session meta write on finalize failed: %s", exc)

    await _broadcast_invalidate(["annotation/sessions", "review/queue", "home"])

    # Audit log: finalize commits annotation results into the canonical
    # store. items_done + items_skipped show how complete the session
    # was; merged tells operators how many records were actually
    # appended (non-duplicate). The merged < items_done case is
    # interesting in its own right - means re-finalize with overlap.
    logger.info(
        "annotation session finalized: id=%s annotator=%s "
        "items_done=%d items_skipped=%d merged=%d",
        session_id,
        meta.get("annotator_id", ""),
        int(meta.get("items_done", 0)),
        int(meta.get("items_skipped", 0)),
        merged,
    )

    return JSONResponse({"ok": True, "merged": merged})


@router.delete("/sessions/{session_id}")
async def abandon_session(session_id: str) -> JSONResponse:
    """Mark a session as abandoned. The jsonl log is preserved on disk."""
    found = _find_session(session_id)
    if found is None:
        raise HTTPException(status_code=404, detail=f"session {session_id} not found")
    ds_dir, meta = found
    if meta.get("status") in _TERMINAL_STATUSES:
        return JSONResponse({"ok": True, "status": meta["status"]})
    meta["status"] = "abandoned"
    meta["last_active_iso"] = _now_iso()
    try:
        _write_meta(_session_meta_path(ds_dir, session_id), meta)
    except OSError as exc:
        logger.warning("session abandon write failed: %s", exc)
        raise HTTPException(status_code=503, detail=f"abandon failed: {exc}") from exc
    await _broadcast_invalidate(["annotation/sessions"])
    # Audit log: abandoned sessions leave their jsonl logs on disk
    # but the verdicts never reach the canonical store. items_done
    # surfaces the lost work for postmortem ("did we abandon at 5%
    # complete or 95%?"). The session_id can be cross-referenced
    # against the create-log to find the annotator + source.
    logger.info(
        "annotation session abandoned: id=%s items_done=%d items_skipped=%d",
        session_id,
        int(meta.get("items_done", 0)),
        int(meta.get("items_skipped", 0)),
    )
    return JSONResponse({"ok": True, "status": "abandoned"})


# ---------------------------------------------------------------------------
# Smart queue: rank items by information-gain proxy
# ---------------------------------------------------------------------------


_SMART_WEIGHTS = {
    "uncertainty": 0.5,
    "disagreement": 0.3,
    "coverage": 0.15,
    "random": 0.05,
}


def _why_label(uncertainty: float, disagreement: float, coverage: float) -> str:
    """Pick the dominant explanation for a smart-queue item."""
    if disagreement >= 0.4:
        return "judge ↔ judge disagreement"
    if uncertainty >= 0.7:
        return "high uncertainty"
    if 0.3 < uncertainty < 0.7:
        return "borderline"
    if coverage >= 0.5:
        return "fills coverage gap"
    return "novel cluster"


@router.get("/smart-queue")
async def smart_queue(
    metric_id: str | None = None, limit: int = 20
) -> JSONResponse:
    """Rank items for human annotation by an information-gain proxy.

    Score formula: ``0.5*uncertainty + 0.3*disagreement + 0.15*coverage +
    0.05*random``. Uncertainty is distance from 0.5 inverted (closer to
    0.5 -> higher score). Disagreement is the stddev of judge scores
    when multi-judge data is available; otherwise 0. Coverage favours
    items whose category has thin past-annotation history. Random adds
    a small jitter so identical scores tie-break stably.
    """
    import random as _random

    rng = _random.Random(42)
    if limit < 1 or limit > 200:
        raise HTTPException(422, "limit must be 1..200")

    runs = load_all_runs()
    if not runs:
        return JSONResponse(
            {
                "metric_id": metric_id or "",
                "items": [],
                "est_to_threshold": 0,
                "weights": _SMART_WEIGHTS,
            }
        )

    # Past annotation coverage: count verdicts per metric across all reviews.
    from collections import defaultdict

    coverage_counter: dict[str, int] = defaultdict(int)
    for root in dataset_roots():
        for ds_dir in list_dataset_dirs(root):
            reviews_dir = ds_dir / "reviews"
            if not reviews_dir.is_dir():
                continue
            for path in reviews_dir.glob("*.jsonl"):
                try:
                    with path.open(encoding="utf-8") as f:
                        for raw in f:
                            line = raw.strip()
                            if not line:
                                continue
                            try:
                                rec = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            for label_entry in rec.get("labels") or []:
                                mid = label_entry.get("metric_id") if isinstance(label_entry, dict) else None
                                if mid:
                                    coverage_counter[mid] += 1
                except OSError:
                    continue
    # Define a "low-representation" threshold relative to median.
    if coverage_counter:
        sorted_counts = sorted(coverage_counter.values())
        mid_pos = len(sorted_counts) // 2
        median_count = sorted_counts[mid_pos]
    else:
        median_count = 0

    candidates: list[dict] = []
    for run in runs:
        rid = run_id(run)
        items_by_id = load_dataset_items(run_dataset_dir(run))
        # Group by item to consolidate per-item judge scores.
        per_item_scores: dict[str, list[float]] = defaultdict(list)
        per_item_target_score: dict[str, float] = {}
        for r in run.get("metric_results") or []:
            iid = r.get("item_id")
            if iid is None:
                continue
            score = r.get("score")
            if not isinstance(score, (int, float)):
                continue
            per_item_scores[iid].append(float(score))
            mid = r.get("metric_id") or ""
            if metric_id is None or mid == metric_id:
                per_item_target_score[iid] = float(score)
        for iid, judge_score in per_item_target_score.items():
            scores = per_item_scores.get(iid) or [judge_score]
            uncertainty = max(0.0, min(1.0, 1.0 - 2 * abs(judge_score - 0.5)))
            if len(scores) >= 2:
                mean_s = sum(scores) / len(scores)
                var = sum((s - mean_s) ** 2 for s in scores) / len(scores)
                stddev = var ** 0.5
                disagreement = max(0.0, min(1.0, stddev * 2))
            else:
                disagreement = 0.0
            coverage_score = (
                1.0 if (coverage_counter.get(metric_id or "", 0) <= median_count) else 0.0
            )
            random_jitter = rng.random()
            combined = (
                _SMART_WEIGHTS["uncertainty"] * uncertainty
                + _SMART_WEIGHTS["disagreement"] * disagreement
                + _SMART_WEIGHTS["coverage"] * coverage_score
                + _SMART_WEIGHTS["random"] * random_jitter
            )
            item = items_by_id.get(iid, {})
            text = input_text(item)
            candidates.append(
                {
                    "score": round(combined, 4),
                    "judge_score": round(judge_score, 4),
                    "text": text[:200],
                    "why": _why_label(uncertainty, disagreement, coverage_score),
                    "item_id": iid,
                    "run_id": rid,
                }
            )

    candidates.sort(key=lambda x: -x["score"])
    top = candidates[:limit]
    for i, row in enumerate(top, start=1):
        row["rank"] = i

    # Estimate annotations to reach kappa >= 0.8 - heuristic: assume each
    # additional annotation buys ~0.005 kappa improvement. Real number
    # would require a per-metric model; we report a reasonable default.
    est = max(0, 50 - len(coverage_counter)) if coverage_counter else 100

    return JSONResponse(
        {
            "metric_id": metric_id or "",
            "items": top,
            "est_to_threshold": est,
            "weights": _SMART_WEIGHTS,
        }
    )
