"""``/api/promote`` router.

Trace-to-dataset promotion (Braintrust-inspired).

POST ``/api/promote/run-failures`` builds a new dataset under
``.evalyn/data/datasets/<dataset_name>/`` from a subset of rows of an
existing run's ``results.json``.

Request body:

.. code-block:: json

    {
      "run_id": "<id>",
      "row_hashes": ["abc", "def"],
      "dataset_name": "regressions-2026-05-02"   // optional
    }

Behavior:

- Look up the run by ``run_id`` under
  ``.evalyn/data/datasets/<dataset>/eval_runs/<run_id>/results.json`` (matches the
  ``runs.py`` convention).
- For each ``row_hash`` (interpreted as ``item_id`` in
  ``metric_results``), include the row in the new dataset. Rows that
  don't match are silently skipped (per spec: "skip or 404 - pick one
  and document"; we skip and surface ``skipped`` in the response).
- If the source dataset can be located on disk
  (``.evalyn/data/datasets/<source_dataset>/dataset.jsonl``), pull the
  original ``input`` / ``output`` for each ``item_id`` so the promoted
  dataset is real evaluation material, not just metadata.
- If a row's source input is unavailable, fall back to a placeholder
  carrying the metric verdict + reason so the regression is still
  captured.
- If ``dataset_name`` is omitted, default to ``regressions-<UTC stamp>``.
- ``dataset_name`` must not contain path-separator characters or
  ``..`` segments (path-traversal protection). ``""``/whitespace is
  rejected as 400.
- 404 if ``run_id`` is not found.
- 400 if ``row_hashes`` is empty.
- 409 if ``dataset_name`` already exists on disk.
- 200 on success: ``{"dataset_path", "dataset_name", "item_count",
  "skipped"}``.

Concurrency: writes go through a process-local lock so two simultaneous
promotes can't race on the directory creation / jsonl write.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)
from pydantic import BaseModel, Field

router = APIRouter()

# Single process-wide lock guarding dataset directory creation. Cheap
# - the operation is fast - and avoids the TOCTOU race between
# "does it exist?" and "mkdir+write".
_PROMOTE_LOCK = threading.Lock()

# Reject any name with path separators, control chars, or parent refs.
# We allow ASCII letters, digits, dash, underscore, and dot (no
# leading dot to avoid hidden files).
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class PromoteRequest(BaseModel):
    run_id: str
    # Cap row_hashes to a sane upper bound so a malformed (or hostile)
    # request cannot DoS the server by forcing it to allocate a giant
    # list + dedup set. 10k rows is well above any realistic eval run.
    row_hashes: list[str] = Field(..., max_length=10_000)
    dataset_name: str | None = None


def _safe_run_id(run_id: str) -> str:
    """Return ``run_id`` if it's a safe directory leaf; otherwise raise 400.

    Even though `_find_run` only does an `.exists()` probe, an attacker
    who can write a `results.json` somewhere on disk + supply a
    traversing `run_id` (e.g. ``../../home/user/...``) would expose the
    existence of arbitrary files via 200/404 status. Anchor the lookup
    to a single directory leaf.
    """
    if not isinstance(run_id, str) or not run_id:
        raise HTTPException(400, "run_id must be a non-empty string")
    if "/" in run_id or "\\" in run_id or ".." in run_id:
        raise HTTPException(400, "run_id contains illegal characters")
    return run_id


def _evalyn_root() -> Path:
    return (Path.cwd() / ".evalyn").resolve()


def _datasets_root() -> Path:
    return _evalyn_root() / "data" / "datasets"


def _validate_dataset_name(name: str) -> str:
    """Return ``name`` if it's a safe directory leaf; otherwise raise 400.

    Rejects empty/whitespace, path separators, ``..`` segments, leading
    ``.``, and anything outside ``[A-Za-z0-9._-]``.
    """
    cleaned = name.strip() if isinstance(name, str) else ""
    if not cleaned:
        raise HTTPException(400, "dataset_name must not be empty")
    if any(token in cleaned for token in ("/", "\\", "..")):
        raise HTTPException(400, "dataset_name contains illegal characters")
    if not _NAME_RE.match(cleaned):
        raise HTTPException(
            400,
            "dataset_name must match [A-Za-z0-9][A-Za-z0-9._-]{0,127}",
        )
    return cleaned


def _default_dataset_name() -> str:
    """``regressions-YYYYMMDD-HHMM`` in UTC."""
    return "regressions-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")


def _find_run(run_id: str) -> tuple[Path, dict[str, Any]] | None:
    """Walk ``.evalyn/data/datasets/*/eval_runs/<run_id>/results.json`` and load it.

    Returns ``(results_path, parsed_json)`` or ``None`` if not found /
    unparseable. Mirrors the lookup in ``api/runs.py::get_run``.
    """
    root = _datasets_root()
    if not root.exists():
        return None
    try:
        for dataset_dir in root.iterdir():
            if not dataset_dir.is_dir():
                continue
            eval_runs_dir = dataset_dir / "eval_runs"
            if not eval_runs_dir.is_dir():
                continue
            candidate = eval_runs_dir / run_id / "results.json"
            if candidate.exists():
                try:
                    data = json.loads(candidate.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    return None
                return candidate, data
    except OSError:
        return None
    return None


def _load_source_dataset_index(source_name: str) -> dict[str, dict[str, Any]]:
    """Return ``{item_id: row}`` for the source dataset, or empty if absent.

    The dataset is expected at
    ``.evalyn/data/datasets/<source_name>/dataset.jsonl`` (one JSON
    record per line). Records without an ``id`` are skipped silently
    (we have no way to match them anyway).
    """
    path = _datasets_root() / source_name / "dataset.jsonl"
    if not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rid = rec.get("id")
                if isinstance(rid, str):
                    out[rid] = rec
    except OSError:
        return {}
    return out


def _build_promoted_row(
    item_id: str,
    metric_rows: list[dict[str, Any]],
    source_row: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build one promoted dataset entry.

    Each entry carries the original ``input`` (from the source dataset
    when available) and an ``expected`` field. Per spec: ``expected`` =
    the agent's output if known, or a placeholder when the row failed.
    Even when the source row is missing entirely, we still emit a usable
    record with metric metadata so the regression is captured.
    """
    # Pull the strongest reason / score across the metrics for this item.
    failed_metrics = [m for m in metric_rows if m.get("passed") is False]
    primary = failed_metrics[0] if failed_metrics else metric_rows[0]
    reason = (
        (primary.get("details") or {}).get("reason")
        or (primary.get("raw_judge") or {}).get("reason")
        or ""
    )
    passed = bool(primary.get("passed", False))

    if source_row is not None:
        inputs = source_row.get("input") or source_row.get("inputs") or {}
        agent_output = source_row.get("output", "")
    else:
        inputs = {}
        agent_output = ""

    expected = agent_output if passed else (agent_output or "TODO: set expected output")

    return {
        "id": item_id,
        "input": inputs,
        "expected": expected,
        "metadata": {
            "promoted_from_item_id": item_id,
            "promoted_passed": passed,
            "promoted_reason": reason,
            "promoted_metrics": [
                {
                    "metric_id": m.get("metric_id"),
                    "score": m.get("score"),
                    "passed": m.get("passed"),
                }
                for m in metric_rows
            ],
        },
    }


def _group_metrics_by_item(
    metric_results: Iterable[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for m in metric_results:
        if not isinstance(m, dict):
            continue
        item_id = m.get("item_id")
        if not isinstance(item_id, str):
            continue
        grouped.setdefault(item_id, []).append(m)
    return grouped


@router.post("/run-failures")
async def promote_run_failures(req: PromoteRequest) -> JSONResponse:
    """Promote selected rows of a run into a new dataset on disk."""
    if not req.row_hashes:
        raise HTTPException(400, "row_hashes must not be empty")

    safe_run_id = _safe_run_id(req.run_id)
    found = _find_run(safe_run_id)
    if found is None:
        raise HTTPException(404, f"run {safe_run_id} not found")
    _, results = found

    source_dataset_name = (
        results.get("dataset_name") if isinstance(results, dict) else None
    )

    raw_name = req.dataset_name if req.dataset_name is not None else _default_dataset_name()
    dataset_name = _validate_dataset_name(raw_name)

    by_item = _group_metrics_by_item(results.get("metric_results") or [])
    source_index: dict[str, dict[str, Any]] = (
        _load_source_dataset_index(source_dataset_name)
        if isinstance(source_dataset_name, str) and source_dataset_name
        else {}
    )

    promoted: list[dict[str, Any]] = []
    skipped: list[str] = []
    seen: set[str] = set()
    for raw_hash in req.row_hashes:
        if not isinstance(raw_hash, str) or not raw_hash:
            # Malformed entries are skipped per spec.
            continue
        if raw_hash in seen:
            continue
        seen.add(raw_hash)
        rows = by_item.get(raw_hash)
        if not rows:
            skipped.append(raw_hash)
            continue
        promoted.append(
            _build_promoted_row(raw_hash, rows, source_index.get(raw_hash))
        )

    if not promoted:
        raise HTTPException(
            404,
            "no matching rows found for any of the supplied row_hashes",
        )

    target_dir = _datasets_root() / dataset_name
    # Verify the resolved path stays inside the datasets root
    # (defence-in-depth - we already filtered the name).
    try:
        target_dir = target_dir.resolve()
        target_dir.relative_to(_datasets_root().resolve())
    except (ValueError, OSError) as exc:
        raise HTTPException(400, "dataset_name resolves outside datasets root") from exc

    # Hold the lock only across the existence check + mkdir; the disk
    # writes that follow can take many seconds on slow filesystems and
    # blocking unrelated promotes is not worth the simplicity. Once the
    # mkdir succeeds the directory itself serves as the per-target lock
    # (any concurrent caller using the same name will hit FileExistsError).
    with _PROMOTE_LOCK:
        if target_dir.exists():
            raise HTTPException(409, f"dataset {dataset_name!r} already exists")
        try:
            target_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            # Lost a race despite the lock (e.g. concurrent process).
            raise HTTPException(409, f"dataset {dataset_name!r} already exists") from exc
        except OSError as exc:
            raise HTTPException(500, f"failed to create dataset directory: {exc}") from exc

    try:
        jsonl_path = target_dir / "dataset.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as fh:
            for row in promoted:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        meta = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_run_id": req.run_id,
            "parent_dataset": source_dataset_name,
            "item_count": len(promoted),
            "skipped_row_hashes": skipped,
        }
        (target_dir / "meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
    except OSError as exc:
        # Atomic cleanup so a partial write doesn't poison subsequent
        # retries with the same name. ``rmtree(ignore_errors=True)``
        # handles the case where the failing operation is the one that
        # would have made an iterdir-based cleanup impossible.
        shutil.rmtree(target_dir, ignore_errors=True)
        raise HTTPException(500, f"failed to write dataset: {exc}") from exc

    # Audit log: dataset creation is a state-changing infrastructure
    # action (writes a new dataset directory + items.jsonl + meta.json).
    # Operators tracking "where did this regression-checkpoint dataset
    # come from?" need the source run + parent dataset on record.
    # Counts (rows requested, rows promoted, rows skipped) help detect
    # silent skips that would otherwise need a meta.json fetch.
    logger.info(
        "promote dataset created: dataset_name=%s parent=%s "
        "source_run=%s requested=%d promoted=%d skipped=%d",
        dataset_name,
        source_dataset_name,
        run_id,
        len(row_hashes),
        len(promoted),
        len(skipped),
    )

    return JSONResponse(
        {
            "dataset_path": str(target_dir),
            "dataset_name": dataset_name,
            "item_count": len(promoted),
            "skipped": skipped,
        }
    )
