"""``/api/v2/datasets/{name}`` sub-router.

Single endpoint:
- ``GET /{name}`` -> ``DatasetDetail``

Mounted by :mod:`datasets` to share the ``/api/v2/datasets`` prefix.
The JSON shape is documented in
``dashboard/frontend/src/v2/api/types.ts`` (``DatasetDetail``).

404s when no ``dataset.jsonl`` exists for ``name`` under any
:func:`dataset_roots` entry. Path traversal attempts (any ``name``
containing a path separator or ``..``) also 404 - the detail walker
intentionally never resolves outside ``dataset_roots()``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from ._shared import (
    dataset_roots,
    fmt_cost,
    input_text,
    load_all_runs,
    load_dataset_items,
    run_id,
    run_pass_rate,
    run_status,
    total_cost,
)
from .datasets import _coverage, _meta
from .rubrics import _metric_kinds_uses

logger = logging.getLogger(__name__)

router = APIRouter()

ITEMS_PREVIEW_CAP = 50
RECENT_RUNS_CAP = 10


def _safe_dataset_dir(name: str) -> Path | None:
    """Return the dataset directory for ``name`` or ``None``.

    Rejects names containing path separators or ``..`` to defeat path
    traversal (e.g. ``../../etc``). Walks every root from
    :func:`dataset_roots` and returns the first hit that has a
    ``dataset.jsonl`` file.
    """
    if not name or "/" in name or "\\" in name or ".." in name:
        return None
    for root in dataset_roots():
        candidate = root / name
        if not candidate.is_dir():
            continue
        # Defense in depth: ensure the resolved path is actually inside root.
        try:
            resolved = candidate.resolve()
        except OSError as exc:
            logger.warning("dataset dir resolve failed at %s: %s", candidate, exc)
            continue
        try:
            resolved.relative_to(root)
        except ValueError:
            continue
        if (resolved / "dataset.jsonl").is_file():
            return resolved
    return None


def _items_preview(dataset_dir: Path) -> tuple[list[dict], int]:
    """Return ``(preview_rows, total_items)`` for the dataset.

    Reuses the cached ``load_dataset_items`` helper so this endpoint
    doesn't bypass the shared cache (no silent IO outside ``_shared``).
    """
    items_by_id = load_dataset_items(dataset_dir)
    total = len(items_by_id)
    preview: list[dict] = []
    for iid, item in items_by_id.items():
        if len(preview) >= ITEMS_PREVIEW_CAP:
            break
        text = input_text(item)
        expected: str | None = None
        exp_val = item.get("expected") or item.get("expected_output")
        if isinstance(exp_val, str):
            expected = exp_val
        elif exp_val is not None:
            try:
                expected = json.dumps(exp_val, ensure_ascii=False)
            except (TypeError, ValueError) as exc:
                logger.warning(
                    "dataset %s item %s expected serialise failed: %s",
                    dataset_dir.name,
                    iid,
                    exc,
                )
                expected = None
        preview.append(
            {
                "id": iid,
                "input_preview": text[:200],
                "expected_preview": (expected[:200] if expected else None),
            }
        )
    return preview, total


def _recent_runs(name: str) -> list[dict]:
    """Return the ``RECENT_RUNS_CAP`` newest runs on this dataset.

    Reuses the cached :func:`load_all_runs` walk. Filtered by
    ``_dataset == name`` and sorted newest-first by ``created_at``.
    """
    runs = [r for r in load_all_runs() if r.get("_dataset") == name]
    runs.sort(key=lambda r: r.get("created_at") or "", reverse=True)
    out: list[dict] = []
    for r in runs[:RECENT_RUNS_CAP]:
        pr = run_pass_rate(r)
        out.append(
            {
                "id": run_id(r),
                "created_at_iso": r.get("created_at") or "",
                "pass": round(pr * 100, 1) if pr is not None else None,
                "status": run_status(r),
                "cost": fmt_cost(total_cost(r)),
            }
        )
    return out


def _observed_metrics(name: str) -> list[dict]:
    """Return deduped metrics seen across this dataset's runs.

    Mirrors :func:`rubrics._metric_kinds_uses` but scoped to this
    dataset's runs only - the kind ('LLM judge' vs 'Programmatic') is
    inferred the same way. ``uses`` is the count of runs the metric
    appeared in.
    """
    runs = [r for r in load_all_runs() if r.get("_dataset") == name]
    if not runs:
        return []
    kind, uses, _ = _metric_kinds_uses(runs)
    out: list[dict] = []
    for mid in sorted(uses):
        out.append(
            {
                "id": mid,
                "kind": kind.get(mid, "Programmatic"),
                "uses": uses[mid],
            }
        )
    return out


@router.get("/{name}")
async def get_dataset_detail(name: str) -> JSONResponse:
    """Return ``DatasetDetail`` for ``name`` or 404.

    The contract matches ``DatasetDetail`` in
    ``dashboard/frontend/src/v2/api/types.ts``.
    """
    dataset_dir = _safe_dataset_dir(name)
    if dataset_dir is None:
        raise HTTPException(404, f"dataset {name} not found")

    jsonl = dataset_dir / "dataset.jsonl"
    item_count, coverage = _coverage(jsonl)
    meta = _meta(dataset_dir)
    items_preview, total_items = _items_preview(dataset_dir)

    # ``meta.json`` may carry ``created_at`` (newer SDK writes); fall back
    # to the dataset directory's own mtime so the UI can render *some*
    # provenance instead of an empty dash.
    created_at_iso = meta.get("created_at")
    if not isinstance(created_at_iso, str) or not created_at_iso:
        created_at_iso = None

    return JSONResponse(
        {
            "name": dataset_dir.name,
            "n": item_count,
            "source": meta.get("source") or "JSONL",
            "tags": meta.get("tags") or [],
            "coverage": coverage,
            "created_at_iso": created_at_iso,
            "items_preview": items_preview,
            "total_items": total_items,
            "recent_runs": _recent_runs(dataset_dir.name),
            "observed_metrics": _observed_metrics(dataset_dir.name),
        }
    )
