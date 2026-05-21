"""``/api/v2/datasets`` router.

Endpoint:
- ``GET /`` -> ``DatasetList``

One card per directory under ``.evalyn/data/datasets/`` that has a
``dataset.jsonl`` file. Source of truth for the JSON shape lives in
``dashboard/frontend/src/v2/api/types.ts``.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ._shared import _caches_disabled, dataset_roots, list_dataset_dirs, load_all_runs

logger = logging.getLogger(__name__)

router = APIRouter()

# The detail sub-router (``GET /{name}``) is mounted on the same prefix so
# both ``/api/v2/datasets`` (list) and ``/api/v2/datasets/{name}`` (detail)
# coexist. Imported below the helper definitions because the detail module
# imports back into this module for ``_coverage`` / ``_meta`` reuse.

# Mtime-keyed cache for ``_coverage`` results.
# 445 dataset.jsonl files (~55MB) on prod take 7s+ to re-parse on every
# /api/v2/datasets request; the file is read-mostly so cache by mtime.
_coverage_cache: dict[Path, tuple[float, tuple[int, list[dict]]]] = {}

# Same trick for ``meta.json`` reads (445 files on prod, ~2s of stat+parse).
_meta_cache: dict[Path, tuple[float | None, dict]] = {}


def _category_of(item: dict) -> str:
    """Pull a category out of the item, falling back to ``uncategorized``."""
    cat = item.get("category")
    if isinstance(cat, str) and cat:
        return cat
    meta = item.get("metadata") or {}
    if isinstance(meta, dict):
        mc = meta.get("category")
        if isinstance(mc, str) and mc:
            return mc
    tags = item.get("tags")
    if isinstance(tags, list) and tags and isinstance(tags[0], str):
        return tags[0]
    return "uncategorized"


def _coverage(jsonl_path: Path) -> tuple[int, list[dict]]:
    """Return ``(item_count, coverage_buckets)`` for the dataset.

    Always includes an ``uncategorized`` bucket if any items lack a
    category. Result is mtime-cached - on prod (~445 dataset.jsonl files,
    55MB total) this cuts the per-request cost from ~7s to ~free.
    """
    try:
        mtime = jsonl_path.stat().st_mtime
    except OSError:
        return 0, []
    cached = _coverage_cache.get(jsonl_path)
    if cached and cached[0] == mtime:
        return cached[1]
    counts: Counter = Counter()
    n = 0
    try:
        with jsonl_path.open(encoding="utf-8") as f:
            for line_num, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                n += 1
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("dataset.jsonl line %d at %s: %s", line_num, jsonl_path, exc)
                    continue
                counts[_category_of(item)] += 1
    except OSError as exc:
        logger.warning("dataset.jsonl read failed at %s: %s", jsonl_path, exc)
        return 0, []
    coverage = [{"label": label, "value": value} for label, value in counts.most_common()]
    result = (n, coverage)
    _coverage_cache[jsonl_path] = (mtime, result)
    return result


def _meta(dataset_dir: Path) -> dict:
    """Cached parse of ``dataset_dir/meta.json``.

    A negative cache (``mtime=None``) records "no meta on disk" so we
    skip the second ``stat()`` on every subsequent request - useful when
    most prod datasets have no ``meta.json`` and the wall-clock cost was
    dominated by repeated ``exists()`` checks.
    """
    p = dataset_dir / "meta.json"
    try:
        mtime: float | None = p.stat().st_mtime
    except OSError:
        mtime = None
    cached = _meta_cache.get(p)
    if cached and cached[0] == mtime:
        return cached[1]
    if mtime is None:
        _meta_cache[p] = (None, {})
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("meta.json read failed at %s: %s", p, exc)
        data = {}
    _meta_cache[p] = (mtime, data)
    return data


def _last_used_index() -> dict[str, str]:
    """Return ``{dataset_name: max(created_at)}`` from the cached runs.

    Reuses the run cache populated by :func:`load_all_runs` instead of
    re-reading every ``results.json`` per dataset (the original
    implementation walked the eval_runs/ tree a second time and added
    ~3.8s per request on prod data).
    """
    latest: dict[str, str] = {}
    for run in load_all_runs():
        ds = run.get("_dataset")
        if not ds:
            continue
        ts = run.get("created_at")
        if not ts:
            continue
        cur = latest.get(ds)
        if cur is None or ts > cur:
            latest[ds] = ts
    return latest


def _build_cards() -> list[dict]:
    """Walk roots once and return the dataset cards list."""
    seen: set[str] = set()
    cards: list[dict] = []
    last_used = _last_used_index()
    for root in dataset_roots():
        for dataset_dir in list_dataset_dirs(root):
            if dataset_dir.name in seen:
                continue
            jsonl = dataset_dir / "dataset.jsonl"
            if not jsonl.exists():
                continue
            seen.add(dataset_dir.name)
            n, coverage = _coverage(jsonl)
            meta = _meta(dataset_dir)
            cards.append(
                {
                    "name": dataset_dir.name,
                    "n": n,
                    "source": meta.get("source") or "JSONL",
                    "tags": meta.get("tags") or [],
                    "coverage": coverage,
                    "last_used_iso": last_used.get(dataset_dir.name),
                }
            )
    return cards


# Response-level cache: 491 dataset dirs => ~3-7s of stat()/parse() per
# request even with per-file caches because each dataset still costs a
# few hundred microseconds of dict lookups + path joins on WSL. We key
# the cached payload by ``(root, root.stat().st_mtime)`` for every
# dataset root - the mtime bumps when a dataset is added or removed,
# which is the only mutation list_datasets cares about.
_response_cache: dict[tuple, list[dict]] = {}


def _response_cache_key() -> tuple | None:
    """Return ``((root, mtime), ...)`` or ``None`` if any root is gone.

    Using all roots' mtimes catches both prod and demo additions. We
    return ``None`` (cache miss) on stat failures so we don't serve
    stale data when something racy happens.
    """
    parts: list[tuple] = []
    for root in dataset_roots():
        try:
            parts.append((root, root.stat().st_mtime))
        except OSError:
            return None
    return tuple(parts)


# Mount the detail sub-router. Imported here (not at module top) so the
# detail module's ``from .datasets import _coverage, _meta`` doesn't hit a
# circular import at startup - by this point both are fully defined.
from . import dataset_detail as _dataset_detail  # noqa: E402

router.include_router(_dataset_detail.router)


@router.get("")
async def list_datasets() -> JSONResponse:
    """Return one card per dataset directory.

    Walks every root from :func:`dataset_roots` so prod runs (under
    ``data/prod/datasets/``) and demo fixture runs (under
    ``.evalyn/data/datasets/``) both surface. First occurrence of a name
    wins so prod entries shadow demo entries on collision.

    The whole response is mtime-cached on every dataset root - even with
    per-file caches the per-dataset stat()+lookup cost dominates on prod
    (~491 directories), so memoising the full response lets warm
    requests skip the loop entirely.
    """
    key = _response_cache_key()
    if key is not None and not _caches_disabled() and key in _response_cache:
        return JSONResponse(_response_cache[key])
    cards = _build_cards()
    if key is not None and not _caches_disabled():
        _response_cache[key] = cards
    return JSONResponse(cards)
