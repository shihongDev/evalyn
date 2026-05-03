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

from ._shared import dataset_roots

logger = logging.getLogger(__name__)

router = APIRouter()


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
    category.
    """
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
                    logger.warning(
                        "dataset.jsonl line %d at %s: %s", line_num, jsonl_path, exc
                    )
                    continue
                counts[_category_of(item)] += 1
    except OSError as exc:
        logger.warning("dataset.jsonl read failed at %s: %s", jsonl_path, exc)
        return 0, []
    coverage = [
        {"label": label, "value": value}
        for label, value in counts.most_common()
    ]
    return n, coverage


def _meta(dataset_dir: Path) -> dict:
    p = dataset_dir / "meta.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("meta.json read failed at %s: %s", p, exc)
        return {}


def _last_used_iso(dataset_dir: Path) -> str | None:
    """Most-recent ``created_at`` across this dataset's runs, or ``None``."""
    eval_runs = dataset_dir / "eval_runs"
    if not eval_runs.is_dir():
        return None
    latest: str | None = None
    for run_dir in eval_runs.iterdir():
        if not run_dir.is_dir():
            continue
        results = run_dir / "results.json"
        if not results.exists():
            continue
        try:
            data = json.loads(results.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("results.json read failed at %s: %s", results, exc)
            continue
        ts = data.get("created_at")
        if ts and (latest is None or ts > latest):
            latest = ts
    return latest


@router.get("")
async def list_datasets() -> JSONResponse:
    """Return one card per dataset directory.

    Walks every root from :func:`dataset_roots` so prod runs (under
    ``data/prod/datasets/``) and demo fixture runs (under
    ``.evalyn/data/datasets/``) both surface. First occurrence of a name
    wins so prod entries shadow demo entries on collision.
    """
    seen: set[str] = set()
    cards: list[dict] = []
    for root in dataset_roots():
        for dataset_dir in sorted(root.iterdir()):
            if not dataset_dir.is_dir() or dataset_dir.name in seen:
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
                    "last_used_iso": _last_used_iso(dataset_dir),
                }
            )
    return JSONResponse(cards)
