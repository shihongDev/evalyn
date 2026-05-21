"""Incremental dataset building: append new traces without full rebuild.

Tracks the last build timestamp and only processes traces created after it.
Deduplicates new items against existing dataset using content hashing.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .datasets import hash_inputs, save_dataset

logger = logging.getLogger(__name__)


@dataclass
class IncrementalBuildResult:
    """Result of an incremental dataset build."""

    new_items_added: int
    duplicates_skipped: int
    total_items: int
    last_build_timestamp: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "new_items_added": self.new_items_added,
            "duplicates_skipped": self.duplicates_skipped,
            "total_items": self.total_items,
            "last_build_timestamp": self.last_build_timestamp,
        }


def load_build_state(dataset_dir: Path) -> str | None:
    """Load the last build timestamp from the dataset's meta.json.

    Returns:
        ISO timestamp string, or None if no previous build.
    """
    meta_path = dataset_dir / "meta.json"
    if not meta_path.exists():
        return None
    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        return meta.get("last_incremental_build")
    except (json.JSONDecodeError, OSError):
        return None


def save_build_state(dataset_dir: Path, timestamp: str) -> None:
    """Save the incremental build timestamp to meta.json.

    Updates or creates the meta.json with the last build timestamp.
    """
    meta_path = dataset_dir / "meta.json"
    meta = {}
    if meta_path.exists():
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError):
            meta = {}

    meta["last_incremental_build"] = timestamp
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def incremental_build(
    existing_items: list,
    new_items: list,
    dataset_path: Path,
) -> IncrementalBuildResult:
    """Append new items to an existing dataset, deduplicating by input hash.

    Args:
        existing_items: Items already in the dataset.
        new_items: New items to consider adding.
        dataset_path: Path to the dataset.jsonl file.

    Returns:
        IncrementalBuildResult with counts and timestamp.
    """
    # Build hash set of existing items for dedup
    existing_hashes: set[str] = set()
    for item in existing_items:
        h = (
            hash_inputs(item.input)
            if isinstance(item.input, dict)
            else hash_inputs({"_raw": item.input})
        )
        existing_hashes.add(h)

    # Filter new items
    added = []
    duplicates = 0
    for item in new_items:
        h = (
            hash_inputs(item.input)
            if isinstance(item.input, dict)
            else hash_inputs({"_raw": item.input})
        )
        if h in existing_hashes:
            duplicates += 1
        else:
            added.append(item)
            existing_hashes.add(h)

    if added:
        # Append to existing dataset file
        all_items = existing_items + added
        save_dataset(all_items, dataset_path)
        logger.info("Added %d new items to dataset (skipped %d duplicates)", len(added), duplicates)

    now = datetime.now(timezone.utc).isoformat()

    # Update build state
    dataset_dir = dataset_path.parent if dataset_path.suffix else dataset_path
    save_build_state(dataset_dir, now)

    return IncrementalBuildResult(
        new_items_added=len(added),
        duplicates_skipped=duplicates,
        total_items=len(existing_items) + len(added),
        last_build_timestamp=now,
    )
