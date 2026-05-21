"""Differential evaluation: only re-evaluate items that changed.

Compares two dataset versions by content hash to detect added, removed,
modified, and unchanged items. Unchanged items carry forward previous
results, avoiding redundant evaluation.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ItemChange:
    """A single item's change status between dataset versions."""

    item_id: str
    change_type: str  # "added", "removed", "modified", "unchanged"
    old_hash: str = ""
    new_hash: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "change_type": self.change_type,
            "old_hash": self.old_hash,
            "new_hash": self.new_hash,
        }


@dataclass
class DiffResult:
    """Result of diffing two dataset versions."""

    changes: list[ItemChange] = field(default_factory=list)
    added_count: int = 0
    removed_count: int = 0
    modified_count: int = 0
    unchanged_count: int = 0
    total_items: int = 0

    @property
    def items_to_evaluate(self) -> list[str]:
        """Item IDs that need re-evaluation: added + modified."""
        return [c.item_id for c in self.changes if c.change_type in ("added", "modified")]

    def as_dict(self) -> dict[str, Any]:
        return {
            "changes": [c.as_dict() for c in self.changes],
            "added_count": self.added_count,
            "removed_count": self.removed_count,
            "modified_count": self.modified_count,
            "unchanged_count": self.unchanged_count,
            "total_items": self.total_items,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiffResult:
        changes = [
            ItemChange(
                item_id=c["item_id"],
                change_type=c["change_type"],
                old_hash=c.get("old_hash", ""),
                new_hash=c.get("new_hash", ""),
            )
            for c in data.get("changes", [])
        ]
        return cls(
            changes=changes,
            added_count=data.get("added_count", 0),
            removed_count=data.get("removed_count", 0),
            modified_count=data.get("modified_count", 0),
            unchanged_count=data.get("unchanged_count", 0),
            total_items=data.get("total_items", 0),
        )

    def format_text(self) -> str:
        lines = [
            "Differential Evaluation Plan",
            f"  Total items: {self.total_items}",
            f"  Added: {self.added_count}",
            f"  Removed: {self.removed_count}",
            f"  Modified: {self.modified_count}",
            f"  Unchanged: {self.unchanged_count}",
            f"  Items to evaluate: {len(self.items_to_evaluate)}",
        ]
        return "\n".join(lines)


def hash_item(item_id: str, input_text: str, output_text: str = "") -> str:
    """SHA-256 hash of concatenated item content. Deterministic."""
    content = f"{item_id}|{input_text}|{output_text}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def compute_diff(
    old_items: dict[str, str],
    new_items: dict[str, str],
) -> DiffResult:
    """Compare two sets of item_id -> hash. Detect changes.

    Args:
        old_items: Mapping of item_id to content hash for old version.
        new_items: Mapping of item_id to content hash for new version.

    Returns:
        DiffResult with all changes categorized.
    """
    changes: list[ItemChange] = []
    added = 0
    removed = 0
    modified = 0
    unchanged = 0

    all_ids = sorted(set(old_items.keys()) | set(new_items.keys()))

    for item_id in all_ids:
        old_hash = old_items.get(item_id, "")
        new_hash = new_items.get(item_id, "")

        if item_id not in old_items:
            changes.append(ItemChange(item_id, "added", new_hash=new_hash))
            added += 1
        elif item_id not in new_items:
            changes.append(ItemChange(item_id, "removed", old_hash=old_hash))
            removed += 1
        elif old_hash != new_hash:
            changes.append(ItemChange(item_id, "modified", old_hash=old_hash, new_hash=new_hash))
            modified += 1
        else:
            changes.append(ItemChange(item_id, "unchanged", old_hash=old_hash, new_hash=new_hash))
            unchanged += 1

    total = len(new_items)

    return DiffResult(
        changes=changes,
        added_count=added,
        removed_count=removed,
        modified_count=modified,
        unchanged_count=unchanged,
        total_items=total,
    )


def carry_forward_results(
    previous_results: list[dict[str, Any]],
    diff: DiffResult,
) -> list[dict[str, Any]]:
    """Return results from previous run for unchanged items only.

    Filters previous_results to keep only those whose item_id
    appears as "unchanged" in the diff.
    """
    unchanged_ids = {c.item_id for c in diff.changes if c.change_type == "unchanged"}
    return [r for r in previous_results if r.get("item_id") in unchanged_ids]


def build_differential_plan(
    old_hashes: dict[str, str],
    new_hashes: dict[str, str],
) -> DiffResult:
    """Build a plan showing what needs re-evaluation.

    Convenience wrapper around compute_diff.
    """
    return compute_diff(old_hashes, new_hashes)
