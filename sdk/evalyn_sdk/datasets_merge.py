"""Dataset merge and diff utilities.

Combine two datasets with deduplication, or compare them to show
added/removed/changed items.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple

from .datasets import hash_inputs


@dataclass
class DatasetDiff:
    """Difference between two datasets."""

    added: list = field(default_factory=list)      # items in B but not A
    removed: list = field(default_factory=list)     # items in A but not B
    modified: list = field(default_factory=list)    # items with same ID but different content
    unchanged: int = 0

    @property
    def has_changes(self) -> bool:
        return len(self.added) > 0 or len(self.removed) > 0 or len(self.modified) > 0

    @property
    def summary(self) -> dict[str, int]:
        return {
            "added": len(self.added),
            "removed": len(self.removed),
            "modified": len(self.modified),
            "unchanged": self.unchanged,
        }

    def format_text(self) -> str:
        lines = ["Dataset Diff"]
        lines.append(f"  Added:     {len(self.added)}")
        lines.append(f"  Removed:   {len(self.removed)}")
        lines.append(f"  Modified:  {len(self.modified)}")
        lines.append(f"  Unchanged: {self.unchanged}")
        if self.added:
            lines.append("")
            lines.append("  Added items:")
            for item in self.added[:10]:
                lines.append(f"    + {item.id}")
            if len(self.added) > 10:
                lines.append(f"    ... and {len(self.added) - 10} more")
        if self.removed:
            lines.append("")
            lines.append("  Removed items:")
            for item in self.removed[:10]:
                lines.append(f"    - {item.id}")
            if len(self.removed) > 10:
                lines.append(f"    ... and {len(self.removed) - 10} more")
        return "\n".join(lines)


def diff_datasets(
    dataset_a: list,
    dataset_b: list,
) -> DatasetDiff:
    """Compare two datasets and return the differences.

    Uses item ID for matching, then compares input/output content.

    Args:
        dataset_a: The "before" dataset.
        dataset_b: The "after" dataset.

    Returns:
        DatasetDiff with added, removed, modified, and unchanged counts.
    """
    a_by_id = {item.id: item for item in dataset_a}
    b_by_id = {item.id: item for item in dataset_b}

    a_ids = set(a_by_id.keys())
    b_ids = set(b_by_id.keys())

    added = [b_by_id[i] for i in sorted(b_ids - a_ids)]
    removed = [a_by_id[i] for i in sorted(a_ids - b_ids)]

    modified = []
    unchanged = 0

    for item_id in sorted(a_ids & b_ids):
        a_item = a_by_id[item_id]
        b_item = b_by_id[item_id]
        if _items_equal(a_item, b_item):
            unchanged += 1
        else:
            modified.append(b_item)

    return DatasetDiff(
        added=added,
        removed=removed,
        modified=modified,
        unchanged=unchanged,
    )


def merge_datasets(
    dataset_a: list,
    dataset_b: list,
    deduplicate: bool = True,
    prefer: str = "b",
) -> tuple[list, int]:
    """Merge two datasets.

    Args:
        dataset_a: First dataset.
        dataset_b: Second dataset.
        deduplicate: If True, remove items with identical input content.
        prefer: When items have same ID, prefer "a" or "b" version.

    Returns:
        Tuple of (merged items, duplicate count).
    """
    if prefer == "a":
        primary, secondary = dataset_a, dataset_b
    else:
        primary, secondary = dataset_b, dataset_a

    # Build ID map: primary items take precedence
    merged_by_id: dict[str, Any] = {}
    for item in secondary:
        merged_by_id[item.id] = item
    for item in primary:
        merged_by_id[item.id] = item  # overwrite with preferred

    merged = list(merged_by_id.values())

    duplicate_count = 0
    if deduplicate:
        seen_hashes: set[str] = set()
        deduped = []
        for item in merged:
            h = _input_hash(item)
            if h in seen_hashes:
                duplicate_count += 1
            else:
                seen_hashes.add(h)
                deduped.append(item)
        merged = deduped

    return merged, duplicate_count


def _items_equal(a, b) -> bool:
    """Check if two items have the same content (input + output)."""
    return _input_hash(a) == _input_hash(b) and str(a.output) == str(b.output)


def _input_hash(item) -> str:
    """Hash the input of an item for dedup."""
    if isinstance(item.input, dict):
        return hash_inputs(item.input)
    return hash_inputs({"_raw": str(item.input)})
