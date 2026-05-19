"""
Dataset snapshot comparison: compare two dataset versions showing
item-level content diffs.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class ItemDiff:
    """Diff result for a single dataset item between two snapshots."""

    item_id: str
    change_type: str  # "added", "removed", "modified", "unchanged"
    old_input: str = ""
    new_input: str = ""
    old_output: str = ""
    new_output: str = ""
    diff_text: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "change_type": self.change_type,
            "old_input": self.old_input,
            "new_input": self.new_input,
            "old_output": self.old_output,
            "new_output": self.new_output,
            "diff_text": self.diff_text,
        }


@dataclass
class SnapshotComparison:
    """Comparison result between two dataset snapshots."""

    version_a: str
    version_b: str
    diffs: list[ItemDiff] = field(default_factory=list)
    added_count: int = 0
    removed_count: int = 0
    modified_count: int = 0
    unchanged_count: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "version_a": self.version_a,
            "version_b": self.version_b,
            "diffs": [d.as_dict() for d in self.diffs],
            "added_count": self.added_count,
            "removed_count": self.removed_count,
            "modified_count": self.modified_count,
            "unchanged_count": self.unchanged_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SnapshotComparison:
        diffs = [ItemDiff(**d) for d in data.get("diffs", [])]
        return cls(
            version_a=data.get("version_a", ""),
            version_b=data.get("version_b", ""),
            diffs=diffs,
            added_count=data.get("added_count", 0),
            removed_count=data.get("removed_count", 0),
            modified_count=data.get("modified_count", 0),
            unchanged_count=data.get("unchanged_count", 0),
        )

    def format_text(self) -> str:
        lines: list[str] = []
        lines.append("Snapshot Comparison")
        lines.append("=" * 40)
        lines.append(f"  {self.version_a} -> {self.version_b}")
        lines.append(f"  Added: {self.added_count}")
        lines.append(f"  Removed: {self.removed_count}")
        lines.append(f"  Modified: {self.modified_count}")
        lines.append(f"  Unchanged: {self.unchanged_count}")
        total = self.added_count + self.removed_count + self.modified_count + self.unchanged_count
        lines.append(f"  Total items: {total}")
        if self.diffs:
            lines.append("")
            changed = [d for d in self.diffs if d.change_type != "unchanged"]
            for d in changed[:20]:
                lines.append(f"  [{d.change_type.upper()}] {d.item_id}")
                if d.diff_text:
                    for diff_line in d.diff_text.splitlines()[:5]:
                        lines.append(f"    {diff_line}")
            if len(changed) > 20:
                lines.append(f"  ... and {len(changed) - 20} more changes")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def compute_item_diff(old_item: dict[str, Any], new_item: dict[str, Any]) -> ItemDiff:
    """Compute diff for one item. Uses difflib.unified_diff on input text."""
    item_id = str(old_item.get("id", new_item.get("id", "")))
    old_input = str(old_item.get("input", ""))
    new_input = str(new_item.get("input", ""))
    old_output = str(old_item.get("output", ""))
    new_output = str(new_item.get("output", ""))

    if old_input == new_input and old_output == new_output:
        return ItemDiff(
            item_id=item_id,
            change_type="unchanged",
            old_input=old_input,
            new_input=new_input,
            old_output=old_output,
            new_output=new_output,
            diff_text="",
        )

    diff_lines = list(difflib.unified_diff(
        old_input.splitlines(keepends=True),
        new_input.splitlines(keepends=True),
        fromfile="old_input",
        tofile="new_input",
        lineterm="",
    ))
    diff_text = "\n".join(diff_lines)

    return ItemDiff(
        item_id=item_id,
        change_type="modified",
        old_input=old_input,
        new_input=new_input,
        old_output=old_output,
        new_output=new_output,
        diff_text=diff_text,
    )


def compare_snapshots(
    items_a: list[dict[str, Any]],
    items_b: list[dict[str, Any]],
    version_a: str = "v1",
    version_b: str = "v2",
) -> SnapshotComparison:
    """Compare two lists of dataset items. Match by 'id' field."""
    index_a: dict[str, dict[str, Any]] = {}
    for item in items_a:
        item_id = str(item.get("id", ""))
        index_a[item_id] = item

    index_b: dict[str, dict[str, Any]] = {}
    for item in items_b:
        item_id = str(item.get("id", ""))
        index_b[item_id] = item

    all_ids = list(dict.fromkeys(list(index_a.keys()) + list(index_b.keys())))

    diffs: list[ItemDiff] = []
    added_count = 0
    removed_count = 0
    modified_count = 0
    unchanged_count = 0

    for item_id in all_ids:
        in_a = item_id in index_a
        in_b = item_id in index_b

        if in_a and not in_b:
            old_item = index_a[item_id]
            diffs.append(ItemDiff(
                item_id=item_id,
                change_type="removed",
                old_input=str(old_item.get("input", "")),
                old_output=str(old_item.get("output", "")),
            ))
            removed_count += 1
        elif in_b and not in_a:
            new_item = index_b[item_id]
            diffs.append(ItemDiff(
                item_id=item_id,
                change_type="added",
                new_input=str(new_item.get("input", "")),
                new_output=str(new_item.get("output", "")),
            ))
            added_count += 1
        else:
            diff = compute_item_diff(index_a[item_id], index_b[item_id])
            diffs.append(diff)
            if diff.change_type == "modified":
                modified_count += 1
            else:
                unchanged_count += 1

    return SnapshotComparison(
        version_a=version_a,
        version_b=version_b,
        diffs=diffs,
        added_count=added_count,
        removed_count=removed_count,
        modified_count=modified_count,
        unchanged_count=unchanged_count,
    )


def filter_diffs(comparison: SnapshotComparison, change_type: str) -> list[ItemDiff]:
    """Filter diffs by change type."""
    return [d for d in comparison.diffs if d.change_type == change_type]


def render_diff_summary(comparison: SnapshotComparison) -> str:
    """ASCII summary of changes."""
    total = (
        comparison.added_count
        + comparison.removed_count
        + comparison.modified_count
        + comparison.unchanged_count
    )
    lines: list[str] = []
    lines.append(f"Diff Summary: {comparison.version_a} -> {comparison.version_b}")
    lines.append("-" * 40)
    lines.append(f"  + Added:     {comparison.added_count}")
    lines.append(f"  - Removed:   {comparison.removed_count}")
    lines.append(f"  ~ Modified:  {comparison.modified_count}")
    lines.append(f"  = Unchanged: {comparison.unchanged_count}")
    lines.append(f"  Total:       {total}")
    if total > 0:
        changed = comparison.added_count + comparison.removed_count + comparison.modified_count
        rate = changed / total
        lines.append(f"  Change rate: {rate:.1%}")
    return "\n".join(lines)


def compute_change_rate(comparison: SnapshotComparison) -> float:
    """(added + removed + modified) / total items."""
    total = (
        comparison.added_count
        + comparison.removed_count
        + comparison.modified_count
        + comparison.unchanged_count
    )
    if total == 0:
        return 0.0
    changed = comparison.added_count + comparison.removed_count + comparison.modified_count
    return changed / total
