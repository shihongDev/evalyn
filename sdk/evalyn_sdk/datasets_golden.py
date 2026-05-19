"""
Golden set management for evaluation benchmarks.

Curate, version, compare, and filter immutable golden evaluation sets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class GoldenItem:
    """A single verified evaluation item in a golden set."""

    id: str
    input_text: str
    expected_output: str
    category: str = ""
    tags: list[str] = field(default_factory=list)
    verified: bool = False
    verified_by: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "input_text": self.input_text,
            "expected_output": self.expected_output,
            "category": self.category,
            "tags": list(self.tags),
            "verified": self.verified,
            "verified_by": self.verified_by,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GoldenItem:
        return cls(
            id=data["id"],
            input_text=data["input_text"],
            expected_output=data["expected_output"],
            category=data.get("category", ""),
            tags=data.get("tags", []),
            verified=data.get("verified", False),
            verified_by=data.get("verified_by", ""),
        )


@dataclass
class GoldenSet:
    """A versioned collection of golden evaluation items."""

    name: str
    items: list[GoldenItem] = field(default_factory=list)
    version: int = 1
    description: str = ""
    created_at: str = ""

    @property
    def count(self) -> int:
        return len(self.items)

    @property
    def verified_count(self) -> int:
        return sum(1 for item in self.items if item.verified)

    @property
    def verification_rate(self) -> float:
        if self.count == 0:
            return 0.0
        return self.verified_count / self.count

    def format_text(self) -> str:
        lines = [
            f"GoldenSet: {self.name} v{self.version}",
            f"  {self.count} items, {self.verified_count} verified "
            f"({self.verification_rate:.0%})",
        ]
        if self.description:
            lines.append(f"  {self.description}")
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "items": [item.as_dict() for item in self.items],
            "version": self.version,
            "description": self.description,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GoldenSet:
        items = [GoldenItem.from_dict(d) for d in data.get("items", [])]
        return cls(
            name=data["name"],
            items=items,
            version=data.get("version", 1),
            description=data.get("description", ""),
            created_at=data.get("created_at", ""),
        )


@dataclass
class GoldenSetComparison:
    """Result of comparing two golden sets."""

    set_a_name: str
    set_b_name: str
    added_ids: list[str] = field(default_factory=list)
    removed_ids: list[str] = field(default_factory=list)
    modified_ids: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "set_a_name": self.set_a_name,
            "set_b_name": self.set_b_name,
            "added_ids": list(self.added_ids),
            "removed_ids": list(self.removed_ids),
            "modified_ids": list(self.modified_ids),
        }

    def format_text(self) -> str:
        lines = [f"Comparison: {self.set_a_name} -> {self.set_b_name}"]
        lines.append(f"  Added: {len(self.added_ids)}")
        lines.append(f"  Removed: {len(self.removed_ids)}")
        lines.append(f"  Modified: {len(self.modified_ids)}")
        if self.added_ids:
            lines.append(f"  + {', '.join(self.added_ids)}")
        if self.removed_ids:
            lines.append(f"  - {', '.join(self.removed_ids)}")
        if self.modified_ids:
            lines.append(f"  ~ {', '.join(self.modified_ids)}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def create_golden_set(
    name: str,
    items: list[dict[str, Any]],
    description: str = "",
) -> GoldenSet:
    """Create a golden set from a list of item dicts."""
    golden_items = [GoldenItem.from_dict(d) for d in items]
    now = datetime.now(timezone.utc).isoformat()
    return GoldenSet(
        name=name,
        items=golden_items,
        version=1,
        description=description,
        created_at=now,
    )


def add_to_golden_set(golden: GoldenSet, item: GoldenItem) -> GoldenSet:
    """Add an item to a golden set. Rejects duplicates. Returns a new set (does not mutate)."""
    existing_ids = {i.id for i in golden.items}
    if item.id in existing_ids:
        raise ValueError(f"Duplicate item ID '{item.id}' in golden set '{golden.name}'")
    new_items = list(golden.items) + [item]
    return GoldenSet(
        name=golden.name,
        items=new_items,
        version=golden.version,
        description=golden.description,
        created_at=golden.created_at,
    )


def remove_from_golden_set(golden: GoldenSet, item_id: str) -> GoldenSet:
    """Remove an item by ID. Returns a new set (does not mutate)."""
    new_items = [item for item in golden.items if item.id != item_id]
    return GoldenSet(
        name=golden.name,
        items=new_items,
        version=golden.version,
        description=golden.description,
        created_at=golden.created_at,
    )


def verify_item(
    golden: GoldenSet,
    item_id: str,
    verified_by: str = "",
) -> GoldenSet:
    """Mark an item as verified. Returns a new set (does not mutate)."""
    new_items: list[GoldenItem] = []
    for item in golden.items:
        if item.id == item_id:
            new_items.append(
                GoldenItem(
                    id=item.id,
                    input_text=item.input_text,
                    expected_output=item.expected_output,
                    category=item.category,
                    tags=list(item.tags),
                    verified=True,
                    verified_by=verified_by,
                )
            )
        else:
            new_items.append(item)
    return GoldenSet(
        name=golden.name,
        items=new_items,
        version=golden.version,
        description=golden.description,
        created_at=golden.created_at,
    )


def compare_golden_sets(
    set_a: GoldenSet,
    set_b: GoldenSet,
) -> GoldenSetComparison:
    """Find added, removed, and modified items between two golden sets.

    "Added" means present in set_b but not set_a.
    "Removed" means present in set_a but not set_b.
    "Modified" means present in both but with different content.
    """
    a_by_id = {item.id: item for item in set_a.items}
    b_by_id = {item.id: item for item in set_b.items}

    a_ids = set(a_by_id.keys())
    b_ids = set(b_by_id.keys())

    added = sorted(b_ids - a_ids)
    removed = sorted(a_ids - b_ids)

    modified: list[str] = []
    for item_id in sorted(a_ids & b_ids):
        a_item = a_by_id[item_id]
        b_item = b_by_id[item_id]
        if (
            a_item.input_text != b_item.input_text
            or a_item.expected_output != b_item.expected_output
            or a_item.category != b_item.category
        ):
            modified.append(item_id)

    return GoldenSetComparison(
        set_a_name=set_a.name,
        set_b_name=set_b.name,
        added_ids=added,
        removed_ids=removed,
        modified_ids=modified,
    )


def filter_golden_set(
    golden: GoldenSet,
    category: str = "",
    tags: list[str] | None = None,
    verified_only: bool = False,
) -> GoldenSet:
    """Filter items in a golden set by category, tags, or verification status.

    Returns a new golden set with only matching items.
    """
    if tags is None:
        tags = []
    filtered: list[GoldenItem] = []
    for item in golden.items:
        if category and item.category != category:
            continue
        if tags and not any(t in item.tags for t in tags):
            continue
        if verified_only and not item.verified:
            continue
        filtered.append(item)

    return GoldenSet(
        name=golden.name,
        items=filtered,
        version=golden.version,
        description=golden.description,
        created_at=golden.created_at,
    )


def bump_version(golden: GoldenSet) -> GoldenSet:
    """Increment the version of a golden set. Returns a new set."""
    return GoldenSet(
        name=golden.name,
        items=list(golden.items),
        version=golden.version + 1,
        description=golden.description,
        created_at=golden.created_at,
    )
