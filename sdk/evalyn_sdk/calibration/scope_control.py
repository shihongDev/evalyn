"""Calibration scope control.

Calibrate only for specific item subsets by applying include, exclude,
metadata, and size filters to item lists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ScopeFilter:
    """Defines which items to include in a calibration scope."""

    include_ids: List[str] = field(default_factory=list)
    exclude_ids: List[str] = field(default_factory=list)
    metadata_key: str = ""
    metadata_value: str = ""
    max_items: Optional[int] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "include_ids": self.include_ids,
            "exclude_ids": self.exclude_ids,
            "metadata_key": self.metadata_key,
            "metadata_value": self.metadata_value,
            "max_items": self.max_items,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ScopeFilter:
        return cls(
            include_ids=data.get("include_ids", []),
            exclude_ids=data.get("exclude_ids", []),
            metadata_key=data.get("metadata_key", ""),
            metadata_value=data.get("metadata_value", ""),
            max_items=data.get("max_items"),
        )


@dataclass
class ScopeResult:
    """Result of applying a scope filter."""

    original_count: int
    filtered_count: int
    filter_description: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "original_count": self.original_count,
            "filtered_count": self.filtered_count,
            "filter_description": self.filter_description,
        }

    def format_text(self) -> str:
        """Human-readable summary of the filtering result."""
        return (
            f"{self.filtered_count}/{self.original_count} items"
            f" after filter: {self.filter_description}"
        )


def apply_scope_filter(
    items: List[Dict[str, Any]], scope: ScopeFilter
) -> Tuple[List[Dict[str, Any]], ScopeResult]:
    """Filter items according to scope rules.

    Order of operations: include_ids, exclude_ids, metadata filter, max_items.
    Each item dict must have an "id" key and may have a "metadata" dict.
    """
    original_count = len(items)
    result = list(items)

    # 1. Include filter - keep only listed ids
    if scope.include_ids:
        include_set = set(scope.include_ids)
        result = [it for it in result if it["id"] in include_set]

    # 2. Exclude filter - remove listed ids
    if scope.exclude_ids:
        exclude_set = set(scope.exclude_ids)
        result = [it for it in result if it["id"] not in exclude_set]

    # 3. Metadata filter
    if scope.metadata_key:
        result = [
            it
            for it in result
            if it.get("metadata", {}).get(scope.metadata_key) == scope.metadata_value
        ]

    # 4. Max items limit
    if scope.max_items is not None:
        result = result[: scope.max_items]

    description = describe_scope(scope)
    return result, ScopeResult(
        original_count=original_count,
        filtered_count=len(result),
        filter_description=description,
    )


def create_include_scope(ids: List[str]) -> ScopeFilter:
    """Factory for an include-only scope."""
    return ScopeFilter(include_ids=ids)


def create_exclude_scope(ids: List[str]) -> ScopeFilter:
    """Factory for an exclude scope."""
    return ScopeFilter(exclude_ids=ids)


def create_metadata_scope(key: str, value: str) -> ScopeFilter:
    """Factory for a metadata filter scope."""
    return ScopeFilter(metadata_key=key, metadata_value=value)


def describe_scope(scope: ScopeFilter) -> str:
    """Return a human-readable description of a scope filter."""
    parts: List[str] = []
    if scope.include_ids:
        parts.append(f"include {len(scope.include_ids)} ids")
    if scope.exclude_ids:
        parts.append(f"exclude {len(scope.exclude_ids)} ids")
    if scope.metadata_key:
        parts.append(f"metadata {scope.metadata_key}={scope.metadata_value}")
    if scope.max_items is not None:
        parts.append(f"max {scope.max_items} items")
    if not parts:
        return "no filter"
    return ", ".join(parts)
