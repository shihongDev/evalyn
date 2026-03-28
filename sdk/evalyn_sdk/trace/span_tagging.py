from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from ..models import Span


STANDARD_TAGS: Set[str] = {"environment", "user_id", "experiment_id", "variant", "version"}


@dataclass
class SpanTag:
    """A single key-value tag."""

    key: str
    value: Any

    def as_dict(self) -> Dict[str, Any]:
        return {"key": self.key, "value": self.value}


@dataclass
class TagSet:
    """Collection of tags as a key-value mapping."""

    tags: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {"tags": dict(self.tags)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TagSet:
        return cls(tags=dict(data.get("tags", {})))

    def merge(self, other: TagSet) -> TagSet:
        """Return a new TagSet with tags from both, other taking precedence."""
        merged = dict(self.tags)
        merged.update(other.tags)
        return TagSet(tags=merged)


def create_tag_set(**tags: Any) -> TagSet:
    """Create a TagSet from keyword arguments."""
    return TagSet(tags=dict(tags))


def apply_tags(span: Span, tags: TagSet) -> Span:
    """Return a new Span with the given tags merged into span.attributes["tags"].

    Does not mutate the original span.
    """
    new_span = copy.deepcopy(span)
    existing = new_span.attributes.get("tags", {})
    merged = dict(existing)
    merged.update(tags.tags)
    new_span.attributes["tags"] = merged
    return new_span


def get_tags(span: Span) -> TagSet:
    """Extract a TagSet from a span's attributes."""
    raw = span.attributes.get("tags", {})
    return TagSet(tags=dict(raw))


def has_tag(span: Span, key: str) -> bool:
    """Check whether a span has a tag with the given key."""
    raw = span.attributes.get("tags", {})
    return key in raw


def filter_spans_by_tag(
    spans: List[Span],
    key: str,
    value: Any = None,
) -> List[Span]:
    """Filter spans that have a tag matching key (and optionally value)."""
    result: List[Span] = []
    for s in spans:
        raw = s.attributes.get("tags", {})
        if key not in raw:
            continue
        if value is not None and raw[key] != value:
            continue
        result.append(s)
    return result
