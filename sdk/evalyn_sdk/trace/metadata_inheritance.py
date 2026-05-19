"""Trace metadata inheritance.

Child spans automatically inherit parent's custom tags from attributes.
Supports three modes: inherit_all, inherit_listed, and no_inherit.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from ..models import Span


@dataclass
class InheritanceConfig:
    """Configuration for tag inheritance between parent and child spans.

    Modes:
    - inherit_all: child inherits all parent tags
    - inherit_listed: child inherits only keys in inherit_keys
    - no_inherit: no inheritance
    """

    mode: str = "inherit_all"
    inherit_keys: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "inherit_keys": list(self.inherit_keys),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InheritanceConfig:
        return cls(
            mode=data.get("mode", "inherit_all"),
            inherit_keys=data.get("inherit_keys", []),
        )


@dataclass
class InheritanceResult:
    """Summary of inheritance application."""

    spans_updated: int
    tags_inherited: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "spans_updated": self.spans_updated,
            "tags_inherited": self.tags_inherited,
        }


def get_inheritable_tags(span: Span, config: InheritanceConfig) -> dict[str, Any]:
    """Extract tags from a span that should be inherited based on config mode.

    Tags are stored in span.attributes["tags"].
    """
    tags = span.attributes.get("tags", {})
    if not isinstance(tags, dict):
        return {}

    if config.mode == "no_inherit":
        return {}
    elif config.mode == "inherit_listed":
        return {k: v for k, v in tags.items() if k in config.inherit_keys}
    else:
        # inherit_all
        return dict(tags)


def apply_inheritance(
    spans: list[Span],
    config: InheritanceConfig | None = None,
) -> tuple[list[Span], InheritanceResult]:
    """Apply tag inheritance from parent spans to child spans.

    Builds the parent-child tree, then for each child inherits parent tags
    according to config. Child's own tags override inherited ones.

    Returns new span copies - originals are not mutated.
    """
    if config is None:
        config = InheritanceConfig()

    if not spans:
        return [], InheritanceResult(spans_updated=0, tags_inherited=0)

    # Deep-copy all spans so originals remain untouched
    new_spans = [copy.deepcopy(s) for s in spans]

    # Build lookup by id
    by_id: dict[str, Span] = {s.id: s for s in new_spans}

    # Build children list per parent
    children_of: dict[str, list[str]] = {}
    for s in new_spans:
        if s.parent_id and s.parent_id in by_id:
            children_of.setdefault(s.parent_id, []).append(s.id)

    # Find roots (spans with no parent or parent not in set)
    roots = [s.id for s in new_spans if not s.parent_id or s.parent_id not in by_id]

    spans_updated = 0
    tags_inherited = 0

    # BFS from roots to propagate tags down the tree
    queue = list(roots)
    while queue:
        current_id = queue.pop(0)
        current = by_id[current_id]
        parent_tags = get_inheritable_tags(current, config)

        for child_id in children_of.get(current_id, []):
            child = by_id[child_id]
            child_tags = child.attributes.get("tags", {})
            if not isinstance(child_tags, dict):
                child_tags = {}

            # Merge: parent tags as base, child tags override
            if parent_tags:
                merged = dict(parent_tags)
                new_count = sum(1 for k in merged if k not in child_tags)
                merged.update(child_tags)
                child.attributes["tags"] = merged

                if new_count > 0:
                    spans_updated += 1
                    tags_inherited += new_count

            queue.append(child_id)

    return new_spans, InheritanceResult(
        spans_updated=spans_updated,
        tags_inherited=tags_inherited,
    )
