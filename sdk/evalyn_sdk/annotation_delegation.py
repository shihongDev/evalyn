"""Annotation delegation: assign items to annotators by expertise and workload."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class AnnotatorProfile:
    """Profile for an annotator with expertise tags and capacity."""

    annotator_id: str
    name: str
    expertise_tags: list[str] = field(default_factory=list)
    max_daily_items: int = 50
    items_assigned: int = 0
    items_completed: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "annotator_id": self.annotator_id,
            "name": self.name,
            "expertise_tags": list(self.expertise_tags),
            "max_daily_items": self.max_daily_items,
            "items_assigned": self.items_assigned,
            "items_completed": self.items_completed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnnotatorProfile:
        return cls(
            annotator_id=data["annotator_id"],
            name=data["name"],
            expertise_tags=data.get("expertise_tags", []),
            max_daily_items=data.get("max_daily_items", 50),
            items_assigned=data.get("items_assigned", 0),
            items_completed=data.get("items_completed", 0),
        )


@dataclass
class DelegationRule:
    """Rule mapping item tags to required annotator expertise."""

    rule_id: str
    item_tag: str
    required_expertise: list[str] = field(default_factory=list)
    priority: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "item_tag": self.item_tag,
            "required_expertise": list(self.required_expertise),
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DelegationRule:
        return cls(
            rule_id=data["rule_id"],
            item_tag=data["item_tag"],
            required_expertise=data.get("required_expertise", []),
            priority=data.get("priority", 0),
        )


@dataclass
class Assignment:
    """Assignment of an item to an annotator."""

    item_id: str
    annotator_id: str
    assigned_at: str
    status: str = "pending"  # pending / in_progress / completed
    rule_id: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "annotator_id": self.annotator_id,
            "assigned_at": self.assigned_at,
            "status": self.status,
            "rule_id": self.rule_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Assignment:
        return cls(
            item_id=data["item_id"],
            annotator_id=data["annotator_id"],
            assigned_at=data["assigned_at"],
            status=data.get("status", "pending"),
            rule_id=data.get("rule_id", ""),
        )


# ---------------------------------------------------------------------------
# Delegation Manager
# ---------------------------------------------------------------------------


class DelegationManager:
    """Manage item-to-annotator assignment using rules and workload balancing."""

    def __init__(self) -> None:
        self.annotators: dict[str, AnnotatorProfile] = {}
        self.rules: list[DelegationRule] = []
        self.assignments: list[Assignment] = []

    def register_annotator(self, profile: AnnotatorProfile) -> None:
        """Register an annotator profile."""
        self.annotators[profile.annotator_id] = profile

    def add_rule(self, rule: DelegationRule) -> None:
        """Add a delegation rule. Rules are kept sorted by descending priority."""
        self.rules.append(rule)
        self.rules.sort(key=lambda r: r.priority, reverse=True)

    # -- assignment helpers --------------------------------------------------

    def _matching_rules(self, item_tags: list[str]) -> list[DelegationRule]:
        """Return rules whose item_tag appears in the given tags, sorted by priority."""
        return [r for r in self.rules if r.item_tag in item_tags]

    def _required_expertise(self, item_tags: list[str]) -> list[str]:
        """Aggregate required expertise across all matching rules."""
        expertise: list[str] = []
        for rule in self._matching_rules(item_tags):
            for tag in rule.required_expertise:
                if tag not in expertise:
                    expertise.append(tag)
        return expertise

    def _best_rule_id(self, item_tags: list[str]) -> str:
        """Return the rule_id of the highest-priority matching rule, or empty."""
        rules = self._matching_rules(item_tags)
        return rules[0].rule_id if rules else ""

    def _score_annotator(self, profile: AnnotatorProfile, required: list[str]) -> tuple[int, int]:
        """Return (expertise_overlap, negative_assigned) for sorting.

        Higher overlap is better; fewer assigned items is better.
        """
        overlap = len(set(profile.expertise_tags) & set(required))
        return (overlap, -profile.items_assigned)

    def _pick_annotator(self, item_tags: list[str]) -> AnnotatorProfile | None:
        """Select the best available annotator for the given item tags."""
        required = self._required_expertise(item_tags)
        candidates = [p for p in self.annotators.values() if p.items_assigned < p.max_daily_items]
        if not candidates:
            return None
        # If no rules matched, required is empty - just pick least loaded
        if not required:
            candidates.sort(key=lambda p: p.items_assigned)
            return candidates[0]
        # Sort by (overlap desc, assigned asc)
        candidates.sort(key=lambda p: self._score_annotator(p, required), reverse=True)
        return candidates[0]

    # -- public API ----------------------------------------------------------

    def assign_item(self, item_id: str, item_tags: list[str]) -> Assignment | None:
        """Assign a single item to the best-matching annotator.

        Returns None if no annotator has capacity.
        """
        annotator = self._pick_annotator(item_tags)
        if annotator is None:
            return None
        rule_id = self._best_rule_id(item_tags)
        assignment = Assignment(
            item_id=item_id,
            annotator_id=annotator.annotator_id,
            assigned_at=_now_iso(),
            status="pending",
            rule_id=rule_id,
        )
        self.assignments.append(assignment)
        annotator.items_assigned += 1
        return assignment

    def assign_batch(self, items: list[tuple[str, list[str]]]) -> list[Assignment]:
        """Assign a batch of (item_id, tags) pairs."""
        results: list[Assignment] = []
        for item_id, tags in items:
            a = self.assign_item(item_id, tags)
            if a is not None:
                results.append(a)
        return results

    def get_annotator_workload(self, annotator_id: str) -> dict[str, Any]:
        """Return workload stats for a single annotator."""
        profile = self.annotators.get(annotator_id)
        if profile is None:
            return {"assigned": 0, "completed": 0, "remaining_capacity": 0}
        return {
            "assigned": profile.items_assigned,
            "completed": profile.items_completed,
            "remaining_capacity": max(0, profile.max_daily_items - profile.items_assigned),
        }

    def get_assignments(
        self,
        annotator_id: str | None = None,
        status: str | None = None,
    ) -> list[Assignment]:
        """Filter assignments by annotator and/or status."""
        result = self.assignments
        if annotator_id is not None:
            result = [a for a in result if a.annotator_id == annotator_id]
        if status is not None:
            result = [a for a in result if a.status == status]
        return result

    def complete_assignment(self, item_id: str) -> bool:
        """Mark an assignment as completed. Returns True if found."""
        for a in self.assignments:
            if a.item_id == item_id and a.status != "completed":
                a.status = "completed"
                profile = self.annotators.get(a.annotator_id)
                if profile is not None:
                    profile.items_completed += 1
                return True
        return False

    def rebalance(self, items: list[tuple[str, list[str]]]) -> list[Assignment]:
        """Redistribute unfinished assignments for better balance.

        Removes all non-completed assignments, resets annotator counts
        accordingly, then re-assigns the provided items from scratch.
        """
        # Collect non-completed assignments to remove
        to_remove = [a for a in self.assignments if a.status != "completed"]
        kept = [a for a in self.assignments if a.status == "completed"]

        # Adjust annotator assigned counts
        for a in to_remove:
            profile = self.annotators.get(a.annotator_id)
            if profile is not None:
                profile.items_assigned = max(0, profile.items_assigned - 1)

        self.assignments = kept
        return self.assign_batch(items)


# ---------------------------------------------------------------------------
# Standalone functions
# ---------------------------------------------------------------------------


def compute_workload_balance(manager: DelegationManager) -> dict[str, Any]:
    """Compute workload balance statistics across annotators.

    Returns min, max, mean items per annotator and the Gini coefficient.
    """
    profiles = list(manager.annotators.values())
    if not profiles:
        return {"min": 0, "max": 0, "mean": 0.0, "gini": 0.0}

    counts = [p.items_assigned for p in profiles]
    n = len(counts)
    total = sum(counts)
    mean = total / n if n else 0.0

    # Gini coefficient via the sort-once linear formula:
    #   G = (2·Σ i·x_i) / (n·Σ x_i) - (n+1)/n   for x sorted ascending
    # Equivalent to the O(n²) pairwise-absolute-difference formula but
    # avoids the nested loop, which dominates for large annotator pools.
    if total == 0 or n == 0:
        gini = 0.0
    else:
        sorted_counts = sorted(counts)
        weighted_sum = sum((i + 1) * x for i, x in enumerate(sorted_counts))
        gini = (2 * weighted_sum) / (n * total) - (n + 1) / n

    return {
        "min": min(counts),
        "max": max(counts),
        "mean": round(mean, 2),
        "gini": round(gini, 4),
    }


def format_delegation_dashboard(manager: DelegationManager) -> str:
    """Format a text table showing per-annotator progress."""
    lines: list[str] = []
    lines.append("Delegation Dashboard")
    lines.append("-" * 60)
    header = f"{'Annotator':<20} {'Assigned':>8} {'Done':>8} {'Capacity':>8}"
    lines.append(header)
    lines.append("-" * 60)

    for profile in sorted(manager.annotators.values(), key=lambda p: p.annotator_id):
        remaining = max(0, profile.max_daily_items - profile.items_assigned)
        row = (
            f"{profile.name:<20} "
            f"{profile.items_assigned:>8} "
            f"{profile.items_completed:>8} "
            f"{remaining:>8}"
        )
        lines.append(row)

    lines.append("-" * 60)
    return "\n".join(lines)
