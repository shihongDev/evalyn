"""
Annotation conflict resolution: side-by-side view when annotators disagree,
with tiebreaker workflow. Pure Python, no external dependencies.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class AnnotatorLabel:
    """A single label assigned by an annotator to an item."""

    item_id: str
    annotator_id: str
    label: float
    confidence: float = 1.0
    reasoning: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "annotator_id": self.annotator_id,
            "label": self.label,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnnotatorLabel:
        return cls(
            item_id=data["item_id"],
            annotator_id=data["annotator_id"],
            label=data["label"],
            confidence=data.get("confidence", 1.0),
            reasoning=data.get("reasoning", ""),
        )


@dataclass
class Conflict:
    """A conflict where annotators disagree on an item."""

    item_id: str
    labels: list[AnnotatorLabel]
    max_disagreement: float
    needs_tiebreaker: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "labels": [lbl.as_dict() for lbl in self.labels],
            "max_disagreement": self.max_disagreement,
            "needs_tiebreaker": self.needs_tiebreaker,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Conflict:
        return cls(
            item_id=data["item_id"],
            labels=[AnnotatorLabel.from_dict(d) for d in data["labels"]],
            max_disagreement=data["max_disagreement"],
            needs_tiebreaker=data.get("needs_tiebreaker", True),
        )


@dataclass
class ResolutionPolicy:
    """Policy for resolving annotation conflicts."""

    policy_name: str  # "majority_vote" / "senior_override" / "discussion_required"
    senior_annotators: list[str] = field(default_factory=list)
    min_agreement: float = 0.5

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "senior_annotators": self.senior_annotators,
            "min_agreement": self.min_agreement,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResolutionPolicy:
        return cls(
            policy_name=data["policy_name"],
            senior_annotators=data.get("senior_annotators", []),
            min_agreement=data.get("min_agreement", 0.5),
        )


@dataclass
class Resolution:
    """The resolved outcome for a conflicting item."""

    item_id: str
    resolved_label: float
    method: str
    resolved_by: str = ""
    original_labels: list[AnnotatorLabel] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "resolved_label": self.resolved_label,
            "method": self.method,
            "resolved_by": self.resolved_by,
            "original_labels": [lbl.as_dict() for lbl in self.original_labels],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Resolution:
        return cls(
            item_id=data["item_id"],
            resolved_label=data["resolved_label"],
            method=data["method"],
            resolved_by=data.get("resolved_by", ""),
            original_labels=[AnnotatorLabel.from_dict(d) for d in data.get("original_labels", [])],
        )


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def detect_conflicts(labels: list[AnnotatorLabel], threshold: float = 0.5) -> list[Conflict]:
    """Find items where annotator labels differ by more than *threshold*.

    Groups labels by item_id, computes pairwise max disagreement, and
    returns a Conflict for each item that exceeds the threshold.
    """
    by_item: dict[str, list[AnnotatorLabel]] = defaultdict(list)
    for lbl in labels:
        by_item[lbl.item_id].append(lbl)

    conflicts: list[Conflict] = []
    for item_id, item_labels in sorted(by_item.items()):
        if len(item_labels) < 2:
            continue
        max_diff = 0.0
        for i in range(len(item_labels)):
            for j in range(i + 1, len(item_labels)):
                diff = abs(item_labels[i].label - item_labels[j].label)
                if diff > max_diff:
                    max_diff = diff
        if max_diff > threshold:
            conflicts.append(
                Conflict(
                    item_id=item_id,
                    labels=item_labels,
                    max_disagreement=max_diff,
                    needs_tiebreaker=True,
                )
            )
    return conflicts


# ---------------------------------------------------------------------------
# Resolution strategies
# ---------------------------------------------------------------------------


def resolve_by_majority_vote(conflict: Conflict, threshold: float = 0.5) -> Resolution:
    """Binarize labels (>= threshold -> 1, else 0) and pick majority.

    Returns the majority binary label. Ties go to 1.
    """
    votes: list[float] = []
    for lbl in conflict.labels:
        votes.append(1.0 if lbl.label >= threshold else 0.0)
    ones = sum(v for v in votes)
    zeros = len(votes) - ones
    resolved = 1.0 if ones >= zeros else 0.0
    return Resolution(
        item_id=conflict.item_id,
        resolved_label=resolved,
        method="majority_vote",
        resolved_by="majority",
        original_labels=list(conflict.labels),
    )


def resolve_by_senior_override(conflict: Conflict, senior_ids: list[str]) -> Resolution | None:
    """Use the first matching senior annotator's label.

    Returns None if no senior annotator is present in the conflict labels.
    """
    for lbl in conflict.labels:
        if lbl.annotator_id in senior_ids:
            return Resolution(
                item_id=conflict.item_id,
                resolved_label=lbl.label,
                method="senior_override",
                resolved_by=lbl.annotator_id,
                original_labels=list(conflict.labels),
            )
    return None


def apply_resolution_policy(conflict: Conflict, policy: ResolutionPolicy) -> Resolution | None:
    """Dispatch to the correct resolution method based on policy.

    Returns None if the policy is "discussion_required" (needs human) or if
    senior_override finds no senior annotator.
    """
    if policy.policy_name == "majority_vote":
        return resolve_by_majority_vote(conflict, threshold=policy.min_agreement)
    if policy.policy_name == "senior_override":
        return resolve_by_senior_override(conflict, policy.senior_annotators)
    # "discussion_required" or unknown - cannot auto-resolve
    return None


def resolve_all_conflicts(
    conflicts: list[Conflict], policy: ResolutionPolicy
) -> tuple[list[Resolution], list[Conflict]]:
    """Resolve what can be resolved; return remaining unresolved.

    Returns (resolutions, unresolved_conflicts).
    """
    resolved: list[Resolution] = []
    unresolved: list[Conflict] = []
    for conflict in conflicts:
        result = apply_resolution_policy(conflict, policy)
        if result is not None:
            resolved.append(result)
        else:
            unresolved.append(conflict)
    return resolved, unresolved


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_conflict_view(conflict: Conflict) -> str:
    """Side-by-side text display showing each annotator's label, confidence, reasoning."""
    lines: list[str] = []
    lines.append(f"CONFLICT: {conflict.item_id}")
    lines.append(f"  Max disagreement: {conflict.max_disagreement:.3f}")
    lines.append(f"  Needs tiebreaker: {conflict.needs_tiebreaker}")
    lines.append("")
    header = f"  {'Annotator':<20} {'Label':>8} {'Confidence':>12}  Reasoning"
    lines.append(header)
    lines.append("  " + "-" * 70)
    for lbl in conflict.labels:
        reasoning_str = lbl.reasoning if lbl.reasoning else "-"
        lines.append(
            f"  {lbl.annotator_id:<20} {lbl.label:>8.3f} {lbl.confidence:>12.3f}  {reasoning_str}"
        )
    return "\n".join(lines)


def format_resolution_summary(resolutions: list[Resolution], unresolved: list[Conflict]) -> str:
    """Summary report of resolutions and remaining conflicts."""
    lines: list[str] = []
    total = len(resolutions) + len(unresolved)
    lines.append("RESOLUTION SUMMARY")
    lines.append(f"  Total conflicts: {total}")
    lines.append(f"  Resolved: {len(resolutions)}")
    lines.append(f"  Unresolved: {len(unresolved)}")
    lines.append("")

    if resolutions:
        lines.append("Resolved items:")
        for res in resolutions:
            lines.append(
                f"  {res.item_id}: label={res.resolved_label:.3f} "
                f"method={res.method} by={res.resolved_by}"
            )
        lines.append("")

    if unresolved:
        lines.append("Unresolved items (need discussion):")
        for conflict in unresolved:
            annotators = ", ".join(lbl.annotator_id for lbl in conflict.labels)
            lines.append(
                f"  {conflict.item_id}: disagreement={conflict.max_disagreement:.3f} "
                f"annotators=[{annotators}]"
            )

    return "\n".join(lines)
