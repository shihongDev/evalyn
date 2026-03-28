"""
Team collaboration: multi-user annotation with conflict resolution.

Provides dataclasses and pure functions for assigning items to annotators,
detecting conflicts, resolving disagreements by majority vote, computing
inter-annotator agreement, and building collaboration reports.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class Annotator:
    """A user who annotates evaluation items."""

    user_id: str = ""
    name: str = ""
    role: str = "annotator"  # "annotator" / "reviewer" / "admin"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "name": self.name,
            "role": self.role,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Annotator:
        return cls(
            user_id=data.get("user_id", ""),
            name=data.get("name", ""),
            role=data.get("role", "annotator"),
        )


@dataclass
class Annotation:
    """A single annotation for an evaluation item."""

    item_id: str = ""
    annotator_id: str = ""
    label: str = ""
    confidence: float = 1.0
    timestamp: str = ""
    notes: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "annotator_id": self.annotator_id,
            "label": self.label,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Annotation:
        return cls(
            item_id=data.get("item_id", ""),
            annotator_id=data.get("annotator_id", ""),
            label=data.get("label", ""),
            confidence=data.get("confidence", 1.0),
            timestamp=data.get("timestamp", ""),
            notes=data.get("notes", ""),
        )


@dataclass
class ConflictResolution:
    """Resolution of a labeling conflict for a single item."""

    item_id: str = ""
    annotations: List[Annotation] = field(default_factory=list)
    resolved_label: str = ""
    resolution_method: str = ""  # "majority" / "reviewer" / "discussion"
    resolved_by: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "annotations": [a.as_dict() for a in self.annotations],
            "resolved_label": self.resolved_label,
            "resolution_method": self.resolution_method,
            "resolved_by": self.resolved_by,
        }


@dataclass
class CollaborationReport:
    """Summary report for a team annotation effort."""

    total_items: int = 0
    annotated_items: int = 0
    conflicts: int = 0
    resolved: int = 0
    agreement_rate: float = 0.0
    annotators: List[Annotator] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "total_items": self.total_items,
            "annotated_items": self.annotated_items,
            "conflicts": self.conflicts,
            "resolved": self.resolved,
            "agreement_rate": self.agreement_rate,
            "annotators": [a.as_dict() for a in self.annotators],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CollaborationReport:
        return cls(
            total_items=data.get("total_items", 0),
            annotated_items=data.get("annotated_items", 0),
            conflicts=data.get("conflicts", 0),
            resolved=data.get("resolved", 0),
            agreement_rate=data.get("agreement_rate", 0.0),
            annotators=[
                Annotator.from_dict(a) for a in data.get("annotators", [])
            ],
        )

    def format_text(self) -> str:
        lines = [
            "Collaboration Report",
            f"  total_items: {self.total_items}",
            f"  annotated_items: {self.annotated_items}",
            f"  conflicts: {self.conflicts}",
            f"  resolved: {self.resolved}",
            f"  agreement_rate: {self.agreement_rate:.2f}",
            f"  annotators: {len(self.annotators)}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def assign_items(
    items: List[str],
    annotators: List[Annotator],
    overlap: int = 2,
) -> Dict[str, List[str]]:
    """Assign items to annotators with overlap using round-robin.

    Each item is assigned to ``overlap`` annotators. The assignment cycles
    through the annotator list so that work is distributed evenly.

    Returns:
        Mapping of annotator_id to list of item_ids.
    """
    if not annotators:
        return {}

    assignments: Dict[str, List[str]] = {a.user_id: [] for a in annotators}
    n = len(annotators)

    for idx, item_id in enumerate(items):
        for k in range(min(overlap, n)):
            annotator = annotators[(idx + k) % n]
            assignments[annotator.user_id].append(item_id)

    return assignments


def detect_conflicts(annotations: List[Annotation]) -> List[str]:
    """Find item_ids where annotators disagree on the label.

    Returns a sorted list of item_ids with conflicting labels.
    """
    items_labels: Dict[str, set] = {}
    for ann in annotations:
        items_labels.setdefault(ann.item_id, set()).add(ann.label)

    return sorted(iid for iid, labels in items_labels.items() if len(labels) > 1)


def resolve_by_majority(
    annotations: List[Annotation],
    item_id: str,
) -> ConflictResolution:
    """Resolve a conflict by majority vote.

    Picks the label that appears most frequently for the given item.
    """
    relevant = [a for a in annotations if a.item_id == item_id]
    label_counts = Counter(a.label for a in relevant)

    if label_counts:
        winner = label_counts.most_common(1)[0][0]
    else:
        winner = ""

    return ConflictResolution(
        item_id=item_id,
        annotations=relevant,
        resolved_label=winner,
        resolution_method="majority",
        resolved_by="system",
    )


def compute_inter_annotator_agreement(annotations: List[Annotation]) -> float:
    """Compute Cohen's kappa approximation for inter-annotator agreement.

    Formula: (agreement - expected) / (1 - expected)

    Agreement is the fraction of items where all annotators gave the same
    label. Expected agreement is the sum of squared label proportions.
    """
    # Group annotations by item
    items_labels: Dict[str, List[str]] = {}
    for ann in annotations:
        items_labels.setdefault(ann.item_id, []).append(ann.label)

    if not items_labels:
        return 0.0

    # Only consider items with multiple annotations
    multi = {iid: labels for iid, labels in items_labels.items() if len(labels) > 1}
    if not multi:
        return 1.0

    # Observed agreement: fraction of items where all labels match
    agree_count = sum(1 for labels in multi.values() if len(set(labels)) == 1)
    agreement = agree_count / len(multi)

    # Expected agreement: sum of squared proportions of each label
    all_labels: List[str] = []
    for labels in multi.values():
        all_labels.extend(labels)

    total = len(all_labels)
    if total == 0:
        return 0.0

    label_counts = Counter(all_labels)
    expected = sum((c / total) ** 2 for c in label_counts.values())

    if expected >= 1.0:
        return 1.0

    return (agreement - expected) / (1.0 - expected)


def build_collaboration_report(
    annotators: List[Annotator],
    annotations: List[Annotation],
) -> CollaborationReport:
    """Build a full collaboration report from annotators and annotations."""
    item_ids = set(a.item_id for a in annotations)
    conflicts = detect_conflicts(annotations)
    agreement = compute_inter_annotator_agreement(annotations)

    return CollaborationReport(
        total_items=len(item_ids),
        annotated_items=len(item_ids),
        conflicts=len(conflicts),
        resolved=0,
        agreement_rate=agreement,
        annotators=list(annotators),
    )
