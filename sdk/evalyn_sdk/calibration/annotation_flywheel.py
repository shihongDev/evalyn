"""Annotation queue flywheel.

Closed loop where human labels feed back to improve the AI judge.
Manages annotation queues, tracks agreement, and determines when
to stop iterating.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AnnotationItem:
    """A single item in the annotation queue."""

    item_id: str
    text: str
    ai_label: str = ""
    human_label: str = ""
    agreement: bool = False
    annotated: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "text": self.text,
            "ai_label": self.ai_label,
            "human_label": self.human_label,
            "agreement": self.agreement,
            "annotated": self.annotated,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnnotationItem:
        return cls(
            item_id=data["item_id"],
            text=data["text"],
            ai_label=data.get("ai_label", ""),
            human_label=data.get("human_label", ""),
            agreement=data.get("agreement", False),
            annotated=data.get("annotated", False),
        )


@dataclass
class FlywheelState:
    """Current state of the annotation flywheel."""

    queue: list[AnnotationItem] = field(default_factory=list)
    annotated_count: int = 0
    agreement_rate: float = 0.0
    improvement_per_batch: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "queue": [item.as_dict() for item in self.queue],
            "annotated_count": self.annotated_count,
            "agreement_rate": self.agreement_rate,
            "improvement_per_batch": list(self.improvement_per_batch),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FlywheelState:
        return cls(
            queue=[AnnotationItem.from_dict(i) for i in data.get("queue", [])],
            annotated_count=data.get("annotated_count", 0),
            agreement_rate=data.get("agreement_rate", 0.0),
            improvement_per_batch=data.get("improvement_per_batch", []),
        )

    def format_text(self) -> str:
        lines = [
            "Flywheel State",
            f"  Queue size: {len(self.queue)}",
            f"  Annotated: {self.annotated_count}",
            f"  Agreement rate: {self.agreement_rate:.4f}",
            f"  Batches completed: {len(self.improvement_per_batch)}",
        ]
        for i, imp in enumerate(self.improvement_per_batch):
            lines.append(f"  Batch {i + 1}: {imp:+.4f}")
        return "\n".join(lines)


@dataclass
class FlywheelConfig:
    """Configuration for the annotation flywheel."""

    batch_size: int = 10
    target_agreement: float = 0.9
    max_iterations: int = 10

    def as_dict(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "target_agreement": self.target_agreement,
            "max_iterations": self.max_iterations,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FlywheelConfig:
        return cls(
            batch_size=data.get("batch_size", 10),
            target_agreement=data.get("target_agreement", 0.9),
            max_iterations=data.get("max_iterations", 10),
        )


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def create_annotation_queue(
    items: list[dict[str, Any]], ai_labels: dict[str, str]
) -> list[AnnotationItem]:
    """Build an annotation queue from raw items and AI predictions.

    Each item dict must have "id" and "text" keys.
    """
    queue: list[AnnotationItem] = []
    for item in items:
        item_id = item["id"]
        queue.append(
            AnnotationItem(
                item_id=item_id,
                text=item["text"],
                ai_label=ai_labels.get(item_id, ""),
            )
        )
    return queue


def record_human_annotation(
    item: AnnotationItem, human_label: str
) -> AnnotationItem:
    """Record a human annotation and compute agreement with the AI label."""
    agrees = item.ai_label == human_label
    return AnnotationItem(
        item_id=item.item_id,
        text=item.text,
        ai_label=item.ai_label,
        human_label=human_label,
        agreement=agrees,
        annotated=True,
    )


def compute_agreement_rate(items: list[AnnotationItem]) -> float:
    """Compute fraction of annotated items where AI and human labels agree."""
    annotated = [i for i in items if i.annotated]
    if not annotated:
        return 0.0
    return sum(1 for i in annotated if i.agreement) / len(annotated)


def select_next_batch(
    queue: list[AnnotationItem], batch_size: int = 10
) -> list[AnnotationItem]:
    """Select the next batch of items to annotate.

    Prioritizes unannotated items with disagreements (previously annotated
    items that disagreed are re-eligible only if still unannotated).
    Among unannotated items, those without an AI label come first,
    then the rest in queue order.
    """
    unannotated = [i for i in queue if not i.annotated]
    return unannotated[:batch_size]


def should_continue(state: FlywheelState, config: FlywheelConfig) -> bool:
    """Determine whether the flywheel should keep iterating.

    Returns True if agreement is below target and we have not exceeded
    the maximum number of iterations.
    """
    iterations = len(state.improvement_per_batch)
    if iterations >= config.max_iterations:
        return False
    if state.agreement_rate >= config.target_agreement:
        return False
    return True


def update_flywheel_state(
    state: FlywheelState, newly_annotated: list[AnnotationItem]
) -> FlywheelState:
    """Update flywheel state after a batch of annotations.

    Replaces matching items in the queue, recomputes agreement rate,
    and records improvement.
    """
    lookup = {i.item_id: i for i in newly_annotated}
    updated_queue: list[AnnotationItem] = []
    for item in state.queue:
        if item.item_id in lookup:
            updated_queue.append(lookup[item.item_id])
        else:
            updated_queue.append(item)

    new_rate = compute_agreement_rate(updated_queue)
    improvement = new_rate - state.agreement_rate
    annotated_count = sum(1 for i in updated_queue if i.annotated)

    return FlywheelState(
        queue=updated_queue,
        annotated_count=annotated_count,
        agreement_rate=new_rate,
        improvement_per_batch=list(state.improvement_per_batch) + [improvement],
    )
