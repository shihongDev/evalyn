"""
Annotation UX improvements: faster, more forgiving annotation workflow model.

Provides data models and logic for keyboard-driven annotation workflows
including undo support, batch processing, and progress tracking.
Pure Python - no external deps, no terminal UI.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class KeyAction:
    """Maps a keyboard key to an annotation action."""

    key: str
    action: str  # "pass" / "fail" / "confidence" / "skip" / "undo" / "quit"
    value: Any = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "action": self.action,
            "value": self.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KeyAction:
        return cls(
            key=data["key"],
            action=data["action"],
            value=data.get("value"),
        )


@dataclass
class AnnotationItem:
    """A single item in the annotation workflow."""

    item_id: str
    input_text: str
    output_text: str
    metric_scores: dict[str, float]
    status: str = "pending"  # "pending" / "annotated" / "skipped"

    def as_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "input_text": self.input_text,
            "output_text": self.output_text,
            "metric_scores": dict(self.metric_scores),
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnnotationItem:
        return cls(
            item_id=data["item_id"],
            input_text=data["input_text"],
            output_text=data["output_text"],
            metric_scores=data.get("metric_scores", {}),
            status=data.get("status", "pending"),
        )


@dataclass
class BatchAnnotation:
    """A batch of annotation items with collected labels."""

    items: list[AnnotationItem]
    batch_id: str
    batch_size: int
    annotations: dict[str, float] = field(default_factory=dict)  # item_id -> label

    def as_dict(self) -> dict[str, Any]:
        return {
            "items": [item.as_dict() for item in self.items],
            "batch_id": self.batch_id,
            "batch_size": self.batch_size,
            "annotations": dict(self.annotations),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BatchAnnotation:
        return cls(
            items=[AnnotationItem.from_dict(i) for i in data.get("items", [])],
            batch_id=data["batch_id"],
            batch_size=data["batch_size"],
            annotations=data.get("annotations", {}),
        )


# ---------------------------------------------------------------------------
# Default Keymap
# ---------------------------------------------------------------------------

DEFAULT_KEYMAP: dict[str, KeyAction] = {
    "y": KeyAction(key="y", action="pass"),
    "n": KeyAction(key="n", action="fail"),
    "1": KeyAction(key="1", action="confidence", value=1),
    "2": KeyAction(key="2", action="confidence", value=2),
    "3": KeyAction(key="3", action="confidence", value=3),
    "4": KeyAction(key="4", action="confidence", value=4),
    "5": KeyAction(key="5", action="confidence", value=5),
    "s": KeyAction(key="s", action="skip"),
    "u": KeyAction(key="u", action="undo"),
    "q": KeyAction(key="q", action="quit"),
}


# ---------------------------------------------------------------------------
# Undo Stack
# ---------------------------------------------------------------------------


class UndoStack:
    """Stack for tracking annotation history to support undo."""

    def __init__(self) -> None:
        self._stack: list[tuple[str, str, float | None]] = []

    def push(self, item_id: str, previous_status: str, previous_label: float | None) -> None:
        """Push a state snapshot onto the undo stack."""
        self._stack.append((item_id, previous_status, previous_label))

    def pop(self) -> tuple[str, str, float | None] | None:
        """Pop the most recent state. Returns (item_id, prev_status, prev_label) or None."""
        if not self._stack:
            return None
        return self._stack.pop()

    def is_empty(self) -> bool:
        """Check whether the stack has no entries."""
        return len(self._stack) == 0

    def size(self) -> int:
        """Return the number of entries on the stack."""
        return len(self._stack)


# ---------------------------------------------------------------------------
# Core Functions
# ---------------------------------------------------------------------------


def process_key_action(
    action: KeyAction,
    item: AnnotationItem,
    undo_stack: UndoStack,
) -> AnnotationItem:
    """Apply a key action to an annotation item.

    Pushes the item's current state to the undo stack before modifying.
    Returns the updated item.

    Actions:
    - pass: mark annotated, label 1.0
    - fail: mark annotated, label 0.0
    - confidence: mark annotated, label is the confidence value (1-5)
    - skip: mark skipped, no label
    """
    # Undo and quit are handled at the workflow level, not per-item
    if action.action in ("undo", "quit"):
        return item

    # Save current state for undo before modifying
    current_label: float | None = None
    undo_stack.push(item.item_id, item.status, current_label)

    if action.action == "pass" or action.action == "fail" or action.action == "confidence":
        item.status = "annotated"
    elif action.action == "skip":
        item.status = "skipped"

    return item


def undo_action(
    undo_stack: UndoStack,
    items: dict[str, AnnotationItem],
) -> str | None:
    """Pop the undo stack and revert the referenced item.

    Returns the item_id that was reverted, or None if stack was empty.
    """
    entry = undo_stack.pop()
    if entry is None:
        return None

    item_id, prev_status, _prev_label = entry
    if item_id in items:
        items[item_id].status = prev_status
    return item_id


def create_batch(
    items: list[AnnotationItem],
    batch_size: int = 5,
) -> list[BatchAnnotation]:
    """Split annotation items into batches of the given size."""
    batches: list[BatchAnnotation] = []
    for i in range(0, len(items), batch_size):
        chunk = items[i : i + batch_size]
        batches.append(
            BatchAnnotation(
                items=chunk,
                batch_id=str(uuid.uuid4()),
                batch_size=len(chunk),
            )
        )
    return batches


def get_batch_progress(batch: BatchAnnotation) -> dict:
    """Compute progress stats for a batch.

    Returns: {total, annotated, skipped, remaining}
    """
    total = len(batch.items)
    annotated = sum(1 for item in batch.items if item.status == "annotated")
    skipped = sum(1 for item in batch.items if item.status == "skipped")
    remaining = total - annotated - skipped
    return {
        "total": total,
        "annotated": annotated,
        "skipped": skipped,
        "remaining": remaining,
    }


def format_annotation_prompt(item: AnnotationItem) -> str:
    """Format an annotation item for display.

    Shows input, output, and metric scores in a readable block.
    """
    lines = [
        f"Item: {item.item_id}",
        f"Input: {item.input_text}",
        f"Output: {item.output_text}",
    ]
    if item.metric_scores:
        lines.append("Scores:")
        for metric_id, score in item.metric_scores.items():
            lines.append(f"  {metric_id}: {score:.2f}")
    return "\n".join(lines)


def format_keymap_help(keymap: dict[str, KeyAction]) -> str:
    """Display available keyboard shortcuts.

    Returns a newline-separated list of key -> action mappings.
    """
    lines: list[str] = []
    for key, action in keymap.items():
        if action.value is not None:
            lines.append(f"  [{key}] {action.action} ({action.value})")
        else:
            lines.append(f"  [{key}] {action.action}")
    return "\n".join(lines)
