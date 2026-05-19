"""
Annotation session persistence: save and resume annotation progress.
Pure Python, no external dependencies.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class AnnotationRecord:
    """A single annotation recorded during a session."""

    item_id: str
    label: float
    annotator_id: str
    timestamp: str  # ISO format
    confidence: float = 1.0
    skipped: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "label": self.label,
            "annotator_id": self.annotator_id,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
            "skipped": self.skipped,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AnnotationRecord:
        return cls(
            item_id=data["item_id"],
            label=data["label"],
            annotator_id=data["annotator_id"],
            timestamp=data["timestamp"],
            confidence=data.get("confidence", 1.0),
            skipped=data.get("skipped", False),
        )


@dataclass
class SessionStats:
    """Statistics for an annotation session."""

    total_items: int
    annotated_count: int
    skipped_count: int
    remaining_count: int
    items_per_hour: float
    elapsed_seconds: float
    agreement_rate: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "total_items": self.total_items,
            "annotated_count": self.annotated_count,
            "skipped_count": self.skipped_count,
            "remaining_count": self.remaining_count,
            "items_per_hour": self.items_per_hour,
            "elapsed_seconds": self.elapsed_seconds,
            "agreement_rate": self.agreement_rate,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SessionStats:
        return cls(
            total_items=data["total_items"],
            annotated_count=data["annotated_count"],
            skipped_count=data["skipped_count"],
            remaining_count=data["remaining_count"],
            items_per_hour=data["items_per_hour"],
            elapsed_seconds=data["elapsed_seconds"],
            agreement_rate=data.get("agreement_rate", 0.0),
        )


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


class AnnotationSession:
    """Manages annotation progress with save/resume support."""

    def __init__(
        self,
        session_id: str,
        annotator_id: str,
        item_ids: List[str],
        session_path: str = "",
    ) -> None:
        self.session_id = session_id
        self.annotator_id = annotator_id
        self.item_ids = list(item_ids)
        self.session_path = session_path
        self._records: List[AnnotationRecord] = []
        self._start_time: str = _now_iso()

    # -- annotation actions --------------------------------------------------

    def annotate(self, item_id: str, label: float, confidence: float = 1.0) -> None:
        """Record an annotation for the given item."""
        record = AnnotationRecord(
            item_id=item_id,
            label=label,
            annotator_id=self.annotator_id,
            timestamp=_now_iso(),
            confidence=confidence,
            skipped=False,
        )
        self._records.append(record)

    def skip(self, item_id: str) -> None:
        """Mark an item as skipped."""
        record = AnnotationRecord(
            item_id=item_id,
            label=0.0,
            annotator_id=self.annotator_id,
            timestamp=_now_iso(),
            confidence=0.0,
            skipped=True,
        )
        self._records.append(record)

    def undo_last(self) -> Optional[AnnotationRecord]:
        """Undo and return the last annotation. Returns None if empty."""
        if not self._records:
            return None
        return self._records.pop()

    # -- navigation ----------------------------------------------------------

    def get_next_item(self) -> Optional[str]:
        """Return next item_id that has not been annotated or skipped."""
        touched = {r.item_id for r in self._records}
        for item_id in self.item_ids:
            if item_id not in touched:
                return item_id
        return None

    # -- progress ------------------------------------------------------------

    def get_progress(self) -> SessionStats:
        """Current progress stats."""
        return compute_session_stats(self)

    def get_annotations(self) -> List[AnnotationRecord]:
        """All recorded annotations (including skipped)."""
        return list(self._records)

    def is_complete(self) -> bool:
        """True if every item has been annotated or skipped."""
        touched = {r.item_id for r in self._records}
        return all(item_id in touched for item_id in self.item_ids)

    # -- persistence ---------------------------------------------------------

    def save(self, path: str = "") -> None:
        """Save session to JSON file. Uses session_path if path is empty."""
        target = path or self.session_path
        if not target:
            raise ValueError("No path provided and session_path is empty")
        data = {
            "session_id": self.session_id,
            "annotator_id": self.annotator_id,
            "item_ids": self.item_ids,
            "session_path": self.session_path,
            "start_time": self._start_time,
            "records": [r.as_dict() for r in self._records],
        }
        with open(target, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> AnnotationSession:
        """Load session from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        session = cls(
            session_id=data["session_id"],
            annotator_id=data["annotator_id"],
            item_ids=data["item_ids"],
            session_path=data.get("session_path", ""),
        )
        session._start_time = data.get("start_time", session._start_time)
        session._records = [
            AnnotationRecord.from_dict(d) for d in data.get("records", [])
        ]
        return session


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------


def compute_session_stats(session: AnnotationSession) -> SessionStats:
    """Compute detailed stats for a session."""
    # Deduplicate by item_id - keep last record for each item
    last_by_item: Dict[str, AnnotationRecord] = {}
    for r in session._records:
        last_by_item[r.item_id] = r
    unique_records = list(last_by_item.values())
    annotated = [r for r in unique_records if not r.skipped]
    skipped = [r for r in unique_records if r.skipped]
    touched = set(last_by_item.keys())
    remaining = sum(1 for iid in session.item_ids if iid not in touched)

    # Elapsed time
    start_dt = datetime.fromisoformat(session._start_time)
    now_dt = datetime.now(timezone.utc)
    elapsed = max((now_dt - start_dt).total_seconds(), 0.0)

    # Items per hour (annotated only, not skipped)
    if elapsed > 0:
        items_per_hour = len(annotated) / (elapsed / 3600.0)
    else:
        items_per_hour = 0.0

    return SessionStats(
        total_items=len(session.item_ids),
        annotated_count=len(annotated),
        skipped_count=len(skipped),
        remaining_count=remaining,
        items_per_hour=items_per_hour,
        elapsed_seconds=elapsed,
        agreement_rate=0.0,
    )


def format_session_progress(stats: SessionStats) -> str:
    """Progress bar and stats display."""
    lines: List[str] = []
    total = stats.total_items
    done = stats.annotated_count + stats.skipped_count
    pct = (done / total * 100) if total > 0 else 0.0

    # Progress bar (40 chars wide)
    bar_width = 40
    filled = int(bar_width * done / total) if total > 0 else 0
    bar = "#" * filled + "-" * (bar_width - filled)
    lines.append(f"[{bar}] {pct:.1f}%")
    lines.append(f"  Annotated: {stats.annotated_count}")
    lines.append(f"  Skipped:   {stats.skipped_count}")
    lines.append(f"  Remaining: {stats.remaining_count}")
    lines.append(f"  Speed:     {stats.items_per_hour:.1f} items/hour")
    lines.append(f"  Elapsed:   {stats.elapsed_seconds:.0f}s")
    if stats.agreement_rate > 0:
        lines.append(f"  Agreement: {stats.agreement_rate:.1%}")
    return "\n".join(lines)
