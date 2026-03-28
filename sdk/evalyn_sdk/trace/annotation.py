from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt else None


def _parse_datetime(raw: Optional[str]) -> Optional[datetime]:
    if raw is None:
        return None
    return datetime.fromisoformat(raw)


@dataclass
class TraceAnnotation:
    """A post-hoc annotation attached to a trace span."""

    span_id: str
    notes: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def add_note(self, note: str) -> None:
        """Append a note and update the timestamp."""
        self.notes.append(note)
        self.updated_at = datetime.now(timezone.utc)

    def add_tag(self, tag: str) -> None:
        """Append a tag if not already present and update the timestamp."""
        if tag not in self.tags:
            self.tags.append(tag)
            self.updated_at = datetime.now(timezone.utc)

    def remove_tag(self, tag: str) -> bool:
        """Remove a tag. Returns True if the tag was found and removed."""
        if tag in self.tags:
            self.tags.remove(tag)
            self.updated_at = datetime.now(timezone.utc)
            return True
        return False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "span_id": self.span_id,
            "notes": list(self.notes),
            "tags": list(self.tags),
            "created_at": _iso(self.created_at),
            "updated_at": _iso(self.updated_at),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TraceAnnotation:
        return cls(
            span_id=data["span_id"],
            notes=data.get("notes", []),
            tags=data.get("tags", []),
            created_at=_parse_datetime(data.get("created_at"))
            or datetime.now(timezone.utc),
            updated_at=_parse_datetime(data.get("updated_at"))
            or datetime.now(timezone.utc),
        )


@dataclass
class AnnotationStore:
    """A keyed collection of trace annotations."""

    annotations: Dict[str, TraceAnnotation] = field(default_factory=dict)

    def annotate(
        self,
        span_id: str,
        note: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> TraceAnnotation:
        """Create or update an annotation for the given span."""
        if span_id not in self.annotations:
            self.annotations[span_id] = TraceAnnotation(span_id=span_id)
        ann = self.annotations[span_id]
        if note is not None:
            ann.add_note(note)
        if tag is not None:
            ann.add_tag(tag)
        return ann

    def get(self, span_id: str) -> Optional[TraceAnnotation]:
        """Look up an annotation by span_id."""
        return self.annotations.get(span_id)

    def find_by_tag(self, tag: str) -> List[TraceAnnotation]:
        """Return all annotations that carry the given tag."""
        return [a for a in self.annotations.values() if tag in a.tags]

    def find_by_note_keyword(self, keyword: str) -> List[TraceAnnotation]:
        """Return annotations where any note contains the keyword."""
        return [
            a
            for a in self.annotations.values()
            if any(keyword in n for n in a.notes)
        ]

    def remove(self, span_id: str) -> bool:
        """Delete an annotation. Returns True if it existed."""
        if span_id in self.annotations:
            del self.annotations[span_id]
            return True
        return False

    @property
    def count(self) -> int:
        return len(self.annotations)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "annotations": {
                sid: ann.as_dict() for sid, ann in self.annotations.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AnnotationStore:
        annotations = {
            sid: TraceAnnotation.from_dict(ad)
            for sid, ad in data.get("annotations", {}).items()
        }
        return cls(annotations=annotations)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.as_dict())

    @classmethod
    def from_json(cls, data: str) -> AnnotationStore:
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(data))
