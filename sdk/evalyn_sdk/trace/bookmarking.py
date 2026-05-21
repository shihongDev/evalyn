from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


def _parse_datetime(raw: str | None) -> datetime | None:
    if raw is None:
        return None
    return datetime.fromisoformat(raw)


@dataclass
class Bookmark:
    """A bookmark marking an interesting span for later review."""

    span_id: str
    label: str = ""
    notes: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tags: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "label": self.label,
            "notes": self.notes,
            "created_at": _iso(self.created_at),
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Bookmark:
        return cls(
            span_id=data["span_id"],
            label=data.get("label", ""),
            notes=data.get("notes", ""),
            created_at=_parse_datetime(data.get("created_at")) or datetime.now(timezone.utc),
            tags=data.get("tags", []),
        )


@dataclass
class BookmarkCollection:
    """A collection of bookmarks with lookup and filtering."""

    bookmarks: list[Bookmark] = field(default_factory=list)

    def add(
        self,
        span_id: str,
        label: str = "",
        notes: str = "",
        tags: list[str] | None = None,
    ) -> Bookmark:
        """Add a bookmark. If span_id already exists, replace it."""
        self.remove(span_id)
        bm = Bookmark(
            span_id=span_id,
            label=label,
            notes=notes,
            tags=tags or [],
        )
        self.bookmarks.append(bm)
        return bm

    def remove(self, span_id: str) -> bool:
        """Remove bookmark by span_id. Returns True if found and removed."""
        before = len(self.bookmarks)
        self.bookmarks = [b for b in self.bookmarks if b.span_id != span_id]
        return len(self.bookmarks) < before

    def get(self, span_id: str) -> Bookmark | None:
        """Look up bookmark by span_id."""
        for b in self.bookmarks:
            if b.span_id == span_id:
                return b
        return None

    def list_by_tag(self, tag: str) -> list[Bookmark]:
        """Filter bookmarks that have the given tag."""
        return [b for b in self.bookmarks if tag in b.tags]

    def list_by_label(self, label: str) -> list[Bookmark]:
        """Filter bookmarks that match the given label."""
        return [b for b in self.bookmarks if b.label == label]

    @property
    def count(self) -> int:
        return len(self.bookmarks)

    def as_dict(self) -> dict[str, Any]:
        return {"bookmarks": [b.as_dict() for b in self.bookmarks]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BookmarkCollection:
        return cls(
            bookmarks=[Bookmark.from_dict(b) for b in data.get("bookmarks", [])],
        )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.as_dict())

    @classmethod
    def from_json(cls, data: str) -> BookmarkCollection:
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(data))
