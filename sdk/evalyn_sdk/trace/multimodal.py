from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..models import Span

_VALID_MEDIA_TYPES = ["image", "audio", "video", "document"]


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class MediaAttachment:
    """A media file attached to a span."""

    media_type: str  # "image", "audio", "video", "document"
    file_path: str = ""
    mime_type: str = ""
    size_bytes: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "media_type": self.media_type,
            "file_path": self.file_path,
            "mime_type": self.mime_type,
            "size_bytes": self.size_bytes,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MediaAttachment:
        return cls(
            media_type=data["media_type"],
            file_path=data.get("file_path", ""),
            mime_type=data.get("mime_type", ""),
            size_bytes=data.get("size_bytes", 0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class MultimodalSpanData:
    """Multimodal data payload for a span."""

    input_media: List[MediaAttachment] = field(default_factory=list)
    output_media: List[MediaAttachment] = field(default_factory=list)
    text_input: str = ""
    text_output: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "input_media": [m.as_dict() for m in self.input_media],
            "output_media": [m.as_dict() for m in self.output_media],
            "text_input": self.text_input,
            "text_output": self.text_output,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MultimodalSpanData:
        return cls(
            input_media=[
                MediaAttachment.from_dict(m) for m in data.get("input_media", [])
            ],
            output_media=[
                MediaAttachment.from_dict(m) for m in data.get("output_media", [])
            ],
            text_input=data.get("text_input", ""),
            text_output=data.get("text_output", ""),
        )


@dataclass
class MultimodalStats:
    """Aggregate statistics across multimodal spans."""

    total_media: int = 0
    by_type: Dict[str, int] = field(default_factory=dict)
    total_size_bytes: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "total_media": self.total_media,
            "by_type": dict(self.by_type),
            "total_size_bytes": self.total_size_bytes,
        }

    def format_text(self) -> str:
        """Human-readable summary of multimodal stats."""
        lines = [
            f"Total media: {self.total_media}",
            f"By type: {self.by_type}",
            f"Total size bytes: {self.total_size_bytes}",
        ]
        return "\n".join(lines)


def supported_media_types() -> List[str]:
    """Return the list of supported media types."""
    return list(_VALID_MEDIA_TYPES)


def create_media_attachment(
    media_type: str,
    file_path: str = "",
    mime_type: str = "",
    size_bytes: int = 0,
) -> MediaAttachment:
    """Factory with validation for MediaAttachment."""
    if media_type not in _VALID_MEDIA_TYPES:
        raise ValueError(
            f"Invalid media_type {media_type!r}, must be one of {_VALID_MEDIA_TYPES}"
        )
    return MediaAttachment(
        media_type=media_type,
        file_path=file_path,
        mime_type=mime_type,
        size_bytes=size_bytes,
    )


def create_multimodal_span(
    span_id: str,
    text_input: str = "",
    text_output: str = "",
    input_media: List[MediaAttachment] = [],
    output_media: List[MediaAttachment] = [],
) -> Span:
    """Create a span with multimodal data stored in attributes."""
    data = MultimodalSpanData(
        input_media=list(input_media),
        output_media=list(output_media),
        text_input=text_input,
        text_output=text_output,
    )
    return Span(
        id=span_id,
        name="multimodal",
        span_type="llm_call",
        parent_id=None,
        start_time=_now_utc(),
        attributes={"multimodal_data": data.as_dict()},
    )


def extract_multimodal_data(span: Span) -> Optional[MultimodalSpanData]:
    """Extract MultimodalSpanData from span attributes. Returns None if not present."""
    raw = span.attributes.get("multimodal_data")
    if not raw or not isinstance(raw, dict):
        return None
    return MultimodalSpanData.from_dict(raw)


def has_media(span: Span) -> bool:
    """Quick check whether a span contains any media attachments."""
    data = extract_multimodal_data(span)
    if data is None:
        return False
    return len(data.input_media) > 0 or len(data.output_media) > 0


def compute_multimodal_stats(spans: List[Span]) -> MultimodalStats:
    """Aggregate media statistics across spans."""
    total_media = 0
    by_type: Dict[str, int] = {}
    total_size_bytes = 0

    for s in spans:
        data = extract_multimodal_data(s)
        if data is None:
            continue
        all_media = list(data.input_media) + list(data.output_media)
        total_media += len(all_media)
        for m in all_media:
            by_type[m.media_type] = by_type.get(m.media_type, 0) + 1
            total_size_bytes += m.size_bytes

    return MultimodalStats(
        total_media=total_media,
        by_type=by_type,
        total_size_bytes=total_size_bytes,
    )
