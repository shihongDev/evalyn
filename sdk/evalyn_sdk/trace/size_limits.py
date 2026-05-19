from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from ..models import Span


@dataclass
class SizeLimitConfig:
    """Configuration for span payload size limits."""

    default_max_bytes: int = 10000
    per_type_limits: dict[str, int] = field(default_factory=dict)
    truncation_marker: str = "... [truncated]"

    def as_dict(self) -> dict[str, Any]:
        return {
            "default_max_bytes": self.default_max_bytes,
            "per_type_limits": dict(self.per_type_limits),
            "truncation_marker": self.truncation_marker,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SizeLimitConfig:
        return cls(
            default_max_bytes=data.get("default_max_bytes", 10000),
            per_type_limits=data.get("per_type_limits", {}),
            truncation_marker=data.get("truncation_marker", "... [truncated]"),
        )


@dataclass
class TruncationResult:
    """Result of a truncation operation on a single field."""

    field_name: str
    original_bytes: int
    truncated_bytes: int
    was_truncated: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "field_name": self.field_name,
            "original_bytes": self.original_bytes,
            "truncated_bytes": self.truncated_bytes,
            "was_truncated": self.was_truncated,
        }


def get_limit(config: SizeLimitConfig, span_type: str) -> int:
    """Return the byte limit for a span type.

    Uses per_type_limits if available, otherwise default_max_bytes.
    """
    return config.per_type_limits.get(span_type, config.default_max_bytes)


def truncate_text(
    text: str, max_bytes: int, marker: str
) -> tuple[str, bool]:
    """Truncate text to fit within max_bytes (UTF-8).

    Appends marker if truncated. Truncates at character boundaries
    to avoid splitting multi-byte characters.

    Returns (result_text, was_truncated).
    """
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text, False

    marker_bytes = len(marker.encode("utf-8"))
    target_bytes = max_bytes - marker_bytes
    if target_bytes <= 0:
        # Not enough room for any content plus the marker
        return marker[:max_bytes] if max_bytes > 0 else "", True

    # Truncate at character boundary by decoding with errors='ignore'
    truncated = encoded[:target_bytes].decode("utf-8", errors="ignore")
    return truncated + marker, True


def enforce_size_limits(
    span: Span, config: SizeLimitConfig
) -> tuple[Span, list[TruncationResult]]:
    """Apply size limits to span string attributes "input" and "output".

    Returns a NEW span (does not mutate the original) and a list of
    truncation results for each field checked.
    """
    limit = get_limit(config, span.span_type)
    new_span = copy.deepcopy(span)
    results: list[TruncationResult] = []

    for field_name in ("input", "output"):
        value = new_span.attributes.get(field_name)
        if not isinstance(value, str):
            continue

        original_bytes = len(value.encode("utf-8"))
        truncated_value, was_truncated = truncate_text(
            value, limit, config.truncation_marker
        )
        new_span.attributes[field_name] = truncated_value
        truncated_bytes = len(truncated_value.encode("utf-8"))

        results.append(
            TruncationResult(
                field_name=field_name,
                original_bytes=original_bytes,
                truncated_bytes=truncated_bytes,
                was_truncated=was_truncated,
            )
        )

    return new_span, results


def default_config() -> SizeLimitConfig:
    """Return a SizeLimitConfig with sensible defaults."""
    return SizeLimitConfig(
        default_max_bytes=10000,
        per_type_limits={
            "llm_call": 50000,
            "tool_call": 5000,
        },
        truncation_marker="... [truncated]",
    )
