"""Trace compression - compress span payloads before storage, transparent decompression on read."""

from __future__ import annotations

import base64
import copy
import gzip
from dataclasses import dataclass
from typing import Any

from ..models import Span


@dataclass
class CompressionConfig:
    """Configuration for span attribute compression."""

    min_size_bytes: int = 1000
    level: int = 6
    enabled: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "min_size_bytes": self.min_size_bytes,
            "level": self.level,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CompressionConfig:
        return cls(
            min_size_bytes=data.get("min_size_bytes", 1000),
            level=data.get("level", 6),
            enabled=data.get("enabled", True),
        )


@dataclass
class CompressionStats:
    """Summary of compression results."""

    original_bytes: int
    compressed_bytes: int
    compression_ratio: float
    fields_compressed: int

    @property
    def savings_pct(self) -> float:
        if self.original_bytes > 0:
            return (1 - self.compressed_bytes / self.original_bytes) * 100
        return 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "original_bytes": self.original_bytes,
            "compressed_bytes": self.compressed_bytes,
            "compression_ratio": self.compression_ratio,
            "fields_compressed": self.fields_compressed,
            "savings_pct": self.savings_pct,
        }


_COMPRESS_KEYS = {"input", "output"}


def compress_text(text: str, level: int = 6) -> bytes:
    """Gzip compress a string. Returns compressed bytes."""
    return gzip.compress(text.encode("utf-8"), compresslevel=level)


def decompress_text(data: bytes) -> str:
    """Gzip decompress bytes back to string."""
    return gzip.decompress(data).decode("utf-8")


def compress_span_attributes(
    span: Span, config: CompressionConfig
) -> tuple[Span, CompressionStats]:
    """Compress string attributes that exceed min_size_bytes.

    Replaces qualifying attribute values with a dict containing:
        {"__compressed__": True, "data": base64_encoded, "original_size": int}

    Returns a NEW span (does not mutate the original) and compression stats.
    """
    new_span = copy.deepcopy(span)
    original_bytes = 0
    compressed_bytes = 0
    fields_compressed = 0

    if not config.enabled:
        return new_span, CompressionStats(
            original_bytes=0,
            compressed_bytes=0,
            compression_ratio=1.0,
            fields_compressed=0,
        )

    for key in list(new_span.attributes.keys()):
        value = new_span.attributes[key]
        if not isinstance(value, str):
            continue
        raw_bytes = len(value.encode("utf-8"))
        if raw_bytes < config.min_size_bytes:
            continue

        original_bytes += raw_bytes
        compressed = compress_text(value, level=config.level)
        compressed_bytes += len(compressed)
        fields_compressed += 1

        encoded = base64.b64encode(compressed).decode("ascii")
        new_span.attributes[key] = {
            "__compressed__": True,
            "data": encoded,
            "original_size": raw_bytes,
        }

    ratio = compressed_bytes / original_bytes if original_bytes > 0 else 1.0

    stats = CompressionStats(
        original_bytes=original_bytes,
        compressed_bytes=compressed_bytes,
        compression_ratio=ratio,
        fields_compressed=fields_compressed,
    )
    return new_span, stats


def decompress_span_attributes(span: Span) -> Span:
    """Decompress compressed attributes in a span.

    Looks for {"__compressed__": True} markers and restores the original text.
    Returns a NEW span (does not mutate the original).
    """
    new_span = copy.deepcopy(span)

    for key in list(new_span.attributes.keys()):
        value = new_span.attributes[key]
        if not isinstance(value, dict):
            continue
        if not value.get("__compressed__"):
            continue

        raw = base64.b64decode(value["data"])
        new_span.attributes[key] = decompress_text(raw)

    return new_span


def compute_compression_stats(
    spans: list[Span], config: CompressionConfig
) -> CompressionStats:
    """Estimate compression savings for a list of spans without persisting changes."""
    total_original = 0
    total_compressed = 0
    total_fields = 0

    for s in spans:
        for key in list(s.attributes.keys()):
            value = s.attributes[key]
            if not isinstance(value, str):
                continue
            raw_bytes = len(value.encode("utf-8"))
            if raw_bytes < config.min_size_bytes:
                continue

            total_original += raw_bytes
            compressed = compress_text(value, level=config.level)
            total_compressed += len(compressed)
            total_fields += 1

    ratio = total_compressed / total_original if total_original > 0 else 1.0

    return CompressionStats(
        original_bytes=total_original,
        compressed_bytes=total_compressed,
        compression_ratio=ratio,
        fields_compressed=total_fields,
    )
