from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from ..models import Span


@dataclass
class StreamChunk:
    """A single chunk from a streaming LLM response."""

    index: int
    content: str
    timestamp_ms: float = 0.0
    token_count: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "content": self.content,
            "timestamp_ms": self.timestamp_ms,
            "token_count": self.token_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StreamChunk:
        return cls(
            index=data["index"],
            content=data.get("content", ""),
            timestamp_ms=data.get("timestamp_ms", 0.0),
            token_count=data.get("token_count", 0),
        )


@dataclass
class StreamCapture:
    """Captured streaming LLM response data."""

    chunks: list[StreamChunk] = field(default_factory=list)
    total_tokens: int = 0
    first_token_ms: float = 0.0
    last_token_ms: float = 0.0
    complete: bool = False

    @property
    def full_content(self) -> str:
        """Concatenated content from all chunks."""
        return "".join(c.content for c in self.chunks)

    @property
    def ttft_ms(self) -> float:
        """Time to first token."""
        return self.first_token_ms

    @property
    def duration_ms(self) -> float:
        """Duration from first to last token."""
        return self.last_token_ms - self.first_token_ms

    @property
    def tokens_per_second(self) -> float:
        """Tokens per second throughput."""
        dur = self.duration_ms
        if dur <= 0:
            return 0.0
        return self.total_tokens / (dur / 1000.0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "chunks": [c.as_dict() for c in self.chunks],
            "total_tokens": self.total_tokens,
            "first_token_ms": self.first_token_ms,
            "last_token_ms": self.last_token_ms,
            "complete": self.complete,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StreamCapture:
        return cls(
            chunks=[StreamChunk.from_dict(c) for c in data.get("chunks", [])],
            total_tokens=data.get("total_tokens", 0),
            first_token_ms=data.get("first_token_ms", 0.0),
            last_token_ms=data.get("last_token_ms", 0.0),
            complete=data.get("complete", False),
        )


@dataclass
class StreamingStats:
    """Aggregate statistics across streaming captures."""

    total_streams: int = 0
    avg_ttft_ms: float = 0.0
    avg_tokens_per_second: float = 0.0
    avg_chunks: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_streams": self.total_streams,
            "avg_ttft_ms": self.avg_ttft_ms,
            "avg_tokens_per_second": self.avg_tokens_per_second,
            "avg_chunks": self.avg_chunks,
        }

    def format_text(self) -> str:
        """Human-readable summary of streaming stats."""
        lines = [
            f"Total streams: {self.total_streams}",
            f"Avg TTFT ms: {self.avg_ttft_ms:.1f}",
            f"Avg tokens/sec: {self.avg_tokens_per_second:.1f}",
            f"Avg chunks: {self.avg_chunks:.1f}",
        ]
        return "\n".join(lines)


def create_stream_capture() -> StreamCapture:
    """Factory for an empty StreamCapture."""
    return StreamCapture()


def add_chunk(
    capture: StreamCapture,
    content: str,
    timestamp_ms: float = 0.0,
    token_count: int = 1,
) -> StreamCapture:
    """Add a chunk to the capture, updating first/last timestamps. Mutates in place."""
    idx = len(capture.chunks)
    chunk = StreamChunk(
        index=idx,
        content=content,
        timestamp_ms=timestamp_ms,
        token_count=token_count,
    )
    capture.chunks.append(chunk)

    if idx == 0 or (timestamp_ms > 0 and (capture.first_token_ms == 0.0 or timestamp_ms < capture.first_token_ms)):
        capture.first_token_ms = timestamp_ms
    if timestamp_ms > capture.last_token_ms:
        capture.last_token_ms = timestamp_ms

    return capture


def finalize_capture(capture: StreamCapture) -> StreamCapture:
    """Mark capture as complete and compute total tokens."""
    capture.complete = True
    capture.total_tokens = sum(c.token_count for c in capture.chunks)
    return capture


def inject_stream_into_span(span: Span, capture: StreamCapture) -> Span:
    """Add stream capture data to span attributes. Returns a NEW span."""
    new_span = copy.deepcopy(span)
    new_span.attributes["stream_capture"] = capture.as_dict()
    return new_span


def extract_stream_from_span(span: Span) -> StreamCapture | None:
    """Extract StreamCapture from span attributes. Returns None if not present."""
    raw = span.attributes.get("stream_capture")
    if not raw or not isinstance(raw, dict):
        return None
    return StreamCapture.from_dict(raw)


def compute_streaming_stats(captures: list[StreamCapture]) -> StreamingStats:
    """Aggregate statistics across streaming captures."""
    if not captures:
        return StreamingStats()

    total = len(captures)
    ttft_sum = 0.0
    tps_sum = 0.0
    chunks_sum = 0

    for cap in captures:
        ttft_sum += cap.ttft_ms
        tps_sum += cap.tokens_per_second
        chunks_sum += len(cap.chunks)

    return StreamingStats(
        total_streams=total,
        avg_ttft_ms=ttft_sum / total,
        avg_tokens_per_second=tps_sum / total,
        avg_chunks=chunks_sum / total,
    )
