"""Streaming span wrapper for capturing streaming response metrics."""
from __future__ import annotations

import time
from typing import Any, Dict, Iterator, TypeVar

T = TypeVar("T")


class StreamingSpanWrapper:
    """Wraps a streaming iterator to capture timing and chunk metrics.

    Yields chunks unchanged while recording:
    - time_to_first_token_ms
    - chunk_count
    - streaming_duration_ms
    """

    def __init__(self, iterator: Iterator[T], request_start_time: float):
        self._iterator = iterator
        self._request_start_time = request_start_time
        self._first_chunk_time: float | None = None
        self._last_chunk_time: float | None = None
        self.chunk_count: int = 0
        self.accumulated_input_tokens: int = 0
        self.accumulated_output_tokens: int = 0

    def __iter__(self):
        return self

    def __next__(self):
        chunk = next(self._iterator)
        now = time.time()
        if self._first_chunk_time is None:
            self._first_chunk_time = now
        self._last_chunk_time = now
        self.chunk_count += 1
        return chunk

    @property
    def time_to_first_token_ms(self) -> float:
        if self._first_chunk_time is None:
            return 0.0
        return (self._first_chunk_time - self._request_start_time) * 1000

    @property
    def streaming_duration_ms(self) -> float:
        if self._first_chunk_time is None or self._last_chunk_time is None:
            return 0.0
        return (self._last_chunk_time - self._first_chunk_time) * 1000

    def as_span_attributes(self) -> Dict[str, Any]:
        return {
            "streaming": True,
            "time_to_first_token_ms": round(self.time_to_first_token_ms, 2),
            "chunk_count": self.chunk_count,
            "streaming_duration_ms": round(self.streaming_duration_ms, 2),
        }
