"""Streaming span wrapper for capturing streaming response metrics."""
from __future__ import annotations

import time
from typing import Any, Dict, TypeVar
from collections.abc import Callable, Iterator

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

    def as_span_attributes(self) -> dict[str, Any]:
        return {
            "streaming": True,
            "time_to_first_token_ms": round(self.time_to_first_token_ms, 2),
            "chunk_count": self.chunk_count,
            "streaming_duration_ms": round(self.streaming_duration_ms, 2),
        }


class LoggingWrapper:
    """Proxy that logs via callback when a sync stream is exhausted.

    Wraps a StreamingSpanWrapper and calls on_exhaust(wrapper) exactly
    once when StopIteration is raised.
    """

    def __init__(self, wrapper: StreamingSpanWrapper, on_exhaust: Callable):
        self._w = wrapper
        self._on_exhaust = on_exhaust
        self._exhausted = False

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self._w)
        except StopIteration:
            if not self._exhausted:
                self._exhausted = True
                self._on_exhaust(self._w)
            raise

    def __getattr__(self, name):
        return getattr(self._w._iterator, name)


class AsyncLoggingWrapper:
    """Proxy that logs via callback when an async stream is exhausted.

    Wraps an async iterator alongside a StreamingSpanWrapper that tracks
    timing/token metrics, then calls on_exhaust(wrapper) on completion.
    """

    def __init__(
        self,
        aiter: Any,
        wrapper: StreamingSpanWrapper,
        on_exhaust: Callable,
    ):
        self._aiter = aiter
        self._wrapper = wrapper
        self._on_exhaust = on_exhaust
        self._exhausted = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            chunk = await self._aiter.__anext__()
            now = time.time()
            if self._wrapper._first_chunk_time is None:
                self._wrapper._first_chunk_time = now
            self._wrapper._last_chunk_time = now
            self._wrapper.chunk_count += 1
            self._extract_tokens(chunk)
            return chunk
        except StopAsyncIteration:
            if not self._exhausted:
                self._exhausted = True
                self._on_exhaust(self._wrapper)
            raise

    def _extract_tokens(self, chunk: Any) -> None:
        """Override in subclasses to extract provider-specific token counts."""

    def __getattr__(self, name):
        return getattr(self._aiter, name)
