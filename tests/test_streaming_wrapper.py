import asyncio
import time

import pytest

from evalyn_sdk.trace.instrumentation.providers._streaming import (
    AsyncLoggingWrapper,
    LoggingWrapper,
    StreamingSpanWrapper,
)


def test_streaming_wrapper_basic():
    chunks = ["Hello", " world", "!"]
    wrapper = StreamingSpanWrapper(iter(chunks), request_start_time=time.time())
    collected = list(wrapper)
    assert collected == ["Hello", " world", "!"]
    assert wrapper.chunk_count == 3
    assert wrapper.time_to_first_token_ms >= 0
    assert wrapper.streaming_duration_ms >= 0


def test_streaming_wrapper_empty():
    wrapper = StreamingSpanWrapper(iter([]), request_start_time=time.time())
    collected = list(wrapper)
    assert collected == []
    assert wrapper.chunk_count == 0


def test_streaming_wrapper_as_attributes():
    wrapper = StreamingSpanWrapper(iter(["a", "b"]), request_start_time=time.time())
    list(wrapper)  # exhaust
    attrs = wrapper.as_span_attributes()
    assert attrs["streaming"] is True
    assert "time_to_first_token_ms" in attrs
    assert "chunk_count" in attrs
    assert attrs["chunk_count"] == 2


# --- LoggingWrapper tests ---


def test_logging_wrapper_fires_callback_on_exhaust():
    """on_exhaust callback fires exactly once when stream ends."""
    calls = []
    inner = StreamingSpanWrapper(iter(["a", "b"]), request_start_time=time.time())
    lw = LoggingWrapper(inner, on_exhaust=lambda w: calls.append(w))

    collected = list(lw)
    assert collected == ["a", "b"]
    assert len(calls) == 1
    assert calls[0] is inner


def test_logging_wrapper_double_exhaustion():
    """Calling next() after StopIteration does not fire callback again."""
    call_count = 0

    def on_exhaust(w):
        nonlocal call_count
        call_count += 1

    inner = StreamingSpanWrapper(iter(["x"]), request_start_time=time.time())
    lw = LoggingWrapper(inner, on_exhaust=on_exhaust)

    assert next(lw) == "x"
    with pytest.raises(StopIteration):
        next(lw)
    assert call_count == 1

    # Second StopIteration should NOT fire callback again
    with pytest.raises(StopIteration):
        next(lw)
    assert call_count == 1


def test_logging_wrapper_getattr_proxy():
    """__getattr__ proxies to the underlying iterator."""

    class FakeStream:
        custom_attr = "hello"

    inner = StreamingSpanWrapper(iter([]), request_start_time=time.time())
    inner._iterator = FakeStream()
    lw = LoggingWrapper(inner, on_exhaust=lambda w: None)

    assert lw.custom_attr == "hello"


def test_logging_wrapper_callback_receives_metrics():
    """Callback receives the wrapper with correct chunk count and timing."""
    captured = {}

    def on_exhaust(w):
        captured["chunk_count"] = w.chunk_count
        captured["attrs"] = w.as_span_attributes()

    inner = StreamingSpanWrapper(iter(["a", "b", "c"]), request_start_time=time.time())
    lw = LoggingWrapper(inner, on_exhaust=on_exhaust)
    list(lw)

    assert captured["chunk_count"] == 3
    assert captured["attrs"]["streaming"] is True
    assert captured["attrs"]["chunk_count"] == 3


# --- AsyncLoggingWrapper tests ---


class _AsyncIter:
    """Simple async iterator for testing."""

    def __init__(self, items):
        self._items = iter(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._items)
        except StopIteration:
            raise StopAsyncIteration


def test_async_logging_wrapper_fires_callback():
    """on_exhaust callback fires exactly once when async stream ends."""
    calls = []
    tracker = StreamingSpanWrapper(iter([]), request_start_time=time.time())
    alw = AsyncLoggingWrapper(
        _AsyncIter(["a", "b"]),
        wrapper=tracker,
        on_exhaust=lambda w: calls.append(w),
    )

    async def consume():
        result = []
        async for chunk in alw:
            result.append(chunk)
        return result

    collected = asyncio.get_event_loop().run_until_complete(consume())
    assert collected == ["a", "b"]
    assert len(calls) == 1
    assert calls[0] is tracker
    assert tracker.chunk_count == 2


def test_async_logging_wrapper_double_exhaustion():
    """Second StopAsyncIteration does not fire callback again."""
    call_count = 0

    def on_exhaust(w):
        nonlocal call_count
        call_count += 1

    tracker = StreamingSpanWrapper(iter([]), request_start_time=time.time())
    alw = AsyncLoggingWrapper(
        _AsyncIter(["x"]),
        wrapper=tracker,
        on_exhaust=on_exhaust,
    )

    async def run():
        assert await alw.__anext__() == "x"
        with pytest.raises(StopAsyncIteration):
            await alw.__anext__()
        assert call_count == 1
        # Second call
        with pytest.raises(StopAsyncIteration):
            await alw.__anext__()
        assert call_count == 1

    asyncio.get_event_loop().run_until_complete(run())


def test_async_logging_wrapper_extract_tokens():
    """Subclass _extract_tokens hook is called per chunk."""
    extracted = []

    class TrackingWrapper(AsyncLoggingWrapper):
        def _extract_tokens(self, chunk):
            extracted.append(chunk)
            self._wrapper.accumulated_input_tokens += 10

    tracker = StreamingSpanWrapper(iter([]), request_start_time=time.time())
    alw = TrackingWrapper(
        _AsyncIter(["a", "b"]),
        wrapper=tracker,
        on_exhaust=lambda w: None,
    )

    async def consume():
        async for _ in alw:
            pass

    asyncio.get_event_loop().run_until_complete(consume())
    assert extracted == ["a", "b"]
    assert tracker.accumulated_input_tokens == 20


def test_async_logging_wrapper_timing():
    """AsyncLoggingWrapper records first/last chunk times."""
    tracker = StreamingSpanWrapper(iter([]), request_start_time=time.time())
    alw = AsyncLoggingWrapper(
        _AsyncIter(["a", "b", "c"]),
        wrapper=tracker,
        on_exhaust=lambda w: None,
    )

    async def consume():
        async for _ in alw:
            pass

    asyncio.get_event_loop().run_until_complete(consume())
    assert tracker._first_chunk_time is not None
    assert tracker._last_chunk_time is not None
    assert tracker._last_chunk_time >= tracker._first_chunk_time
    attrs = tracker.as_span_attributes()
    assert attrs["chunk_count"] == 3
    assert attrs["time_to_first_token_ms"] >= 0


def test_async_logging_wrapper_getattr_proxy():
    """__getattr__ proxies to the underlying async iterator."""

    class FakeAsyncStream:
        custom_attr = "async_hello"

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    tracker = StreamingSpanWrapper(iter([]), request_start_time=time.time())
    alw = AsyncLoggingWrapper(
        FakeAsyncStream(),
        wrapper=tracker,
        on_exhaust=lambda w: None,
    )

    assert alw.custom_attr == "async_hello"
