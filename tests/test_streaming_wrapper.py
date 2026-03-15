import time
from evalyn_sdk.trace.instrumentation.providers._streaming import StreamingSpanWrapper


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
