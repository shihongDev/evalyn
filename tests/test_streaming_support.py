from __future__ import annotations

from datetime import datetime, timezone

from evalyn_sdk.models import Span
from evalyn_sdk.trace.streaming_support import (
    StreamCapture,
    StreamChunk,
    StreamingStats,
    add_chunk,
    compute_streaming_stats,
    create_stream_capture,
    extract_stream_from_span,
    finalize_capture,
    inject_stream_into_span,
)


def _span(sid="s1", **attrs):
    return Span(
        id=sid,
        name="test",
        span_type="llm_call",
        parent_id=None,
        start_time=datetime.now(timezone.utc),
        attributes=attrs,
    )


# -- StreamChunk --


class TestStreamChunk:
    def test_as_dict(self):
        c = StreamChunk(index=0, content="hello", timestamp_ms=100.0, token_count=1)
        d = c.as_dict()
        assert d["index"] == 0
        assert d["content"] == "hello"
        assert d["timestamp_ms"] == 100.0

    def test_from_dict(self):
        d = {"index": 2, "content": "world", "timestamp_ms": 200.0, "token_count": 3}
        c = StreamChunk.from_dict(d)
        assert c.index == 2
        assert c.content == "world"
        assert c.token_count == 3

    def test_defaults(self):
        c = StreamChunk(index=0, content="x")
        assert c.timestamp_ms == 0.0
        assert c.token_count == 0


# -- StreamCapture --


class TestStreamCapture:
    def test_empty(self):
        cap = StreamCapture()
        assert cap.chunks == []
        assert cap.total_tokens == 0
        assert cap.complete is False

    def test_full_content(self):
        cap = StreamCapture(
            chunks=[
                StreamChunk(index=0, content="Hello"),
                StreamChunk(index=1, content=" world"),
            ]
        )
        assert cap.full_content == "Hello world"

    def test_ttft_ms(self):
        cap = StreamCapture(first_token_ms=42.0)
        assert cap.ttft_ms == 42.0

    def test_duration_ms(self):
        cap = StreamCapture(first_token_ms=100.0, last_token_ms=500.0)
        assert cap.duration_ms == 400.0

    def test_tokens_per_second(self):
        cap = StreamCapture(first_token_ms=0.0, last_token_ms=1000.0, total_tokens=50)
        assert cap.tokens_per_second == 50.0

    def test_tokens_per_second_zero_duration(self):
        cap = StreamCapture(first_token_ms=100.0, last_token_ms=100.0, total_tokens=10)
        assert cap.tokens_per_second == 0.0

    def test_as_dict(self):
        cap = StreamCapture(total_tokens=5, first_token_ms=10.0, last_token_ms=20.0, complete=True)
        d = cap.as_dict()
        assert d["total_tokens"] == 5
        assert d["complete"] is True

    def test_from_dict(self):
        d = {
            "chunks": [{"index": 0, "content": "hi", "timestamp_ms": 1.0, "token_count": 1}],
            "total_tokens": 1,
            "first_token_ms": 1.0,
            "last_token_ms": 1.0,
            "complete": True,
        }
        cap = StreamCapture.from_dict(d)
        assert len(cap.chunks) == 1
        assert cap.complete is True

    def test_roundtrip(self):
        cap = StreamCapture(
            chunks=[StreamChunk(index=0, content="a", timestamp_ms=10.0, token_count=1)],
            total_tokens=1,
            first_token_ms=10.0,
            last_token_ms=10.0,
            complete=True,
        )
        cap2 = StreamCapture.from_dict(cap.as_dict())
        assert cap2.full_content == "a"
        assert cap2.total_tokens == 1
        assert cap2.complete is True


# -- create_stream_capture --


class TestCreateStreamCapture:
    def test_empty(self):
        cap = create_stream_capture()
        assert cap.chunks == []
        assert cap.complete is False
        assert cap.total_tokens == 0


# -- add_chunk --


class TestAddChunk:
    def test_accumulates(self):
        cap = create_stream_capture()
        add_chunk(cap, "Hello", timestamp_ms=100.0)
        add_chunk(cap, " world", timestamp_ms=200.0)
        assert len(cap.chunks) == 2
        assert cap.chunks[0].index == 0
        assert cap.chunks[1].index == 1

    def test_updates_first_token(self):
        cap = create_stream_capture()
        add_chunk(cap, "a", timestamp_ms=50.0)
        assert cap.first_token_ms == 50.0

    def test_updates_last_token(self):
        cap = create_stream_capture()
        add_chunk(cap, "a", timestamp_ms=50.0)
        add_chunk(cap, "b", timestamp_ms=150.0)
        assert cap.last_token_ms == 150.0

    def test_returns_same_capture(self):
        cap = create_stream_capture()
        result = add_chunk(cap, "x")
        assert result is cap

    def test_default_token_count(self):
        cap = create_stream_capture()
        add_chunk(cap, "x")
        assert cap.chunks[0].token_count == 1


# -- finalize_capture --


class TestFinalizeCapture:
    def test_marks_complete(self):
        cap = create_stream_capture()
        add_chunk(cap, "a", token_count=2)
        add_chunk(cap, "b", token_count=3)
        finalize_capture(cap)
        assert cap.complete is True
        assert cap.total_tokens == 5

    def test_empty_capture(self):
        cap = create_stream_capture()
        finalize_capture(cap)
        assert cap.complete is True
        assert cap.total_tokens == 0


# -- inject / extract --


class TestInjectExtract:
    def test_roundtrip(self):
        span = _span()
        cap = create_stream_capture()
        add_chunk(cap, "Hello", timestamp_ms=10.0)
        add_chunk(cap, " there", timestamp_ms=20.0)
        finalize_capture(cap)

        new_span = inject_stream_into_span(span, cap)
        extracted = extract_stream_from_span(new_span)
        assert extracted is not None
        assert extracted.full_content == "Hello there"
        assert extracted.complete is True

    def test_inject_returns_new_span(self):
        span = _span()
        cap = create_stream_capture()
        new_span = inject_stream_into_span(span, cap)
        assert new_span is not span
        assert "stream_capture" not in span.attributes

    def test_extract_returns_none_no_data(self):
        span = _span()
        assert extract_stream_from_span(span) is None

    def test_extract_returns_none_bad_type(self):
        span = _span(stream_capture="not_a_dict")
        assert extract_stream_from_span(span) is None


# -- compute_streaming_stats --


class TestComputeStreamingStats:
    def test_averages(self):
        cap1 = StreamCapture(
            chunks=[StreamChunk(0, "a")],
            total_tokens=10,
            first_token_ms=100.0,
            last_token_ms=1100.0,
            complete=True,
        )
        cap2 = StreamCapture(
            chunks=[StreamChunk(0, "b"), StreamChunk(1, "c")],
            total_tokens=20,
            first_token_ms=200.0,
            last_token_ms=1200.0,
            complete=True,
        )
        stats = compute_streaming_stats([cap1, cap2])
        assert stats.total_streams == 2
        assert stats.avg_ttft_ms == 150.0  # (100+200)/2
        assert stats.avg_chunks == 1.5  # (1+2)/2

    def test_empty_list(self):
        stats = compute_streaming_stats([])
        assert stats.total_streams == 0
        assert stats.avg_ttft_ms == 0.0

    def test_single_capture(self):
        cap = StreamCapture(
            chunks=[StreamChunk(0, "x")],
            total_tokens=5,
            first_token_ms=50.0,
            last_token_ms=550.0,
            complete=True,
        )
        stats = compute_streaming_stats([cap])
        assert stats.total_streams == 1
        assert stats.avg_ttft_ms == 50.0
        # tokens_per_second: 5 / (500/1000) = 10.0
        assert stats.avg_tokens_per_second == 10.0
        assert stats.avg_chunks == 1.0


# -- StreamingStats --


class TestStreamingStats:
    def test_as_dict(self):
        s = StreamingStats(total_streams=3, avg_ttft_ms=10.0, avg_tokens_per_second=5.0, avg_chunks=2.0)
        d = s.as_dict()
        assert d["total_streams"] == 3
        assert d["avg_ttft_ms"] == 10.0

    def test_format_text(self):
        s = StreamingStats(total_streams=2, avg_ttft_ms=15.5, avg_tokens_per_second=8.3, avg_chunks=4.0)
        txt = s.format_text()
        assert "Total streams: 2" in txt
        assert "Avg TTFT ms: 15.5" in txt
        assert "Avg tokens/sec: 8.3" in txt
        assert "Avg chunks: 4.0" in txt

    def test_defaults(self):
        s = StreamingStats()
        assert s.total_streams == 0
        assert s.avg_ttft_ms == 0.0
