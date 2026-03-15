"""Integration tests for tracing improvements."""
import tempfile
import time

from evalyn_sdk.models import SpanMetricLink, MetricResult
from evalyn_sdk.storage.sqlite import SQLiteStorage
from evalyn_sdk.attribution import extract_span_metric_links
from evalyn_sdk.trace.instrumentation.providers._shared import (
    get_model_context_window,
    MODEL_CONTEXT_WINDOWS,
)
from evalyn_sdk.trace.instrumentation.providers._streaming import StreamingSpanWrapper


def test_full_attribution_flow():
    """End-to-end: judge result -> attribution extraction -> storage -> retrieval."""
    result = MetricResult(
        metric_id="quality",
        item_id="item-1",
        call_id="call-1",
        score=0.3,
        passed=False,
        raw_judge={
            "passed": False,
            "reason": "Low quality",
            "span_attribution": [
                {"span_id": "llm-span-1", "relevance": 0.95, "reason": "hallucinated"},
                {"span_id": "tool-span-2", "relevance": 0.4, "reason": "wrong tool input"},
            ],
        },
    )

    # Extract links
    links = extract_span_metric_links(result, run_id="run-abc")
    assert len(links) == 2

    # Store and retrieve
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        store = SQLiteStorage(f.name)
        store.store_span_metric_links(links)

        # Query by run
        all_links = store.list_span_metric_links(run_id="run-abc")
        assert len(all_links) == 2

        # Query by span
        span_links = store.list_span_metric_links(
            run_id="run-abc", span_id="llm-span-1"
        )
        assert len(span_links) == 1
        assert span_links[0].relevance == 0.95

        # Query by metric_result_id
        metric_links = store.list_span_metric_links(
            run_id="run-abc",
            metric_result_id="quality:item-1:call-1",
        )
        assert len(metric_links) == 2

        store.close()


def test_context_window_coverage():
    """All major model families should have context window entries."""
    families = ["gpt-4o", "claude-sonnet-4", "gemini-2.5-flash"]
    for model in families:
        window = get_model_context_window(model)
        assert window is not None, f"Missing context window for {model}"
        assert window > 0


def test_streaming_wrapper_full_lifecycle():
    """StreamingSpanWrapper captures all metrics through full lifecycle."""
    start = time.time()
    chunks = ["chunk1", "chunk2", "chunk3", "chunk4", "chunk5"]
    wrapper = StreamingSpanWrapper(iter(chunks), request_start_time=start)

    collected = list(wrapper)
    assert collected == chunks
    assert wrapper.chunk_count == 5

    attrs = wrapper.as_span_attributes()
    assert attrs["streaming"] is True
    assert attrs["chunk_count"] == 5
    assert isinstance(attrs["time_to_first_token_ms"], float)
    assert isinstance(attrs["streaming_duration_ms"], float)
    assert attrs["time_to_first_token_ms"] >= 0
    assert attrs["streaming_duration_ms"] >= 0


def test_span_metric_link_roundtrip():
    """SpanMetricLink survives as_dict -> from_dict roundtrip."""
    link = SpanMetricLink(
        id="sml-test",
        metric_result_id="m:i:c",
        span_id="span-1",
        relevance=0.75,
        reason="test reason",
        run_id="run-1",
    )
    d = link.as_dict()
    restored = SpanMetricLink.from_dict(d)
    assert restored.id == link.id
    assert restored.metric_result_id == link.metric_result_id
    assert restored.span_id == link.span_id
    assert restored.relevance == link.relevance
    assert restored.reason == link.reason
    assert restored.run_id == link.run_id


def test_attribution_with_no_raw_judge():
    """Objective metrics produce no attribution links."""
    result = MetricResult(
        metric_id="latency_check",
        item_id="item-1",
        call_id="call-1",
        score=1.0,
        passed=True,
        raw_judge=None,
    )
    links = extract_span_metric_links(result, run_id="run-1")
    assert links == []
