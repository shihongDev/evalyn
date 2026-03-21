import tempfile

from evalyn_sdk.models import SpanMetricLink
from evalyn_sdk.storage.sqlite import SQLiteStorage


def test_span_metric_link_creation():
    link = SpanMetricLink(
        id="sml-1",
        metric_result_id="metric1:item1:call1",
        span_id="span-abc",
        relevance=0.85,
        reason="This span contained the hallucinated claim",
        run_id="run-123",
    )
    assert link.relevance == 0.85
    assert link.span_id == "span-abc"


def test_span_metric_link_as_dict_from_dict():
    link = SpanMetricLink(
        id="sml-2",
        metric_result_id="m:i:c",
        span_id="s1",
        relevance=0.5,
        reason="test",
        run_id="r1",
    )
    d = link.as_dict()
    restored = SpanMetricLink.from_dict(d)
    assert restored.id == link.id
    assert restored.relevance == link.relevance


def test_store_and_list_span_metric_links():
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        store = SQLiteStorage(f.name)
        link = SpanMetricLink(
            id="sml-1",
            metric_result_id="m:i:c",
            span_id="s1",
            relevance=0.9,
            reason="caused failure",
            run_id="run-1",
        )
        store.store_span_metric_links([link])
        results = store.list_span_metric_links(run_id="run-1")
        assert len(results) == 1
        assert results[0].span_id == "s1"
        store.close()


def test_list_span_metric_links_by_span():
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        store = SQLiteStorage(f.name)
        links = [
            SpanMetricLink("sml-1", "m1:i:c", "span-a", 0.9, "r1", "run-1"),
            SpanMetricLink("sml-2", "m2:i:c", "span-a", 0.5, "r2", "run-1"),
            SpanMetricLink("sml-3", "m3:i:c", "span-b", 0.7, "r3", "run-1"),
        ]
        store.store_span_metric_links(links)
        results = store.list_span_metric_links(run_id="run-1", span_id="span-a")
        assert len(results) == 2
        store.close()


def test_store_span_metric_links_empty():
    """Empty iterable should not raise."""
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        store = SQLiteStorage(f.name)
        store.store_span_metric_links([])
        results = store.list_span_metric_links(run_id="run-1")
        assert results == []
        store.close()


def test_store_span_metric_links_duplicate_upsert():
    """INSERT OR REPLACE should overwrite on duplicate (run_id, metric_result_id, span_id)."""
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        store = SQLiteStorage(f.name)
        link_v1 = SpanMetricLink(
            id="sml-1",
            metric_result_id="m:i:c",
            span_id="s1",
            relevance=0.5,
            reason="first version",
            run_id="run-1",
        )
        link_v2 = SpanMetricLink(
            id="sml-1",
            metric_result_id="m:i:c",
            span_id="s1",
            relevance=0.9,
            reason="updated version",
            run_id="run-1",
        )
        store.store_span_metric_links([link_v1])
        store.store_span_metric_links([link_v2])
        results = store.list_span_metric_links(run_id="run-1")
        assert len(results) == 1
        assert results[0].relevance == 0.9
        assert results[0].reason == "updated version"
        store.close()
