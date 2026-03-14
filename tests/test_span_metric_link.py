from evalyn_sdk.models import SpanMetricLink


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
