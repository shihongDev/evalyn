"""Tests for attribution extraction pipeline."""
from evalyn_sdk.attribution import extract_span_metric_links
from evalyn_sdk.models import MetricResult


def test_extract_links_from_metric_result():
    """Extract SpanMetricLinks from a MetricResult with raw_judge attribution."""
    result = MetricResult(
        metric_id="helpfulness",
        item_id="item-1",
        call_id="call-1",
        score=0.9,
        passed=True,
        raw_judge={
            "span_attribution": [
                {"span_id": "s1", "relevance": 0.8, "reason": "main call"},
            ]
        },
    )
    links = extract_span_metric_links(result, run_id="run-1")
    assert len(links) == 1
    assert links[0].span_id == "s1"
    assert links[0].relevance == 0.8
    assert links[0].run_id == "run-1"
    assert links[0].metric_result_id == "helpfulness:item-1:call-1"


def test_extract_links_no_attribution():
    """Objective metrics without raw_judge should return empty list."""
    result = MetricResult(
        metric_id="accuracy",
        item_id="item-1",
        call_id="call-1",
        score=1.0,
        passed=True,
    )
    links = extract_span_metric_links(result, run_id="run-1")
    assert links == []


def test_extract_links_empty_attribution():
    """Empty span_attribution array should return empty list."""
    result = MetricResult(
        metric_id="quality",
        item_id="item-1",
        call_id="call-1",
        score=0.5,
        passed=True,
        raw_judge={"span_attribution": []},
    )
    links = extract_span_metric_links(result, run_id="run-1")
    assert links == []


def test_extract_links_multiple_spans():
    """Multiple span attributions should produce multiple links."""
    result = MetricResult(
        metric_id="quality",
        item_id="item-1",
        call_id="call-1",
        score=0.3,
        passed=False,
        raw_judge={
            "span_attribution": [
                {"span_id": "s1", "relevance": 0.95, "reason": "hallucinated"},
                {"span_id": "s2", "relevance": 0.4, "reason": "wrong tool"},
            ]
        },
    )
    links = extract_span_metric_links(result, run_id="run-abc")
    assert len(links) == 2
    assert links[0].span_id == "s1"
    assert links[1].span_id == "s2"
    # Each link should have a unique ID
    assert links[0].id != links[1].id


def test_extract_links_skips_missing_span_id():
    """Attributions without span_id should be skipped."""
    result = MetricResult(
        metric_id="quality",
        item_id="item-1",
        call_id="call-1",
        score=0.5,
        passed=True,
        raw_judge={
            "span_attribution": [
                {"relevance": 0.8, "reason": "no span_id"},
                {"span_id": "s1", "relevance": 0.6, "reason": "valid"},
            ]
        },
    )
    links = extract_span_metric_links(result, run_id="run-1")
    assert len(links) == 1
    assert links[0].span_id == "s1"
