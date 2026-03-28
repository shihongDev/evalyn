"""Tests for subagent_cost module."""

from datetime import datetime, timezone

from evalyn_sdk.models import Span
from evalyn_sdk.analysis.subagent_cost import (
    SubagentCost,
    CostAllocationReport,
    extract_agent_id,
    allocate_costs,
    compare_agent_costs,
    find_cost_anomalies,
)


def _span(sid, name="agent", span_type="llm_call", parent_id=None, **attrs):
    return Span(
        id=sid,
        name=name,
        span_type=span_type,
        parent_id=parent_id,
        start_time=datetime.now(timezone.utc),
        attributes=attrs,
    )


# --- extract_agent_id ---


def test_extract_agent_id_from_agent_id_attr():
    s = _span("s1", agent_id="planner")
    assert extract_agent_id(s) == "planner"


def test_extract_agent_id_from_agent_name_attr():
    s = _span("s1", agent_name="researcher")
    assert extract_agent_id(s) == "researcher"


def test_extract_agent_id_prefers_agent_id_over_name():
    s = _span("s1", agent_id="id1", agent_name="name1")
    assert extract_agent_id(s) == "id1"


def test_extract_agent_id_fallback_to_span_name():
    s = _span("s1", name="my_agent")
    assert extract_agent_id(s) == "my_agent"


def test_extract_agent_id_numeric_value():
    s = _span("s1", agent_id=42)
    assert extract_agent_id(s) == "42"


# --- allocate_costs: single agent ---


def test_allocate_costs_single_agent():
    spans = [
        _span("s1", agent_id="a1", cost=0.5, tokens=100),
        _span("s2", agent_id="a1", cost=0.3, tokens=50),
    ]
    report = allocate_costs(spans)
    assert len(report.agents) == 1
    a = report.agents[0]
    assert a.agent_id == "a1"
    assert a.total_cost_usd == 0.8
    assert a.total_tokens == 150
    assert a.call_count == 2
    assert abs(a.avg_cost_per_call - 0.4) < 1e-9


def test_allocate_costs_single_agent_total():
    spans = [_span("s1", agent_id="a1", cost=1.0, tokens=200)]
    report = allocate_costs(spans)
    assert report.total_cost_usd == 1.0


# --- allocate_costs: multiple agents ---


def test_allocate_costs_multiple_agents():
    spans = [
        _span("s1", agent_id="planner", cost=1.0, tokens=500),
        _span("s2", agent_id="executor", cost=0.2, tokens=100),
        _span("s3", agent_id="planner", cost=0.5, tokens=200),
    ]
    report = allocate_costs(spans)
    assert len(report.agents) == 2
    by_id = {a.agent_id: a for a in report.agents}
    assert by_id["planner"].total_cost_usd == 1.5
    assert by_id["executor"].total_cost_usd == 0.2


def test_allocate_costs_total_across_agents():
    spans = [
        _span("s1", agent_id="a", cost=1.0, tokens=10),
        _span("s2", agent_id="b", cost=2.0, tokens=20),
    ]
    report = allocate_costs(spans)
    assert report.total_cost_usd == 3.0


# --- most_expensive / cheapest ---


def test_most_expensive_agent():
    spans = [
        _span("s1", agent_id="cheap", cost=0.1, tokens=10),
        _span("s2", agent_id="pricey", cost=5.0, tokens=1000),
    ]
    report = allocate_costs(spans)
    assert report.most_expensive_agent == "pricey"


def test_cheapest_agent():
    spans = [
        _span("s1", agent_id="cheap", cost=0.1, tokens=10),
        _span("s2", agent_id="pricey", cost=5.0, tokens=1000),
    ]
    report = allocate_costs(spans)
    assert report.cheapest_agent == "cheap"


def test_most_expensive_and_cheapest_same_when_single_agent():
    spans = [_span("s1", agent_id="solo", cost=1.0, tokens=100)]
    report = allocate_costs(spans)
    assert report.most_expensive_agent == "solo"
    assert report.cheapest_agent == "solo"


# --- compare_agent_costs ---


def test_compare_agent_costs_ratios():
    spans = [
        _span("s1", agent_id="a", cost=3.0, tokens=300),
        _span("s2", agent_id="b", cost=1.0, tokens=100),
    ]
    report = allocate_costs(spans)
    comparison = compare_agent_costs(report)
    assert abs(comparison["cost_ratio"]["a"] - 0.75) < 1e-9
    assert abs(comparison["cost_ratio"]["b"] - 0.25) < 1e-9


def test_compare_agent_costs_token_ratios():
    spans = [
        _span("s1", agent_id="a", cost=1.0, tokens=800),
        _span("s2", agent_id="b", cost=1.0, tokens=200),
    ]
    report = allocate_costs(spans)
    comparison = compare_agent_costs(report)
    assert abs(comparison["token_ratio"]["a"] - 0.8) < 1e-9
    assert abs(comparison["token_ratio"]["b"] - 0.2) < 1e-9


def test_compare_agent_costs_efficiency_ranking():
    spans = [
        _span("s1", agent_id="expensive", cost=10.0, tokens=100),
        _span("s2", agent_id="cheap", cost=1.0, tokens=100),
    ]
    report = allocate_costs(spans)
    comparison = compare_agent_costs(report)
    # cheap has lower avg cost, ranked first
    assert comparison["efficiency_ranking"][0] == "cheap"


def test_compare_agent_costs_empty():
    report = CostAllocationReport()
    comparison = compare_agent_costs(report)
    assert comparison["cost_ratio"] == {}
    assert comparison["efficiency_ranking"] == []


# --- find_cost_anomalies ---


def test_find_cost_anomalies_detects_outlier():
    spans = [
        _span("s1", agent_id="normal1", cost=1.0, tokens=100),
        _span("s2", agent_id="normal2", cost=1.0, tokens=100),
        _span("s3", agent_id="outlier", cost=10.0, tokens=100),
    ]
    report = allocate_costs(spans)
    # overall avg = 12/3 = 4.0; threshold = 2*4 = 8; outlier avg = 10 > 8
    anomalies = find_cost_anomalies(report, threshold_multiplier=2.0)
    assert "outlier" in anomalies
    assert "normal1" not in anomalies


def test_find_cost_anomalies_no_anomalies():
    spans = [
        _span("s1", agent_id="a", cost=1.0, tokens=100),
        _span("s2", agent_id="b", cost=1.0, tokens=100),
    ]
    report = allocate_costs(spans)
    anomalies = find_cost_anomalies(report)
    assert anomalies == []


def test_find_cost_anomalies_custom_threshold():
    spans = [
        _span("s1", agent_id="a", cost=1.0, tokens=100),
        _span("s2", agent_id="b", cost=3.0, tokens=100),
    ]
    report = allocate_costs(spans)
    # overall avg = 4/2 = 2.0; threshold = 1.0 * 2.0 = 2.0; b avg = 3.0 > 2.0
    anomalies = find_cost_anomalies(report, threshold_multiplier=1.0)
    assert "b" in anomalies


def test_find_cost_anomalies_empty():
    report = CostAllocationReport()
    assert find_cost_anomalies(report) == []


# --- CostAllocationReport format_text ---


def test_format_text_contains_header():
    spans = [_span("s1", agent_id="a1", cost=1.0, tokens=100)]
    report = allocate_costs(spans)
    text = report.format_text()
    assert "Subagent Cost Allocation" in text


def test_format_text_contains_agent():
    spans = [_span("s1", agent_id="planner", cost=2.5, tokens=500)]
    report = allocate_costs(spans)
    text = report.format_text()
    assert "planner" in text
    assert "$2.5000" in text


def test_format_text_contains_total():
    spans = [_span("s1", agent_id="a", cost=1.5, tokens=100)]
    report = allocate_costs(spans)
    text = report.format_text()
    assert "Total:" in text


# --- as_dict / from_dict roundtrips ---


def test_subagent_cost_as_dict():
    sc = SubagentCost(
        agent_id="a1",
        agent_name="planner",
        total_cost_usd=1.5,
        total_tokens=300,
        call_count=2,
        avg_cost_per_call=0.75,
    )
    d = sc.as_dict()
    assert d["agent_id"] == "a1"
    assert d["total_cost_usd"] == 1.5
    assert d["avg_cost_per_call"] == 0.75


def test_cost_allocation_report_roundtrip():
    spans = [
        _span("s1", agent_id="a", cost=1.0, tokens=100),
        _span("s2", agent_id="b", cost=2.0, tokens=200),
    ]
    original = allocate_costs(spans)
    d = original.as_dict()
    restored = CostAllocationReport.from_dict(d)
    assert restored.total_cost_usd == original.total_cost_usd
    assert restored.most_expensive_agent == original.most_expensive_agent
    assert restored.cheapest_agent == original.cheapest_agent
    assert len(restored.agents) == len(original.agents)


def test_cost_allocation_report_from_dict_defaults():
    restored = CostAllocationReport.from_dict({})
    assert restored.agents == []
    assert restored.total_cost_usd == 0.0


# --- Edge cases ---


def test_allocate_costs_empty_spans():
    report = allocate_costs([])
    assert report.agents == []
    assert report.total_cost_usd == 0.0
    assert report.most_expensive_agent == ""
    assert report.cheapest_agent == ""


def test_allocate_costs_single_span():
    spans = [_span("s1", agent_id="solo", cost=0.5, tokens=50)]
    report = allocate_costs(spans)
    assert len(report.agents) == 1
    assert report.agents[0].call_count == 1
    assert report.agents[0].avg_cost_per_call == 0.5


def test_allocate_costs_zero_cost_spans():
    spans = [_span("s1", agent_id="free", tokens=100)]
    report = allocate_costs(spans)
    assert report.agents[0].total_cost_usd == 0.0
    assert report.total_cost_usd == 0.0


def test_allocate_costs_agent_name_display():
    spans = [
        _span("s1", agent_id="a1", agent_name="Planner Agent", cost=1.0, tokens=100),
    ]
    report = allocate_costs(spans)
    # agent_id takes priority for ID, but agent_name is stored
    assert report.agents[0].agent_id == "a1"
    assert report.agents[0].agent_name == "Planner Agent"
