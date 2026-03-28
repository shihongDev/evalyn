"""Tests for analysis.cost_phase module."""

from __future__ import annotations

import pytest
from datetime import datetime, timezone, timedelta

from evalyn_sdk.models import Span
from evalyn_sdk.analysis.cost_phase import (
    PHASE_MAPPING,
    PhaseCost,
    CostBreakdown,
    classify_phase,
    compute_cost_breakdown,
)


def _span(sid, span_type="llm_call", cost=0.0, tokens=0, duration_ms=100):
    t = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return Span(
        id=sid,
        name=f"span-{sid}",
        span_type=span_type,
        parent_id=None,
        start_time=t,
        end_time=t + timedelta(milliseconds=duration_ms),
        attributes={"cost": cost, "tokens": tokens},
    )


# ---------------------------------------------------------------------------
# classify_phase
# ---------------------------------------------------------------------------


class TestClassifyPhase:
    def test_llm_call(self):
        assert classify_phase("llm_call") == "reasoning"

    def test_tool_call(self):
        assert classify_phase("tool_call") == "tool_use"

    def test_retrieval(self):
        assert classify_phase("retrieval") == "retrieval"

    def test_scorer(self):
        assert classify_phase("scorer") == "evaluation"

    def test_node(self):
        assert classify_phase("node") == "orchestration"

    def test_agent(self):
        assert classify_phase("agent") == "orchestration"

    def test_custom(self):
        assert classify_phase("custom") == "other"

    def test_unknown_maps_to_other(self):
        assert classify_phase("totally_unknown") == "other"

    def test_empty_string(self):
        assert classify_phase("") == "other"


# ---------------------------------------------------------------------------
# compute_cost_breakdown
# ---------------------------------------------------------------------------


class TestComputeCostBreakdown:
    def test_single_phase(self):
        spans = [_span("1", "llm_call", cost=0.01, tokens=100)]
        bd = compute_cost_breakdown(spans)
        assert len(bd.phases) == 1
        assert bd.phases[0].phase == "reasoning"
        assert bd.total_cost_usd == pytest.approx(0.01)
        assert bd.total_tokens == 100

    def test_multiple_phases(self):
        spans = [
            _span("1", "llm_call", cost=0.03, tokens=300),
            _span("2", "tool_call", cost=0.01, tokens=100),
        ]
        bd = compute_cost_breakdown(spans)
        assert len(bd.phases) == 2
        assert bd.total_cost_usd == pytest.approx(0.04)
        assert bd.total_tokens == 400

    def test_percentage_sums_to_100(self):
        spans = [
            _span("1", "llm_call", cost=0.03, tokens=300),
            _span("2", "tool_call", cost=0.01, tokens=100),
            _span("3", "retrieval", cost=0.06, tokens=600),
        ]
        bd = compute_cost_breakdown(spans)
        total_pct = sum(p.percentage_cost for p in bd.phases)
        assert total_pct == pytest.approx(100.0)

    def test_dominant_phase(self):
        spans = [
            _span("1", "llm_call", cost=0.01, tokens=100),
            _span("2", "tool_call", cost=0.05, tokens=500),
        ]
        bd = compute_cost_breakdown(spans)
        assert bd.dominant_phase == "tool_use"

    def test_zero_cost_spans(self):
        spans = [_span("1", "llm_call", cost=0.0, tokens=0)]
        bd = compute_cost_breakdown(spans)
        assert bd.total_cost_usd == 0.0
        assert bd.phases[0].percentage_cost == 0.0

    def test_empty_spans(self):
        bd = compute_cost_breakdown([])
        assert bd.phases == []
        assert bd.total_cost_usd == 0.0
        assert bd.total_tokens == 0
        assert bd.dominant_phase == "other"

    def test_same_phase_aggregation(self):
        spans = [
            _span("1", "llm_call", cost=0.01, tokens=50),
            _span("2", "llm_call", cost=0.02, tokens=150),
        ]
        bd = compute_cost_breakdown(spans)
        assert len(bd.phases) == 1
        assert bd.phases[0].span_count == 2
        assert bd.phases[0].total_cost_usd == pytest.approx(0.03)
        assert bd.phases[0].total_tokens == 200

    def test_duration_aggregation(self):
        spans = [
            _span("1", "llm_call", cost=0.01, tokens=50, duration_ms=200),
            _span("2", "llm_call", cost=0.02, tokens=100, duration_ms=300),
        ]
        bd = compute_cost_breakdown(spans)
        assert bd.phases[0].total_duration_ms == pytest.approx(500.0)


# ---------------------------------------------------------------------------
# format_text
# ---------------------------------------------------------------------------


class TestFormatText:
    def test_format_text_output(self):
        spans = [
            _span("1", "llm_call", cost=0.03, tokens=300),
            _span("2", "tool_call", cost=0.01, tokens=100),
        ]
        bd = compute_cost_breakdown(spans)
        text = bd.format_text()
        assert "Cost Breakdown by Phase" in text
        assert "reasoning" in text
        assert "tool_use" in text
        assert "Dominant phase" in text

    def test_format_text_empty(self):
        bd = compute_cost_breakdown([])
        text = bd.format_text()
        assert "Total: $0.0000" in text


# ---------------------------------------------------------------------------
# as_dict / from_dict roundtrip
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_phase_cost_as_dict(self):
        pc = PhaseCost("reasoning", 2, 0.05, 500, 300.0, 62.5)
        d = pc.as_dict()
        assert d["phase"] == "reasoning"
        assert d["span_count"] == 2
        assert d["total_cost_usd"] == 0.05

    def test_cost_breakdown_roundtrip(self):
        spans = [
            _span("1", "llm_call", cost=0.03, tokens=300),
            _span("2", "tool_call", cost=0.01, tokens=100),
        ]
        original = compute_cost_breakdown(spans)
        d = original.as_dict()
        restored = CostBreakdown.from_dict(d)
        assert restored.total_cost_usd == pytest.approx(original.total_cost_usd)
        assert restored.total_tokens == original.total_tokens
        assert restored.dominant_phase == original.dominant_phase
        assert len(restored.phases) == len(original.phases)

    def test_from_dict_defaults(self):
        bd = CostBreakdown.from_dict({})
        assert bd.total_cost_usd == 0.0
        assert bd.total_tokens == 0
        assert bd.dominant_phase == "other"
        assert bd.phases == []
