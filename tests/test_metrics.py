"""
Tests for metrics functionality.

Run with: pytest tests/test_metrics.py -v
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest


class TestMetricRegistry:
    """Tests for MetricRegistry."""

    def test_registry_creation(self):
        """Test creating a MetricRegistry."""
        from evalyn_sdk import MetricRegistry
        registry = MetricRegistry()
        assert registry is not None

    def test_register_and_get_metric(self):
        """Test registering and retrieving a metric."""
        from evalyn_sdk import MetricRegistry, Metric, MetricSpec

        registry = MetricRegistry()

        spec = MetricSpec(
            id="test_metric",
            type="objective",
            name="Test Metric",
            description="A test metric",
        )

        def handler(call, item):
            from evalyn_sdk import MetricResult
            return MetricResult(
                metric_id=spec.id,
                item_id=item.id,
                call_id=call.id,
                score=1.0,
                passed=True,
            )

        metric = Metric(spec=spec, handler=handler)
        registry.register(metric)

        # Verify retrieval
        retrieved = registry.get("test_metric")
        assert retrieved is not None
        assert retrieved.spec.id == "test_metric"

    def test_get_nonexistent_metric(self):
        """Test getting a nonexistent metric returns None."""
        from evalyn_sdk import MetricRegistry
        registry = MetricRegistry()
        assert registry.get("nonexistent") is None


class TestObjectiveMetrics:
    """Tests for objective metrics."""

    def _make_call(self, output, duration_ms=100, trace=None, call_id="test"):
        """Helper to create a FunctionCall for testing."""
        from evalyn_sdk.models import FunctionCall

        return FunctionCall(
            id=call_id,
            function_name="test",
            inputs={},
            output=output,
            error=None,
            started_at=datetime.now(timezone.utc),
            ended_at=datetime.now(timezone.utc),
            duration_ms=duration_ms,
            session_id=None,
            trace=trace or [],
        )

    def _make_item(self, expected=None, item_id="item-1"):
        """Helper to create a DatasetItem for testing."""
        from evalyn_sdk.models import DatasetItem

        return DatasetItem(
            id=item_id,
            input={"query": "test"},
            output=expected,
        )

    def test_latency_metric(self):
        """Test latency_ms metric."""
        from evalyn_sdk import latency_metric

        metric = latency_metric()
        assert metric is not None
        assert metric.spec.id == "latency_ms"

        call = self._make_call("result", duration_ms=150.5)
        item = self._make_item()

        result = metric.handler(call, item)
        assert result.score == 150.5

    def test_json_valid_metric_valid(self):
        """Test json_valid metric with valid JSON."""
        from evalyn_sdk import json_valid_metric

        metric = json_valid_metric()
        call = self._make_call('{"key": "value"}')
        item = self._make_item()

        result = metric.handler(call, item)
        assert result.passed is True

    def test_json_valid_metric_invalid(self):
        """Test json_valid metric with invalid JSON."""
        from evalyn_sdk import json_valid_metric

        metric = json_valid_metric()
        call = self._make_call("not json")
        item = self._make_item()

        result = metric.handler(call, item)
        assert result.passed is False

    def test_exact_match_metric(self):
        """Test exact_match metric."""
        from evalyn_sdk import exact_match_metric

        metric = exact_match_metric()

        # Match
        call = self._make_call("expected output")
        item = self._make_item(expected="expected output")
        result = metric.handler(call, item)
        assert result.passed is True
        assert result.score == 1.0

        # No match
        item2 = self._make_item(expected="different output")
        result2 = metric.handler(call, item2)
        assert result2.passed is False
        assert result2.score == 0.0

    def test_regex_match_metric(self):
        """Test regex_match metric."""
        from evalyn_sdk import regex_match_metric

        metric = regex_match_metric(pattern=r"\d+")
        call = self._make_call("The answer is 42.")
        item = self._make_item()

        result = metric.handler(call, item)
        assert result.passed is True

    def test_token_length_metric(self):
        """Test token_length metric."""
        from evalyn_sdk import token_length_metric

        metric = token_length_metric(max_chars=100)
        call = self._make_call("short")
        item = self._make_item()

        result = metric.handler(call, item)
        assert result.passed is True

    def test_tool_call_count_metric(self):
        """Test tool_call_count metric."""
        from evalyn_sdk import tool_call_count_metric
        from evalyn_sdk.models import TraceEvent

        metric = tool_call_count_metric()

        trace = [
            TraceEvent(kind="tool.call", timestamp=datetime.now(timezone.utc), detail={}),
            TraceEvent(kind="tool.result", timestamp=datetime.now(timezone.utc), detail={}),
            TraceEvent(kind="tool.call", timestamp=datetime.now(timezone.utc), detail={}),
        ]
        call = self._make_call("result", trace=trace)
        item = self._make_item()

        result = metric.handler(call, item)
        # Counts all tool-related events (tool.call, tool.result, etc.)
        assert result.score == 3


class TestMetricTemplates:
    """Tests for metric templates."""

    def test_objective_templates_exist(self):
        """Test objective templates are defined."""
        from evalyn_sdk import OBJECTIVE_REGISTRY
        assert isinstance(OBJECTIVE_REGISTRY, list)
        assert len(OBJECTIVE_REGISTRY) > 0
        # Check some expected templates exist
        ids = [t["id"] for t in OBJECTIVE_REGISTRY]
        assert "latency_ms" in ids
        assert "json_valid" in ids

    def test_subjective_templates_exist(self):
        """Test subjective templates are defined."""
        from evalyn_sdk import SUBJECTIVE_REGISTRY
        assert isinstance(SUBJECTIVE_REGISTRY, list)
        assert len(SUBJECTIVE_REGISTRY) > 0

    def test_list_template_ids(self):
        """Test list_template_ids function."""
        from evalyn_sdk import list_template_ids
        ids = list_template_ids()
        assert isinstance(ids, list)
        assert "latency_ms" in ids
        assert "json_valid" in ids

    def test_build_objective_metric(self):
        """Test building an objective metric from template."""
        from evalyn_sdk import build_objective_metric

        metric = build_objective_metric("latency_ms")
        assert metric is not None
        assert metric.spec.id == "latency_ms"
        assert metric.spec.type == "objective"

    def test_build_objective_metric_with_config(self):
        """Test building metric with custom config."""
        from evalyn_sdk import build_objective_metric

        metric = build_objective_metric("token_length", config={"max_chars": 500})
        assert metric is not None

    def test_build_invalid_template(self):
        """Test building from invalid template ID."""
        from evalyn_sdk import build_objective_metric

        with pytest.raises((ValueError, KeyError)):
            build_objective_metric("nonexistent_template")


class TestJudges:
    """Tests for LLM judges."""

    def test_echo_judge_creation(self):
        """Test EchoJudge creation."""
        from evalyn_sdk import EchoJudge

        judge = EchoJudge()
        assert judge is not None

    def test_llm_judge_base(self):
        """Test LLMJudge base class exists."""
        from evalyn_sdk import LLMJudge
        assert LLMJudge is not None


class TestSuggesters:
    """Tests for metric suggesters."""

    def test_heuristic_suggester(self):
        """Test HeuristicSuggester creation."""
        from evalyn_sdk import HeuristicSuggester

        suggester = HeuristicSuggester()
        assert suggester is not None


class TestMetricSpec:
    """Tests for MetricSpec model."""

    def test_metric_spec_creation(self):
        """Test creating a MetricSpec."""
        from evalyn_sdk import MetricSpec

        spec = MetricSpec(
            id="test_spec",
            type="objective",
            name="Test Spec",
            description="A test specification",
        )

        assert spec.id == "test_spec"
        assert spec.type == "objective"
        assert spec.name == "Test Spec"

    def test_metric_spec_with_config(self):
        """Test MetricSpec with config."""
        from evalyn_sdk import MetricSpec

        spec = MetricSpec(
            id="configurable",
            type="objective",
            name="Configurable Metric",
            description="Has config",
            config={"threshold": 0.5, "max_retries": 3},
        )

        assert spec.config["threshold"] == 0.5
        assert spec.config["max_retries"] == 3


class TestMetricResult:
    """Tests for MetricResult model."""

    def test_metric_result_creation(self):
        """Test creating a MetricResult."""
        from evalyn_sdk import MetricResult

        result = MetricResult(
            metric_id="test_metric",
            item_id="item-1",
            call_id="call-1",
            score=0.85,
            passed=True,
            details={"reason": "Good output"},
        )

        assert result.metric_id == "test_metric"
        assert result.score == 0.85
        assert result.passed is True

    def test_metric_result_serialization(self):
        """Test MetricResult serialization."""
        from evalyn_sdk import MetricResult

        result = MetricResult(
            metric_id="serialize_result",
            item_id="item-1",
            call_id="call-1",
            score=0.5,
            passed=False,
            details={"issues": ["too short", "missing context"]},
        )

        data = result.as_dict()
        assert data["score"] == 0.5
        assert data["passed"] is False

        restored = MetricResult.from_dict(data)
        assert restored.score == result.score
        assert restored.item_id == result.item_id
