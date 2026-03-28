"""Tests for public tracing API and metric benchmarking."""

import pytest
from evalyn_sdk.tracing_api import trace_span, trace_llm_call, trace_tool_call, trace_retrieval
from evalyn_sdk.models import MetricResult, MetricSpec
from evalyn_sdk.analysis.metric_benchmark import (
    benchmark_metrics, MetricBenchmark, BenchmarkReport,
)


# =============================================================================
# Tracing API tests
# =============================================================================

class TestTraceSpan:
    def test_basic_span(self):
        with trace_span("my_operation") as s:
            assert s is not None
            assert s.name == "my_operation"
            assert s.span_type == "custom"

    def test_span_with_type(self):
        with trace_span("my_op", "agent") as s:
            assert s.span_type == "agent"

    def test_span_with_attributes(self):
        with trace_span("my_op", "custom", key1="value1", key2=42) as s:
            assert s.attributes.get("key1") == "value1"
            assert s.attributes.get("key2") == 42

    def test_span_attribute_mutation(self):
        with trace_span("my_op") as s:
            s.attributes["result"] = "success"
            assert s.attributes["result"] == "success"

    def test_span_has_id(self):
        with trace_span("test") as s:
            assert s.id is not None
            assert len(s.id) > 0

    def test_span_has_start_time(self):
        with trace_span("test") as s:
            assert s.start_time is not None

    def test_nested_spans(self):
        with trace_span("outer") as outer:
            with trace_span("inner") as inner:
                assert inner.parent_id == outer.id


class TestTraceLLMCall:
    def test_basic(self):
        with trace_llm_call("gpt-4") as s:
            assert s.span_type == "llm_call"
            assert s.attributes.get("model") == "gpt-4"
            assert "gpt-4" in s.name

    def test_with_attributes(self):
        with trace_llm_call("claude-3", temperature=0.7) as s:
            assert s.attributes["temperature"] == 0.7


class TestTraceToolCall:
    def test_basic(self):
        with trace_tool_call("search") as s:
            assert s.span_type == "tool_call"
            assert s.attributes.get("tool.name") == "search"
            assert "search" in s.name


class TestTraceRetrieval:
    def test_basic(self):
        with trace_retrieval("what is python?", source="chroma") as s:
            assert s.span_type == "retrieval"
            assert s.attributes.get("retrieval.query") == "what is python?"
            assert s.attributes.get("retrieval.source") == "chroma"


# =============================================================================
# Metric benchmark tests
# =============================================================================

def _result(metric_id, input_tokens=0, output_tokens=0, model=None):
    return MetricResult(
        metric_id=metric_id, item_id="i1", call_id="c1",
        score=1.0, passed=True, details={},
        input_tokens=input_tokens, output_tokens=output_tokens, model=model,
    )


class TestBenchmarkMetrics:
    def test_objective_free(self):
        results = [_result("json_valid"), _result("json_valid")]
        specs = [MetricSpec(id="json_valid", name="json", type="objective")]
        report = benchmark_metrics(results, specs)
        assert len(report.metrics) == 1
        assert report.metrics[0].total_cost_usd == 0.0

    def test_subjective_has_cost(self):
        results = [
            _result("help", 500, 100, "gemini-2.5-flash-lite"),
            _result("help", 600, 120, "gemini-2.5-flash-lite"),
        ]
        specs = [MetricSpec(id="help", name="help", type="subjective")]
        report = benchmark_metrics(results, specs)
        assert report.metrics[0].total_input_tokens == 1100
        assert report.total_cost_usd > 0

    def test_multiple_metrics(self):
        results = [_result("a"), _result("a"), _result("b", 100, 50)]
        specs = [
            MetricSpec(id="a", name="a", type="objective"),
            MetricSpec(id="b", name="b", type="subjective"),
        ]
        report = benchmark_metrics(results, specs)
        assert len(report.metrics) == 2

    def test_most_expensive(self):
        results = [_result("cheap"), _result("exp", 10000, 5000, "gpt-4o")]
        report = benchmark_metrics(results)
        assert report.most_expensive.metric_id == "exp"

    def test_avg_tokens(self):
        results = [_result("m", 100, 50)] * 4
        report = benchmark_metrics(results)
        assert report.metrics[0].avg_tokens_per_item == 150

    def test_empty(self):
        report = benchmark_metrics([])
        assert len(report.metrics) == 0
        assert report.most_expensive is None

    def test_as_dict(self):
        results = [_result("m", 100, 50)]
        d = benchmark_metrics(results).as_dict()
        assert "total_cost_usd" in d
        assert "metrics" in d

    def test_format_text(self):
        results = [_result("json_valid"), _result("help", 500, 100)]
        specs = [
            MetricSpec(id="json_valid", name="json", type="objective"),
            MetricSpec(id="help", name="help", type="subjective"),
        ]
        text = benchmark_metrics(results, specs).format_text()
        assert "Benchmark Report" in text

    def test_zero_items_benchmark(self):
        bench = MetricBenchmark(metric_id="m", metric_type="objective")
        assert bench.avg_cost_per_item == 0.0
        assert bench.avg_tokens_per_item == 0
