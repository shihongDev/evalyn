"""
Extended tests for model roundtrips in evalyn_sdk.models.

Run with: pytest tests/test_models_extended.py -v
"""
from __future__ import annotations

import pytest

from conftest import T0, T1, T2
from evalyn_sdk.models import (
    Annotation,
    AnnotationItem,
    CalibrationRecord,
    DatasetItem,
    EvalRun,
    EvalUnit,
    EvalView,
    FunctionCall,
    HumanLabel,
    JudgeConfig,
    MetricLabel,
    MetricResult,
    MetricSpec,
    Span,
    TraceEvent,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_span(
    span_id="span-001",
    name="llm_call",
    span_type="llm_call",
    parent_id=None,
    start=T0,
    end=T1,
    status="ok",
    attrs=None,
):
    return Span(
        id=span_id,
        name=name,
        span_type=span_type,
        parent_id=parent_id,
        start_time=start,
        end_time=end,
        status=status,
        attributes=attrs or {},
    )


def _make_trace_event(kind="llm_start", ts=T0, detail=None, span_id=None):
    return TraceEvent(
        kind=kind,
        timestamp=ts,
        detail=detail or {"model": "gemini-2.5-flash"},
        span_id=span_id,
    )


def _make_function_call(**overrides):
    defaults = dict(
        id="call-abc-123",
        function_name="run_agent",
        inputs={"query": "What is the capital of France?", "max_tokens": 256},
        output={"answer": "Paris", "confidence": 0.98},
        error=None,
        started_at=T0,
        ended_at=T2,
        duration_ms=3000.0,
        session_id="sess-xyz",
        trace=[_make_trace_event()],
        metadata={"project_id": "demo", "version": "v3"},
        parent_call_id=None,
        spans=[_make_span()],
    )
    defaults.update(overrides)
    return FunctionCall(**defaults)


# ---------------------------------------------------------------------------
# Span tests
# ---------------------------------------------------------------------------

class TestSpan:
    def test_roundtrip(self):
        span = _make_span(attrs={"model": "gemini-2.5-flash", "tokens": 42})
        d = span.as_dict()
        restored = Span.from_dict(d)
        assert restored.id == span.id
        assert restored.name == span.name
        assert restored.span_type == span.span_type
        assert restored.parent_id == span.parent_id
        assert restored.status == span.status
        assert restored.attributes == span.attributes
        # Timestamps survive ISO round-trip
        assert restored.start_time.isoformat() == span.start_time.isoformat()
        assert restored.end_time.isoformat() == span.end_time.isoformat()

    def test_duration_ms_computed(self):
        span = _make_span(start=T0, end=T1)
        assert span.duration_ms == pytest.approx(1500.0)

    def test_duration_ms_none_when_no_end(self):
        span = _make_span(end=None)
        assert span.duration_ms is None

    def test_duration_ms_in_as_dict(self):
        span = _make_span()
        d = span.as_dict()
        assert "duration_ms" in d
        assert d["duration_ms"] == pytest.approx(1500.0)

    def test_new_factory(self):
        span = Span.new("generate_query", "node", parent_id="root-1", model="gpt-4o")
        assert span.name == "generate_query"
        assert span.span_type == "node"
        assert span.parent_id == "root-1"
        assert span.status == "running"
        assert span.end_time is None
        assert span.attributes["model"] == "gpt-4o"
        assert len(span.id) > 0  # auto-generated UUID

    def test_finish_sets_end_and_status(self):
        span = Span.new("task", "custom")
        assert span.end_time is None
        assert span.status == "running"
        returned = span.finish(status="error", reason="timeout")
        assert span.end_time is not None
        assert span.status == "error"
        assert span.attributes["reason"] == "timeout"
        assert returned is span  # returns self

    def test_from_dict_defaults(self):
        minimal = {"id": "s1", "name": "test"}
        span = Span.from_dict(minimal)
        assert span.span_type == "custom"
        assert span.status == "ok"
        assert span.parent_id is None
        assert span.attributes == {}


# ---------------------------------------------------------------------------
# FunctionCall tests
# ---------------------------------------------------------------------------

class TestFunctionCall:
    def test_roundtrip_with_spans_and_events(self):
        call = _make_function_call()
        d = call.as_dict()
        restored = FunctionCall.from_dict(d)
        assert restored.id == call.id
        assert restored.function_name == call.function_name
        assert restored.inputs == call.inputs
        assert restored.output == call.output
        assert restored.error is None
        assert restored.duration_ms == 3000.0
        assert restored.session_id == call.session_id
        assert restored.metadata == call.metadata
        assert len(restored.trace) == 1
        assert restored.trace[0].kind == "llm_start"
        assert len(restored.spans) == 1
        assert restored.spans[0].id == "span-001"

    def test_roundtrip_preserves_parent_call_id(self):
        call = _make_function_call(parent_call_id="parent-999")
        d = call.as_dict()
        assert d["parent_call_id"] == "parent-999"
        restored = FunctionCall.from_dict(d)
        assert restored.parent_call_id == "parent-999"

    def test_as_dict_omits_parent_and_spans_when_empty(self):
        call = _make_function_call(parent_call_id=None, spans=[])
        d = call.as_dict()
        assert "parent_call_id" not in d
        assert "spans" not in d

    def test_new_factory(self):
        call = FunctionCall.new(
            "summarize",
            {"text": "long document..."},
            session_id="s1",
            metadata={"env": "test"},
            parent_call_id="p1",
        )
        assert call.function_name == "summarize"
        assert call.output is None
        assert call.error is None
        assert call.ended_at is None
        assert call.duration_ms is None
        assert call.parent_call_id == "p1"
        assert call.spans == []

    def test_add_span(self):
        call = _make_function_call(spans=[])
        assert len(call.spans) == 0
        span = _make_span(span_id="new-span")
        call.add_span(span)
        assert len(call.spans) == 1
        assert call.spans[0].id == "new-span"

    def test_get_span_tree_parent_child(self):
        root = _make_span(span_id="root", name="agent", span_type="agent")
        child1 = _make_span(span_id="c1", name="node1", span_type="node", parent_id="root")
        child2 = _make_span(span_id="c2", name="node2", span_type="node", parent_id="root")
        grandchild = _make_span(span_id="gc1", name="llm", span_type="llm_call", parent_id="c1")

        call = _make_function_call(spans=[root, child1, child2, grandchild])
        tree = call.get_span_tree()

        assert len(tree["roots"]) == 1
        root_node = tree["roots"][0]
        assert root_node["span"].id == "root"
        assert len(root_node["children"]) == 2

        child_ids = {c["span"].id for c in root_node["children"]}
        assert child_ids == {"c1", "c2"}

        c1_node = tree["by_id"]["c1"]
        assert len(c1_node["children"]) == 1
        assert c1_node["children"][0]["span"].id == "gc1"

    def test_get_span_tree_multiple_roots(self):
        s1 = _make_span(span_id="r1", name="root1", span_type="custom")
        s2 = _make_span(span_id="r2", name="root2", span_type="custom")
        call = _make_function_call(spans=[s1, s2])
        tree = call.get_span_tree()
        assert len(tree["roots"]) == 2


# ---------------------------------------------------------------------------
# EvalUnit tests
# ---------------------------------------------------------------------------

class TestEvalUnit:
    def test_roundtrip(self):
        unit = EvalUnit(
            id="unit-1",
            unit_type="single_turn",
            call_id="call-abc-123",
            span_ids=["span-001", "span-002"],
            context={"turn_index": 0, "model": "gpt-4o"},
        )
        d = unit.as_dict()
        restored = EvalUnit.from_dict(d)
        assert restored.id == unit.id
        assert restored.unit_type == unit.unit_type
        assert restored.call_id == unit.call_id
        assert restored.span_ids == unit.span_ids
        assert restored.context == unit.context


# ---------------------------------------------------------------------------
# EvalView tests
# ---------------------------------------------------------------------------

class TestEvalView:
    def test_as_dict(self):
        view = EvalView(
            unit_id="unit-1",
            unit_type="tool_use",
            input={"tool": "search", "query": "weather"},
            output={"results": ["sunny", "25C"]},
            context={"turn": 3},
        )
        d = view.as_dict()
        assert d["unit_id"] == "unit-1"
        assert d["unit_type"] == "tool_use"
        assert d["input"]["tool"] == "search"
        assert d["output"]["results"][0] == "sunny"
        assert d["context"]["turn"] == 3


# ---------------------------------------------------------------------------
# DatasetItem tests
# ---------------------------------------------------------------------------

class TestDatasetItem:
    def test_from_payload_new_format(self):
        payload = {
            "id": "item-1",
            "input": {"query": "hello"},
            "output": {"answer": "hi"},
            "human_label": {"passed": True},
            "metadata": {"source": "prod"},
        }
        item = DatasetItem.from_payload(payload)
        assert item.id == "item-1"
        assert item.input == {"query": "hello"}
        assert item.output == {"answer": "hi"}
        # Backwards compat fields sync
        assert item.inputs == {"query": "hello"}
        assert item.expected == {"answer": "hi"}

    def test_from_payload_old_format(self):
        payload = {
            "id": "item-2",
            "inputs": {"question": "2+2?"},
            "expected": "4",
        }
        item = DatasetItem.from_payload(payload)
        assert item.input == {"question": "2+2?"}
        assert item.output == "4"
        assert item.inputs == {"question": "2+2?"}
        assert item.expected == "4"

    def test_backwards_compat_inputs_syncs_to_input(self):
        item = DatasetItem(id="x", inputs={"a": 1})
        assert item.input == {"a": 1}

    def test_backwards_compat_input_syncs_to_inputs(self):
        item = DatasetItem(id="y", input={"b": 2})
        assert item.inputs == {"b": 2}

    def test_from_call(self):
        call = _make_function_call()
        item = DatasetItem.from_call(call)
        assert item.input == call.inputs
        assert item.output == call.output
        assert item.metadata["call_id"] == call.id
        assert item.metadata["function"] == call.function_name
        assert item.metadata["duration_ms"] == call.duration_ms
        assert item.expected == call.output

    def test_as_dict_uses_new_format(self):
        item = DatasetItem.from_payload({"id": "d1", "input": {"q": "hi"}, "output": "bye"})
        d = item.as_dict()
        assert "input" in d
        assert "output" in d
        # Old keys not in as_dict output
        assert "inputs" not in d
        assert "expected" not in d


# ---------------------------------------------------------------------------
# HumanLabel tests
# ---------------------------------------------------------------------------

class TestHumanLabel:
    def test_roundtrip(self):
        label = HumanLabel(
            passed=False,
            scores={"helpfulness": 0.3, "accuracy": 0.8},
            notes="Partially correct but missed key detail",
            annotator="reviewer-1",
            timestamp=T0,
        )
        d = label.as_dict()
        restored = HumanLabel.from_dict(d)
        assert restored.passed is False
        assert restored.scores == {"helpfulness": 0.3, "accuracy": 0.8}
        assert restored.notes == "Partially correct but missed key detail"
        assert restored.annotator == "reviewer-1"

    def test_from_dict_defaults(self):
        restored = HumanLabel.from_dict({})
        assert restored.passed is True
        assert restored.scores == {}
        assert restored.notes == ""
        assert restored.annotator == ""


# ---------------------------------------------------------------------------
# AnnotationItem tests
# ---------------------------------------------------------------------------

class TestAnnotationItem:
    def test_roundtrip_with_human_label(self):
        label = HumanLabel(passed=True, scores={"accuracy": 0.95}, annotator="alice")
        item = AnnotationItem(
            id="ann-item-1",
            input={"query": "capital of Japan?"},
            output="Tokyo",
            eval_results={"accuracy": {"score": 0.9, "passed": True}},
            human_label=label,
            metadata={"run_id": "run-42"},
        )
        d = item.as_dict()
        restored = AnnotationItem.from_dict(d)
        assert restored.id == item.id
        assert restored.input == item.input
        assert restored.output == item.output
        assert restored.eval_results == item.eval_results
        assert restored.human_label is not None
        assert restored.human_label.passed is True
        assert restored.human_label.scores["accuracy"] == 0.95
        assert restored.metadata == item.metadata

    def test_roundtrip_without_human_label(self):
        item = AnnotationItem(
            id="ann-item-2",
            input={"text": "summarize this"},
            output="summary here",
        )
        d = item.as_dict()
        assert d["human_label"] is None
        restored = AnnotationItem.from_dict(d)
        assert restored.human_label is None


# ---------------------------------------------------------------------------
# MetricLabel tests
# ---------------------------------------------------------------------------

class TestMetricLabel:
    def test_roundtrip(self):
        ml = MetricLabel(
            metric_id="helpfulness",
            agree_with_llm=False,
            human_label=True,
            notes="LLM was too harsh",
        )
        d = ml.as_dict()
        restored = MetricLabel.from_dict(d)
        assert restored.metric_id == "helpfulness"
        assert restored.agree_with_llm is False
        assert restored.human_label is True
        assert restored.notes == "LLM was too harsh"

    def test_from_dict_defaults(self):
        restored = MetricLabel.from_dict({})
        assert restored.metric_id == ""
        assert restored.agree_with_llm is True
        assert restored.human_label is True
        assert restored.notes == ""


# ---------------------------------------------------------------------------
# Annotation tests
# ---------------------------------------------------------------------------

class TestAnnotation:
    def test_roundtrip_with_metric_labels(self):
        ann = Annotation(
            id="ann-1",
            target_id="item-42",
            label=True,
            rationale="Output is factually correct and well-structured",
            annotator="bob",
            source="human",
            confidence=4,
            metric_labels={
                "accuracy": MetricLabel("accuracy", True, True, ""),
                "helpfulness": MetricLabel("helpfulness", False, True, "Should be higher"),
            },
            created_at=T0,
        )
        d = ann.as_dict()
        restored = Annotation.from_dict(d)
        assert restored.id == "ann-1"
        assert restored.target_id == "item-42"
        assert restored.label is True
        assert restored.rationale == "Output is factually correct and well-structured"
        assert restored.annotator == "bob"
        assert restored.confidence == 4
        assert len(restored.metric_labels) == 2
        assert restored.metric_labels["helpfulness"].notes == "Should be higher"

    def test_roundtrip_without_metric_labels(self):
        ann = Annotation(
            id="ann-2",
            target_id="item-99",
            label=False,
            rationale="Incorrect answer",
            annotator="carol",
        )
        d = ann.as_dict()
        restored = Annotation.from_dict(d)
        assert restored.label is False
        assert restored.metric_labels == {}


# ---------------------------------------------------------------------------
# CalibrationRecord tests
# ---------------------------------------------------------------------------

class TestCalibrationRecord:
    def test_as_dict(self):
        rec = CalibrationRecord(
            id="cal-1",
            judge_config_id="judge-v2",
            gold_items=["item-1", "item-2", "item-3"],
            adjustments={"threshold": 0.7, "prompt_patch": "Be stricter"},
            created_at=T0,
            usage_summary={"total_tokens": 15000, "total_cost_usd": 0.03},
        )
        d = rec.as_dict()
        assert d["id"] == "cal-1"
        assert d["judge_config_id"] == "judge-v2"
        assert len(d["gold_items"]) == 3
        assert d["adjustments"]["threshold"] == 0.7
        assert d["created_at"] == T0.isoformat()
        assert d["usage_summary"]["total_tokens"] == 15000


# ---------------------------------------------------------------------------
# EvalRun tests
# ---------------------------------------------------------------------------

class TestEvalRun:
    def test_roundtrip(self):
        metric_results = [
            MetricResult(
                metric_id="accuracy",
                item_id="item-1",
                call_id="call-abc-123",
                score=0.95,
                passed=True,
                details={"reason": "exact match"},
                input_tokens=100,
                output_tokens=50,
                model="gpt-4o",
            ),
            MetricResult(
                metric_id="helpfulness",
                item_id="item-1",
                call_id="call-abc-123",
                score=0.8,
                passed=True,
                details={"reason": "good but verbose"},
                unit_id="unit-1",
                unit_type="single_turn",
                span_ids=["span-001"],
            ),
        ]
        metrics = [
            MetricSpec(
                id="accuracy",
                name="Accuracy",
                type="objective",
                description="Exact match check",
            ),
            MetricSpec(
                id="helpfulness",
                name="Helpfulness",
                type="subjective",
                description="LLM judge for helpfulness",
                unit_types=["outcome", "single_turn"],
            ),
        ]
        judge_configs = [
            JudgeConfig(
                id="judge-v1",
                model="gpt-4o",
                prompt="Rate the helpfulness 1-5",
                parameters={"temperature": 0.0},
                version="v1",
            ),
        ]
        run = EvalRun(
            id="run-001",
            dataset_name="qa-benchmark",
            created_at=T0,
            metric_results=metric_results,
            metrics=metrics,
            judge_configs=judge_configs,
            summary={"total": 10, "passed": 8, "pass_rate": 0.8},
            usage_summary={"total_input_tokens": 5000, "total_output_tokens": 2000},
        )
        d = run.as_dict()
        restored = EvalRun.from_dict(d)

        assert restored.id == "run-001"
        assert restored.dataset_name == "qa-benchmark"
        assert len(restored.metric_results) == 2
        assert restored.metric_results[0].score == 0.95
        assert restored.metric_results[1].unit_type == "single_turn"
        assert len(restored.metrics) == 2
        assert restored.metrics[0].name == "Accuracy"
        assert restored.metrics[1].unit_types == ["outcome", "single_turn"]
        assert len(restored.judge_configs) == 1
        assert restored.judge_configs[0].model == "gpt-4o"
        assert restored.summary["pass_rate"] == 0.8
        # usage_summary is not stored in DB schema but is on the model
        # from_dict restores it if present
        assert restored.usage_summary == run.usage_summary


# ---------------------------------------------------------------------------
# MetricSpec tests
# ---------------------------------------------------------------------------

class TestMetricSpec:
    def test_metric_spec_roundtrip(self):
        spec = MetricSpec(
            id="helpfulness",
            name="Helpfulness",
            type="subjective",
            description="Rates helpfulness of response",
            config={"temperature": 0.0, "model": "gpt-4o"},
            why="Core quality signal",
            unit_types=["outcome", "single_turn"],
        )
        d = spec.as_dict()
        restored = MetricSpec.from_dict(d)
        assert restored.id == "helpfulness"
        assert restored.name == "Helpfulness"
        assert restored.type == "subjective"
        assert restored.description == "Rates helpfulness of response"
        assert restored.config == {"temperature": 0.0, "model": "gpt-4o"}
        assert restored.why == "Core quality signal"
        assert restored.unit_types == ["outcome", "single_turn"]

    def test_metric_spec_from_dict_defaults(self):
        restored = MetricSpec.from_dict({"id": "m1", "name": "M1"})
        assert restored.type == "objective"
        assert restored.description == ""
        assert restored.config == {}
        assert restored.why == ""
        assert restored.unit_types == ["outcome"]

    def test_metric_spec_from_dict_ignores_extra_keys(self):
        data = {
            "id": "m2",
            "name": "M2",
            "unknown_field": "should not crash",
            "extra": 42,
        }
        restored = MetricSpec.from_dict(data)
        assert restored.id == "m2"
        assert restored.name == "M2"


# ---------------------------------------------------------------------------
# JudgeConfig tests
# ---------------------------------------------------------------------------

class TestJudgeConfig:
    def test_judge_config_roundtrip(self):
        jc = JudgeConfig(
            id="judge-v2",
            model="gemini-2.5-flash",
            prompt="Rate quality 1-5",
            parameters={"temperature": 0.1, "max_tokens": 200},
            version="v2",
        )
        d = jc.as_dict()
        restored = JudgeConfig.from_dict(d)
        assert restored.id == "judge-v2"
        assert restored.model == "gemini-2.5-flash"
        assert restored.prompt == "Rate quality 1-5"
        assert restored.parameters == {"temperature": 0.1, "max_tokens": 200}
        assert restored.version == "v2"

    def test_judge_config_from_dict_defaults(self):
        restored = JudgeConfig.from_dict({"id": "j1", "model": "gpt-4o"})
        assert restored.prompt == ""
        assert restored.parameters == {}
        assert restored.version == "v0"

    def test_judge_config_from_dict_ignores_extra_keys(self):
        data = {
            "id": "j2",
            "model": "claude-3",
            "unrecognized": True,
            "foo": [1, 2, 3],
        }
        restored = JudgeConfig.from_dict(data)
        assert restored.id == "j2"
        assert restored.model == "claude-3"


# ---------------------------------------------------------------------------
# MetricResult error-path tests
# ---------------------------------------------------------------------------

class TestMetricResultErrorPath:
    def test_metric_result_requires_item_id(self):
        """MetricResult must have item_id - the error handler in execution.py
        uses item_id_hint to ensure it is always set."""
        result = MetricResult(
            metric_id="accuracy",
            item_id="item-42",
            call_id="error-item-42",
            score=None,
            passed=False,
            details={"error": "something failed", "error_type": "RuntimeError"},
        )
        assert result.item_id == "item-42"
        assert result.score is None
        assert result.passed is False
        assert result.details["error_type"] == "RuntimeError"

        # Verify it roundtrips correctly even with error data
        d = result.as_dict()
        restored = MetricResult.from_dict(d)
        assert restored.item_id == "item-42"
        assert restored.call_id == "error-item-42"
        assert restored.details["error"] == "something failed"


# ---------------------------------------------------------------------------
# TraceEvent tests
# ---------------------------------------------------------------------------

class TestTraceEvent:
    def test_roundtrip(self):
        ev = TraceEvent(
            kind="llm_response",
            timestamp=T1,
            detail={"model": "gpt-4o", "tokens": 128, "finish_reason": "stop"},
            span_id="span-001",
            parent_span_id="root-span",
        )
        d = ev.as_dict()
        restored = TraceEvent.from_dict(d)
        assert restored.kind == "llm_response"
        assert restored.detail["tokens"] == 128
        assert restored.span_id == "span-001"
        assert restored.parent_span_id == "root-span"

    def test_as_dict_omits_none_span_ids(self):
        ev = TraceEvent(kind="start", timestamp=T0)
        d = ev.as_dict()
        assert "span_id" not in d
        assert "parent_span_id" not in d
