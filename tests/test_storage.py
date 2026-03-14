"""
Tests for SQLiteStorage in evalyn_sdk.storage.sqlite.

Run with: pytest tests/test_storage.py -v
"""
from __future__ import annotations

from datetime import timedelta

import pytest

from conftest import T0, T1, T2, T3
from evalyn_sdk.models import (
    Annotation,
    EvalRun,
    FunctionCall,
    JudgeConfig,
    MetricLabel,
    MetricResult,
    MetricSpec,
    Span,
    TraceEvent,
)
from evalyn_sdk.storage.sqlite import SQLiteStorage


@pytest.fixture
def temp_db(tmp_path):
    db_path = tmp_path / "test_traces.sqlite"
    storage = SQLiteStorage(path=str(db_path))
    yield storage
    storage.close()


def _make_span(span_id="span-001", name="llm_call", parent_id=None):
    return Span(
        id=span_id,
        name=name,
        span_type="llm_call",
        parent_id=parent_id,
        start_time=T0,
        end_time=T1,
        status="ok",
        attributes={"model": "gemini-2.5-flash", "tokens": 128},
    )


def _make_trace_event(kind="llm_start", ts=T0, span_id=None):
    return TraceEvent(
        kind=kind,
        timestamp=ts,
        detail={"model": "gemini-2.5-flash", "temperature": 0.0},
        span_id=span_id,
    )


def _make_call(
    call_id="call-001",
    function_name="run_agent",
    project_id="demo-project",
    session_id="sess-1",
    started_at=T0,
    ended_at=T2,
    with_spans=True,
    with_trace=True,
    parent_call_id=None,
):
    spans = [
        _make_span("span-root", "agent"),
        _make_span("span-child", "llm_call", parent_id="span-root"),
    ] if with_spans else []
    trace = [
        _make_trace_event("llm_start", T0, span_id="span-root"),
        _make_trace_event("llm_end", T1, span_id="span-child"),
    ] if with_trace else []
    return FunctionCall(
        id=call_id,
        function_name=function_name,
        inputs={"query": "What is 2+2?", "max_tokens": 256},
        output={"answer": "4", "confidence": 0.99},
        error=None,
        started_at=started_at,
        ended_at=ended_at,
        duration_ms=3000.0,
        session_id=session_id,
        trace=trace,
        metadata={"project_id": project_id, "version": "v3"},
        parent_call_id=parent_call_id,
        spans=spans,
    )


def _make_eval_run(
    run_id="run-001",
    dataset_name="qa-benchmark",
    created_at=T0,
):
    return EvalRun(
        id=run_id,
        dataset_name=dataset_name,
        created_at=created_at,
        metric_results=[
            MetricResult(
                metric_id="accuracy",
                item_id="item-1",
                call_id="call-001",
                score=0.95,
                passed=True,
                details={"reason": "exact match"},
                input_tokens=100,
                output_tokens=50,
                model="gpt-4o",
            ),
        ],
        metrics=[
            MetricSpec(
                id="accuracy",
                name="Accuracy",
                type="objective",
                description="Exact match",
            ),
        ],
        judge_configs=[
            JudgeConfig(
                id="judge-v1",
                model="gpt-4o",
                prompt="Check correctness",
                parameters={"temperature": 0.0},
            ),
        ],
        summary={"total": 5, "passed": 4, "pass_rate": 0.8},
    )


def _make_annotation(
    ann_id="ann-001",
    target_id="item-1",
    label=True,
    annotator="reviewer-1",
    confidence=4,
):
    return Annotation(
        id=ann_id,
        target_id=target_id,
        label=label,
        rationale="Correct and well-formed answer",
        annotator=annotator,
        source="human",
        confidence=confidence,
        metric_labels={
            "accuracy": MetricLabel("accuracy", True, True, ""),
        },
        created_at=T0,
    )


# ---------------------------------------------------------------------------
# Store/get call roundtrip
# ---------------------------------------------------------------------------

class TestStoreGetCall:
    def test_store_and_get_roundtrip(self, temp_db):
        call = _make_call()
        temp_db.store_call(call)
        restored = temp_db.get_call("call-001")

        assert restored is not None
        assert restored.id == "call-001"
        assert restored.function_name == "run_agent"
        assert restored.inputs["query"] == "What is 2+2?"
        assert restored.output["answer"] == "4"
        assert restored.error is None
        assert restored.duration_ms == 3000.0
        assert restored.session_id == "sess-1"
        assert restored.metadata["project_id"] == "demo-project"
        assert restored.parent_call_id is None

    def test_spans_survive_roundtrip(self, temp_db):
        call = _make_call()
        temp_db.store_call(call)
        restored = temp_db.get_call("call-001")

        assert len(restored.spans) == 2
        assert restored.spans[0].id == "span-root"
        assert restored.spans[1].id == "span-child"
        assert restored.spans[1].parent_id == "span-root"
        assert restored.spans[0].attributes["model"] == "gemini-2.5-flash"

    def test_trace_events_survive_roundtrip(self, temp_db):
        call = _make_call()
        temp_db.store_call(call)
        restored = temp_db.get_call("call-001")

        assert len(restored.trace) == 2
        assert restored.trace[0].kind == "llm_start"
        assert restored.trace[1].kind == "llm_end"
        assert restored.trace[0].detail["model"] == "gemini-2.5-flash"

    def test_get_nonexistent_returns_none(self, temp_db):
        assert temp_db.get_call("does-not-exist") is None

    def test_store_call_with_parent_call_id(self, temp_db):
        call = _make_call(call_id="child-call", parent_call_id="parent-call")
        temp_db.store_call(call)
        restored = temp_db.get_call("child-call")
        assert restored.parent_call_id == "parent-call"

    def test_store_call_without_spans(self, temp_db):
        call = _make_call(call_id="no-spans", with_spans=False, with_trace=False)
        temp_db.store_call(call)
        restored = temp_db.get_call("no-spans")
        assert restored.spans == []
        assert restored.trace == []


# ---------------------------------------------------------------------------
# list_calls with project filter
# ---------------------------------------------------------------------------

class TestListCalls:
    def test_list_all(self, temp_db):
        temp_db.store_call(_make_call("c1", started_at=T0))
        temp_db.store_call(_make_call("c2", started_at=T1))
        temp_db.store_call(_make_call("c3", started_at=T2))

        calls = temp_db.list_calls(limit=10)
        assert len(calls) == 3
        # Ordered by started_at DESC
        assert calls[0].id == "c3"
        assert calls[2].id == "c1"

    def test_list_with_limit(self, temp_db):
        for i in range(5):
            temp_db.store_call(
                _make_call(f"c{i}", started_at=T0 + timedelta(seconds=i))
            )
        calls = temp_db.list_calls(limit=2)
        assert len(calls) == 2

    def test_list_with_project_filter_project_id(self, temp_db):
        temp_db.store_call(_make_call("c1", project_id="alpha"))
        temp_db.store_call(_make_call("c2", project_id="beta"))
        temp_db.store_call(_make_call("c3", project_id="alpha"))

        calls = temp_db.list_calls(project="alpha")
        assert len(calls) == 2
        ids = {c.id for c in calls}
        assert ids == {"c1", "c3"}

    def test_list_with_project_filter_project_name(self, temp_db):
        call = _make_call("c-named")
        call.metadata = {"project_name": "my-project"}
        temp_db.store_call(call)

        calls = temp_db.list_calls(project="my-project")
        assert len(calls) == 1
        assert calls[0].id == "c-named"

    def test_list_with_project_filter_no_match(self, temp_db):
        temp_db.store_call(_make_call("c1", project_id="alpha"))
        calls = temp_db.list_calls(project="nonexistent")
        assert len(calls) == 0


# ---------------------------------------------------------------------------
# delete_calls
# ---------------------------------------------------------------------------

class TestDeleteCalls:
    def test_delete_calls(self, temp_db):
        temp_db.store_call(_make_call("c1"))
        temp_db.store_call(_make_call("c2"))
        temp_db.store_call(_make_call("c3"))

        deleted = temp_db.delete_calls(["c1", "c3"])
        assert deleted == 2
        assert temp_db.get_call("c1") is None
        assert temp_db.get_call("c2") is not None
        assert temp_db.get_call("c3") is None

    def test_delete_empty_list(self, temp_db):
        assert temp_db.delete_calls([]) == 0

    def test_delete_nonexistent(self, temp_db):
        deleted = temp_db.delete_calls(["no-such-id"])
        assert deleted == 0


# ---------------------------------------------------------------------------
# Store/list eval runs
# ---------------------------------------------------------------------------

class TestEvalRuns:
    def test_store_and_get_roundtrip(self, temp_db):
        run = _make_eval_run()
        temp_db.store_eval_run(run)
        restored = temp_db.get_eval_run("run-001")

        assert restored is not None
        assert restored.id == "run-001"
        assert restored.dataset_name == "qa-benchmark"
        assert len(restored.metric_results) == 1
        assert restored.metric_results[0].score == 0.95
        assert len(restored.metrics) == 1
        assert restored.metrics[0].name == "Accuracy"
        assert len(restored.judge_configs) == 1
        assert restored.judge_configs[0].model == "gpt-4o"
        assert restored.summary["pass_rate"] == 0.8

    def test_get_nonexistent_returns_none(self, temp_db):
        assert temp_db.get_eval_run("does-not-exist") is None

    def test_list_eval_runs(self, temp_db):
        temp_db.store_eval_run(_make_eval_run("r1", created_at=T0))
        temp_db.store_eval_run(_make_eval_run("r2", created_at=T1))
        temp_db.store_eval_run(_make_eval_run("r3", created_at=T2))

        runs = temp_db.list_eval_runs(limit=10)
        assert len(runs) == 3
        # Ordered by created_at DESC
        assert runs[0].id == "r3"

    def test_list_eval_runs_by_project(self, temp_db):
        temp_db.store_eval_run(_make_eval_run("r1", dataset_name="proj-a"))
        temp_db.store_eval_run(_make_eval_run("r2", dataset_name="proj-b"))
        temp_db.store_eval_run(_make_eval_run("r3", dataset_name="proj-a"))

        runs = temp_db.list_eval_runs_by_project("proj-a")
        assert len(runs) == 2
        ids = {r.id for r in runs}
        assert ids == {"r1", "r3"}

    def test_list_eval_runs_by_project_no_match(self, temp_db):
        temp_db.store_eval_run(_make_eval_run("r1", dataset_name="proj-a"))
        runs = temp_db.list_eval_runs_by_project("proj-z")
        assert len(runs) == 0


# ---------------------------------------------------------------------------
# Store/list annotations
# ---------------------------------------------------------------------------

class TestAnnotations:
    def test_store_and_list_all(self, temp_db):
        a1 = _make_annotation("ann-1", target_id="item-1")
        a2 = _make_annotation("ann-2", target_id="item-2")
        temp_db.store_annotations([a1, a2])

        anns = temp_db.list_annotations()
        assert len(anns) == 2

    def test_list_filter_by_target_id(self, temp_db):
        a1 = _make_annotation("ann-1", target_id="item-1")
        a2 = _make_annotation("ann-2", target_id="item-2")
        a3 = _make_annotation("ann-3", target_id="item-1")
        temp_db.store_annotations([a1, a2, a3])

        anns = temp_db.list_annotations(target_id="item-1")
        assert len(anns) == 2
        ids = {a.id for a in anns}
        assert ids == {"ann-1", "ann-3"}

    def test_annotation_fields_survive_roundtrip(self, temp_db):
        ann = _make_annotation(
            "ann-rt",
            target_id="item-5",
            label=False,
            annotator="bob",
            confidence=3,
        )
        temp_db.store_annotations([ann])
        restored_list = temp_db.list_annotations(target_id="item-5")
        assert len(restored_list) == 1
        r = restored_list[0]
        assert r.id == "ann-rt"
        assert r.target_id == "item-5"
        assert r.label is False
        assert r.rationale == "Correct and well-formed answer"
        assert r.annotator == "bob"
        assert r.source == "human"
        assert r.confidence == 3

    def test_list_annotations_empty(self, temp_db):
        anns = temp_db.list_annotations()
        assert anns == []

    def test_list_annotations_no_match(self, temp_db):
        temp_db.store_annotations([_make_annotation("a1", target_id="item-1")])
        anns = temp_db.list_annotations(target_id="item-999")
        assert anns == []


# ---------------------------------------------------------------------------
# resolve_call_id and resolve_eval_run_id
# ---------------------------------------------------------------------------

class TestResolveIds:
    def test_resolve_call_id_exact_match(self, temp_db):
        temp_db.store_call(_make_call("abcdef12-3456-7890-abcd-ef1234567890"))
        result = temp_db.resolve_call_id("abcdef12-3456-7890-abcd-ef1234567890")
        assert result == "abcdef12-3456-7890-abcd-ef1234567890"

    def test_resolve_call_id_prefix_unique(self, temp_db):
        temp_db.store_call(_make_call("abcdef12-3456-7890-abcd-ef1234567890"))
        temp_db.store_call(_make_call("xyz99912-3456-7890-abcd-ef1234567890"))
        result = temp_db.resolve_call_id("abcdef")
        assert result == "abcdef12-3456-7890-abcd-ef1234567890"

    def test_resolve_call_id_ambiguous_returns_none(self, temp_db):
        temp_db.store_call(_make_call("abc00001-1111-1111-1111-111111111111"))
        temp_db.store_call(_make_call("abc00002-2222-2222-2222-222222222222"))
        result = temp_db.resolve_call_id("abc")
        assert result is None

    def test_resolve_call_id_not_found(self, temp_db):
        result = temp_db.resolve_call_id("zzzzzzz")
        assert result is None

    def test_resolve_eval_run_id_exact_match(self, temp_db):
        temp_db.store_eval_run(_make_eval_run("run-aaa-111-bbb"))
        result = temp_db.resolve_eval_run_id("run-aaa-111-bbb")
        assert result == "run-aaa-111-bbb"

    def test_resolve_eval_run_id_prefix_unique(self, temp_db):
        temp_db.store_eval_run(_make_eval_run("run-aaa-111"))
        temp_db.store_eval_run(_make_eval_run("run-bbb-222"))
        result = temp_db.resolve_eval_run_id("run-aaa")
        assert result == "run-aaa-111"

    def test_resolve_eval_run_id_ambiguous_returns_none(self, temp_db):
        temp_db.store_eval_run(_make_eval_run("run-same-prefix-001"))
        temp_db.store_eval_run(_make_eval_run("run-same-prefix-002"))
        result = temp_db.resolve_eval_run_id("run-same-prefix")
        assert result is None

    def test_resolve_eval_run_id_not_found(self, temp_db):
        result = temp_db.resolve_eval_run_id("nonexistent")
        assert result is None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_store_call_replace_on_conflict(self, temp_db):
        """INSERT OR REPLACE should update existing call."""
        call1 = _make_call("same-id")
        temp_db.store_call(call1)

        call2 = _make_call("same-id", function_name="updated_agent")
        temp_db.store_call(call2)

        restored = temp_db.get_call("same-id")
        assert restored.function_name == "updated_agent"

    def test_store_eval_run_replace_on_conflict(self, temp_db):
        run1 = _make_eval_run("same-run")
        temp_db.store_eval_run(run1)

        run2 = _make_eval_run("same-run", dataset_name="updated-dataset")
        temp_db.store_eval_run(run2)

        restored = temp_db.get_eval_run("same-run")
        assert restored.dataset_name == "updated-dataset"

    def test_call_with_error_field(self, temp_db):
        call = _make_call("err-call")
        call.error = "TimeoutError: request timed out after 30s"
        call.output = None
        temp_db.store_call(call)

        restored = temp_db.get_call("err-call")
        assert restored.error == "TimeoutError: request timed out after 30s"
        assert restored.output is None

    def test_multiple_annotations_same_target(self, temp_db):
        anns = [
            _make_annotation(f"ann-{i}", target_id="shared-target", annotator=f"r{i}")
            for i in range(5)
        ]
        temp_db.store_annotations(anns)
        result = temp_db.list_annotations(target_id="shared-target")
        assert len(result) == 5
