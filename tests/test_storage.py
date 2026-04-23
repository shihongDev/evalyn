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

    def test_function_filter_underscore_is_literal(self, temp_db):
        """Regression: --function 'run_agent' must not match 'runXagent'.

        SQL LIKE treats `_` as a single-char wildcard. The substring filter
        must use literal matching (e.g. instr) so dataset items with similar
        names aren't silently included.
        """
        temp_db.store_call(_make_call("c1", function_name="run_agent"))
        temp_db.store_call(_make_call("c2", function_name="runXagent"))
        temp_db.store_call(_make_call("c3", function_name="run1agent"))
        temp_db.store_call(_make_call("c4", function_name="myfunc"))

        calls = temp_db.list_calls(function_name="run_agent")
        names = {c.function_name for c in calls}
        assert names == {"run_agent"}

    def test_function_filter_percent_is_literal(self, temp_db):
        """Regression: --function 'foo%bar' must not act as a wildcard."""
        temp_db.store_call(_make_call("c1", function_name="foo%bar"))
        temp_db.store_call(_make_call("c2", function_name="fooXXXbar"))

        calls = temp_db.list_calls(function_name="foo%bar")
        names = {c.function_name for c in calls}
        assert names == {"foo%bar"}

    def test_function_filter_case_insensitive(self, temp_db):
        temp_db.store_call(_make_call("c1", function_name="RunAgent"))
        calls = temp_db.list_calls(function_name="runagent")
        assert len(calls) == 1


# ---------------------------------------------------------------------------
# get_calls_batch
# ---------------------------------------------------------------------------

class TestGetCallsBatch:
    def test_batch_returns_dict_keyed_by_id(self, temp_db):
        temp_db.store_call(_make_call("c1"))
        temp_db.store_call(_make_call("c2"))
        result = temp_db.get_calls_batch(["c1", "c2", "missing"])
        assert set(result.keys()) == {"c1", "c2"}

    def test_batch_empty(self, temp_db):
        assert temp_db.get_calls_batch([]) == {}

    def test_batch_chunks_above_sqlite_variable_limit(self, temp_db):
        """Regression: large batches must not raise 'too many SQL variables'.

        SQLITE_MAX_VARIABLE_NUMBER is 999 on SQLite <3.32. Pass well above
        that to ensure chunking kicks in.
        """
        ids = [f"c{i:05d}" for i in range(1500)]
        for cid in ids:
            temp_db.store_call(_make_call(cid))
        # Should not raise sqlite3.OperationalError
        result = temp_db.get_calls_batch(ids)
        assert len(result) == 1500
        assert set(result.keys()) == set(ids)

    def test_batch_small_chunk_size(self, temp_db):
        for i in range(10):
            temp_db.store_call(_make_call(f"c{i}"))
        ids = [f"c{i}" for i in range(10)]
        # chunk_size=3 forces 4 chunks; verifies merge logic
        result = temp_db.get_calls_batch(ids, chunk_size=3)
        assert len(result) == 10


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


# ---------------------------------------------------------------------------
# WAL mode and connection pragmas
# ---------------------------------------------------------------------------

class TestWALMode:
    def test_wal_mode_enabled(self, temp_db):
        cur = temp_db.conn.execute("PRAGMA journal_mode")
        mode = cur.fetchone()[0]
        assert mode == "wal"

    def test_busy_timeout_set(self, temp_db):
        cur = temp_db.conn.execute("PRAGMA busy_timeout")
        timeout = cur.fetchone()[0]
        assert timeout == 5000


# ---------------------------------------------------------------------------
# Thread-local connections
# ---------------------------------------------------------------------------

class TestThreadLocalConnections:
    def test_main_thread_returns_primary_conn(self, temp_db):
        conn = temp_db.get_connection()
        assert conn is temp_db.conn

    def test_worker_thread_gets_separate_conn(self, temp_db):
        import threading
        results = {}

        def check():
            results["conn"] = temp_db.get_connection()
            results["is_same"] = results["conn"] is temp_db.conn

        t = threading.Thread(target=check)
        t.start()
        t.join()

        assert results["is_same"] is False
        assert results["conn"] is not None

    def test_worker_thread_conn_reused_on_same_thread(self, temp_db):
        import threading
        results = {}

        def check():
            c1 = temp_db.get_connection()
            c2 = temp_db.get_connection()
            results["same"] = c1 is c2

        t = threading.Thread(target=check)
        t.start()
        t.join()

        assert results["same"] is True

    def test_worker_thread_conn_has_wal(self, temp_db):
        import threading
        results = {}

        def check():
            conn = temp_db.get_connection()
            cur = conn.execute("PRAGMA journal_mode")
            results["mode"] = cur.fetchone()[0]

        t = threading.Thread(target=check)
        t.start()
        t.join()

        assert results["mode"] == "wal"


# ---------------------------------------------------------------------------
# Relational metric results (metric_results_rows table)
# ---------------------------------------------------------------------------

class TestRelationalMetricResults:
    def test_store_run_writes_to_relational_table(self, temp_db):
        """New runs store metric results in metric_results_rows, not JSON blob."""
        run = _make_eval_run("rel-run-1")
        temp_db.store_eval_run(run)

        # JSON blob should be NULL
        cur = temp_db.conn.cursor()
        cur.execute("SELECT metric_results FROM eval_runs WHERE id = ?", ("rel-run-1",))
        blob = cur.fetchone()[0]
        assert blob is None

        # Relational table should have the results
        cur.execute("SELECT COUNT(*) FROM metric_results_rows WHERE run_id = ?", ("rel-run-1",))
        count = cur.fetchone()[0]
        assert count == 1

    def test_load_metric_results_roundtrip(self, temp_db):
        run = _make_eval_run("rel-run-2")
        temp_db.store_eval_run(run)

        results = temp_db.load_metric_results("rel-run-2")
        assert len(results) == 1
        assert results[0].metric_id == "accuracy"
        assert results[0].score == 0.95
        assert results[0].passed is True
        assert results[0].item_id == "item-1"
        assert results[0].details == {"reason": "exact match"}

    def test_get_eval_run_uses_relational_results(self, temp_db):
        """_row_to_eval_run loads from relational table when JSON blob is NULL."""
        run = _make_eval_run("rel-run-3")
        temp_db.store_eval_run(run)

        restored = temp_db.get_eval_run("rel-run-3")
        assert len(restored.metric_results) == 1
        assert restored.metric_results[0].score == 0.95
        assert restored.metric_results[0].metric_id == "accuracy"

    def test_json_blob_fallback(self, temp_db):
        """Old runs with JSON blob still load correctly."""
        # Manually insert with JSON blob (simulating old data)
        import json
        results_json = json.dumps([{
            "metric_id": "old-metric",
            "item_id": "old-item",
            "call_id": "old-call",
            "score": 0.5,
            "passed": False,
            "details": {},
        }])
        cur = temp_db.conn.cursor()
        cur.execute(
            """INSERT INTO eval_runs
            (id, dataset_name, created_at, metric_results, metrics, judge_configs, summary, usage_summary)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            ("old-run", "old-ds", "2026-01-01T00:00:00+00:00", results_json, "[]", "[]", "{}", "{}"),
        )
        temp_db.conn.commit()

        restored = temp_db.get_eval_run("old-run")
        assert len(restored.metric_results) == 1
        assert restored.metric_results[0].metric_id == "old-metric"
        assert restored.metric_results[0].score == 0.5

    def test_batch_insert_many_results(self, temp_db):
        """batch_insert_metric_results handles large batches."""
        results = [
            MetricResult(
                metric_id=f"m-{i}",
                item_id=f"item-{i}",
                call_id=f"call-{i}",
                score=i / 100.0,
                passed=i % 2 == 0,
                details={"idx": i},
            )
            for i in range(50)
        ]
        temp_db.batch_insert_metric_results("batch-run", results)

        loaded = temp_db.load_metric_results("batch-run")
        assert len(loaded) == 50
        scores = sorted(r.score for r in loaded)
        assert scores[0] == 0.0
        assert scores[-1] == 0.49

    def test_batch_insert_small_batch_size(self, temp_db):
        """Verify batching works with a small batch_size."""
        results = [
            MetricResult(
                metric_id=f"m-{i}",
                item_id=f"item-{i}",
                call_id="c",
                score=float(i),
                passed=True,
            )
            for i in range(7)
        ]
        temp_db.batch_insert_metric_results("small-batch-run", results, batch_size=3)

        loaded = temp_db.load_metric_results("small-batch-run")
        assert len(loaded) == 7

    def test_store_run_no_metric_results(self, temp_db):
        """Run with empty metric_results doesn't crash."""
        run = EvalRun(
            id="empty-mr-run",
            dataset_name="ds",
            created_at=T0,
            metric_results=[],
            metrics=[],
            judge_configs=[],
        )
        temp_db.store_eval_run(run)
        restored = temp_db.get_eval_run("empty-mr-run")
        assert restored.metric_results == []

    def test_passed_none_preserved(self, temp_db):
        """MetricResult with passed=None survives roundtrip."""
        results = [
            MetricResult(
                metric_id="m-none",
                item_id="item-none",
                call_id="c",
                score=0.5,
                passed=None,
                details={},
            )
        ]
        temp_db.batch_insert_metric_results("none-passed-run", results)
        loaded = temp_db.load_metric_results("none-passed-run")
        assert len(loaded) == 1
        assert loaded[0].passed is None

    def test_load_metric_results_empty_run(self, temp_db):
        """Loading results for a run with no rows returns empty list."""
        results = temp_db.load_metric_results("nonexistent-run")
        assert results == []


# ---------------------------------------------------------------------------
# Performance indexes
# ---------------------------------------------------------------------------

class TestPerformanceIndexes:
    def test_indexes_created(self, temp_db):
        cur = temp_db.conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='index'")
        indexes = {row[0] for row in cur.fetchall()}
        expected = {
            "idx_fc_started_at",
            "idx_er_created_at",
            "idx_er_dataset",
            "idx_ann_target",
            "idx_ann_created_at",
            "idx_mr_run",
            "idx_mr_run_item",
            "idx_mr_config",
        }
        assert expected.issubset(indexes)
