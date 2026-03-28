from __future__ import annotations

import pytest

from evalyn_sdk.evaluation.pipeline_definitions import (
    FULL_PIPELINE,
    QUICK_EVAL_PIPELINE,
    PipelineDefinition,
    PipelineRunResult,
    PipelineStep,
    compare_pipeline_runs,
    create_pipeline,
    get_execution_order,
    get_pipeline,
    list_pipelines,
)


# ---------------------------------------------------------------------------
# PipelineStep
# ---------------------------------------------------------------------------


class TestPipelineStep:
    def test_as_dict(self):
        s = PipelineStep("load", "trace", {"key": "val"}, ["dep1"])
        d = s.as_dict()
        assert d["name"] == "load"
        assert d["step_type"] == "trace"
        assert d["config"] == {"key": "val"}
        assert d["depends_on"] == ["dep1"]

    def test_from_dict_roundtrip(self):
        s = PipelineStep("run", "evaluate", {"p": 1}, ["load"])
        restored = PipelineStep.from_dict(s.as_dict())
        assert restored == s

    def test_from_dict_defaults(self):
        s = PipelineStep.from_dict({"name": "x", "step_type": "trace"})
        assert s.config == {}
        assert s.depends_on == []


# ---------------------------------------------------------------------------
# PipelineDefinition
# ---------------------------------------------------------------------------


class TestPipelineDefinition:
    def test_step_names(self):
        p = PipelineDefinition(
            name="test",
            steps=[PipelineStep("a", "trace"), PipelineStep("b", "evaluate")],
        )
        assert p.step_names == ["a", "b"]

    def test_step_names_empty(self):
        p = PipelineDefinition(name="empty")
        assert p.step_names == []

    def test_as_dict_from_dict_roundtrip(self):
        p = PipelineDefinition(
            name="roundtrip",
            description="test desc",
            steps=[PipelineStep("s1", "trace"), PipelineStep("s2", "analyze")],
            version=2,
        )
        d = p.as_dict()
        restored = PipelineDefinition.from_dict(d)
        assert restored.name == p.name
        assert restored.description == p.description
        assert len(restored.steps) == 2
        assert restored.version == 2

    def test_format_text(self):
        p = PipelineDefinition(
            name="demo",
            description="A demo pipeline",
            steps=[
                PipelineStep("a", "trace"),
                PipelineStep("b", "evaluate", depends_on=["a"]),
            ],
        )
        text = p.format_text()
        assert "Pipeline: demo" in text
        assert "A demo pipeline" in text
        assert "a [trace]" in text
        assert "(after: a)" in text

    def test_format_text_no_description(self):
        p = PipelineDefinition(name="bare", steps=[PipelineStep("x", "trace")])
        text = p.format_text()
        assert "bare" in text

    def test_validate_valid(self):
        p = PipelineDefinition(
            name="ok",
            steps=[
                PipelineStep("a", "trace"),
                PipelineStep("b", "evaluate", depends_on=["a"]),
            ],
        )
        valid, errors = p.validate()
        assert valid is True
        assert errors == []

    def test_validate_missing_dep(self):
        p = PipelineDefinition(
            name="bad",
            steps=[PipelineStep("a", "trace", depends_on=["nonexistent"])],
        )
        valid, errors = p.validate()
        assert valid is False
        assert any("nonexistent" in e for e in errors)

    def test_validate_circular_dep(self):
        p = PipelineDefinition(
            name="cycle",
            steps=[
                PipelineStep("a", "trace", depends_on=["b"]),
                PipelineStep("b", "evaluate", depends_on=["a"]),
            ],
        )
        valid, errors = p.validate()
        assert valid is False
        assert any("Circular" in e for e in errors)

    def test_validate_empty_steps(self):
        p = PipelineDefinition(name="empty")
        valid, errors = p.validate()
        assert valid is True

    def test_validate_self_dep(self):
        p = PipelineDefinition(
            name="self",
            steps=[PipelineStep("a", "trace", depends_on=["a"])],
        )
        valid, errors = p.validate()
        assert valid is False


# ---------------------------------------------------------------------------
# PipelineRunResult
# ---------------------------------------------------------------------------


class TestPipelineRunResult:
    def test_as_dict_from_dict_roundtrip(self):
        r = PipelineRunResult(
            pipeline_name="test",
            step_results={"s1": {"status": "ok"}},
            completed_steps=["s1"],
            failed_step="",
            total_duration_ms=123.4,
        )
        d = r.as_dict()
        restored = PipelineRunResult.from_dict(d)
        assert restored.pipeline_name == "test"
        assert restored.completed_steps == ["s1"]
        assert restored.total_duration_ms == 123.4

    def test_format_text(self):
        r = PipelineRunResult(
            pipeline_name="demo",
            completed_steps=["a", "b"],
            total_duration_ms=500.0,
        )
        text = r.format_text()
        assert "Pipeline Run: demo" in text
        assert "500.0 ms" in text
        assert "- a" in text
        assert "- b" in text

    def test_format_text_with_failure(self):
        r = PipelineRunResult(
            pipeline_name="fail",
            completed_steps=["a"],
            failed_step="b",
            total_duration_ms=100.0,
        )
        text = r.format_text()
        assert "Failed at: b" in text

    def test_defaults(self):
        r = PipelineRunResult(pipeline_name="p")
        assert r.step_results == {}
        assert r.completed_steps == []
        assert r.failed_step == ""
        assert r.total_duration_ms == 0.0


# ---------------------------------------------------------------------------
# Built-in pipelines
# ---------------------------------------------------------------------------


class TestBuiltinPipelines:
    def test_get_pipeline_quick(self):
        p = get_pipeline("quick-eval")
        assert p.name == "quick-eval"
        assert len(p.steps) == 3

    def test_get_pipeline_full(self):
        p = get_pipeline("full")
        assert p.name == "full"
        assert len(p.steps) == 5

    def test_get_pipeline_unknown(self):
        with pytest.raises(KeyError):
            get_pipeline("nonexistent")

    def test_list_pipelines_count(self):
        pipelines = list_pipelines()
        assert len(pipelines) == 2

    def test_list_pipelines_names(self):
        names = [p.name for p in list_pipelines()]
        assert "quick-eval" in names
        assert "full" in names

    def test_quick_pipeline_validates(self):
        valid, errors = QUICK_EVAL_PIPELINE.validate()
        assert valid is True

    def test_full_pipeline_validates(self):
        valid, errors = FULL_PIPELINE.validate()
        assert valid is True


# ---------------------------------------------------------------------------
# create_pipeline
# ---------------------------------------------------------------------------


class TestCreatePipeline:
    def test_from_dicts(self):
        steps = [
            {"name": "s1", "step_type": "trace"},
            {"name": "s2", "step_type": "evaluate", "depends_on": ["s1"]},
        ]
        p = create_pipeline("custom", steps)
        assert p.name == "custom"
        assert len(p.steps) == 2
        assert p.steps[1].depends_on == ["s1"]

    def test_empty_steps(self):
        p = create_pipeline("empty", [])
        assert p.step_names == []


# ---------------------------------------------------------------------------
# get_execution_order
# ---------------------------------------------------------------------------


class TestGetExecutionOrder:
    def test_linear(self):
        p = PipelineDefinition(
            name="linear",
            steps=[
                PipelineStep("a", "trace"),
                PipelineStep("b", "evaluate"),
                PipelineStep("c", "analyze"),
            ],
        )
        order = get_execution_order(p)
        assert order == ["a", "b", "c"]

    def test_with_deps(self):
        p = PipelineDefinition(
            name="deps",
            steps=[
                PipelineStep("a", "trace"),
                PipelineStep("b", "evaluate", depends_on=["a"]),
                PipelineStep("c", "analyze", depends_on=["b"]),
            ],
        )
        order = get_execution_order(p)
        assert order.index("a") < order.index("b")
        assert order.index("b") < order.index("c")

    def test_diamond_deps(self):
        p = PipelineDefinition(
            name="diamond",
            steps=[
                PipelineStep("a", "trace"),
                PipelineStep("b", "evaluate", depends_on=["a"]),
                PipelineStep("c", "analyze", depends_on=["a"]),
                PipelineStep("d", "export", depends_on=["b", "c"]),
            ],
        )
        order = get_execution_order(p)
        assert order[0] == "a"
        assert order[-1] == "d"
        assert set(order) == {"a", "b", "c", "d"}

    def test_empty_pipeline(self):
        p = PipelineDefinition(name="empty")
        order = get_execution_order(p)
        assert order == []

    def test_single_step(self):
        p = PipelineDefinition(name="one", steps=[PipelineStep("only", "trace")])
        order = get_execution_order(p)
        assert order == ["only"]


# ---------------------------------------------------------------------------
# compare_pipeline_runs
# ---------------------------------------------------------------------------


class TestComparePipelineRuns:
    def test_compare(self):
        a = PipelineRunResult(
            pipeline_name="test",
            completed_steps=["s1", "s2"],
            total_duration_ms=100.0,
        )
        b = PipelineRunResult(
            pipeline_name="test",
            completed_steps=["s2", "s3"],
            total_duration_ms=200.0,
        )
        result = compare_pipeline_runs(a, b)
        assert result["completed_a"] == 2
        assert result["completed_b"] == 2
        assert result["common_steps"] == ["s2"]
        assert result["only_in_a"] == ["s1"]
        assert result["only_in_b"] == ["s3"]
        assert result["duration_diff_ms"] == 100.0

    def test_compare_identical(self):
        r = PipelineRunResult(
            pipeline_name="same",
            completed_steps=["a", "b"],
            total_duration_ms=50.0,
        )
        result = compare_pipeline_runs(r, r)
        assert result["only_in_a"] == []
        assert result["only_in_b"] == []
        assert result["duration_diff_ms"] == 0.0

    def test_compare_with_failures(self):
        a = PipelineRunResult(pipeline_name="p", completed_steps=["s1"], failed_step="s2")
        b = PipelineRunResult(pipeline_name="p", completed_steps=["s1", "s2"])
        result = compare_pipeline_runs(a, b)
        assert result["a_failed"] == "s2"
        assert result["b_failed"] == ""
