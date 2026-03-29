"""Tests for sampling_pipeline module."""

from __future__ import annotations

import functools

import pytest

from evalyn_sdk.sampling_pipeline import (
    PipelineConfig,
    PipelineResult,
    PipelineStage,
    create_stage_fn,
    deduplicate_stage,
    format_pipeline_report,
    limit_stage,
    run_pipeline,
    shuffle_stage,
)


# ---------------------------------------------------------------------------
# PipelineStage
# ---------------------------------------------------------------------------


class TestPipelineStage:
    def test_defaults(self):
        s = PipelineStage(stage_name="dedup", strategy="deduplicate")
        assert s.input_count == 0
        assert s.output_count == 0
        assert s.duration_seconds == 0.0

    def test_full_init(self):
        s = PipelineStage(
            stage_name="limit",
            strategy="truncate",
            input_count=100,
            output_count=50,
            duration_seconds=0.01,
        )
        assert s.stage_name == "limit"
        assert s.output_count == 50

    def test_as_dict(self):
        s = PipelineStage(stage_name="s", strategy="x", input_count=5, output_count=3)
        d = s.as_dict()
        assert d["stage_name"] == "s"
        assert d["strategy"] == "x"
        assert d["input_count"] == 5
        assert d["output_count"] == 3
        assert d["duration_seconds"] == 0.0

    def test_from_dict_full(self):
        d = {
            "stage_name": "a",
            "strategy": "b",
            "input_count": 10,
            "output_count": 5,
            "duration_seconds": 0.1,
        }
        s = PipelineStage.from_dict(d)
        assert s.stage_name == "a"
        assert s.duration_seconds == 0.1

    def test_from_dict_minimal(self):
        s = PipelineStage.from_dict({"stage_name": "x"})
        assert s.stage_name == "x"
        assert s.strategy == ""
        assert s.input_count == 0

    def test_roundtrip(self):
        s = PipelineStage(stage_name="s", strategy="t", input_count=7, output_count=3, duration_seconds=0.5)
        assert PipelineStage.from_dict(s.as_dict()).as_dict() == s.as_dict()


# ---------------------------------------------------------------------------
# PipelineConfig
# ---------------------------------------------------------------------------


class TestPipelineConfig:
    def test_defaults(self):
        c = PipelineConfig()
        assert c.stages == []

    def test_with_stages(self):
        c = PipelineConfig(stages=["dedup", "shuffle", "limit"])
        assert len(c.stages) == 3

    def test_as_dict(self):
        c = PipelineConfig(stages=["a", "b"])
        assert c.as_dict() == {"stages": ["a", "b"]}

    def test_from_dict(self):
        c = PipelineConfig.from_dict({"stages": ["x", "y"]})
        assert c.stages == ["x", "y"]

    def test_from_dict_empty(self):
        c = PipelineConfig.from_dict({})
        assert c.stages == []

    def test_roundtrip(self):
        c = PipelineConfig(stages=["a", "b", "c"])
        assert PipelineConfig.from_dict(c.as_dict()).as_dict() == c.as_dict()

    def test_list_is_copied(self):
        original = ["a", "b"]
        c = PipelineConfig.from_dict({"stages": original})
        original.append("c")
        assert len(c.stages) == 2


# ---------------------------------------------------------------------------
# PipelineResult
# ---------------------------------------------------------------------------


class TestPipelineResult:
    def test_defaults(self):
        r = PipelineResult()
        assert r.final_ids == []
        assert r.stage_results == []
        assert r.total_input == 0
        assert r.total_output == 0

    def test_as_dict(self):
        stage = PipelineStage(stage_name="s", strategy="x", input_count=5, output_count=3)
        r = PipelineResult(
            final_ids=["a", "b"],
            stage_results=[stage],
            total_input=5,
            total_output=2,
        )
        d = r.as_dict()
        assert d["final_ids"] == ["a", "b"]
        assert len(d["stage_results"]) == 1
        assert d["total_input"] == 5

    def test_from_dict(self):
        d = {
            "final_ids": ["x"],
            "stage_results": [{"stage_name": "a", "strategy": "b", "input_count": 3, "output_count": 1}],
            "total_input": 3,
            "total_output": 1,
        }
        r = PipelineResult.from_dict(d)
        assert r.final_ids == ["x"]
        assert r.stage_results[0].stage_name == "a"

    def test_roundtrip(self):
        stage = PipelineStage(stage_name="s", strategy="x", input_count=5, output_count=3, duration_seconds=0.01)
        r = PipelineResult(final_ids=["a"], stage_results=[stage], total_input=5, total_output=1)
        assert PipelineResult.from_dict(r.as_dict()).as_dict() == r.as_dict()


# ---------------------------------------------------------------------------
# deduplicate_stage
# ---------------------------------------------------------------------------


class TestDeduplicateStage:
    def test_no_duplicates(self):
        assert deduplicate_stage(["a", "b", "c"]) == ["a", "b", "c"]

    def test_with_duplicates(self):
        assert deduplicate_stage(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]

    def test_empty(self):
        assert deduplicate_stage([]) == []

    def test_all_same(self):
        assert deduplicate_stage(["x", "x", "x"]) == ["x"]

    def test_preserves_order(self):
        result = deduplicate_stage(["c", "a", "b", "a", "c"])
        assert result == ["c", "a", "b"]

    def test_single_item(self):
        assert deduplicate_stage(["only"]) == ["only"]


# ---------------------------------------------------------------------------
# shuffle_stage
# ---------------------------------------------------------------------------


class TestShuffleStage:
    def test_deterministic(self):
        ids = ["a", "b", "c", "d", "e"]
        assert shuffle_stage(ids, seed=42) == shuffle_stage(ids, seed=42)

    def test_different_seeds(self):
        ids = list("abcdefghij")
        r1 = shuffle_stage(ids, seed=1)
        r2 = shuffle_stage(ids, seed=2)
        # Extremely unlikely to be identical with different seeds and 10 items
        assert r1 != r2 or len(ids) <= 1

    def test_same_elements(self):
        ids = ["a", "b", "c"]
        result = shuffle_stage(ids, seed=7)
        assert sorted(result) == sorted(ids)

    def test_empty(self):
        assert shuffle_stage([], seed=1) == []

    def test_single(self):
        assert shuffle_stage(["x"], seed=1) == ["x"]

    def test_does_not_mutate_input(self):
        ids = ["a", "b", "c"]
        original = list(ids)
        shuffle_stage(ids, seed=42)
        assert ids == original

    def test_default_seed(self):
        ids = ["a", "b", "c", "d"]
        r1 = shuffle_stage(ids)
        r2 = shuffle_stage(ids)
        assert r1 == r2  # default seed=42 both times


# ---------------------------------------------------------------------------
# limit_stage
# ---------------------------------------------------------------------------


class TestLimitStage:
    def test_below_limit(self):
        assert limit_stage(["a", "b"], max_count=5) == ["a", "b"]

    def test_at_limit(self):
        assert limit_stage(["a", "b", "c"], max_count=3) == ["a", "b", "c"]

    def test_above_limit(self):
        assert limit_stage(["a", "b", "c", "d"], max_count=2) == ["a", "b"]

    def test_zero_limit(self):
        assert limit_stage(["a", "b"], max_count=0) == []

    def test_empty(self):
        assert limit_stage([], max_count=5) == []

    def test_default_limit(self):
        ids = [str(i) for i in range(200)]
        assert len(limit_stage(ids)) == 100


# ---------------------------------------------------------------------------
# create_stage_fn
# ---------------------------------------------------------------------------


class TestCreateStageFn:
    def test_returns_tuple(self):
        result = create_stage_fn("my_stage", lambda ids: ids)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] == "my_stage"

    def test_fn_callable(self):
        name, fn = create_stage_fn("echo", lambda ids: ids)
        assert fn(["a", "b"]) == ["a", "b"]

    def test_with_real_function(self):
        name, fn = create_stage_fn("dedup", deduplicate_stage)
        assert fn(["a", "a", "b"]) == ["a", "b"]


# ---------------------------------------------------------------------------
# run_pipeline
# ---------------------------------------------------------------------------


class TestRunPipeline:
    def test_single_stage(self):
        result = run_pipeline(["a", "b", "c"], [("dedup", deduplicate_stage)])
        assert result.final_ids == ["a", "b", "c"]
        assert result.total_input == 3
        assert result.total_output == 3
        assert len(result.stage_results) == 1

    def test_multi_stage(self):
        ids = ["a", "b", "a", "c", "b", "d", "e"]
        stages = [
            ("dedup", deduplicate_stage),
            ("limit", lambda x: limit_stage(x, max_count=3)),
        ]
        result = run_pipeline(ids, stages)
        assert result.total_input == 7
        assert result.final_ids == ["a", "b", "c"]
        assert result.total_output == 3
        assert len(result.stage_results) == 2

    def test_empty_input(self):
        result = run_pipeline([], [("dedup", deduplicate_stage)])
        assert result.final_ids == []
        assert result.total_input == 0
        assert result.total_output == 0

    def test_no_stages(self):
        result = run_pipeline(["a", "b"], [])
        assert result.final_ids == ["a", "b"]
        assert result.total_input == 2
        assert result.total_output == 2
        assert result.stage_results == []

    def test_stage_results_have_counts(self):
        ids = ["a", "b", "a", "c"]
        result = run_pipeline(ids, [("dedup", deduplicate_stage)])
        s = result.stage_results[0]
        assert s.input_count == 4
        assert s.output_count == 3
        assert s.stage_name == "dedup"

    def test_stage_timing(self):
        result = run_pipeline(["a"], [("echo", lambda x: x)])
        assert result.stage_results[0].duration_seconds >= 0.0

    def test_chained_stages_funnel(self):
        ids = ["a", "b", "c", "a", "b", "d", "e", "f", "g", "h"]
        stages = [
            ("dedup", deduplicate_stage),
            ("shuffle", lambda x: shuffle_stage(x, seed=1)),
            ("limit", lambda x: limit_stage(x, max_count=3)),
        ]
        result = run_pipeline(ids, stages)
        assert result.total_input == 10
        assert result.total_output == 3
        assert len(result.stage_results) == 3
        # After dedup: 8 items
        assert result.stage_results[0].output_count == 8
        # After shuffle: still 8
        assert result.stage_results[1].output_count == 8
        # After limit: 3
        assert result.stage_results[2].output_count == 3

    def test_input_not_mutated(self):
        ids = ["a", "b", "c"]
        original = list(ids)
        run_pipeline(ids, [("dedup", deduplicate_stage)])
        assert ids == original

    def test_stage_strategy_matches_name(self):
        result = run_pipeline(["a"], [("my_stage", lambda x: x)])
        assert result.stage_results[0].strategy == "my_stage"

    def test_create_stage_fn_integration(self):
        stage = create_stage_fn("dedup", deduplicate_stage)
        result = run_pipeline(["a", "a", "b"], [stage])
        assert result.final_ids == ["a", "b"]

    def test_filter_stage(self):
        """Custom filter stage that keeps only items starting with 'a'."""
        def keep_a(ids):
            return [x for x in ids if x.startswith("a")]

        result = run_pipeline(
            ["apple", "banana", "avocado", "cherry"],
            [("filter_a", keep_a)],
        )
        assert result.final_ids == ["apple", "avocado"]
        assert result.total_output == 2

    def test_all_dropped(self):
        result = run_pipeline(["a", "b"], [("drop_all", lambda x: [])])
        assert result.final_ids == []
        assert result.total_output == 0
        assert result.stage_results[0].input_count == 2
        assert result.stage_results[0].output_count == 0


# ---------------------------------------------------------------------------
# format_pipeline_report
# ---------------------------------------------------------------------------


class TestFormatPipelineReport:
    def test_basic(self):
        result = run_pipeline(
            ["a", "b", "a", "c"],
            [("dedup", deduplicate_stage)],
        )
        report = format_pipeline_report(result)
        assert "Pipeline Report" in report
        assert "dedup" in report
        assert "4" in report  # total input
        assert "3" in report  # total output

    def test_multi_stage(self):
        result = run_pipeline(
            ["a", "b", "a"],
            [
                ("dedup", deduplicate_stage),
                ("limit", lambda x: limit_stage(x, max_count=1)),
            ],
        )
        report = format_pipeline_report(result)
        assert "dedup" in report
        assert "limit" in report

    def test_empty_pipeline(self):
        result = run_pipeline(["a"], [])
        report = format_pipeline_report(result)
        assert "Pipeline Report" in report

    def test_shows_dropped(self):
        result = run_pipeline(
            ["a", "b", "c"],
            [("limit", lambda x: limit_stage(x, max_count=1))],
        )
        report = format_pipeline_report(result)
        assert "dropped 2" in report

    def test_report_is_string(self):
        result = run_pipeline([], [])
        assert isinstance(format_pipeline_report(result), str)
