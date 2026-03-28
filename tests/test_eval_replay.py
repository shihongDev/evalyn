"""Tests for evaluation replay module."""

import pytest
from evalyn_sdk.evaluation.replay import (
    ReplayConfig,
    ReplayPlan,
    ReplayComparison,
    ReplayReport,
    build_replay_plan,
    compare_replay_results,
    summarize_replay,
)


class TestReplayConfig:
    def test_as_dict_minimal(self):
        c = ReplayConfig(original_run_id="run-1")
        d = c.as_dict()
        assert d["original_run_id"] == "run-1"
        assert d["override_provider"] is None
        assert d["override_metrics"] is None

    def test_as_dict_full(self):
        c = ReplayConfig(
            original_run_id="run-2",
            override_provider="openai",
            override_model="gpt-4",
            override_metrics=["accuracy", "helpfulness"],
            override_prompts={"accuracy": "Rate accuracy 1-5"},
        )
        d = c.as_dict()
        assert d["override_provider"] == "openai"
        assert d["override_model"] == "gpt-4"
        assert d["override_metrics"] == ["accuracy", "helpfulness"]
        assert d["override_prompts"] == {"accuracy": "Rate accuracy 1-5"}

    def test_from_dict_roundtrip(self):
        c = ReplayConfig(
            original_run_id="run-3",
            override_provider="gemini",
            override_metrics=["safety"],
        )
        restored = ReplayConfig.from_dict(c.as_dict())
        assert restored.original_run_id == "run-3"
        assert restored.override_provider == "gemini"
        assert restored.override_metrics == ["safety"]

    def test_from_dict_minimal(self):
        c = ReplayConfig.from_dict({"original_run_id": "run-x"})
        assert c.original_run_id == "run-x"
        assert c.override_model is None


class TestReplayPlan:
    def test_as_dict(self):
        config = ReplayConfig(original_run_id="run-1")
        plan = ReplayPlan(
            config=config,
            items_to_replay=["item-1", "item-2"],
            metrics_to_use=["accuracy"],
            estimated_calls=2,
        )
        d = plan.as_dict()
        assert d["items_to_replay"] == ["item-1", "item-2"]
        assert d["estimated_calls"] == 2

    def test_format_text_contains_key_info(self):
        config = ReplayConfig(
            original_run_id="run-1",
            override_provider="openai",
            override_model="gpt-4",
            override_prompts={"m1": "prompt"},
        )
        plan = ReplayPlan(
            config=config,
            items_to_replay=["a", "b"],
            metrics_to_use=["m1"],
            estimated_calls=2,
        )
        text = plan.format_text()
        assert "run-1" in text
        assert "2" in text
        assert "openai" in text
        assert "gpt-4" in text
        assert "1 metric(s)" in text


class TestReplayComparison:
    def test_as_dict(self):
        c = ReplayComparison(
            item_id="i1",
            metric_id="m1",
            original_score=0.5,
            replayed_score=0.8,
            delta=0.3,
        )
        d = c.as_dict()
        assert d["item_id"] == "i1"
        assert d["delta"] == 0.3


class TestReplayReport:
    def test_as_dict(self):
        report = ReplayReport(
            comparisons=[
                ReplayComparison("i1", "m1", 0.5, 0.8, 0.3),
            ],
            improved_count=1,
            degraded_count=0,
            unchanged_count=0,
            mean_delta=0.3,
        )
        d = report.as_dict()
        assert len(d["comparisons"]) == 1
        assert d["improved_count"] == 1
        assert d["mean_delta"] == 0.3

    def test_from_dict_roundtrip(self):
        report = ReplayReport(
            comparisons=[
                ReplayComparison("i1", "m1", 0.5, 0.8, 0.3),
                ReplayComparison("i2", "m1", 0.9, 0.7, -0.2),
            ],
            improved_count=1,
            degraded_count=1,
            unchanged_count=0,
            mean_delta=0.05,
        )
        restored = ReplayReport.from_dict(report.as_dict())
        assert len(restored.comparisons) == 2
        assert restored.improved_count == 1
        assert restored.degraded_count == 1
        assert restored.mean_delta == pytest.approx(0.05)

    def test_format_text(self):
        report = ReplayReport(
            comparisons=[
                ReplayComparison("i1", "m1", 0.5, 0.8, 0.3),
            ],
            improved_count=1,
            degraded_count=0,
            unchanged_count=0,
            mean_delta=0.3,
        )
        text = report.format_text()
        assert "Replay Report" in text
        assert "Improved" in text
        assert "1" in text

    def test_empty_report(self):
        report = ReplayReport()
        assert report.comparisons == []
        assert report.improved_count == 0
        assert report.mean_delta == 0.0


class TestBuildReplayPlan:
    def test_uses_original_metrics(self):
        config = ReplayConfig(original_run_id="run-1")
        plan = build_replay_plan(config, ["i1", "i2"], ["accuracy", "safety"])
        assert plan.metrics_to_use == ["accuracy", "safety"]
        assert plan.items_to_replay == ["i1", "i2"]

    def test_with_override_metrics(self):
        config = ReplayConfig(
            original_run_id="run-1",
            override_metrics=["helpfulness"],
        )
        plan = build_replay_plan(config, ["i1", "i2"], ["accuracy", "safety"])
        assert plan.metrics_to_use == ["helpfulness"]

    def test_estimated_calls(self):
        config = ReplayConfig(original_run_id="run-1")
        plan = build_replay_plan(config, ["i1", "i2", "i3"], ["m1", "m2"])
        assert plan.estimated_calls == 6  # 3 items * 2 metrics

    def test_estimated_calls_with_override(self):
        config = ReplayConfig(
            original_run_id="run-1",
            override_metrics=["m1"],
        )
        plan = build_replay_plan(config, ["i1", "i2"], ["m1", "m2", "m3"])
        assert plan.estimated_calls == 2  # 2 items * 1 override metric

    def test_empty_items(self):
        config = ReplayConfig(original_run_id="run-1")
        plan = build_replay_plan(config, [], ["m1"])
        assert plan.estimated_calls == 0
        assert plan.items_to_replay == []

    def test_empty_metrics(self):
        config = ReplayConfig(original_run_id="run-1")
        plan = build_replay_plan(config, ["i1"], [])
        assert plan.estimated_calls == 0
        assert plan.metrics_to_use == []


class TestCompareReplayResults:
    def test_all_improved(self):
        original = [
            {"item_id": "i1", "metric_id": "m1", "score": 0.5},
            {"item_id": "i2", "metric_id": "m1", "score": 0.6},
        ]
        replayed = [
            {"item_id": "i1", "metric_id": "m1", "score": 0.9},
            {"item_id": "i2", "metric_id": "m1", "score": 0.8},
        ]
        report = compare_replay_results(original, replayed)
        assert report.improved_count == 2
        assert report.degraded_count == 0
        assert report.unchanged_count == 0

    def test_all_degraded(self):
        original = [
            {"item_id": "i1", "metric_id": "m1", "score": 0.9},
            {"item_id": "i2", "metric_id": "m1", "score": 0.8},
        ]
        replayed = [
            {"item_id": "i1", "metric_id": "m1", "score": 0.3},
            {"item_id": "i2", "metric_id": "m1", "score": 0.2},
        ]
        report = compare_replay_results(original, replayed)
        assert report.improved_count == 0
        assert report.degraded_count == 2

    def test_mixed_results(self):
        original = [
            {"item_id": "i1", "metric_id": "m1", "score": 0.5},
            {"item_id": "i2", "metric_id": "m1", "score": 0.9},
            {"item_id": "i3", "metric_id": "m1", "score": 0.7},
        ]
        replayed = [
            {"item_id": "i1", "metric_id": "m1", "score": 0.8},  # improved
            {"item_id": "i2", "metric_id": "m1", "score": 0.5},  # degraded
            {"item_id": "i3", "metric_id": "m1", "score": 0.7},  # unchanged
        ]
        report = compare_replay_results(original, replayed)
        assert report.improved_count == 1
        assert report.degraded_count == 1
        assert report.unchanged_count == 1

    def test_unchanged_within_threshold(self):
        original = [
            {"item_id": "i1", "metric_id": "m1", "score": 0.5},
        ]
        replayed = [
            {"item_id": "i1", "metric_id": "m1", "score": 0.505},
        ]
        report = compare_replay_results(original, replayed, threshold=0.01)
        assert report.unchanged_count == 1
        assert report.improved_count == 0

    def test_mean_delta_correct(self):
        original = [
            {"item_id": "i1", "metric_id": "m1", "score": 0.5},
            {"item_id": "i2", "metric_id": "m1", "score": 0.6},
        ]
        replayed = [
            {"item_id": "i1", "metric_id": "m1", "score": 0.7},  # +0.2
            {"item_id": "i2", "metric_id": "m1", "score": 0.5},  # -0.1
        ]
        report = compare_replay_results(original, replayed)
        assert report.mean_delta == pytest.approx(0.05)

    def test_empty_results(self):
        report = compare_replay_results([], [])
        assert len(report.comparisons) == 0
        assert report.mean_delta == 0.0
        assert report.improved_count == 0

    def test_custom_threshold(self):
        original = [
            {"item_id": "i1", "metric_id": "m1", "score": 0.5},
        ]
        replayed = [
            {"item_id": "i1", "metric_id": "m1", "score": 0.55},
        ]
        # With large threshold, this should be unchanged
        report = compare_replay_results(original, replayed, threshold=0.1)
        assert report.unchanged_count == 1
        # With small threshold, this should be improved
        report2 = compare_replay_results(original, replayed, threshold=0.01)
        assert report2.improved_count == 1

    def test_missing_replayed_result_skipped(self):
        original = [
            {"item_id": "i1", "metric_id": "m1", "score": 0.5},
            {"item_id": "i2", "metric_id": "m1", "score": 0.6},
        ]
        replayed = [
            {"item_id": "i1", "metric_id": "m1", "score": 0.9},
        ]
        report = compare_replay_results(original, replayed)
        assert len(report.comparisons) == 1


class TestSummarizeReplay:
    def test_summary_keys(self):
        report = ReplayReport(
            comparisons=[
                ReplayComparison("i1", "m1", 0.5, 0.8, 0.3),
                ReplayComparison("i2", "m1", 0.9, 0.7, -0.2),
            ],
            improved_count=1,
            degraded_count=1,
            unchanged_count=0,
            mean_delta=0.05,
        )
        s = summarize_replay(report)
        assert s["total_comparisons"] == 2
        assert s["improved_count"] == 1
        assert s["degraded_count"] == 1
        assert s["mean_delta"] == pytest.approx(0.05)
        assert s["improved_pct"] == pytest.approx(50.0)
        assert s["degraded_pct"] == pytest.approx(50.0)

    def test_summary_empty(self):
        report = ReplayReport()
        s = summarize_replay(report)
        assert s["total_comparisons"] == 0
        assert s["improved_pct"] == 0.0
        assert s["degraded_pct"] == 0.0
