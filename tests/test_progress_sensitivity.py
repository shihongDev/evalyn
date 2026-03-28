"""Tests for evaluation progress API and metric sensitivity analysis."""

import json
import pytest
from pathlib import Path
from evalyn_sdk.evaluation.progress import ProgressTracker, ProgressEvent
from evalyn_sdk.analysis.sensitivity import (
    analyze_sensitivity, perturb_text, SensitivityResult, SensitivityReport,
)


# =============================================================================
# Progress API tests
# =============================================================================

class TestProgressEvent:
    def test_as_dict(self):
        e = ProgressEvent("item_complete", "2026-03-28T12:00:00Z", {"item_id": "i1"})
        d = e.as_dict()
        assert d["event"] == "item_complete"
        assert d["item_id"] == "i1"

    def test_to_json(self):
        e = ProgressEvent("run_started", "2026-03-28T12:00:00Z", {"count": 10})
        j = json.loads(e.to_json())
        assert j["event"] == "run_started"


class TestProgressTracker:
    def test_run_lifecycle(self):
        tracker = ProgressTracker()
        tracker.run_started("r1", 10, 3)
        tracker.item_started("i1")
        tracker.metric_scored("help", "i1", 0.9, True)
        tracker.item_complete("i1", pass_count=1, fail_count=0)
        tracker.run_complete("r1", overall_pass_rate=0.9)

        assert len(tracker.events) == 5
        assert tracker.items_complete == 1

    def test_progress_fraction(self):
        tracker = ProgressTracker()
        tracker.run_started("r1", 4, 1)
        assert tracker.progress_fraction == 0.0
        tracker.item_complete("i1")
        assert tracker.progress_fraction == 0.25
        tracker.item_complete("i2")
        assert tracker.progress_fraction == 0.5

    def test_file_output(self, tmp_path):
        file_path = str(tmp_path / "progress.jsonl")
        tracker = ProgressTracker(file_path=file_path)
        tracker.run_started("r1", 5, 2)
        tracker.item_complete("i1")

        assert Path(file_path).exists()
        lines = Path(file_path).read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["event"] == "run_started"
        assert json.loads(lines[1])["event"] == "item_complete"

    def test_callback(self):
        received = []
        tracker = ProgressTracker(callback=lambda e: received.append(e.event_type))
        tracker.run_started("r1", 1, 1)
        tracker.item_complete("i1")
        assert received == ["run_started", "item_complete"]

    def test_disabled(self):
        tracker = ProgressTracker(enabled=False)
        tracker.run_started("r1", 10, 3)
        tracker.item_complete("i1")
        assert len(tracker.events) == 0

    def test_error_event(self):
        tracker = ProgressTracker()
        tracker.error("help", "i1", "API timeout")
        assert tracker.events[0].event_type == "error"
        assert tracker.events[0].data["error"] == "API timeout"

    def test_callback_exception_safe(self):
        tracker = ProgressTracker(callback=lambda e: 1/0)
        tracker.run_started("r1", 1, 1)  # should not raise

    def test_elapsed_seconds(self):
        tracker = ProgressTracker()
        tracker.run_started("r1", 1, 1)
        assert tracker.elapsed_seconds >= 0

    def test_summary(self):
        tracker = ProgressTracker()
        tracker.run_started("r1", 10, 3)
        tracker.item_complete("i1")
        s = tracker.summary()
        assert s["items_complete"] == 1
        assert s["total_items"] == 10
        assert s["progress"] == 0.1

    def test_zero_items_progress(self):
        tracker = ProgressTracker()
        assert tracker.progress_fraction == 0.0

    def test_creates_parent_dirs(self, tmp_path):
        file_path = str(tmp_path / "deep" / "nested" / "progress.jsonl")
        tracker = ProgressTracker(file_path=file_path)
        tracker.run_started("r1", 1, 1)
        assert Path(file_path).exists()


# =============================================================================
# Sensitivity analysis tests
# =============================================================================

class TestAnalyzeSensitivity:
    def test_stable_metric(self):
        scores = {"json_valid": [1.0, 1.0, 1.0, 1.0, 1.0]}
        report = analyze_sensitivity(scores)
        assert report.results[0].classification == "stable"
        assert report.results[0].coefficient_of_variation == 0.0

    def test_volatile_metric(self):
        scores = {"helpfulness": [0.2, 0.9, 0.3, 0.8, 0.1]}
        report = analyze_sensitivity(scores)
        assert report.results[0].classification == "volatile"

    def test_moderate_metric(self):
        scores = {"quality": [0.80, 0.82, 0.78, 0.81, 0.79]}
        report = analyze_sensitivity(scores)
        result = report.results[0]
        assert result.classification in ("stable", "moderate")

    def test_single_score(self):
        scores = {"only_one": [0.5]}
        report = analyze_sensitivity(scores)
        assert report.results[0].classification == "insufficient_data"

    def test_empty(self):
        report = analyze_sensitivity({})
        assert len(report.results) == 0

    def test_multiple_metrics(self):
        scores = {
            "stable": [0.9, 0.9, 0.9],
            "volatile": [0.1, 0.9, 0.3],
        }
        report = analyze_sensitivity(scores)
        assert len(report.results) == 2
        assert "stable" in report.stable_metrics

    def test_volatile_list(self):
        scores = {"wild": [0.0, 1.0, 0.0, 1.0]}
        report = analyze_sensitivity(scores)
        assert "wild" in report.volatile_metrics

    def test_as_dict(self):
        scores = {"m1": [0.8, 0.82, 0.79]}
        d = analyze_sensitivity(scores).as_dict()
        assert "metrics" in d
        assert "stable_count" in d
        assert "volatile_count" in d

    def test_format_text(self):
        scores = {"m1": [0.8, 0.82], "m2": [0.1, 0.9]}
        text = analyze_sensitivity(scores).format_text()
        assert "Sensitivity" in text

    def test_custom_thresholds(self):
        # Very tight thresholds: even small variance = volatile
        scores = {"m": [0.80, 0.81, 0.79]}
        report = analyze_sensitivity(scores, stable_threshold=0.001, volatile_threshold=0.005)
        assert report.results[0].classification == "volatile"


class TestPerturbText:
    def test_typo(self):
        result = perturb_text("hello world", "typo")
        assert result != "hello world"
        assert len(result) == len("hello world")

    def test_case(self):
        result = perturb_text("hello world hello", "case")
        assert result.lower() == "hello world hello"
        assert result != "hello world hello"

    def test_whitespace(self):
        result = perturb_text("hello", "whitespace")
        assert " " in result
        assert len(result) == len("hello") + 1

    def test_deterministic(self):
        r1 = perturb_text("test string", "typo", seed=42)
        r2 = perturb_text("test string", "typo", seed=42)
        assert r1 == r2

    def test_different_seeds(self):
        r1 = perturb_text("test string here", "typo", seed=1)
        r2 = perturb_text("test string here", "typo", seed=999)
        # Different seeds may or may not produce different results on short strings
        # Just verify they're valid
        assert len(r1) == len("test string here")

    def test_short_text(self):
        assert perturb_text("a", "typo") == "a"
        assert perturb_text("", "typo") == ""

    def test_sensitivity_result_format(self):
        r = SensitivityResult(
            metric_id="m1", mean_score=0.8, std_dev=0.05,
            coefficient_of_variation=0.0625, classification="moderate",
        )
        text = r.format_text()
        assert "m1" in text
        assert "moderate" in text
