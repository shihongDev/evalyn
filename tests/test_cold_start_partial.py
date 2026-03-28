"""Tests for cold start detection and partial result access."""

import json
import pytest
from pathlib import Path
from evalyn_sdk.analysis.cold_start import (
    detect_cold_start, ColdStartResult, ColdStartReport,
)
from evalyn_sdk.evaluation.partial_results import (
    load_partial_results, estimate_remaining_time, PartialResults,
)


# =============================================================================
# Cold start detection tests
# =============================================================================

class TestDetectColdStart:
    def test_no_cold_start(self):
        # Consistent scores throughout
        scores = {"m": [0.8, 0.82, 0.79, 0.81, 0.80, 0.78, 0.83, 0.80, 0.79, 0.81]}
        report = detect_cold_start(scores, cold_window=3)
        assert not report.has_cold_starts

    def test_cold_start_detected(self):
        # First 5 items score much lower than rest
        cold = [0.2, 0.3, 0.25, 0.28, 0.22]
        warm = [0.85, 0.9, 0.88, 0.87, 0.92, 0.89, 0.91, 0.86, 0.88, 0.90]
        scores = {"m": cold + warm}
        report = detect_cold_start(scores, cold_window=5)
        assert report.has_cold_starts
        assert "m" in report.affected_metrics

    def test_too_few_items(self):
        scores = {"m": [0.5, 0.6, 0.7]}
        report = detect_cold_start(scores, cold_window=5)
        assert not report.has_cold_starts
        assert report.results[0].p_value_approx == 1.0

    def test_empty(self):
        report = detect_cold_start({})
        assert not report.has_cold_starts
        assert len(report.results) == 0

    def test_multiple_metrics(self):
        scores = {
            "stable": [0.8] * 20,
            "drifting": [0.2] * 5 + [0.9] * 15,
        }
        report = detect_cold_start(scores, cold_window=5)
        assert "drifting" in report.affected_metrics
        assert "stable" not in report.affected_metrics

    def test_result_fields(self):
        scores = {"m": [0.3, 0.3, 0.3, 0.3, 0.3, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]}
        report = detect_cold_start(scores, cold_window=5)
        r = report.results[0]
        assert r.cold_count == 5
        assert r.warm_count == 7
        assert r.cold_mean < r.warm_mean
        assert r.delta > 0

    def test_as_dict(self):
        scores = {"m": [0.5] * 10}
        d = detect_cold_start(scores).as_dict()
        assert "affected_metrics" in d
        assert "results" in d

    def test_format_text(self):
        scores = {"m": [0.5] * 10}
        text = detect_cold_start(scores).format_text()
        assert "Cold Start" in text

    def test_cold_start_result_format(self):
        r = ColdStartResult(
            metric_id="m", cold_start_detected=True,
            cold_mean=0.3, warm_mean=0.9, delta=0.6,
            p_value_approx=0.001, cold_count=5, warm_count=15,
        )
        assert "DETECTED" in r.format_text()


# =============================================================================
# Partial results tests
# =============================================================================

class TestLoadPartialResults:
    def test_load_checkpoint(self, tmp_path):
        checkpoint = tmp_path / "checkpoint.json"
        data = {
            "run_id": "run-123",
            "completed_items": ["i1", "i2", "i3"],
            "results": [
                {"metric_id": "a", "item_id": "i1", "passed": True, "score": 0.9},
                {"metric_id": "a", "item_id": "i2", "passed": False, "score": 0.3},
                {"metric_id": "b", "item_id": "i1", "passed": True, "score": 0.8},
            ],
        }
        checkpoint.write_text(json.dumps(data))

        partial = load_partial_results(checkpoint)
        assert partial is not None
        assert partial.run_id == "run-123"
        assert partial.completed_items == 3
        assert partial.total_results == 3

    def test_missing_checkpoint(self, tmp_path):
        assert load_partial_results(tmp_path / "nonexistent.json") is None

    def test_corrupt_checkpoint(self, tmp_path):
        checkpoint = tmp_path / "bad.json"
        checkpoint.write_text("not valid json{{{")
        assert load_partial_results(checkpoint) is None

    def test_pass_rate_estimate(self):
        partial = PartialResults(
            run_id="r1", completed_items=3, total_results=4,
            metric_results=[
                {"metric_id": "a", "item_id": "i1", "passed": True},
                {"metric_id": "a", "item_id": "i2", "passed": True},
                {"metric_id": "a", "item_id": "i3", "passed": False},
                {"metric_id": "b", "item_id": "i1", "passed": True},
            ],
        )
        # i1: all pass, i2: pass, i3: fail -> 2/3 items pass
        assert partial.pass_rate_estimate == pytest.approx(2/3, abs=0.01)

    def test_metric_pass_rates(self):
        partial = PartialResults(
            run_id="r1", completed_items=2, total_results=4,
            metric_results=[
                {"metric_id": "a", "item_id": "i1", "passed": True},
                {"metric_id": "a", "item_id": "i2", "passed": False},
                {"metric_id": "b", "item_id": "i1", "passed": True},
                {"metric_id": "b", "item_id": "i2", "passed": True},
            ],
        )
        rates = partial.metric_pass_rates
        assert rates["a"] == 0.5
        assert rates["b"] == 1.0

    def test_empty_results(self):
        partial = PartialResults(run_id="r1", completed_items=0, total_results=0)
        assert partial.pass_rate_estimate == 0.0
        assert partial.metric_pass_rates == {}

    def test_as_dict(self):
        partial = PartialResults(
            run_id="r1", completed_items=5, total_results=10,
            metric_results=[{"metric_id": "a", "item_id": "i1", "passed": True}],
        )
        d = partial.as_dict()
        assert "run_id" in d
        assert "pass_rate_estimate" in d
        assert "metric_pass_rates" in d

    def test_format_text(self):
        partial = PartialResults(
            run_id="r1", completed_items=5, total_results=10,
            metric_results=[{"metric_id": "a", "item_id": "i1", "passed": True}],
        )
        text = partial.format_text()
        assert "Partial Results" in text


class TestEstimateRemainingTime:
    def test_basic_estimate(self):
        partial = PartialResults(run_id="r1", completed_items=50, total_results=100)
        est = estimate_remaining_time(partial, total_items=100, elapsed_seconds=60.0)
        assert est["remaining_items"] == 50
        assert est["eta_seconds"] > 0
        assert est["items_per_second"] > 0

    def test_zero_completed(self):
        partial = PartialResults(run_id="r1", completed_items=0, total_results=0)
        est = estimate_remaining_time(partial, total_items=100, elapsed_seconds=10.0)
        assert est["items_per_second"] == 0.0
        assert est["remaining_items"] == 100

    def test_complete(self):
        partial = PartialResults(run_id="r1", completed_items=100, total_results=200)
        est = estimate_remaining_time(partial, total_items=100, elapsed_seconds=120.0)
        assert est["remaining_items"] == 0
        assert est["progress_pct"] == 100.0
