"""Tests for adaptive consistency sampling.

All tests use synthetic data - no storage or ML deps required.
"""

from __future__ import annotations

import pytest

from evalyn_sdk.evaluation.adaptive_consistency import (
    AdaptiveConfig,
    AdaptiveReport,
    AdaptiveSampleResult,
    compute_adaptive_report,
    compute_agreement_rate,
    run_adaptive_sampling,
    should_stop_early,
)


# ---------------------------------------------------------------------------
# compute_agreement_rate
# ---------------------------------------------------------------------------


class TestComputeAgreementRate:
    def test_all_true(self):
        assert compute_agreement_rate([True, True, True]) == 1.0

    def test_all_false(self):
        assert compute_agreement_rate([False, False, False]) == 1.0

    def test_two_thirds(self):
        rate = compute_agreement_rate([True, True, False])
        assert abs(rate - 2.0 / 3.0) < 1e-9

    def test_even_split(self):
        rate = compute_agreement_rate([True, False])
        assert rate == 0.5

    def test_single_decision(self):
        assert compute_agreement_rate([True]) == 1.0

    def test_empty(self):
        assert compute_agreement_rate([]) == 0.0

    def test_four_out_of_five(self):
        rate = compute_agreement_rate([True, True, True, True, False])
        assert abs(rate - 0.8) < 1e-9


# ---------------------------------------------------------------------------
# should_stop_early
# ---------------------------------------------------------------------------


class TestShouldStopEarly:
    def test_below_min_samples(self):
        config = AdaptiveConfig(max_samples=5, min_samples=3, agreement_threshold=1.0)
        assert not should_stop_early([True, True], config)

    def test_at_min_samples_full_agreement(self):
        config = AdaptiveConfig(max_samples=5, min_samples=3, agreement_threshold=1.0)
        assert should_stop_early([True, True, True], config)

    def test_at_min_samples_partial_agreement(self):
        config = AdaptiveConfig(max_samples=5, min_samples=3, agreement_threshold=1.0)
        assert not should_stop_early([True, True, False], config)

    def test_threshold_below_one(self):
        config = AdaptiveConfig(max_samples=5, min_samples=3, agreement_threshold=0.6)
        # 2/3 = 0.667 >= 0.6
        assert should_stop_early([True, True, False], config)

    def test_min_samples_one(self):
        config = AdaptiveConfig(max_samples=5, min_samples=1, agreement_threshold=1.0)
        assert should_stop_early([False], config)

    def test_empty_decisions(self):
        config = AdaptiveConfig(max_samples=5, min_samples=3, agreement_threshold=1.0)
        assert not should_stop_early([], config)


# ---------------------------------------------------------------------------
# run_adaptive_sampling
# ---------------------------------------------------------------------------


class TestRunAdaptiveSampling:
    def test_all_agree_early_stop(self):
        config = AdaptiveConfig(max_samples=5, min_samples=3, agreement_threshold=1.0)
        call_count = 0

        def always_true():
            nonlocal call_count
            call_count += 1
            return True

        result = run_adaptive_sampling("item-1", always_true, config)
        assert result.item_id == "item-1"
        assert result.samples_taken == 3
        assert result.early_stopped is True
        assert result.agreement_rate == 1.0
        assert result.final_decision is True
        assert call_count == 3

    def test_disagreement_runs_to_max(self):
        config = AdaptiveConfig(max_samples=5, min_samples=3, agreement_threshold=1.0)
        toggle = iter([True, False, True, False, True])

        result = run_adaptive_sampling("item-2", lambda: next(toggle), config)
        assert result.samples_taken == 5
        assert result.early_stopped is False

    def test_savings_pct_correct(self):
        config = AdaptiveConfig(max_samples=10, min_samples=3, agreement_threshold=1.0)
        result = run_adaptive_sampling("item-3", lambda: False, config)
        # Took 3 of 10 -> saved 70%
        assert result.savings_pct == 70.0

    def test_final_decision_majority_true(self):
        config = AdaptiveConfig(max_samples=3, min_samples=3, agreement_threshold=0.5)
        decisions = iter([True, True, False])
        result = run_adaptive_sampling("item-4", lambda: next(decisions), config)
        assert result.final_decision is True

    def test_final_decision_majority_false(self):
        config = AdaptiveConfig(max_samples=3, min_samples=3, agreement_threshold=0.5)
        decisions = iter([False, False, True])
        result = run_adaptive_sampling("item-5", lambda: next(decisions), config)
        assert result.final_decision is False

    def test_max_samples_one(self):
        config = AdaptiveConfig(max_samples=1, min_samples=1, agreement_threshold=1.0)
        result = run_adaptive_sampling("item-6", lambda: True, config)
        assert result.samples_taken == 1
        assert result.savings_pct == 0.0
        # With max=1, min=1, it stops after 1 sample but that equals max
        assert result.early_stopped is False

    def test_samples_max_in_result(self):
        config = AdaptiveConfig(max_samples=7, min_samples=2, agreement_threshold=1.0)
        result = run_adaptive_sampling("item-7", lambda: True, config)
        assert result.samples_max == 7


# ---------------------------------------------------------------------------
# compute_adaptive_report
# ---------------------------------------------------------------------------


class TestComputeAdaptiveReport:
    def test_aggregate_savings(self):
        r1 = AdaptiveSampleResult("a", 3, 5, True, 1.0, True, 40.0)
        r2 = AdaptiveSampleResult("b", 5, 5, False, 0.8, True, 0.0)
        report = compute_adaptive_report([r1, r2])
        assert report.total_samples_taken == 8
        assert report.total_samples_possible == 10
        assert abs(report.overall_savings_pct - 20.0) < 1e-9

    def test_empty_results(self):
        report = compute_adaptive_report([])
        assert report.total_samples_taken == 0
        assert report.overall_savings_pct == 0.0

    def test_no_savings(self):
        r1 = AdaptiveSampleResult("a", 5, 5, False, 0.6, True, 0.0)
        report = compute_adaptive_report([r1])
        assert report.overall_savings_pct == 0.0


# ---------------------------------------------------------------------------
# Serialization roundtrips
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_config_roundtrip(self):
        cfg = AdaptiveConfig(max_samples=7, min_samples=2, agreement_threshold=0.8)
        d = cfg.as_dict()
        restored = AdaptiveConfig.from_dict(d)
        assert restored.max_samples == 7
        assert restored.min_samples == 2
        assert restored.agreement_threshold == 0.8

    def test_config_from_dict_defaults(self):
        cfg = AdaptiveConfig.from_dict({})
        assert cfg.max_samples == 5
        assert cfg.min_samples == 3
        assert cfg.agreement_threshold == 1.0

    def test_sample_result_as_dict(self):
        r = AdaptiveSampleResult("x", 3, 5, True, 1.0, True, 40.0)
        d = r.as_dict()
        assert d["item_id"] == "x"
        assert d["early_stopped"] is True

    def test_report_roundtrip(self):
        r1 = AdaptiveSampleResult("a", 3, 5, True, 1.0, True, 40.0)
        report = compute_adaptive_report([r1])
        d = report.as_dict()
        restored = AdaptiveReport.from_dict(d)
        assert len(restored.results) == 1
        assert restored.total_samples_taken == 3


# ---------------------------------------------------------------------------
# format_text
# ---------------------------------------------------------------------------


class TestFormatText:
    def test_report_format_text_contains_header(self):
        r1 = AdaptiveSampleResult("a", 3, 5, True, 1.0, True, 40.0)
        report = compute_adaptive_report([r1])
        text = report.format_text()
        assert "Adaptive Consistency Report" in text
        assert "Samples taken:" in text
        assert "Early stopped:" in text

    def test_report_format_text_savings(self):
        r1 = AdaptiveSampleResult("a", 3, 10, True, 1.0, True, 70.0)
        report = compute_adaptive_report([r1])
        text = report.format_text()
        assert "70.0%" in text
