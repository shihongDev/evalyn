"""Tests for confidence method comparison."""

import pytest
from evalyn_sdk.evaluation.confidence_compare import (
    CalibrationBin,
    MethodCalibration,
    ComparisonReport,
    compute_calibration,
    compare_methods,
)


# =============================================================================
# CalibrationBin
# =============================================================================

class TestCalibrationBin:
    def test_as_dict(self):
        b = CalibrationBin(0.0, 0.1, 0.05, 0.06, 10)
        d = b.as_dict()
        assert d["bin_start"] == 0.0
        assert d["count"] == 10


# =============================================================================
# MethodCalibration
# =============================================================================

class TestMethodCalibration:
    def test_as_dict(self):
        m = MethodCalibration(method="logprobs", ece=0.05, mce=0.1, brier_score=0.2, num_samples=50)
        d = m.as_dict()
        assert d["method"] == "logprobs"
        assert d["ece"] == 0.05

    def test_from_dict(self):
        d = {"method": "consistency", "ece": 0.03, "brier_score": 0.15, "num_samples": 30, "bins": []}
        m = MethodCalibration.from_dict(d)
        assert m.method == "consistency"
        assert m.ece == 0.03

    def test_format_text(self):
        m = MethodCalibration(method="logprobs", ece=0.05, mce=0.1, brier_score=0.2, num_samples=50)
        text = m.format_text()
        assert "logprobs" in text
        assert "ECE" in text


# =============================================================================
# compute_calibration
# =============================================================================

class TestComputeCalibration:
    def test_empty(self):
        m = compute_calibration("test", [])
        assert m.num_samples == 0
        assert m.ece == 0.0

    def test_perfect_calibration(self):
        # Confidence = actual accuracy in each bin
        preds = [(0.1, False)] * 9 + [(0.1, True)] * 1  # 10% conf, 10% correct
        preds += [(0.9, True)] * 9 + [(0.9, False)] * 1  # 90% conf, 90% correct
        m = compute_calibration("perfect", preds, num_bins=10)
        assert m.ece < 0.05  # should be very low

    def test_overconfident(self):
        # High confidence, low accuracy
        preds = [(0.9, False)] * 10
        m = compute_calibration("overconf", preds, num_bins=10)
        assert m.ece > 0.5
        assert m.brier_score > 0.5

    def test_underconfident(self):
        # Low confidence, high accuracy
        preds = [(0.1, True)] * 10
        m = compute_calibration("underconf", preds, num_bins=10)
        assert m.ece > 0.5

    def test_brier_score_range(self):
        preds = [(0.5, True)] * 5 + [(0.5, False)] * 5
        m = compute_calibration("mid", preds)
        assert 0.0 <= m.brier_score <= 1.0

    def test_bins_populated(self):
        preds = [(i / 10, i % 2 == 0) for i in range(10)]
        m = compute_calibration("spread", preds, num_bins=5)
        assert len(m.bins) > 0
        assert all(b.count > 0 for b in m.bins)

    def test_all_correct(self):
        preds = [(0.9, True)] * 20
        m = compute_calibration("all_right", preds)
        assert m.brier_score < 0.05

    def test_all_wrong(self):
        preds = [(0.9, False)] * 20
        m = compute_calibration("all_wrong", preds)
        assert m.brier_score > 0.5


# =============================================================================
# compare_methods
# =============================================================================

class TestCompareMethods:
    def test_empty(self):
        r = compare_methods({})
        assert len(r.calibrations) == 0
        assert r.recommended_method == ""

    def test_single_method(self):
        preds = {"logprobs": [(0.8, True)] * 10}
        r = compare_methods(preds)
        assert len(r.calibrations) == 1
        assert r.recommended_method == "logprobs"

    def test_recommends_lowest_ece(self):
        # Well-calibrated method
        good = [(0.1, False)] * 9 + [(0.1, True)] + [(0.9, True)] * 9 + [(0.9, False)]
        # Poorly calibrated method
        bad = [(0.9, False)] * 10 + [(0.1, True)] * 10

        r = compare_methods({"good": good, "bad": bad})
        assert r.recommended_method == "good"
        assert r.best_by_ece == "good"

    def test_as_dict(self):
        preds = {"m1": [(0.5, True)]}
        r = compare_methods(preds)
        d = r.as_dict()
        assert "recommended_method" in d
        assert "methods" in d

    def test_from_dict(self):
        r = ComparisonReport(
            calibrations=[MethodCalibration(method="x", ece=0.05)],
            recommended_method="x",
        )
        d = r.as_dict()
        restored = ComparisonReport.from_dict(d)
        assert restored.recommended_method == "x"
        assert len(restored.calibrations) == 1

    def test_format_text(self):
        preds = {"logprobs": [(0.8, True)] * 5}
        r = compare_methods(preds)
        text = r.format_text()
        assert "Comparison" in text
        assert "logprobs" in text
        assert "Recommended" in text
