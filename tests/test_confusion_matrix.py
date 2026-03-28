"""Tests for the confusion matrix module."""

from __future__ import annotations

import math
import sys
from pathlib import Path

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.analysis.confusion_matrix import (
    ConfusionCell,
    ConfusionMatrix,
    ConfusionReport,
    build_confusion_matrix,
    build_confusion_report,
    compute_f1_scores,
    compute_kappa,
    compute_precision_recall,
    render_confusion_ascii,
)


# ---------------------------------------------------------------------------
# build_confusion_matrix
# ---------------------------------------------------------------------------


class TestBuildConfusionMatrix:
    def test_basic_binary(self):
        """Simple binary classification matrix."""
        preds = ["yes", "no", "yes", "yes", "no"]
        actual = ["yes", "no", "no", "yes", "no"]
        matrix = build_confusion_matrix(preds, actual)
        assert matrix.total == 5
        assert set(matrix.labels) == {"yes", "no"}
        # TP for yes: 2, FP for yes: 1
        assert matrix.get_cell("yes", "yes") == 2
        assert matrix.get_cell("yes", "no") == 1
        assert matrix.get_cell("no", "no") == 2
        assert matrix.get_cell("no", "yes") == 0

    def test_multiclass(self):
        """Three-class confusion matrix."""
        preds = ["a", "b", "c", "a", "b", "c"]
        actual = ["a", "b", "c", "b", "c", "a"]
        matrix = build_confusion_matrix(preds, actual)
        assert matrix.total == 6
        assert len(matrix.labels) == 3
        # Diagonal: 3 correct
        assert matrix.get_cell("a", "a") == 1
        assert matrix.get_cell("b", "b") == 1
        assert matrix.get_cell("c", "c") == 1

    def test_empty(self):
        """Empty inputs produce empty matrix."""
        matrix = build_confusion_matrix([], [])
        assert matrix.total == 0
        assert matrix.labels == []
        assert matrix.cells == []

    def test_length_mismatch_raises(self):
        """Mismatched lengths should raise ValueError."""
        try:
            build_confusion_matrix(["a", "b"], ["a"])
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_all_same_label(self):
        """All predictions and actuals the same."""
        preds = ["x", "x", "x"]
        actual = ["x", "x", "x"]
        matrix = build_confusion_matrix(preds, actual)
        assert matrix.total == 3
        assert matrix.labels == ["x"]
        assert matrix.get_cell("x", "x") == 3


# ---------------------------------------------------------------------------
# accuracy property
# ---------------------------------------------------------------------------


class TestAccuracy:
    def test_perfect_accuracy(self):
        """All correct predictions give accuracy 1.0."""
        matrix = build_confusion_matrix(["a", "b", "c"], ["a", "b", "c"])
        assert matrix.accuracy == 1.0

    def test_zero_accuracy(self):
        """All wrong predictions give accuracy 0.0."""
        matrix = build_confusion_matrix(["a", "a", "a"], ["b", "b", "b"])
        assert matrix.accuracy == 0.0

    def test_partial_accuracy(self):
        """Partial correctness."""
        matrix = build_confusion_matrix(["a", "b", "a"], ["a", "a", "a"])
        # 2 out of 3 correct (first and third)
        assert abs(matrix.accuracy - 2.0 / 3.0) < 1e-10

    def test_empty_accuracy(self):
        """Empty matrix accuracy is 0.0."""
        matrix = ConfusionMatrix()
        assert matrix.accuracy == 0.0


# ---------------------------------------------------------------------------
# get_cell
# ---------------------------------------------------------------------------


class TestGetCell:
    def test_existing_cell(self):
        """Should return count for existing cell."""
        matrix = build_confusion_matrix(["a", "b"], ["a", "b"])
        assert matrix.get_cell("a", "a") == 1

    def test_missing_cell(self):
        """Should return 0 for non-existent label pair."""
        matrix = build_confusion_matrix(["a", "b"], ["a", "b"])
        assert matrix.get_cell("z", "z") == 0

    def test_zero_count_cell(self):
        """Cell with zero observations."""
        matrix = build_confusion_matrix(["a", "a"], ["a", "b"])
        # "a" predicted as "b" actual: 1 time. "b" predicted as anything: 0.
        assert matrix.get_cell("b", "a") == 0


# ---------------------------------------------------------------------------
# compute_precision_recall
# ---------------------------------------------------------------------------


class TestPrecisionRecall:
    def test_perfect_classifier(self):
        """Perfect classifier has precision=1 and recall=1 for all labels."""
        matrix = build_confusion_matrix(
            ["a", "b", "c", "a", "b", "c"], ["a", "b", "c", "a", "b", "c"]
        )
        precision, recall = compute_precision_recall(matrix)
        for label in ["a", "b", "c"]:
            assert precision[label] == 1.0
            assert recall[label] == 1.0

    def test_partial_precision(self):
        """Check precision calculation with false positives."""
        # Predictions: a, a, b. Actuals: a, b, b.
        # For "a": TP=1, FP=1 -> precision=0.5
        # For "b": TP=1, FP=0 -> precision=1.0
        matrix = build_confusion_matrix(["a", "a", "b"], ["a", "b", "b"])
        precision, recall = compute_precision_recall(matrix)
        assert abs(precision["a"] - 0.5) < 1e-10
        assert abs(precision["b"] - 1.0) < 1e-10

    def test_partial_recall(self):
        """Check recall calculation with false negatives."""
        # Predictions: a, a, b. Actuals: a, b, b.
        # For "a": TP=1, FN=0 -> recall=1.0
        # For "b": TP=1, FN=1 -> recall=0.5
        matrix = build_confusion_matrix(["a", "a", "b"], ["a", "b", "b"])
        precision, recall = compute_precision_recall(matrix)
        assert abs(recall["a"] - 1.0) < 1e-10
        assert abs(recall["b"] - 0.5) < 1e-10

    def test_empty_matrix(self):
        """Empty matrix returns empty dicts."""
        matrix = ConfusionMatrix()
        precision, recall = compute_precision_recall(matrix)
        assert precision == {}
        assert recall == {}


# ---------------------------------------------------------------------------
# compute_f1_scores
# ---------------------------------------------------------------------------


class TestF1Scores:
    def test_perfect_f1(self):
        """Perfect precision and recall give F1=1.0."""
        f1 = compute_f1_scores({"a": 1.0, "b": 1.0}, {"a": 1.0, "b": 1.0})
        assert f1["a"] == 1.0
        assert f1["b"] == 1.0

    def test_zero_precision_and_recall(self):
        """Zero precision and recall give F1=0.0."""
        f1 = compute_f1_scores({"a": 0.0}, {"a": 0.0})
        assert f1["a"] == 0.0

    def test_harmonic_mean(self):
        """F1 should be harmonic mean of precision and recall."""
        p, r = 0.6, 0.8
        expected = 2 * p * r / (p + r)
        f1 = compute_f1_scores({"x": p}, {"x": r})
        assert abs(f1["x"] - expected) < 1e-10

    def test_asymmetric(self):
        """F1 penalizes imbalance between precision and recall."""
        f1 = compute_f1_scores({"a": 1.0}, {"a": 0.01})
        # With P=1.0 and R=0.01, F1 should be very low
        assert f1["a"] < 0.05


# ---------------------------------------------------------------------------
# build_confusion_report
# ---------------------------------------------------------------------------


class TestBuildConfusionReport:
    def test_full_report(self):
        """Report should contain matrix, accuracy, precision, recall, f1."""
        preds = ["a", "b", "a", "b"]
        actual = ["a", "b", "b", "a"]
        report = build_confusion_report(preds, actual)
        assert report.accuracy == 0.5
        assert "a" in report.precision
        assert "b" in report.recall
        assert "a" in report.f1

    def test_report_consistency(self):
        """Report accuracy should match matrix accuracy."""
        preds = ["yes", "no", "yes"]
        actual = ["yes", "no", "no"]
        report = build_confusion_report(preds, actual)
        assert abs(report.accuracy - report.matrix.accuracy) < 1e-10


# ---------------------------------------------------------------------------
# render_confusion_ascii
# ---------------------------------------------------------------------------


class TestRenderConfusionAscii:
    def test_basic_rendering(self):
        """Should produce a readable table."""
        matrix = build_confusion_matrix(["a", "b", "a"], ["a", "b", "b"])
        text = render_confusion_ascii(matrix)
        assert "Predicted" in text
        assert "Actual" in text
        # Should contain the labels
        assert "a" in text
        assert "b" in text

    def test_empty_matrix(self):
        """Empty matrix should produce placeholder text."""
        matrix = ConfusionMatrix()
        text = render_confusion_ascii(matrix)
        assert "empty" in text.lower()

    def test_contains_counts(self):
        """Rendered table should contain the cell counts."""
        preds = ["a", "a", "b", "b"]
        actual = ["a", "b", "a", "b"]
        matrix = build_confusion_matrix(preds, actual)
        text = render_confusion_ascii(matrix)
        assert "1" in text  # All cells are 1


# ---------------------------------------------------------------------------
# compute_kappa
# ---------------------------------------------------------------------------


class TestComputeKappa:
    def test_perfect_agreement(self):
        """Perfect agreement should give kappa=1.0."""
        matrix = build_confusion_matrix(
            ["a", "b", "c", "a", "b", "c"], ["a", "b", "c", "a", "b", "c"]
        )
        kappa = compute_kappa(matrix)
        assert abs(kappa - 1.0) < 1e-10

    def test_random_agreement(self):
        """Random-level agreement should give kappa near 0.

        With a balanced 2x2 matrix where predictions are random,
        kappa should be near zero.
        """
        # Uniform confusion: each cell has equal count
        preds = ["a", "a", "b", "b"]
        actual = ["a", "b", "a", "b"]
        matrix = build_confusion_matrix(preds, actual)
        kappa = compute_kappa(matrix)
        assert abs(kappa) < 0.01  # Near zero for chance agreement

    def test_empty_matrix_kappa(self):
        """Empty matrix gives kappa=0.0."""
        kappa = compute_kappa(ConfusionMatrix())
        assert kappa == 0.0

    def test_kappa_range(self):
        """Kappa should be in [-1, 1] range."""
        matrix = build_confusion_matrix(
            ["a", "b", "a", "b", "a"], ["a", "a", "b", "b", "a"]
        )
        kappa = compute_kappa(matrix)
        assert -1.0 <= kappa <= 1.0


# ---------------------------------------------------------------------------
# ConfusionReport format_text
# ---------------------------------------------------------------------------


class TestConfusionReportFormat:
    def test_format_includes_accuracy(self):
        """Format text should include accuracy."""
        report = build_confusion_report(["a", "b"], ["a", "b"])
        text = report.format_text()
        assert "accuracy" in text.lower()

    def test_format_includes_labels(self):
        """Format text should include per-label metrics."""
        report = build_confusion_report(["a", "b", "a"], ["a", "b", "b"])
        text = report.format_text()
        assert "a:" in text
        assert "b:" in text
        assert "P=" in text
        assert "R=" in text
        assert "F1=" in text


# ---------------------------------------------------------------------------
# as_dict / from_dict roundtrips
# ---------------------------------------------------------------------------


class TestRoundtrips:
    def test_confusion_cell_as_dict(self):
        """ConfusionCell as_dict should include all fields."""
        cell = ConfusionCell(predicted="yes", actual="no", count=5)
        d = cell.as_dict()
        assert d["predicted"] == "yes"
        assert d["actual"] == "no"
        assert d["count"] == 5

    def test_confusion_matrix_roundtrip(self):
        """ConfusionMatrix should survive as_dict -> from_dict."""
        original = build_confusion_matrix(["a", "b", "a"], ["a", "b", "b"])
        restored = ConfusionMatrix.from_dict(original.as_dict())
        assert restored.total == original.total
        assert restored.labels == original.labels
        assert len(restored.cells) == len(original.cells)
        assert restored.get_cell("a", "a") == original.get_cell("a", "a")

    def test_confusion_report_roundtrip(self):
        """ConfusionReport should survive as_dict -> from_dict."""
        original = build_confusion_report(["a", "b", "c"], ["a", "b", "c"])
        restored = ConfusionReport.from_dict(original.as_dict())
        assert abs(restored.accuracy - original.accuracy) < 1e-10
        assert restored.precision == original.precision
        assert restored.recall == original.recall
        assert restored.f1 == original.f1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_class(self):
        """Single class should work without errors."""
        report = build_confusion_report(["x", "x", "x"], ["x", "x", "x"])
        assert report.accuracy == 1.0
        assert report.precision["x"] == 1.0
        assert report.recall["x"] == 1.0
        assert report.f1["x"] == 1.0

    def test_many_classes(self):
        """Should handle many classes."""
        labels = [str(i) for i in range(20)]
        preds = labels * 2
        actual = labels * 2
        report = build_confusion_report(preds, actual)
        assert report.accuracy == 1.0

    def test_asymmetric_predictions(self):
        """Labels only in predictions but not actuals."""
        preds = ["a", "b", "c"]
        actual = ["a", "a", "a"]
        matrix = build_confusion_matrix(preds, actual)
        assert "b" in matrix.labels
        assert "c" in matrix.labels
        assert matrix.get_cell("b", "a") == 1
        assert matrix.get_cell("c", "a") == 1

    def test_kappa_with_single_class(self):
        """Kappa with single class: expected agreement = 1.0, so kappa = 0.0."""
        matrix = build_confusion_matrix(["a", "a"], ["a", "a"])
        kappa = compute_kappa(matrix)
        assert kappa == 0.0
