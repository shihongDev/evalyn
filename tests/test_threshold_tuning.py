"""Tests for confidence threshold tuning."""

import pytest
from evalyn_sdk.evaluation.threshold_tuning import (
    ThresholdCandidate,
    ThresholdResult,
    compute_f1,
    compute_accuracy,
    evaluate_threshold,
    find_optimal_threshold,
    tune_thresholds,
)


# --- compute_f1 ---


class TestComputeF1:
    def test_perfect_prediction(self):
        preds = [True, False, True, False]
        truth = [True, False, True, False]
        assert compute_f1(preds, truth) == 1.0

    def test_all_wrong(self):
        preds = [True, True, True]
        truth = [False, False, False]
        assert compute_f1(preds, truth) == 0.0

    def test_empty_lists(self):
        assert compute_f1([], []) == 0.0

    def test_all_positive_ground_truth(self):
        preds = [True, True, True]
        truth = [True, True, True]
        assert compute_f1(preds, truth) == 1.0

    def test_all_negative_ground_truth(self):
        # No true positives possible
        preds = [False, False, False]
        truth = [False, False, False]
        assert compute_f1(preds, truth) == 0.0

    def test_all_false_predictions_with_true_labels(self):
        preds = [False, False, False]
        truth = [True, True, True]
        assert compute_f1(preds, truth) == 0.0

    def test_partial_overlap(self):
        preds = [True, True, False, False]
        truth = [True, False, True, False]
        # tp=1, fp=1, fn=1 -> precision=0.5, recall=0.5 -> f1=0.5
        assert abs(compute_f1(preds, truth) - 0.5) < 1e-9

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            compute_f1([True], [True, False])

    def test_single_true_positive(self):
        assert compute_f1([True], [True]) == 1.0

    def test_single_false_positive(self):
        assert compute_f1([True], [False]) == 0.0


# --- compute_accuracy ---


class TestComputeAccuracy:
    def test_basic(self):
        preds = [True, False, True, False]
        truth = [True, False, True, False]
        assert compute_accuracy(preds, truth) == 1.0

    def test_half_correct(self):
        preds = [True, True, False, False]
        truth = [True, False, True, False]
        assert compute_accuracy(preds, truth) == 0.5

    def test_empty(self):
        assert compute_accuracy([], []) == 0.0

    def test_all_wrong(self):
        preds = [True, True]
        truth = [False, False]
        assert compute_accuracy(preds, truth) == 0.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            compute_accuracy([True], [True, False])


# --- evaluate_threshold ---


class TestEvaluateThreshold:
    def test_low_threshold_all_positive(self):
        scores = [0.1, 0.5, 0.9]
        truth = [False, True, True]
        result = evaluate_threshold(scores, truth, 0.0)
        assert result.threshold == 0.0
        # All predicted True: tp=2, fp=1, precision=2/3, recall=1
        assert result.recall == 1.0
        assert abs(result.precision - 2.0 / 3.0) < 1e-9

    def test_high_threshold_all_negative(self):
        scores = [0.1, 0.5, 0.9]
        truth = [False, True, True]
        result = evaluate_threshold(scores, truth, 1.0)
        # All predicted False: tp=0
        assert result.f1_score == 0.0

    def test_mid_threshold(self):
        scores = [0.2, 0.4, 0.6, 0.8]
        truth = [False, False, True, True]
        result = evaluate_threshold(scores, truth, 0.5)
        # predictions: [F, F, T, T] -> perfect match
        assert result.f1_score == 1.0
        assert result.accuracy == 1.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            evaluate_threshold([0.5], [True, False], 0.5)

    def test_exact_threshold_value_included(self):
        scores = [0.5]
        truth = [True]
        result = evaluate_threshold(scores, truth, 0.5)
        # 0.5 >= 0.5 -> True
        assert result.f1_score == 1.0


# --- find_optimal_threshold ---


class TestFindOptimalThreshold:
    def test_selects_best_f1(self):
        scores = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
        truth = [False, False, False, True, True, True]
        result = find_optimal_threshold("my_metric", scores, truth, num_candidates=10)
        assert result.metric_id == "my_metric"
        assert result.best_f1 > 0.0
        assert result.num_samples == 6
        assert len(result.candidates) == 10

    def test_perfect_data(self):
        scores = [0.0, 0.0, 1.0, 1.0]
        truth = [False, False, True, True]
        result = find_optimal_threshold("perfect", scores, truth, num_candidates=20)
        assert result.best_f1 == 1.0

    def test_empty_scores(self):
        result = find_optimal_threshold("empty", [], [], num_candidates=10)
        assert result.best_f1 == 0.0
        assert result.num_samples == 0
        assert result.candidates == []

    def test_single_score(self):
        result = find_optimal_threshold("single", [0.5], [True], num_candidates=5)
        assert result.num_samples == 1
        # Single score: only one candidate at score value
        assert len(result.candidates) == 1

    def test_all_same_scores(self):
        scores = [0.5, 0.5, 0.5]
        truth = [True, False, True]
        result = find_optimal_threshold("same", scores, truth, num_candidates=10)
        assert result.num_samples == 3
        assert len(result.candidates) == 1

    def test_grid_covers_range(self):
        scores = [0.0, 1.0]
        truth = [False, True]
        result = find_optimal_threshold("range", scores, truth, num_candidates=5)
        thresholds = [c.threshold for c in result.candidates]
        assert min(thresholds) == pytest.approx(0.0)
        assert max(thresholds) == pytest.approx(1.0)
        assert len(thresholds) == 5

    def test_optimal_threshold_in_candidates(self):
        scores = [0.1, 0.3, 0.5, 0.7, 0.9]
        truth = [False, False, True, True, True]
        result = find_optimal_threshold("check", scores, truth, num_candidates=20)
        candidate_thresholds = [c.threshold for c in result.candidates]
        assert result.optimal_threshold in candidate_thresholds


# --- tune_thresholds ---


class TestTuneThresholds:
    def test_multiple_metrics(self):
        metrics = {
            "accuracy": ([0.1, 0.9], [False, True]),
            "helpfulness": ([0.2, 0.8, 0.5], [False, True, True]),
        }
        results = tune_thresholds(metrics)
        assert "accuracy" in results
        assert "helpfulness" in results
        assert results["accuracy"].metric_id == "accuracy"
        assert results["helpfulness"].metric_id == "helpfulness"

    def test_empty_metrics(self):
        results = tune_thresholds({})
        assert results == {}

    def test_single_metric(self):
        metrics = {"m1": ([0.0, 1.0], [False, True])}
        results = tune_thresholds(metrics)
        assert len(results) == 1
        assert results["m1"].best_f1 == 1.0


# --- ThresholdResult format_text / as_dict / from_dict ---


class TestThresholdResultSerialization:
    def test_format_text(self):
        result = ThresholdResult(
            metric_id="test_metric",
            optimal_threshold=0.65,
            best_f1=0.88,
            candidates=[],
            num_samples=100,
        )
        text = result.format_text()
        assert "test_metric" in text
        assert "0.6500" in text
        assert "0.8800" in text
        assert "100" in text

    def test_as_dict_from_dict_roundtrip(self):
        candidate = ThresholdCandidate(
            threshold=0.5,
            accuracy=0.9,
            precision=0.85,
            recall=0.95,
            f1_score=0.8974,
        )
        original = ThresholdResult(
            metric_id="roundtrip",
            optimal_threshold=0.5,
            best_f1=0.8974,
            candidates=[candidate],
            num_samples=50,
        )
        data = original.as_dict()
        restored = ThresholdResult.from_dict(data)
        assert restored.metric_id == original.metric_id
        assert restored.optimal_threshold == pytest.approx(original.optimal_threshold)
        assert restored.best_f1 == pytest.approx(original.best_f1)
        assert restored.num_samples == original.num_samples
        assert len(restored.candidates) == 1
        assert restored.candidates[0].threshold == pytest.approx(0.5)

    def test_as_dict_rounds_values(self):
        candidate = ThresholdCandidate(
            threshold=0.123456789,
            accuracy=0.999999999,
            precision=0.5,
            recall=0.5,
            f1_score=0.5,
        )
        d = candidate.as_dict()
        assert d["threshold"] == 0.123457  # rounded to 6 decimals

    def test_from_dict_missing_candidates(self):
        data = {
            "metric_id": "m",
            "optimal_threshold": 0.5,
            "best_f1": 0.8,
        }
        result = ThresholdResult.from_dict(data)
        assert result.candidates == []
        assert result.num_samples == 0
