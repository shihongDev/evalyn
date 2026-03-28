"""Tests for per-score-level calibration."""

import pytest
from evalyn_sdk.calibration.per_score_level import (
    ScoreLevelStats,
    PerLevelCalibration,
    classify_score_level,
    compute_per_level_stats,
    analyze_per_level,
    correct_bias,
)


# ---------------------------------------------------------------------------
# classify_score_level
# ---------------------------------------------------------------------------


class TestClassifyScoreLevel:
    def test_low(self):
        assert classify_score_level(0.1) == "low"
        assert classify_score_level(0.0) == "low"
        assert classify_score_level(0.39) == "low"

    def test_medium(self):
        assert classify_score_level(0.4) == "medium"
        assert classify_score_level(0.5) == "medium"
        assert classify_score_level(0.69) == "medium"

    def test_high(self):
        assert classify_score_level(0.7) == "high"
        assert classify_score_level(0.85) == "high"
        assert classify_score_level(1.0) == "high"

    def test_boundary_low_medium(self):
        assert classify_score_level(0.399) == "low"
        assert classify_score_level(0.4) == "medium"

    def test_boundary_medium_high(self):
        assert classify_score_level(0.699) == "medium"
        assert classify_score_level(0.7) == "high"


# ---------------------------------------------------------------------------
# compute_per_level_stats
# ---------------------------------------------------------------------------


class TestComputePerLevelStats:
    def test_accuracy_all_correct(self):
        predictions = [(0.8, True), (0.9, True)]  # both >= 0.5 and truth True
        stats = compute_per_level_stats(predictions, "high", (0.7, 1.0))
        assert stats.accuracy == pytest.approx(1.0)

    def test_accuracy_none_correct(self):
        predictions = [(0.3, True)]  # 0.3 < 0.5 -> predicted False, truth True
        stats = compute_per_level_stats(predictions, "low", (0.0, 0.4))
        assert stats.accuracy == pytest.approx(0.0)

    def test_accuracy_mixed(self):
        predictions = [
            (0.8, True),   # correct
            (0.3, True),   # wrong
        ]
        stats = compute_per_level_stats(predictions, "medium", (0.4, 0.7))
        assert stats.accuracy == pytest.approx(0.5)

    def test_bias_positive(self):
        # score=0.8, truth=False -> bias = 0.8 - 0.0 = 0.8
        predictions = [(0.8, False)]
        stats = compute_per_level_stats(predictions, "high", (0.7, 1.0))
        assert stats.bias == pytest.approx(0.8)

    def test_bias_negative(self):
        # score=0.1, truth=True -> bias = 0.1 - 1.0 = -0.9
        predictions = [(0.1, True)]
        stats = compute_per_level_stats(predictions, "low", (0.0, 0.4))
        assert stats.bias == pytest.approx(-0.9)

    def test_bias_zero(self):
        # score=1.0, truth=True -> bias = 1.0 - 1.0 = 0.0
        predictions = [(1.0, True)]
        stats = compute_per_level_stats(predictions, "high", (0.7, 1.0))
        assert stats.bias == pytest.approx(0.0)

    def test_item_count(self):
        predictions = [(0.5, True), (0.6, False), (0.55, True)]
        stats = compute_per_level_stats(predictions, "medium", (0.4, 0.7))
        assert stats.item_count == 3

    def test_empty_predictions(self):
        stats = compute_per_level_stats([], "low", (0.0, 0.4))
        assert stats.item_count == 0
        assert stats.accuracy == 0.0
        assert stats.bias == 0.0


# ---------------------------------------------------------------------------
# analyze_per_level
# ---------------------------------------------------------------------------


class TestAnalyzePerLevel:
    def test_finds_worst_level(self):
        predictions = [
            (0.1, True),   # low, wrong (0.1 < 0.5)
            (0.2, True),   # low, wrong
            (0.5, True),   # medium, correct
            (0.6, True),   # medium, correct
            (0.8, True),   # high, correct
        ]
        result = analyze_per_level(predictions)
        assert result.worst_level == "low"

    def test_overall_accuracy(self):
        predictions = [
            (0.8, True),   # correct
            (0.3, False),  # correct
            (0.8, False),  # wrong
            (0.3, True),   # wrong
        ]
        result = analyze_per_level(predictions)
        assert result.overall_accuracy == pytest.approx(0.5)

    def test_overall_bias(self):
        # All predictions 0.5, all truths True (1.0)
        # bias per item = 0.5 - 1.0 = -0.5
        predictions = [(0.5, True), (0.5, True)]
        result = analyze_per_level(predictions)
        assert result.overall_bias == pytest.approx(-0.5)

    def test_all_same_level(self):
        predictions = [(0.8, True), (0.9, True), (0.75, True)]
        result = analyze_per_level(predictions)
        # All are high
        high_stats = [lv for lv in result.levels if lv.level == "high"][0]
        assert high_stats.item_count == 3
        low_stats = [lv for lv in result.levels if lv.level == "low"][0]
        assert low_stats.item_count == 0

    def test_empty_predictions(self):
        result = analyze_per_level([])
        assert result.overall_accuracy == 0.0
        assert result.worst_level == ""
        assert result.levels == []

    def test_three_levels_present(self):
        predictions = [
            (0.1, False),  # low
            (0.5, True),   # medium
            (0.8, True),   # high
        ]
        result = analyze_per_level(predictions)
        assert len(result.levels) == 3
        names = [lv.level for lv in result.levels]
        assert "low" in names
        assert "medium" in names
        assert "high" in names


# ---------------------------------------------------------------------------
# correct_bias
# ---------------------------------------------------------------------------


class TestCorrectBias:
    def test_reduces_positive_bias(self):
        stats = {
            "high": ScoreLevelStats("high", (0.7, 1.0), 10, 0.9, 0.1),
        }
        corrected = correct_bias(0.8, stats)
        assert corrected == pytest.approx(0.7)

    def test_reduces_negative_bias(self):
        stats = {
            "low": ScoreLevelStats("low", (0.0, 0.4), 10, 0.5, -0.2),
        }
        corrected = correct_bias(0.2, stats)
        assert corrected == pytest.approx(0.4)

    def test_clamps_to_zero(self):
        stats = {
            "low": ScoreLevelStats("low", (0.0, 0.4), 10, 0.5, 0.5),
        }
        corrected = correct_bias(0.1, stats)
        assert corrected == pytest.approx(0.0)

    def test_clamps_to_one(self):
        stats = {
            "high": ScoreLevelStats("high", (0.7, 1.0), 10, 0.9, -0.5),
        }
        corrected = correct_bias(0.9, stats)
        assert corrected == pytest.approx(1.0)

    def test_missing_level_returns_original(self):
        corrected = correct_bias(0.5, {})
        assert corrected == pytest.approx(0.5)

    def test_zero_bias_no_change(self):
        stats = {
            "medium": ScoreLevelStats("medium", (0.4, 0.7), 10, 0.8, 0.0),
        }
        corrected = correct_bias(0.5, stats)
        assert corrected == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# PerLevelCalibration serialization and format
# ---------------------------------------------------------------------------


class TestPerLevelCalibration:
    def _make(self):
        return PerLevelCalibration(
            levels=[
                ScoreLevelStats("low", (0.0, 0.4), 5, 0.6, -0.1),
                ScoreLevelStats("medium", (0.4, 0.7), 10, 0.8, 0.05),
                ScoreLevelStats("high", (0.7, 1.0), 8, 0.9, 0.02),
            ],
            overall_accuracy=0.8,
            overall_bias=0.01,
            worst_level="low",
        )

    def test_as_dict(self):
        d = self._make().as_dict()
        assert d["overall_accuracy"] == 0.8
        assert d["worst_level"] == "low"
        assert len(d["levels"]) == 3

    def test_from_dict(self):
        d = self._make().as_dict()
        restored = PerLevelCalibration.from_dict(d)
        assert restored.overall_accuracy == 0.8
        assert restored.worst_level == "low"
        assert len(restored.levels) == 3
        assert restored.levels[0].level == "low"

    def test_roundtrip(self):
        original = self._make()
        restored = PerLevelCalibration.from_dict(original.as_dict())
        assert original.as_dict() == restored.as_dict()

    def test_score_range_is_tuple_after_roundtrip(self):
        original = self._make()
        restored = PerLevelCalibration.from_dict(original.as_dict())
        for lv in restored.levels:
            assert isinstance(lv.score_range, tuple)

    def test_format_text(self):
        text = self._make().format_text()
        assert "Per-Level Calibration" in text
        assert "0.8000" in text
        assert "low" in text
        assert "medium" in text
        assert "high" in text

    def test_empty_from_dict(self):
        restored = PerLevelCalibration.from_dict({})
        assert restored.levels == []
        assert restored.worst_level == ""


# ---------------------------------------------------------------------------
# ScoreLevelStats as_dict
# ---------------------------------------------------------------------------


class TestScoreLevelStats:
    def test_as_dict(self):
        stats = ScoreLevelStats("low", (0.0, 0.4), 5, 0.6, -0.1)
        d = stats.as_dict()
        assert d["level"] == "low"
        assert d["score_range"] == [0.0, 0.4]
        assert d["item_count"] == 5
        assert d["accuracy"] == 0.6
        assert d["bias"] == -0.1
