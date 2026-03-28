"""Tests for calibration difficulty weighting."""

import pytest
from evalyn_sdk.calibration.difficulty_weighting import (
    WeightedItem,
    WeightedAlignmentResult,
    assign_difficulty_weights,
    compute_weighted_accuracy,
    identify_hard_failures,
)


def _item(item_id, score, truth, difficulty):
    return WeightedItem(
        item_id=item_id,
        score=score,
        ground_truth=truth,
        difficulty=difficulty,
    )


# ---------------------------------------------------------------------------
# assign_difficulty_weights
# ---------------------------------------------------------------------------


class TestAssignDifficultyWeights:
    def test_inverse_method(self):
        items = [_item("a", 0.8, True, 0.2), _item("b", 0.3, False, 0.9)]
        assign_difficulty_weights(items, method="inverse")
        assert items[0].weight == pytest.approx(1.2)
        assert items[1].weight == pytest.approx(1.9)

    def test_exponential_method(self):
        items = [_item("a", 0.8, True, 0.0), _item("b", 0.3, False, 1.0)]
        assign_difficulty_weights(items, method="exponential")
        assert items[0].weight == pytest.approx(1.0)  # 2^0
        assert items[1].weight == pytest.approx(2.0)  # 2^1

    def test_exponential_mid(self):
        items = [_item("a", 0.5, True, 0.5)]
        assign_difficulty_weights(items, method="exponential")
        assert items[0].weight == pytest.approx(2.0 ** 0.5)

    def test_equal_method(self):
        items = [_item("a", 0.8, True, 0.9), _item("b", 0.3, False, 0.1)]
        assign_difficulty_weights(items, method="equal")
        assert items[0].weight == 1.0
        assert items[1].weight == 1.0

    def test_unknown_method_defaults_equal(self):
        items = [_item("a", 0.8, True, 0.7)]
        assign_difficulty_weights(items, method="unknown")
        assert items[0].weight == 1.0

    def test_empty_items(self):
        result = assign_difficulty_weights([], method="inverse")
        assert result == []

    def test_returns_same_list(self):
        items = [_item("a", 0.5, True, 0.5)]
        returned = assign_difficulty_weights(items, method="inverse")
        assert returned is items

    def test_all_same_difficulty(self):
        items = [_item(str(i), 0.5, True, 0.6) for i in range(5)]
        assign_difficulty_weights(items, method="inverse")
        for item in items:
            assert item.weight == pytest.approx(1.6)


# ---------------------------------------------------------------------------
# compute_weighted_accuracy
# ---------------------------------------------------------------------------


class TestComputeWeightedAccuracy:
    def test_basic_all_correct(self):
        items = [
            _item("a", 0.8, True, 0.2),
            _item("b", 0.3, False, 0.1),
        ]
        assign_difficulty_weights(items, method="equal")
        result = compute_weighted_accuracy(items)
        assert result.weighted_accuracy == pytest.approx(1.0)
        assert result.unweighted_accuracy == pytest.approx(1.0)

    def test_basic_none_correct(self):
        items = [
            _item("a", 0.3, True, 0.2),   # predicted False, truth True
            _item("b", 0.8, False, 0.1),   # predicted True, truth False
        ]
        assign_difficulty_weights(items, method="equal")
        result = compute_weighted_accuracy(items)
        assert result.weighted_accuracy == pytest.approx(0.0)
        assert result.unweighted_accuracy == pytest.approx(0.0)

    def test_weighted_vs_unweighted_differ(self):
        # Hard item correct, easy item wrong
        items = [
            _item("hard", 0.8, True, 0.9),  # correct, high weight
            _item("easy", 0.8, False, 0.1),  # wrong, low weight
        ]
        assign_difficulty_weights(items, method="inverse")
        result = compute_weighted_accuracy(items)
        # Weighted should be higher than unweighted (0.5)
        assert result.unweighted_accuracy == pytest.approx(0.5)
        assert result.weighted_accuracy > result.unweighted_accuracy

    def test_hard_vs_easy_accuracy(self):
        items = [
            _item("h1", 0.8, True, 0.8),   # hard, correct
            _item("h2", 0.3, True, 0.7),   # hard, wrong
            _item("e1", 0.8, True, 0.2),   # easy, correct
            _item("e2", 0.8, True, 0.3),   # easy, correct
        ]
        assign_difficulty_weights(items, method="equal")
        result = compute_weighted_accuracy(items)
        assert result.hard_item_accuracy == pytest.approx(0.5)
        assert result.easy_item_accuracy == pytest.approx(1.0)

    def test_num_items_and_total_weight(self):
        items = [_item("a", 0.8, True, 0.5), _item("b", 0.3, False, 0.5)]
        assign_difficulty_weights(items, method="equal")
        result = compute_weighted_accuracy(items)
        assert result.num_items == 2
        assert result.total_weight == pytest.approx(2.0)

    def test_custom_threshold(self):
        items = [_item("a", 0.6, True, 0.5)]
        assign_difficulty_weights(items, method="equal")
        # threshold 0.7: 0.6 < 0.7 -> predicted False, truth True -> wrong
        result = compute_weighted_accuracy(items, threshold=0.7)
        assert result.weighted_accuracy == pytest.approx(0.0)

    def test_empty_items(self):
        result = compute_weighted_accuracy([])
        assert result.num_items == 0
        assert result.weighted_accuracy == 0.0
        assert result.total_weight == 0.0

    def test_all_hard_items(self):
        items = [_item("h", 0.8, True, 0.9)]
        assign_difficulty_weights(items, method="equal")
        result = compute_weighted_accuracy(items)
        assert result.hard_item_accuracy == pytest.approx(1.0)
        assert result.easy_item_accuracy == pytest.approx(0.0)  # no easy items


# ---------------------------------------------------------------------------
# identify_hard_failures
# ---------------------------------------------------------------------------


class TestIdentifyHardFailures:
    def test_finds_hard_failures(self):
        items = [
            _item("h1", 0.3, True, 0.8),   # hard, wrong
            _item("h2", 0.8, True, 0.9),   # hard, correct
            _item("e1", 0.3, True, 0.2),   # easy, wrong - not hard
        ]
        failures = identify_hard_failures(items)
        assert len(failures) == 1
        assert failures[0].item_id == "h1"

    def test_none_when_all_correct(self):
        items = [
            _item("h1", 0.8, True, 0.8),
            _item("h2", 0.3, False, 0.9),
        ]
        failures = identify_hard_failures(items)
        assert failures == []

    def test_boundary_difficulty(self):
        # difficulty == 0.5 is NOT hard (need > 0.5)
        items = [_item("a", 0.3, True, 0.5)]
        failures = identify_hard_failures(items)
        assert failures == []

    def test_custom_threshold(self):
        items = [_item("a", 0.6, True, 0.8)]
        # threshold 0.7: 0.6 < 0.7 -> predicted False, truth True -> wrong
        failures = identify_hard_failures(items, threshold=0.7)
        assert len(failures) == 1

    def test_empty_items(self):
        assert identify_hard_failures([]) == []


# ---------------------------------------------------------------------------
# WeightedItem as_dict
# ---------------------------------------------------------------------------


class TestWeightedItemAsDict:
    def test_roundtrip(self):
        item = _item("x", 0.75, True, 0.6)
        item.weight = 1.5
        d = item.as_dict()
        assert d["item_id"] == "x"
        assert d["score"] == 0.75
        assert d["ground_truth"] is True
        assert d["difficulty"] == 0.6
        assert d["weight"] == 1.5


# ---------------------------------------------------------------------------
# WeightedAlignmentResult serialization and format
# ---------------------------------------------------------------------------


class TestWeightedAlignmentResult:
    def _make(self):
        return WeightedAlignmentResult(
            weighted_accuracy=0.85,
            unweighted_accuracy=0.80,
            num_items=10,
            total_weight=15.0,
            hard_item_accuracy=0.70,
            easy_item_accuracy=0.90,
        )

    def test_as_dict(self):
        d = self._make().as_dict()
        assert d["weighted_accuracy"] == 0.85
        assert d["num_items"] == 10

    def test_from_dict(self):
        d = self._make().as_dict()
        restored = WeightedAlignmentResult.from_dict(d)
        assert restored.weighted_accuracy == 0.85
        assert restored.num_items == 10

    def test_roundtrip(self):
        original = self._make()
        restored = WeightedAlignmentResult.from_dict(original.as_dict())
        assert original.as_dict() == restored.as_dict()

    def test_format_text(self):
        text = self._make().format_text()
        assert "Weighted Alignment" in text
        assert "0.8500" in text
        assert "0.8000" in text
        assert "Items: 10" in text
