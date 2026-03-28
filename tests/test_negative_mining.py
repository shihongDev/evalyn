"""Tests for calibration negative mining."""

from __future__ import annotations

import pytest

from evalyn_sdk.calibration.negative_mining import (
    MiningResult,
    NegativeExample,
    analyze_failure_patterns,
    find_negatives,
    mine_hard_negatives,
    rank_by_difficulty,
)


# ---------------------------------------------------------------------------
# NegativeExample
# ---------------------------------------------------------------------------


class TestNegativeExample:
    def test_as_dict(self):
        ne = NegativeExample("a", 0.9, False, 0.9, 0.5)
        d = ne.as_dict()
        assert d == {
            "item_id": "a",
            "predicted_score": 0.9,
            "ground_truth": False,
            "error_magnitude": 0.9,
            "difficulty": 0.5,
        }

    def test_default_difficulty(self):
        ne = NegativeExample("b", 0.3, True, 0.7)
        assert ne.difficulty == 0.0


# ---------------------------------------------------------------------------
# MiningResult
# ---------------------------------------------------------------------------


class TestMiningResult:
    def test_as_dict_roundtrip(self):
        neg = NegativeExample("x", 0.8, False, 0.8)
        result = MiningResult(
            negatives=[neg],
            total_failures=1,
            hardest_failures=[neg],
            failure_rate=0.5,
            common_patterns=["pattern1"],
        )
        d = result.as_dict()
        restored = MiningResult.from_dict(d)
        assert restored.total_failures == 1
        assert restored.failure_rate == 0.5
        assert len(restored.negatives) == 1
        assert restored.negatives[0].item_id == "x"
        assert len(restored.hardest_failures) == 1
        assert restored.common_patterns == ["pattern1"]

    def test_from_dict_defaults(self):
        result = MiningResult.from_dict({})
        assert result.negatives == []
        assert result.total_failures == 0
        assert result.failure_rate == 0.0
        assert result.common_patterns == []

    def test_format_text_basic(self):
        neg = NegativeExample("item1", 0.9, False, 0.9)
        result = MiningResult(
            negatives=[neg],
            total_failures=1,
            hardest_failures=[neg],
            failure_rate=0.25,
            common_patterns=["too many false positives"],
        )
        text = result.format_text()
        assert "Negative Mining Results" in text
        assert "Total failures: 1" in text
        assert "25.00%" in text
        assert "item1" in text
        assert "too many false positives" in text

    def test_format_text_empty(self):
        result = MiningResult([], 0, [], 0.0)
        text = result.format_text()
        assert "Total failures: 0" in text
        assert "Hardest" not in text


# ---------------------------------------------------------------------------
# find_negatives
# ---------------------------------------------------------------------------


class TestFindNegatives:
    def test_finds_false_positive(self):
        # score=0.8 >= 0.5 predicts True, but truth is False
        preds = [("a", 0.8, False)]
        negs = find_negatives(preds)
        assert len(negs) == 1
        assert negs[0].item_id == "a"
        assert negs[0].ground_truth is False
        assert negs[0].error_magnitude == pytest.approx(0.8)

    def test_finds_false_negative(self):
        # score=0.2 < 0.5 predicts False, but truth is True
        preds = [("b", 0.2, True)]
        negs = find_negatives(preds)
        assert len(negs) == 1
        assert negs[0].item_id == "b"
        assert negs[0].ground_truth is True
        assert negs[0].error_magnitude == pytest.approx(0.8)

    def test_no_negatives_when_all_correct(self):
        preds = [("a", 0.9, True), ("b", 0.1, False)]
        negs = find_negatives(preds)
        assert negs == []

    def test_custom_threshold(self):
        # score=0.6, threshold=0.7 -> predicts False, truth=True -> negative
        preds = [("a", 0.6, True)]
        negs = find_negatives(preds, threshold=0.7)
        assert len(negs) == 1

    def test_boundary_at_threshold(self):
        # score exactly at threshold counts as positive
        preds = [("a", 0.5, True)]
        negs = find_negatives(preds, threshold=0.5)
        assert negs == []

    def test_empty_predictions(self):
        assert find_negatives([]) == []

    def test_mixed_predictions(self):
        preds = [
            ("a", 0.9, True),   # correct
            ("b", 0.8, False),  # false positive
            ("c", 0.1, False),  # correct
            ("d", 0.2, True),   # false negative
        ]
        negs = find_negatives(preds)
        assert len(negs) == 2
        ids = {n.item_id for n in negs}
        assert ids == {"b", "d"}

    def test_all_failures(self):
        preds = [("a", 0.9, False), ("b", 0.1, True)]
        negs = find_negatives(preds)
        assert len(negs) == 2


# ---------------------------------------------------------------------------
# rank_by_difficulty
# ---------------------------------------------------------------------------


class TestRankByDifficulty:
    def test_sorts_descending(self):
        negs = [
            NegativeExample("a", 0.6, False, 0.6),
            NegativeExample("b", 0.9, False, 0.9),
            NegativeExample("c", 0.3, True, 0.7),
        ]
        ranked = rank_by_difficulty(negs)
        assert ranked[0].item_id == "b"
        assert ranked[1].item_id == "c"
        assert ranked[2].item_id == "a"

    def test_empty_list(self):
        assert rank_by_difficulty([]) == []

    def test_single_item(self):
        negs = [NegativeExample("x", 0.5, True, 0.5)]
        ranked = rank_by_difficulty(negs)
        assert len(ranked) == 1


# ---------------------------------------------------------------------------
# mine_hard_negatives
# ---------------------------------------------------------------------------


class TestMineHardNegatives:
    def test_top_k_limit(self):
        preds = [(f"item{i}", 0.9, False) for i in range(20)]
        result = mine_hard_negatives(preds, top_k=5)
        assert len(result.hardest_failures) == 5
        assert result.total_failures == 20

    def test_fewer_than_top_k(self):
        preds = [("a", 0.9, False), ("b", 0.1, False)]
        result = mine_hard_negatives(preds, top_k=10)
        assert len(result.hardest_failures) == 1  # only 1 failure

    def test_failure_rate(self):
        preds = [
            ("a", 0.9, True),    # correct
            ("b", 0.8, False),   # failure
            ("c", 0.1, False),   # correct
            ("d", 0.2, True),    # failure
        ]
        result = mine_hard_negatives(preds)
        assert result.failure_rate == pytest.approx(0.5)

    def test_no_failures(self):
        preds = [("a", 0.9, True), ("b", 0.1, False)]
        result = mine_hard_negatives(preds)
        assert result.total_failures == 0
        assert result.failure_rate == 0.0
        assert result.hardest_failures == []

    def test_empty_predictions(self):
        result = mine_hard_negatives([])
        assert result.total_failures == 0
        assert result.failure_rate == 0.0

    def test_hardest_first_in_result(self):
        preds = [
            ("a", 0.6, False),  # error_mag = 0.6
            ("b", 0.9, False),  # error_mag = 0.9
            ("c", 0.7, False),  # error_mag = 0.7
        ]
        result = mine_hard_negatives(preds, top_k=2)
        assert result.hardest_failures[0].item_id == "b"
        assert result.hardest_failures[1].item_id == "c"


# ---------------------------------------------------------------------------
# analyze_failure_patterns
# ---------------------------------------------------------------------------


class TestAnalyzeFailurePatterns:
    def test_empty_negatives(self):
        assert analyze_failure_patterns([]) == []

    def test_more_false_positives(self):
        negs = [
            NegativeExample("a", 0.9, False, 0.9),
            NegativeExample("b", 0.8, False, 0.8),
            NegativeExample("c", 0.2, True, 0.8),
        ]
        patterns = analyze_failure_patterns(negs)
        assert any("false positives (2)" in p for p in patterns)

    def test_more_false_negatives(self):
        negs = [
            NegativeExample("a", 0.2, True, 0.8),
            NegativeExample("b", 0.1, True, 0.9),
            NegativeExample("c", 0.9, False, 0.9),
        ]
        patterns = analyze_failure_patterns(negs)
        assert any("false negatives (2)" in p for p in patterns)

    def test_equal_fp_fn(self):
        negs = [
            NegativeExample("a", 0.9, False, 0.9),
            NegativeExample("b", 0.2, True, 0.8),
        ]
        patterns = analyze_failure_patterns(negs)
        assert any("Equal" in p for p in patterns)

    def test_common_words_in_texts(self):
        negs = [
            NegativeExample("a", 0.9, False, 0.9),
            NegativeExample("b", 0.8, False, 0.8),
            NegativeExample("c", 0.7, False, 0.7),
        ]
        texts = {
            "a": "the model produced wrong answer",
            "b": "the model gave wrong output",
            "c": "the model was wrong again",
        }
        patterns = analyze_failure_patterns(negs, texts)
        # "the", "model", "wrong" appear in all 3 texts
        combined = " ".join(patterns)
        assert "wrong" in combined or "model" in combined

    def test_no_texts_provided(self):
        negs = [NegativeExample("a", 0.9, False, 0.9)]
        patterns = analyze_failure_patterns(negs)
        # Should still have FP/FN pattern but no word analysis
        assert len(patterns) >= 1

    def test_texts_for_nonexistent_ids(self):
        negs = [NegativeExample("a", 0.9, False, 0.9)]
        texts = {"z": "unrelated text"}
        patterns = analyze_failure_patterns(negs, texts)
        # No crash, just FP/FN pattern
        assert len(patterns) >= 1
