"""Tests for active learning sample selection."""

from __future__ import annotations

import sys
from pathlib import Path

# Add SDK to path
SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.calibration.active_learning import (
    ActiveLearningConfig,
    ScoredItem,
    SelectionResult,
    select_by_disagreement,
    select_by_diversity,
    select_by_uncertainty,
    select_for_annotation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_items(n: int = 10, with_heuristic: bool = False) -> list[ScoredItem]:
    items = []
    for i in range(n):
        h = (1.0 - i / max(n - 1, 1)) if with_heuristic else None
        items.append(
            ScoredItem(
                item_id=f"item_{i}",
                score=i / max(n - 1, 1),
                confidence=i / max(n - 1, 1),
                heuristic_score=h,
            )
        )
    return items


# ---------------------------------------------------------------------------
# ScoredItem
# ---------------------------------------------------------------------------


class TestScoredItem:
    def test_as_dict_basic(self):
        item = ScoredItem(item_id="a", score=0.5, confidence=0.8)
        d = item.as_dict()
        assert d["item_id"] == "a"
        assert d["score"] == 0.5
        assert d["confidence"] == 0.8
        assert "heuristic_score" not in d

    def test_as_dict_with_heuristic(self):
        item = ScoredItem(item_id="b", score=0.5, confidence=0.8, heuristic_score=0.3)
        d = item.as_dict()
        assert d["heuristic_score"] == 0.3


# ---------------------------------------------------------------------------
# SelectionResult
# ---------------------------------------------------------------------------


class TestSelectionResult:
    def test_as_dict(self):
        r = SelectionResult(
            selected_ids=["a", "b"],
            method="uncertainty",
            total_candidates=10,
            selected_count=2,
        )
        d = r.as_dict()
        assert d["selected_ids"] == ["a", "b"]
        assert d["method"] == "uncertainty"
        assert d["total_candidates"] == 10
        assert d["selected_count"] == 2

    def test_format_text_contains_method(self):
        r = SelectionResult(
            selected_ids=["x"],
            method="diversity",
            total_candidates=5,
            selected_count=1,
        )
        text = r.format_text()
        assert "diversity" in text
        assert "x" in text
        assert "5" in text

    def test_format_text_empty(self):
        r = SelectionResult(
            selected_ids=[],
            method="uncertainty",
            total_candidates=0,
            selected_count=0,
        )
        text = r.format_text()
        assert "uncertainty" in text


# ---------------------------------------------------------------------------
# ActiveLearningConfig
# ---------------------------------------------------------------------------


class TestActiveLearningConfig:
    def test_defaults(self):
        cfg = ActiveLearningConfig()
        assert cfg.method == "uncertainty"
        assert cfg.batch_size == 10
        assert cfg.diversity_weight == 0.3

    def test_as_dict(self):
        cfg = ActiveLearningConfig(method="diversity", batch_size=5, diversity_weight=0.5)
        d = cfg.as_dict()
        assert d["method"] == "diversity"
        assert d["batch_size"] == 5
        assert d["diversity_weight"] == 0.5

    def test_from_dict(self):
        d = {"method": "disagreement", "batch_size": 20, "diversity_weight": 0.1}
        cfg = ActiveLearningConfig.from_dict(d)
        assert cfg.method == "disagreement"
        assert cfg.batch_size == 20
        assert cfg.diversity_weight == 0.1

    def test_from_dict_defaults(self):
        cfg = ActiveLearningConfig.from_dict({})
        assert cfg.method == "uncertainty"
        assert cfg.batch_size == 10
        assert cfg.diversity_weight == 0.3

    def test_roundtrip(self):
        cfg = ActiveLearningConfig(method="combined", batch_size=7, diversity_weight=0.6)
        cfg2 = ActiveLearningConfig.from_dict(cfg.as_dict())
        assert cfg2.method == cfg.method
        assert cfg2.batch_size == cfg.batch_size
        assert cfg2.diversity_weight == cfg.diversity_weight


# ---------------------------------------------------------------------------
# select_by_uncertainty
# ---------------------------------------------------------------------------


class TestSelectByUncertainty:
    def test_picks_lowest_confidence(self):
        items = _make_items(10)
        result = select_by_uncertainty(items, batch_size=3)
        # Items are sorted by confidence ascending; item_0 has conf=0.0
        assert result.selected_ids[0] == "item_0"
        assert result.selected_count == 3
        assert result.method == "uncertainty"

    def test_batch_size_respected(self):
        items = _make_items(20)
        result = select_by_uncertainty(items, batch_size=5)
        assert len(result.selected_ids) == 5
        assert result.selected_count == 5
        assert result.total_candidates == 20

    def test_batch_larger_than_items(self):
        items = _make_items(3)
        result = select_by_uncertainty(items, batch_size=10)
        assert len(result.selected_ids) == 3
        assert result.selected_count == 3

    def test_empty_items(self):
        result = select_by_uncertainty([], batch_size=5)
        assert result.selected_ids == []
        assert result.selected_count == 0
        assert result.total_candidates == 0

    def test_single_item(self):
        items = [ScoredItem(item_id="only", score=0.5, confidence=0.9)]
        result = select_by_uncertainty(items, batch_size=5)
        assert result.selected_ids == ["only"]


# ---------------------------------------------------------------------------
# select_by_disagreement
# ---------------------------------------------------------------------------


class TestSelectByDisagreement:
    def test_picks_highest_disagreement(self):
        items = _make_items(10, with_heuristic=True)
        result = select_by_disagreement(items, batch_size=3)
        # item_0: score=0.0, heuristic=1.0 -> disagreement=1.0
        # item_9: score=1.0, heuristic=0.0 -> disagreement=1.0
        assert result.selected_count == 3
        assert result.method == "disagreement"
        # Both extremes should be in the top picks
        assert "item_0" in result.selected_ids or "item_9" in result.selected_ids

    def test_skips_items_without_heuristic(self):
        items = [
            ScoredItem(item_id="no_h", score=0.5, confidence=0.5),
            ScoredItem(item_id="has_h", score=0.5, confidence=0.5, heuristic_score=0.1),
        ]
        result = select_by_disagreement(items, batch_size=5)
        assert "has_h" in result.selected_ids
        assert "no_h" not in result.selected_ids

    def test_all_without_heuristic(self):
        items = [
            ScoredItem(item_id="a", score=0.5, confidence=0.5),
            ScoredItem(item_id="b", score=0.3, confidence=0.3),
        ]
        result = select_by_disagreement(items, batch_size=5)
        assert result.selected_ids == []
        assert result.selected_count == 0

    def test_empty_items(self):
        result = select_by_disagreement([], batch_size=5)
        assert result.selected_ids == []
        assert result.total_candidates == 0


# ---------------------------------------------------------------------------
# select_by_diversity
# ---------------------------------------------------------------------------


class TestSelectByDiversity:
    def test_spreads_across_range(self):
        items = _make_items(20)
        result = select_by_diversity(items, batch_size=5)
        assert result.selected_count == 5
        assert result.method == "diversity"
        # Selected items should span different parts of the score range
        scores = []
        id_to_score = {item.item_id: item.score for item in items}
        for sid in result.selected_ids:
            scores.append(id_to_score[sid])
        # Check that min and max are separated
        assert max(scores) - min(scores) > 0.5

    def test_batch_larger_than_items(self):
        items = _make_items(3)
        result = select_by_diversity(items, batch_size=10)
        assert result.selected_count == 3

    def test_empty_items(self):
        result = select_by_diversity([], batch_size=5)
        assert result.selected_ids == []
        assert result.total_candidates == 0

    def test_single_item(self):
        items = [ScoredItem(item_id="solo", score=0.5, confidence=0.5)]
        result = select_by_diversity(items, batch_size=3)
        assert result.selected_ids == ["solo"]
        assert result.selected_count == 1


# ---------------------------------------------------------------------------
# select_for_annotation
# ---------------------------------------------------------------------------


class TestSelectForAnnotation:
    def test_routes_uncertainty(self):
        items = _make_items(10)
        cfg = ActiveLearningConfig(method="uncertainty", batch_size=3)
        result = select_for_annotation(items, cfg)
        assert result.method == "uncertainty"
        assert result.selected_count == 3

    def test_routes_disagreement(self):
        items = _make_items(10, with_heuristic=True)
        cfg = ActiveLearningConfig(method="disagreement", batch_size=3)
        result = select_for_annotation(items, cfg)
        assert result.method == "disagreement"

    def test_routes_diversity(self):
        items = _make_items(10)
        cfg = ActiveLearningConfig(method="diversity", batch_size=3)
        result = select_for_annotation(items, cfg)
        assert result.method == "diversity"

    def test_routes_combined(self):
        items = _make_items(10)
        cfg = ActiveLearningConfig(method="combined", batch_size=5, diversity_weight=0.4)
        result = select_for_annotation(items, cfg)
        assert result.method == "combined"
        assert result.selected_count <= 5

    def test_combined_includes_diverse_and_uncertain(self):
        items = _make_items(20)
        cfg = ActiveLearningConfig(method="combined", batch_size=10, diversity_weight=0.3)
        result = select_for_annotation(items, cfg)
        assert result.selected_count <= 10
        # Should have some items selected
        assert result.selected_count > 0

    def test_unknown_method_raises(self):
        items = _make_items(5)
        cfg = ActiveLearningConfig(method="bogus", batch_size=3)
        try:
            select_for_annotation(items, cfg)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "bogus" in str(e)
