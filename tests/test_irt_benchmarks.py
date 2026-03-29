"""Tests for IRT-based tiny benchmarks module."""

from __future__ import annotations

import math

import pytest

from evalyn_sdk.irt_benchmarks import (
    IRTConfig,
    IRTItem,
    IRTResult,
    compute_item_information,
    compute_total_information,
    estimate_all_parameters,
    estimate_item_parameters,
    format_irt_report,
    select_informative_items,
)


# ---------------------------------------------------------------------------
# IRTItem dataclass
# ---------------------------------------------------------------------------


class TestIRTItem:
    def test_basic_creation(self):
        item = IRTItem(item_id="q1", difficulty=0.5, discrimination=1.2)
        assert item.item_id == "q1"
        assert item.difficulty == 0.5
        assert item.discrimination == 1.2
        assert item.guessing == 0.0

    def test_guessing_default(self):
        item = IRTItem(item_id="q1", difficulty=0.0, discrimination=1.0)
        assert item.guessing == 0.0

    def test_guessing_custom(self):
        item = IRTItem(item_id="q1", difficulty=0.0, discrimination=1.0, guessing=0.25)
        assert item.guessing == 0.25

    def test_as_dict(self):
        item = IRTItem(item_id="q1", difficulty=1.5, discrimination=0.8, guessing=0.1)
        d = item.as_dict()
        assert d["item_id"] == "q1"
        assert d["difficulty"] == 1.5
        assert d["discrimination"] == 0.8
        assert d["guessing"] == 0.1

    def test_from_dict(self):
        data = {"item_id": "q2", "difficulty": -0.5, "discrimination": 2.0, "guessing": 0.2}
        item = IRTItem.from_dict(data)
        assert item.item_id == "q2"
        assert item.difficulty == -0.5
        assert item.discrimination == 2.0
        assert item.guessing == 0.2

    def test_from_dict_default_guessing(self):
        data = {"item_id": "q3", "difficulty": 0.0, "discrimination": 1.0}
        item = IRTItem.from_dict(data)
        assert item.guessing == 0.0

    def test_roundtrip(self):
        original = IRTItem(item_id="q1", difficulty=2.3, discrimination=1.5, guessing=0.15)
        restored = IRTItem.from_dict(original.as_dict())
        assert restored.item_id == original.item_id
        assert restored.difficulty == original.difficulty
        assert restored.discrimination == original.discrimination
        assert restored.guessing == original.guessing


# ---------------------------------------------------------------------------
# IRTConfig dataclass
# ---------------------------------------------------------------------------


class TestIRTConfig:
    def test_defaults(self):
        cfg = IRTConfig()
        assert cfg.target_size == 100
        assert cfg.ability_range == (-3.0, 3.0)
        assert cfg.n_ability_points == 21

    def test_custom(self):
        cfg = IRTConfig(target_size=50, ability_range=(-2.0, 2.0), n_ability_points=11)
        assert cfg.target_size == 50
        assert cfg.ability_range == (-2.0, 2.0)

    def test_as_dict(self):
        cfg = IRTConfig(target_size=30)
        d = cfg.as_dict()
        assert d["target_size"] == 30
        assert d["ability_range"] == [-3.0, 3.0]

    def test_from_dict(self):
        data = {"target_size": 20, "ability_range": [-1.0, 1.0], "n_ability_points": 5}
        cfg = IRTConfig.from_dict(data)
        assert cfg.target_size == 20
        assert cfg.ability_range == (-1.0, 1.0)
        assert cfg.n_ability_points == 5

    def test_from_dict_defaults(self):
        cfg = IRTConfig.from_dict({})
        assert cfg.target_size == 100
        assert cfg.n_ability_points == 21

    def test_roundtrip(self):
        original = IRTConfig(target_size=42, ability_range=(-2.5, 2.5), n_ability_points=15)
        restored = IRTConfig.from_dict(original.as_dict())
        assert restored.target_size == original.target_size
        assert restored.ability_range == tuple(original.ability_range)
        assert restored.n_ability_points == original.n_ability_points


# ---------------------------------------------------------------------------
# IRTResult dataclass
# ---------------------------------------------------------------------------


class TestIRTResult:
    def test_empty(self):
        r = IRTResult()
        assert r.selected_items == []
        assert r.total_pool == 0
        assert r.compression_ratio == 0.0
        assert r.information_retained == 0.0

    def test_as_dict(self):
        item = IRTItem(item_id="q1", difficulty=0.0, discrimination=1.0)
        r = IRTResult(selected_items=[item], total_pool=10, compression_ratio=0.1, information_retained=0.5)
        d = r.as_dict()
        assert len(d["selected_items"]) == 1
        assert d["total_pool"] == 10

    def test_from_dict(self):
        data = {
            "selected_items": [{"item_id": "q1", "difficulty": 0.0, "discrimination": 1.0}],
            "total_pool": 5,
            "compression_ratio": 0.2,
            "information_retained": 0.8,
        }
        r = IRTResult.from_dict(data)
        assert len(r.selected_items) == 1
        assert r.selected_items[0].item_id == "q1"
        assert r.total_pool == 5

    def test_roundtrip(self):
        items = [IRTItem(item_id=f"q{i}", difficulty=float(i), discrimination=1.0) for i in range(3)]
        original = IRTResult(selected_items=items, total_pool=10, compression_ratio=0.3, information_retained=0.9)
        restored = IRTResult.from_dict(original.as_dict())
        assert len(restored.selected_items) == 3
        assert restored.compression_ratio == 0.3


# ---------------------------------------------------------------------------
# Parameter Estimation
# ---------------------------------------------------------------------------


class TestEstimateItemParameters:
    def test_empty_scores(self):
        item = estimate_item_parameters("q1", [])
        assert item.item_id == "q1"
        assert item.difficulty == 0.0
        assert item.discrimination == 1.0

    def test_easy_item(self):
        # High mean score -> low (negative) difficulty
        scores = [0.9, 0.95, 0.85, 0.9, 0.92]
        item = estimate_item_parameters("easy", scores)
        assert item.difficulty < 0.0  # Easy items have negative difficulty

    def test_hard_item(self):
        # Low mean score -> high (positive) difficulty
        scores = [0.1, 0.05, 0.15, 0.08, 0.12]
        item = estimate_item_parameters("hard", scores)
        assert item.difficulty > 0.0  # Hard items have positive difficulty

    def test_medium_item(self):
        # 50% mean -> difficulty near 0
        scores = [0.5, 0.5, 0.5, 0.5]
        item = estimate_item_parameters("mid", scores)
        assert abs(item.difficulty) < 0.1

    def test_high_discrimination(self):
        # Very consistent scores -> high discrimination
        scores = [0.5, 0.5, 0.5, 0.5, 0.5]
        item = estimate_item_parameters("consistent", scores)
        assert item.discrimination >= 2.0

    def test_low_discrimination(self):
        # Highly variable scores -> low discrimination
        scores = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        item = estimate_item_parameters("variable", scores)
        assert item.discrimination < 1.5

    def test_single_score(self):
        item = estimate_item_parameters("single", [0.7])
        assert item.item_id == "single"
        assert item.discrimination == 1.0  # Default for n < 2

    def test_item_id_preserved(self):
        item = estimate_item_parameters("my-item-42", [0.5])
        assert item.item_id == "my-item-42"


class TestEstimateAllParameters:
    def test_empty(self):
        result = estimate_all_parameters({})
        assert result == []

    def test_multiple_items(self):
        item_scores = {
            "q1": [0.9, 0.8, 0.85],
            "q2": [0.2, 0.1, 0.15],
            "q3": [0.5, 0.5, 0.5],
        }
        items = estimate_all_parameters(item_scores)
        assert len(items) == 3
        ids = {item.item_id for item in items}
        assert ids == {"q1", "q2", "q3"}

    def test_preserves_order(self):
        item_scores = {"c": [0.5], "a": [0.5], "b": [0.5]}
        items = estimate_all_parameters(item_scores)
        assert [it.item_id for it in items] == list(item_scores.keys())


# ---------------------------------------------------------------------------
# Information Functions
# ---------------------------------------------------------------------------


class TestComputeItemInformation:
    def test_at_difficulty_peak(self):
        # Information is maximized when ability == difficulty (for D=1)
        item = IRTItem(item_id="q1", difficulty=0.0, discrimination=1.0)
        info_at_zero = compute_item_information(item, 0.0)
        info_away = compute_item_information(item, 3.0)
        assert info_at_zero > info_away

    def test_non_negative(self):
        item = IRTItem(item_id="q1", difficulty=1.0, discrimination=2.0)
        for theta in [-3.0, -1.0, 0.0, 1.0, 3.0]:
            assert compute_item_information(item, theta) >= 0.0

    def test_higher_discrimination_more_info(self):
        low_d = IRTItem(item_id="low", difficulty=0.0, discrimination=0.5)
        high_d = IRTItem(item_id="high", difficulty=0.0, discrimination=2.0)
        # At theta=0 (matching difficulty), higher disc -> more info
        assert compute_item_information(high_d, 0.0) > compute_item_information(low_d, 0.0)

    def test_symmetric_around_difficulty(self):
        item = IRTItem(item_id="q1", difficulty=1.0, discrimination=1.5)
        info_above = compute_item_information(item, 2.0)
        info_below = compute_item_information(item, 0.0)
        # Due to symmetry of logistic, should be equal
        assert abs(info_above - info_below) < 1e-10

    def test_zero_discrimination(self):
        item = IRTItem(item_id="q1", difficulty=0.0, discrimination=0.0)
        assert compute_item_information(item, 0.0) == 0.0

    def test_known_value(self):
        # At ability = difficulty, P = 0.5, so I = D^2 * 0.25
        item = IRTItem(item_id="q1", difficulty=0.0, discrimination=2.0)
        expected = 4.0 * 0.25  # D^2 * 0.25 = 1.0
        assert abs(compute_item_information(item, 0.0) - expected) < 1e-10


class TestComputeTotalInformation:
    def test_empty_items(self):
        assert compute_total_information([], [0.0]) == 0.0

    def test_empty_ability_points(self):
        item = IRTItem(item_id="q1", difficulty=0.0, discrimination=1.0)
        assert compute_total_information([item], []) == 0.0

    def test_single_item_single_point(self):
        item = IRTItem(item_id="q1", difficulty=0.0, discrimination=1.0)
        total = compute_total_information([item], [0.0])
        single = compute_item_information(item, 0.0)
        assert abs(total - single) < 1e-10

    def test_additive(self):
        items = [
            IRTItem(item_id="q1", difficulty=0.0, discrimination=1.0),
            IRTItem(item_id="q2", difficulty=1.0, discrimination=1.5),
        ]
        points = [0.0, 1.0]
        total = compute_total_information(items, points)
        manual = sum(
            compute_item_information(item, theta)
            for item in items
            for theta in points
        )
        assert abs(total - manual) < 1e-10

    def test_more_items_more_info(self):
        items_few = [IRTItem(item_id="q1", difficulty=0.0, discrimination=1.0)]
        items_many = items_few + [IRTItem(item_id="q2", difficulty=1.0, discrimination=1.0)]
        points = [-1.0, 0.0, 1.0]
        assert compute_total_information(items_many, points) > compute_total_information(items_few, points)


# ---------------------------------------------------------------------------
# Greedy Selection
# ---------------------------------------------------------------------------


class TestSelectInformativeItems:
    def test_empty_pool(self):
        result = select_informative_items([])
        assert result.total_pool == 0
        assert result.selected_items == []
        assert result.compression_ratio == 0.0

    def test_pool_smaller_than_target(self):
        items = [IRTItem(item_id=f"q{i}", difficulty=float(i), discrimination=1.0) for i in range(5)]
        config = IRTConfig(target_size=100)
        result = select_informative_items(items, config)
        assert len(result.selected_items) == 5
        assert result.total_pool == 5

    def test_selects_target_size(self):
        items = [IRTItem(item_id=f"q{i}", difficulty=float(i) - 5, discrimination=1.0) for i in range(20)]
        config = IRTConfig(target_size=10)
        result = select_informative_items(items, config)
        assert len(result.selected_items) == 10
        assert result.total_pool == 20

    def test_compression_ratio(self):
        items = [IRTItem(item_id=f"q{i}", difficulty=0.0, discrimination=1.0) for i in range(10)]
        config = IRTConfig(target_size=5)
        result = select_informative_items(items, config)
        assert abs(result.compression_ratio - 0.5) < 1e-10

    def test_information_retained_positive(self):
        items = [
            IRTItem(item_id=f"q{i}", difficulty=float(i) - 2, discrimination=1.5)
            for i in range(10)
        ]
        config = IRTConfig(target_size=5)
        result = select_informative_items(items, config)
        assert 0.0 < result.information_retained <= 1.0

    def test_full_selection_retains_all_info(self):
        items = [IRTItem(item_id=f"q{i}", difficulty=float(i), discrimination=1.0) for i in range(5)]
        config = IRTConfig(target_size=5)
        result = select_informative_items(items, config)
        assert abs(result.information_retained - 1.0) < 1e-10

    def test_prefers_high_discrimination(self):
        items = [
            IRTItem(item_id="low_d", difficulty=0.0, discrimination=0.5),
            IRTItem(item_id="high_d", difficulty=0.0, discrimination=3.0),
        ]
        config = IRTConfig(target_size=1)
        result = select_informative_items(items, config)
        assert result.selected_items[0].item_id == "high_d"

    def test_default_config(self):
        items = [IRTItem(item_id=f"q{i}", difficulty=0.0, discrimination=1.0) for i in range(3)]
        result = select_informative_items(items)
        assert len(result.selected_items) == 3  # Pool < default target_size

    def test_no_duplicates(self):
        items = [IRTItem(item_id=f"q{i}", difficulty=float(i), discrimination=1.0) for i in range(10)]
        config = IRTConfig(target_size=5)
        result = select_informative_items(items, config)
        ids = [item.item_id for item in result.selected_items]
        assert len(ids) == len(set(ids))

    def test_result_as_dict_roundtrip(self):
        items = [IRTItem(item_id=f"q{i}", difficulty=float(i), discrimination=1.0) for i in range(5)]
        config = IRTConfig(target_size=3)
        result = select_informative_items(items, config)
        restored = IRTResult.from_dict(result.as_dict())
        assert len(restored.selected_items) == len(result.selected_items)
        assert restored.total_pool == result.total_pool


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


class TestFormatIRTReport:
    def test_empty_result(self):
        result = IRTResult()
        report = format_irt_report(result)
        assert "IRT Benchmark Selection Report" in report
        assert "Total pool size: 0" in report

    def test_with_items(self):
        items = [
            IRTItem(item_id="q1", difficulty=1.5, discrimination=2.0),
            IRTItem(item_id="q2", difficulty=-0.5, discrimination=1.0),
        ]
        result = IRTResult(
            selected_items=items,
            total_pool=10,
            compression_ratio=0.2,
            information_retained=0.75,
        )
        report = format_irt_report(result)
        assert "q1" in report
        assert "q2" in report
        assert "Selected items: 2" in report
        assert "Total pool size: 10" in report
        assert "20.00%" in report  # compression ratio
        assert "75.00%" in report  # information retained

    def test_report_is_string(self):
        result = IRTResult(total_pool=5)
        assert isinstance(format_irt_report(result), str)

    def test_contains_header(self):
        result = IRTResult()
        report = format_irt_report(result)
        assert "=" in report  # separator line
