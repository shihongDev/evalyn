"""Tests for coreset_sampling module."""

from __future__ import annotations

import pytest

from evalyn_sdk.coreset_sampling import (
    CoresetConfig,
    CoresetItem,
    CoresetResult,
    compute_approximation_error,
    compute_item_features,
    format_coreset_report,
    greedy_coreset,
    run_coreset_sampling,
    validate_coreset,
)


# ---------------------------------------------------------------------------
# CoresetConfig
# ---------------------------------------------------------------------------


class TestCoresetConfig:
    def test_defaults(self):
        cfg = CoresetConfig()
        assert cfg.target_size == 50
        assert cfg.max_error == 0.1
        assert cfg.seed is None

    def test_custom(self):
        cfg = CoresetConfig(target_size=10, max_error=0.05, seed=7)
        assert cfg.target_size == 10
        assert cfg.max_error == 0.05
        assert cfg.seed == 7

    def test_as_dict(self):
        cfg = CoresetConfig(target_size=20, max_error=0.2, seed=3)
        d = cfg.as_dict()
        assert d == {"target_size": 20, "max_error": 0.2, "seed": 3}

    def test_from_dict(self):
        cfg = CoresetConfig.from_dict(
            {"target_size": 30, "max_error": 0.15, "seed": 5}
        )
        assert cfg.target_size == 30
        assert cfg.max_error == 0.15
        assert cfg.seed == 5

    def test_from_dict_defaults(self):
        cfg = CoresetConfig.from_dict({})
        assert cfg.target_size == 50
        assert cfg.max_error == 0.1
        assert cfg.seed is None

    def test_roundtrip(self):
        cfg = CoresetConfig(target_size=15, max_error=0.08, seed=99)
        assert CoresetConfig.from_dict(cfg.as_dict()) == cfg


# ---------------------------------------------------------------------------
# CoresetItem
# ---------------------------------------------------------------------------


class TestCoresetItem:
    def test_creation(self):
        ci = CoresetItem(item_id="a", weight=0.5, representative_of=10)
        assert ci.item_id == "a"
        assert ci.weight == 0.5
        assert ci.representative_of == 10

    def test_as_dict(self):
        ci = CoresetItem(item_id="b", weight=0.3, representative_of=5)
        d = ci.as_dict()
        assert d == {"item_id": "b", "weight": 0.3, "representative_of": 5}

    def test_from_dict(self):
        ci = CoresetItem.from_dict(
            {"item_id": "c", "weight": 0.7, "representative_of": 3}
        )
        assert ci.item_id == "c"
        assert ci.weight == 0.7

    def test_roundtrip(self):
        ci = CoresetItem(item_id="x", weight=0.25, representative_of=8)
        assert CoresetItem.from_dict(ci.as_dict()) == ci


# ---------------------------------------------------------------------------
# CoresetResult
# ---------------------------------------------------------------------------


class TestCoresetResult:
    def test_defaults(self):
        r = CoresetResult()
        assert r.selected == []
        assert r.total_pool == 0
        assert r.compression_ratio == 0.0
        assert r.max_approximation_error == 0.0

    def test_as_dict(self):
        item = CoresetItem(item_id="a", weight=0.5, representative_of=2)
        r = CoresetResult(
            selected=[item],
            total_pool=10,
            compression_ratio=0.1,
            max_approximation_error=0.05,
        )
        d = r.as_dict()
        assert len(d["selected"]) == 1
        assert d["total_pool"] == 10
        assert d["compression_ratio"] == 0.1

    def test_from_dict(self):
        data = {
            "selected": [{"item_id": "z", "weight": 0.4, "representative_of": 6}],
            "total_pool": 20,
            "compression_ratio": 0.05,
            "max_approximation_error": 0.03,
        }
        r = CoresetResult.from_dict(data)
        assert len(r.selected) == 1
        assert r.selected[0].item_id == "z"
        assert r.total_pool == 20

    def test_from_dict_defaults(self):
        r = CoresetResult.from_dict({})
        assert r.selected == []
        assert r.total_pool == 0

    def test_roundtrip(self):
        item = CoresetItem(item_id="q", weight=0.1, representative_of=4)
        r = CoresetResult(
            selected=[item],
            total_pool=40,
            compression_ratio=0.025,
            max_approximation_error=0.08,
        )
        r2 = CoresetResult.from_dict(r.as_dict())
        assert r2.total_pool == r.total_pool
        assert len(r2.selected) == len(r.selected)
        assert r2.selected[0].item_id == r.selected[0].item_id


# ---------------------------------------------------------------------------
# compute_item_features
# ---------------------------------------------------------------------------


class TestComputeItemFeatures:
    def test_basic(self):
        features = compute_item_features("Hello World")
        assert features == {"hello", "world"}

    def test_short_words_filtered(self):
        features = compute_item_features("I am a big cat")
        assert "i" not in features
        assert "a" not in features
        assert "am" in features
        assert "big" in features
        assert "cat" in features

    def test_empty_string(self):
        assert compute_item_features("") == set()

    def test_all_short_words(self):
        assert compute_item_features("I a") == set()

    def test_lowercased(self):
        features = compute_item_features("ABC DEF")
        assert "abc" in features
        assert "ABC" not in features

    def test_duplicates(self):
        features = compute_item_features("the the the big big")
        assert features == {"the", "big"}

    def test_punctuation_attached(self):
        # Punctuation stays attached to words (split by whitespace only)
        features = compute_item_features("hello, world!")
        assert "hello," in features
        assert "world!" in features


# ---------------------------------------------------------------------------
# greedy_coreset
# ---------------------------------------------------------------------------


class TestGreedyCoreset:
    def _make_items(self, n: int) -> dict[str, str]:
        topics = ["python programming", "java enterprise", "rust systems",
                  "data science ml", "web frontend react"]
        items = {}
        for i in range(n):
            topic = topics[i % len(topics)]
            items[f"item_{i}"] = f"{topic} example number {i}"
        return items

    def test_empty_items(self):
        result = greedy_coreset({}, target_size=5)
        assert result.selected == []
        assert result.total_pool == 0

    def test_target_larger_than_pool(self):
        items = {"a": "hello world", "b": "foo bar"}
        result = greedy_coreset(items, target_size=10)
        assert len(result.selected) == 2
        assert result.total_pool == 2

    def test_single_item(self):
        result = greedy_coreset({"only": "just one item"}, target_size=5)
        assert len(result.selected) == 1
        assert result.selected[0].item_id == "only"

    def test_correct_count(self):
        items = self._make_items(20)
        result = greedy_coreset(items, target_size=5, seed=42)
        assert len(result.selected) == 5

    def test_total_pool(self):
        items = self._make_items(30)
        result = greedy_coreset(items, target_size=10)
        assert result.total_pool == 30

    def test_compression_ratio(self):
        items = self._make_items(20)
        result = greedy_coreset(items, target_size=5)
        assert result.compression_ratio == pytest.approx(0.25)

    def test_weights_sum_to_one(self):
        items = self._make_items(50)
        result = greedy_coreset(items, target_size=10, seed=7)
        total_weight = sum(s.weight for s in result.selected)
        assert total_weight == pytest.approx(1.0)

    def test_representative_of_sums_to_total(self):
        items = self._make_items(40)
        result = greedy_coreset(items, target_size=8, seed=0)
        total_rep = sum(s.representative_of for s in result.selected)
        assert total_rep == 40

    def test_deterministic_same_seed(self):
        items = self._make_items(30)
        r1 = greedy_coreset(items, target_size=5, seed=42)
        r2 = greedy_coreset(items, target_size=5, seed=42)
        ids1 = [s.item_id for s in r1.selected]
        ids2 = [s.item_id for s in r2.selected]
        assert ids1 == ids2

    def test_different_seeds_may_differ(self):
        items = self._make_items(50)
        r1 = greedy_coreset(items, target_size=5, seed=1)
        r2 = greedy_coreset(items, target_size=5, seed=999)
        ids1 = set(s.item_id for s in r1.selected)
        ids2 = set(s.item_id for s in r2.selected)
        # With 50 items and target 5, different seeds may pick
        # different items (not guaranteed but very likely with ties)
        # At minimum, both should be valid subsets
        all_ids = set(items.keys())
        assert ids1.issubset(all_ids)
        assert ids2.issubset(all_ids)

    def test_selected_ids_are_from_items(self):
        items = self._make_items(25)
        result = greedy_coreset(items, target_size=7)
        for s in result.selected:
            assert s.item_id in items

    def test_no_duplicates_in_selected(self):
        items = self._make_items(30)
        result = greedy_coreset(items, target_size=10, seed=42)
        ids = [s.item_id for s in result.selected]
        assert len(ids) == len(set(ids))

    def test_approximation_error_recorded(self):
        items = self._make_items(20)
        result = greedy_coreset(items, target_size=5)
        assert result.max_approximation_error >= 0.0
        assert result.max_approximation_error <= 1.0

    def test_identical_items(self):
        items = {f"item_{i}": "same text here" for i in range(10)}
        result = greedy_coreset(items, target_size=3)
        assert len(result.selected) == 3
        assert result.max_approximation_error == 0.0

    def test_completely_different_items(self):
        items = {
            "a": "alpha bravo charlie",
            "b": "delta echo foxtrot",
            "c": "golf hotel india",
        }
        result = greedy_coreset(items, target_size=3)
        assert len(result.selected) == 3


# ---------------------------------------------------------------------------
# compute_approximation_error
# ---------------------------------------------------------------------------


class TestApproximationError:
    def test_empty_coreset(self):
        assert compute_approximation_error([], {"a": "text"}) == 0.0

    def test_empty_items(self):
        assert compute_approximation_error(["a"], {}) == 0.0

    def test_all_items_in_coreset(self):
        items = {"a": "hello world", "b": "foo bar"}
        error = compute_approximation_error(["a", "b"], items)
        assert error == 0.0

    def test_identical_texts_zero_error(self):
        items = {f"i{k}": "same words" for k in range(5)}
        error = compute_approximation_error(["i0"], items)
        assert error == 0.0

    def test_disjoint_texts_high_error(self):
        items = {"a": "alpha bravo", "b": "charlie delta"}
        error = compute_approximation_error(["a"], items)
        assert error == 1.0  # disjoint word sets -> distance 1.0

    def test_error_decreases_with_more_coreset(self):
        items = {
            "a": "alpha bravo",
            "b": "charlie delta",
            "c": "echo foxtrot",
        }
        e1 = compute_approximation_error(["a"], items)
        e2 = compute_approximation_error(["a", "b"], items)
        assert e2 <= e1

    def test_error_bounded_zero_one(self):
        items = {f"i{k}": f"word{k} unique{k}" for k in range(10)}
        error = compute_approximation_error(["i0"], items)
        assert 0.0 <= error <= 1.0


# ---------------------------------------------------------------------------
# validate_coreset
# ---------------------------------------------------------------------------


class TestValidateCoreset:
    def test_valid_coreset(self):
        items = {f"item_{i}": "same text" for i in range(10)}
        result = greedy_coreset(items, target_size=3)
        assert validate_coreset(result, items, max_error=0.1) is True

    def test_invalid_coreset_tight_bound(self):
        items = {
            "a": "alpha bravo charlie",
            "b": "delta echo foxtrot",
            "c": "golf hotel india",
        }
        # Coreset of 1 from dissimilar items, very tight bound
        result = greedy_coreset(items, target_size=1)
        # With dissimilar items and only 1 in coreset, error is likely > 0.0001
        is_valid = validate_coreset(result, items, max_error=0.0001)
        assert is_valid is False

    def test_full_coreset_always_valid(self):
        items = {"a": "hello", "b": "world", "c": "foo"}
        result = greedy_coreset(items, target_size=3)
        assert validate_coreset(result, items, max_error=0.0) is True


# ---------------------------------------------------------------------------
# run_coreset_sampling
# ---------------------------------------------------------------------------


class TestRunCoresetSampling:
    def test_basic_pipeline(self):
        items = {f"item_{i}": f"topic {i % 3} content {i}" for i in range(30)}
        config = CoresetConfig(target_size=5, seed=42)
        result = run_coreset_sampling(items, config)
        assert len(result.selected) == 5
        assert result.total_pool == 30

    def test_empty_items(self):
        config = CoresetConfig(target_size=5)
        result = run_coreset_sampling({}, config)
        assert result.selected == []

    def test_config_seed_used(self):
        items = {f"i{k}": f"word {k} data {k}" for k in range(20)}
        c1 = CoresetConfig(target_size=5, seed=42)
        c2 = CoresetConfig(target_size=5, seed=42)
        r1 = run_coreset_sampling(items, c1)
        r2 = run_coreset_sampling(items, c2)
        ids1 = [s.item_id for s in r1.selected]
        ids2 = [s.item_id for s in r2.selected]
        assert ids1 == ids2

    def test_config_target_respected(self):
        items = {f"i{k}": f"text {k}" for k in range(100)}
        config = CoresetConfig(target_size=10)
        result = run_coreset_sampling(items, config)
        assert len(result.selected) == 10


# ---------------------------------------------------------------------------
# format_coreset_report
# ---------------------------------------------------------------------------


class TestFormatCoresetReport:
    def test_basic_format(self):
        item = CoresetItem(item_id="a", weight=0.5, representative_of=10)
        result = CoresetResult(
            selected=[item],
            total_pool=20,
            compression_ratio=0.05,
            max_approximation_error=0.03,
        )
        text = format_coreset_report(result)
        assert "Coreset Report" in text
        assert "Total pool:          20" in text
        assert "Selected:            1" in text
        assert "0.0500" in text  # compression ratio
        assert "0.0300" in text  # approx error
        assert "a:" in text
        assert "weight=0.5000" in text
        assert "represents=10" in text

    def test_empty_report(self):
        result = CoresetResult()
        text = format_coreset_report(result)
        assert "Coreset Report" in text
        assert "Selected:            0" in text

    def test_multiple_items(self):
        items = [
            CoresetItem(item_id=f"i{k}", weight=0.2, representative_of=5)
            for k in range(3)
        ]
        result = CoresetResult(
            selected=items,
            total_pool=15,
            compression_ratio=0.2,
            max_approximation_error=0.05,
        )
        text = format_coreset_report(result)
        assert "i0:" in text
        assert "i1:" in text
        assert "i2:" in text


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_word_items(self):
        items = {"a": "hello", "b": "world", "c": "python"}
        result = greedy_coreset(items, target_size=2)
        assert len(result.selected) == 2

    def test_empty_text_items(self):
        items = {"a": "", "b": "", "c": ""}
        result = greedy_coreset(items, target_size=2)
        assert len(result.selected) == 2
        # All empty -> distance 0 between all -> error 0
        assert result.max_approximation_error == 0.0

    def test_very_long_text(self):
        items = {"a": " ".join(f"word{i}" for i in range(1000)),
                 "b": " ".join(f"other{i}" for i in range(1000))}
        result = greedy_coreset(items, target_size=2)
        assert len(result.selected) == 2

    def test_target_size_one(self):
        items = {f"i{k}": f"text {k}" for k in range(10)}
        result = greedy_coreset(items, target_size=1)
        assert len(result.selected) == 1
        assert result.selected[0].representative_of == 10

    def test_all_same_text(self):
        items = {f"i{k}": "identical content here" for k in range(20)}
        result = greedy_coreset(items, target_size=5)
        assert len(result.selected) == 5
        assert result.max_approximation_error == 0.0
        total_rep = sum(s.representative_of for s in result.selected)
        assert total_rep == 20
