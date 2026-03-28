"""Tests for dataset interleaving."""

import pytest
from evalyn_sdk.datasets_interleaving import (
    InterleavedItem,
    InterleavingConfig,
    InterleavingResult,
    compute_balance_score,
    interleave,
    interleave_random,
    interleave_round_robin,
    interleave_weighted,
    validate_interleaving,
)


def _ds(prefix: str, n: int) -> list:
    return [{"text": f"{prefix}-{i}"} for i in range(n)]


class TestInterleavingConfig:
    def test_as_dict(self):
        cfg = InterleavingConfig(strategy="weighted", weights={"a": 2.0}, seed=42)
        d = cfg.as_dict()
        assert d["strategy"] == "weighted"
        assert d["weights"] == {"a": 2.0}
        assert d["seed"] == 42

    def test_from_dict(self):
        cfg = InterleavingConfig.from_dict({"strategy": "random", "seed": 7})
        assert cfg.strategy == "random"
        assert cfg.seed == 7
        assert cfg.weights == {}

    def test_roundtrip(self):
        cfg = InterleavingConfig(strategy="weighted", weights={"x": 3.0}, seed=99)
        assert InterleavingConfig.from_dict(cfg.as_dict()).strategy == "weighted"
        assert InterleavingConfig.from_dict(cfg.as_dict()).seed == 99


class TestInterleavedItem:
    def test_as_dict(self):
        item = InterleavedItem(id="a_0", source_dataset="a", original_index=0, data={"k": "v"})
        d = item.as_dict()
        assert d["id"] == "a_0"
        assert d["data"] == {"k": "v"}

    def test_from_dict(self):
        item = InterleavedItem.from_dict({"id": "b_1", "source_dataset": "b", "original_index": 1})
        assert item.source_dataset == "b"
        assert item.data == {}

    def test_roundtrip(self):
        item = InterleavedItem(id="c_2", source_dataset="c", original_index=2, data={"x": 1})
        restored = InterleavedItem.from_dict(item.as_dict())
        assert restored.id == item.id
        assert restored.data == item.data


class TestInterleavingResult:
    def test_as_dict_from_dict_roundtrip(self):
        result = InterleavingResult(
            items=[InterleavedItem(id="a_0", source_dataset="a", original_index=0)],
            total_items=1,
            source_distribution={"a": 1},
        )
        restored = InterleavingResult.from_dict(result.as_dict())
        assert restored.total_items == 1
        assert restored.items[0].id == "a_0"

    def test_format_text(self):
        result = InterleavingResult(
            items=[], total_items=10, source_distribution={"a": 6, "b": 4}
        )
        text = result.format_text()
        assert "10 items" in text
        assert "2 datasets" in text
        assert "a:" in text


class TestInterleaveRoundRobin:
    def test_alternates(self):
        datasets = {"a": _ds("a", 3), "b": _ds("b", 3)}
        result = interleave_round_robin(datasets)
        sources = [item.source_dataset for item in result.items]
        assert sources == ["a", "b", "a", "b", "a", "b"]
        assert result.total_items == 6

    def test_uneven_datasets(self):
        datasets = {"a": _ds("a", 2), "b": _ds("b", 4)}
        result = interleave_round_robin(datasets)
        assert result.total_items == 6
        assert result.source_distribution["a"] == 2
        assert result.source_distribution["b"] == 4
        # First round: a, b; second round: a, b; then b fills remaining
        sources = [item.source_dataset for item in result.items]
        assert sources[:4] == ["a", "b", "a", "b"]

    def test_single_dataset(self):
        datasets = {"only": _ds("only", 5)}
        result = interleave_round_robin(datasets)
        assert result.total_items == 5
        assert all(item.source_dataset == "only" for item in result.items)

    def test_empty_datasets(self):
        result = interleave_round_robin({})
        assert result.total_items == 0
        assert result.items == []

    def test_one_empty_one_full(self):
        datasets = {"a": [], "b": _ds("b", 3)}
        result = interleave_round_robin(datasets)
        assert result.total_items == 3
        assert result.source_distribution.get("a", 0) == 0


class TestInterleaveWeighted:
    def test_respects_weights(self):
        datasets = {"a": _ds("a", 100), "b": _ds("b", 100)}
        result = interleave_weighted(datasets, {"a": 3.0, "b": 1.0}, seed=42)
        assert result.total_items == 200
        # With 3:1 weighting, a should appear more often in first half
        first_half = result.items[:100]
        a_count = sum(1 for item in first_half if item.source_dataset == "a")
        # a should dominate the first half (not exact, but significantly more)
        assert a_count > 50

    def test_all_items_included(self):
        datasets = {"x": _ds("x", 10), "y": _ds("y", 10)}
        result = interleave_weighted(datasets, {"x": 1.0, "y": 1.0}, seed=7)
        assert result.total_items == 20

    def test_deterministic_with_seed(self):
        datasets = {"a": _ds("a", 20), "b": _ds("b", 20)}
        r1 = interleave_weighted(datasets, {"a": 2.0, "b": 1.0}, seed=123)
        r2 = interleave_weighted(datasets, {"a": 2.0, "b": 1.0}, seed=123)
        ids1 = [item.id for item in r1.items]
        ids2 = [item.id for item in r2.items]
        assert ids1 == ids2


class TestInterleaveRandom:
    def test_includes_all(self):
        datasets = {"a": _ds("a", 5), "b": _ds("b", 5)}
        result = interleave_random(datasets, seed=42)
        assert result.total_items == 10
        assert result.source_distribution == {"a": 5, "b": 5}

    def test_deterministic_with_seed(self):
        datasets = {"a": _ds("a", 10), "b": _ds("b", 10)}
        r1 = interleave_random(datasets, seed=99)
        r2 = interleave_random(datasets, seed=99)
        ids1 = [item.id for item in r1.items]
        ids2 = [item.id for item in r2.items]
        assert ids1 == ids2

    def test_different_seeds_differ(self):
        datasets = {"a": _ds("a", 20), "b": _ds("b", 20)}
        r1 = interleave_random(datasets, seed=1)
        r2 = interleave_random(datasets, seed=2)
        ids1 = [item.id for item in r1.items]
        ids2 = [item.id for item in r2.items]
        assert ids1 != ids2

    def test_empty(self):
        result = interleave_random({}, seed=1)
        assert result.total_items == 0


class TestInterleaveRouter:
    def test_routes_round_robin(self):
        datasets = {"a": _ds("a", 3)}
        cfg = InterleavingConfig(strategy="round_robin")
        result = interleave(datasets, cfg)
        assert result.total_items == 3

    def test_routes_weighted(self):
        datasets = {"a": _ds("a", 5), "b": _ds("b", 5)}
        cfg = InterleavingConfig(strategy="weighted", weights={"a": 1.0, "b": 1.0}, seed=42)
        result = interleave(datasets, cfg)
        assert result.total_items == 10

    def test_routes_random(self):
        datasets = {"a": _ds("a", 5)}
        cfg = InterleavingConfig(strategy="random", seed=42)
        result = interleave(datasets, cfg)
        assert result.total_items == 5

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            interleave({}, InterleavingConfig(strategy="bogus"))


class TestValidateInterleaving:
    def test_valid(self):
        result = interleave_round_robin({"a": _ds("a", 3), "b": _ds("b", 3)})
        valid, errors = validate_interleaving(result)
        assert valid is True
        assert errors == []

    def test_duplicate_ids(self):
        result = InterleavingResult(
            items=[
                InterleavedItem(id="dup", source_dataset="a", original_index=0),
                InterleavedItem(id="dup", source_dataset="b", original_index=0),
            ],
            total_items=2,
            source_distribution={"a": 1, "b": 1},
        )
        valid, errors = validate_interleaving(result)
        assert valid is False
        assert any("Duplicate" in e for e in errors)

    def test_mismatched_total(self):
        result = InterleavingResult(
            items=[InterleavedItem(id="a_0", source_dataset="a", original_index=0)],
            total_items=5,
            source_distribution={"a": 5},
        )
        valid, errors = validate_interleaving(result)
        assert valid is False


class TestComputeBalanceScore:
    def test_perfectly_even(self):
        datasets = {"a": _ds("a", 5), "b": _ds("b", 5)}
        result = interleave_round_robin(datasets)
        score = compute_balance_score(result)
        assert score == pytest.approx(1.0)

    def test_uneven(self):
        result = InterleavingResult(
            items=[], total_items=10, source_distribution={"a": 9, "b": 1}
        )
        score = compute_balance_score(result)
        assert 0.0 < score < 1.0

    def test_single_source(self):
        result = InterleavingResult(
            items=[], total_items=5, source_distribution={"a": 5}
        )
        score = compute_balance_score(result)
        assert score == 1.0

    def test_empty(self):
        result = InterleavingResult(items=[], total_items=0, source_distribution={})
        score = compute_balance_score(result)
        assert score == 1.0
