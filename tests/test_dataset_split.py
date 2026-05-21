"""Tests for dataset splitting."""

import pytest
from evalyn_sdk.models import DatasetItem
from evalyn_sdk.datasets.split import split_dataset, split_two_way, SplitResult


def _item(item_id: str, metadata: dict = None) -> DatasetItem:
    return DatasetItem(id=item_id, input=f"input-{item_id}", output=f"output-{item_id}",
                      metadata=metadata or {})


class TestSplitDataset:
    def test_basic_split(self):
        items = [_item(str(i)) for i in range(100)]
        result = split_dataset(items, train_ratio=0.7, test_ratio=0.15, validation_ratio=0.15)
        assert result.sizes["train"] == 70
        assert result.sizes["test"] == 15
        assert result.sizes["validation"] == 15

    def test_no_overlap(self):
        items = [_item(str(i)) for i in range(50)]
        result = split_dataset(items)
        all_ids = (
            {i.id for i in result.train}
            | {i.id for i in result.test}
            | {i.id for i in result.validation}
        )
        assert len(all_ids) == 50  # all items accounted for

    def test_deterministic_with_seed(self):
        items = [_item(str(i)) for i in range(100)]
        r1 = split_dataset(items, seed=42)
        r2 = split_dataset(items, seed=42)
        assert [i.id for i in r1.train] == [i.id for i in r2.train]
        assert [i.id for i in r1.test] == [i.id for i in r2.test]

    def test_different_seeds_differ(self):
        items = [_item(str(i)) for i in range(100)]
        r1 = split_dataset(items, seed=1)
        r2 = split_dataset(items, seed=999)
        assert [i.id for i in r1.train] != [i.id for i in r2.train]

    def test_invalid_ratios_raise(self):
        items = [_item("1")]
        with pytest.raises(ValueError, match="sum to 1.0"):
            split_dataset(items, train_ratio=0.5, test_ratio=0.5, validation_ratio=0.5)

    def test_empty_dataset(self):
        result = split_dataset([])
        assert result.sizes == {"train": 0, "test": 0, "validation": 0}

    def test_small_dataset(self):
        items = [_item(str(i)) for i in range(3)]
        result = split_dataset(items, train_ratio=0.7, test_ratio=0.15, validation_ratio=0.15)
        total = sum(result.sizes.values())
        assert total == 3

    def test_80_20_split(self):
        items = [_item(str(i)) for i in range(100)]
        result = split_dataset(items, train_ratio=0.8, test_ratio=0.2, validation_ratio=0.0)
        assert result.sizes["train"] == 80
        assert result.sizes["test"] == 20
        assert result.sizes["validation"] == 0

    def test_ratios_property(self):
        items = [_item(str(i)) for i in range(100)]
        result = split_dataset(items)
        ratios = result.ratios
        assert ratios["train"] == pytest.approx(0.7, abs=0.05)


class TestStratifiedSplit:
    def test_stratified_by_metadata(self):
        items = (
            [_item(f"a-{i}", metadata={"source": "prod"}) for i in range(50)]
            + [_item(f"b-{i}", metadata={"source": "test"}) for i in range(50)]
        )
        result = split_dataset(items, stratify_by="source")

        # Both groups should be represented in train
        train_sources = {(i.metadata or {}).get("source") for i in result.train}
        assert "prod" in train_sources
        assert "test" in train_sources

    def test_stratified_preserves_all(self):
        items = [
            _item("1", metadata={"cat": "A"}),
            _item("2", metadata={"cat": "A"}),
            _item("3", metadata={"cat": "B"}),
            _item("4", metadata={"cat": "B"}),
            _item("5", metadata={"cat": "B"}),
        ]
        result = split_dataset(items, stratify_by="cat")
        total = sum(result.sizes.values())
        assert total == 5


class TestSplitTwoWay:
    def test_basic(self):
        items = [_item(str(i)) for i in range(100)]
        first, second = split_two_way(items, ratio=0.8)
        assert len(first) == 80
        # second gets remainder from int rounding
        assert len(first) + len(second) == 100

    def test_no_overlap(self):
        items = [_item(str(i)) for i in range(50)]
        first, second = split_two_way(items)
        first_ids = {i.id for i in first}
        second_ids = {i.id for i in second}
        assert first_ids.isdisjoint(second_ids)
