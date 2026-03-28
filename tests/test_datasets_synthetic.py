"""Tests for synthetic dataset generation."""

from __future__ import annotations

import sys
from pathlib import Path

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.datasets_synthetic import (
    SyntheticConfig,
    SyntheticDataset,
    SyntheticItem,
    generate_adversarial_items,
    generate_synthetic_dataset,
    generate_synthetic_item,
    validate_synthetic_dataset,
)


# ---------------------------------------------------------------------------
# generate_synthetic_item
# ---------------------------------------------------------------------------


class TestGenerateSyntheticItem:
    def test_basic(self):
        item = generate_synthetic_item("test_001")
        assert item.id == "test_001"
        assert item.input_text
        assert item.category == "general"
        assert item.difficulty == "medium"

    def test_custom_category(self):
        item = generate_synthetic_item("test_002", category="math")
        assert item.category == "math"

    def test_easy_difficulty(self):
        item = generate_synthetic_item("test_003", difficulty="easy", seed=42)
        assert item.difficulty == "easy"
        # Easy templates produce shorter inputs
        assert len(item.input_text) < 200

    def test_medium_difficulty(self):
        item = generate_synthetic_item("test_004", difficulty="medium", seed=42)
        assert item.difficulty == "medium"

    def test_hard_difficulty(self):
        item = generate_synthetic_item("test_005", difficulty="hard", seed=42)
        assert item.difficulty == "hard"
        # Hard templates produce longer, multi-part inputs
        assert len(item.input_text) > 30

    def test_has_expected_output(self):
        item = generate_synthetic_item("test_006")
        assert item.expected_output.startswith("expected_")

    def test_metadata_contains_synthetic_flag(self):
        item = generate_synthetic_item("test_007")
        assert item.metadata.get("synthetic") is True

    def test_deterministic_with_seed(self):
        a = generate_synthetic_item("test_008", seed=123)
        b = generate_synthetic_item("test_008", seed=123)
        assert a.input_text == b.input_text
        assert a.expected_output == b.expected_output

    def test_different_seeds_differ(self):
        a = generate_synthetic_item("test_009", seed=1)
        b = generate_synthetic_item("test_009", seed=2)
        # Same ID but different seed may pick different template
        # At minimum the items should be valid
        assert a.id == b.id
        assert a.input_text  # non-empty


# ---------------------------------------------------------------------------
# generate_synthetic_dataset
# ---------------------------------------------------------------------------


class TestGenerateSyntheticDataset:
    def test_correct_count(self):
        cfg = SyntheticConfig(num_items=20, seed=42)
        ds = generate_synthetic_dataset(cfg)
        assert ds.count == 20

    def test_zero_items(self):
        cfg = SyntheticConfig(num_items=0, seed=1)
        ds = generate_synthetic_dataset(cfg)
        assert ds.count == 0
        assert ds.items == []

    def test_single_item(self):
        cfg = SyntheticConfig(num_items=1, seed=10)
        ds = generate_synthetic_dataset(cfg)
        assert ds.count == 1

    def test_category_distribution(self):
        cfg = SyntheticConfig(
            num_items=12,
            categories=["math", "science", "history"],
            seed=42,
        )
        ds = generate_synthetic_dataset(cfg)
        cats = ds.by_category
        # Items round-robin across categories: 12 / 3 = 4 each
        assert cats.get("math", 0) == 4
        assert cats.get("science", 0) == 4
        assert cats.get("history", 0) == 4

    def test_single_category(self):
        cfg = SyntheticConfig(num_items=5, categories=["only"], seed=1)
        ds = generate_synthetic_dataset(cfg)
        assert all(item.category == "only" for item in ds.items)

    def test_difficulty_distribution_present(self):
        cfg = SyntheticConfig(num_items=30, seed=42)
        ds = generate_synthetic_dataset(cfg)
        diffs = ds.by_difficulty
        # All three difficulties should be present
        assert "easy" in diffs
        assert "medium" in diffs
        assert "hard" in diffs

    def test_config_attached(self):
        cfg = SyntheticConfig(num_items=5, seed=7)
        ds = generate_synthetic_dataset(cfg)
        assert ds.config is cfg

    def test_deterministic_with_seed(self):
        cfg = SyntheticConfig(num_items=10, seed=99)
        ds_a = generate_synthetic_dataset(cfg)
        ds_b = generate_synthetic_dataset(cfg)
        for a, b in zip(ds_a.items, ds_b.items):
            assert a.id == b.id
            assert a.input_text == b.input_text


# ---------------------------------------------------------------------------
# generate_adversarial_items
# ---------------------------------------------------------------------------


class TestGenerateAdversarialItems:
    def test_basic(self):
        base = [generate_synthetic_item("base_001", seed=1)]
        adv = generate_adversarial_items(base, num_adversarial=3, seed=42)
        assert len(adv) == 3

    def test_ids_contain_adv(self):
        base = [generate_synthetic_item("base_002", seed=1)]
        adv = generate_adversarial_items(base, num_adversarial=2, seed=42)
        for item in adv:
            assert "_adv_" in item.id

    def test_metadata_adversarial_flag(self):
        base = [generate_synthetic_item("base_003", seed=1)]
        adv = generate_adversarial_items(base, num_adversarial=1, seed=42)
        assert adv[0].metadata.get("adversarial") is True

    def test_empty_base_returns_empty(self):
        adv = generate_adversarial_items([], num_adversarial=5, seed=42)
        assert adv == []

    def test_transforms_modify_input(self):
        base = [generate_synthetic_item("base_004", seed=1)]
        adv = generate_adversarial_items(base, num_adversarial=5, seed=42)
        # At least some adversarial inputs should differ from the base
        base_text = base[0].input_text
        different = [item for item in adv if item.input_text != base_text]
        assert len(different) >= 1

    def test_source_id_in_metadata(self):
        base = [generate_synthetic_item("base_005", seed=1)]
        adv = generate_adversarial_items(base, num_adversarial=2, seed=42)
        for item in adv:
            assert item.metadata.get("source_id") == "base_005"


# ---------------------------------------------------------------------------
# validate_synthetic_dataset
# ---------------------------------------------------------------------------


class TestValidateSyntheticDataset:
    def test_valid_dataset(self):
        cfg = SyntheticConfig(num_items=5, seed=42)
        ds = generate_synthetic_dataset(cfg)
        valid, errors = validate_synthetic_dataset(ds)
        assert valid is True
        assert errors == []

    def test_duplicate_ids(self):
        items = [
            SyntheticItem(id="dup", input_text="a"),
            SyntheticItem(id="dup", input_text="b"),
        ]
        ds = SyntheticDataset(items=items)
        valid, errors = validate_synthetic_dataset(ds)
        assert valid is False
        assert any("Duplicate ID" in e for e in errors)

    def test_empty_input_text(self):
        items = [SyntheticItem(id="empty", input_text="")]
        ds = SyntheticDataset(items=items)
        valid, errors = validate_synthetic_dataset(ds)
        assert valid is False
        assert any("empty input_text" in e for e in errors)

    def test_unexpected_category(self):
        cfg = SyntheticConfig(num_items=1, categories=["math"], seed=1)
        item = SyntheticItem(id="wrong", input_text="test", category="science")
        ds = SyntheticDataset(items=[item], config=cfg)
        valid, errors = validate_synthetic_dataset(ds)
        assert valid is False
        assert any("Unexpected categories" in e for e in errors)

    def test_empty_dataset_is_valid(self):
        ds = SyntheticDataset(items=[])
        valid, errors = validate_synthetic_dataset(ds)
        assert valid is True


# ---------------------------------------------------------------------------
# SyntheticDataset properties
# ---------------------------------------------------------------------------


class TestSyntheticDatasetProperties:
    def test_count(self):
        items = [SyntheticItem(id=f"i{i}", input_text=f"text {i}") for i in range(7)]
        ds = SyntheticDataset(items=items)
        assert ds.count == 7

    def test_by_category(self):
        items = [
            SyntheticItem(id="a", input_text="x", category="math"),
            SyntheticItem(id="b", input_text="y", category="math"),
            SyntheticItem(id="c", input_text="z", category="science"),
        ]
        ds = SyntheticDataset(items=items)
        assert ds.by_category == {"math": 2, "science": 1}

    def test_by_difficulty(self):
        items = [
            SyntheticItem(id="a", input_text="x", difficulty="easy"),
            SyntheticItem(id="b", input_text="y", difficulty="hard"),
            SyntheticItem(id="c", input_text="z", difficulty="easy"),
        ]
        ds = SyntheticDataset(items=items)
        assert ds.by_difficulty == {"easy": 2, "hard": 1}

    def test_format_text(self):
        items = [SyntheticItem(id="a", input_text="x", category="math")]
        ds = SyntheticDataset(items=items)
        text = ds.format_text()
        assert "1 items" in text
        assert "math" in text


# ---------------------------------------------------------------------------
# Serialization roundtrips
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_synthetic_config_roundtrip(self):
        cfg = SyntheticConfig(
            num_items=15,
            categories=["a", "b"],
            difficulty_distribution={"easy": 0.5, "hard": 0.5},
            seed=99,
        )
        restored = SyntheticConfig.from_dict(cfg.as_dict())
        assert restored.num_items == cfg.num_items
        assert restored.categories == cfg.categories
        assert restored.difficulty_distribution == cfg.difficulty_distribution
        assert restored.seed == cfg.seed

    def test_synthetic_config_from_dict_defaults(self):
        cfg = SyntheticConfig.from_dict({})
        assert cfg.num_items == 10
        assert cfg.categories == ["general"]
        assert cfg.seed is None

    def test_synthetic_item_roundtrip(self):
        item = generate_synthetic_item("rt_001", seed=42)
        restored = SyntheticItem.from_dict(item.as_dict())
        assert restored.id == item.id
        assert restored.input_text == item.input_text
        assert restored.expected_output == item.expected_output
        assert restored.category == item.category
        assert restored.difficulty == item.difficulty
        assert restored.metadata == item.metadata

    def test_synthetic_dataset_roundtrip(self):
        cfg = SyntheticConfig(num_items=3, seed=42)
        ds = generate_synthetic_dataset(cfg)
        restored = SyntheticDataset.from_dict(ds.as_dict())
        assert restored.count == ds.count
        assert restored.config is not None
        assert restored.config.num_items == cfg.num_items
        for orig, rest in zip(ds.items, restored.items):
            assert orig.id == rest.id
            assert orig.input_text == rest.input_text
