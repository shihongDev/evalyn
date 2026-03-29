"""Tests for evalyn_sdk.reproducibility_seed module."""

from __future__ import annotations

import json
import random

import pytest

from evalyn_sdk.reproducibility_seed import (
    ReproducibilityRecord,
    SeedConfig,
    SeededRandom,
    compute_items_checksum,
    create_reproducibility_record,
    create_seeded_rng,
    format_reproducibility_info,
    verify_reproducibility,
)


# -----------------------------------------------------------------------
# SeedConfig
# -----------------------------------------------------------------------


class TestSeedConfig:
    def test_defaults(self):
        cfg = SeedConfig(seed=42)
        assert cfg.seed == 42
        assert cfg.temperature == 0.0
        assert cfg.record_in_metadata is True

    def test_custom_values(self):
        cfg = SeedConfig(seed=7, temperature=0.5, record_in_metadata=False)
        assert cfg.seed == 7
        assert cfg.temperature == 0.5
        assert cfg.record_in_metadata is False

    def test_as_dict(self):
        cfg = SeedConfig(seed=1, temperature=0.3, record_in_metadata=False)
        d = cfg.as_dict()
        assert d == {"seed": 1, "temperature": 0.3, "record_in_metadata": False}

    def test_from_dict_full(self):
        d = {"seed": 99, "temperature": 1.0, "record_in_metadata": False}
        cfg = SeedConfig.from_dict(d)
        assert cfg.seed == 99
        assert cfg.temperature == 1.0
        assert cfg.record_in_metadata is False

    def test_from_dict_defaults(self):
        cfg = SeedConfig.from_dict({"seed": 5})
        assert cfg.temperature == 0.0
        assert cfg.record_in_metadata is True

    def test_roundtrip(self):
        cfg = SeedConfig(seed=123, temperature=0.7, record_in_metadata=False)
        assert SeedConfig.from_dict(cfg.as_dict()) == cfg


# -----------------------------------------------------------------------
# ReproducibilityRecord
# -----------------------------------------------------------------------


class TestReproducibilityRecord:
    def test_creation(self):
        rec = ReproducibilityRecord(
            seed=42, timestamp="2026-01-01T00:00:00+00:00",
            item_count=3, checksum="abc123",
        )
        assert rec.seed == 42
        assert rec.item_count == 3

    def test_default_metadata(self):
        rec = ReproducibilityRecord(
            seed=1, timestamp="t", item_count=0, checksum="x",
        )
        assert rec.metadata == {}

    def test_as_dict(self):
        rec = ReproducibilityRecord(
            seed=1, timestamp="t", item_count=2, checksum="c",
            metadata={"k": "v"},
        )
        d = rec.as_dict()
        assert d["seed"] == 1
        assert d["metadata"] == {"k": "v"}

    def test_from_dict(self):
        d = {
            "seed": 10, "timestamp": "ts", "item_count": 5,
            "checksum": "chk", "metadata": {"a": 1},
        }
        rec = ReproducibilityRecord.from_dict(d)
        assert rec.seed == 10
        assert rec.metadata == {"a": 1}

    def test_from_dict_defaults(self):
        d = {"seed": 1, "timestamp": "t", "item_count": 0, "checksum": "c"}
        rec = ReproducibilityRecord.from_dict(d)
        assert rec.metadata == {}

    def test_roundtrip(self):
        rec = ReproducibilityRecord(
            seed=7, timestamp="2026-03-29T12:00:00Z",
            item_count=10, checksum="deadbeef", metadata={"run": 1},
        )
        assert ReproducibilityRecord.from_dict(rec.as_dict()) == rec


# -----------------------------------------------------------------------
# SeededRandom
# -----------------------------------------------------------------------


class TestSeededRandom:
    def test_get_seed(self):
        sr = SeededRandom(42)
        assert sr.get_seed() == 42

    def test_choice(self):
        sr = SeededRandom(0)
        result = sr.choice([1, 2, 3, 4, 5])
        assert result in [1, 2, 3, 4, 5]

    def test_choice_deterministic(self):
        a = SeededRandom(99).choice(list(range(100)))
        b = SeededRandom(99).choice(list(range(100)))
        assert a == b

    def test_shuffle_returns_new_list(self):
        sr = SeededRandom(0)
        original = [1, 2, 3, 4, 5]
        copy = list(original)
        shuffled = sr.shuffle(original)
        assert original == copy  # original unchanged
        assert sorted(shuffled) == sorted(original)

    def test_shuffle_deterministic(self):
        data = list(range(20))
        a = SeededRandom(7).shuffle(data)
        b = SeededRandom(7).shuffle(data)
        assert a == b

    def test_sample(self):
        sr = SeededRandom(0)
        result = sr.sample(list(range(50)), 5)
        assert len(result) == 5
        assert len(set(result)) == 5  # no duplicates

    def test_sample_deterministic(self):
        pop = list(range(100))
        a = SeededRandom(3).sample(pop, 10)
        b = SeededRandom(3).sample(pop, 10)
        assert a == b

    def test_uniform(self):
        sr = SeededRandom(0)
        v = sr.uniform(0.0, 1.0)
        assert 0.0 <= v <= 1.0

    def test_uniform_deterministic(self):
        a = SeededRandom(5).uniform(0.0, 100.0)
        b = SeededRandom(5).uniform(0.0, 100.0)
        assert a == b

    def test_randint(self):
        sr = SeededRandom(0)
        v = sr.randint(1, 10)
        assert 1 <= v <= 10

    def test_randint_deterministic(self):
        a = SeededRandom(11).randint(0, 1000)
        b = SeededRandom(11).randint(0, 1000)
        assert a == b

    def test_different_seeds_differ(self):
        a = SeededRandom(0).shuffle(list(range(50)))
        b = SeededRandom(1).shuffle(list(range(50)))
        assert a != b


# -----------------------------------------------------------------------
# create_seeded_rng
# -----------------------------------------------------------------------


class TestCreateSeededRng:
    def test_returns_seeded_random(self):
        rng = create_seeded_rng(42)
        assert isinstance(rng, SeededRandom)
        assert rng.get_seed() == 42

    def test_deterministic(self):
        a = create_seeded_rng(10).shuffle(list(range(30)))
        b = create_seeded_rng(10).shuffle(list(range(30)))
        assert a == b


# -----------------------------------------------------------------------
# compute_items_checksum
# -----------------------------------------------------------------------


class TestComputeItemsChecksum:
    def test_basic(self):
        items = [{"a": 1}, {"b": 2}]
        chk = compute_items_checksum(items)
        assert isinstance(chk, str)
        assert len(chk) == 64  # SHA-256 hex digest

    def test_deterministic(self):
        items = [{"x": "hello"}, {"y": "world"}]
        assert compute_items_checksum(items) == compute_items_checksum(items)

    def test_order_matters(self):
        a = compute_items_checksum([{"a": 1}, {"b": 2}])
        b = compute_items_checksum([{"b": 2}, {"a": 1}])
        assert a != b

    def test_key_order_irrelevant(self):
        """Keys within a single dict are sorted, so order does not matter."""
        a = compute_items_checksum([{"b": 2, "a": 1}])
        b = compute_items_checksum([{"a": 1, "b": 2}])
        assert a == b

    def test_empty_list(self):
        chk = compute_items_checksum([])
        assert isinstance(chk, str)
        assert len(chk) == 64

    def test_unicode(self):
        items = [{"text": "hello"}]
        chk = compute_items_checksum(items)
        assert len(chk) == 64


# -----------------------------------------------------------------------
# create_reproducibility_record
# -----------------------------------------------------------------------


class TestCreateReproducibilityRecord:
    def test_basic(self):
        items = [{"a": 1}]
        rec = create_reproducibility_record(seed=42, items=items)
        assert rec.seed == 42
        assert rec.item_count == 1
        assert rec.checksum == compute_items_checksum(items)
        assert rec.metadata == {}

    def test_with_metadata(self):
        rec = create_reproducibility_record(
            seed=1, items=[], metadata={"run": "test"},
        )
        assert rec.metadata == {"run": "test"}

    def test_timestamp_present(self):
        rec = create_reproducibility_record(seed=0, items=[])
        assert rec.timestamp  # non-empty string


# -----------------------------------------------------------------------
# verify_reproducibility
# -----------------------------------------------------------------------


class TestVerifyReproducibility:
    def test_matching(self):
        items = [{"k": "v"}]
        rec = create_reproducibility_record(seed=0, items=items)
        assert verify_reproducibility(rec, items) is True

    def test_mismatch(self):
        items = [{"k": "v"}]
        rec = create_reproducibility_record(seed=0, items=items)
        assert verify_reproducibility(rec, [{"k": "other"}]) is False

    def test_empty(self):
        rec = create_reproducibility_record(seed=0, items=[])
        assert verify_reproducibility(rec, []) is True

    def test_extra_item(self):
        items = [{"a": 1}]
        rec = create_reproducibility_record(seed=0, items=items)
        assert verify_reproducibility(rec, items + [{"b": 2}]) is False


# -----------------------------------------------------------------------
# format_reproducibility_info
# -----------------------------------------------------------------------


class TestFormatReproducibilityInfo:
    def test_basic(self):
        rec = ReproducibilityRecord(
            seed=42, timestamp="2026-01-01T00:00:00Z",
            item_count=5, checksum="abc",
        )
        text = format_reproducibility_info(rec)
        assert "Seed: 42" in text
        assert "Items: 5" in text
        assert "Checksum: abc" in text

    def test_with_metadata(self):
        rec = ReproducibilityRecord(
            seed=1, timestamp="t", item_count=0,
            checksum="c", metadata={"k": "v"},
        )
        text = format_reproducibility_info(rec)
        assert "Metadata:" in text
        assert '"k"' in text

    def test_no_metadata(self):
        rec = ReproducibilityRecord(
            seed=1, timestamp="t", item_count=0, checksum="c",
        )
        text = format_reproducibility_info(rec)
        assert "Metadata" not in text
