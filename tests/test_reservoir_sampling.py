"""Tests for reservoir_sampling module."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from evalyn_sdk.reservoir_sampling import (
    Reservoir,
    ReservoirConfig,
    ReservoirState,
    create_reservoir,
    format_reservoir_stats,
    stream_from_file,
)


# ---------------------------------------------------------------------------
# ReservoirConfig
# ---------------------------------------------------------------------------


class TestReservoirConfig:
    def test_defaults(self):
        cfg = ReservoirConfig()
        assert cfg.capacity == 100
        assert cfg.seed == 42

    def test_custom_values(self):
        cfg = ReservoirConfig(capacity=10, seed=99)
        assert cfg.capacity == 10
        assert cfg.seed == 99

    def test_as_dict(self):
        cfg = ReservoirConfig(capacity=5, seed=7)
        d = cfg.as_dict()
        assert d == {"capacity": 5, "seed": 7}

    def test_from_dict(self):
        cfg = ReservoirConfig.from_dict({"capacity": 20, "seed": 3})
        assert cfg.capacity == 20
        assert cfg.seed == 3

    def test_from_dict_defaults(self):
        cfg = ReservoirConfig.from_dict({})
        assert cfg.capacity == 100
        assert cfg.seed == 42

    def test_roundtrip(self):
        cfg = ReservoirConfig(capacity=77, seed=11)
        assert ReservoirConfig.from_dict(cfg.as_dict()) == cfg


# ---------------------------------------------------------------------------
# ReservoirState
# ---------------------------------------------------------------------------


class TestReservoirState:
    def test_defaults(self):
        state = ReservoirState()
        assert state.items == []
        assert state.seen_count == 0
        assert state.capacity == 100
        assert state.is_full is False

    def test_as_dict(self):
        state = ReservoirState(
            items=[{"a": 1}], seen_count=5, capacity=3, is_full=True
        )
        d = state.as_dict()
        assert d["items"] == [{"a": 1}]
        assert d["seen_count"] == 5
        assert d["is_full"] is True

    def test_from_dict(self):
        state = ReservoirState.from_dict(
            {"items": [{"x": 2}], "seen_count": 10, "capacity": 5, "is_full": True}
        )
        assert state.items == [{"x": 2}]
        assert state.seen_count == 10

    def test_from_dict_defaults(self):
        state = ReservoirState.from_dict({})
        assert state.items == []
        assert state.seen_count == 0

    def test_roundtrip(self):
        state = ReservoirState(
            items=[{"k": "v"}], seen_count=3, capacity=2, is_full=True
        )
        assert ReservoirState.from_dict(state.as_dict()) == state


# ---------------------------------------------------------------------------
# Reservoir - basic operations
# ---------------------------------------------------------------------------


class TestReservoirBasic:
    def test_empty_reservoir(self):
        r = Reservoir(capacity=5)
        assert r.size() == 0
        assert r.seen() == 0
        assert r.is_full() is False
        assert r.get_sample() == []

    def test_add_below_capacity(self):
        r = Reservoir(capacity=5)
        assert r.add({"id": 1}) is True
        assert r.size() == 1
        assert r.seen() == 1

    def test_fill_to_capacity(self):
        r = Reservoir(capacity=3)
        for i in range(3):
            r.add({"id": i})
        assert r.size() == 3
        assert r.is_full() is True

    def test_add_returns_bool(self):
        r = Reservoir(capacity=2)
        r1 = r.add({"a": 1})
        r2 = r.add({"a": 2})
        assert r1 is True
        assert r2 is True
        # After full, add may return True or False
        r3 = r.add({"a": 3})
        assert isinstance(r3, bool)

    def test_get_sample_returns_copy(self):
        r = Reservoir(capacity=5)
        r.add({"x": 1})
        sample = r.get_sample()
        sample.append({"x": 2})
        assert r.size() == 1

    def test_get_state(self):
        r = Reservoir(capacity=3, seed=10)
        r.add({"id": "a"})
        r.add({"id": "b"})
        state = r.get_state()
        assert state.seen_count == 2
        assert state.capacity == 3
        assert state.is_full is False
        assert len(state.items) == 2

    def test_reset(self):
        r = Reservoir(capacity=5)
        for i in range(10):
            r.add({"id": i})
        r.reset()
        assert r.size() == 0
        assert r.seen() == 0
        assert r.is_full() is False


# ---------------------------------------------------------------------------
# Reservoir - Algorithm R correctness
# ---------------------------------------------------------------------------


class TestAlgorithmR:
    def test_capacity_never_exceeded(self):
        r = Reservoir(capacity=10, seed=0)
        for i in range(1000):
            r.add({"id": i})
        assert r.size() == 10
        assert r.seen() == 1000

    def test_deterministic_with_same_seed(self):
        def run(seed):
            r = Reservoir(capacity=5, seed=seed)
            for i in range(100):
                r.add({"id": i})
            return [item["id"] for item in r.get_sample()]

        assert run(42) == run(42)

    def test_different_seeds_differ(self):
        def run(seed):
            r = Reservoir(capacity=5, seed=seed)
            for i in range(200):
                r.add({"id": i})
            return [item["id"] for item in r.get_sample()]

        # Very unlikely to be identical with different seeds
        assert run(1) != run(999)

    def test_uniform_distribution_approximate(self):
        """Each item should appear with roughly equal probability."""
        counts = [0] * 20
        trials = 5000
        cap = 5
        n_items = 20
        for trial in range(trials):
            r = Reservoir(capacity=cap, seed=trial)
            for i in range(n_items):
                r.add({"id": i})
            for item in r.get_sample():
                counts[item["id"]] += 1
        expected = trials * cap / n_items  # 1250
        for c in counts:
            # Allow 20% deviation
            assert abs(c - expected) / expected < 0.20, (
                f"count {c} too far from expected {expected}"
            )

    def test_all_items_present_when_under_capacity(self):
        r = Reservoir(capacity=100)
        items = [{"id": i} for i in range(50)]
        for item in items:
            r.add(item)
        sample = r.get_sample()
        assert len(sample) == 50
        ids = {item["id"] for item in sample}
        assert ids == set(range(50))

    def test_is_full_boundary(self):
        r = Reservoir(capacity=5)
        for i in range(4):
            r.add({"id": i})
        assert r.is_full() is False
        r.add({"id": 4})
        assert r.is_full() is True


# ---------------------------------------------------------------------------
# Reservoir.add_batch
# ---------------------------------------------------------------------------


class TestAddBatch:
    def test_add_batch_empty(self):
        r = Reservoir(capacity=5)
        count = r.add_batch([])
        assert count == 0
        assert r.size() == 0

    def test_add_batch_below_capacity(self):
        r = Reservoir(capacity=10)
        count = r.add_batch([{"id": i} for i in range(5)])
        assert count == 5
        assert r.size() == 5

    def test_add_batch_above_capacity(self):
        r = Reservoir(capacity=3, seed=7)
        items = [{"id": i} for i in range(100)]
        count = r.add_batch(items)
        assert r.size() == 3
        assert r.seen() == 100
        assert count >= 3  # at least the first 3 were added

    def test_add_batch_updates_seen(self):
        r = Reservoir(capacity=5)
        r.add_batch([{"id": i} for i in range(20)])
        assert r.seen() == 20


# ---------------------------------------------------------------------------
# create_reservoir factory
# ---------------------------------------------------------------------------


class TestCreateReservoir:
    def test_default_factory(self):
        r = create_reservoir()
        assert r.size() == 0
        assert r.seen() == 0

    def test_custom_factory(self):
        r = create_reservoir(capacity=7, seed=99)
        for i in range(20):
            r.add({"id": i})
        assert r.size() == 7


# ---------------------------------------------------------------------------
# stream_from_file
# ---------------------------------------------------------------------------


class TestStreamFromFile:
    def test_stream_jsonl(self, tmp_path):
        fp = tmp_path / "data.jsonl"
        lines = [json.dumps({"id": i, "text": f"item {i}"}) for i in range(50)]
        fp.write_text("\n".join(lines) + "\n")

        r = Reservoir(capacity=10, seed=42)
        state = stream_from_file(str(fp), r)
        assert state.seen_count == 50
        assert len(state.items) == 10
        assert state.is_full is True

    def test_stream_empty_file(self, tmp_path):
        fp = tmp_path / "empty.jsonl"
        fp.write_text("")
        r = Reservoir(capacity=10)
        state = stream_from_file(str(fp), r)
        assert state.seen_count == 0
        assert len(state.items) == 0

    def test_stream_blank_lines_skipped(self, tmp_path):
        fp = tmp_path / "blanks.jsonl"
        content = '{"a":1}\n\n{"b":2}\n\n\n{"c":3}\n'
        fp.write_text(content)
        r = Reservoir(capacity=10)
        state = stream_from_file(str(fp), r)
        assert state.seen_count == 3

    def test_stream_accumulates(self, tmp_path):
        fp = tmp_path / "data.jsonl"
        lines = [json.dumps({"id": i}) for i in range(5)]
        fp.write_text("\n".join(lines) + "\n")

        r = Reservoir(capacity=100)
        r.add({"id": "pre"})
        stream_from_file(str(fp), r)
        assert r.seen() == 6  # 1 pre-added + 5 from file


# ---------------------------------------------------------------------------
# format_reservoir_stats
# ---------------------------------------------------------------------------


class TestFormatStats:
    def test_basic_format(self):
        state = ReservoirState(
            items=[{"a": 1}] * 5,
            seen_count=100,
            capacity=10,
            is_full=True,
        )
        text = format_reservoir_stats(state)
        assert "Reservoir Statistics" in text
        assert "Capacity:    10" in text
        assert "Items held:  5" in text
        assert "Items seen:  100" in text
        assert "Is full:     True" in text
        assert "Sample rate:" in text

    def test_empty_state(self):
        state = ReservoirState()
        text = format_reservoir_stats(state)
        assert "Items seen:  0" in text
        # No sample rate line when seen is 0
        assert "Sample rate:" not in text

    def test_sample_rate_value(self):
        state = ReservoirState(
            items=[{}] * 25, seen_count=100, capacity=50, is_full=True
        )
        text = format_reservoir_stats(state)
        assert "0.2500" in text


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_capacity_one(self):
        r = Reservoir(capacity=1, seed=0)
        for i in range(100):
            r.add({"id": i})
        assert r.size() == 1
        assert r.seen() == 100

    def test_large_items(self):
        r = Reservoir(capacity=2)
        big = {"data": "x" * 10000}
        r.add(big)
        assert r.get_sample()[0]["data"] == "x" * 10000

    def test_nested_dict_items(self):
        r = Reservoir(capacity=5)
        r.add({"nested": {"a": [1, 2, 3]}})
        assert r.get_sample()[0]["nested"]["a"] == [1, 2, 3]

    def test_reset_then_reuse(self):
        r = Reservoir(capacity=5, seed=42)
        for i in range(50):
            r.add({"id": i})
        first_sample = [item["id"] for item in r.get_sample()]
        r.reset()
        for i in range(50):
            r.add({"id": i})
        second_sample = [item["id"] for item in r.get_sample()]
        # Same seed after reset should give same results
        assert first_sample == second_sample

    def test_state_items_are_independent(self):
        r = Reservoir(capacity=5)
        r.add({"id": 1})
        state = r.get_state()
        state.items.append({"id": 999})
        assert r.size() == 1
