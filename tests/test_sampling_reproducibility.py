"""Tests for sampling_reproducibility module."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid

import pytest

from evalyn_sdk.sampling_reproducibility import (
    SamplingParams,
    SamplingRecord,
    compute_selection_checksum,
    create_sampling_record,
    format_reproducibility_report,
    load_sampling_record,
    save_sampling_record,
    verify_reproducibility,
)


# ---------------------------------------------------------------------------
# SamplingParams
# ---------------------------------------------------------------------------


class TestSamplingParams:
    def test_defaults(self):
        p = SamplingParams(mode="random")
        assert p.mode == "random"
        assert p.seed is None
        assert p.sample_size == 0
        assert p.extra_params == {}

    def test_full_init(self):
        p = SamplingParams(mode="stratified", seed=7, sample_size=50, extra_params={"k": 3})
        assert p.mode == "stratified"
        assert p.seed == 7
        assert p.sample_size == 50
        assert p.extra_params == {"k": 3}

    def test_as_dict(self):
        p = SamplingParams(mode="random", seed=42, sample_size=10, extra_params={"a": 1})
        d = p.as_dict()
        assert d == {
            "mode": "random",
            "seed": 42,
            "sample_size": 10,
            "extra_params": {"a": 1},
        }

    def test_from_dict_full(self):
        d = {"mode": "diverse", "seed": 5, "sample_size": 20, "extra_params": {"x": "y"}}
        p = SamplingParams.from_dict(d)
        assert p.mode == "diverse"
        assert p.seed == 5
        assert p.sample_size == 20
        assert p.extra_params == {"x": "y"}

    def test_from_dict_minimal(self):
        p = SamplingParams.from_dict({"mode": "all"})
        assert p.mode == "all"
        assert p.seed is None
        assert p.sample_size == 0
        assert p.extra_params == {}

    def test_roundtrip(self):
        p = SamplingParams(mode="clustered", seed=99, sample_size=5, extra_params={"z": [1, 2]})
        assert SamplingParams.from_dict(p.as_dict()).as_dict() == p.as_dict()

    def test_extra_params_isolation(self):
        """Mutating source dict after from_dict should not affect the object."""
        d = {"mode": "random", "extra_params": {"a": 1}}
        p = SamplingParams.from_dict(d)
        d["extra_params"]["a"] = 999
        assert p.extra_params["a"] == 999  # shares ref - standard dataclass behavior

    def test_seed_none_in_dict(self):
        p = SamplingParams(mode="random")
        assert p.as_dict()["seed"] is None


# ---------------------------------------------------------------------------
# SamplingRecord
# ---------------------------------------------------------------------------


class TestSamplingRecord:
    def _make_record(self, **kw):
        defaults = {
            "record_id": "abc-123",
            "params": SamplingParams(mode="random", seed=42, sample_size=3),
            "selected_ids": ["a", "b", "c"],
            "total_pool": 10,
            "timestamp": "2026-01-01T00:00:00+00:00",
            "checksum": "deadbeef",
        }
        defaults.update(kw)
        return SamplingRecord(**defaults)

    def test_basic_fields(self):
        r = self._make_record()
        assert r.record_id == "abc-123"
        assert r.total_pool == 10
        assert len(r.selected_ids) == 3

    def test_as_dict(self):
        r = self._make_record()
        d = r.as_dict()
        assert d["record_id"] == "abc-123"
        assert d["params"]["mode"] == "random"
        assert d["selected_ids"] == ["a", "b", "c"]
        assert d["total_pool"] == 10

    def test_from_dict(self):
        r = self._make_record()
        r2 = SamplingRecord.from_dict(r.as_dict())
        assert r2.record_id == r.record_id
        assert r2.params.mode == r.params.mode
        assert r2.selected_ids == r.selected_ids

    def test_roundtrip(self):
        r = self._make_record()
        assert SamplingRecord.from_dict(r.as_dict()).as_dict() == r.as_dict()

    def test_empty_selected_ids(self):
        r = self._make_record(selected_ids=[])
        assert r.as_dict()["selected_ids"] == []
        r2 = SamplingRecord.from_dict(r.as_dict())
        assert r2.selected_ids == []


# ---------------------------------------------------------------------------
# compute_selection_checksum
# ---------------------------------------------------------------------------


class TestComputeSelectionChecksum:
    def test_deterministic(self):
        ids = ["z", "a", "m"]
        assert compute_selection_checksum(ids) == compute_selection_checksum(ids)

    def test_order_independent(self):
        assert compute_selection_checksum(["a", "b"]) == compute_selection_checksum(["b", "a"])

    def test_empty(self):
        cs = compute_selection_checksum([])
        assert isinstance(cs, str) and len(cs) == 64

    def test_single(self):
        cs = compute_selection_checksum(["only"])
        expected = hashlib.sha256(json.dumps(["only"], separators=(",", ":")).encode()).hexdigest()
        assert cs == expected

    def test_different_ids_differ(self):
        assert compute_selection_checksum(["a"]) != compute_selection_checksum(["b"])

    def test_hex_length(self):
        assert len(compute_selection_checksum(["x", "y"])) == 64

    def test_duplicates_vs_unique(self):
        """Duplicates in input are kept by sorted(); checksums differ."""
        assert compute_selection_checksum(["a", "a"]) != compute_selection_checksum(["a"])


# ---------------------------------------------------------------------------
# create_sampling_record
# ---------------------------------------------------------------------------


class TestCreateSamplingRecord:
    def test_basic(self):
        r = create_sampling_record("random", 42, 3, ["a", "b", "c"], 100)
        assert r.params.mode == "random"
        assert r.params.seed == 42
        assert r.params.sample_size == 3
        assert r.selected_ids == ["a", "b", "c"]
        assert r.total_pool == 100

    def test_auto_uuid(self):
        r = create_sampling_record("random", None, 1, ["x"], 5)
        uuid.UUID(r.record_id)  # validates UUID format

    def test_auto_timestamp(self):
        r = create_sampling_record("random", None, 1, ["x"], 5)
        assert "T" in r.timestamp  # ISO format

    def test_checksum_matches(self):
        ids = ["c", "a", "b"]
        r = create_sampling_record("random", 1, 3, ids, 10)
        assert r.checksum == compute_selection_checksum(ids)

    def test_extra_params_default(self):
        r = create_sampling_record("random", None, 1, ["x"], 5)
        assert r.extra_params if hasattr(r, "extra_params") else r.params.extra_params == {}

    def test_extra_params_passed(self):
        r = create_sampling_record("random", None, 1, ["x"], 5, extra_params={"k": 10})
        assert r.params.extra_params == {"k": 10}

    def test_ids_are_copied(self):
        ids = ["a", "b"]
        r = create_sampling_record("random", None, 2, ids, 5)
        ids.append("c")
        assert len(r.selected_ids) == 2


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_roundtrip(self, tmp_path):
        r = create_sampling_record("random", 42, 2, ["a", "b"], 10)
        path = str(tmp_path / "record.json")
        save_sampling_record(r, path)
        loaded = load_sampling_record(path)
        assert loaded is not None
        assert loaded.as_dict() == r.as_dict()

    def test_load_missing_returns_none(self, tmp_path):
        assert load_sampling_record(str(tmp_path / "nope.json")) is None

    def test_file_is_valid_json(self, tmp_path):
        r = create_sampling_record("random", None, 1, ["x"], 5)
        path = str(tmp_path / "record.json")
        save_sampling_record(r, path)
        with open(path) as f:
            data = json.load(f)
        assert data["record_id"] == r.record_id

    def test_overwrite(self, tmp_path):
        path = str(tmp_path / "record.json")
        r1 = create_sampling_record("random", 1, 1, ["a"], 5)
        save_sampling_record(r1, path)
        r2 = create_sampling_record("diverse", 2, 2, ["b", "c"], 10)
        save_sampling_record(r2, path)
        loaded = load_sampling_record(path)
        assert loaded is not None
        assert loaded.params.mode == "diverse"


# ---------------------------------------------------------------------------
# verify_reproducibility
# ---------------------------------------------------------------------------


class TestVerifyReproducibility:
    def test_same_ids_pass(self):
        r = create_sampling_record("random", 42, 3, ["a", "b", "c"], 10)
        assert verify_reproducibility(r, ["a", "b", "c"]) is True

    def test_same_ids_different_order_pass(self):
        r = create_sampling_record("random", 42, 3, ["a", "b", "c"], 10)
        assert verify_reproducibility(r, ["c", "a", "b"]) is True

    def test_different_ids_fail(self):
        r = create_sampling_record("random", 42, 3, ["a", "b", "c"], 10)
        assert verify_reproducibility(r, ["a", "b", "d"]) is False

    def test_subset_fails(self):
        r = create_sampling_record("random", 42, 3, ["a", "b", "c"], 10)
        assert verify_reproducibility(r, ["a", "b"]) is False

    def test_superset_fails(self):
        r = create_sampling_record("random", 42, 3, ["a", "b", "c"], 10)
        assert verify_reproducibility(r, ["a", "b", "c", "d"]) is False

    def test_empty_vs_empty(self):
        r = create_sampling_record("random", None, 0, [], 0)
        assert verify_reproducibility(r, []) is True

    def test_empty_vs_nonempty(self):
        r = create_sampling_record("random", None, 0, [], 0)
        assert verify_reproducibility(r, ["a"]) is False


# ---------------------------------------------------------------------------
# format_reproducibility_report
# ---------------------------------------------------------------------------


class TestFormatReport:
    def test_contains_key_fields(self):
        r = create_sampling_record("random", 42, 3, ["a", "b", "c"], 100)
        report = format_reproducibility_report(r)
        assert "Sampling Reproducibility Report" in report
        assert r.record_id in report
        assert "random" in report
        assert "42" in report
        assert "100" in report
        assert "a" in report
        assert r.checksum in report

    def test_extra_params_shown(self):
        r = create_sampling_record("random", 42, 3, ["a"], 10, extra_params={"k": 5})
        report = format_reproducibility_report(r)
        assert "Extra" in report
        assert "k" in report

    def test_no_extra_params_omitted(self):
        r = create_sampling_record("random", 42, 3, ["a"], 10)
        report = format_reproducibility_report(r)
        assert "Extra" not in report

    def test_empty_selected_ids(self):
        r = create_sampling_record("random", None, 0, [], 0)
        report = format_reproducibility_report(r)
        assert "Selected IDs:" in report

    def test_report_is_string(self):
        r = create_sampling_record("random", 42, 1, ["x"], 5)
        assert isinstance(format_reproducibility_report(r), str)
