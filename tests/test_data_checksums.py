"""Tests for evalyn_sdk.storage.data_checksums.

Run with: uv run pytest tests/test_data_checksums.py -v
"""

from __future__ import annotations

import pytest

from evalyn_sdk.storage.data_checksums import (
    ChecksumRecord,
    ChecksumReport,
    add_checksum,
    compute_checksum,
    find_corrupted,
    strip_checksums,
    verify_batch,
    verify_checksum,
)


# ---------------------------------------------------------------------------
# compute_checksum
# ---------------------------------------------------------------------------


class TestComputeChecksum:
    def test_deterministic(self):
        data = {"a": 1, "b": 2}
        assert compute_checksum(data) == compute_checksum(data)

    def test_key_order_irrelevant(self):
        d1 = {"z": 1, "a": 2}
        d2 = {"a": 2, "z": 1}
        assert compute_checksum(d1) == compute_checksum(d2)

    def test_different_data_different_hash(self):
        assert compute_checksum({"x": 1}) != compute_checksum({"x": 2})

    def test_returns_hex_string(self):
        h = compute_checksum({"k": "v"})
        assert isinstance(h, str)
        assert len(h) == 64  # sha256 hex digest

    def test_nested_dict(self):
        data = {"outer": {"inner": [1, 2, 3]}}
        h = compute_checksum(data)
        assert len(h) == 64

    def test_empty_dict(self):
        h = compute_checksum({})
        assert isinstance(h, str)
        assert len(h) == 64


# ---------------------------------------------------------------------------
# add_checksum / verify_checksum
# ---------------------------------------------------------------------------


class TestAddAndVerify:
    def test_add_checksum_adds_field(self):
        rec = {"id": "r1", "value": 42}
        result = add_checksum(rec)
        assert "_checksum" in result
        assert "id" in result
        assert "value" in result

    def test_add_checksum_does_not_mutate_original(self):
        rec = {"id": "r1", "value": 42}
        add_checksum(rec)
        assert "_checksum" not in rec

    def test_verify_valid(self):
        rec = add_checksum({"id": "r1", "score": 0.9})
        assert verify_checksum(rec) is True

    def test_verify_tampered(self):
        rec = add_checksum({"id": "r1", "score": 0.9})
        rec["score"] = 0.1
        assert verify_checksum(rec) is False

    def test_verify_missing_checksum(self):
        assert verify_checksum({"id": "r1"}) is False

    def test_add_replaces_existing_checksum(self):
        rec = add_checksum({"id": "r1", "v": 1})
        cs1 = rec["_checksum"]
        rec["v"] = 2
        rec2 = add_checksum(rec)
        assert rec2["_checksum"] != cs1
        assert verify_checksum(rec2) is True

    def test_nested_dict_roundtrip(self):
        rec = add_checksum({"id": "r1", "meta": {"tags": ["a", "b"]}})
        assert verify_checksum(rec) is True


# ---------------------------------------------------------------------------
# verify_batch
# ---------------------------------------------------------------------------


class TestVerifyBatch:
    def test_all_valid(self):
        records = [add_checksum({"id": str(i), "v": i}) for i in range(5)]
        report = verify_batch(records)
        assert report.total_records == 5
        assert report.verified == 5
        assert report.failed == 0
        assert report.missing == 0
        assert report.integrity_rate == 1.0

    def test_mixed_results(self):
        good = add_checksum({"id": "1", "v": 1})
        bad = add_checksum({"id": "2", "v": 2})
        bad["v"] = 999
        no_cs = {"id": "3", "v": 3}
        report = verify_batch([good, bad, no_cs])
        assert report.total_records == 3
        assert report.verified == 1
        assert report.failed == 1
        assert report.missing == 1

    def test_empty_list(self):
        report = verify_batch([])
        assert report.total_records == 0
        assert report.integrity_rate == 1.0

    def test_all_missing(self):
        records = [{"id": "1"}, {"id": "2"}]
        report = verify_batch(records)
        assert report.missing == 2
        assert report.verified == 0

    def test_integrity_rate_calculation(self):
        good = [add_checksum({"id": str(i), "v": i}) for i in range(3)]
        bad = add_checksum({"id": "bad", "v": 0})
        bad["v"] = -1
        report = verify_batch(good + [bad])
        assert report.integrity_rate == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# find_corrupted
# ---------------------------------------------------------------------------


class TestFindCorrupted:
    def test_returns_corrupted_ids(self):
        good = add_checksum({"id": "ok", "v": 1})
        bad = add_checksum({"id": "broken", "v": 2})
        bad["v"] = -1
        result = find_corrupted([good, bad])
        assert result == ["broken"]

    def test_no_corrupted(self):
        records = [add_checksum({"id": str(i), "v": i}) for i in range(3)]
        assert find_corrupted(records) == []

    def test_skips_records_without_checksum(self):
        records = [{"id": "no_cs", "v": 1}]
        assert find_corrupted(records) == []

    def test_uses_record_id_fallback(self):
        bad = add_checksum({"record_id": "rid", "v": 1})
        bad["v"] = -1
        result = find_corrupted([bad])
        assert result == ["rid"]

    def test_unknown_id(self):
        bad = add_checksum({"v": 1})
        bad["v"] = -1
        result = find_corrupted([bad])
        assert result == ["unknown"]


# ---------------------------------------------------------------------------
# strip_checksums
# ---------------------------------------------------------------------------


class TestStripChecksums:
    def test_removes_checksum_field(self):
        records = [add_checksum({"id": "1", "v": 1})]
        stripped = strip_checksums(records)
        assert "_checksum" not in stripped[0]
        assert stripped[0]["id"] == "1"

    def test_does_not_mutate_original(self):
        records = [add_checksum({"id": "1", "v": 1})]
        strip_checksums(records)
        assert "_checksum" in records[0]

    def test_empty_list(self):
        assert strip_checksums([]) == []

    def test_records_without_checksum(self):
        records = [{"id": "1", "v": 1}]
        stripped = strip_checksums(records)
        assert stripped == records


# ---------------------------------------------------------------------------
# ChecksumReport
# ---------------------------------------------------------------------------


class TestChecksumReport:
    def test_format_text(self):
        report = ChecksumReport(
            total_records=10, verified=8, failed=1, missing=1, integrity_rate=0.8
        )
        text = report.format_text()
        assert "Checksum Report" in text
        assert "Total records: 10" in text
        assert "Verified: 8" in text
        assert "Failed: 1" in text
        assert "Missing: 1" in text
        assert "80.0%" in text

    def test_as_dict(self):
        report = ChecksumReport(total_records=5, verified=5)
        d = report.as_dict()
        assert d["total_records"] == 5
        assert d["verified"] == 5

    def test_from_dict(self):
        d = {
            "total_records": 3,
            "verified": 2,
            "failed": 1,
            "missing": 0,
            "integrity_rate": 0.667,
        }
        report = ChecksumReport.from_dict(d)
        assert report.total_records == 3
        assert report.failed == 1

    def test_roundtrip(self):
        original = ChecksumReport(
            total_records=10,
            verified=7,
            failed=2,
            missing=1,
            integrity_rate=0.7,
        )
        restored = ChecksumReport.from_dict(original.as_dict())
        assert restored.as_dict() == original.as_dict()


# ---------------------------------------------------------------------------
# ChecksumRecord
# ---------------------------------------------------------------------------


class TestChecksumRecord:
    def test_as_dict(self):
        rec = ChecksumRecord(record_id="r1", checksum="abc123")
        d = rec.as_dict()
        assert d["record_id"] == "r1"
        assert d["algorithm"] == "sha256"
        assert d["verified"] is True

    def test_from_dict(self):
        d = {"record_id": "r2", "checksum": "def456", "algorithm": "md5"}
        rec = ChecksumRecord.from_dict(d)
        assert rec.record_id == "r2"
        assert rec.algorithm == "md5"

    def test_roundtrip(self):
        original = ChecksumRecord(
            record_id="r1", checksum="abc", algorithm="sha256", verified=False
        )
        restored = ChecksumRecord.from_dict(original.as_dict())
        assert restored.as_dict() == original.as_dict()
