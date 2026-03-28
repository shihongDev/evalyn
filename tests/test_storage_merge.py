"""
Tests for evalyn_sdk.storage.merge.

Run with: uv run pytest tests/test_storage_merge.py -v
"""

from __future__ import annotations

import pytest

from evalyn_sdk.storage.merge import (
    MergeConfig,
    MergeConflict,
    MergeResult,
    detect_conflicts,
    merge_records,
    plan_merge,
    resolve_conflict,
    validate_merge,
)


# ---------------------------------------------------------------------------
# MergeConflict
# ---------------------------------------------------------------------------


class TestMergeConflict:
    def test_as_dict(self):
        c = MergeConflict(
            table="runs",
            record_id="r1",
            field="status",
            value_a="pass",
            value_b="fail",
            resolution="keep_a",
        )
        d = c.as_dict()
        assert d["table"] == "runs"
        assert d["record_id"] == "r1"
        assert d["field"] == "status"
        assert d["resolution"] == "keep_a"

    def test_defaults(self):
        c = MergeConflict(table="t", record_id="1")
        assert c.field == ""
        assert c.value_a == ""
        assert c.value_b == ""
        assert c.resolution == ""


# ---------------------------------------------------------------------------
# MergeConfig
# ---------------------------------------------------------------------------


class TestMergeConfig:
    def test_as_dict(self):
        cfg = MergeConfig(strategy="keep_a", tables=["runs", "metrics"])
        d = cfg.as_dict()
        assert d["strategy"] == "keep_a"
        assert d["tables"] == ["runs", "metrics"]

    def test_from_dict(self):
        cfg = MergeConfig.from_dict({"strategy": "keep_b", "tables": ["t1"]})
        assert cfg.strategy == "keep_b"
        assert cfg.tables == ["t1"]

    def test_from_dict_defaults(self):
        cfg = MergeConfig.from_dict({})
        assert cfg.strategy == "keep_newest"
        assert cfg.tables == []

    def test_roundtrip(self):
        cfg = MergeConfig(strategy="fail_on_conflict", tables=["a", "b"])
        restored = MergeConfig.from_dict(cfg.as_dict())
        assert restored.strategy == cfg.strategy
        assert restored.tables == cfg.tables


# ---------------------------------------------------------------------------
# MergeResult
# ---------------------------------------------------------------------------


class TestMergeResult:
    def test_as_dict(self):
        r = MergeResult(merged_count=10, conflict_count=1, skipped_count=0)
        d = r.as_dict()
        assert d["merged_count"] == 10
        assert d["conflict_count"] == 1

    def test_from_dict(self):
        data = {
            "merged_count": 5,
            "conflict_count": 2,
            "conflicts": [
                {"table": "t", "record_id": "1", "field": "x", "value_a": "a", "value_b": "b", "resolution": "keep_a"},
            ],
            "skipped_count": 1,
        }
        r = MergeResult.from_dict(data)
        assert r.merged_count == 5
        assert len(r.conflicts) == 1
        assert r.conflicts[0].field == "x"

    def test_from_dict_defaults(self):
        r = MergeResult.from_dict({})
        assert r.merged_count == 0
        assert r.conflicts == []

    def test_roundtrip(self):
        r = MergeResult(
            merged_count=3,
            conflict_count=1,
            conflicts=[MergeConflict("t", "1", "f", "a", "b", "keep_a")],
            skipped_count=0,
        )
        restored = MergeResult.from_dict(r.as_dict())
        assert restored.merged_count == r.merged_count
        assert len(restored.conflicts) == 1

    def test_format_text(self):
        r = MergeResult(
            merged_count=10,
            conflict_count=1,
            conflicts=[
                MergeConflict("runs", "r1", "status", "pass", "fail", "keep_a"),
            ],
            skipped_count=2,
        )
        text = r.format_text()
        assert "Merge Result" in text
        assert "Merged records: 10" in text
        assert "Conflicts: 1" in text
        assert "Skipped: 2" in text
        assert "runs" in text
        assert "keep_a" in text

    def test_format_text_no_conflicts(self):
        r = MergeResult(merged_count=5)
        text = r.format_text()
        assert "Merge Result" in text
        assert "Conflicts: 0" in text


# ---------------------------------------------------------------------------
# detect_conflicts
# ---------------------------------------------------------------------------


class TestDetectConflicts:
    def test_finds_conflicts(self):
        a = [{"id": "1", "name": "Alice", "score": 90}]
        b = [{"id": "1", "name": "Alice", "score": 95}]
        conflicts = detect_conflicts(a, b)
        assert len(conflicts) == 1
        assert conflicts[0].field == "score"
        assert conflicts[0].value_a == "90"
        assert conflicts[0].value_b == "95"

    def test_no_conflicts_different_ids(self):
        a = [{"id": "1", "name": "Alice"}]
        b = [{"id": "2", "name": "Bob"}]
        conflicts = detect_conflicts(a, b)
        assert conflicts == []

    def test_no_conflicts_same_values(self):
        a = [{"id": "1", "name": "Alice"}]
        b = [{"id": "1", "name": "Alice"}]
        conflicts = detect_conflicts(a, b)
        assert conflicts == []

    def test_empty_records(self):
        assert detect_conflicts([], []) == []
        assert detect_conflicts([{"id": "1"}], []) == []
        assert detect_conflicts([], [{"id": "1"}]) == []

    def test_multiple_field_conflicts(self):
        a = [{"id": "1", "x": "a", "y": "b"}]
        b = [{"id": "1", "x": "c", "y": "d"}]
        conflicts = detect_conflicts(a, b)
        assert len(conflicts) == 2
        fields = {c.field for c in conflicts}
        assert fields == {"x", "y"}

    def test_custom_id_field(self):
        a = [{"uid": "1", "val": "old"}]
        b = [{"uid": "1", "val": "new"}]
        conflicts = detect_conflicts(a, b, id_field="uid")
        assert len(conflicts) == 1


# ---------------------------------------------------------------------------
# resolve_conflict
# ---------------------------------------------------------------------------


class TestResolveConflict:
    def test_keep_a(self):
        c = MergeConflict("t", "1", "f", "a", "b")
        resolved = resolve_conflict(c, "keep_a")
        assert resolved.resolution == "keep_a"

    def test_keep_b(self):
        c = MergeConflict("t", "1", "f", "a", "b")
        resolved = resolve_conflict(c, "keep_b")
        assert resolved.resolution == "keep_b"

    def test_keep_newest_defaults_to_keep_a(self):
        c = MergeConflict("t", "1", "f", "a", "b")
        resolved = resolve_conflict(c, "keep_newest")
        assert resolved.resolution == "keep_a"

    def test_preserves_fields(self):
        c = MergeConflict("runs", "r1", "status", "pass", "fail")
        resolved = resolve_conflict(c, "keep_a")
        assert resolved.table == "runs"
        assert resolved.record_id == "r1"
        assert resolved.field == "status"
        assert resolved.value_a == "pass"
        assert resolved.value_b == "fail"


# ---------------------------------------------------------------------------
# merge_records
# ---------------------------------------------------------------------------


class TestMergeRecords:
    def test_no_conflicts(self):
        a = [{"id": "1", "name": "Alice"}]
        b = [{"id": "2", "name": "Bob"}]
        config = MergeConfig(strategy="keep_a")
        merged, result = merge_records(a, b, config)
        assert result.conflict_count == 0
        assert result.merged_count == 2
        ids = {r["id"] for r in merged}
        assert ids == {"1", "2"}

    def test_with_conflicts_keep_a(self):
        a = [{"id": "1", "val": "old"}]
        b = [{"id": "1", "val": "new"}]
        config = MergeConfig(strategy="keep_a")
        merged, result = merge_records(a, b, config)
        assert result.conflict_count == 1
        # keep_a: value from a should win
        rec = merged[0]
        assert rec["val"] == "old"

    def test_with_conflicts_keep_b(self):
        a = [{"id": "1", "val": "old"}]
        b = [{"id": "1", "val": "new"}]
        config = MergeConfig(strategy="keep_b")
        merged, result = merge_records(a, b, config)
        assert result.conflict_count == 1
        rec = merged[0]
        assert rec["val"] == "new"

    def test_fail_on_conflict(self):
        a = [{"id": "1", "val": "old"}]
        b = [{"id": "1", "val": "new"}]
        config = MergeConfig(strategy="fail_on_conflict")
        merged, result = merge_records(a, b, config)
        assert result.conflict_count > 0
        assert result.skipped_count == 1

    def test_empty_records(self):
        config = MergeConfig()
        merged, result = merge_records([], [], config)
        assert merged == []
        assert result.merged_count == 0

    def test_one_sided_a_only(self):
        a = [{"id": "1", "val": "x"}]
        config = MergeConfig()
        merged, result = merge_records(a, [], config)
        assert result.merged_count == 1
        assert result.conflict_count == 0

    def test_one_sided_b_only(self):
        b = [{"id": "1", "val": "x"}]
        config = MergeConfig()
        merged, result = merge_records([], b, config)
        assert result.merged_count == 1
        assert result.conflict_count == 0

    def test_identical_records_no_conflict(self):
        rec = {"id": "1", "val": "same"}
        config = MergeConfig()
        merged, result = merge_records([rec], [dict(rec)], config)
        assert result.conflict_count == 0
        assert result.merged_count == 1


# ---------------------------------------------------------------------------
# plan_merge
# ---------------------------------------------------------------------------


class TestPlanMerge:
    def test_shows_tables(self):
        a = {"runs": 100, "metrics": 200}
        b = {"runs": 50, "spans": 300}
        plan = plan_merge(a, b)
        assert plan["total_tables"] == 3
        table_names = [t["table"] for t in plan["tables"]]
        assert "runs" in table_names
        assert "metrics" in table_names
        assert "spans" in table_names

    def test_only_in_a(self):
        plan = plan_merge({"t1": 10}, {})
        assert plan["tables"][0]["only_in_a"] is True
        assert plan["tables"][0]["only_in_b"] is False

    def test_only_in_b(self):
        plan = plan_merge({}, {"t1": 10})
        assert plan["tables"][0]["only_in_b"] is True
        assert plan["tables"][0]["only_in_a"] is False

    def test_estimated_merged(self):
        plan = plan_merge({"t": 10}, {"t": 20})
        assert plan["tables"][0]["estimated_merged"] == 30

    def test_empty(self):
        plan = plan_merge({}, {})
        assert plan["total_tables"] == 0
        assert plan["tables"] == []


# ---------------------------------------------------------------------------
# validate_merge
# ---------------------------------------------------------------------------


class TestValidateMerge:
    def test_valid_same_schema(self):
        a = [{"id": "1", "name": "Alice"}]
        b = [{"id": "2", "name": "Bob"}]
        valid, errors = validate_merge(a, b)
        assert valid is True
        assert errors == []

    def test_invalid_different_schema(self):
        a = [{"id": "1", "name": "Alice"}]
        b = [{"id": "2", "email": "bob@example.com"}]
        valid, errors = validate_merge(a, b)
        assert valid is False
        assert len(errors) > 0

    def test_missing_id_field(self):
        a = [{"name": "Alice"}]
        b = [{"id": "2", "name": "Bob"}]
        valid, errors = validate_merge(a, b)
        assert valid is False
        assert any("id" in e.lower() or "ID" in e for e in errors)

    def test_empty_records(self):
        valid, errors = validate_merge([], [])
        assert valid is True

    def test_one_side_empty(self):
        valid, errors = validate_merge([{"id": "1"}], [])
        assert valid is True
        valid2, errors2 = validate_merge([], [{"id": "1"}])
        assert valid2 is True

    def test_custom_id_field(self):
        a = [{"uid": "1", "val": "x"}]
        b = [{"uid": "2", "val": "y"}]
        valid, errors = validate_merge(a, b, id_field="uid")
        assert valid is True
