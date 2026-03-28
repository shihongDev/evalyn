"""Tests for storage integrity_checks module."""

from __future__ import annotations

import pytest

from evalyn_sdk.storage.integrity_checks import (
    IntegrityCheck,
    IntegrityReport,
    check_data_types,
    check_duplicate_ids,
    check_orphan_references,
    check_required_fields,
    run_integrity_checks,
    suggest_fixes,
)


# ---------------------------------------------------------------------------
# check_orphan_references
# ---------------------------------------------------------------------------


class TestCheckOrphanReferences:
    def test_finds_orphans(self):
        records = [
            {"parent_id": "a"},
            {"parent_id": "b"},
            {"parent_id": "missing"},
        ]
        result = check_orphan_references(records, valid_ids={"a", "b"})
        assert not result.passed
        assert result.affected_rows == 1

    def test_no_orphans(self):
        records = [
            {"parent_id": "a"},
            {"parent_id": "b"},
        ]
        result = check_orphan_references(records, valid_ids={"a", "b"})
        assert result.passed
        assert result.affected_rows == 0

    def test_none_values_ignored(self):
        records = [
            {"parent_id": None},
            {"parent_id": "a"},
        ]
        result = check_orphan_references(records, valid_ids={"a"})
        assert result.passed

    def test_custom_id_field(self):
        records = [{"ref": "x"}]
        result = check_orphan_references(
            records, valid_ids={"y"}, id_field="ref"
        )
        assert not result.passed

    def test_empty_records(self):
        result = check_orphan_references([], valid_ids={"a"})
        assert result.passed

    def test_all_orphans(self):
        records = [{"parent_id": "x"}, {"parent_id": "y"}]
        result = check_orphan_references(records, valid_ids=set())
        assert not result.passed
        assert result.affected_rows == 2


# ---------------------------------------------------------------------------
# check_duplicate_ids
# ---------------------------------------------------------------------------


class TestCheckDuplicateIds:
    def test_finds_dupes(self):
        records = [{"id": "a"}, {"id": "a"}, {"id": "b"}]
        result = check_duplicate_ids(records)
        assert not result.passed
        assert result.affected_rows == 1  # one duplicate value

    def test_no_dupes(self):
        records = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
        result = check_duplicate_ids(records)
        assert result.passed

    def test_empty_records(self):
        result = check_duplicate_ids([])
        assert result.passed

    def test_custom_id_field(self):
        records = [{"key": 1}, {"key": 1}]
        result = check_duplicate_ids(records, id_field="key")
        assert not result.passed

    def test_none_ids_skipped(self):
        records = [{"id": None}, {"id": None}]
        result = check_duplicate_ids(records)
        assert result.passed

    def test_multiple_duplicate_values(self):
        records = [
            {"id": "a"}, {"id": "a"},
            {"id": "b"}, {"id": "b"},
        ]
        result = check_duplicate_ids(records)
        assert result.affected_rows == 2


# ---------------------------------------------------------------------------
# check_required_fields
# ---------------------------------------------------------------------------


class TestCheckRequiredFields:
    def test_all_present(self):
        records = [{"name": "x", "age": 1}]
        result = check_required_fields(records, required=["name", "age"])
        assert result.passed

    def test_missing_field(self):
        records = [{"name": "x"}]
        result = check_required_fields(records, required=["name", "age"])
        assert not result.passed
        assert result.affected_rows >= 1

    def test_none_value_counts_as_missing(self):
        records = [{"name": None}]
        result = check_required_fields(records, required=["name"])
        assert not result.passed

    def test_empty_records(self):
        result = check_required_fields([], required=["id"])
        assert result.passed

    def test_empty_required_list(self):
        records = [{"a": 1}]
        result = check_required_fields(records, required=[])
        assert result.passed

    def test_multiple_records_partial_missing(self):
        records = [
            {"name": "ok", "val": 1},
            {"name": "bad"},
        ]
        result = check_required_fields(records, required=["name", "val"])
        assert not result.passed
        assert result.affected_rows == 1


# ---------------------------------------------------------------------------
# check_data_types
# ---------------------------------------------------------------------------


class TestCheckDataTypes:
    def test_correct_types(self):
        records = [{"count": 5, "label": "x"}]
        result = check_data_types(
            records, field_types={"count": int, "label": str}
        )
        assert result.passed

    def test_wrong_type(self):
        records = [{"count": "five"}]
        result = check_data_types(records, field_types={"count": int})
        assert not result.passed

    def test_none_values_skipped(self):
        records = [{"count": None}]
        result = check_data_types(records, field_types={"count": int})
        assert result.passed

    def test_empty_records(self):
        result = check_data_types([], field_types={"x": int})
        assert result.passed

    def test_missing_field_skipped(self):
        records = [{"other": 1}]
        result = check_data_types(records, field_types={"count": int})
        assert result.passed

    def test_multiple_violations(self):
        records = [
            {"a": "wrong", "b": "wrong"},
        ]
        result = check_data_types(
            records, field_types={"a": int, "b": float}
        )
        assert not result.passed
        assert result.affected_rows == 2


# ---------------------------------------------------------------------------
# run_integrity_checks
# ---------------------------------------------------------------------------


class TestRunIntegrityChecks:
    def test_all_pass(self):
        records = [{"id": "a", "parent_id": "p1", "name": "x", "val": 1}]
        report = run_integrity_checks(
            records,
            valid_parent_ids={"p1"},
            required_fields=["name"],
            field_types={"val": int},
        )
        assert report.all_passed
        assert report.failed_checks == 0

    def test_mixed_results(self):
        records = [
            {"id": "a", "parent_id": "missing", "name": "x"},
        ]
        report = run_integrity_checks(
            records,
            valid_parent_ids={"p1"},
            required_fields=["name"],
        )
        assert not report.all_passed
        assert report.failed_checks >= 1

    def test_only_duplicate_check_when_no_params(self):
        records = [{"id": "a"}]
        report = run_integrity_checks(records)
        assert report.total_checks == 1
        check_names = [c.name for c in report.checks]
        assert "duplicate_ids" in check_names

    def test_empty_records(self):
        report = run_integrity_checks([])
        assert report.all_passed


# ---------------------------------------------------------------------------
# suggest_fixes
# ---------------------------------------------------------------------------


class TestSuggestFixes:
    def test_generates_suggestions_for_failures(self):
        report = IntegrityReport(
            checks=[
                IntegrityCheck("orphan_references", False, "bad", 2),
                IntegrityCheck("duplicate_ids", False, "bad", 1),
            ],
            all_passed=False,
            total_checks=2,
            failed_checks=2,
        )
        fixes = suggest_fixes(report)
        assert len(fixes) == 2
        assert any("orphan" in f for f in fixes)
        assert any("duplicate" in f.lower() or "Deduplicate" in f for f in fixes)

    def test_no_suggestions_when_all_pass(self):
        report = IntegrityReport(
            checks=[IntegrityCheck("duplicate_ids", True)],
            all_passed=True,
            total_checks=1,
            failed_checks=0,
        )
        fixes = suggest_fixes(report)
        assert fixes == []

    def test_required_fields_suggestion(self):
        report = IntegrityReport(
            checks=[IntegrityCheck("required_fields", False, "missing", 1)],
            all_passed=False,
            total_checks=1,
            failed_checks=1,
        )
        fixes = suggest_fixes(report)
        assert len(fixes) == 1
        assert "required" in fixes[0].lower() or "missing" in fixes[0].lower()

    def test_data_types_suggestion(self):
        report = IntegrityReport(
            checks=[IntegrityCheck("data_types", False, "wrong", 1)],
            all_passed=False,
            total_checks=1,
            failed_checks=1,
        )
        fixes = suggest_fixes(report)
        assert len(fixes) == 1
        assert "type" in fixes[0].lower()

    def test_unknown_check_name(self):
        report = IntegrityReport(
            checks=[IntegrityCheck("custom_check", False, "issue", 1)],
            all_passed=False,
            total_checks=1,
            failed_checks=1,
        )
        fixes = suggest_fixes(report)
        assert len(fixes) == 1
        assert "custom_check" in fixes[0]


# ---------------------------------------------------------------------------
# IntegrityReport format_text
# ---------------------------------------------------------------------------


class TestIntegrityReportFormatText:
    def test_contains_header(self):
        report = IntegrityReport(
            checks=[IntegrityCheck("duplicate_ids", True)],
            all_passed=True,
            total_checks=1,
            failed_checks=0,
        )
        text = report.format_text()
        assert "Integrity Report" in text
        assert "PASS" in text

    def test_shows_failures(self):
        report = IntegrityReport(
            checks=[IntegrityCheck("orphan_references", False, "2 bad", 2)],
            all_passed=False,
            total_checks=1,
            failed_checks=1,
        )
        text = report.format_text()
        assert "FAIL" in text
        assert "orphan_references" in text
        assert "affected rows: 2" in text

    def test_empty_report(self):
        text = IntegrityReport().format_text()
        assert "Total checks: 0" in text


# ---------------------------------------------------------------------------
# as_dict / from_dict roundtrips
# ---------------------------------------------------------------------------


class TestRoundtrips:
    def test_integrity_check_as_dict(self):
        check = IntegrityCheck("test", True, "ok", 0)
        d = check.as_dict()
        assert d["name"] == "test"
        assert d["passed"] is True

    def test_integrity_report_roundtrip(self):
        report = IntegrityReport(
            checks=[
                IntegrityCheck("dup", True, "ok", 0),
                IntegrityCheck("orphan", False, "bad", 3),
            ],
            all_passed=False,
            total_checks=2,
            failed_checks=1,
        )
        d = report.as_dict()
        restored = IntegrityReport.from_dict(d)
        assert len(restored.checks) == 2
        assert restored.all_passed is False
        assert restored.failed_checks == 1
        assert restored.checks[1].affected_rows == 3

    def test_empty_report_roundtrip(self):
        report = IntegrityReport()
        d = report.as_dict()
        restored = IntegrityReport.from_dict(d)
        assert restored.checks == []
        assert restored.all_passed is True
