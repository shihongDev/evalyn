"""Tests for data governance metadata module."""

from __future__ import annotations

import json

from evalyn_sdk.data_governance import (
    ComplianceFlag,
    DataGovernanceRecord,
    GovernanceReport,
    GovernanceTag,
    add_compliance_check,
    check_pii_compliance,
    create_governance_record,
    export_governance_report_json,
    format_governance_report,
    generate_governance_report,
    validate_tags,
)


# --- GovernanceTag ---


class TestGovernanceTag:
    def test_known_tags_are_strings(self):
        assert isinstance(GovernanceTag.PII_PRESENT, str)
        assert isinstance(GovernanceTag.INTERNAL_ONLY, str)
        assert isinstance(GovernanceTag.CUSTOMER_DATA, str)
        assert isinstance(GovernanceTag.SYNTHETIC, str)
        assert isinstance(GovernanceTag.PUBLIC, str)
        assert isinstance(GovernanceTag.ANONYMIZED, str)

    def test_all_contains_all_tags(self):
        assert GovernanceTag.PII_PRESENT in GovernanceTag.ALL
        assert GovernanceTag.INTERNAL_ONLY in GovernanceTag.ALL
        assert GovernanceTag.CUSTOMER_DATA in GovernanceTag.ALL
        assert GovernanceTag.SYNTHETIC in GovernanceTag.ALL
        assert GovernanceTag.PUBLIC in GovernanceTag.ALL
        assert GovernanceTag.ANONYMIZED in GovernanceTag.ALL
        assert len(GovernanceTag.ALL) == 6

    def test_all_is_frozenset(self):
        assert isinstance(GovernanceTag.ALL, frozenset)

    def test_tag_values_are_lowercase(self):
        for tag in GovernanceTag.ALL:
            assert tag == tag.lower()


# --- ComplianceFlag ---


class TestComplianceFlag:
    def test_create(self):
        flag = ComplianceFlag(
            flag_name="test_check",
            passed=True,
            checked_at="2026-01-01T00:00:00+00:00",
            details="all good",
        )
        assert flag.flag_name == "test_check"
        assert flag.passed is True
        assert flag.details == "all good"

    def test_as_dict(self):
        flag = ComplianceFlag(
            flag_name="check",
            passed=False,
            checked_at="2026-01-01T00:00:00+00:00",
            details="failed",
        )
        d = flag.as_dict()
        assert d["flag_name"] == "check"
        assert d["passed"] is False
        assert d["checked_at"] == "2026-01-01T00:00:00+00:00"
        assert d["details"] == "failed"

    def test_from_dict(self):
        d = {
            "flag_name": "check",
            "passed": True,
            "checked_at": "2026-01-01T00:00:00+00:00",
            "details": "ok",
        }
        flag = ComplianceFlag.from_dict(d)
        assert flag.flag_name == "check"
        assert flag.passed is True
        assert flag.details == "ok"

    def test_from_dict_missing_details_defaults_empty(self):
        d = {
            "flag_name": "check",
            "passed": True,
            "checked_at": "2026-01-01T00:00:00+00:00",
        }
        flag = ComplianceFlag.from_dict(d)
        assert flag.details == ""

    def test_roundtrip(self):
        flag = ComplianceFlag(
            flag_name="roundtrip",
            passed=False,
            checked_at="2026-03-01T12:00:00+00:00",
            details="test roundtrip",
        )
        restored = ComplianceFlag.from_dict(flag.as_dict())
        assert restored.flag_name == flag.flag_name
        assert restored.passed == flag.passed
        assert restored.checked_at == flag.checked_at
        assert restored.details == flag.details


# --- DataGovernanceRecord ---


class TestDataGovernanceRecord:
    def test_create(self):
        rec = DataGovernanceRecord(
            dataset_id="ds-1",
            tags=["pii_present"],
            compliance_flags=[],
            owner="alice",
            created_at="2026-01-01T00:00:00+00:00",
        )
        assert rec.dataset_id == "ds-1"
        assert rec.retention_days == 365
        assert rec.classification == "internal"

    def test_as_dict(self):
        flag = ComplianceFlag("check", True, "2026-01-01T00:00:00+00:00", "ok")
        rec = DataGovernanceRecord(
            dataset_id="ds-2",
            tags=["synthetic"],
            compliance_flags=[flag],
            owner="bob",
            created_at="2026-01-01T00:00:00+00:00",
            retention_days=90,
            classification="public",
        )
        d = rec.as_dict()
        assert d["dataset_id"] == "ds-2"
        assert d["tags"] == ["synthetic"]
        assert len(d["compliance_flags"]) == 1
        assert d["compliance_flags"][0]["flag_name"] == "check"
        assert d["retention_days"] == 90
        assert d["classification"] == "public"

    def test_from_dict(self):
        d = {
            "dataset_id": "ds-3",
            "tags": ["public"],
            "compliance_flags": [],
            "owner": "carol",
            "created_at": "2026-01-01T00:00:00+00:00",
            "retention_days": 180,
            "classification": "confidential",
        }
        rec = DataGovernanceRecord.from_dict(d)
        assert rec.dataset_id == "ds-3"
        assert rec.owner == "carol"
        assert rec.retention_days == 180
        assert rec.classification == "confidential"

    def test_from_dict_defaults(self):
        d = {
            "dataset_id": "ds-4",
            "created_at": "2026-01-01T00:00:00+00:00",
        }
        rec = DataGovernanceRecord.from_dict(d)
        assert rec.tags == []
        assert rec.compliance_flags == []
        assert rec.owner == ""
        assert rec.retention_days == 365
        assert rec.classification == "internal"

    def test_roundtrip(self):
        flag = ComplianceFlag("check", True, "2026-01-01T00:00:00+00:00", "ok")
        rec = DataGovernanceRecord(
            dataset_id="ds-5",
            tags=["pii_present", "internal_only"],
            compliance_flags=[flag],
            owner="dave",
            created_at="2026-01-01T00:00:00+00:00",
            retention_days=730,
            classification="restricted",
        )
        restored = DataGovernanceRecord.from_dict(rec.as_dict())
        assert restored.dataset_id == rec.dataset_id
        assert restored.tags == rec.tags
        assert len(restored.compliance_flags) == 1
        assert restored.retention_days == rec.retention_days

    def test_tags_list_is_copy(self):
        original_tags = ["synthetic"]
        rec = DataGovernanceRecord(
            dataset_id="ds-6",
            tags=original_tags,
            compliance_flags=[],
            owner="",
            created_at="2026-01-01T00:00:00+00:00",
        )
        d = rec.as_dict()
        d["tags"].append("public")
        assert len(rec.tags) == 1


# --- GovernanceReport ---


class TestGovernanceReport:
    def test_as_dict(self):
        report = GovernanceReport(
            records=[],
            summary={"total_records": 0},
            generated_at="2026-01-01T00:00:00+00:00",
        )
        d = report.as_dict()
        assert d["records"] == []
        assert d["summary"]["total_records"] == 0

    def test_from_dict(self):
        d = {
            "records": [],
            "summary": {"total_records": 0},
            "generated_at": "2026-01-01T00:00:00+00:00",
        }
        report = GovernanceReport.from_dict(d)
        assert report.records == []
        assert report.generated_at == "2026-01-01T00:00:00+00:00"

    def test_roundtrip_with_records(self):
        rec = DataGovernanceRecord(
            dataset_id="ds-7",
            tags=["synthetic"],
            compliance_flags=[],
            owner="eve",
            created_at="2026-01-01T00:00:00+00:00",
        )
        report = GovernanceReport(
            records=[rec],
            summary={"total_records": 1},
            generated_at="2026-01-01T00:00:00+00:00",
        )
        restored = GovernanceReport.from_dict(report.as_dict())
        assert len(restored.records) == 1
        assert restored.records[0].dataset_id == "ds-7"


# --- create_governance_record ---


class TestCreateGovernanceRecord:
    def test_basic(self):
        rec = create_governance_record("ds-10", ["synthetic"])
        assert rec.dataset_id == "ds-10"
        assert rec.tags == ["synthetic"]
        assert rec.compliance_flags == []
        assert rec.classification == "internal"
        assert rec.owner == ""
        assert rec.created_at  # non-empty

    def test_with_owner_and_classification(self):
        rec = create_governance_record(
            "ds-11",
            ["public"],
            owner="alice",
            classification="public",
        )
        assert rec.owner == "alice"
        assert rec.classification == "public"

    def test_tags_are_copied(self):
        tags = ["synthetic"]
        rec = create_governance_record("ds-12", tags)
        tags.append("public")
        assert rec.tags == ["synthetic"]

    def test_auto_timestamp_is_iso(self):
        rec = create_governance_record("ds-13", [])
        # Should be parseable as ISO format
        assert "T" in rec.created_at


# --- add_compliance_check ---


class TestAddComplianceCheck:
    def test_adds_flag(self):
        rec = create_governance_record("ds-20", ["synthetic"])
        rec2 = add_compliance_check(rec, "test_check", True, "passed")
        assert len(rec2.compliance_flags) == 1
        assert rec2.compliance_flags[0].flag_name == "test_check"
        assert rec2.compliance_flags[0].passed is True

    def test_original_unchanged(self):
        rec = create_governance_record("ds-21", ["synthetic"])
        add_compliance_check(rec, "test_check", True)
        assert len(rec.compliance_flags) == 0

    def test_multiple_flags(self):
        rec = create_governance_record("ds-22", [])
        rec = add_compliance_check(rec, "check_a", True)
        rec = add_compliance_check(rec, "check_b", False, "issue")
        assert len(rec.compliance_flags) == 2
        assert rec.compliance_flags[0].flag_name == "check_a"
        assert rec.compliance_flags[1].flag_name == "check_b"
        assert rec.compliance_flags[1].passed is False

    def test_preserves_record_fields(self):
        rec = create_governance_record(
            "ds-23", ["pii_present"], owner="alice", classification="restricted"
        )
        rec2 = add_compliance_check(rec, "check", True)
        assert rec2.dataset_id == "ds-23"
        assert rec2.owner == "alice"
        assert rec2.classification == "restricted"
        assert rec2.tags == ["pii_present"]


# --- validate_tags ---


class TestValidateTags:
    def test_all_valid(self):
        assert validate_tags(["synthetic", "public"]) == []

    def test_some_invalid(self):
        result = validate_tags(["synthetic", "bogus_tag", "public"])
        assert result == ["bogus_tag"]

    def test_all_invalid(self):
        result = validate_tags(["foo", "bar"])
        assert result == ["foo", "bar"]

    def test_empty_list(self):
        assert validate_tags([]) == []

    def test_every_known_tag_is_valid(self):
        assert validate_tags(list(GovernanceTag.ALL)) == []


# --- check_pii_compliance ---


class TestCheckPiiCompliance:
    def test_no_pii_tag(self):
        rec = create_governance_record("ds-30", ["synthetic"])
        flag = check_pii_compliance(rec)
        assert flag.passed is True
        assert flag.flag_name == "pii_compliance"
        assert "No PII" in flag.details

    def test_pii_with_internal_classification(self):
        rec = create_governance_record("ds-31", ["pii_present"])
        flag = check_pii_compliance(rec)
        assert flag.passed is True
        assert "internal" in flag.details

    def test_pii_with_public_classification_fails(self):
        rec = create_governance_record(
            "ds-32", ["pii_present"], classification="public"
        )
        flag = check_pii_compliance(rec)
        assert flag.passed is False
        assert "public" in flag.details

    def test_pii_with_restricted_classification(self):
        rec = create_governance_record(
            "ds-33", ["pii_present"], classification="restricted"
        )
        flag = check_pii_compliance(rec)
        assert flag.passed is True

    def test_flag_has_timestamp(self):
        rec = create_governance_record("ds-34", [])
        flag = check_pii_compliance(rec)
        assert flag.checked_at  # non-empty


# --- generate_governance_report ---


class TestGenerateGovernanceReport:
    def test_empty_records(self):
        report = generate_governance_report([])
        assert report.summary["total_records"] == 0
        assert report.summary["tag_counts"] == {}
        assert report.records == []

    def test_single_record(self):
        rec = create_governance_record("ds-40", ["synthetic"], owner="alice")
        report = generate_governance_report([rec])
        assert report.summary["total_records"] == 1
        assert report.summary["tag_counts"]["synthetic"] == 1

    def test_multiple_records_tag_counting(self):
        r1 = create_governance_record("ds-41", ["synthetic", "public"])
        r2 = create_governance_record("ds-42", ["synthetic"])
        report = generate_governance_report([r1, r2])
        assert report.summary["tag_counts"]["synthetic"] == 2
        assert report.summary["tag_counts"]["public"] == 1

    def test_classification_counting(self):
        r1 = create_governance_record("ds-43", [], classification="internal")
        r2 = create_governance_record("ds-44", [], classification="public")
        r3 = create_governance_record("ds-45", [], classification="internal")
        report = generate_governance_report([r1, r2, r3])
        assert report.summary["classification_counts"]["internal"] == 2
        assert report.summary["classification_counts"]["public"] == 1

    def test_compliance_flag_counting(self):
        rec = create_governance_record("ds-46", [])
        rec = add_compliance_check(rec, "check_a", True)
        rec = add_compliance_check(rec, "check_b", False)
        report = generate_governance_report([rec])
        assert report.summary["total_compliance_checks"] == 2
        assert report.summary["passed_checks"] == 1
        assert report.summary["failed_checks"] == 1

    def test_report_has_timestamp(self):
        report = generate_governance_report([])
        assert report.generated_at
        assert "T" in report.generated_at


# --- format_governance_report ---


class TestFormatGovernanceReport:
    def test_empty_report(self):
        report = generate_governance_report([])
        text = format_governance_report(report)
        assert "Data Governance Report" in text
        assert "Total records: 0" in text

    def test_includes_dataset_info(self):
        rec = create_governance_record(
            "ds-50", ["synthetic"], owner="alice"
        )
        report = generate_governance_report([rec])
        text = format_governance_report(report)
        assert "ds-50" in text
        assert "alice" in text
        assert "synthetic" in text

    def test_includes_compliance_status(self):
        rec = create_governance_record("ds-51", [])
        rec = add_compliance_check(rec, "quality", True, "looks good")
        rec = add_compliance_check(rec, "security", False, "needs fix")
        report = generate_governance_report([rec])
        text = format_governance_report(report)
        assert "[PASS]" in text
        assert "[FAIL]" in text
        assert "quality" in text
        assert "security" in text


# --- export_governance_report_json ---


class TestExportGovernanceReportJson:
    def test_valid_json(self):
        report = generate_governance_report([])
        raw = export_governance_report_json(report)
        parsed = json.loads(raw)
        assert "records" in parsed
        assert "summary" in parsed
        assert "generated_at" in parsed

    def test_records_included(self):
        rec = create_governance_record("ds-60", ["public"])
        report = generate_governance_report([rec])
        raw = export_governance_report_json(report)
        parsed = json.loads(raw)
        assert len(parsed["records"]) == 1
        assert parsed["records"][0]["dataset_id"] == "ds-60"

    def test_roundtrip_through_json(self):
        rec = create_governance_record("ds-61", ["synthetic"], owner="bob")
        rec = add_compliance_check(rec, "check", True, "ok")
        report = generate_governance_report([rec])
        raw = export_governance_report_json(report)
        parsed = json.loads(raw)
        restored = GovernanceReport.from_dict(parsed)
        assert restored.summary["total_records"] == 1
        assert len(restored.records) == 1
        assert restored.records[0].dataset_id == "ds-61"
