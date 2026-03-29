"""Tests for the embedding PII safety check module."""

from __future__ import annotations

import sys
from pathlib import Path

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.embedding_pii_check import (
    DEFAULT_PII_PATTERNS,
    PIIDetectionResult,
    PIIPattern,
    PIISafetyReport,
    assess_embedding_risk,
    check_dataset_pii_safety,
    detect_pii,
    format_pii_report,
    suggest_mitigations,
)


# ---------------------------------------------------------------------------
# PIIPattern dataclass
# ---------------------------------------------------------------------------


class TestPIIPattern:
    def test_as_dict(self):
        p = PIIPattern("email", r".+@.+", "Email", "high")
        d = p.as_dict()
        assert d["pattern_name"] == "email"
        assert d["severity"] == "high"

    def test_from_dict(self):
        d = {"pattern_name": "phone", "regex": r"\d+", "description": "Phone", "severity": "medium"}
        p = PIIPattern.from_dict(d)
        assert p.pattern_name == "phone"
        assert p.severity == "medium"

    def test_roundtrip(self):
        original = PIIPattern("ssn", r"\d{3}-\d{2}-\d{4}", "SSN", "high")
        restored = PIIPattern.from_dict(original.as_dict())
        assert restored.pattern_name == original.pattern_name
        assert restored.regex == original.regex
        assert restored.severity == original.severity


# ---------------------------------------------------------------------------
# PIIDetectionResult dataclass
# ---------------------------------------------------------------------------


class TestPIIDetectionResult:
    def test_as_dict(self):
        r = PIIDetectionResult(
            text_id="t1", pii_found=["email"], pii_count=1,
            has_embedding=True, risk_level="critical",
            recommendations=["Redact PII"],
        )
        d = r.as_dict()
        assert d["text_id"] == "t1"
        assert d["pii_found"] == ["email"]
        assert d["has_embedding"] is True

    def test_from_dict(self):
        d = {
            "text_id": "t2", "pii_found": ["phone", "ssn"], "pii_count": 2,
            "has_embedding": False, "risk_level": "high",
            "recommendations": [],
        }
        r = PIIDetectionResult.from_dict(d)
        assert r.pii_count == 2
        assert r.risk_level == "high"

    def test_from_dict_defaults(self):
        r = PIIDetectionResult.from_dict({"text_id": "x"})
        assert r.pii_count == 0
        assert r.risk_level == "none"
        assert r.pii_found == []

    def test_roundtrip(self):
        original = PIIDetectionResult(
            text_id="id1", pii_found=["email", "phone"], pii_count=2,
            has_embedding=True, risk_level="critical",
            recommendations=["Fix it"],
        )
        restored = PIIDetectionResult.from_dict(original.as_dict())
        assert restored.text_id == original.text_id
        assert restored.pii_found == original.pii_found
        assert restored.recommendations == original.recommendations


# ---------------------------------------------------------------------------
# PIISafetyReport dataclass
# ---------------------------------------------------------------------------


class TestPIISafetyReport:
    def test_as_dict(self):
        report = PIISafetyReport(
            results=[], total_items=0, items_with_pii=0,
            items_at_risk=0, summary={},
        )
        d = report.as_dict()
        assert d["total_items"] == 0
        assert d["results"] == []

    def test_from_dict_with_nested_results(self):
        d = {
            "results": [
                {"text_id": "a", "pii_found": ["email"], "pii_count": 1,
                 "has_embedding": True, "risk_level": "critical",
                 "recommendations": []},
            ],
            "total_items": 1, "items_with_pii": 1, "items_at_risk": 1,
            "summary": {"risk_distribution": {"critical": 1}},
        }
        report = PIISafetyReport.from_dict(d)
        assert len(report.results) == 1
        assert report.results[0].text_id == "a"

    def test_roundtrip(self):
        r = PIIDetectionResult("id1", ["ssn"], 1, False, "high", ["Remove it"])
        original = PIISafetyReport(
            results=[r], total_items=1, items_with_pii=1,
            items_at_risk=1, summary={"key": "val"},
        )
        restored = PIISafetyReport.from_dict(original.as_dict())
        assert restored.total_items == 1
        assert restored.results[0].pii_found == ["ssn"]


# ---------------------------------------------------------------------------
# DEFAULT_PII_PATTERNS
# ---------------------------------------------------------------------------


class TestDefaultPatterns:
    def test_has_expected_patterns(self):
        names = {p.pattern_name for p in DEFAULT_PII_PATTERNS}
        for expected in ("email", "phone", "ssn", "credit_card", "ip_address", "date_of_birth"):
            assert expected in names

    def test_all_have_valid_severity(self):
        for p in DEFAULT_PII_PATTERNS:
            assert p.severity in ("high", "medium", "low")


# ---------------------------------------------------------------------------
# detect_pii
# ---------------------------------------------------------------------------


class TestDetectPII:
    def test_email(self):
        found = detect_pii("Contact us at user@example.com for details")
        assert "email" in found

    def test_phone(self):
        found = detect_pii("Call 555-123-4567 now")
        assert "phone" in found

    def test_phone_with_parens(self):
        found = detect_pii("Call (555) 123-4567")
        assert "phone" in found

    def test_ssn(self):
        found = detect_pii("SSN: 123-45-6789")
        assert "ssn" in found

    def test_ssn_no_dashes(self):
        found = detect_pii("SSN: 123456789")
        assert "ssn" in found

    def test_credit_card(self):
        found = detect_pii("Card: 4111 1111 1111 1111")
        assert "credit_card" in found

    def test_ip_address(self):
        found = detect_pii("Server at 192.168.1.1")
        assert "ip_address" in found

    def test_date_of_birth_mmddyyyy(self):
        found = detect_pii("DOB: 01/15/1990")
        assert "date_of_birth" in found

    def test_date_of_birth_yyyy_mm_dd(self):
        found = detect_pii("Born 1990-01-15")
        assert "date_of_birth" in found

    def test_no_pii(self):
        found = detect_pii("The quick brown fox jumps over the lazy dog")
        assert found == []

    def test_multiple_patterns(self):
        text = "Email: a@b.com, SSN: 123-45-6789"
        found = detect_pii(text)
        assert "email" in found
        assert "ssn" in found

    def test_custom_patterns(self):
        custom = [PIIPattern("custom_id", r"ID-\d{4}", "Custom ID", "low")]
        found = detect_pii("Reference: ID-1234", patterns=custom)
        assert found == ["custom_id"]

    def test_custom_patterns_no_match(self):
        custom = [PIIPattern("custom_id", r"ID-\d{4}", "Custom ID", "low")]
        found = detect_pii("No ID here", patterns=custom)
        assert found == []

    def test_empty_text(self):
        found = detect_pii("")
        assert found == []


# ---------------------------------------------------------------------------
# assess_embedding_risk
# ---------------------------------------------------------------------------


class TestAssessEmbeddingRisk:
    def test_critical_with_high_pii_and_embedding(self):
        result = assess_embedding_risk(
            "Email: user@example.com", has_embedding=True, text_id="t1",
        )
        assert result.risk_level == "critical"
        assert result.has_embedding is True
        assert "email" in result.pii_found

    def test_high_with_medium_pii_and_embedding(self):
        result = assess_embedding_risk(
            "IP: 10.0.0.1", has_embedding=True, text_id="t2",
        )
        assert result.risk_level == "high"

    def test_high_with_high_pii_no_embedding(self):
        result = assess_embedding_risk(
            "SSN: 123-45-6789", has_embedding=False, text_id="t3",
        )
        assert result.risk_level == "high"

    def test_medium_with_medium_pii_no_embedding(self):
        result = assess_embedding_risk(
            "IP: 192.168.1.1", has_embedding=False, text_id="t4",
        )
        assert result.risk_level == "medium"

    def test_none_risk_no_pii(self):
        result = assess_embedding_risk(
            "Just a normal sentence", has_embedding=True, text_id="t5",
        )
        assert result.risk_level == "none"
        assert result.pii_count == 0

    def test_recommendations_present_for_critical(self):
        result = assess_embedding_risk(
            "user@test.com", has_embedding=True, text_id="t6",
        )
        assert len(result.recommendations) > 0

    def test_no_recommendations_for_clean_text(self):
        result = assess_embedding_risk(
            "Safe text", has_embedding=False, text_id="t7",
        )
        assert result.recommendations == []

    def test_text_id_propagated(self):
        result = assess_embedding_risk("hi", has_embedding=False, text_id="my-id")
        assert result.text_id == "my-id"

    def test_default_text_id(self):
        result = assess_embedding_risk("hi", has_embedding=False)
        assert result.text_id == ""


# ---------------------------------------------------------------------------
# check_dataset_pii_safety
# ---------------------------------------------------------------------------


class TestCheckDatasetPIISafety:
    def test_empty_dataset(self):
        report = check_dataset_pii_safety([])
        assert report.total_items == 0
        assert report.items_with_pii == 0

    def test_dataset_with_pii_and_embeddings(self):
        items = [
            {"id": "a", "input": "user@test.com", "embedding": [0.1, 0.2]},
            {"id": "b", "input": "Safe text", "embedding": [0.3, 0.4]},
        ]
        report = check_dataset_pii_safety(items)
        assert report.total_items == 2
        assert report.items_with_pii == 1
        assert report.items_at_risk == 1

    def test_dataset_no_embeddings(self):
        items = [
            {"id": "a", "input": "SSN: 123-45-6789"},
        ]
        report = check_dataset_pii_safety(items)
        assert report.items_with_pii == 1
        # high risk (SSN is high severity, no embedding)
        assert report.results[0].risk_level == "high"

    def test_custom_field_names(self):
        items = [
            {"id": "x", "text": "user@test.com", "vec": [0.1]},
        ]
        report = check_dataset_pii_safety(items, text_field="text", embedding_field="vec")
        assert report.items_with_pii == 1

    def test_summary_risk_distribution(self):
        items = [
            {"input": "user@a.com", "embedding": [1]},
            {"input": "safe", "embedding": [2]},
        ]
        report = check_dataset_pii_safety(items)
        dist = report.summary["risk_distribution"]
        assert "critical" in dist or "high" in dist or "none" in dist

    def test_summary_pattern_distribution(self):
        items = [
            {"input": "user@a.com and 192.168.0.1", "embedding": [1]},
        ]
        report = check_dataset_pii_safety(items)
        pat_dist = report.summary["pattern_distribution"]
        assert "email" in pat_dist
        assert "ip_address" in pat_dist

    def test_auto_id_when_missing(self):
        items = [{"input": "user@x.com"}]
        report = check_dataset_pii_safety(items)
        assert report.results[0].text_id == "0"

    def test_none_embedding_not_treated_as_present(self):
        items = [{"input": "user@x.com", "embedding": None}]
        report = check_dataset_pii_safety(items)
        assert report.results[0].has_embedding is False


# ---------------------------------------------------------------------------
# format_pii_report
# ---------------------------------------------------------------------------


class TestFormatPIIReport:
    def test_empty_report(self):
        report = PIISafetyReport(
            results=[], total_items=0, items_with_pii=0,
            items_at_risk=0, summary={},
        )
        text = format_pii_report(report)
        assert "PII Safety Report" in text
        assert "Total items scanned: 0" in text

    def test_report_with_items(self):
        items = [
            {"id": "a", "input": "user@test.com", "embedding": [0.1]},
        ]
        report = check_dataset_pii_safety(items)
        text = format_pii_report(report)
        assert "Items with PII: 1" in text
        assert "CRITICAL" in text or "HIGH" in text

    def test_risk_distribution_shown(self):
        items = [
            {"input": "user@a.com", "embedding": [1]},
            {"input": "safe"},
        ]
        report = check_dataset_pii_safety(items)
        text = format_pii_report(report)
        assert "Risk distribution" in text

    def test_pattern_distribution_shown(self):
        items = [{"input": "user@a.com", "embedding": [1]}]
        report = check_dataset_pii_safety(items)
        text = format_pii_report(report)
        assert "PII patterns detected" in text
        assert "email" in text


# ---------------------------------------------------------------------------
# suggest_mitigations
# ---------------------------------------------------------------------------


class TestSuggestMitigations:
    def test_no_pii_returns_no_action(self):
        result = PIIDetectionResult("t1", [], 0, False, "none", [])
        suggestions = suggest_mitigations(result)
        assert len(suggestions) == 1
        assert "No PII" in suggestions[0]

    def test_critical_has_redact_recommendation(self):
        result = PIIDetectionResult(
            "t1", ["email"], 1, True, "critical", [],
        )
        suggestions = suggest_mitigations(result)
        assert any("redact" in s.lower() for s in suggestions)

    def test_embedding_specific_suggestions(self):
        result = PIIDetectionResult(
            "t1", ["email"], 1, True, "high", [],
        )
        suggestions = suggest_mitigations(result)
        assert any("differential-privacy" in s.lower() for s in suggestions)
        assert any("dimensionality" in s.lower() for s in suggestions)

    def test_ssn_removal_suggestion(self):
        result = PIIDetectionResult(
            "t1", ["ssn"], 1, True, "critical", [],
        )
        suggestions = suggest_mitigations(result)
        assert any("remove" in s.lower() for s in suggestions)

    def test_credit_card_removal_suggestion(self):
        result = PIIDetectionResult(
            "t1", ["credit_card"], 1, True, "critical", [],
        )
        suggestions = suggest_mitigations(result)
        assert any("remove" in s.lower() for s in suggestions)

    def test_email_specific(self):
        result = PIIDetectionResult(
            "t1", ["email"], 1, False, "high", [],
        )
        suggestions = suggest_mitigations(result)
        assert any("email" in s.lower() for s in suggestions)

    def test_phone_specific(self):
        result = PIIDetectionResult(
            "t1", ["phone"], 1, False, "high", [],
        )
        suggestions = suggest_mitigations(result)
        assert any("phone" in s.lower() for s in suggestions)

    def test_ip_address_specific(self):
        result = PIIDetectionResult(
            "t1", ["ip_address"], 1, False, "medium", [],
        )
        suggestions = suggest_mitigations(result)
        assert any("ip" in s.lower() for s in suggestions)

    def test_dob_specific(self):
        result = PIIDetectionResult(
            "t1", ["date_of_birth"], 1, False, "medium", [],
        )
        suggestions = suggest_mitigations(result)
        assert any("age" in s.lower() for s in suggestions)

    def test_low_risk_monitor_suggestion(self):
        result = PIIDetectionResult(
            "t1", ["date_of_birth"], 1, False, "low",
            [],
        )
        # Use custom patterns where dob is low severity so risk is low
        custom = [PIIPattern("date_of_birth", r"\d{4}", "DOB", "low")]
        # Directly construct a low-risk result
        low = PIIDetectionResult(
            "t1", ["date_of_birth"], 1, False, "low", [],
        )
        suggestions = suggest_mitigations(low)
        assert any("monitor" in s.lower() for s in suggestions)
