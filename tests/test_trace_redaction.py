"""Tests for the trace redaction module."""

from __future__ import annotations

import pytest

from evalyn_sdk.trace_redaction import (
    RELAXED_POLICY,
    STRICT_POLICY,
    RedactionAuditEntry,
    RedactionPolicy,
    RedactionResult,
    RedactionRule,
    apply_redaction,
    apply_truncation,
    build_default_policy,
    format_redaction_audit,
    generate_redaction_audit,
)


# ---------------------------------------------------------------------------
# RedactionRule
# ---------------------------------------------------------------------------


class TestRedactionRule:
    def test_defaults(self):
        r = RedactionRule(rule_id="r1", pattern=r"\d+")
        assert r.replacement == "[REDACTED]"
        assert r.description == ""
        assert r.applies_to == ["all"]

    def test_as_dict(self):
        r = RedactionRule(rule_id="r1", pattern=r"\d+", replacement="X", description="numbers")
        d = r.as_dict()
        assert d["rule_id"] == "r1"
        assert d["replacement"] == "X"

    def test_from_dict_minimal(self):
        r = RedactionRule.from_dict({"rule_id": "r1", "pattern": r"\w+"})
        assert r.replacement == "[REDACTED]"
        assert r.applies_to == ["all"]

    def test_roundtrip(self):
        r = RedactionRule(rule_id="test", pattern=r"\d{3}", replacement="***", applies_to=["input"])
        r2 = RedactionRule.from_dict(r.as_dict())
        assert r2.rule_id == r.rule_id
        assert r2.pattern == r.pattern
        assert r2.applies_to == ["input"]

    def test_applies_to_independent(self):
        r = RedactionRule(rule_id="r", pattern="x", applies_to=["input"])
        d = r.as_dict()
        d["applies_to"].append("output")
        assert r.applies_to == ["input"]


# ---------------------------------------------------------------------------
# RedactionPolicy
# ---------------------------------------------------------------------------


class TestRedactionPolicy:
    def test_defaults(self):
        p = RedactionPolicy(policy_id="p1", name="Test")
        assert p.rules == []
        assert p.truncate_after == 0
        assert p.mode == "strict"

    def test_as_dict(self):
        rule = RedactionRule(rule_id="r1", pattern="x")
        p = RedactionPolicy(policy_id="p1", name="P", rules=[rule], truncate_after=100)
        d = p.as_dict()
        assert len(d["rules"]) == 1
        assert d["truncate_after"] == 100

    def test_from_dict(self):
        data = {
            "policy_id": "p1",
            "name": "Test",
            "rules": [{"rule_id": "r1", "pattern": "x"}],
            "mode": "relaxed",
        }
        p = RedactionPolicy.from_dict(data)
        assert p.mode == "relaxed"
        assert len(p.rules) == 1

    def test_roundtrip(self):
        rule = RedactionRule(rule_id="r1", pattern=r"\d+")
        p = RedactionPolicy(policy_id="p1", name="Test", rules=[rule], keep_first_n=10, keep_last_n=5)
        p2 = RedactionPolicy.from_dict(p.as_dict())
        assert p2.keep_first_n == 10
        assert p2.keep_last_n == 5


# ---------------------------------------------------------------------------
# RedactionResult
# ---------------------------------------------------------------------------


class TestRedactionResult:
    def test_defaults(self):
        r = RedactionResult(original_length=100, redacted_length=80)
        assert r.rules_applied == []
        assert r.truncated is False

    def test_roundtrip(self):
        r = RedactionResult(original_length=50, redacted_length=30, rules_applied=["email"], truncated=True)
        r2 = RedactionResult.from_dict(r.as_dict())
        assert r2.original_length == 50
        assert r2.truncated is True


# ---------------------------------------------------------------------------
# RedactionAuditEntry
# ---------------------------------------------------------------------------


class TestRedactionAuditEntry:
    def test_as_dict(self):
        result = RedactionResult(original_length=10, redacted_length=5)
        entry = RedactionAuditEntry(trace_id="t1", field_name="input", result=result)
        d = entry.as_dict()
        assert d["trace_id"] == "t1"
        assert d["result"]["original_length"] == 10

    def test_roundtrip(self):
        result = RedactionResult(original_length=20, redacted_length=10, rules_applied=["ssn"])
        entry = RedactionAuditEntry(trace_id="t2", field_name="output", result=result)
        entry2 = RedactionAuditEntry.from_dict(entry.as_dict())
        assert entry2.result.rules_applied == ["ssn"]


# ---------------------------------------------------------------------------
# apply_redaction
# ---------------------------------------------------------------------------


class TestApplyRedaction:
    def test_email_redaction(self):
        policy = build_default_policy("strict")
        text = "Contact user@example.com for info"
        redacted, result = apply_redaction(text, policy)
        assert "user@example.com" not in redacted
        assert "[EMAIL_REDACTED]" in redacted
        assert "email" in result.rules_applied

    def test_phone_redaction(self):
        policy = build_default_policy("strict")
        text = "Call 555-123-4567 now"
        redacted, result = apply_redaction(text, policy)
        assert "555-123-4567" not in redacted
        assert "[PHONE_REDACTED]" in redacted

    def test_ssn_redaction(self):
        policy = build_default_policy("strict")
        text = "SSN is 123-45-6789"
        redacted, result = apply_redaction(text, policy)
        assert "123-45-6789" not in redacted
        assert "[SSN_REDACTED]" in redacted

    def test_api_key_redaction(self):
        policy = build_default_policy("strict")
        text = "Use token-abcdefghijklmnopqrstuvwxyz to authenticate"
        redacted, result = apply_redaction(text, policy)
        assert "token-abcdefghijklmnopqrstuvwxyz" not in redacted
        assert "[API_KEY_REDACTED]" in redacted

    def test_relaxed_skips_email(self):
        policy = build_default_policy("relaxed")
        text = "Email user@example.com please"
        redacted, result = apply_redaction(text, policy)
        assert "user@example.com" in redacted
        assert "email" not in result.rules_applied

    def test_relaxed_catches_api_key(self):
        policy = build_default_policy("relaxed")
        text = "token_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef"
        redacted, result = apply_redaction(text, policy)
        assert "ABCDEFGHIJ" not in redacted

    def test_no_redaction_needed(self):
        policy = build_default_policy("strict")
        text = "This is a clean text with nothing to redact."
        redacted, result = apply_redaction(text, policy)
        assert redacted == text
        assert result.rules_applied == []

    def test_field_name_filter(self):
        rule = RedactionRule(rule_id="input_only", pattern=r"\d+", replacement="N", applies_to=["input"])
        policy = RedactionPolicy(policy_id="p", name="P", rules=[rule])
        # Applying to "input" field should trigger the rule
        redacted, result = apply_redaction("abc 123 def", policy, field_name="input")
        assert "123" not in redacted
        assert "input_only" in result.rules_applied
        # Applying to "output" field should NOT trigger the rule
        redacted2, result2 = apply_redaction("abc 123 def", policy, field_name="output")
        assert "123" in redacted2
        assert result2.rules_applied == []

    def test_truncation_applied(self):
        policy = RedactionPolicy(policy_id="p", name="P", truncate_after=20)
        text = "A" * 100
        redacted, result = apply_redaction(text, policy)
        assert len(redacted) <= 20
        assert result.truncated is True

    def test_no_truncation_when_short(self):
        policy = RedactionPolicy(policy_id="p", name="P", truncate_after=200)
        text = "short"
        redacted, result = apply_redaction(text, policy)
        assert redacted == "short"
        assert result.truncated is False

    def test_multiple_rules(self):
        policy = build_default_policy("strict")
        text = "Email: a@b.com, SSN: 111-22-3333"
        redacted, result = apply_redaction(text, policy)
        assert "a@b.com" not in redacted
        assert "111-22-3333" not in redacted
        assert len(result.rules_applied) >= 2

    def test_result_lengths(self):
        policy = build_default_policy("strict")
        text = "Contact user@example.com"
        redacted, result = apply_redaction(text, policy)
        assert result.original_length == len(text)
        assert result.redacted_length == len(redacted)


# ---------------------------------------------------------------------------
# apply_truncation
# ---------------------------------------------------------------------------


class TestApplyTruncation:
    def test_short_text_unchanged(self):
        assert apply_truncation("hello", 100) == "hello"

    def test_long_text_truncated(self):
        text = "A" * 100
        result = apply_truncation(text, 30)
        assert len(result) <= 30
        assert "[truncated]" in result

    def test_keep_first_n(self):
        text = "ABCDE" + "x" * 100
        result = apply_truncation(text, 50, keep_first=5)
        assert result.startswith("ABCDE")
        assert "[truncated]" in result

    def test_keep_last_n(self):
        text = "x" * 100 + "ZZZZZ"
        result = apply_truncation(text, 50, keep_last=5)
        assert result.endswith("ZZZZZ")
        assert "[truncated]" in result

    def test_keep_first_and_last(self):
        text = "START" + "x" * 200 + "END"
        result = apply_truncation(text, 50, keep_first=5, keep_last=3)
        assert result.startswith("START")
        assert result.endswith("END")
        assert "[truncated]" in result

    def test_very_small_max_len(self):
        text = "A" * 50
        result = apply_truncation(text, 5)
        assert len(result) == 5


# ---------------------------------------------------------------------------
# build_default_policy
# ---------------------------------------------------------------------------


class TestBuildDefaultPolicy:
    def test_strict(self):
        p = build_default_policy("strict")
        assert p.mode == "strict"
        rule_ids = {r.rule_id for r in p.rules}
        assert "email" in rule_ids
        assert "phone" in rule_ids
        assert "ssn" in rule_ids
        assert "api_key" in rule_ids

    def test_relaxed(self):
        p = build_default_policy("relaxed")
        assert p.mode == "relaxed"
        rule_ids = {r.rule_id for r in p.rules}
        assert "api_key" in rule_ids
        assert "email" not in rule_ids


# ---------------------------------------------------------------------------
# generate_redaction_audit
# ---------------------------------------------------------------------------


class TestGenerateRedactionAudit:
    def _make_entry(self, trace_id, rules, orig=100, redacted=80):
        result = RedactionResult(original_length=orig, redacted_length=redacted, rules_applied=rules)
        return RedactionAuditEntry(trace_id=trace_id, field_name="input", result=result)

    def test_empty(self):
        audit = generate_redaction_audit([])
        assert audit["total_traces"] == 0
        assert audit["total_redactions"] == 0
        assert audit["bytes_saved"] == 0
        assert audit["most_triggered_rule"] == ""

    def test_single_entry(self):
        entry = self._make_entry("t1", ["email"])
        audit = generate_redaction_audit([entry])
        assert audit["total_traces"] == 1
        assert audit["total_redactions"] == 1
        assert audit["bytes_saved"] == 20
        assert audit["most_triggered_rule"] == "email"

    def test_multiple_entries_same_trace(self):
        e1 = self._make_entry("t1", ["email"], orig=100, redacted=80)
        e2 = self._make_entry("t1", ["ssn"], orig=50, redacted=30)
        audit = generate_redaction_audit([e1, e2])
        assert audit["total_traces"] == 1
        assert audit["total_redactions"] == 2
        assert audit["bytes_saved"] == 40

    def test_multiple_traces(self):
        e1 = self._make_entry("t1", ["email"])
        e2 = self._make_entry("t2", ["phone"])
        audit = generate_redaction_audit([e1, e2])
        assert audit["total_traces"] == 2

    def test_most_triggered_rule(self):
        entries = [
            self._make_entry("t1", ["email"]),
            self._make_entry("t2", ["email", "ssn"]),
            self._make_entry("t3", ["email"]),
        ]
        audit = generate_redaction_audit(entries)
        assert audit["most_triggered_rule"] == "email"
        assert audit["rule_counts"]["email"] == 3
        assert audit["rule_counts"]["ssn"] == 1


# ---------------------------------------------------------------------------
# format_redaction_audit
# ---------------------------------------------------------------------------


class TestFormatRedactionAudit:
    def test_basic_format(self):
        audit = {
            "total_traces": 5,
            "total_redactions": 12,
            "bytes_saved": 340,
            "most_triggered_rule": "email",
            "rule_counts": {"email": 8, "ssn": 4},
        }
        report = format_redaction_audit(audit)
        assert "Redaction Audit Report" in report
        assert "5" in report
        assert "12" in report
        assert "340" in report
        assert "email" in report
        assert "ssn" in report

    def test_empty_audit(self):
        audit = {"total_traces": 0, "total_redactions": 0, "bytes_saved": 0, "most_triggered_rule": "", "rule_counts": {}}
        report = format_redaction_audit(audit)
        assert "Redaction Audit Report" in report
        assert "Rule Breakdown" not in report


# ---------------------------------------------------------------------------
# Pre-built Policies
# ---------------------------------------------------------------------------


class TestPrebuiltPolicies:
    def test_strict_policy_exists(self):
        assert STRICT_POLICY.mode == "strict"
        assert len(STRICT_POLICY.rules) >= 4

    def test_relaxed_policy_exists(self):
        assert RELAXED_POLICY.mode == "relaxed"
        assert len(RELAXED_POLICY.rules) >= 1

    def test_strict_catches_email(self):
        text = "hello foo@bar.com world"
        redacted, _ = apply_redaction(text, STRICT_POLICY)
        assert "foo@bar.com" not in redacted

    def test_relaxed_ignores_email(self):
        text = "hello foo@bar.com world"
        redacted, _ = apply_redaction(text, RELAXED_POLICY)
        assert "foo@bar.com" in redacted
