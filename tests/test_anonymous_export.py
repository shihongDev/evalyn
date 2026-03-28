from __future__ import annotations

import pytest

from evalyn_sdk.storage.anonymous_export import (
    DEFAULT_RULES,
    AnonymizationRule,
    AnonymousExportConfig,
    ExportResult,
    anonymize_batch,
    anonymize_record,
    apply_rule,
    get_default_config,
    matches_pattern,
)


# --- matches_pattern ---


class TestMatchesPattern:
    def test_exact_match(self):
        assert matches_pattern("email", "email") is True

    def test_no_match(self):
        assert matches_pattern("foo", "bar") is False

    def test_leading_wildcard(self):
        assert matches_pattern("user_email", "*email") is True

    def test_trailing_wildcard(self):
        assert matches_pattern("email_address", "email*") is True

    def test_both_wildcards(self):
        assert matches_pattern("my_email_field", "*email*") is True

    def test_case_insensitive(self):
        assert matches_pattern("UserEmail", "*email*") is True
        assert matches_pattern("EMAIL", "*email*") is True

    def test_no_wildcard_no_match(self):
        assert matches_pattern("user_email", "email") is False

    def test_all_wildcard(self):
        assert matches_pattern("anything", "*") is True

    def test_empty_pattern(self):
        assert matches_pattern("anything", "") is False

    def test_empty_field(self):
        assert matches_pattern("", "*email*") is False

    def test_middle_wildcard(self):
        assert matches_pattern("user_name_field", "user*field") is True

    def test_middle_wildcard_no_match(self):
        assert matches_pattern("user_name_item", "user*field") is False


# --- apply_rule ---


class TestApplyRule:
    def test_replace(self):
        rule = AnonymizationRule("*", "[GONE]", "replace")
        assert apply_rule("hello", rule) == "[GONE]"

    def test_hash(self):
        rule = AnonymizationRule("*", "", "hash")
        result = apply_rule("secret", rule)
        assert len(result) == 16
        # Deterministic
        assert apply_rule("secret", rule) == result

    def test_hash_with_salt(self):
        rule = AnonymizationRule("*", "", "hash")
        a = apply_rule("secret", rule, salt="salt1")
        b = apply_rule("secret", rule, salt="salt2")
        assert a != b

    def test_remove(self):
        rule = AnonymizationRule("*", "[X]", "remove")
        assert apply_rule("anything", rule) == ""

    def test_unknown_method_falls_back_to_replacement(self):
        rule = AnonymizationRule("*", "[FALLBACK]", "unknown")
        assert apply_rule("data", rule) == "[FALLBACK]"


# --- AnonymizationRule ---


class TestAnonymizationRule:
    def test_defaults(self):
        r = AnonymizationRule("*email*")
        assert r.replacement == "[REDACTED]"
        assert r.method == "replace"

    def test_as_dict(self):
        r = AnonymizationRule("*name*", "[N]", "hash")
        d = r.as_dict()
        assert d["field_pattern"] == "*name*"
        assert d["method"] == "hash"

    def test_from_dict(self):
        d = {"field_pattern": "*pw*", "replacement": "[PW]", "method": "remove"}
        r = AnonymizationRule.from_dict(d)
        assert r.field_pattern == "*pw*"
        assert r.method == "remove"

    def test_roundtrip(self):
        original = AnonymizationRule("*token*", "[T]", "hash")
        restored = AnonymizationRule.from_dict(original.as_dict())
        assert restored.field_pattern == original.field_pattern
        assert restored.replacement == original.replacement
        assert restored.method == original.method


# --- AnonymousExportConfig ---


class TestAnonymousExportConfig:
    def test_defaults(self):
        c = AnonymousExportConfig()
        assert c.rules == []
        assert c.preserve_structure is True
        assert c.salt == ""

    def test_as_dict(self):
        c = AnonymousExportConfig(
            rules=[AnonymizationRule("*email*")],
            salt="abc",
        )
        d = c.as_dict()
        assert len(d["rules"]) == 1
        assert d["salt"] == "abc"

    def test_from_dict(self):
        d = {
            "rules": [{"field_pattern": "*x*", "method": "remove"}],
            "preserve_structure": False,
            "salt": "s",
        }
        c = AnonymousExportConfig.from_dict(d)
        assert len(c.rules) == 1
        assert c.preserve_structure is False
        assert c.salt == "s"

    def test_roundtrip(self):
        original = AnonymousExportConfig(
            rules=[AnonymizationRule("*a*", "[A]", "hash")],
            salt="pepper",
        )
        restored = AnonymousExportConfig.from_dict(original.as_dict())
        assert len(restored.rules) == len(original.rules)
        assert restored.salt == original.salt


# --- ExportResult ---


class TestExportResult:
    def test_defaults(self):
        r = ExportResult()
        assert r.total_records == 0
        assert r.fields_anonymized == 0
        assert r.rules_applied == 0

    def test_as_dict(self):
        r = ExportResult(total_records=10, fields_anonymized=5, rules_applied=3)
        d = r.as_dict()
        assert d["total_records"] == 10
        assert d["fields_anonymized"] == 5

    def test_format_text(self):
        r = ExportResult(total_records=7, fields_anonymized=3, rules_applied=2)
        text = r.format_text()
        assert "Export Result" in text
        assert "7" in text
        assert "3" in text


# --- anonymize_record ---


class TestAnonymizeRecord:
    def test_applies_matching_rules(self):
        config = AnonymousExportConfig(
            rules=[AnonymizationRule("*email*", "[EMAIL]", "replace")],
        )
        record = {"user_email": "foo@bar.com", "age": "25"}
        new, changed = anonymize_record(record, config)
        assert new["user_email"] == "[EMAIL]"
        assert new["age"] == "25"
        assert changed == 1

    def test_preserves_non_matching(self):
        config = AnonymousExportConfig(
            rules=[AnonymizationRule("*email*", "[EMAIL]", "replace")],
        )
        record = {"id": "123", "score": "0.9"}
        new, changed = anonymize_record(record, config)
        assert new == record
        assert changed == 0

    def test_hash_method(self):
        config = AnonymousExportConfig(
            rules=[AnonymizationRule("*user*", "", "hash")],
            salt="test",
        )
        record = {"username": "alice"}
        new, changed = anonymize_record(record, config)
        assert new["username"] != "alice"
        assert len(new["username"]) == 16
        assert changed == 1

    def test_remove_method(self):
        config = AnonymousExportConfig(
            rules=[AnonymizationRule("*password*", "", "remove")],
        )
        record = {"password": "secret123"}
        new, changed = anonymize_record(record, config)
        assert new["password"] == ""
        assert changed == 1

    def test_no_rules(self):
        config = AnonymousExportConfig(rules=[])
        record = {"email": "x@y.com"}
        new, changed = anonymize_record(record, config)
        assert new == record
        assert changed == 0

    def test_empty_record(self):
        config = get_default_config()
        new, changed = anonymize_record({}, config)
        assert new == {}
        assert changed == 0


# --- anonymize_batch ---


class TestAnonymizeBatch:
    def test_batch_all_records(self):
        records = [
            {"user_email": "a@b.com", "id": "1"},
            {"user_email": "c@d.com", "id": "2"},
        ]
        config = AnonymousExportConfig(
            rules=[AnonymizationRule("*email*", "[E]", "replace")],
        )
        results, export = anonymize_batch(records, config)
        assert len(results) == 2
        assert results[0]["user_email"] == "[E]"
        assert results[1]["user_email"] == "[E]"
        assert export.total_records == 2
        assert export.fields_anonymized == 2

    def test_batch_default_config(self):
        records = [{"user_email": "x@y.com", "user_name": "Alice"}]
        results, export = anonymize_batch(records)
        assert results[0]["user_email"] != "x@y.com"
        assert results[0]["user_name"] != "Alice"
        assert export.total_records == 1

    def test_batch_empty_records(self):
        results, export = anonymize_batch([])
        assert results == []
        assert export.total_records == 0
        assert export.fields_anonymized == 0

    def test_batch_no_matching_fields(self):
        records = [{"score": "0.5", "id": "abc"}]
        config = AnonymousExportConfig(
            rules=[AnonymizationRule("*email*", "[E]", "replace")],
        )
        results, export = anonymize_batch(records, config)
        assert results[0] == records[0]
        assert export.fields_anonymized == 0
        assert export.rules_applied == 0

    def test_batch_rules_applied_count(self):
        records = [
            {"email": "a@b.com", "name": "Bob"},
        ]
        config = AnonymousExportConfig(
            rules=[
                AnonymizationRule("*email*", "[E]", "replace"),
                AnonymizationRule("*name*", "[N]", "replace"),
            ],
        )
        _, export = anonymize_batch(records, config)
        assert export.rules_applied == 2


# --- get_default_config ---


class TestGetDefaultConfig:
    def test_has_rules(self):
        config = get_default_config()
        assert len(config.rules) == len(DEFAULT_RULES)

    def test_preserve_structure_default(self):
        config = get_default_config()
        assert config.preserve_structure is True

    def test_salt_empty(self):
        config = get_default_config()
        assert config.salt == ""

    def test_rules_are_copies(self):
        config = get_default_config()
        config.rules.pop()
        assert len(get_default_config().rules) == len(DEFAULT_RULES)
