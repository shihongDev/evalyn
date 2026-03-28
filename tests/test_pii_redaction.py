"""Tests for PII redaction module."""

from __future__ import annotations

import re
from datetime import datetime, timezone

import pytest

from evalyn_sdk.models import Span
from evalyn_sdk.trace.pii_redaction import (
    CREDIT_CARD_PATTERN,
    EMAIL_PATTERN,
    IP_ADDRESS_PATTERN,
    PHONE_PATTERN,
    SSN_PATTERN,
    RedactionConfig,
    RedactionPattern,
    RedactionResult,
    default_config,
    redact_span,
    redact_spans,
    redact_text,
)


def _span(sid="s1", **attrs):
    return Span(
        id=sid,
        name="test",
        span_type="llm_call",
        parent_id=None,
        start_time=datetime.now(timezone.utc),
        attributes=attrs,
    )


# ---------------------------------------------------------------------------
# Built-in pattern detection
# ---------------------------------------------------------------------------


class TestEmailPattern:
    def test_simple_email(self):
        cfg = default_config("mask")
        result = redact_text("contact user@example.com please", cfg)
        assert "[REDACTED:EMAIL]" in result
        assert "user@example.com" not in result

    def test_email_with_plus(self):
        cfg = default_config("mask")
        result = redact_text("send to user+tag@example.co.uk", cfg)
        assert "[REDACTED:EMAIL]" in result

    def test_no_email(self):
        cfg = default_config("mask")
        result = redact_text("no email here", cfg)
        assert result == "no email here"


class TestPhonePattern:
    def test_us_phone_dashes(self):
        cfg = default_config("mask")
        result = redact_text("call 555-123-4567", cfg)
        assert "[REDACTED:PHONE]" in result
        assert "555-123-4567" not in result

    def test_us_phone_dots(self):
        cfg = default_config("mask")
        result = redact_text("call 555.123.4567", cfg)
        assert "[REDACTED:PHONE]" in result

    def test_us_phone_with_country_code(self):
        cfg = default_config("mask")
        result = redact_text("call +1-555-123-4567", cfg)
        assert "[REDACTED:PHONE]" in result

    def test_us_phone_parens(self):
        cfg = default_config("mask")
        result = redact_text("call (555) 123-4567", cfg)
        assert "[REDACTED:PHONE]" in result


class TestSSNPattern:
    def test_ssn(self):
        cfg = default_config("mask")
        result = redact_text("SSN: 123-45-6789", cfg)
        assert "[REDACTED:SSN]" in result
        assert "123-45-6789" not in result

    def test_not_ssn(self):
        cfg = default_config("mask")
        result = redact_text("code: 12-345-6789", cfg)
        # Should not match SSN pattern (wrong grouping)
        assert "[REDACTED:SSN]" not in result


class TestCreditCardPattern:
    def test_cc_spaces(self):
        cfg = default_config("mask")
        result = redact_text("card: 4111 1111 1111 1111", cfg)
        assert "[REDACTED:CREDIT_CARD]" in result
        assert "4111 1111 1111 1111" not in result

    def test_cc_dashes(self):
        cfg = default_config("mask")
        result = redact_text("card: 4111-1111-1111-1111", cfg)
        assert "[REDACTED:CREDIT_CARD]" in result

    def test_cc_no_separator(self):
        cfg = default_config("mask")
        result = redact_text("card: 4111111111111111", cfg)
        assert "[REDACTED:CREDIT_CARD]" in result


class TestIPAddressPattern:
    def test_ipv4(self):
        cfg = default_config("mask")
        result = redact_text("server at 192.168.1.1", cfg)
        assert "[REDACTED:IP_ADDRESS]" in result
        assert "192.168.1.1" not in result

    def test_localhost(self):
        cfg = default_config("mask")
        result = redact_text("connect to 127.0.0.1", cfg)
        assert "[REDACTED:IP_ADDRESS]" in result


# ---------------------------------------------------------------------------
# Strategy tests
# ---------------------------------------------------------------------------


class TestMaskStrategy:
    def test_mask_email(self):
        cfg = default_config("mask")
        result = redact_text("email: a@b.com", cfg)
        assert result == "email: [REDACTED:EMAIL]"

    def test_mask_multiple(self):
        cfg = default_config("mask")
        text = "email a@b.com and SSN 111-22-3333"
        result = redact_text(text, cfg)
        assert "[REDACTED:EMAIL]" in result
        assert "[REDACTED:SSN]" in result


class TestHashStrategy:
    def test_hash_email(self):
        cfg = default_config("hash")
        result = redact_text("email: a@b.com", cfg)
        # Should be replaced with 8-char hex hash
        assert "a@b.com" not in result
        assert "[REDACTED" not in result
        # Hash is deterministic
        result2 = redact_text("email: a@b.com", cfg)
        assert result == result2

    def test_hash_length(self):
        cfg = default_config("hash")
        result = redact_text("a@b.com", cfg)
        # Result should be exactly 8 hex chars
        assert len(result) == 8
        assert all(c in "0123456789abcdef" for c in result)


class TestRemoveStrategy:
    def test_remove_email(self):
        cfg = default_config("remove")
        result = redact_text("email: a@b.com end", cfg)
        assert result == "email:  end"
        assert "a@b.com" not in result

    def test_remove_ssn(self):
        cfg = default_config("remove")
        result = redact_text("SSN: 111-22-3333 done", cfg)
        assert "111-22-3333" not in result


# ---------------------------------------------------------------------------
# Multiple PII in same text
# ---------------------------------------------------------------------------


class TestMultiplePII:
    def test_email_and_phone(self):
        cfg = default_config("mask")
        text = "Contact: user@example.com or 555-123-4567"
        result = redact_text(text, cfg)
        assert "[REDACTED:EMAIL]" in result
        assert "[REDACTED:PHONE]" in result

    def test_all_types(self):
        cfg = default_config("mask")
        text = (
            "Email: a@b.com, Phone: 555-123-4567, "
            "SSN: 111-22-3333, Card: 4111 1111 1111 1111, "
            "IP: 10.0.0.1"
        )
        result = redact_text(text, cfg)
        assert "[REDACTED:EMAIL]" in result
        assert "[REDACTED:PHONE]" in result
        assert "[REDACTED:SSN]" in result
        assert "[REDACTED:CREDIT_CARD]" in result
        assert "[REDACTED:IP_ADDRESS]" in result


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_text(self):
        cfg = default_config("mask")
        result = redact_text("", cfg)
        assert result == ""

    def test_no_pii(self):
        cfg = default_config("mask")
        text = "This is perfectly clean text with no sensitive data."
        result = redact_text(text, cfg)
        assert result == text

    def test_partial_email_not_matched(self):
        cfg = default_config("mask")
        # Just a username without domain should not match
        result = redact_text("user@ is not valid", cfg)
        assert "[REDACTED:EMAIL]" not in result

    def test_repeated_pii(self):
        cfg = default_config("mask")
        text = "a@b.com and a@b.com again"
        result = redact_text(text, cfg)
        assert result.count("[REDACTED:EMAIL]") == 2


# ---------------------------------------------------------------------------
# Custom patterns
# ---------------------------------------------------------------------------


class TestCustomPatterns:
    def test_custom_pattern_applied(self):
        custom = RedactionPattern(
            name="API_KEY",
            pattern=re.compile(r"sk-[a-zA-Z0-9]{10,}"),
            replacement="[REDACTED:API_KEY]",
        )
        cfg = RedactionConfig(
            strategy="mask", patterns=[], custom_patterns=[custom]
        )
        result = redact_text("key: sk-abcdefghij1234", cfg)
        assert "[REDACTED:API_KEY]" in result
        assert "sk-abcdefghij1234" not in result

    def test_custom_with_builtin(self):
        custom = RedactionPattern(
            name="API_KEY",
            pattern=re.compile(r"sk-[a-zA-Z0-9]{10,}"),
            replacement="[REDACTED:API_KEY]",
        )
        cfg = default_config("mask")
        cfg.custom_patterns.append(custom)
        text = "key: sk-abcdefghij1234 email: a@b.com"
        result = redact_text(text, cfg)
        assert "[REDACTED:API_KEY]" in result
        assert "[REDACTED:EMAIL]" in result


# ---------------------------------------------------------------------------
# Span redaction
# ---------------------------------------------------------------------------


class TestRedactSpan:
    def test_creates_new_span(self):
        original = _span(input="email: a@b.com")
        cfg = default_config("mask")
        new_span, result = redact_span(original, cfg)
        # Original is unchanged
        assert original.attributes["input"] == "email: a@b.com"
        # New span has redacted value
        assert "[REDACTED:EMAIL]" in new_span.attributes["input"]
        assert new_span is not original

    def test_redacts_input_key(self):
        s = _span(input="SSN: 111-22-3333")
        cfg = default_config("mask")
        new_s, _ = redact_span(s, cfg)
        assert "[REDACTED:SSN]" in new_s.attributes["input"]

    def test_redacts_output_key(self):
        s = _span(output="call 555-123-4567")
        cfg = default_config("mask")
        new_s, _ = redact_span(s, cfg)
        assert "[REDACTED:PHONE]" in new_s.attributes["output"]

    def test_redacts_inputs_key(self):
        s = _span(inputs="card: 4111 1111 1111 1111")
        cfg = default_config("mask")
        new_s, _ = redact_span(s, cfg)
        assert "[REDACTED:CREDIT_CARD]" in new_s.attributes["inputs"]

    def test_redacts_outputs_key(self):
        s = _span(outputs="IP: 10.0.0.1")
        cfg = default_config("mask")
        new_s, _ = redact_span(s, cfg)
        assert "[REDACTED:IP_ADDRESS]" in new_s.attributes["outputs"]

    def test_ignores_non_target_keys(self):
        s = _span(input="a@b.com", model="gpt-4")
        cfg = default_config("mask")
        new_s, _ = redact_span(s, cfg)
        assert new_s.attributes["model"] == "gpt-4"

    def test_nested_dict_redaction(self):
        s = _span(input={"query": "email me at a@b.com", "context": "safe"})
        cfg = default_config("mask")
        new_s, result = redact_span(s, cfg)
        assert "[REDACTED:EMAIL]" in new_s.attributes["input"]["query"]
        assert new_s.attributes["input"]["context"] == "safe"
        assert result.redactions_count == 1

    def test_span_id_preserved(self):
        s = _span(sid="my-id", input="a@b.com")
        cfg = default_config("mask")
        new_s, _ = redact_span(s, cfg)
        assert new_s.id == "my-id"
        assert new_s.name == "test"
        assert new_s.span_type == "llm_call"


# ---------------------------------------------------------------------------
# Batch span redaction
# ---------------------------------------------------------------------------


class TestRedactSpans:
    def test_batch_redaction(self):
        spans = [
            _span(sid="s1", input="a@b.com"),
            _span(sid="s2", output="555-123-4567"),
        ]
        cfg = default_config("mask")
        new_spans, result = redact_spans(spans, cfg)
        assert len(new_spans) == 2
        assert "[REDACTED:EMAIL]" in new_spans[0].attributes["input"]
        assert "[REDACTED:PHONE]" in new_spans[1].attributes["output"]
        assert result.redactions_count == 2

    def test_batch_empty_list(self):
        cfg = default_config("mask")
        new_spans, result = redact_spans([], cfg)
        assert new_spans == []
        assert result.redactions_count == 0

    def test_batch_no_pii(self):
        spans = [_span(sid="s1", input="clean text")]
        cfg = default_config("mask")
        new_spans, result = redact_spans(spans, cfg)
        assert new_spans[0].attributes["input"] == "clean text"
        assert result.redactions_count == 0

    def test_originals_unchanged(self):
        originals = [_span(sid="s1", input="a@b.com")]
        cfg = default_config("mask")
        redact_spans(originals, cfg)
        assert originals[0].attributes["input"] == "a@b.com"


# ---------------------------------------------------------------------------
# RedactionResult
# ---------------------------------------------------------------------------


class TestRedactionResult:
    def test_counts_correct(self):
        s = _span(input="a@b.com and c@d.com")
        cfg = default_config("mask")
        _, result = redact_span(s, cfg)
        assert result.redactions_count == 2
        assert "input" in result.redacted_fields

    def test_no_redactions(self):
        s = _span(input="clean")
        cfg = default_config("mask")
        _, result = redact_span(s, cfg)
        assert result.redactions_count == 0
        assert result.redacted_fields == []

    def test_original_vs_redacted_length(self):
        s = _span(input="a@b.com")
        cfg = default_config("mask")
        _, result = redact_span(s, cfg)
        assert result.original_length == len("a@b.com")
        assert result.redacted_length == len("[REDACTED:EMAIL]")


# ---------------------------------------------------------------------------
# as_dict / from_dict roundtrip
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_redaction_pattern_roundtrip(self):
        pat = RedactionPattern(
            name="TEST",
            pattern=re.compile(r"\btest\b"),
            replacement="[REDACTED:TEST]",
        )
        d = pat.as_dict()
        restored = RedactionPattern.from_dict(d)
        assert restored.name == pat.name
        assert restored.pattern.pattern == pat.pattern.pattern
        assert restored.replacement == pat.replacement

    def test_redaction_config_roundtrip(self):
        cfg = default_config("hash")
        d = cfg.as_dict()
        restored = RedactionConfig.from_dict(d)
        assert restored.strategy == "hash"
        assert len(restored.patterns) == 5
        assert restored.patterns[0].name == "EMAIL"

    def test_redaction_result_roundtrip(self):
        result = RedactionResult(
            original_length=100,
            redacted_length=120,
            redactions_count=3,
            redacted_fields=["input", "output"],
        )
        d = result.as_dict()
        restored = RedactionResult.from_dict(d)
        assert restored.original_length == 100
        assert restored.redacted_length == 120
        assert restored.redactions_count == 3
        assert restored.redacted_fields == ["input", "output"]

    def test_config_with_custom_patterns_roundtrip(self):
        custom = RedactionPattern(
            name="MY_PAT",
            pattern=re.compile(r"secret-\d+"),
            replacement="[REDACTED:MY_PAT]",
        )
        cfg = RedactionConfig(
            strategy="remove",
            patterns=[],
            custom_patterns=[custom],
        )
        d = cfg.as_dict()
        restored = RedactionConfig.from_dict(d)
        assert restored.strategy == "remove"
        assert len(restored.custom_patterns) == 1
        assert restored.custom_patterns[0].name == "MY_PAT"


# ---------------------------------------------------------------------------
# default_config
# ---------------------------------------------------------------------------


class TestDefaultConfig:
    def test_default_mask(self):
        cfg = default_config()
        assert cfg.strategy == "mask"
        assert len(cfg.patterns) == 5
        assert cfg.custom_patterns == []

    def test_default_hash(self):
        cfg = default_config("hash")
        assert cfg.strategy == "hash"

    def test_default_remove(self):
        cfg = default_config("remove")
        assert cfg.strategy == "remove"

    def test_pattern_names(self):
        cfg = default_config()
        names = [p.name for p in cfg.patterns]
        assert names == ["EMAIL", "PHONE", "SSN", "CREDIT_CARD", "IP_ADDRESS"]
