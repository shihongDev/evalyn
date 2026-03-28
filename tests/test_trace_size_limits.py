from __future__ import annotations

from datetime import datetime, timedelta, timezone

from evalyn_sdk.models import Span
from evalyn_sdk.trace.size_limits import (
    SizeLimitConfig,
    TruncationResult,
    default_config,
    enforce_size_limits,
    get_limit,
    truncate_text,
)


def _now():
    return datetime.now(timezone.utc)


def _span(sid="s1", name="s", span_type="custom", duration_ms=100, status="ok", **attrs):
    t = _now()
    return Span(
        id=sid,
        name=name,
        span_type=span_type,
        parent_id=None,
        start_time=t,
        end_time=t + timedelta(milliseconds=duration_ms),
        status=status,
        attributes=attrs,
    )


# -- SizeLimitConfig as_dict / from_dict --


class TestSizeLimitConfigRoundtrip:
    def test_as_dict_contains_all_fields(self):
        cfg = SizeLimitConfig(
            default_max_bytes=5000,
            per_type_limits={"llm_call": 20000},
            truncation_marker="[cut]",
        )
        d = cfg.as_dict()
        assert d["default_max_bytes"] == 5000
        assert d["per_type_limits"] == {"llm_call": 20000}
        assert d["truncation_marker"] == "[cut]"

    def test_from_dict_roundtrip(self):
        original = SizeLimitConfig(
            default_max_bytes=8000,
            per_type_limits={"tool_call": 3000},
            truncation_marker="--snip--",
        )
        restored = SizeLimitConfig.from_dict(original.as_dict())
        assert restored.default_max_bytes == original.default_max_bytes
        assert restored.per_type_limits == original.per_type_limits
        assert restored.truncation_marker == original.truncation_marker

    def test_from_dict_defaults(self):
        cfg = SizeLimitConfig.from_dict({})
        assert cfg.default_max_bytes == 10000
        assert cfg.per_type_limits == {}
        assert cfg.truncation_marker == "... [truncated]"


# -- TruncationResult --


class TestTruncationResult:
    def test_as_dict(self):
        r = TruncationResult(
            field_name="input",
            original_bytes=20000,
            truncated_bytes=10000,
            was_truncated=True,
        )
        d = r.as_dict()
        assert d == {
            "field_name": "input",
            "original_bytes": 20000,
            "truncated_bytes": 10000,
            "was_truncated": True,
        }

    def test_as_dict_not_truncated(self):
        r = TruncationResult(
            field_name="output",
            original_bytes=500,
            truncated_bytes=500,
            was_truncated=False,
        )
        assert r.as_dict()["was_truncated"] is False


# -- get_limit --


class TestGetLimit:
    def test_per_type_limit(self):
        cfg = SizeLimitConfig(
            default_max_bytes=10000,
            per_type_limits={"llm_call": 50000},
        )
        assert get_limit(cfg, "llm_call") == 50000

    def test_default_fallback(self):
        cfg = SizeLimitConfig(
            default_max_bytes=10000,
            per_type_limits={"llm_call": 50000},
        )
        assert get_limit(cfg, "tool_call") == 10000

    def test_empty_per_type(self):
        cfg = SizeLimitConfig(default_max_bytes=7000)
        assert get_limit(cfg, "custom") == 7000


# -- truncate_text --


class TestTruncateText:
    def test_short_text_not_truncated(self):
        text, was = truncate_text("hello", 100, "... [truncated]")
        assert text == "hello"
        assert was is False

    def test_exact_fit_not_truncated(self):
        text = "abc"
        result, was = truncate_text(text, 3, "[x]")
        assert result == "abc"
        assert was is False

    def test_long_text_truncated(self):
        text = "a" * 200
        result, was = truncate_text(text, 50, "...")
        assert was is True
        assert result.endswith("...")
        assert len(result.encode("utf-8")) <= 50

    def test_marker_appended(self):
        text = "abcdefghij"
        result, was = truncate_text(text, 8, "[!]")
        assert was is True
        assert result.endswith("[!]")

    def test_utf8_multibyte_not_split(self):
        # Each CJK character is 3 bytes in UTF-8
        text = "abcde"  # 5 bytes ASCII, then add multibyte
        text = "abc\u4e16\u754c"  # 3 + 3 + 3 = 9 bytes
        result, was = truncate_text(text, 8, ".")
        assert was is True
        # Should not produce invalid UTF-8
        result.encode("utf-8")  # Should not raise
        assert len(result.encode("utf-8")) <= 8

    def test_empty_text_not_truncated(self):
        result, was = truncate_text("", 100, "...")
        assert result == ""
        assert was is False

    def test_marker_larger_than_limit(self):
        result, was = truncate_text("hello world", 3, "... [truncated]")
        assert was is True
        # Should still return something within limit
        assert len(result.encode("utf-8")) <= 3

    def test_unicode_emoji_boundary(self):
        # Emoji can be 4 bytes
        text = "hi" + "\U0001F600" * 5  # 2 + 20 = 22 bytes
        result, was = truncate_text(text, 10, ".")
        assert was is True
        result.encode("utf-8")  # valid UTF-8
        assert len(result.encode("utf-8")) <= 10


# -- enforce_size_limits --


class TestEnforceSizeLimits:
    def test_creates_new_span(self):
        s = _span(input="short", output="short")
        cfg = SizeLimitConfig(default_max_bytes=10000)
        new_s, _ = enforce_size_limits(s, cfg)
        assert new_s is not s
        assert new_s.id == s.id

    def test_does_not_mutate_original(self):
        original_input = "x" * 50000
        s = _span(input=original_input)
        cfg = SizeLimitConfig(default_max_bytes=100)
        enforce_size_limits(s, cfg)
        # Original unchanged
        assert s.attributes["input"] == original_input

    def test_short_fields_untouched(self):
        s = _span(input="hi", output="there")
        cfg = SizeLimitConfig(default_max_bytes=10000)
        new_s, results = enforce_size_limits(s, cfg)
        assert new_s.attributes["input"] == "hi"
        assert new_s.attributes["output"] == "there"
        for r in results:
            assert r.was_truncated is False

    def test_long_input_truncated(self):
        s = _span(input="a" * 50000)
        cfg = SizeLimitConfig(default_max_bytes=100)
        new_s, results = enforce_size_limits(s, cfg)
        assert len(new_s.attributes["input"].encode("utf-8")) <= 100
        input_result = [r for r in results if r.field_name == "input"][0]
        assert input_result.was_truncated is True
        assert input_result.original_bytes == 50000

    def test_long_output_truncated(self):
        s = _span(output="b" * 30000)
        cfg = SizeLimitConfig(default_max_bytes=200)
        new_s, results = enforce_size_limits(s, cfg)
        assert len(new_s.attributes["output"].encode("utf-8")) <= 200
        output_result = [r for r in results if r.field_name == "output"][0]
        assert output_result.was_truncated is True

    def test_multiple_fields_truncated_independently(self):
        s = _span(input="a" * 20000, output="b" * 20000)
        cfg = SizeLimitConfig(default_max_bytes=500)
        new_s, results = enforce_size_limits(s, cfg)
        assert len(results) == 2
        for r in results:
            assert r.was_truncated is True

    def test_per_type_limit_used(self):
        s = _span(span_type="llm_call", input="a" * 60000)
        cfg = SizeLimitConfig(
            default_max_bytes=100,
            per_type_limits={"llm_call": 50000},
        )
        new_s, results = enforce_size_limits(s, cfg)
        input_result = [r for r in results if r.field_name == "input"][0]
        assert input_result.was_truncated is True
        assert len(new_s.attributes["input"].encode("utf-8")) <= 50000

    def test_non_string_attributes_ignored(self):
        s = _span(input=12345, output={"key": "val"})
        cfg = SizeLimitConfig(default_max_bytes=100)
        new_s, results = enforce_size_limits(s, cfg)
        assert len(results) == 0
        assert new_s.attributes["input"] == 12345

    def test_truncation_result_fields(self):
        s = _span(input="x" * 500)
        cfg = SizeLimitConfig(default_max_bytes=100)
        _, results = enforce_size_limits(s, cfg)
        r = results[0]
        assert r.field_name == "input"
        assert r.original_bytes == 500
        assert r.truncated_bytes <= 100
        assert r.was_truncated is True

    def test_preserves_other_attributes(self):
        s = _span(input="hello", custom_field="keep me")
        cfg = SizeLimitConfig(default_max_bytes=10000)
        new_s, _ = enforce_size_limits(s, cfg)
        assert new_s.attributes["custom_field"] == "keep me"


# -- default_config --


class TestDefaultConfig:
    def test_returns_config(self):
        cfg = default_config()
        assert isinstance(cfg, SizeLimitConfig)

    def test_has_per_type_limits(self):
        cfg = default_config()
        assert "llm_call" in cfg.per_type_limits
        assert "tool_call" in cfg.per_type_limits

    def test_default_max_bytes(self):
        cfg = default_config()
        assert cfg.default_max_bytes == 10000
