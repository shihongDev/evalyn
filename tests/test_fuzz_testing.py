from __future__ import annotations

import json

import pytest

from evalyn_sdk.testing.fuzz_testing import (
    FuzzInput,
    FuzzReport,
    FuzzResult,
    find_crash_patterns,
    fuzz_parser,
    generate_fuzz_inputs,
    generate_injection_input,
    generate_malformed_json,
    generate_truncated_input,
    generate_unicode_input,
)


# --- generate_malformed_json ---


class TestGenerateMalformedJson:
    def test_produces_invalid_json(self):
        for seed in range(12):
            text = generate_malformed_json(seed)
            with pytest.raises((json.JSONDecodeError, ValueError)):
                json.loads(text)

    def test_returns_string(self):
        assert isinstance(generate_malformed_json(0), str)

    def test_different_seeds_can_differ(self):
        outputs = {generate_malformed_json(s) for s in range(20)}
        assert len(outputs) > 1


# --- generate_truncated_input ---


class TestGenerateTruncatedInput:
    def test_shorter_than_base(self):
        base = '{"score": 0.5}'
        result = generate_truncated_input(base, ratio=0.5)
        assert len(result) < len(base)

    def test_very_small_ratio(self):
        base = '{"score": 0.5}'
        result = generate_truncated_input(base, ratio=0.1)
        assert len(result) >= 1
        assert len(result) < len(base)

    def test_empty_base(self):
        result = generate_truncated_input("", ratio=0.5)
        assert result == ""

    def test_ratio_one_returns_full(self):
        base = '{"score": 0.5}'
        result = generate_truncated_input(base, ratio=1.0)
        assert result == base


# --- generate_unicode_input ---


class TestGenerateUnicodeInput:
    def test_non_ascii(self):
        result = generate_unicode_input(0)
        assert isinstance(result, str)
        # Should contain non-ASCII characters
        has_non_ascii = any(ord(c) > 127 for c in result)
        assert has_non_ascii

    def test_nonempty(self):
        result = generate_unicode_input(42)
        assert len(result) > 0

    def test_different_seeds(self):
        a = generate_unicode_input(0)
        b = generate_unicode_input(99)
        # Different seeds should produce different outputs (with high probability)
        outputs = {generate_unicode_input(s) for s in range(20)}
        assert len(outputs) > 1


# --- generate_injection_input ---


class TestGenerateInjectionInput:
    def test_has_injection_patterns(self):
        markers = [
            "DROP", "script", "alert", "IGNORE", "__import__", "system",
            "OR", "{{", "${", "cat /etc", "constructor",
        ]
        for seed in range(12):
            text = generate_injection_input(seed)
            assert any(m in text for m in markers), f"No injection pattern in: {text}"

    def test_returns_string(self):
        assert isinstance(generate_injection_input(0), str)

    def test_different_seeds_vary(self):
        outputs = {generate_injection_input(s) for s in range(20)}
        assert len(outputs) > 1


# --- generate_fuzz_inputs ---


class TestGenerateFuzzInputs:
    def test_correct_count(self):
        inputs = generate_fuzz_inputs(count=30, seed=0)
        assert len(inputs) == 30

    def test_default_count(self):
        inputs = generate_fuzz_inputs()
        assert len(inputs) == 50

    def test_deterministic_with_seed(self):
        a = generate_fuzz_inputs(count=20, seed=123)
        b = generate_fuzz_inputs(count=20, seed=123)
        for x, y in zip(a, b):
            assert x.input_text == y.input_text
            assert x.category == y.category

    def test_diverse_categories(self):
        inputs = generate_fuzz_inputs(count=50, seed=42)
        cats = {fi.category for fi in inputs}
        assert "malformed_json" in cats
        assert "unicode" in cats
        assert "injection" in cats

    def test_all_are_fuzz_input(self):
        inputs = generate_fuzz_inputs(count=10)
        for fi in inputs:
            assert isinstance(fi, FuzzInput)


# --- fuzz_parser ---


class TestFuzzParser:
    def test_catches_exceptions(self):
        def bad_parser(text: str):
            raise ValueError("parse error")

        inputs = [FuzzInput(input_text="test", category="random")]
        report = fuzz_parser(bad_parser, inputs)
        assert report.crashes == 1
        assert report.results[0].crashed is True
        assert "parse error" in report.results[0].error

    def test_no_crashes_on_robust_parser(self):
        def robust_parser(text: str):
            return {"raw": text}

        inputs = generate_fuzz_inputs(count=20, seed=0)
        report = fuzz_parser(robust_parser, inputs)
        assert report.crashes == 0
        assert report.crash_rate == 0.0

    def test_report_totals(self):
        def half_crash(text: str):
            if len(text) % 2 == 0:
                raise RuntimeError("boom")
            return text

        inputs = generate_fuzz_inputs(count=10, seed=0)
        report = fuzz_parser(half_crash, inputs)
        assert report.total_inputs == 10
        assert report.crashes + sum(
            1 for r in report.results if not r.crashed
        ) == 10

    def test_crash_rate_calculation(self):
        def always_crash(text: str):
            raise RuntimeError("fail")

        inputs = [FuzzInput(input_text=str(i)) for i in range(4)]
        report = fuzz_parser(always_crash, inputs)
        assert report.crash_rate == 1.0

    def test_empty_inputs(self):
        def parser(text: str):
            return text

        report = fuzz_parser(parser, [])
        assert report.total_inputs == 0
        assert report.crashes == 0
        assert report.crash_rate == 0.0

    def test_categories_tracked(self):
        def parser(text: str):
            return text

        inputs = [
            FuzzInput(input_text="a", category="unicode"),
            FuzzInput(input_text="b", category="injection"),
        ]
        report = fuzz_parser(parser, inputs)
        assert "unicode" in report.categories_tested
        assert "injection" in report.categories_tested


# --- find_crash_patterns ---


class TestFindCrashPatterns:
    def test_counts_by_category(self):
        results = [
            FuzzResult(input_text='{"bad json', parser_name="p", crashed=True, error="e"),
            FuzzResult(input_text='{"also bad,}', parser_name="p", crashed=True, error="e"),
            FuzzResult(input_text="<script>alert(1)</script>", parser_name="p", crashed=True, error="e"),
            FuzzResult(input_text="ok", parser_name="p", crashed=False),
        ]
        report = FuzzReport(results=results, total_inputs=4, crashes=3)
        patterns = find_crash_patterns(report)
        assert sum(patterns.values()) == 3

    def test_empty_report(self):
        report = FuzzReport()
        patterns = find_crash_patterns(report)
        assert patterns == {}

    def test_no_crashes(self):
        results = [
            FuzzResult(input_text="good", parser_name="p", crashed=False),
        ]
        report = FuzzReport(results=results, total_inputs=1, crashes=0)
        patterns = find_crash_patterns(report)
        assert patterns == {}


# --- FuzzInput dataclass ---


class TestFuzzInputDataclass:
    def test_as_dict(self):
        fi = FuzzInput(input_text="hello", category="random", seed=5)
        d = fi.as_dict()
        assert d["input_text"] == "hello"
        assert d["category"] == "random"
        assert d["seed"] == 5

    def test_defaults(self):
        fi = FuzzInput(input_text="x")
        assert fi.category == "random"
        assert fi.seed == 0


# --- FuzzResult dataclass ---


class TestFuzzResultDataclass:
    def test_as_dict(self):
        fr = FuzzResult(input_text="t", parser_name="p", crashed=True, error="err")
        d = fr.as_dict()
        assert d["crashed"] is True
        assert d["error"] == "err"

    def test_defaults(self):
        fr = FuzzResult(input_text="t", parser_name="p")
        assert fr.crashed is False
        assert fr.error == ""
        assert fr.output is None


# --- FuzzReport ---


class TestFuzzReport:
    def test_as_dict_from_dict_roundtrip(self):
        report = FuzzReport(
            results=[
                FuzzResult(input_text="a", parser_name="p", crashed=True, error="e"),
                FuzzResult(input_text="b", parser_name="p"),
            ],
            total_inputs=2,
            crashes=1,
            crash_rate=0.5,
            categories_tested=["random"],
        )
        d = report.as_dict()
        restored = FuzzReport.from_dict(d)
        assert restored.total_inputs == 2
        assert restored.crashes == 1
        assert restored.crash_rate == 0.5
        assert len(restored.results) == 2
        assert restored.results[0].crashed is True

    def test_format_text_includes_crash_details(self):
        report = FuzzReport(
            results=[
                FuzzResult(input_text="x", parser_name="myparser", crashed=True, error="boom"),
            ],
            total_inputs=1,
            crashes=1,
            crash_rate=1.0,
            categories_tested=["random"],
        )
        text = report.format_text()
        assert "Fuzz Report" in text
        assert "1" in text
        assert "myparser" in text
        assert "boom" in text

    def test_format_text_no_crashes(self):
        report = FuzzReport(
            total_inputs=5,
            crashes=0,
            crash_rate=0.0,
            categories_tested=["random"],
        )
        text = report.format_text()
        assert "Crashes: 0" in text

    def test_from_dict_empty(self):
        report = FuzzReport.from_dict({})
        assert report.total_inputs == 0
        assert report.results == []
