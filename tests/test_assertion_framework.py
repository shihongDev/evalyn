from __future__ import annotations

import pytest

from evalyn_sdk.testing.assertion_framework import (
    AssertionConfig,
    AssertionResult,
    all_passed,
    assert_contains,
    assert_equals,
    assert_json_valid,
    assert_matches_regex,
    assert_max_length,
    assert_min_length,
    assert_not_contains,
    assert_score_above,
    assert_score_below,
    run_assertions,
)


# --- assert_contains ---


class TestAssertContains:
    def test_pass(self):
        r = assert_contains("hello world", "world")
        assert r.passed is True
        assert r.assertion_type == "contains"

    def test_fail(self):
        r = assert_contains("hello world", "xyz")
        assert r.passed is False
        assert "xyz" in r.message

    def test_empty_substring(self):
        r = assert_contains("anything", "")
        assert r.passed is True


# --- assert_not_contains ---


class TestAssertNotContains:
    def test_pass(self):
        r = assert_not_contains("hello world", "xyz")
        assert r.passed is True

    def test_fail(self):
        r = assert_not_contains("hello world", "hello")
        assert r.passed is False
        assert "hello" in r.message

    def test_empty_output(self):
        r = assert_not_contains("", "something")
        assert r.passed is True


# --- assert_matches_regex ---


class TestAssertMatchesRegex:
    def test_pass(self):
        r = assert_matches_regex("score: 0.95", r"\d+\.\d+")
        assert r.passed is True

    def test_fail(self):
        r = assert_matches_regex("no numbers here", r"^\d+$")
        assert r.passed is False

    def test_invalid_regex(self):
        r = assert_matches_regex("test", "[invalid")
        assert r.passed is False
        assert "Invalid regex" in r.message


# --- assert_json_valid ---


class TestAssertJsonValid:
    def test_valid(self):
        r = assert_json_valid('{"score": 0.5}')
        assert r.passed is True

    def test_invalid(self):
        r = assert_json_valid("{bad json}")
        assert r.passed is False
        assert "Invalid JSON" in r.message

    def test_valid_array(self):
        r = assert_json_valid("[1, 2, 3]")
        assert r.passed is True

    def test_empty_string(self):
        r = assert_json_valid("")
        assert r.passed is False


# --- assert_min_length ---


class TestAssertMinLength:
    def test_pass(self):
        r = assert_min_length("hello", 3)
        assert r.passed is True

    def test_fail(self):
        r = assert_min_length("hi", 5)
        assert r.passed is False

    def test_exact(self):
        r = assert_min_length("abc", 3)
        assert r.passed is True


# --- assert_max_length ---


class TestAssertMaxLength:
    def test_pass(self):
        r = assert_max_length("hi", 5)
        assert r.passed is True

    def test_fail(self):
        r = assert_max_length("hello world", 5)
        assert r.passed is False

    def test_exact(self):
        r = assert_max_length("abc", 3)
        assert r.passed is True


# --- assert_equals ---


class TestAssertEquals:
    def test_pass(self):
        r = assert_equals("hello", "hello")
        assert r.passed is True

    def test_fail(self):
        r = assert_equals("hello", "world")
        assert r.passed is False

    def test_empty_strings(self):
        r = assert_equals("", "")
        assert r.passed is True


# --- assert_score_above ---


class TestAssertScoreAbove:
    def test_pass(self):
        r = assert_score_above(0.9, 0.5)
        assert r.passed is True

    def test_fail(self):
        r = assert_score_above(0.3, 0.5)
        assert r.passed is False

    def test_equal_fails(self):
        r = assert_score_above(0.5, 0.5)
        assert r.passed is False


# --- assert_score_below ---


class TestAssertScoreBelow:
    def test_pass(self):
        r = assert_score_below(0.3, 0.5)
        assert r.passed is True

    def test_fail(self):
        r = assert_score_below(0.9, 0.5)
        assert r.passed is False

    def test_equal_fails(self):
        r = assert_score_below(0.5, 0.5)
        assert r.passed is False


# --- run_assertions ---


class TestRunAssertions:
    def test_from_config(self):
        config = AssertionConfig(assertions=[
            {"type": "contains", "value": "hello"},
            {"type": "min_length", "value": 3},
        ])
        results = run_assertions("hello world", config)
        assert len(results) == 2
        assert results[0].passed is True
        assert results[1].passed is True

    def test_mixed_pass_fail(self):
        config = AssertionConfig(assertions=[
            {"type": "contains", "value": "hello"},
            {"type": "contains", "value": "xyz"},
        ])
        results = run_assertions("hello world", config)
        assert results[0].passed is True
        assert results[1].passed is False

    def test_score_assertions(self):
        config = AssertionConfig(assertions=[
            {"type": "score_above", "value": 0.5},
            {"type": "score_below", "value": 1.0},
        ])
        results = run_assertions("output", config, score=0.8)
        assert results[0].passed is True
        assert results[1].passed is True

    def test_empty_config(self):
        config = AssertionConfig()
        results = run_assertions("output", config)
        assert results == []

    def test_unknown_type(self):
        config = AssertionConfig(assertions=[
            {"type": "nonexistent", "value": "x"},
        ])
        results = run_assertions("output", config)
        assert len(results) == 1
        assert results[0].passed is False
        assert "Unknown" in results[0].message

    def test_json_valid_assertion(self):
        config = AssertionConfig(assertions=[
            {"type": "json_valid"},
        ])
        results = run_assertions('{"key": "val"}', config)
        assert results[0].passed is True

    def test_all_types(self):
        config = AssertionConfig(assertions=[
            {"type": "contains", "value": "hello"},
            {"type": "not_contains", "value": "xyz"},
            {"type": "matches_regex", "value": "h.*o"},
            {"type": "min_length", "value": 1},
            {"type": "max_length", "value": 100},
            {"type": "equals", "value": "hello"},
            {"type": "score_above", "value": 0.1},
            {"type": "score_below", "value": 0.9},
        ])
        results = run_assertions("hello", config, score=0.5)
        assert all(r.passed for r in results)


# --- all_passed ---


class TestAllPassed:
    def test_true(self):
        results = [
            AssertionResult(passed=True, assertion_type="contains"),
            AssertionResult(passed=True, assertion_type="min_length"),
        ]
        assert all_passed(results) is True

    def test_false(self):
        results = [
            AssertionResult(passed=True, assertion_type="contains"),
            AssertionResult(passed=False, assertion_type="min_length"),
        ]
        assert all_passed(results) is False

    def test_empty(self):
        assert all_passed([]) is True


# --- AssertionConfig ---


class TestAssertionConfig:
    def test_from_dict(self):
        data = {"assertions": [{"type": "contains", "value": "x"}]}
        config = AssertionConfig.from_dict(data)
        assert len(config.assertions) == 1
        assert config.assertions[0]["type"] == "contains"

    def test_from_dict_empty(self):
        config = AssertionConfig.from_dict({})
        assert config.assertions == []

    def test_as_dict_roundtrip(self):
        config = AssertionConfig(assertions=[
            {"type": "contains", "value": "test"},
        ])
        d = config.as_dict()
        restored = AssertionConfig.from_dict(d)
        assert restored.assertions == config.assertions


# --- AssertionResult ---


class TestAssertionResult:
    def test_as_dict(self):
        r = AssertionResult(
            passed=True,
            assertion_type="contains",
            message="ok",
            actual="text",
            expected="te",
        )
        d = r.as_dict()
        assert d["passed"] is True
        assert d["assertion_type"] == "contains"
        assert d["actual"] == "text"

    def test_defaults(self):
        r = AssertionResult(passed=False, assertion_type="test")
        assert r.message == ""
        assert r.actual is None
        assert r.expected is None


# --- Edge cases ---


class TestEdgeCases:
    def test_empty_output_contains(self):
        r = assert_contains("", "something")
        assert r.passed is False

    def test_empty_output_not_contains(self):
        r = assert_not_contains("", "something")
        assert r.passed is True

    def test_empty_output_json_valid(self):
        r = assert_json_valid("")
        assert r.passed is False

    def test_empty_output_min_length(self):
        r = assert_min_length("", 0)
        assert r.passed is True

    def test_empty_output_max_length(self):
        r = assert_max_length("", 0)
        assert r.passed is True
