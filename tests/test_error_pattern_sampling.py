"""Tests for error-pattern sampling module."""

from __future__ import annotations

import pytest

from evalyn_sdk.error_pattern_sampling import (
    ErrorPattern,
    ErrorPatternResult,
    PatternMatch,
    compute_pattern_match,
    extract_patterns_from_failures,
    format_error_pattern_report,
    match_items_to_patterns,
    run_error_pattern_sampling,
    sample_by_error_patterns,
)


# ---- ErrorPattern ----


class TestErrorPattern:
    def test_defaults(self):
        p = ErrorPattern(pattern_id="p0", description="timeout error")
        assert p.pattern_id == "p0"
        assert p.description == "timeout error"
        assert p.keywords == []
        assert p.severity == "medium"

    def test_custom_values(self):
        p = ErrorPattern(
            pattern_id="p1",
            description="parse failure",
            keywords=["parse", "json"],
            severity="high",
        )
        assert p.keywords == ["parse", "json"]
        assert p.severity == "high"

    def test_as_dict(self):
        p = ErrorPattern(
            pattern_id="p0",
            description="timeout",
            keywords=["timeout", "server"],
            severity="low",
        )
        d = p.as_dict()
        assert d == {
            "pattern_id": "p0",
            "description": "timeout",
            "keywords": ["timeout", "server"],
            "severity": "low",
        }

    def test_from_dict_full(self):
        d = {
            "pattern_id": "p2",
            "description": "auth error",
            "keywords": ["auth", "token"],
            "severity": "high",
        }
        p = ErrorPattern.from_dict(d)
        assert p.pattern_id == "p2"
        assert p.keywords == ["auth", "token"]

    def test_from_dict_defaults(self):
        p = ErrorPattern.from_dict({"pattern_id": "p0", "description": "x"})
        assert p.keywords == []
        assert p.severity == "medium"

    def test_roundtrip(self):
        p = ErrorPattern(
            pattern_id="p3",
            description="crash",
            keywords=["crash", "null"],
            severity="high",
        )
        assert ErrorPattern.from_dict(p.as_dict()) == p


# ---- PatternMatch ----


class TestPatternMatch:
    def test_creation(self):
        m = PatternMatch(item_id="i1", pattern_id="p0", match_score=0.75)
        assert m.item_id == "i1"
        assert m.match_score == 0.75

    def test_as_dict(self):
        m = PatternMatch(item_id="i2", pattern_id="p1", match_score=0.5)
        d = m.as_dict()
        assert d == {"item_id": "i2", "pattern_id": "p1", "match_score": 0.5}

    def test_from_dict(self):
        d = {"item_id": "i3", "pattern_id": "p2", "match_score": 1.0}
        m = PatternMatch.from_dict(d)
        assert m.item_id == "i3"
        assert m.match_score == 1.0

    def test_roundtrip(self):
        m = PatternMatch(item_id="i4", pattern_id="p3", match_score=0.33)
        assert PatternMatch.from_dict(m.as_dict()) == m


# ---- ErrorPatternResult ----


class TestErrorPatternResult:
    def test_defaults(self):
        r = ErrorPatternResult()
        assert r.selected_ids == []
        assert r.matches == []
        assert r.patterns_covered == 0

    def test_as_dict(self):
        r = ErrorPatternResult(
            selected_ids=["a", "b"],
            matches=[PatternMatch(item_id="a", pattern_id="p0", match_score=0.9)],
            patterns_covered=1,
            total_patterns=2,
            total_pool=10,
        )
        d = r.as_dict()
        assert d["selected_ids"] == ["a", "b"]
        assert len(d["matches"]) == 1
        assert d["patterns_covered"] == 1

    def test_from_dict(self):
        d = {
            "selected_ids": ["x"],
            "matches": [{"item_id": "x", "pattern_id": "p0", "match_score": 0.8}],
            "patterns_covered": 1,
            "total_patterns": 3,
            "total_pool": 20,
        }
        r = ErrorPatternResult.from_dict(d)
        assert r.selected_ids == ["x"]
        assert len(r.matches) == 1
        assert r.matches[0].match_score == 0.8

    def test_from_dict_empty(self):
        r = ErrorPatternResult.from_dict({})
        assert r.selected_ids == []
        assert r.matches == []

    def test_roundtrip(self):
        r = ErrorPatternResult(
            selected_ids=["a"],
            matches=[PatternMatch("a", "p0", 0.5)],
            patterns_covered=1,
            total_patterns=1,
            total_pool=5,
        )
        assert ErrorPatternResult.from_dict(r.as_dict()).selected_ids == r.selected_ids


# ---- extract_patterns_from_failures ----


class TestExtractPatterns:
    def test_basic_extraction(self):
        failures = [
            {"failure_reason": "timeout error"},
            {"failure_reason": "timeout error"},
            {"failure_reason": "parse failure"},
        ]
        patterns = extract_patterns_from_failures(failures)
        assert len(patterns) == 2
        descs = {p.description for p in patterns}
        assert "timeout error" in descs
        assert "parse failure" in descs

    def test_severity_assignment_high(self):
        failures = [{"failure_reason": "crash"} for _ in range(5)]
        patterns = extract_patterns_from_failures(failures)
        assert len(patterns) == 1
        assert patterns[0].severity == "high"

    def test_severity_assignment_medium(self):
        failures = [{"failure_reason": "error"} for _ in range(3)]
        patterns = extract_patterns_from_failures(failures)
        assert patterns[0].severity == "medium"

    def test_severity_assignment_low(self):
        failures = [{"failure_reason": "rare bug"}]
        patterns = extract_patterns_from_failures(failures)
        assert patterns[0].severity == "low"

    def test_empty_failures(self):
        assert extract_patterns_from_failures([]) == []

    def test_missing_reason_field(self):
        failures = [{"other_field": "something"}]
        patterns = extract_patterns_from_failures(failures)
        assert patterns == []

    def test_custom_reason_field(self):
        failures = [{"error_msg": "network timeout"}]
        patterns = extract_patterns_from_failures(failures, reason_field="error_msg")
        assert len(patterns) == 1
        assert "network" in patterns[0].keywords

    def test_keywords_are_lowercase(self):
        failures = [{"failure_reason": "JSON Parse Error"}]
        patterns = extract_patterns_from_failures(failures)
        for kw in patterns[0].keywords:
            assert kw == kw.lower()

    def test_keywords_sorted(self):
        failures = [{"failure_reason": "connection timeout retry"}]
        patterns = extract_patterns_from_failures(failures)
        assert patterns[0].keywords == sorted(patterns[0].keywords)

    def test_deduplicates_same_reason(self):
        failures = [
            {"failure_reason": "timeout"},
            {"failure_reason": "timeout"},
            {"failure_reason": "timeout"},
        ]
        patterns = extract_patterns_from_failures(failures)
        assert len(patterns) == 1


# ---- compute_pattern_match ----


class TestComputePatternMatch:
    def test_full_match(self):
        p = ErrorPattern(pattern_id="p0", description="x", keywords=["timeout", "error"])
        score = compute_pattern_match("This is a timeout error", p)
        assert score == 1.0

    def test_partial_match(self):
        p = ErrorPattern(pattern_id="p0", description="x", keywords=["timeout", "error"])
        score = compute_pattern_match("This is a timeout", p)
        assert score == 0.5

    def test_no_match(self):
        p = ErrorPattern(pattern_id="p0", description="x", keywords=["timeout", "error"])
        score = compute_pattern_match("Everything is fine", p)
        assert score == 0.0

    def test_case_insensitive(self):
        p = ErrorPattern(pattern_id="p0", description="x", keywords=["timeout"])
        score = compute_pattern_match("TIMEOUT occurred", p)
        assert score == 1.0

    def test_empty_keywords(self):
        p = ErrorPattern(pattern_id="p0", description="x", keywords=[])
        score = compute_pattern_match("anything", p)
        assert score == 0.0

    def test_empty_text(self):
        p = ErrorPattern(pattern_id="p0", description="x", keywords=["timeout"])
        score = compute_pattern_match("", p)
        assert score == 0.0


# ---- match_items_to_patterns ----


class TestMatchItemsToPatterns:
    def test_basic_matching(self):
        items = {
            "i1": "timeout error occurred",
            "i2": "parse failure in json",
            "i3": "all good",
        }
        patterns = [
            ErrorPattern(pattern_id="p0", description="timeout", keywords=["timeout", "error"]),
        ]
        matches = match_items_to_patterns(items, patterns)
        assert len(matches) == 1
        assert matches[0].item_id == "i1"
        assert matches[0].match_score == 1.0

    def test_min_match_filtering(self):
        items = {"i1": "timeout happened"}
        patterns = [
            ErrorPattern(pattern_id="p0", description="x", keywords=["timeout", "error", "server"]),
        ]
        # 1/3 = 0.33, threshold 0.3
        matches = match_items_to_patterns(items, patterns, min_match=0.3)
        assert len(matches) == 1

        # threshold 0.5 should filter it out
        matches = match_items_to_patterns(items, patterns, min_match=0.5)
        assert len(matches) == 0

    def test_sorted_by_score_descending(self):
        items = {
            "i1": "timeout",
            "i2": "timeout error",
        }
        patterns = [
            ErrorPattern(pattern_id="p0", description="x", keywords=["timeout", "error"]),
        ]
        matches = match_items_to_patterns(items, patterns, min_match=0.3)
        assert len(matches) == 2
        assert matches[0].match_score >= matches[1].match_score

    def test_empty_items(self):
        patterns = [ErrorPattern(pattern_id="p0", description="x", keywords=["timeout"])]
        assert match_items_to_patterns({}, patterns) == []

    def test_empty_patterns(self):
        items = {"i1": "something"}
        assert match_items_to_patterns(items, []) == []

    def test_multiple_patterns(self):
        items = {"i1": "timeout error and parse failure"}
        patterns = [
            ErrorPattern(pattern_id="p0", description="timeout", keywords=["timeout"]),
            ErrorPattern(pattern_id="p1", description="parse", keywords=["parse"]),
        ]
        matches = match_items_to_patterns(items, patterns, min_match=0.3)
        pattern_ids = {m.pattern_id for m in matches}
        assert "p0" in pattern_ids
        assert "p1" in pattern_ids


# ---- sample_by_error_patterns ----


class TestSampleByErrorPatterns:
    def test_basic_sampling(self):
        matches = [
            PatternMatch(item_id="i1", pattern_id="p0", match_score=1.0),
            PatternMatch(item_id="i2", pattern_id="p1", match_score=0.8),
            PatternMatch(item_id="i3", pattern_id="p0", match_score=0.6),
        ]
        selected = sample_by_error_patterns(matches, sample_size=3)
        assert len(selected) == 3
        # i1 and i2 should be first (one per pattern)
        assert "i1" in selected
        assert "i2" in selected
        assert "i3" in selected

    def test_pattern_coverage_first(self):
        matches = [
            PatternMatch(item_id="i1", pattern_id="p0", match_score=1.0),
            PatternMatch(item_id="i2", pattern_id="p1", match_score=0.5),
        ]
        selected = sample_by_error_patterns(matches, sample_size=2)
        # Both patterns should be covered
        assert "i1" in selected
        assert "i2" in selected

    def test_respects_sample_size(self):
        matches = [
            PatternMatch(item_id=f"i{k}", pattern_id="p0", match_score=0.5)
            for k in range(10)
        ]
        selected = sample_by_error_patterns(matches, sample_size=3)
        assert len(selected) == 3

    def test_empty_matches(self):
        assert sample_by_error_patterns([], sample_size=10) == []

    def test_sample_size_larger_than_matches(self):
        matches = [
            PatternMatch(item_id="i1", pattern_id="p0", match_score=1.0),
            PatternMatch(item_id="i2", pattern_id="p1", match_score=0.8),
        ]
        selected = sample_by_error_patterns(matches, sample_size=100)
        assert len(selected) == 2

    def test_deterministic_with_seed(self):
        matches = [
            PatternMatch(item_id=f"i{k}", pattern_id=f"p{k % 3}", match_score=0.5)
            for k in range(20)
        ]
        a = sample_by_error_patterns(matches, sample_size=5, seed=42)
        b = sample_by_error_patterns(matches, sample_size=5, seed=42)
        assert a == b

    def test_one_per_pattern_minimum(self):
        matches = [
            PatternMatch(item_id="i1", pattern_id="p0", match_score=1.0),
            PatternMatch(item_id="i2", pattern_id="p0", match_score=0.9),
            PatternMatch(item_id="i3", pattern_id="p1", match_score=0.4),
        ]
        selected = sample_by_error_patterns(matches, sample_size=2)
        pattern_items = {}
        for m in matches:
            if m.item_id in selected:
                pattern_items.setdefault(m.pattern_id, []).append(m.item_id)
        # Both patterns should have at least one item
        assert "p0" in pattern_items
        assert "p1" in pattern_items


# ---- run_error_pattern_sampling ----


class TestRunErrorPatternSampling:
    def test_basic_run(self):
        items = {
            "i1": "timeout error on server",
            "i2": "parse failure in json handler",
            "i3": "everything works fine",
        }
        patterns = [
            ErrorPattern(pattern_id="p0", description="timeout", keywords=["timeout", "error"]),
            ErrorPattern(pattern_id="p1", description="parse", keywords=["parse", "failure"]),
        ]
        result = run_error_pattern_sampling(items, patterns, sample_size=3)
        assert isinstance(result, ErrorPatternResult)
        assert result.total_pool == 3
        assert result.total_patterns == 2
        assert result.patterns_covered <= 2

    def test_empty_items(self):
        patterns = [ErrorPattern(pattern_id="p0", description="x", keywords=["timeout"])]
        result = run_error_pattern_sampling({}, patterns)
        assert result.selected_ids == []
        assert result.patterns_covered == 0

    def test_empty_patterns(self):
        items = {"i1": "something"}
        result = run_error_pattern_sampling(items, [])
        assert result.selected_ids == []
        assert result.total_patterns == 0

    def test_coverage_counted_correctly(self):
        items = {
            "i1": "timeout error",
            "i2": "parse failure",
        }
        patterns = [
            ErrorPattern(pattern_id="p0", description="timeout", keywords=["timeout", "error"]),
            ErrorPattern(pattern_id="p1", description="parse", keywords=["parse", "failure"]),
        ]
        result = run_error_pattern_sampling(items, patterns, sample_size=10)
        assert result.patterns_covered == 2

    def test_result_matches_are_for_selected_only(self):
        items = {
            "i1": "timeout error",
            "i2": "another timeout error",
            "i3": "no match at all",
        }
        patterns = [
            ErrorPattern(pattern_id="p0", description="timeout", keywords=["timeout", "error"]),
        ]
        result = run_error_pattern_sampling(items, patterns, sample_size=1)
        for m in result.matches:
            assert m.item_id in result.selected_ids


# ---- format_error_pattern_report ----


class TestFormatErrorPatternReport:
    def test_basic_format(self):
        result = ErrorPatternResult(
            selected_ids=["i1", "i2"],
            matches=[PatternMatch("i1", "p0", 0.9)],
            patterns_covered=1,
            total_patterns=2,
            total_pool=10,
        )
        report = format_error_pattern_report(result)
        assert "Error Pattern Sampling Report" in report
        assert "Total pool size: 10" in report
        assert "Selected items: 2" in report
        assert "Patterns covered: 1/2" in report
        assert "i1" in report
        assert "i2" in report

    def test_empty_result(self):
        result = ErrorPatternResult()
        report = format_error_pattern_report(result)
        assert "Error Pattern Sampling Report" in report
        assert "Selected items: 0" in report

    def test_match_scores_in_report(self):
        result = ErrorPatternResult(
            selected_ids=["i1"],
            matches=[PatternMatch("i1", "p0", 0.85)],
            patterns_covered=1,
            total_patterns=1,
            total_pool=5,
        )
        report = format_error_pattern_report(result)
        assert "0.85" in report
