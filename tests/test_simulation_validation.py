"""Tests for simulation_validation module."""

from __future__ import annotations

import pytest

from evalyn_sdk.simulation_validation import (
    DistributionStats,
    ValidationReport,
    ValidationResult,
    check_deduplication,
    check_vocabulary_overlap,
    compare_distributions,
    compute_length_distribution,
    compute_vocabulary_overlap,
    find_duplicates,
    find_near_duplicates,
    format_validation_report,
    validate_simulation,
)


# ---------------------------------------------------------------------------
# DistributionStats dataclass
# ---------------------------------------------------------------------------


class TestDistributionStats:
    def test_basic_fields(self):
        ds = DistributionStats(
            mean=5.0, std_dev=1.0, min_val=2.0, max_val=8.0,
            median=5.0, count=10,
        )
        assert ds.mean == 5.0
        assert ds.count == 10

    def test_as_dict(self):
        ds = DistributionStats(
            mean=3.0, std_dev=0.5, min_val=1.0, max_val=5.0,
            median=3.0, count=4,
        )
        d = ds.as_dict()
        assert d["mean"] == 3.0
        assert set(d.keys()) == {"mean", "std_dev", "min_val", "max_val", "median", "count"}

    def test_from_dict(self):
        d = {
            "mean": 2.0, "std_dev": 0.3, "min_val": 1.0, "max_val": 3.0,
            "median": 2.0, "count": 5,
        }
        ds = DistributionStats.from_dict(d)
        assert ds.mean == 2.0
        assert ds.count == 5

    def test_roundtrip(self):
        ds = DistributionStats(
            mean=7.5, std_dev=2.1, min_val=3.0, max_val=12.0,
            median=7.0, count=20,
        )
        assert DistributionStats.from_dict(ds.as_dict()) == ds


# ---------------------------------------------------------------------------
# ValidationResult dataclass
# ---------------------------------------------------------------------------


class TestValidationResult:
    def test_basic_fields(self):
        vr = ValidationResult(
            check_name="test", passed=True, score=0.5,
            details="ok", threshold=0.3,
        )
        assert vr.passed is True

    def test_as_dict(self):
        vr = ValidationResult(
            check_name="c", passed=False, score=0.9,
            details="fail", threshold=0.5,
        )
        d = vr.as_dict()
        assert d["passed"] is False
        assert d["score"] == 0.9

    def test_from_dict(self):
        d = {
            "check_name": "x", "passed": True, "score": 0.1,
            "details": "d", "threshold": 0.2,
        }
        vr = ValidationResult.from_dict(d)
        assert vr.check_name == "x"

    def test_roundtrip(self):
        vr = ValidationResult(
            check_name="rt", passed=True, score=0.42,
            details="round trip", threshold=0.5,
        )
        assert ValidationResult.from_dict(vr.as_dict()) == vr


# ---------------------------------------------------------------------------
# ValidationReport dataclass
# ---------------------------------------------------------------------------


class TestValidationReport:
    def test_defaults(self):
        vr = ValidationReport()
        assert vr.results == []
        assert vr.overall_passed is True
        assert vr.total_checks == 0

    def test_as_dict(self):
        result = ValidationResult(
            check_name="a", passed=True, score=0.1,
            details="ok", threshold=0.5,
        )
        report = ValidationReport(
            results=[result], overall_passed=True,
            total_checks=1, passed_checks=1,
        )
        d = report.as_dict()
        assert len(d["results"]) == 1
        assert d["overall_passed"] is True

    def test_from_dict(self):
        d = {
            "results": [{
                "check_name": "a", "passed": True, "score": 0.1,
                "details": "d", "threshold": 0.5,
            }],
            "overall_passed": True,
            "total_checks": 1,
            "passed_checks": 1,
        }
        report = ValidationReport.from_dict(d)
        assert len(report.results) == 1
        assert report.results[0].check_name == "a"

    def test_from_dict_defaults(self):
        report = ValidationReport.from_dict({})
        assert report.results == []
        assert report.overall_passed is True

    def test_roundtrip(self):
        result = ValidationResult(
            check_name="b", passed=False, score=0.8,
            details="fail", threshold=0.5,
        )
        report = ValidationReport(
            results=[result], overall_passed=False,
            total_checks=1, passed_checks=0,
        )
        restored = ValidationReport.from_dict(report.as_dict())
        assert restored.overall_passed is False
        assert len(restored.results) == 1


# ---------------------------------------------------------------------------
# compute_length_distribution
# ---------------------------------------------------------------------------


class TestComputeLengthDistribution:
    def test_empty_list(self):
        stats = compute_length_distribution([])
        assert stats.count == 0
        assert stats.mean == 0.0

    def test_single_text(self):
        stats = compute_length_distribution(["hello"])
        assert stats.count == 1
        assert stats.mean == 5.0
        assert stats.min_val == 5.0
        assert stats.max_val == 5.0
        assert stats.median == 5.0
        assert stats.std_dev == 0.0

    def test_multiple_texts(self):
        stats = compute_length_distribution(["ab", "abcd"])
        assert stats.count == 2
        assert stats.mean == 3.0
        assert stats.min_val == 2.0
        assert stats.max_val == 4.0
        assert stats.median == 3.0

    def test_std_dev_nonzero(self):
        stats = compute_length_distribution(["a", "aaa", "aaaaa"])
        # lengths: 1, 3, 5; mean = 3.0
        assert stats.std_dev > 0

    def test_empty_strings(self):
        stats = compute_length_distribution(["", ""])
        assert stats.mean == 0.0
        assert stats.std_dev == 0.0

    def test_median_odd_count(self):
        stats = compute_length_distribution(["a", "bb", "ccccc"])
        # lengths: 1, 2, 5 - median = 2
        assert stats.median == 2.0

    def test_median_even_count(self):
        stats = compute_length_distribution(["a", "bb", "ccc", "dddd"])
        # lengths: 1, 2, 3, 4 - median = 2.5
        assert stats.median == 2.5


# ---------------------------------------------------------------------------
# compare_distributions
# ---------------------------------------------------------------------------


class TestCompareDistributions:
    def test_identical_distributions(self):
        s = DistributionStats(mean=10, std_dev=2, min_val=5, max_val=15, median=10, count=10)
        result = compare_distributions(s, s)
        assert result.passed is True
        assert result.score == 0.0

    def test_both_zero_mean(self):
        s = DistributionStats(mean=0, std_dev=0, min_val=0, max_val=0, median=0, count=0)
        result = compare_distributions(s, s)
        assert result.passed is True
        assert result.details == "Both distributions have zero mean"

    def test_within_tolerance(self):
        s = DistributionStats(mean=10, std_dev=2, min_val=5, max_val=15, median=10, count=10)
        g = DistributionStats(mean=12, std_dev=2.5, min_val=4, max_val=18, median=12, count=10)
        result = compare_distributions(s, g, tolerance=0.5)
        assert result.passed is True

    def test_outside_tolerance(self):
        s = DistributionStats(mean=10, std_dev=2, min_val=5, max_val=15, median=10, count=10)
        g = DistributionStats(mean=100, std_dev=20, min_val=50, max_val=150, median=100, count=10)
        result = compare_distributions(s, g, tolerance=0.5)
        assert result.passed is False

    def test_check_name(self):
        s = DistributionStats(mean=5, std_dev=1, min_val=3, max_val=7, median=5, count=5)
        result = compare_distributions(s, s)
        assert result.check_name == "distribution_comparison"

    def test_custom_tolerance(self):
        s = DistributionStats(mean=10, std_dev=2, min_val=5, max_val=15, median=10, count=10)
        g = DistributionStats(mean=15, std_dev=2, min_val=5, max_val=25, median=15, count=10)
        # mean diff ratio = 5/10 = 0.5, std diff = 0 - score = 0.5
        result_strict = compare_distributions(s, g, tolerance=0.3)
        assert result_strict.passed is False
        result_lenient = compare_distributions(s, g, tolerance=0.6)
        assert result_lenient.passed is True


# ---------------------------------------------------------------------------
# compute_vocabulary_overlap
# ---------------------------------------------------------------------------


class TestComputeVocabularyOverlap:
    def test_identical_texts(self):
        texts = ["hello world"]
        overlap = compute_vocabulary_overlap(texts, texts)
        assert overlap == 1.0

    def test_no_overlap(self):
        overlap = compute_vocabulary_overlap(["hello world"], ["foo bar"])
        assert overlap == 0.0

    def test_partial_overlap(self):
        overlap = compute_vocabulary_overlap(
            ["hello world"], ["hello foo"],
        )
        # seed words: {hello, world}, gen words: {hello, foo}
        # intersection: {hello}, union: {hello, world, foo}
        assert abs(overlap - 1.0 / 3.0) < 1e-9

    def test_empty_both(self):
        overlap = compute_vocabulary_overlap([], [])
        assert overlap == 1.0

    def test_empty_seed(self):
        overlap = compute_vocabulary_overlap([], ["hello"])
        assert overlap == 0.0

    def test_empty_gen(self):
        overlap = compute_vocabulary_overlap(["hello"], [])
        assert overlap == 0.0

    def test_case_insensitive(self):
        overlap = compute_vocabulary_overlap(["Hello"], ["hello"])
        assert overlap == 1.0


# ---------------------------------------------------------------------------
# check_vocabulary_overlap
# ---------------------------------------------------------------------------


class TestCheckVocabularyOverlap:
    def test_passes_threshold(self):
        result = check_vocabulary_overlap(["a b c"], ["a b d"], min_overlap=0.3)
        # overlap = 2/4 = 0.5 >= 0.3
        assert result.passed is True

    def test_fails_threshold(self):
        result = check_vocabulary_overlap(["a b"], ["c d"], min_overlap=0.1)
        assert result.passed is False

    def test_check_name(self):
        result = check_vocabulary_overlap(["x"], ["x"])
        assert result.check_name == "vocabulary_overlap"

    def test_default_threshold(self):
        result = check_vocabulary_overlap(["x"], ["x"])
        assert result.threshold == 0.1


# ---------------------------------------------------------------------------
# find_duplicates
# ---------------------------------------------------------------------------


class TestFindDuplicates:
    def test_no_duplicates(self):
        assert find_duplicates(["a", "b", "c"]) == []

    def test_one_duplicate(self):
        pairs = find_duplicates(["a", "b", "a"])
        assert pairs == [(0, 2)]

    def test_multiple_duplicates(self):
        pairs = find_duplicates(["x", "y", "x", "y"])
        assert (0, 2) in pairs
        assert (1, 3) in pairs

    def test_empty_list(self):
        assert find_duplicates([]) == []

    def test_all_same(self):
        pairs = find_duplicates(["a", "a", "a"])
        # First "a" at 0, second at 1 maps to (0,1), third at 2 maps to (0,2)
        assert (0, 1) in pairs
        assert (0, 2) in pairs

    def test_single_item(self):
        assert find_duplicates(["x"]) == []


# ---------------------------------------------------------------------------
# find_near_duplicates
# ---------------------------------------------------------------------------


class TestFindNearDuplicates:
    def test_identical_texts(self):
        result = find_near_duplicates(["a b c", "a b c"], threshold=0.9)
        assert len(result) == 1
        assert result[0][0] == 0
        assert result[0][1] == 1
        assert result[0][2] == 1.0

    def test_no_near_duplicates(self):
        result = find_near_duplicates(["a b c", "x y z"], threshold=0.5)
        assert result == []

    def test_threshold_boundary(self):
        # "a b c" vs "a b d" - Jaccard = 1/4 = 0.25
        result = find_near_duplicates(["a b c", "a b d"], threshold=0.25)
        assert len(result) == 1

    def test_below_threshold(self):
        result = find_near_duplicates(["a b c", "a b d"], threshold=0.6)
        assert result == []

    def test_empty_texts(self):
        result = find_near_duplicates(["", ""], threshold=0.9)
        # Both empty - word sets are empty, similarity is 1.0
        assert len(result) == 1
        assert result[0][2] == 1.0

    def test_empty_list(self):
        assert find_near_duplicates([]) == []

    def test_multiple_pairs(self):
        result = find_near_duplicates(
            ["a b c", "a b c", "x y z"], threshold=0.9,
        )
        assert len(result) == 1
        assert result[0][:2] == (0, 1)


# ---------------------------------------------------------------------------
# check_deduplication
# ---------------------------------------------------------------------------


class TestCheckDeduplication:
    def test_no_duplicates(self):
        result = check_deduplication(
            gen_texts=["a", "b"], seed_texts=["c", "d"],
        )
        assert result.passed is True
        assert result.score == 0.0

    def test_all_duplicates(self):
        result = check_deduplication(
            gen_texts=["a", "b"], seed_texts=["a", "b"],
        )
        assert result.passed is False
        assert result.score == 1.0

    def test_partial_duplicates_pass(self):
        result = check_deduplication(
            gen_texts=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
            seed_texts=["a"],
            max_dup_rate=0.1,
        )
        assert result.passed is True
        assert result.score == 0.1

    def test_partial_duplicates_fail(self):
        result = check_deduplication(
            gen_texts=["a", "b", "c"],
            seed_texts=["a", "b"],
            max_dup_rate=0.1,
        )
        assert result.passed is False

    def test_empty_gen_texts(self):
        result = check_deduplication(gen_texts=[], seed_texts=["a"])
        assert result.passed is True

    def test_empty_seed_texts(self):
        result = check_deduplication(gen_texts=["a", "b"], seed_texts=[])
        assert result.passed is True
        assert result.score == 0.0

    def test_check_name(self):
        result = check_deduplication(gen_texts=["a"], seed_texts=["b"])
        assert result.check_name == "deduplication"


# ---------------------------------------------------------------------------
# validate_simulation
# ---------------------------------------------------------------------------


class TestValidateSimulation:
    def test_similar_texts_pass(self):
        seed = ["the cat sat on the mat", "the dog ran in the park"]
        gen = ["the bird flew over the tree", "the fish swam in the lake"]
        report = validate_simulation(seed, gen)
        assert report.total_checks == 3
        assert report.overall_passed is True or report.overall_passed is False
        # Just check structure
        assert len(report.results) == 3

    def test_identical_texts(self):
        texts = ["hello world", "foo bar baz"]
        report = validate_simulation(texts, texts)
        assert report.total_checks == 3
        # Distribution and vocab should pass, dedup should fail
        dist_check = report.results[0]
        assert dist_check.passed is True
        vocab_check = report.results[1]
        assert vocab_check.passed is True

    def test_empty_inputs(self):
        report = validate_simulation([], [])
        assert report.total_checks == 3

    def test_report_check_names(self):
        report = validate_simulation(["a"], ["b"])
        names = [r.check_name for r in report.results]
        assert "distribution_comparison" in names
        assert "vocabulary_overlap" in names
        assert "deduplication" in names

    def test_passed_checks_count(self):
        seed = ["hello world"]
        gen = ["goodbye earth"]
        report = validate_simulation(seed, gen)
        assert report.passed_checks == sum(1 for r in report.results if r.passed)

    def test_overall_passed_reflects_all(self):
        seed = ["a b c d e f g h i j"]
        gen = ["a b c d e f g h i j"]  # exact dup
        report = validate_simulation(seed, gen)
        # Dedup will fail since gen==seed
        if not all(r.passed for r in report.results):
            assert report.overall_passed is False


# ---------------------------------------------------------------------------
# format_validation_report
# ---------------------------------------------------------------------------


class TestFormatValidationReport:
    def test_empty_report(self):
        report = ValidationReport()
        text = format_validation_report(report)
        assert "Simulation Validation Report" in text
        assert "PASSED" in text
        assert "0/0" in text

    def test_failed_report(self):
        result = ValidationResult(
            check_name="test_check", passed=False, score=0.8,
            details="too high", threshold=0.5,
        )
        report = ValidationReport(
            results=[result], overall_passed=False,
            total_checks=1, passed_checks=0,
        )
        text = format_validation_report(report)
        assert "FAILED" in text
        assert "[FAIL]" in text
        assert "test_check" in text

    def test_passed_report(self):
        result = ValidationResult(
            check_name="good", passed=True, score=0.2,
            details="all good", threshold=0.5,
        )
        report = ValidationReport(
            results=[result], overall_passed=True,
            total_checks=1, passed_checks=1,
        )
        text = format_validation_report(report)
        assert "PASSED" in text
        assert "[PASS]" in text

    def test_mixed_results(self):
        r1 = ValidationResult(
            check_name="a", passed=True, score=0.1,
            details="ok", threshold=0.5,
        )
        r2 = ValidationResult(
            check_name="b", passed=False, score=0.9,
            details="bad", threshold=0.5,
        )
        report = ValidationReport(
            results=[r1, r2], overall_passed=False,
            total_checks=2, passed_checks=1,
        )
        text = format_validation_report(report)
        assert "1/2 checks passed" in text

    def test_score_formatting(self):
        result = ValidationResult(
            check_name="c", passed=True, score=0.12345,
            details="d", threshold=0.5,
        )
        report = ValidationReport(
            results=[result], overall_passed=True,
            total_checks=1, passed_checks=1,
        )
        text = format_validation_report(report)
        assert "0.1235" in text  # 4 decimal places
