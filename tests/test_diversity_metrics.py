"""Tests for diversity_metrics module."""

from __future__ import annotations

import pytest

from evalyn_sdk.diversity_metrics import (
    DiversityReport,
    DiversityScore,
    build_diversity_report,
    compare_diversity,
    compute_length_diversity,
    compute_novelty_score,
    compute_pairwise_distance,
    compute_uniqueness_ratio,
    compute_vocabulary_spread,
    format_diversity_report,
)


# ---------------------------------------------------------------------------
# DiversityScore dataclass
# ---------------------------------------------------------------------------


class TestDiversityScore:
    def test_basic_fields(self):
        s = DiversityScore(metric_name="vocab", score=0.75, details="test")
        assert s.metric_name == "vocab"
        assert s.score == 0.75
        assert s.details == "test"

    def test_as_dict(self):
        s = DiversityScore("vocab", 0.8, "details here")
        d = s.as_dict()
        assert d == {"metric_name": "vocab", "score": 0.8, "details": "details here"}

    def test_from_dict(self):
        s = DiversityScore.from_dict({"metric_name": "x", "score": 0.5, "details": "d"})
        assert s.metric_name == "x"
        assert s.score == 0.5

    def test_roundtrip(self):
        original = DiversityScore("test", 0.42, "info")
        restored = DiversityScore.from_dict(original.as_dict())
        assert restored == original


# ---------------------------------------------------------------------------
# DiversityReport dataclass
# ---------------------------------------------------------------------------


class TestDiversityReport:
    def test_defaults(self):
        r = DiversityReport()
        assert r.scores == []
        assert r.overall_diversity == 0.0
        assert r.total_items == 0

    def test_as_dict(self):
        r = DiversityReport(
            scores=[DiversityScore("a", 0.5, "d")],
            overall_diversity=0.5,
            total_items=10,
        )
        d = r.as_dict()
        assert d["overall_diversity"] == 0.5
        assert d["total_items"] == 10
        assert len(d["scores"]) == 1

    def test_from_dict(self):
        data = {
            "scores": [{"metric_name": "a", "score": 0.5, "details": "d"}],
            "overall_diversity": 0.5,
            "total_items": 10,
        }
        r = DiversityReport.from_dict(data)
        assert len(r.scores) == 1
        assert r.total_items == 10

    def test_roundtrip(self):
        r = DiversityReport(
            scores=[DiversityScore("a", 0.1, "x"), DiversityScore("b", 0.9, "y")],
            overall_diversity=0.5,
            total_items=5,
        )
        restored = DiversityReport.from_dict(r.as_dict())
        assert restored == r


# ---------------------------------------------------------------------------
# compute_vocabulary_spread
# ---------------------------------------------------------------------------


class TestComputeVocabularySpread:
    def test_empty(self):
        assert compute_vocabulary_spread([]) == 0.0

    def test_single_word(self):
        assert compute_vocabulary_spread(["hello"]) == 1.0

    def test_repeated_words(self):
        result = compute_vocabulary_spread(["the the the the"])
        assert result == pytest.approx(0.25)

    def test_all_unique(self):
        result = compute_vocabulary_spread(["one two three four"])
        assert result == pytest.approx(1.0)

    def test_multiple_texts(self):
        result = compute_vocabulary_spread(["hello world", "hello there"])
        # tokens: hello, world, hello, there => 3 unique / 4 total = 0.75
        assert result == pytest.approx(0.75)

    def test_empty_strings(self):
        assert compute_vocabulary_spread(["", ""]) == 0.0


# ---------------------------------------------------------------------------
# compute_pairwise_distance
# ---------------------------------------------------------------------------


class TestComputePairwiseDistance:
    def test_empty(self):
        assert compute_pairwise_distance([]) == 0.0

    def test_single(self):
        assert compute_pairwise_distance(["hello"]) == 0.0

    def test_identical(self):
        result = compute_pairwise_distance(["hello world", "hello world"])
        assert result == pytest.approx(0.0)

    def test_completely_different(self):
        result = compute_pairwise_distance(["aaa bbb", "ccc ddd"])
        # No overlap: Jaccard distance = 1.0
        assert result == pytest.approx(1.0)

    def test_partial_overlap(self):
        result = compute_pairwise_distance(["hello world", "hello there"])
        # sets: {hello, world} {hello, there} => intersection=1, union=3 => sim=1/3, dist=2/3
        assert result == pytest.approx(2 / 3, abs=0.01)

    def test_multiple_texts(self):
        texts = ["a b c", "d e f", "a b c"]
        result = compute_pairwise_distance(texts)
        # pairs: (0,1)=1.0, (0,2)=0.0, (1,2)=1.0 => mean=2/3
        assert result == pytest.approx(2 / 3, abs=0.01)


# ---------------------------------------------------------------------------
# compute_uniqueness_ratio
# ---------------------------------------------------------------------------


class TestComputeUniquenessRatio:
    def test_empty_gen(self):
        assert compute_uniqueness_ratio([], ["seed"]) == 0.0

    def test_empty_seed(self):
        assert compute_uniqueness_ratio(["gen"], []) == 1.0

    def test_identical_to_seed(self):
        result = compute_uniqueness_ratio(["hello world"], ["hello world"])
        # Jaccard sim = 1.0, which is >= 0.5 => not unique
        assert result == 0.0

    def test_completely_different(self):
        result = compute_uniqueness_ratio(["aaa bbb ccc"], ["xxx yyy zzz"])
        # Jaccard sim = 0.0, which is < 0.5 => unique
        assert result == 1.0

    def test_mixed(self):
        gen = ["hello world", "completely different text here"]
        seed = ["hello world"]
        result = compute_uniqueness_ratio(gen, seed)
        # First: sim=1.0 (not unique). Second: very low sim (unique).
        assert result == pytest.approx(0.5)

    def test_partial_overlap(self):
        gen = ["hello world foo bar baz"]
        seed = ["hello world"]
        # gen set: {hello, world, foo, bar, baz}, seed set: {hello, world}
        # sim = 2/5 = 0.4, which is < 0.5 => unique
        result = compute_uniqueness_ratio(gen, seed)
        assert result == 1.0


# ---------------------------------------------------------------------------
# compute_novelty_score
# ---------------------------------------------------------------------------


class TestComputeNoveltyScore:
    def test_empty_gen(self):
        assert compute_novelty_score([], ["seed"]) == 0.0

    def test_empty_seed(self):
        assert compute_novelty_score(["gen"], []) == 1.0

    def test_identical(self):
        result = compute_novelty_score(["hello world"], ["hello world"])
        # sim = 1.0 >= 0.3 => not novel
        assert result == 0.0

    def test_completely_different(self):
        result = compute_novelty_score(["aaa bbb"], ["xxx yyy"])
        # sim = 0.0 < 0.3 => novel
        assert result == 1.0

    def test_custom_threshold(self):
        gen = ["hello world foo bar"]
        seed = ["hello world"]
        # sim = 2/4 = 0.5
        result_low = compute_novelty_score(gen, seed, threshold=0.6)
        result_high = compute_novelty_score(gen, seed, threshold=0.4)
        assert result_low == 1.0  # 0.5 < 0.6 => novel
        assert result_high == 0.0  # 0.5 >= 0.4 => not novel

    def test_multiple_seeds(self):
        gen = ["aaa bbb ccc"]
        seed = ["xxx yyy", "aaa bbb ccc"]
        # max sim is 1.0 (against second seed) => not novel
        result = compute_novelty_score(gen, seed)
        assert result == 0.0


# ---------------------------------------------------------------------------
# compute_length_diversity
# ---------------------------------------------------------------------------


class TestComputeLengthDiversity:
    def test_empty(self):
        assert compute_length_diversity([]) == 0.0

    def test_single(self):
        assert compute_length_diversity(["hello"]) == 0.0

    def test_identical_lengths(self):
        result = compute_length_diversity(["abc", "def", "ghi"])
        assert result == pytest.approx(0.0)

    def test_varied_lengths(self):
        result = compute_length_diversity(["a", "abcdefghij"])
        # lengths: 1, 10 => mean=5.5, var=20.25, stdev=4.5, cv=4.5/5.5
        assert result == pytest.approx(4.5 / 5.5, abs=0.01)

    def test_all_empty(self):
        result = compute_length_diversity(["", ""])
        assert result == 0.0

    def test_three_texts(self):
        result = compute_length_diversity(["a", "bb", "ccc"])
        # lengths: 1, 2, 3 => mean=2.0
        # var = ((1-2)^2 + (2-2)^2 + (3-2)^2) / 3 = 2/3
        # stdev = sqrt(2/3) ~= 0.8165
        # cv = 0.8165 / 2.0 ~= 0.4082
        assert result == pytest.approx(0.4082, abs=0.01)


# ---------------------------------------------------------------------------
# build_diversity_report
# ---------------------------------------------------------------------------


class TestBuildDiversityReport:
    def test_basic(self):
        gen = ["hello world", "foo bar baz", "completely different text"]
        seed = ["seed text here"]
        report = build_diversity_report(gen, seed)
        assert report.total_items == 3
        assert len(report.scores) == 5
        assert report.overall_diversity >= 0.0

    def test_empty_gen(self):
        report = build_diversity_report([], ["seed"])
        assert report.total_items == 0

    def test_empty_seed(self):
        report = build_diversity_report(["gen text"], [])
        assert report.total_items == 1

    def test_metric_names(self):
        report = build_diversity_report(["a", "b"], ["c"])
        names = {s.metric_name for s in report.scores}
        expected = {"vocabulary_spread", "pairwise_distance", "uniqueness_ratio", "novelty_score", "length_diversity"}
        assert names == expected

    def test_overall_is_mean(self):
        report = build_diversity_report(["hello world", "foo bar"], ["seed"])
        expected_mean = sum(s.score for s in report.scores) / len(report.scores)
        assert report.overall_diversity == pytest.approx(expected_mean)


# ---------------------------------------------------------------------------
# format_diversity_report
# ---------------------------------------------------------------------------


class TestFormatDiversityReport:
    def test_basic(self):
        report = build_diversity_report(["a b c", "d e f"], ["x y z"])
        text = format_diversity_report(report)
        assert "Diversity Report" in text
        assert "2 items" in text
        assert "Overall diversity" in text

    def test_empty_report(self):
        report = DiversityReport()
        text = format_diversity_report(report)
        assert "0 items" in text

    def test_contains_metrics(self):
        report = build_diversity_report(["hello", "world"], ["seed"])
        text = format_diversity_report(report)
        assert "vocabulary_spread" in text
        assert "pairwise_distance" in text


# ---------------------------------------------------------------------------
# compare_diversity
# ---------------------------------------------------------------------------


class TestCompareDiversity:
    def test_basic(self):
        a = build_diversity_report(["hello world"], ["seed"])
        b = build_diversity_report(["hello world", "foo bar baz"], ["seed"])
        text = compare_diversity(a, b)
        assert "Diversity Comparison" in text
        assert "Report A" in text
        assert "Report B" in text

    def test_higher_lower(self):
        a = DiversityReport(
            scores=[DiversityScore("test", 0.3, "d")],
            overall_diversity=0.3,
            total_items=1,
        )
        b = DiversityReport(
            scores=[DiversityScore("test", 0.7, "d")],
            overall_diversity=0.7,
            total_items=2,
        )
        text = compare_diversity(a, b)
        assert "higher" in text

    def test_lower(self):
        a = DiversityReport(
            scores=[DiversityScore("test", 0.9, "d")],
            overall_diversity=0.9,
            total_items=5,
        )
        b = DiversityReport(
            scores=[DiversityScore("test", 0.1, "d")],
            overall_diversity=0.1,
            total_items=5,
        )
        text = compare_diversity(a, b)
        assert "lower" in text

    def test_same(self):
        a = DiversityReport(
            scores=[DiversityScore("test", 0.5, "d")],
            overall_diversity=0.5,
            total_items=3,
        )
        b = DiversityReport(
            scores=[DiversityScore("test", 0.5, "d")],
            overall_diversity=0.5,
            total_items=3,
        )
        text = compare_diversity(a, b)
        assert "same" in text
