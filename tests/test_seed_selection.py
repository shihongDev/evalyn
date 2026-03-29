"""Tests for seed selection optimization module."""

from __future__ import annotations

import sys
from pathlib import Path

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.seed_selection import (
    SeedScore,
    SelectionResult,
    compute_text_diversity,
    score_seed,
    greedy_select,
    detect_redundant_seeds,
    format_selection_report,
)


# ---------------------------------------------------------------------------
# SeedScore dataclass
# ---------------------------------------------------------------------------


class TestSeedScore:
    def test_basic_construction(self):
        s = SeedScore(seed_id="s1", diversity_score=0.5, unique_outputs=3, near_duplicate_count=1)
        assert s.seed_id == "s1"
        assert s.diversity_score == 0.5
        assert s.unique_outputs == 3
        assert s.near_duplicate_count == 1

    def test_as_dict(self):
        s = SeedScore(seed_id="s1", diversity_score=0.75, unique_outputs=5, near_duplicate_count=2)
        d = s.as_dict()
        assert d["seed_id"] == "s1"
        assert d["diversity_score"] == 0.75
        assert d["unique_outputs"] == 5
        assert d["near_duplicate_count"] == 2

    def test_from_dict(self):
        d = {"seed_id": "s2", "diversity_score": 0.3, "unique_outputs": 2, "near_duplicate_count": 0}
        s = SeedScore.from_dict(d)
        assert s.seed_id == "s2"
        assert s.diversity_score == 0.3

    def test_roundtrip(self):
        original = SeedScore(seed_id="x", diversity_score=0.9, unique_outputs=10, near_duplicate_count=3)
        restored = SeedScore.from_dict(original.as_dict())
        assert restored.seed_id == original.seed_id
        assert restored.diversity_score == original.diversity_score
        assert restored.unique_outputs == original.unique_outputs
        assert restored.near_duplicate_count == original.near_duplicate_count


# ---------------------------------------------------------------------------
# SelectionResult dataclass
# ---------------------------------------------------------------------------


class TestSelectionResult:
    def test_basic_construction(self):
        r = SelectionResult(
            selected_seeds=["a", "b"],
            dropped_seeds=["c"],
            scores=[SeedScore("a", 0.5, 3, 0), SeedScore("b", 0.6, 4, 1)],
            coverage_improvement=0.8,
        )
        assert r.selected_seeds == ["a", "b"]
        assert r.coverage_improvement == 0.8

    def test_as_dict(self):
        r = SelectionResult(
            selected_seeds=["a"],
            dropped_seeds=[],
            scores=[SeedScore("a", 0.5, 3, 0)],
            coverage_improvement=1.0,
        )
        d = r.as_dict()
        assert d["selected_seeds"] == ["a"]
        assert len(d["scores"]) == 1
        assert d["scores"][0]["seed_id"] == "a"

    def test_from_dict(self):
        d = {
            "selected_seeds": ["x"],
            "dropped_seeds": ["y"],
            "scores": [
                {"seed_id": "x", "diversity_score": 0.5, "unique_outputs": 2, "near_duplicate_count": 0},
            ],
            "coverage_improvement": 0.7,
        }
        r = SelectionResult.from_dict(d)
        assert r.selected_seeds == ["x"]
        assert r.dropped_seeds == ["y"]
        assert len(r.scores) == 1

    def test_roundtrip(self):
        original = SelectionResult(
            selected_seeds=["a", "b"],
            dropped_seeds=["c"],
            scores=[SeedScore("a", 0.5, 3, 0), SeedScore("b", 0.6, 4, 1), SeedScore("c", 0.05, 1, 0)],
            coverage_improvement=0.85,
        )
        restored = SelectionResult.from_dict(original.as_dict())
        assert restored.selected_seeds == original.selected_seeds
        assert restored.dropped_seeds == original.dropped_seeds
        assert len(restored.scores) == len(original.scores)
        assert restored.coverage_improvement == original.coverage_improvement


# ---------------------------------------------------------------------------
# compute_text_diversity
# ---------------------------------------------------------------------------


class TestComputeTextDiversity:
    def test_empty_list(self):
        assert compute_text_diversity([]) == 0.0

    def test_single_text(self):
        assert compute_text_diversity(["hello world foo"]) == 0.0

    def test_identical_texts(self):
        texts = ["the quick brown fox", "the quick brown fox"]
        assert compute_text_diversity(texts) == 0.0

    def test_completely_different_texts(self):
        texts = ["alpha bravo charlie", "delta echo foxtrot"]
        diversity = compute_text_diversity(texts)
        assert diversity == 1.0  # no overlap

    def test_partial_overlap(self):
        texts = ["python machine learning", "python deep learning"]
        diversity = compute_text_diversity(texts)
        assert 0.0 < diversity < 1.0

    def test_multiple_texts(self):
        texts = [
            "cats dogs animals pets",
            "cars trucks vehicles transport",
            "books reading library novels",
        ]
        diversity = compute_text_diversity(texts)
        assert diversity > 0.5  # fairly different topics

    def test_stop_words_ignored(self):
        texts = ["the and but", "the and but"]
        # All words are stop words or too short, so word sets are empty
        assert compute_text_diversity(texts) == 0.0

    def test_short_words_filtered(self):
        texts = ["ab cd ef", "ab cd ef"]
        # All words are < 3 chars, so empty sets
        assert compute_text_diversity(texts) == 0.0

    def test_diversity_symmetry(self):
        texts_a = ["alpha bravo charlie", "delta echo foxtrot"]
        texts_b = ["delta echo foxtrot", "alpha bravo charlie"]
        assert compute_text_diversity(texts_a) == compute_text_diversity(texts_b)


# ---------------------------------------------------------------------------
# score_seed
# ---------------------------------------------------------------------------


class TestScoreSeed:
    def test_single_output(self):
        score = score_seed("s1", ["hello world foo bar"])
        assert score.seed_id == "s1"
        assert score.diversity_score == 0.0  # only one text
        assert score.unique_outputs == 1
        assert score.near_duplicate_count == 0

    def test_identical_outputs(self):
        score = score_seed("s2", ["alpha bravo charlie", "alpha bravo charlie"])
        assert score.unique_outputs == 1  # duplicates
        assert score.diversity_score == 0.0
        assert score.near_duplicate_count == 1  # one near-dup pair

    def test_diverse_outputs(self):
        outputs = [
            "machine learning algorithms optimization",
            "database systems query processing",
            "network security encryption protocols",
        ]
        score = score_seed("s3", outputs)
        assert score.diversity_score > 0.5
        assert score.unique_outputs == 3
        assert score.near_duplicate_count == 0

    def test_near_duplicates_detected(self):
        base = "python programming language features design patterns best practices algorithms optimization"
        outputs = [
            base,
            base + " overview",
        ]
        score = score_seed("s4", outputs)
        # 10 shared words out of 11 total unique = Jaccard ~0.909 > 0.9
        assert score.near_duplicate_count >= 1

    def test_empty_outputs(self):
        score = score_seed("s5", [])
        assert score.diversity_score == 0.0
        assert score.unique_outputs == 0
        assert score.near_duplicate_count == 0

    def test_score_returns_seed_id(self):
        score = score_seed("my_seed", ["foo bar baz"])
        assert score.seed_id == "my_seed"


# ---------------------------------------------------------------------------
# greedy_select
# ---------------------------------------------------------------------------


class TestGreedySelect:
    def test_empty_seeds(self):
        result = greedy_select({}, max_seeds=5)
        assert result.selected_seeds == []
        assert result.dropped_seeds == []
        assert result.coverage_improvement == 0.0

    def test_single_seed_selected(self):
        seeds = {"s1": ["alpha bravo charlie delta"]}
        result = greedy_select(seeds, max_seeds=5, min_diversity=0.0)
        assert "s1" in result.selected_seeds

    def test_max_seeds_limit(self):
        seeds = {
            "s1": ["alpha bravo charlie"],
            "s2": ["delta echo foxtrot"],
            "s3": ["golf hotel india"],
        }
        result = greedy_select(seeds, max_seeds=2, min_diversity=0.0)
        assert len(result.selected_seeds) <= 2

    def test_low_diversity_dropped(self):
        # One seed with identical outputs (diversity 0), one with diverse outputs
        seeds = {
            "boring": ["aaa bbb ccc", "aaa bbb ccc"],
            "diverse": ["alpha bravo charlie", "delta echo foxtrot"],
        }
        result = greedy_select(seeds, max_seeds=5, min_diversity=0.1)
        assert "diverse" in result.selected_seeds
        assert "boring" in result.dropped_seeds

    def test_greedy_picks_most_new_words(self):
        seeds = {
            "small": ["tiny word"],
            "large": ["alpha bravo charlie delta echo foxtrot golf hotel india"],
        }
        result = greedy_select(seeds, max_seeds=1, min_diversity=0.0)
        # large has more unique words, should be picked first
        assert result.selected_seeds[0] == "large"

    def test_coverage_improvement_ranges_zero_to_one(self):
        seeds = {
            "s1": ["alpha bravo charlie"],
            "s2": ["delta echo foxtrot"],
        }
        result = greedy_select(seeds, max_seeds=5, min_diversity=0.0)
        assert 0.0 <= result.coverage_improvement <= 1.0

    def test_all_seeds_selected_when_max_is_large(self):
        seeds = {
            "s1": ["alpha bravo charlie"],
            "s2": ["delta echo foxtrot"],
        }
        result = greedy_select(seeds, max_seeds=100, min_diversity=0.0)
        assert len(result.selected_seeds) == 2
        assert result.coverage_improvement == 1.0

    def test_scores_present_for_all_seeds(self):
        seeds = {
            "s1": ["alpha bravo charlie"],
            "s2": ["delta echo foxtrot"],
        }
        result = greedy_select(seeds, max_seeds=5, min_diversity=0.0)
        score_ids = {s.seed_id for s in result.scores}
        assert "s1" in score_ids
        assert "s2" in score_ids

    def test_dropped_seeds_not_in_selected(self):
        seeds = {
            "s1": ["alpha bravo charlie"],
            "s2": ["delta echo foxtrot"],
        }
        result = greedy_select(seeds, max_seeds=5, min_diversity=0.0)
        selected_set = set(result.selected_seeds)
        dropped_set = set(result.dropped_seeds)
        assert not selected_set.intersection(dropped_set)

    def test_incremental_coverage(self):
        seeds = {
            "s1": ["alpha bravo charlie"],
            "s2": ["alpha bravo delta"],  # overlaps s1, adds 'delta'
            "s3": ["echo foxtrot golf"],  # completely new words
        }
        result = greedy_select(seeds, max_seeds=2, min_diversity=0.0)
        # s1 and s3 should be selected (no overlap), not s2
        selected = set(result.selected_seeds)
        assert "s3" in selected or "s1" in selected


# ---------------------------------------------------------------------------
# detect_redundant_seeds
# ---------------------------------------------------------------------------


class TestDetectRedundantSeeds:
    def test_no_seeds(self):
        assert detect_redundant_seeds({}) == []

    def test_single_seed(self):
        assert detect_redundant_seeds({"s1": ["alpha bravo"]}) == []

    def test_identical_outputs(self):
        seeds = {
            "s1": ["alpha bravo charlie delta echo"],
            "s2": ["alpha bravo charlie delta echo"],
        }
        pairs = detect_redundant_seeds(seeds, threshold=0.8)
        assert len(pairs) == 1
        assert ("s1", "s2") in pairs

    def test_completely_different(self):
        seeds = {
            "s1": ["alpha bravo charlie"],
            "s2": ["delta echo foxtrot"],
        }
        pairs = detect_redundant_seeds(seeds, threshold=0.8)
        assert pairs == []

    def test_threshold_sensitivity(self):
        seeds = {
            "s1": ["alpha bravo charlie delta"],
            "s2": ["alpha bravo charlie echo"],
        }
        # With low threshold, should detect similarity
        pairs_low = detect_redundant_seeds(seeds, threshold=0.3)
        # With high threshold, might not
        pairs_high = detect_redundant_seeds(seeds, threshold=0.95)
        assert len(pairs_low) >= len(pairs_high)

    def test_multiple_redundant_pairs(self):
        seeds = {
            "s1": ["alpha bravo charlie delta echo"],
            "s2": ["alpha bravo charlie delta echo"],
            "s3": ["alpha bravo charlie delta echo"],
        }
        pairs = detect_redundant_seeds(seeds, threshold=0.8)
        # All three pairs should be redundant
        assert len(pairs) == 3

    def test_sorted_pair_order(self):
        seeds = {
            "b_seed": ["alpha bravo charlie delta echo"],
            "a_seed": ["alpha bravo charlie delta echo"],
        }
        pairs = detect_redundant_seeds(seeds, threshold=0.8)
        # Pairs should be in sorted order
        assert pairs[0][0] < pairs[0][1]


# ---------------------------------------------------------------------------
# format_selection_report
# ---------------------------------------------------------------------------


class TestFormatSelectionReport:
    def test_empty_result(self):
        result = SelectionResult(
            selected_seeds=[], dropped_seeds=[], scores=[], coverage_improvement=0.0,
        )
        report = format_selection_report(result)
        assert "SEED SELECTION REPORT" in report
        assert "Selected: 0" in report

    def test_includes_selected_seeds(self):
        result = SelectionResult(
            selected_seeds=["s1"],
            dropped_seeds=[],
            scores=[SeedScore("s1", 0.5, 3, 0)],
            coverage_improvement=0.8,
        )
        report = format_selection_report(result)
        assert "s1" in report
        assert "diversity=0.500" in report

    def test_includes_dropped_seeds(self):
        result = SelectionResult(
            selected_seeds=[],
            dropped_seeds=["s2"],
            scores=[SeedScore("s2", 0.01, 1, 0)],
            coverage_improvement=0.0,
        )
        report = format_selection_report(result)
        assert "Dropped Seeds:" in report
        assert "s2" in report

    def test_coverage_shown(self):
        result = SelectionResult(
            selected_seeds=["s1"],
            dropped_seeds=[],
            scores=[SeedScore("s1", 0.5, 3, 0)],
            coverage_improvement=0.85,
        )
        report = format_selection_report(result)
        assert "85.0%" in report

    def test_report_is_string(self):
        result = SelectionResult(
            selected_seeds=[], dropped_seeds=[], scores=[], coverage_improvement=0.0,
        )
        assert isinstance(format_selection_report(result), str)
