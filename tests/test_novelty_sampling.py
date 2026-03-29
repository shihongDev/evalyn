"""Tests for novelty-based sampling module."""

from __future__ import annotations

import pytest

from evalyn_sdk.novelty_sampling import (
    NoveltyConfig,
    NoveltyResult,
    NoveltyScore,
    compute_novelty,
    format_novelty_report,
    iterative_novelty_sampling,
    run_novelty_sampling,
    score_all_items,
    select_most_novel,
)


# ---------------------------------------------------------------------------
# NoveltyScore
# ---------------------------------------------------------------------------


class TestNoveltyScore:
    def test_defaults(self):
        s = NoveltyScore(item_id="a", novelty=0.8)
        assert s.item_id == "a"
        assert s.novelty == 0.8
        assert s.nearest_labeled_id == ""

    def test_custom_nearest(self):
        s = NoveltyScore(item_id="a", novelty=0.5, nearest_labeled_id="b")
        assert s.nearest_labeled_id == "b"

    def test_as_dict(self):
        s = NoveltyScore(item_id="x", novelty=0.9, nearest_labeled_id="y")
        d = s.as_dict()
        assert d == {
            "item_id": "x",
            "novelty": 0.9,
            "nearest_labeled_id": "y",
        }

    def test_from_dict_full(self):
        d = {"item_id": "a", "novelty": 0.3, "nearest_labeled_id": "b"}
        s = NoveltyScore.from_dict(d)
        assert s.item_id == "a"
        assert s.novelty == 0.3
        assert s.nearest_labeled_id == "b"

    def test_from_dict_minimal(self):
        d = {"item_id": "a", "novelty": 0.5}
        s = NoveltyScore.from_dict(d)
        assert s.nearest_labeled_id == ""

    def test_roundtrip(self):
        s = NoveltyScore(item_id="z", novelty=0.7, nearest_labeled_id="q")
        assert NoveltyScore.from_dict(s.as_dict()) == s


# ---------------------------------------------------------------------------
# NoveltyConfig
# ---------------------------------------------------------------------------


class TestNoveltyConfig:
    def test_defaults(self):
        c = NoveltyConfig()
        assert c.sample_size == 50
        assert c.seed is None

    def test_custom_values(self):
        c = NoveltyConfig(sample_size=10, seed=42)
        assert c.sample_size == 10
        assert c.seed == 42

    def test_as_dict(self):
        c = NoveltyConfig(sample_size=20, seed=7)
        d = c.as_dict()
        assert d == {"sample_size": 20, "seed": 7}

    def test_from_dict_full(self):
        d = {"sample_size": 30, "seed": 99}
        c = NoveltyConfig.from_dict(d)
        assert c.sample_size == 30
        assert c.seed == 99

    def test_from_dict_minimal(self):
        c = NoveltyConfig.from_dict({})
        assert c.sample_size == 50
        assert c.seed is None

    def test_roundtrip(self):
        c = NoveltyConfig(sample_size=15, seed=3)
        c2 = NoveltyConfig.from_dict(c.as_dict())
        assert c2.sample_size == c.sample_size
        assert c2.seed == c.seed


# ---------------------------------------------------------------------------
# NoveltyResult
# ---------------------------------------------------------------------------


class TestNoveltyResult:
    def test_defaults(self):
        r = NoveltyResult()
        assert r.selected_ids == []
        assert r.scores == []
        assert r.mean_novelty == 0.0
        assert r.total_unlabeled == 0
        assert r.total_labeled == 0

    def test_as_dict(self):
        r = NoveltyResult(
            selected_ids=["a"],
            scores=[NoveltyScore(item_id="a", novelty=0.8)],
            mean_novelty=0.8,
            total_unlabeled=5,
            total_labeled=3,
        )
        d = r.as_dict()
        assert d["selected_ids"] == ["a"]
        assert len(d["scores"]) == 1
        assert d["mean_novelty"] == 0.8

    def test_from_dict(self):
        d = {
            "selected_ids": ["x"],
            "scores": [{"item_id": "x", "novelty": 0.6}],
            "mean_novelty": 0.6,
            "total_unlabeled": 10,
            "total_labeled": 5,
        }
        r = NoveltyResult.from_dict(d)
        assert r.selected_ids == ["x"]
        assert len(r.scores) == 1
        assert r.total_labeled == 5

    def test_from_dict_empty(self):
        r = NoveltyResult.from_dict({})
        assert r.selected_ids == []
        assert r.mean_novelty == 0.0

    def test_roundtrip(self):
        r = NoveltyResult(
            selected_ids=["a"],
            scores=[NoveltyScore(item_id="a", novelty=0.5, nearest_labeled_id="b")],
            mean_novelty=0.5,
            total_unlabeled=3,
            total_labeled=2,
        )
        r2 = NoveltyResult.from_dict(r.as_dict())
        assert r2.selected_ids == r.selected_ids
        assert r2.mean_novelty == r.mean_novelty


# ---------------------------------------------------------------------------
# compute_novelty
# ---------------------------------------------------------------------------


class TestComputeNovelty:
    def test_empty_labeled_set(self):
        novelty, idx = compute_novelty("hello world", [])
        assert novelty == 1.0
        assert idx == -1

    def test_identical_text(self):
        novelty, idx = compute_novelty("hello world", ["hello world"])
        assert novelty == pytest.approx(0.0)
        assert idx == 0

    def test_completely_different(self):
        novelty, idx = compute_novelty("aaa bbb", ["ccc ddd"])
        assert novelty == pytest.approx(1.0)
        assert idx == 0

    def test_partial_overlap(self):
        novelty, idx = compute_novelty("hello world foo", ["hello world bar"])
        # Jaccard: intersection={"hello","world"}=2, union={"hello","world","foo","bar"}=4
        # similarity = 0.5, novelty = 0.5
        assert novelty == pytest.approx(0.5)
        assert idx == 0

    def test_nearest_labeled_index(self):
        labeled = ["aaa bbb", "hello world", "ccc ddd"]
        novelty, idx = compute_novelty("hello world extra", labeled)
        # Most similar to labeled[1] = "hello world"
        assert idx == 1

    def test_case_insensitive(self):
        novelty, _ = compute_novelty("Hello World", ["hello world"])
        assert novelty == pytest.approx(0.0)

    def test_empty_text(self):
        novelty, idx = compute_novelty("", ["hello"])
        # empty vs non-empty: Jaccard = 0, novelty = 1
        assert novelty == pytest.approx(1.0)

    def test_both_empty(self):
        novelty, idx = compute_novelty("", [""])
        # Both empty: Jaccard = 0, novelty = 1
        assert novelty == pytest.approx(1.0)

    def test_multiple_labeled_picks_best(self):
        labeled = ["alpha beta", "alpha beta gamma"]
        novelty, idx = compute_novelty("alpha beta gamma delta", labeled)
        # More overlap with labeled[1]: {"alpha","beta","gamma"}/{"alpha","beta","gamma","delta"} = 3/4
        # vs labeled[0]: {"alpha","beta"}/{"alpha","beta","gamma","delta"} = 2/4
        assert idx == 1
        assert novelty == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# score_all_items
# ---------------------------------------------------------------------------


class TestScoreAllItems:
    def test_basic_scoring(self):
        unlabeled = {"u1": "aaa bbb", "u2": "ccc ddd"}
        labeled = {"l1": "aaa bbb"}
        scores = score_all_items(unlabeled, labeled)
        assert len(scores) == 2
        # u1 is identical to l1 (novelty 0), u2 is totally different (novelty 1)
        ids = [s.item_id for s in scores]
        # Sorted by novelty desc
        assert ids[0] == "u2"
        assert ids[1] == "u1"

    def test_empty_unlabeled(self):
        scores = score_all_items({}, {"l1": "hello"})
        assert scores == []

    def test_empty_labeled(self):
        scores = score_all_items({"u1": "hello"}, {})
        assert len(scores) == 1
        assert scores[0].novelty == 1.0
        assert scores[0].nearest_labeled_id == ""

    def test_nearest_labeled_id_set(self):
        unlabeled = {"u1": "hello world"}
        labeled = {"l1": "aaa", "l2": "hello world"}
        scores = score_all_items(unlabeled, labeled)
        assert scores[0].nearest_labeled_id == "l2"

    def test_sorted_by_novelty_desc(self):
        unlabeled = {
            "u1": "aaa",
            "u2": "aaa bbb",
            "u3": "ccc ddd eee",
        }
        labeled = {"l1": "aaa bbb"}
        scores = score_all_items(unlabeled, labeled)
        novelties = [s.novelty for s in scores]
        assert novelties == sorted(novelties, reverse=True)


# ---------------------------------------------------------------------------
# select_most_novel
# ---------------------------------------------------------------------------


class TestSelectMostNovel:
    def test_basic(self):
        scores = [
            NoveltyScore(item_id="a", novelty=0.3),
            NoveltyScore(item_id="b", novelty=0.9),
            NoveltyScore(item_id="c", novelty=0.6),
        ]
        selected = select_most_novel(scores, 2)
        assert selected == ["b", "c"]

    def test_n_larger_than_list(self):
        scores = [NoveltyScore(item_id="a", novelty=0.5)]
        selected = select_most_novel(scores, 10)
        assert selected == ["a"]

    def test_n_zero(self):
        scores = [NoveltyScore(item_id="a", novelty=0.5)]
        selected = select_most_novel(scores, 0)
        assert selected == []

    def test_empty_scores(self):
        selected = select_most_novel([], 5)
        assert selected == []

    def test_tie_broken_by_id(self):
        scores = [
            NoveltyScore(item_id="b", novelty=0.5),
            NoveltyScore(item_id="a", novelty=0.5),
        ]
        selected = select_most_novel(scores, 2)
        assert selected == ["a", "b"]


# ---------------------------------------------------------------------------
# run_novelty_sampling
# ---------------------------------------------------------------------------


class TestRunNoveltySampling:
    def test_basic_pipeline(self):
        unlabeled = {
            "u1": "the quick brown fox",
            "u2": "completely unique words here",
            "u3": "the quick brown dog",
        }
        labeled = {"l1": "the quick brown cat"}
        config = NoveltyConfig(sample_size=2)
        result = run_novelty_sampling(unlabeled, labeled, config)
        assert len(result.selected_ids) == 2
        assert result.total_unlabeled == 3
        assert result.total_labeled == 1
        # u2 should be most novel (no overlap)
        assert result.selected_ids[0] == "u2"

    def test_sample_size_larger_than_pool(self):
        unlabeled = {"u1": "hello", "u2": "world"}
        labeled = {"l1": "foo"}
        config = NoveltyConfig(sample_size=10)
        result = run_novelty_sampling(unlabeled, labeled, config)
        assert len(result.selected_ids) == 2

    def test_mean_novelty_calculated(self):
        unlabeled = {"u1": "aaa bbb", "u2": "ccc ddd"}
        labeled = {"l1": "aaa bbb"}
        config = NoveltyConfig(sample_size=2)
        result = run_novelty_sampling(unlabeled, labeled, config)
        # u1 novelty=0, u2 novelty=1, mean=(0+1)/2=0.5
        assert result.mean_novelty == pytest.approx(0.5)

    def test_empty_unlabeled(self):
        config = NoveltyConfig(sample_size=5)
        result = run_novelty_sampling({}, {"l1": "hello"}, config)
        assert result.selected_ids == []
        assert result.total_unlabeled == 0
        assert result.mean_novelty == 0.0

    def test_empty_labeled(self):
        unlabeled = {"u1": "hello", "u2": "world"}
        config = NoveltyConfig(sample_size=1)
        result = run_novelty_sampling(unlabeled, {}, config)
        assert len(result.selected_ids) == 1
        assert result.total_labeled == 0
        assert result.mean_novelty == pytest.approx(1.0)

    def test_scores_included(self):
        unlabeled = {"u1": "hello", "u2": "world"}
        labeled = {"l1": "hello"}
        config = NoveltyConfig(sample_size=2)
        result = run_novelty_sampling(unlabeled, labeled, config)
        assert len(result.scores) == 2


# ---------------------------------------------------------------------------
# iterative_novelty_sampling
# ---------------------------------------------------------------------------


class TestIterativeNoveltySampling:
    def test_basic_iterative(self):
        unlabeled = {f"u{i}": f"word{i} extra{i}" for i in range(20)}
        labeled = {"l1": "word0 extra0"}
        selected = iterative_novelty_sampling(
            unlabeled, labeled, n_rounds=2, per_round=5
        )
        assert len(selected) == 10

    def test_single_round(self):
        unlabeled = {"u1": "aaa", "u2": "bbb", "u3": "ccc"}
        labeled = {"l1": "ddd"}
        selected = iterative_novelty_sampling(
            unlabeled, labeled, n_rounds=1, per_round=2
        )
        assert len(selected) == 2

    def test_more_rounds_than_items(self):
        unlabeled = {"u1": "aaa", "u2": "bbb"}
        labeled = {"l1": "ccc"}
        selected = iterative_novelty_sampling(
            unlabeled, labeled, n_rounds=10, per_round=1
        )
        assert len(selected) == 2
        assert set(selected) == {"u1", "u2"}

    def test_empty_unlabeled(self):
        selected = iterative_novelty_sampling(
            {}, {"l1": "hello"}, n_rounds=3, per_round=5
        )
        assert selected == []

    def test_empty_labeled(self):
        unlabeled = {"u1": "aaa", "u2": "bbb"}
        selected = iterative_novelty_sampling(
            unlabeled, {}, n_rounds=1, per_round=2
        )
        assert len(selected) == 2

    def test_no_duplicates(self):
        unlabeled = {f"u{i}": f"text{i}" for i in range(30)}
        labeled = {"l1": "reference text"}
        selected = iterative_novelty_sampling(
            unlabeled, labeled, n_rounds=3, per_round=5
        )
        assert len(selected) == len(set(selected))

    def test_original_dicts_not_mutated(self):
        unlabeled = {"u1": "aaa", "u2": "bbb"}
        labeled = {"l1": "ccc"}
        unlabeled_copy = dict(unlabeled)
        labeled_copy = dict(labeled)
        iterative_novelty_sampling(unlabeled, labeled, n_rounds=2, per_round=1)
        assert unlabeled == unlabeled_copy
        assert labeled == labeled_copy

    def test_per_round_larger_than_remaining(self):
        unlabeled = {"u1": "aaa", "u2": "bbb", "u3": "ccc"}
        labeled = {"l1": "ddd"}
        selected = iterative_novelty_sampling(
            unlabeled, labeled, n_rounds=2, per_round=10
        )
        assert len(selected) == 3


# ---------------------------------------------------------------------------
# format_novelty_report
# ---------------------------------------------------------------------------


class TestFormatNoveltyReport:
    def test_report_contains_header(self):
        result = NoveltyResult(
            selected_ids=["u1"],
            scores=[NoveltyScore(item_id="u1", novelty=0.8, nearest_labeled_id="l1")],
            mean_novelty=0.8,
            total_unlabeled=5,
            total_labeled=3,
        )
        report = format_novelty_report(result)
        assert "Novelty Sampling Report" in report
        assert "unlabeled=5" in report
        assert "labeled=3" in report

    def test_report_contains_items(self):
        result = NoveltyResult(
            selected_ids=["u1", "u2"],
            scores=[
                NoveltyScore(item_id="u1", novelty=0.9, nearest_labeled_id="l1"),
                NoveltyScore(item_id="u2", novelty=0.5, nearest_labeled_id="l2"),
            ],
            mean_novelty=0.7,
            total_unlabeled=10,
            total_labeled=5,
        )
        report = format_novelty_report(result)
        assert "u1" in report
        assert "u2" in report
        assert "l1" in report

    def test_report_shows_mean_novelty(self):
        result = NoveltyResult(mean_novelty=0.1234)
        report = format_novelty_report(result)
        assert "0.1234" in report

    def test_report_empty_result(self):
        result = NoveltyResult()
        report = format_novelty_report(result)
        assert "Novelty Sampling Report" in report

    def test_report_is_string(self):
        result = NoveltyResult(
            selected_ids=["a"],
            scores=[NoveltyScore(item_id="a", novelty=0.5)],
            mean_novelty=0.5,
        )
        report = format_novelty_report(result)
        assert isinstance(report, str)
        assert len(report) > 0
