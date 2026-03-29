"""Tests for human disagreement sampling module."""

from __future__ import annotations

import pytest

from evalyn_sdk.disagreement_sampling import (
    DisagreementConfig,
    DisagreementResult,
    DisagreementScore,
    compute_disagreement,
    compute_severity,
    format_disagreement_report,
    run_disagreement_sampling,
    sample_by_disagreement,
    score_items,
)


# ---- DisagreementScore ----


class TestDisagreementScore:
    def test_create(self):
        s = DisagreementScore(item_id="a", score=0.5, n_annotators=3, label_spread=0.8)
        assert s.item_id == "a"
        assert s.score == 0.5
        assert s.n_annotators == 3
        assert s.label_spread == 0.8

    def test_as_dict(self):
        s = DisagreementScore(item_id="b", score=0.3, n_annotators=2, label_spread=0.4)
        d = s.as_dict()
        assert d == {
            "item_id": "b",
            "score": 0.3,
            "n_annotators": 2,
            "label_spread": 0.4,
        }

    def test_from_dict(self):
        d = {"item_id": "c", "score": 0.9, "n_annotators": 5, "label_spread": 1.0}
        s = DisagreementScore.from_dict(d)
        assert s.item_id == "c"
        assert s.score == 0.9
        assert s.n_annotators == 5

    def test_roundtrip(self):
        s = DisagreementScore(item_id="x", score=0.7, n_annotators=4, label_spread=0.6)
        assert DisagreementScore.from_dict(s.as_dict()) == s


# ---- DisagreementConfig ----


class TestDisagreementConfig:
    def test_defaults(self):
        c = DisagreementConfig()
        assert c.sample_size == 50
        assert c.min_annotators == 2
        assert c.severity_weight == 1.0
        assert c.seed is None

    def test_custom(self):
        c = DisagreementConfig(sample_size=10, min_annotators=3, severity_weight=2.0, seed=42)
        assert c.sample_size == 10
        assert c.min_annotators == 3
        assert c.severity_weight == 2.0
        assert c.seed == 42

    def test_as_dict(self):
        c = DisagreementConfig(seed=7)
        d = c.as_dict()
        assert d == {
            "sample_size": 50,
            "min_annotators": 2,
            "severity_weight": 1.0,
            "seed": 7,
        }

    def test_from_dict_full(self):
        d = {"sample_size": 20, "min_annotators": 4, "severity_weight": 0.5, "seed": 99}
        c = DisagreementConfig.from_dict(d)
        assert c.sample_size == 20
        assert c.min_annotators == 4
        assert c.severity_weight == 0.5
        assert c.seed == 99

    def test_from_dict_defaults(self):
        c = DisagreementConfig.from_dict({})
        assert c.sample_size == 50
        assert c.min_annotators == 2
        assert c.severity_weight == 1.0
        assert c.seed is None

    def test_roundtrip(self):
        c = DisagreementConfig(sample_size=30, min_annotators=3, severity_weight=1.5, seed=5)
        assert DisagreementConfig.from_dict(c.as_dict()) == c


# ---- DisagreementResult ----


class TestDisagreementResult:
    def test_defaults(self):
        r = DisagreementResult()
        assert r.selected_ids == []
        assert r.scores == []
        assert r.mean_disagreement == 0.0
        assert r.total_pool == 0

    def test_as_dict(self):
        s = DisagreementScore(item_id="a", score=0.5, n_annotators=2, label_spread=0.3)
        r = DisagreementResult(
            selected_ids=["a"],
            scores=[s],
            mean_disagreement=0.5,
            total_pool=1,
        )
        d = r.as_dict()
        assert d["selected_ids"] == ["a"]
        assert len(d["scores"]) == 1
        assert d["mean_disagreement"] == 0.5
        assert d["total_pool"] == 1

    def test_from_dict(self):
        d = {
            "selected_ids": ["x", "y"],
            "scores": [
                {"item_id": "x", "score": 0.8, "n_annotators": 3, "label_spread": 0.5},
            ],
            "mean_disagreement": 0.8,
            "total_pool": 5,
        }
        r = DisagreementResult.from_dict(d)
        assert r.selected_ids == ["x", "y"]
        assert len(r.scores) == 1
        assert r.scores[0].item_id == "x"

    def test_from_dict_defaults(self):
        r = DisagreementResult.from_dict({})
        assert r.selected_ids == []
        assert r.scores == []
        assert r.mean_disagreement == 0.0
        assert r.total_pool == 0

    def test_roundtrip(self):
        s = DisagreementScore(item_id="a", score=0.5, n_annotators=2, label_spread=0.3)
        r = DisagreementResult(
            selected_ids=["a"],
            scores=[s],
            mean_disagreement=0.5,
            total_pool=1,
        )
        r2 = DisagreementResult.from_dict(r.as_dict())
        assert r2.selected_ids == r.selected_ids
        assert r2.mean_disagreement == r.mean_disagreement
        assert r2.total_pool == r.total_pool


# ---- compute_disagreement ----


class TestComputeDisagreement:
    def test_empty_labels(self):
        assert compute_disagreement([]) == 0.0

    def test_single_label(self):
        assert compute_disagreement([0.5]) == 0.0

    def test_identical_labels(self):
        assert compute_disagreement([0.7, 0.7, 0.7]) == 0.0

    def test_max_disagreement_two_labels(self):
        # Two labels at extremes: [0.0, 1.0]
        # std = 0.5, max_std = 0.5, ratio = 1.0
        result = compute_disagreement([0.0, 1.0])
        assert result == pytest.approx(1.0)

    def test_moderate_disagreement(self):
        # [0.0, 0.5, 1.0]: mean=0.5, var=(0.25+0+0.25)/3=0.1667, std=0.408
        # max_std = 0.5, ratio = 0.816
        result = compute_disagreement([0.0, 0.5, 1.0])
        assert 0.7 < result < 0.9

    def test_low_disagreement(self):
        # With equal spread from center, disagreement < 1.0
        # [0.0, 0.5, 1.0]: mean=0.5, std=0.408, max_std=0.5, ratio=0.816
        result = compute_disagreement([0.0, 0.5, 1.0])
        assert result < 1.0

    def test_narrow_range(self):
        result = compute_disagreement([0.4, 0.6])
        # range=0.2, std=0.1, max_std=0.1, ratio=1.0
        assert result == pytest.approx(1.0)

    def test_clamped_to_one(self):
        # Should never exceed 1.0
        result = compute_disagreement([0.0, 1.0, 0.0, 1.0])
        assert result <= 1.0


# ---- compute_severity ----


class TestComputeSeverity:
    def test_empty(self):
        assert compute_severity([]) == 0.0

    def test_single(self):
        assert compute_severity([0.3]) == 0.0

    def test_binary_flip(self):
        # One above 0.5, one below
        assert compute_severity([0.3, 0.8]) == 1.0

    def test_all_above_threshold(self):
        # All above, spread < 0.25
        assert compute_severity([0.6, 0.7]) == 0.0

    def test_moderate_spread_above(self):
        # All above, spread > 0.25
        assert compute_severity([0.6, 0.9]) == 0.5

    def test_all_below_threshold(self):
        assert compute_severity([0.1, 0.2]) == 0.0

    def test_moderate_spread_below(self):
        assert compute_severity([0.1, 0.4]) == 0.5

    def test_custom_threshold(self):
        # With threshold=0.3: 0.2 < 0.3 < 0.4 -> binary flip
        assert compute_severity([0.2, 0.4], threshold=0.3) == 1.0

    def test_exactly_at_threshold_no_flip(self):
        # 0.5 is not above threshold, not below -> no flip between [0.5, 0.5]
        assert compute_severity([0.5, 0.5]) == 0.0

    def test_one_at_threshold_one_below(self):
        # 0.5 is not above, 0.3 is below -> no flip (need strictly above)
        assert compute_severity([0.5, 0.3]) == 0.0


# ---- score_items ----


class TestScoreItems:
    def test_empty(self):
        assert score_items({}, DisagreementConfig()) == []

    def test_filters_below_min_annotators(self):
        annotations = {"a": [0.5]}  # only 1 annotator
        scores = score_items(annotations, DisagreementConfig(min_annotators=2))
        assert len(scores) == 0

    def test_keeps_items_at_min_annotators(self):
        annotations = {"a": [0.3, 0.7]}
        scores = score_items(annotations, DisagreementConfig(min_annotators=2))
        assert len(scores) == 1
        assert scores[0].item_id == "a"

    def test_score_includes_severity(self):
        # Use 3 annotators so base disagreement is < 1.0
        # [0.2, 0.5, 0.8]: has binary flip (0.2 < 0.5, 0.8 > 0.5)
        annotations = {"a": [0.2, 0.5, 0.8]}
        scores_w1 = score_items(annotations, DisagreementConfig(severity_weight=1.0))
        scores_w0 = score_items(annotations, DisagreementConfig(severity_weight=0.0))
        assert scores_w1[0].score > scores_w0[0].score

    def test_sorted_by_item_id(self):
        annotations = {"c": [0.1, 0.9], "a": [0.3, 0.7], "b": [0.4, 0.6]}
        scores = score_items(annotations, DisagreementConfig())
        ids = [s.item_id for s in scores]
        assert ids == ["a", "b", "c"]

    def test_label_spread(self):
        annotations = {"a": [0.2, 0.5, 0.9]}
        scores = score_items(annotations, DisagreementConfig())
        assert scores[0].label_spread == pytest.approx(0.7)

    def test_n_annotators(self):
        annotations = {"a": [0.1, 0.2, 0.3, 0.4]}
        scores = score_items(annotations, DisagreementConfig())
        assert scores[0].n_annotators == 4

    def test_score_clamped_to_one(self):
        annotations = {"a": [0.0, 1.0]}
        scores = score_items(
            annotations, DisagreementConfig(severity_weight=5.0)
        )
        assert scores[0].score <= 1.0


# ---- sample_by_disagreement ----


class TestSampleByDisagreement:
    def test_empty(self):
        assert sample_by_disagreement([], 10) == []

    def test_sample_smaller_than_pool(self):
        scores = [
            DisagreementScore(item_id="a", score=0.9, n_annotators=2, label_spread=0.5),
            DisagreementScore(item_id="b", score=0.3, n_annotators=2, label_spread=0.2),
            DisagreementScore(item_id="c", score=0.7, n_annotators=2, label_spread=0.4),
        ]
        result = sample_by_disagreement(scores, 2, seed=42)
        assert len(result) == 2
        # Top 2 by score should be "a" (0.9) and "c" (0.7)
        assert "a" in result
        assert "c" in result

    def test_sample_larger_than_pool(self):
        scores = [
            DisagreementScore(item_id="a", score=0.5, n_annotators=2, label_spread=0.3),
        ]
        result = sample_by_disagreement(scores, 10, seed=1)
        assert result == ["a"]

    def test_deterministic_with_seed(self):
        scores = [
            DisagreementScore(item_id=f"i{i}", score=0.5, n_annotators=2, label_spread=0.3)
            for i in range(20)
        ]
        r1 = sample_by_disagreement(scores, 5, seed=42)
        r2 = sample_by_disagreement(scores, 5, seed=42)
        assert r1 == r2

    def test_different_seeds_may_differ(self):
        scores = [
            DisagreementScore(item_id=f"i{i}", score=0.5, n_annotators=2, label_spread=0.3)
            for i in range(20)
        ]
        r1 = sample_by_disagreement(scores, 5, seed=1)
        r2 = sample_by_disagreement(scores, 5, seed=2)
        # With different seeds, tiebreaking differs
        assert r1 != r2

    def test_highest_score_first(self):
        scores = [
            DisagreementScore(item_id="low", score=0.1, n_annotators=2, label_spread=0.1),
            DisagreementScore(item_id="high", score=0.9, n_annotators=2, label_spread=0.8),
        ]
        result = sample_by_disagreement(scores, 1, seed=0)
        assert result == ["high"]


# ---- run_disagreement_sampling ----


class TestRunDisagreementSampling:
    def test_empty(self):
        result = run_disagreement_sampling({}, DisagreementConfig())
        assert result.selected_ids == []
        assert result.scores == []
        assert result.mean_disagreement == 0.0
        assert result.total_pool == 0

    def test_basic(self):
        annotations = {
            "a": [0.2, 0.8],
            "b": [0.5, 0.5],
            "c": [0.1, 0.9],
        }
        config = DisagreementConfig(sample_size=2, seed=42)
        result = run_disagreement_sampling(annotations, config)
        assert len(result.selected_ids) == 2
        assert result.total_pool == 3
        assert len(result.scores) == 3

    def test_filters_low_annotator_count(self):
        annotations = {
            "a": [0.5],  # only 1 annotator
            "b": [0.2, 0.8],  # 2 annotators
        }
        config = DisagreementConfig(min_annotators=2, sample_size=10)
        result = run_disagreement_sampling(annotations, config)
        assert len(result.scores) == 1
        assert result.scores[0].item_id == "b"
        assert result.total_pool == 2

    def test_mean_disagreement(self):
        annotations = {
            "a": [0.0, 1.0],
            "b": [0.5, 0.5],
        }
        config = DisagreementConfig(sample_size=10, seed=1)
        result = run_disagreement_sampling(annotations, config)
        # "a" has high disagreement, "b" has 0 - mean should be > 0
        assert result.mean_disagreement > 0.0

    def test_selects_most_contentious(self):
        annotations = {
            "agree": [0.5, 0.5, 0.5],
            "disagree": [0.0, 1.0, 0.5],
        }
        config = DisagreementConfig(sample_size=1, seed=42)
        result = run_disagreement_sampling(annotations, config)
        assert result.selected_ids == ["disagree"]

    def test_deterministic(self):
        annotations = {f"i{i}": [0.3, 0.7] for i in range(20)}
        config = DisagreementConfig(sample_size=5, seed=42)
        r1 = run_disagreement_sampling(annotations, config)
        r2 = run_disagreement_sampling(annotations, config)
        assert r1.selected_ids == r2.selected_ids


# ---- format_disagreement_report ----


class TestFormatDisagreementReport:
    def test_empty_result(self):
        result = DisagreementResult()
        report = format_disagreement_report(result)
        assert "Disagreement Sampling Report" in report
        assert "Total pool size: 0" in report

    def test_with_data(self):
        s = DisagreementScore(item_id="a", score=0.8, n_annotators=3, label_spread=0.6)
        result = DisagreementResult(
            selected_ids=["a"],
            scores=[s],
            mean_disagreement=0.8,
            total_pool=1,
        )
        report = format_disagreement_report(result)
        assert "Selected count: 1" in report
        assert "Mean disagreement: 0.8000" in report
        assert "a:" in report

    def test_report_contains_header(self):
        result = DisagreementResult(total_pool=5)
        report = format_disagreement_report(result)
        assert "=" * 40 in report

    def test_top_disagreements_section(self):
        scores = [
            DisagreementScore(item_id=f"i{i}", score=i * 0.1, n_annotators=2, label_spread=0.3)
            for i in range(3)
        ]
        result = DisagreementResult(scores=scores, total_pool=3)
        report = format_disagreement_report(result)
        assert "Top disagreements:" in report
