"""Tests for judge ensemble evaluation."""

from __future__ import annotations

import pytest

from evalyn_sdk.evaluation.judge_ensemble import (
    EnsembleConfig,
    EnsembleReport,
    EnsembleResult,
    JudgeVote,
    build_ensemble_report,
    evaluate_ensemble,
    majority_vote,
    weighted_average,
)


# -- JudgeVote --


class TestJudgeVote:
    def test_as_dict(self):
        v = JudgeVote("judge_a", 0.9, True, 2.0)
        d = v.as_dict()
        assert d == {
            "judge_id": "judge_a",
            "score": 0.9,
            "passed": True,
            "weight": 2.0,
        }

    def test_default_weight(self):
        v = JudgeVote("j1", 0.5, False)
        assert v.weight == 1.0


# -- EnsembleResult as_dict / from_dict --


class TestEnsembleResultRoundtrip:
    def test_roundtrip(self):
        votes = [JudgeVote("a", 0.8, True), JudgeVote("b", 0.6, True)]
        result = EnsembleResult(
            item_id="item_1",
            votes=votes,
            final_score=0.7,
            final_passed=True,
            agreement_rate=1.0,
            method="majority_vote",
            flagged_for_review=False,
        )
        d = result.as_dict()
        restored = EnsembleResult.from_dict(d)
        assert restored.item_id == "item_1"
        assert len(restored.votes) == 2
        assert restored.final_score == 0.7
        assert restored.method == "majority_vote"
        assert restored.flagged_for_review is False

    def test_from_dict_defaults(self):
        d = {
            "item_id": "x",
            "votes": [],
            "final_score": 0.0,
            "final_passed": False,
            "agreement_rate": 0.0,
            "method": "majority_vote",
        }
        result = EnsembleResult.from_dict(d)
        assert result.flagged_for_review is False


# -- EnsembleConfig --


class TestEnsembleConfig:
    def test_defaults(self):
        cfg = EnsembleConfig()
        assert cfg.method == "majority_vote"
        assert cfg.disagreement_threshold == 0.3
        assert cfg.min_judges == 3

    def test_roundtrip(self):
        cfg = EnsembleConfig(method="weighted_average", disagreement_threshold=0.5, min_judges=5)
        d = cfg.as_dict()
        restored = EnsembleConfig.from_dict(d)
        assert restored.method == "weighted_average"
        assert restored.disagreement_threshold == 0.5
        assert restored.min_judges == 5

    def test_from_dict_defaults(self):
        cfg = EnsembleConfig.from_dict({})
        assert cfg.method == "majority_vote"
        assert cfg.min_judges == 3


# -- majority_vote --


class TestMajorityVote:
    def test_all_agree_pass(self):
        votes = [
            JudgeVote("a", 0.9, True),
            JudgeVote("b", 0.8, True),
            JudgeVote("c", 0.7, True),
        ]
        passed, agreement = majority_vote(votes)
        assert passed is True
        assert agreement == 1.0

    def test_all_agree_fail(self):
        votes = [
            JudgeVote("a", 0.2, False),
            JudgeVote("b", 0.3, False),
            JudgeVote("c", 0.1, False),
        ]
        passed, agreement = majority_vote(votes)
        assert passed is False
        assert agreement == 1.0

    def test_split_decision(self):
        votes = [
            JudgeVote("a", 0.9, True),
            JudgeVote("b", 0.8, True),
            JudgeVote("c", 0.2, False),
        ]
        passed, agreement = majority_vote(votes)
        assert passed is True
        assert agreement == pytest.approx(2 / 3)

    def test_majority_fail(self):
        votes = [
            JudgeVote("a", 0.3, False),
            JudgeVote("b", 0.2, False),
            JudgeVote("c", 0.9, True),
        ]
        passed, agreement = majority_vote(votes)
        assert passed is False
        assert agreement == pytest.approx(2 / 3)

    def test_empty_votes(self):
        passed, agreement = majority_vote([])
        assert passed is False
        assert agreement == 0.0


# -- weighted_average --


class TestWeightedAverage:
    def test_equal_weights(self):
        votes = [
            JudgeVote("a", 0.8, True, 1.0),
            JudgeVote("b", 0.6, True, 1.0),
            JudgeVote("c", 0.4, False, 1.0),
        ]
        result = weighted_average(votes)
        assert result == pytest.approx(0.6)

    def test_unequal_weights(self):
        votes = [
            JudgeVote("a", 1.0, True, 3.0),
            JudgeVote("b", 0.0, False, 1.0),
        ]
        result = weighted_average(votes)
        # (1.0*3 + 0.0*1) / (3+1) = 0.75
        assert result == pytest.approx(0.75)

    def test_empty(self):
        assert weighted_average([]) == 0.0

    def test_zero_weights(self):
        votes = [JudgeVote("a", 0.5, True, 0.0)]
        assert weighted_average(votes) == 0.0


# -- evaluate_ensemble --


class TestEvaluateEnsemble:
    def test_majority_method(self):
        votes = [
            JudgeVote("a", 0.9, True),
            JudgeVote("b", 0.8, True),
            JudgeVote("c", 0.3, False),
        ]
        config = EnsembleConfig(method="majority_vote")
        result = evaluate_ensemble("item_1", votes, config)
        assert result.final_passed is True
        assert result.method == "majority_vote"
        assert result.item_id == "item_1"

    def test_weighted_method(self):
        votes = [
            JudgeVote("a", 0.9, True, 2.0),
            JudgeVote("b", 0.1, False, 1.0),
        ]
        config = EnsembleConfig(method="weighted_average")
        result = evaluate_ensemble("item_2", votes, config)
        # (0.9*2 + 0.1*1) / 3 = 0.633...
        assert result.final_score == pytest.approx(0.6333, abs=0.01)
        assert result.final_passed is True
        assert result.method == "weighted_average"

    def test_cost_aware_method(self):
        votes = [
            JudgeVote("cheap", 0.7, True, 1.0),
            JudgeVote("expensive", 0.9, True, 5.0),
        ]
        config = EnsembleConfig(method="cost_aware")
        result = evaluate_ensemble("item_3", votes, config)
        assert result.method == "cost_aware"
        # Weighted toward expensive judge
        assert result.final_score > 0.7

    def test_flagged_for_review_on_disagreement(self):
        votes = [
            JudgeVote("a", 0.9, True),
            JudgeVote("b", 0.1, False),
        ]
        # disagreement_threshold=0.3 means flag when agreement < 0.7
        config = EnsembleConfig(method="majority_vote", disagreement_threshold=0.3)
        result = evaluate_ensemble("item_4", votes, config)
        # agreement = 0.5 < 0.7, should be flagged
        assert result.flagged_for_review is True

    def test_not_flagged_on_agreement(self):
        votes = [
            JudgeVote("a", 0.9, True),
            JudgeVote("b", 0.8, True),
            JudgeVote("c", 0.7, True),
        ]
        config = EnsembleConfig(method="majority_vote", disagreement_threshold=0.3)
        result = evaluate_ensemble("item_5", votes, config)
        # agreement = 1.0 >= 0.7, not flagged
        assert result.flagged_for_review is False

    def test_empty_votes(self):
        config = EnsembleConfig()
        result = evaluate_ensemble("empty", [], config)
        assert result.final_passed is False
        assert result.final_score == 0.0
        assert result.flagged_for_review is True


# -- EnsembleReport --


class TestEnsembleReport:
    def _make_result(self, item_id: str, flagged: bool = False) -> EnsembleResult:
        return EnsembleResult(
            item_id=item_id,
            votes=[JudgeVote("j1", 0.8, True)],
            final_score=0.8,
            final_passed=True,
            agreement_rate=1.0,
            method="majority_vote",
            flagged_for_review=flagged,
        )

    def test_build_report(self):
        results = [
            self._make_result("a"),
            self._make_result("b", flagged=True),
            self._make_result("c"),
        ]
        report = build_ensemble_report(results)
        assert report.total_items == 3
        assert report.flagged_count == 1
        assert report.avg_agreement == 1.0

    def test_roundtrip(self):
        results = [self._make_result("a"), self._make_result("b")]
        report = build_ensemble_report(results)
        d = report.as_dict()
        restored = EnsembleReport.from_dict(d)
        assert restored.total_items == 2
        assert restored.flagged_count == 0
        assert len(restored.results) == 2

    def test_format_text(self):
        results = [
            self._make_result("item_1"),
            self._make_result("item_2", flagged=True),
        ]
        report = build_ensemble_report(results)
        text = report.format_text()
        assert "Ensemble Evaluation Report" in text
        assert "item_1" in text
        assert "[REVIEW]" in text
        assert "Total items: 2" in text

    def test_empty_report(self):
        report = build_ensemble_report([])
        assert report.total_items == 0
        assert report.flagged_count == 0
        assert report.avg_agreement == 0.0


# -- Edge cases --


class TestEdgeCases:
    def test_single_vote_majority(self):
        votes = [JudgeVote("only", 0.9, True)]
        passed, agreement = majority_vote(votes)
        assert passed is True
        assert agreement == 1.0

    def test_single_vote_ensemble(self):
        config = EnsembleConfig()
        result = evaluate_ensemble("solo", [JudgeVote("j1", 0.8, True)], config)
        assert result.final_passed is True
        assert result.agreement_rate == 1.0

    def test_even_split_fails(self):
        # When evenly split, pass_count == fail_count, so passed is False
        votes = [
            JudgeVote("a", 0.9, True),
            JudgeVote("b", 0.1, False),
        ]
        passed, agreement = majority_vote(votes)
        assert passed is False
        assert agreement == 0.5

    def test_weighted_average_single(self):
        result = weighted_average([JudgeVote("j", 0.42, False, 1.0)])
        assert result == pytest.approx(0.42)
