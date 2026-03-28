"""Tests for evaluation canary and inter-rater reliability."""

import pytest
from evalyn_sdk.models import DatasetItem, MetricResult
from evalyn_sdk.evaluation.canary import (
    select_canary_sample, evaluate_canary, CanaryResult,
)
from evalyn_sdk.analysis.inter_rater import (
    compute_agreement, AgreementStats, _cohens_kappa, _fleiss_kappa,
)


def _item(item_id):
    return DatasetItem(id=item_id, input=f"input-{item_id}", output=f"output-{item_id}")


def _result(item_id, passed=True):
    return MetricResult(
        metric_id="m", item_id=item_id, call_id="c",
        score=1.0 if passed else 0.0, passed=passed, details={},
    )


# =============================================================================
# Canary tests
# =============================================================================

class TestSelectCanarySample:
    def test_small_dataset_returns_all(self):
        items = [_item(str(i)) for i in range(5)]
        sample = select_canary_sample(items, sample_size=10)
        assert len(sample) == 5

    def test_sample_correct_size(self):
        items = [_item(str(i)) for i in range(100)]
        sample = select_canary_sample(items, sample_size=10)
        assert len(sample) == 10

    def test_deterministic(self):
        items = [_item(str(i)) for i in range(50)]
        s1 = select_canary_sample(items, seed=42)
        s2 = select_canary_sample(items, seed=42)
        assert [i.id for i in s1] == [i.id for i in s2]

    def test_different_seeds(self):
        items = [_item(str(i)) for i in range(50)]
        s1 = select_canary_sample(items, seed=1)
        s2 = select_canary_sample(items, seed=999)
        assert [i.id for i in s1] != [i.id for i in s2]

    def test_empty_dataset(self):
        assert select_canary_sample([], sample_size=10) == []


class TestEvaluateCanary:
    def test_proceed_when_passing(self):
        results = [_result(f"i{i}", passed=True) for i in range(10)]
        canary = evaluate_canary(results, abort_threshold=0.2)
        assert not canary.should_abort
        assert canary.status == "PROCEED"
        assert canary.pass_rate == 1.0

    def test_abort_when_failing(self):
        results = [_result(f"i{i}", passed=False) for i in range(10)]
        canary = evaluate_canary(results, abort_threshold=0.2)
        assert canary.should_abort
        assert canary.status == "ABORT"
        assert canary.pass_rate == 0.0

    def test_borderline(self):
        # 2/10 pass = 20%, threshold is 20%
        results = ([_result(f"p{i}", True) for i in range(2)]
                  + [_result(f"f{i}", False) for i in range(8)])
        canary = evaluate_canary(results, abort_threshold=0.2)
        assert not canary.should_abort  # 20% >= 20%

    def test_just_below_threshold(self):
        # 1/10 pass = 10%, threshold is 20%
        results = ([_result("p0", True)]
                  + [_result(f"f{i}", False) for i in range(9)])
        canary = evaluate_canary(results, abort_threshold=0.2)
        assert canary.should_abort

    def test_cost_savings_estimate(self):
        results = [_result(f"i{i}", passed=False) for i in range(10)]
        canary = evaluate_canary(
            results, abort_threshold=0.5,
            total_items=100, estimated_cost_per_item=0.01,
        )
        assert canary.should_abort
        assert canary.estimated_cost_saved > 0

    def test_empty_results_aborts(self):
        canary = evaluate_canary([], abort_threshold=0.2)
        assert canary.should_abort
        assert canary.sample_size == 0

    def test_multiple_metrics_per_item(self):
        results = [
            _result("i1", True), _result("i1", True),
            _result("i2", True), _result("i2", False),
        ]
        canary = evaluate_canary(results, abort_threshold=0.3)
        # i1: all pass, i2: one fails -> 1/2 items pass = 50%
        assert canary.pass_rate == 0.5
        assert canary.sample_size == 2

    def test_as_dict(self):
        results = [_result("i1", True)]
        d = evaluate_canary(results).as_dict()
        assert "status" in d
        assert "pass_rate" in d

    def test_format_text(self):
        results = [_result("i1", True)]
        text = evaluate_canary(results).format_text()
        assert "Canary Check" in text


# =============================================================================
# Inter-rater reliability tests
# =============================================================================

class TestCohensKappa:
    def test_perfect_agreement(self):
        a = [True, True, False, False, True]
        b = [True, True, False, False, True]
        assert _cohens_kappa(a, b) == pytest.approx(1.0)

    def test_complete_disagreement(self):
        # When one rater is constant True and other constant False,
        # pe=0 and po=0, so kappa=0 (not negative)
        a = [True, True, True, True, True]
        b = [False, False, False, False, False]
        kappa = _cohens_kappa(a, b)
        assert kappa <= 0

    def test_random_agreement(self):
        a = [True, False, True, False, True]
        b = [True, True, False, False, True]
        kappa = _cohens_kappa(a, b)
        # Should be between -1 and 1, moderate agreement
        assert -1 <= kappa <= 1

    def test_empty(self):
        assert _cohens_kappa([], []) == 0.0


class TestFleissKappa:
    def test_perfect_agreement(self):
        ratings = [
            [True, True, True],
            [False, False, False],
            [True, True, True],
        ]
        assert _fleiss_kappa(ratings) == pytest.approx(1.0)

    def test_empty(self):
        assert _fleiss_kappa([]) == 0.0


class TestComputeAgreement:
    def test_two_raters_perfect(self):
        ratings = {
            "judge_a": {"i1": True, "i2": False, "i3": True},
            "judge_b": {"i1": True, "i2": False, "i3": True},
        }
        stats = compute_agreement(ratings)
        assert stats.raw_agreement == 1.0
        assert stats.cohens_kappa == pytest.approx(1.0)
        assert stats.agreement_level == "almost perfect"
        assert len(stats.items_with_disagreement) == 0

    def test_two_raters_partial(self):
        ratings = {
            "judge_a": {"i1": True, "i2": True, "i3": False, "i4": True},
            "judge_b": {"i1": True, "i2": False, "i3": False, "i4": True},
        }
        stats = compute_agreement(ratings)
        assert stats.raw_agreement == 0.75
        assert len(stats.items_with_disagreement) == 1
        assert "i2" in stats.items_with_disagreement

    def test_three_raters(self):
        ratings = {
            "a": {"i1": True, "i2": True, "i3": False},
            "b": {"i1": True, "i2": True, "i3": False},
            "c": {"i1": True, "i2": False, "i3": False},
        }
        stats = compute_agreement(ratings)
        assert stats.num_raters == 3
        assert stats.fleiss_kappa is not None
        assert stats.cohens_kappa is None  # only for 2 raters

    def test_single_rater(self):
        stats = compute_agreement({"a": {"i1": True}})
        assert stats.num_raters == 1
        assert stats.raw_agreement == 0.0

    def test_no_common_items(self):
        ratings = {
            "a": {"i1": True},
            "b": {"i2": True},
        }
        stats = compute_agreement(ratings)
        assert stats.num_items == 0

    def test_as_dict(self):
        ratings = {
            "a": {"i1": True, "i2": False},
            "b": {"i1": True, "i2": False},
        }
        d = compute_agreement(ratings).as_dict()
        assert "raw_agreement" in d
        assert "cohens_kappa" in d

    def test_format_text(self):
        ratings = {
            "a": {"i1": True, "i2": False},
            "b": {"i1": True, "i2": False},
        }
        text = compute_agreement(ratings).format_text()
        assert "Inter-Rater" in text
        assert "kappa" in text.lower()

    def test_agreement_levels(self):
        stats = AgreementStats(num_raters=2, num_items=10, raw_agreement=0.9, cohens_kappa=0.85)
        assert stats.agreement_level == "almost perfect"
        stats.cohens_kappa = 0.65
        assert stats.agreement_level == "substantial"
        stats.cohens_kappa = 0.45
        assert stats.agreement_level == "moderate"
        stats.cohens_kappa = 0.25
        assert stats.agreement_level == "fair"
        stats.cohens_kappa = 0.05
        assert stats.agreement_level == "slight"
        stats.cohens_kappa = -0.1
        assert stats.agreement_level == "poor"
