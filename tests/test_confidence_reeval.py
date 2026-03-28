"""Tests for confidence-based re-evaluation."""

import pytest
from evalyn_sdk.evaluation.confidence_reeval import (
    UncertainItem,
    ReEvalPlan,
    ReEvalResult,
    ReEvalReport,
    identify_uncertain_items,
    create_reeval_plan,
    merge_reeval_results,
    build_reeval_report,
)


# =============================================================================
# UncertainItem
# =============================================================================


class TestUncertainItem:
    def test_as_dict(self):
        item = UncertainItem("i1", "m1", 0.6, 0.3, "gpt-4")
        d = item.as_dict()
        assert d["item_id"] == "i1"
        assert d["metric_id"] == "m1"
        assert d["original_score"] == 0.6
        assert d["confidence"] == 0.3
        assert d["original_model"] == "gpt-4"

    def test_default_model(self):
        item = UncertainItem("i1", "m1", 0.5, 0.4)
        assert item.original_model == ""


# =============================================================================
# ReEvalPlan
# =============================================================================


class TestReEvalPlan:
    def test_as_dict(self):
        plan = ReEvalPlan(
            uncertain_items=[UncertainItem("i1", "m1", 0.6, 0.3)],
            target_model="gemini-2.5-pro",
            estimated_calls=1,
            confidence_threshold=0.5,
        )
        d = plan.as_dict()
        assert d["target_model"] == "gemini-2.5-pro"
        assert d["estimated_calls"] == 1
        assert len(d["uncertain_items"]) == 1

    def test_from_dict(self):
        d = {
            "uncertain_items": [
                {"item_id": "i1", "metric_id": "m1", "original_score": 0.6, "confidence": 0.3}
            ],
            "target_model": "gemini-2.5-pro",
            "estimated_calls": 1,
            "confidence_threshold": 0.5,
        }
        plan = ReEvalPlan.from_dict(d)
        assert plan.target_model == "gemini-2.5-pro"
        assert len(plan.uncertain_items) == 1
        assert plan.uncertain_items[0].item_id == "i1"

    def test_from_dict_roundtrip(self):
        plan = ReEvalPlan(
            uncertain_items=[UncertainItem("i1", "m1", 0.6, 0.3, "gpt-4")],
            target_model="gemini-2.5-pro",
            estimated_calls=1,
            confidence_threshold=0.4,
        )
        restored = ReEvalPlan.from_dict(plan.as_dict())
        assert restored.target_model == plan.target_model
        assert restored.estimated_calls == plan.estimated_calls
        assert restored.confidence_threshold == plan.confidence_threshold
        assert len(restored.uncertain_items) == 1

    def test_format_text(self):
        plan = ReEvalPlan(
            uncertain_items=[UncertainItem("i1", "m1", 0.6, 0.3)],
            target_model="gemini-2.5-pro",
            estimated_calls=1,
            confidence_threshold=0.5,
        )
        text = plan.format_text()
        assert "Re-Evaluation Plan" in text
        assert "gemini-2.5-pro" in text
        assert "1" in text

    def test_format_text_empty(self):
        plan = ReEvalPlan()
        text = plan.format_text()
        assert "Uncertain items:       0" in text


# =============================================================================
# ReEvalResult
# =============================================================================


class TestReEvalResult:
    def test_as_dict(self):
        r = ReEvalResult("i1", "m1", 0.5, 0.9, "gemini-2.5-pro", True)
        d = r.as_dict()
        assert d["item_id"] == "i1"
        assert d["new_score"] == 0.9
        assert d["improved"] is True


# =============================================================================
# ReEvalReport
# =============================================================================


class TestReEvalReport:
    def test_as_dict(self):
        r = ReEvalResult("i1", "m1", 0.5, 0.9, "gemini-2.5-pro", True)
        report = ReEvalReport(results=[r], total_re_evaluated=1, improved_count=1)
        d = report.as_dict()
        assert d["total_re_evaluated"] == 1
        assert len(d["results"]) == 1

    def test_from_dict(self):
        d = {
            "results": [
                {"item_id": "i1", "metric_id": "m1", "original_score": 0.5,
                 "new_score": 0.9, "new_model": "gemini-2.5-pro", "improved": True}
            ],
            "total_re_evaluated": 1,
            "improved_count": 1,
            "unchanged_count": 0,
            "degraded_count": 0,
        }
        report = ReEvalReport.from_dict(d)
        assert report.total_re_evaluated == 1
        assert report.results[0].new_score == 0.9

    def test_from_dict_roundtrip(self):
        r = ReEvalResult("i1", "m1", 0.5, 0.9, "gemini-2.5-pro", True)
        report = ReEvalReport(results=[r], total_re_evaluated=1, improved_count=1)
        restored = ReEvalReport.from_dict(report.as_dict())
        assert restored.total_re_evaluated == report.total_re_evaluated
        assert restored.improved_count == report.improved_count
        assert len(restored.results) == 1

    def test_format_text(self):
        report = ReEvalReport(
            total_re_evaluated=3, improved_count=2, unchanged_count=1, degraded_count=0
        )
        text = report.format_text()
        assert "Re-Evaluation Report" in text
        assert "Improved:            2" in text

    def test_format_text_empty(self):
        report = ReEvalReport()
        text = report.format_text()
        assert "Total re-evaluated:  0" in text


# =============================================================================
# identify_uncertain_items
# =============================================================================


class TestIdentifyUncertainItems:
    def test_finds_low_confidence(self):
        scores = [
            {"item_id": "i1", "metric_id": "m1", "score": 0.8, "confidence": 0.3},
            {"item_id": "i2", "metric_id": "m1", "score": 0.9, "confidence": 0.7},
            {"item_id": "i3", "metric_id": "m1", "score": 0.4, "confidence": 0.2},
        ]
        result = identify_uncertain_items(scores, confidence_threshold=0.5)
        assert len(result) == 2
        ids = {r.item_id for r in result}
        assert ids == {"i1", "i3"}

    def test_none_uncertain(self):
        scores = [
            {"item_id": "i1", "metric_id": "m1", "score": 0.8, "confidence": 0.9},
            {"item_id": "i2", "metric_id": "m1", "score": 0.7, "confidence": 0.8},
        ]
        result = identify_uncertain_items(scores, confidence_threshold=0.5)
        assert result == []

    def test_all_uncertain(self):
        scores = [
            {"item_id": "i1", "metric_id": "m1", "score": 0.5, "confidence": 0.1},
            {"item_id": "i2", "metric_id": "m1", "score": 0.6, "confidence": 0.2},
        ]
        result = identify_uncertain_items(scores, confidence_threshold=0.5)
        assert len(result) == 2

    def test_empty_scores(self):
        result = identify_uncertain_items([], confidence_threshold=0.5)
        assert result == []

    def test_preserves_model(self):
        scores = [
            {"item_id": "i1", "metric_id": "m1", "score": 0.5, "confidence": 0.2, "model": "gpt-4"},
        ]
        result = identify_uncertain_items(scores, confidence_threshold=0.5)
        assert result[0].original_model == "gpt-4"

    def test_missing_model_defaults_empty(self):
        scores = [
            {"item_id": "i1", "metric_id": "m1", "score": 0.5, "confidence": 0.2},
        ]
        result = identify_uncertain_items(scores, confidence_threshold=0.5)
        assert result[0].original_model == ""

    def test_custom_threshold(self):
        scores = [
            {"item_id": "i1", "metric_id": "m1", "score": 0.5, "confidence": 0.6},
        ]
        # 0.6 < 0.7 threshold
        result = identify_uncertain_items(scores, confidence_threshold=0.7)
        assert len(result) == 1


# =============================================================================
# create_reeval_plan
# =============================================================================


class TestCreateReevalPlan:
    def test_correct_counts(self):
        items = [
            UncertainItem("i1", "m1", 0.5, 0.3),
            UncertainItem("i2", "m1", 0.6, 0.2),
        ]
        plan = create_reeval_plan(items, target_model="gemini-2.5-pro")
        assert plan.estimated_calls == 2
        assert plan.target_model == "gemini-2.5-pro"
        assert len(plan.uncertain_items) == 2

    def test_default_target_model(self):
        items = [UncertainItem("i1", "m1", 0.5, 0.3)]
        plan = create_reeval_plan(items)
        assert plan.target_model == "gemini-2.5-pro"


# =============================================================================
# merge_reeval_results
# =============================================================================


class TestMergeReevalResults:
    def test_replaces_scores(self):
        originals = [
            {"item_id": "i1", "metric_id": "m1", "score": 0.5, "confidence": 0.3},
        ]
        reeval = [ReEvalResult("i1", "m1", 0.5, 0.9, "gemini-2.5-pro", True)]
        merged = merge_reeval_results(originals, reeval)
        assert merged[0]["score"] == 0.9
        assert merged[0]["model"] == "gemini-2.5-pro"

    def test_preserves_non_re_evaluated(self):
        originals = [
            {"item_id": "i1", "metric_id": "m1", "score": 0.5, "confidence": 0.3},
            {"item_id": "i2", "metric_id": "m1", "score": 0.9, "confidence": 0.95},
        ]
        reeval = [ReEvalResult("i1", "m1", 0.5, 0.8, "gemini-2.5-pro", True)]
        merged = merge_reeval_results(originals, reeval)
        assert merged[0]["score"] == 0.8  # replaced
        assert merged[1]["score"] == 0.9  # preserved

    def test_empty_reeval(self):
        originals = [
            {"item_id": "i1", "metric_id": "m1", "score": 0.5, "confidence": 0.3},
        ]
        merged = merge_reeval_results(originals, [])
        assert merged[0]["score"] == 0.5

    def test_does_not_mutate_originals(self):
        originals = [
            {"item_id": "i1", "metric_id": "m1", "score": 0.5, "confidence": 0.3},
        ]
        reeval = [ReEvalResult("i1", "m1", 0.5, 0.9, "gemini-2.5-pro", True)]
        merge_reeval_results(originals, reeval)
        assert originals[0]["score"] == 0.5  # original untouched


# =============================================================================
# build_reeval_report
# =============================================================================


class TestBuildReevalReport:
    def test_improved(self):
        results = [ReEvalResult("i1", "m1", 0.5, 0.9, "gemini-2.5-pro", True)]
        report = build_reeval_report(results, threshold=0.01)
        assert report.improved_count == 1
        assert report.degraded_count == 0
        assert report.unchanged_count == 0

    def test_degraded(self):
        results = [ReEvalResult("i1", "m1", 0.9, 0.3, "gemini-2.5-pro", False)]
        report = build_reeval_report(results, threshold=0.01)
        assert report.degraded_count == 1
        assert report.improved_count == 0

    def test_unchanged(self):
        results = [ReEvalResult("i1", "m1", 0.5, 0.505, "gemini-2.5-pro", False)]
        report = build_reeval_report(results, threshold=0.01)
        assert report.unchanged_count == 1

    def test_mixed(self):
        results = [
            ReEvalResult("i1", "m1", 0.5, 0.9, "gemini-2.5-pro", True),
            ReEvalResult("i2", "m1", 0.9, 0.3, "gemini-2.5-pro", False),
            ReEvalResult("i3", "m1", 0.5, 0.505, "gemini-2.5-pro", False),
        ]
        report = build_reeval_report(results, threshold=0.01)
        assert report.total_re_evaluated == 3
        assert report.improved_count == 1
        assert report.degraded_count == 1
        assert report.unchanged_count == 1

    def test_empty_results(self):
        report = build_reeval_report([], threshold=0.01)
        assert report.total_re_evaluated == 0
        assert report.improved_count == 0
