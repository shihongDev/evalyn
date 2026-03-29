"""Tests for the pre-annotation pipeline module."""

from __future__ import annotations

import sys
from pathlib import Path

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.pre_annotation import (
    PreAnnotationConfig,
    PreAnnotation,
    PreAnnotationReport,
    triage_annotation,
    triage_batch,
    build_pre_annotation_prompt,
    compute_pre_annotation_accuracy,
    build_pre_annotation_report,
    format_pre_annotation_report,
)


# ---------------------------------------------------------------------------
# PreAnnotationConfig
# ---------------------------------------------------------------------------


class TestPreAnnotationConfig:
    def test_defaults(self):
        cfg = PreAnnotationConfig()
        assert cfg.confidence_threshold == 0.8
        assert cfg.auto_accept_above == 0.95
        assert cfg.human_review_below == 0.8
        assert cfg.batch_size == 10

    def test_custom_values(self):
        cfg = PreAnnotationConfig(
            confidence_threshold=0.7,
            auto_accept_above=0.9,
            human_review_below=0.6,
            batch_size=20,
        )
        assert cfg.auto_accept_above == 0.9
        assert cfg.batch_size == 20

    def test_as_dict(self):
        cfg = PreAnnotationConfig(batch_size=5)
        d = cfg.as_dict()
        assert d["batch_size"] == 5
        assert d["confidence_threshold"] == 0.8

    def test_from_dict(self):
        d = {"confidence_threshold": 0.6, "auto_accept_above": 0.85, "batch_size": 15}
        cfg = PreAnnotationConfig.from_dict(d)
        assert cfg.confidence_threshold == 0.6
        assert cfg.auto_accept_above == 0.85
        assert cfg.batch_size == 15

    def test_from_dict_defaults(self):
        cfg = PreAnnotationConfig.from_dict({})
        assert cfg.confidence_threshold == 0.8
        assert cfg.batch_size == 10

    def test_roundtrip(self):
        cfg = PreAnnotationConfig(confidence_threshold=0.5, batch_size=3)
        restored = PreAnnotationConfig.from_dict(cfg.as_dict())
        assert restored.confidence_threshold == cfg.confidence_threshold
        assert restored.batch_size == cfg.batch_size


# ---------------------------------------------------------------------------
# PreAnnotation
# ---------------------------------------------------------------------------


class TestPreAnnotation:
    def test_creation(self):
        a = PreAnnotation(item_id="i1", predicted_label=0.9, confidence=0.85)
        assert a.item_id == "i1"
        assert a.status == ""
        assert a.reasoning == ""

    def test_as_dict(self):
        a = PreAnnotation(
            item_id="i1",
            predicted_label=0.8,
            confidence=0.7,
            status="needs_review",
            reasoning="unsure",
        )
        d = a.as_dict()
        assert d["item_id"] == "i1"
        assert d["predicted_label"] == 0.8
        assert d["status"] == "needs_review"

    def test_from_dict(self):
        d = {
            "item_id": "x",
            "predicted_label": 0.6,
            "confidence": 0.5,
            "status": "rejected",
            "reasoning": "low conf",
        }
        a = PreAnnotation.from_dict(d)
        assert a.item_id == "x"
        assert a.status == "rejected"

    def test_from_dict_defaults(self):
        a = PreAnnotation.from_dict({})
        assert a.item_id == ""
        assert a.predicted_label == 0.0
        assert a.confidence == 0.0

    def test_roundtrip(self):
        a = PreAnnotation(
            item_id="t",
            predicted_label=0.7,
            confidence=0.6,
            status="auto_accepted",
            reasoning="good",
        )
        restored = PreAnnotation.from_dict(a.as_dict())
        assert restored.item_id == a.item_id
        assert restored.status == a.status
        assert restored.reasoning == a.reasoning


# ---------------------------------------------------------------------------
# PreAnnotationReport
# ---------------------------------------------------------------------------


class TestPreAnnotationReport:
    def test_defaults(self):
        r = PreAnnotationReport()
        assert r.annotations == []
        assert r.total == 0
        assert r.accuracy == 0.0

    def test_as_dict(self):
        ann = PreAnnotation(item_id="i", predicted_label=0.5, confidence=0.5)
        r = PreAnnotationReport(annotations=[ann], total=1, auto_accepted=1)
        d = r.as_dict()
        assert d["total"] == 1
        assert len(d["annotations"]) == 1

    def test_from_dict(self):
        d = {
            "annotations": [
                {"item_id": "a", "predicted_label": 0.5, "confidence": 0.5}
            ],
            "total": 1,
            "auto_accepted": 0,
            "needs_review": 1,
            "rejected": 0,
            "accuracy": 0.75,
        }
        r = PreAnnotationReport.from_dict(d)
        assert r.total == 1
        assert r.accuracy == 0.75
        assert len(r.annotations) == 1

    def test_roundtrip(self):
        ann = PreAnnotation(item_id="i", predicted_label=0.8, confidence=0.9, status="auto_accepted")
        r = PreAnnotationReport(annotations=[ann], total=1, auto_accepted=1, accuracy=0.9)
        restored = PreAnnotationReport.from_dict(r.as_dict())
        assert restored.total == r.total
        assert restored.accuracy == r.accuracy
        assert len(restored.annotations) == 1


# ---------------------------------------------------------------------------
# triage_annotation
# ---------------------------------------------------------------------------


class TestTriageAnnotation:
    def test_auto_accepted(self):
        cfg = PreAnnotationConfig(auto_accept_above=0.95, human_review_below=0.8)
        a = PreAnnotation(item_id="i", predicted_label=0.9, confidence=0.97)
        result = triage_annotation(a, cfg)
        assert result.status == "auto_accepted"

    def test_needs_review(self):
        cfg = PreAnnotationConfig(auto_accept_above=0.95, human_review_below=0.8)
        a = PreAnnotation(item_id="i", predicted_label=0.7, confidence=0.85)
        result = triage_annotation(a, cfg)
        assert result.status == "needs_review"

    def test_rejected(self):
        cfg = PreAnnotationConfig(auto_accept_above=0.95, human_review_below=0.8)
        a = PreAnnotation(item_id="i", predicted_label=0.5, confidence=0.3)
        result = triage_annotation(a, cfg)
        assert result.status == "rejected"

    def test_boundary_auto_accept(self):
        cfg = PreAnnotationConfig(auto_accept_above=0.95)
        a = PreAnnotation(item_id="i", predicted_label=0.5, confidence=0.95)
        result = triage_annotation(a, cfg)
        assert result.status == "auto_accepted"

    def test_boundary_review(self):
        cfg = PreAnnotationConfig(auto_accept_above=0.95, human_review_below=0.8)
        a = PreAnnotation(item_id="i", predicted_label=0.5, confidence=0.8)
        result = triage_annotation(a, cfg)
        assert result.status == "needs_review"

    def test_boundary_rejected(self):
        cfg = PreAnnotationConfig(human_review_below=0.8)
        a = PreAnnotation(item_id="i", predicted_label=0.5, confidence=0.79)
        result = triage_annotation(a, cfg)
        assert result.status == "rejected"

    def test_mutates_in_place(self):
        cfg = PreAnnotationConfig()
        a = PreAnnotation(item_id="i", predicted_label=0.5, confidence=0.99)
        result = triage_annotation(a, cfg)
        assert result is a

    def test_custom_thresholds(self):
        cfg = PreAnnotationConfig(auto_accept_above=0.5, human_review_below=0.3)
        a = PreAnnotation(item_id="i", predicted_label=0.5, confidence=0.4)
        result = triage_annotation(a, cfg)
        assert result.status == "needs_review"


# ---------------------------------------------------------------------------
# triage_batch
# ---------------------------------------------------------------------------


class TestTriageBatch:
    def test_empty_batch(self):
        result = triage_batch([], PreAnnotationConfig())
        assert result == []

    def test_mixed_batch(self):
        cfg = PreAnnotationConfig(auto_accept_above=0.9, human_review_below=0.5)
        annotations = [
            PreAnnotation(item_id="a", predicted_label=0.9, confidence=0.95),
            PreAnnotation(item_id="b", predicted_label=0.6, confidence=0.7),
            PreAnnotation(item_id="c", predicted_label=0.3, confidence=0.2),
        ]
        results = triage_batch(annotations, cfg)
        assert results[0].status == "auto_accepted"
        assert results[1].status == "needs_review"
        assert results[2].status == "rejected"

    def test_all_auto_accepted(self):
        cfg = PreAnnotationConfig(auto_accept_above=0.5)
        annotations = [
            PreAnnotation(item_id=f"i{i}", predicted_label=0.8, confidence=0.9)
            for i in range(5)
        ]
        results = triage_batch(annotations, cfg)
        assert all(r.status == "auto_accepted" for r in results)

    def test_preserves_order(self):
        cfg = PreAnnotationConfig()
        annotations = [
            PreAnnotation(item_id=f"item_{i}", predicted_label=0.5, confidence=0.99)
            for i in range(3)
        ]
        results = triage_batch(annotations, cfg)
        for i, r in enumerate(results):
            assert r.item_id == f"item_{i}"


# ---------------------------------------------------------------------------
# build_pre_annotation_prompt
# ---------------------------------------------------------------------------


class TestBuildPreAnnotationPrompt:
    def test_contains_metric_id(self):
        prompt = build_pre_annotation_prompt({"query": "hello"}, "accuracy")
        assert "accuracy" in prompt

    def test_contains_item_content(self):
        prompt = build_pre_annotation_prompt({"query": "test query"}, "relevance")
        assert "test query" in prompt

    def test_contains_rubric(self):
        prompt = build_pre_annotation_prompt(
            {"q": "x"}, "quality", rubric="Rate from 0 to 1"
        )
        assert "Rubric:" in prompt
        assert "Rate from 0 to 1" in prompt

    def test_no_rubric(self):
        prompt = build_pre_annotation_prompt({"q": "x"}, "quality")
        assert "Rubric:" not in prompt

    def test_response_format_instructions(self):
        prompt = build_pre_annotation_prompt({"q": "x"}, "m")
        assert "score:" in prompt
        assert "confidence:" in prompt
        assert "reasoning:" in prompt

    def test_multiple_item_fields(self):
        item = {"input": "question", "output": "answer", "context": "background"}
        prompt = build_pre_annotation_prompt(item, "faithfulness")
        assert "input:" in prompt
        assert "output:" in prompt
        assert "context:" in prompt


# ---------------------------------------------------------------------------
# compute_pre_annotation_accuracy
# ---------------------------------------------------------------------------


class TestComputePreAnnotationAccuracy:
    def test_perfect_accuracy(self):
        annotations = [
            PreAnnotation(item_id="a", predicted_label=0.9, confidence=0.8),
            PreAnnotation(item_id="b", predicted_label=0.2, confidence=0.8),
        ]
        human = {"a": 0.8, "b": 0.1}
        acc = compute_pre_annotation_accuracy(annotations, human)
        assert acc == 1.0

    def test_zero_accuracy(self):
        annotations = [
            PreAnnotation(item_id="a", predicted_label=0.9, confidence=0.8),
            PreAnnotation(item_id="b", predicted_label=0.2, confidence=0.8),
        ]
        human = {"a": 0.1, "b": 0.9}
        acc = compute_pre_annotation_accuracy(annotations, human)
        assert acc == 0.0

    def test_partial_accuracy(self):
        annotations = [
            PreAnnotation(item_id="a", predicted_label=0.9, confidence=0.8),
            PreAnnotation(item_id="b", predicted_label=0.9, confidence=0.8),
        ]
        human = {"a": 0.8, "b": 0.2}
        acc = compute_pre_annotation_accuracy(annotations, human)
        assert acc == 0.5

    def test_empty_annotations(self):
        acc = compute_pre_annotation_accuracy([], {"a": 0.5})
        assert acc == 0.0

    def test_empty_human_labels(self):
        annotations = [PreAnnotation(item_id="a", predicted_label=0.5, confidence=0.5)]
        acc = compute_pre_annotation_accuracy(annotations, {})
        assert acc == 0.0

    def test_no_matching_ids(self):
        annotations = [PreAnnotation(item_id="a", predicted_label=0.5, confidence=0.5)]
        acc = compute_pre_annotation_accuracy(annotations, {"b": 0.5})
        assert acc == 0.0

    def test_custom_threshold(self):
        annotations = [
            PreAnnotation(item_id="a", predicted_label=0.6, confidence=0.8),
        ]
        human = {"a": 0.8}
        # threshold=0.7: predicted 0.6 < 0.7 (fail), human 0.8 >= 0.7 (pass) - disagree
        acc = compute_pre_annotation_accuracy(annotations, human, threshold=0.7)
        assert acc == 0.0

    def test_threshold_boundary(self):
        annotations = [
            PreAnnotation(item_id="a", predicted_label=0.5, confidence=0.8),
        ]
        human = {"a": 0.5}
        # Both at exactly threshold - both pass
        acc = compute_pre_annotation_accuracy(annotations, human, threshold=0.5)
        assert acc == 1.0


# ---------------------------------------------------------------------------
# build_pre_annotation_report
# ---------------------------------------------------------------------------


class TestBuildPreAnnotationReport:
    def test_empty_annotations(self):
        r = build_pre_annotation_report([])
        assert r.total == 0
        assert r.auto_accepted == 0

    def test_counts_statuses(self):
        annotations = [
            PreAnnotation(item_id="a", predicted_label=0.9, confidence=0.9, status="auto_accepted"),
            PreAnnotation(item_id="b", predicted_label=0.7, confidence=0.7, status="needs_review"),
            PreAnnotation(item_id="c", predicted_label=0.3, confidence=0.3, status="rejected"),
            PreAnnotation(item_id="d", predicted_label=0.8, confidence=0.8, status="needs_review"),
        ]
        r = build_pre_annotation_report(annotations)
        assert r.total == 4
        assert r.auto_accepted == 1
        assert r.needs_review == 2
        assert r.rejected == 1

    def test_accuracy_with_human_labels(self):
        annotations = [
            PreAnnotation(item_id="a", predicted_label=0.9, confidence=0.9, status="auto_accepted"),
            PreAnnotation(item_id="b", predicted_label=0.1, confidence=0.9, status="auto_accepted"),
        ]
        human = {"a": 0.8, "b": 0.2}
        r = build_pre_annotation_report(annotations, human_labels=human)
        assert r.accuracy == 1.0

    def test_accuracy_without_human_labels(self):
        annotations = [
            PreAnnotation(item_id="a", predicted_label=0.9, confidence=0.9, status="auto_accepted"),
        ]
        r = build_pre_annotation_report(annotations)
        assert r.accuracy == 0.0

    def test_report_contains_annotations(self):
        annotations = [
            PreAnnotation(item_id="x", predicted_label=0.5, confidence=0.5, status="needs_review"),
        ]
        r = build_pre_annotation_report(annotations)
        assert len(r.annotations) == 1
        assert r.annotations[0].item_id == "x"


# ---------------------------------------------------------------------------
# format_pre_annotation_report
# ---------------------------------------------------------------------------


class TestFormatPreAnnotationReport:
    def test_contains_title(self):
        r = PreAnnotationReport(total=0)
        text = format_pre_annotation_report(r)
        assert "Pre-Annotation Report" in text

    def test_contains_counts(self):
        r = PreAnnotationReport(total=10, auto_accepted=5, needs_review=3, rejected=2)
        text = format_pre_annotation_report(r)
        assert "10" in text
        assert "5" in text
        assert "3" in text
        assert "2" in text

    def test_accuracy_shown_when_positive(self):
        r = PreAnnotationReport(total=2, accuracy=0.85)
        text = format_pre_annotation_report(r)
        assert "85.0%" in text

    def test_accuracy_hidden_when_zero(self):
        r = PreAnnotationReport(total=2, accuracy=0.0)
        text = format_pre_annotation_report(r)
        assert "Accuracy" not in text

    def test_shows_annotation_details(self):
        ann = PreAnnotation(
            item_id="item_1",
            predicted_label=0.75,
            confidence=0.6,
            status="needs_review",
            reasoning="uncertain about tone",
        )
        r = PreAnnotationReport(annotations=[ann], total=1, needs_review=1)
        text = format_pre_annotation_report(r)
        assert "item_1" in text
        assert "needs_review" in text
        assert "uncertain about tone" in text

    def test_empty_report_format(self):
        r = PreAnnotationReport()
        text = format_pre_annotation_report(r)
        assert "Total annotations: 0" in text


# ---------------------------------------------------------------------------
# Integration / end-to-end
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_full_pipeline(self):
        """End-to-end: create annotations, triage, build report, format."""
        cfg = PreAnnotationConfig(auto_accept_above=0.9, human_review_below=0.5)

        annotations = [
            PreAnnotation(item_id="a", predicted_label=0.95, confidence=0.98, reasoning="clear pass"),
            PreAnnotation(item_id="b", predicted_label=0.7, confidence=0.75, reasoning="borderline"),
            PreAnnotation(item_id="c", predicted_label=0.2, confidence=0.3, reasoning="likely fail"),
        ]

        triaged = triage_batch(annotations, cfg)
        assert triaged[0].status == "auto_accepted"
        assert triaged[1].status == "needs_review"
        assert triaged[2].status == "rejected"

        human = {"a": 1.0, "b": 0.6, "c": 0.1}
        report = build_pre_annotation_report(triaged, human_labels=human)

        assert report.total == 3
        assert report.auto_accepted == 1
        assert report.accuracy > 0

        text = format_pre_annotation_report(report)
        assert "Pre-Annotation Report" in text
        assert "3" in text

    def test_roundtrip_report(self):
        """Report serialization roundtrip."""
        ann = PreAnnotation(item_id="z", predicted_label=0.5, confidence=0.5, status="needs_review")
        r = PreAnnotationReport(annotations=[ann], total=1, needs_review=1, accuracy=0.5)
        restored = PreAnnotationReport.from_dict(r.as_dict())
        assert restored.total == 1
        assert restored.accuracy == 0.5
        assert restored.annotations[0].item_id == "z"
