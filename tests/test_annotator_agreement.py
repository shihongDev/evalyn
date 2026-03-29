"""Tests for the inter-annotator agreement module."""

from __future__ import annotations

import pytest

from evalyn_sdk.annotator_agreement import (
    AgreementReport,
    AgreementScore,
    AnnotatorLabel,
    build_agreement_report,
    build_pairwise_matrix,
    compute_cohens_kappa,
    compute_krippendorff_alpha,
    compute_percent_agreement,
    find_disagreement_items,
    format_agreement_report,
)


# ---------------------------------------------------------------------------
# AnnotatorLabel dataclass
# ---------------------------------------------------------------------------


class TestAnnotatorLabel:
    def test_create_minimal(self):
        lbl = AnnotatorLabel(item_id="i1", annotator_id="a1", metric_id="m1", label=1.0)
        assert lbl.confidence == 1.0

    def test_create_full(self):
        lbl = AnnotatorLabel("i1", "a1", "m1", 0.8, confidence=0.9)
        assert lbl.confidence == 0.9

    def test_as_dict(self):
        lbl = AnnotatorLabel("i1", "a1", "m1", 0.5, 0.7)
        d = lbl.as_dict()
        assert d["item_id"] == "i1"
        assert d["confidence"] == 0.7

    def test_from_dict_minimal(self):
        lbl = AnnotatorLabel.from_dict(
            {"item_id": "i1", "annotator_id": "a1", "metric_id": "m1", "label": 1.0}
        )
        assert lbl.confidence == 1.0

    def test_from_dict_full(self):
        lbl = AnnotatorLabel.from_dict(
            {
                "item_id": "x",
                "annotator_id": "y",
                "metric_id": "z",
                "label": 0.3,
                "confidence": 0.6,
            }
        )
        assert lbl.label == 0.3
        assert lbl.confidence == 0.6

    def test_roundtrip(self):
        original = AnnotatorLabel("i1", "a1", "m1", 0.75, 0.9)
        restored = AnnotatorLabel.from_dict(original.as_dict())
        assert restored.item_id == original.item_id
        assert restored.label == original.label
        assert restored.confidence == original.confidence


# ---------------------------------------------------------------------------
# AgreementScore dataclass
# ---------------------------------------------------------------------------


class TestAgreementScore:
    def test_create(self):
        s = AgreementScore("m1", ("a1", "a2"), 0.8, 0.9, 10)
        assert s.annotator_pair == ("a1", "a2")

    def test_as_dict_pair_is_list(self):
        s = AgreementScore("m1", ("a1", "a2"), 0.8, 0.9, 10)
        d = s.as_dict()
        assert d["annotator_pair"] == ["a1", "a2"]

    def test_from_dict(self):
        d = {
            "metric_id": "m1",
            "annotator_pair": ["a1", "a2"],
            "cohens_kappa": 0.5,
            "percent_agreement": 0.75,
            "n_items": 20,
        }
        s = AgreementScore.from_dict(d)
        assert s.annotator_pair == ("a1", "a2")
        assert s.n_items == 20

    def test_roundtrip(self):
        original = AgreementScore("m1", ("a", "b"), 0.6, 0.8, 15)
        restored = AgreementScore.from_dict(original.as_dict())
        assert restored.metric_id == original.metric_id
        assert restored.annotator_pair == original.annotator_pair


# ---------------------------------------------------------------------------
# AgreementReport dataclass
# ---------------------------------------------------------------------------


class TestAgreementReport:
    def _sample_report(self):
        return AgreementReport(
            scores=[AgreementScore("m1", ("a1", "a2"), 0.7, 0.85, 10)],
            overall_kappa=0.7,
            overall_agreement=0.85,
            high_disagreement_items=["item3", "item7"],
            total_items=10,
            total_annotators=2,
        )

    def test_as_dict(self):
        r = self._sample_report()
        d = r.as_dict()
        assert len(d["scores"]) == 1
        assert d["overall_kappa"] == 0.7
        assert d["high_disagreement_items"] == ["item3", "item7"]

    def test_from_dict(self):
        r = self._sample_report()
        restored = AgreementReport.from_dict(r.as_dict())
        assert restored.total_items == 10
        assert len(restored.scores) == 1
        assert restored.scores[0].annotator_pair == ("a1", "a2")

    def test_roundtrip(self):
        original = self._sample_report()
        restored = AgreementReport.from_dict(original.as_dict())
        assert restored.overall_agreement == original.overall_agreement
        assert restored.high_disagreement_items == original.high_disagreement_items


# ---------------------------------------------------------------------------
# compute_cohens_kappa
# ---------------------------------------------------------------------------


class TestCohensKappa:
    def test_perfect_agreement(self):
        a = [1.0, 0.0, 1.0, 0.0, 1.0]
        b = [1.0, 0.0, 1.0, 0.0, 1.0]
        kappa = compute_cohens_kappa(a, b)
        assert kappa == pytest.approx(1.0)

    def test_no_agreement(self):
        # Systematic disagreement: each pair disagrees
        a = [1.0, 0.0, 1.0, 0.0]
        b = [0.0, 1.0, 0.0, 1.0]
        kappa = compute_cohens_kappa(a, b)
        assert kappa < 0

    def test_partial_agreement(self):
        a = [1.0, 0.0, 1.0, 0.0]
        b = [1.0, 0.0, 0.0, 0.0]
        kappa = compute_cohens_kappa(a, b)
        assert 0 < kappa < 1

    def test_empty_lists(self):
        assert compute_cohens_kappa([], []) == 0.0

    def test_mismatched_lengths(self):
        assert compute_cohens_kappa([1.0], [1.0, 0.0]) == 0.0

    def test_custom_threshold(self):
        a = [0.6, 0.4, 0.8, 0.2]
        b = [0.7, 0.3, 0.9, 0.1]
        kappa = compute_cohens_kappa(a, b, threshold=0.5)
        assert kappa == pytest.approx(1.0)

    def test_all_same_class(self):
        # Both assign all positive: kappa = 0 (can't do better than chance)
        a = [1.0, 1.0, 1.0]
        b = [1.0, 1.0, 1.0]
        kappa = compute_cohens_kappa(a, b)
        # p_e = 1.0, p_o = 1.0, degenerate case
        assert kappa == pytest.approx(1.0)

    def test_kappa_range(self):
        a = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        b = [1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]
        kappa = compute_cohens_kappa(a, b)
        assert -1 <= kappa <= 1

    def test_threshold_effect(self):
        a = [0.6, 0.4, 0.3, 0.7]
        b = [0.6, 0.4, 0.3, 0.7]
        # With threshold 0.5: binarized same -> perfect agreement
        k1 = compute_cohens_kappa(a, b, threshold=0.5)
        assert k1 == pytest.approx(1.0)
        # With threshold 0.35: different binarization, still same
        k2 = compute_cohens_kappa(a, b, threshold=0.35)
        assert k2 == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# compute_percent_agreement
# ---------------------------------------------------------------------------


class TestPercentAgreement:
    def test_perfect(self):
        a = [1.0, 0.0, 1.0]
        b = [1.0, 0.0, 1.0]
        assert compute_percent_agreement(a, b) == pytest.approx(1.0)

    def test_none(self):
        a = [1.0, 1.0, 1.0]
        b = [0.0, 0.0, 0.0]
        assert compute_percent_agreement(a, b) == pytest.approx(0.0)

    def test_half(self):
        a = [1.0, 0.0, 1.0, 0.0]
        b = [0.0, 1.0, 1.0, 0.0]
        assert compute_percent_agreement(a, b) == pytest.approx(0.5)

    def test_empty(self):
        assert compute_percent_agreement([], []) == 0.0

    def test_mismatched(self):
        assert compute_percent_agreement([1.0], [0.0, 1.0]) == 0.0

    def test_threshold_matters(self):
        a = [0.6, 0.4]
        b = [0.6, 0.4]
        # threshold 0.5: both agree
        assert compute_percent_agreement(a, b, threshold=0.5) == pytest.approx(1.0)
        # threshold 0.7: a[0]=0, b[0]=0, a[1]=0, b[1]=0 -> agree
        assert compute_percent_agreement(a, b, threshold=0.7) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# compute_krippendorff_alpha
# ---------------------------------------------------------------------------


class TestKrippendorffAlpha:
    def test_perfect_agreement(self):
        labels = [
            AnnotatorLabel("i1", "a1", "m1", 1.0),
            AnnotatorLabel("i1", "a2", "m1", 1.0),
            AnnotatorLabel("i2", "a1", "m1", 0.0),
            AnnotatorLabel("i2", "a2", "m1", 0.0),
        ]
        alpha = compute_krippendorff_alpha(labels)
        assert alpha == pytest.approx(1.0)

    def test_no_agreement(self):
        labels = [
            AnnotatorLabel("i1", "a1", "m1", 1.0),
            AnnotatorLabel("i1", "a2", "m1", 0.0),
            AnnotatorLabel("i2", "a1", "m1", 0.0),
            AnnotatorLabel("i2", "a2", "m1", 1.0),
        ]
        alpha = compute_krippendorff_alpha(labels)
        assert alpha <= 0

    def test_empty(self):
        assert compute_krippendorff_alpha([]) == 0.0

    def test_single_item(self):
        labels = [
            AnnotatorLabel("i1", "a1", "m1", 1.0),
            AnnotatorLabel("i1", "a2", "m1", 1.0),
        ]
        # Only one unit: all values identical -> 1.0
        alpha = compute_krippendorff_alpha(labels)
        assert alpha == pytest.approx(1.0)

    def test_three_annotators(self):
        labels = [
            AnnotatorLabel("i1", "a1", "m1", 1.0),
            AnnotatorLabel("i1", "a2", "m1", 1.0),
            AnnotatorLabel("i1", "a3", "m1", 1.0),
            AnnotatorLabel("i2", "a1", "m1", 0.0),
            AnnotatorLabel("i2", "a2", "m1", 0.0),
            AnnotatorLabel("i2", "a3", "m1", 0.0),
        ]
        alpha = compute_krippendorff_alpha(labels)
        assert alpha == pytest.approx(1.0)

    def test_partial_agreement_range(self):
        labels = [
            AnnotatorLabel("i1", "a1", "m1", 1.0),
            AnnotatorLabel("i1", "a2", "m1", 0.8),
            AnnotatorLabel("i2", "a1", "m1", 0.3),
            AnnotatorLabel("i2", "a2", "m1", 0.2),
            AnnotatorLabel("i3", "a1", "m1", 0.5),
            AnnotatorLabel("i3", "a2", "m1", 0.6),
        ]
        alpha = compute_krippendorff_alpha(labels)
        assert -1 <= alpha <= 1

    def test_missing_data_handled(self):
        # a2 did not label i2
        labels = [
            AnnotatorLabel("i1", "a1", "m1", 1.0),
            AnnotatorLabel("i1", "a2", "m1", 1.0),
            AnnotatorLabel("i2", "a1", "m1", 0.0),
            AnnotatorLabel("i3", "a1", "m1", 0.5),
            AnnotatorLabel("i3", "a2", "m1", 0.5),
        ]
        alpha = compute_krippendorff_alpha(labels)
        # Should not raise, result in valid range
        assert -1 <= alpha <= 1

    def test_single_annotator_returns_zero(self):
        labels = [
            AnnotatorLabel("i1", "a1", "m1", 1.0),
            AnnotatorLabel("i2", "a1", "m1", 0.0),
        ]
        # Each item has only 1 annotator, so no pairs
        alpha = compute_krippendorff_alpha(labels)
        assert alpha == 0.0


# ---------------------------------------------------------------------------
# build_pairwise_matrix
# ---------------------------------------------------------------------------


class TestBuildPairwiseMatrix:
    def _make_labels(self):
        return [
            AnnotatorLabel("i1", "a1", "m1", 1.0),
            AnnotatorLabel("i1", "a2", "m1", 1.0),
            AnnotatorLabel("i2", "a1", "m1", 0.0),
            AnnotatorLabel("i2", "a2", "m1", 0.0),
            AnnotatorLabel("i3", "a1", "m1", 1.0),
            AnnotatorLabel("i3", "a2", "m1", 0.0),
        ]

    def test_basic_matrix(self):
        matrix = build_pairwise_matrix(self._make_labels())
        assert ("a1", "a2") in matrix

    def test_score_values(self):
        matrix = build_pairwise_matrix(self._make_labels())
        score = matrix[("a1", "a2")]
        assert score.n_items == 3
        assert 0 <= score.percent_agreement <= 1

    def test_metric_id_preserved(self):
        matrix = build_pairwise_matrix(self._make_labels())
        score = matrix[("a1", "a2")]
        assert score.metric_id == "m1"

    def test_empty_labels(self):
        matrix = build_pairwise_matrix([])
        assert len(matrix) == 0

    def test_no_shared_items(self):
        labels = [
            AnnotatorLabel("i1", "a1", "m1", 1.0),
            AnnotatorLabel("i2", "a2", "m1", 0.0),
        ]
        matrix = build_pairwise_matrix(labels)
        assert len(matrix) == 0

    def test_three_annotators(self):
        labels = [
            AnnotatorLabel("i1", "a1", "m1", 1.0),
            AnnotatorLabel("i1", "a2", "m1", 1.0),
            AnnotatorLabel("i1", "a3", "m1", 1.0),
            AnnotatorLabel("i2", "a1", "m1", 0.0),
            AnnotatorLabel("i2", "a2", "m1", 0.0),
            AnnotatorLabel("i2", "a3", "m1", 0.0),
        ]
        matrix = build_pairwise_matrix(labels)
        # 3 choose 2 = 3 pairs
        assert len(matrix) == 3
        for score in matrix.values():
            assert score.cohens_kappa == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# find_disagreement_items
# ---------------------------------------------------------------------------


class TestFindDisagreementItems:
    def test_high_disagreement(self):
        labels = [
            AnnotatorLabel("i1", "a1", "m1", 1.0),
            AnnotatorLabel("i1", "a2", "m1", 0.0),
            AnnotatorLabel("i2", "a1", "m1", 1.0),
            AnnotatorLabel("i2", "a2", "m1", 1.0),
        ]
        # i1 has variance 0.25, i2 has variance 0.0
        # threshold=0.2 -> i1 included
        items = find_disagreement_items(labels, threshold=0.2)
        assert "i1" in items
        assert "i2" not in items

    def test_no_disagreement(self):
        labels = [
            AnnotatorLabel("i1", "a1", "m1", 1.0),
            AnnotatorLabel("i1", "a2", "m1", 1.0),
        ]
        items = find_disagreement_items(labels, threshold=0.1)
        assert len(items) == 0

    def test_empty(self):
        assert find_disagreement_items([], threshold=0.5) == []

    def test_single_annotator_excluded(self):
        labels = [AnnotatorLabel("i1", "a1", "m1", 1.0)]
        items = find_disagreement_items(labels, threshold=0.0)
        assert len(items) == 0

    def test_sorted_by_variance(self):
        labels = [
            # i1: variance = 0.25 (1.0 vs 0.0)
            AnnotatorLabel("i1", "a1", "m1", 1.0),
            AnnotatorLabel("i1", "a2", "m1", 0.0),
            # i2: variance = 0.0625 (0.5 vs 0.0)
            AnnotatorLabel("i2", "a1", "m1", 0.5),
            AnnotatorLabel("i2", "a2", "m1", 0.0),
        ]
        items = find_disagreement_items(labels, threshold=0.01)
        assert items[0] == "i1"
        assert items[1] == "i2"


# ---------------------------------------------------------------------------
# build_agreement_report
# ---------------------------------------------------------------------------


class TestBuildAgreementReport:
    def _make_labels(self):
        return [
            AnnotatorLabel("i1", "a1", "m1", 1.0),
            AnnotatorLabel("i1", "a2", "m1", 1.0),
            AnnotatorLabel("i2", "a1", "m1", 0.0),
            AnnotatorLabel("i2", "a2", "m1", 0.0),
            AnnotatorLabel("i3", "a1", "m1", 1.0),
            AnnotatorLabel("i3", "a2", "m1", 0.0),
        ]

    def test_report_type(self):
        report = build_agreement_report(self._make_labels())
        assert isinstance(report, AgreementReport)

    def test_report_totals(self):
        report = build_agreement_report(self._make_labels())
        assert report.total_items == 3
        assert report.total_annotators == 2

    def test_report_has_scores(self):
        report = build_agreement_report(self._make_labels())
        assert len(report.scores) >= 1

    def test_report_overall_agreement_range(self):
        report = build_agreement_report(self._make_labels())
        assert 0 <= report.overall_agreement <= 1

    def test_empty_labels(self):
        report = build_agreement_report([])
        assert report.total_items == 0
        assert report.overall_kappa == 0.0

    def test_perfect_agreement_report(self):
        labels = [
            AnnotatorLabel("i1", "a1", "m1", 1.0),
            AnnotatorLabel("i1", "a2", "m1", 1.0),
            AnnotatorLabel("i2", "a1", "m1", 0.0),
            AnnotatorLabel("i2", "a2", "m1", 0.0),
        ]
        report = build_agreement_report(labels)
        assert report.overall_kappa == pytest.approx(1.0)
        assert report.overall_agreement == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# format_agreement_report
# ---------------------------------------------------------------------------


class TestFormatAgreementReport:
    def test_format_nonempty(self):
        report = AgreementReport(
            scores=[AgreementScore("m1", ("a1", "a2"), 0.7, 0.85, 10)],
            overall_kappa=0.7,
            overall_agreement=0.85,
            high_disagreement_items=["item3"],
            total_items=10,
            total_annotators=2,
        )
        text = format_agreement_report(report)
        assert "Inter-Annotator Agreement Report" in text
        assert "0.7000" in text
        assert "a1 vs a2" in text
        assert "item3" in text

    def test_format_empty(self):
        report = AgreementReport(
            scores=[], overall_kappa=0.0, overall_agreement=0.0,
            high_disagreement_items=[], total_items=0, total_annotators=0,
        )
        text = format_agreement_report(report)
        assert "Total items: 0" in text

    def test_format_no_disagreement(self):
        report = AgreementReport(
            scores=[AgreementScore("m1", ("a1", "a2"), 1.0, 1.0, 5)],
            overall_kappa=1.0,
            overall_agreement=1.0,
            high_disagreement_items=[],
            total_items=5,
            total_annotators=2,
        )
        text = format_agreement_report(report)
        assert "High Disagreement Items" not in text

    def test_format_contains_percentages(self):
        report = AgreementReport(
            scores=[AgreementScore("m1", ("a1", "a2"), 0.5, 0.75, 8)],
            overall_kappa=0.5,
            overall_agreement=0.75,
            high_disagreement_items=[],
            total_items=8,
            total_annotators=2,
        )
        text = format_agreement_report(report)
        assert "75.00%" in text
