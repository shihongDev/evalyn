"""Tests for the team collaboration module."""

from __future__ import annotations

import sys
from pathlib import Path

# Add SDK to path
SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.integration.team_collaboration import (
    Annotator,
    Annotation,
    CollaborationReport,
    ConflictResolution,
    assign_items,
    build_collaboration_report,
    compute_inter_annotator_agreement,
    detect_conflicts,
    resolve_by_majority,
)


# ---------------------------------------------------------------------------
# Annotator
# ---------------------------------------------------------------------------


class TestAnnotator:
    def test_defaults(self):
        a = Annotator()
        assert a.user_id == ""
        assert a.name == ""
        assert a.role == "annotator"

    def test_as_dict(self):
        a = Annotator(user_id="u1", name="Alice", role="reviewer")
        d = a.as_dict()
        assert d["user_id"] == "u1"
        assert d["name"] == "Alice"
        assert d["role"] == "reviewer"

    def test_from_dict(self):
        d = {"user_id": "u2", "name": "Bob", "role": "admin"}
        a = Annotator.from_dict(d)
        assert a.user_id == "u2"
        assert a.name == "Bob"
        assert a.role == "admin"

    def test_from_dict_defaults(self):
        a = Annotator.from_dict({})
        assert a.user_id == ""
        assert a.role == "annotator"

    def test_roundtrip(self):
        a = Annotator(user_id="u3", name="Carol", role="reviewer")
        restored = Annotator.from_dict(a.as_dict())
        assert restored.user_id == a.user_id
        assert restored.name == a.name
        assert restored.role == a.role


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------


class TestAnnotation:
    def test_defaults(self):
        a = Annotation()
        assert a.confidence == 1.0
        assert a.timestamp == ""

    def test_as_dict(self):
        a = Annotation(item_id="i1", annotator_id="u1", label="good", confidence=0.9)
        d = a.as_dict()
        assert d["item_id"] == "i1"
        assert d["label"] == "good"
        assert d["confidence"] == 0.9

    def test_from_dict(self):
        d = {"item_id": "i2", "annotator_id": "u2", "label": "bad", "confidence": 0.5, "notes": "unsure"}
        a = Annotation.from_dict(d)
        assert a.item_id == "i2"
        assert a.label == "bad"
        assert a.confidence == 0.5
        assert a.notes == "unsure"

    def test_roundtrip(self):
        a = Annotation(item_id="i3", annotator_id="u3", label="ok", confidence=0.8, timestamp="2026-01-01", notes="fine")
        restored = Annotation.from_dict(a.as_dict())
        assert restored.item_id == a.item_id
        assert restored.label == a.label
        assert restored.confidence == a.confidence


# ---------------------------------------------------------------------------
# ConflictResolution
# ---------------------------------------------------------------------------


class TestConflictResolution:
    def test_as_dict(self):
        cr = ConflictResolution(
            item_id="i1",
            annotations=[Annotation(item_id="i1", annotator_id="u1", label="good")],
            resolved_label="good",
            resolution_method="majority",
            resolved_by="system",
        )
        d = cr.as_dict()
        assert d["item_id"] == "i1"
        assert d["resolved_label"] == "good"
        assert len(d["annotations"]) == 1


# ---------------------------------------------------------------------------
# CollaborationReport
# ---------------------------------------------------------------------------


class TestCollaborationReport:
    def test_defaults(self):
        r = CollaborationReport()
        assert r.total_items == 0
        assert r.agreement_rate == 0.0

    def test_as_dict(self):
        r = CollaborationReport(total_items=10, annotated_items=8, conflicts=2, agreement_rate=0.75)
        d = r.as_dict()
        assert d["total_items"] == 10
        assert d["conflicts"] == 2

    def test_from_dict(self):
        d = {
            "total_items": 5,
            "annotated_items": 5,
            "conflicts": 1,
            "resolved": 1,
            "agreement_rate": 0.8,
            "annotators": [{"user_id": "u1", "name": "Alice", "role": "annotator"}],
        }
        r = CollaborationReport.from_dict(d)
        assert r.total_items == 5
        assert len(r.annotators) == 1
        assert r.annotators[0].user_id == "u1"

    def test_roundtrip(self):
        r = CollaborationReport(
            total_items=10,
            annotated_items=10,
            conflicts=3,
            resolved=2,
            agreement_rate=0.7,
            annotators=[Annotator(user_id="u1", name="Alice")],
        )
        restored = CollaborationReport.from_dict(r.as_dict())
        assert restored.total_items == r.total_items
        assert restored.agreement_rate == r.agreement_rate
        assert len(restored.annotators) == 1

    def test_format_text(self):
        r = CollaborationReport(total_items=10, annotated_items=8, conflicts=2, agreement_rate=0.85)
        text = r.format_text()
        assert "Collaboration Report" in text
        assert "total_items: 10" in text
        assert "agreement_rate: 0.85" in text


# ---------------------------------------------------------------------------
# assign_items
# ---------------------------------------------------------------------------


class TestAssignItems:
    def test_round_robin(self):
        annotators = [Annotator(user_id="u1"), Annotator(user_id="u2")]
        items = ["i1", "i2", "i3", "i4"]
        assignments = assign_items(items, annotators, overlap=1)
        # With overlap=1 and round-robin, items alternate
        assert "i1" in assignments["u1"]
        assert "i2" in assignments["u2"]
        assert "i3" in assignments["u1"]
        assert "i4" in assignments["u2"]

    def test_with_overlap(self):
        annotators = [Annotator(user_id="u1"), Annotator(user_id="u2"), Annotator(user_id="u3")]
        items = ["i1", "i2"]
        assignments = assign_items(items, annotators, overlap=2)
        # Each item assigned to 2 annotators
        total_assignments = sum(len(v) for v in assignments.values())
        assert total_assignments == 4  # 2 items x 2 overlap

    def test_overlap_greater_than_annotators(self):
        annotators = [Annotator(user_id="u1")]
        items = ["i1", "i2"]
        assignments = assign_items(items, annotators, overlap=3)
        # overlap clamped to number of annotators
        assert len(assignments["u1"]) == 2

    def test_empty_annotators(self):
        assignments = assign_items(["i1", "i2"], [], overlap=2)
        assert assignments == {}

    def test_empty_items(self):
        annotators = [Annotator(user_id="u1")]
        assignments = assign_items([], annotators, overlap=2)
        assert assignments["u1"] == []

    def test_single_annotator(self):
        annotators = [Annotator(user_id="u1")]
        items = ["i1", "i2", "i3"]
        assignments = assign_items(items, annotators, overlap=1)
        assert assignments["u1"] == ["i1", "i2", "i3"]


# ---------------------------------------------------------------------------
# detect_conflicts
# ---------------------------------------------------------------------------


class TestDetectConflicts:
    def test_finds_disagreements(self):
        annotations = [
            Annotation(item_id="i1", annotator_id="u1", label="good"),
            Annotation(item_id="i1", annotator_id="u2", label="bad"),
            Annotation(item_id="i2", annotator_id="u1", label="ok"),
            Annotation(item_id="i2", annotator_id="u2", label="ok"),
        ]
        conflicts = detect_conflicts(annotations)
        assert conflicts == ["i1"]

    def test_no_conflicts(self):
        annotations = [
            Annotation(item_id="i1", annotator_id="u1", label="good"),
            Annotation(item_id="i1", annotator_id="u2", label="good"),
        ]
        conflicts = detect_conflicts(annotations)
        assert conflicts == []

    def test_empty_annotations(self):
        assert detect_conflicts([]) == []

    def test_multiple_conflicts(self):
        annotations = [
            Annotation(item_id="i1", annotator_id="u1", label="a"),
            Annotation(item_id="i1", annotator_id="u2", label="b"),
            Annotation(item_id="i2", annotator_id="u1", label="x"),
            Annotation(item_id="i2", annotator_id="u2", label="y"),
        ]
        conflicts = detect_conflicts(annotations)
        assert conflicts == ["i1", "i2"]


# ---------------------------------------------------------------------------
# resolve_by_majority
# ---------------------------------------------------------------------------


class TestResolveByMajority:
    def test_picks_winner(self):
        annotations = [
            Annotation(item_id="i1", annotator_id="u1", label="good"),
            Annotation(item_id="i1", annotator_id="u2", label="bad"),
            Annotation(item_id="i1", annotator_id="u3", label="good"),
        ]
        result = resolve_by_majority(annotations, "i1")
        assert result.resolved_label == "good"
        assert result.resolution_method == "majority"
        assert len(result.annotations) == 3

    def test_filters_to_item(self):
        annotations = [
            Annotation(item_id="i1", annotator_id="u1", label="good"),
            Annotation(item_id="i2", annotator_id="u2", label="bad"),
        ]
        result = resolve_by_majority(annotations, "i1")
        assert len(result.annotations) == 1
        assert result.resolved_label == "good"

    def test_no_annotations_for_item(self):
        result = resolve_by_majority([], "i99")
        assert result.resolved_label == ""

    def test_tie_picks_most_common(self):
        # With Counter.most_common, ties are resolved by insertion order
        annotations = [
            Annotation(item_id="i1", annotator_id="u1", label="a"),
            Annotation(item_id="i1", annotator_id="u2", label="b"),
        ]
        result = resolve_by_majority(annotations, "i1")
        assert result.resolved_label in ("a", "b")


# ---------------------------------------------------------------------------
# compute_inter_annotator_agreement
# ---------------------------------------------------------------------------


class TestInterAnnotatorAgreement:
    def test_perfect_agreement(self):
        annotations = [
            Annotation(item_id="i1", annotator_id="u1", label="good"),
            Annotation(item_id="i1", annotator_id="u2", label="good"),
            Annotation(item_id="i2", annotator_id="u1", label="bad"),
            Annotation(item_id="i2", annotator_id="u2", label="bad"),
        ]
        kappa = compute_inter_annotator_agreement(annotations)
        assert kappa == 1.0

    def test_no_agreement(self):
        annotations = [
            Annotation(item_id="i1", annotator_id="u1", label="good"),
            Annotation(item_id="i1", annotator_id="u2", label="bad"),
            Annotation(item_id="i2", annotator_id="u1", label="bad"),
            Annotation(item_id="i2", annotator_id="u2", label="good"),
        ]
        kappa = compute_inter_annotator_agreement(annotations)
        # 0 agreement, 0.5 expected => (0 - 0.5)/(1 - 0.5) = -1.0
        assert kappa == -1.0

    def test_empty_annotations(self):
        assert compute_inter_annotator_agreement([]) == 0.0

    def test_single_item_agreement(self):
        annotations = [
            Annotation(item_id="i1", annotator_id="u1", label="good"),
            Annotation(item_id="i1", annotator_id="u2", label="good"),
        ]
        kappa = compute_inter_annotator_agreement(annotations)
        assert kappa == 1.0

    def test_single_annotation_per_item(self):
        annotations = [
            Annotation(item_id="i1", annotator_id="u1", label="good"),
            Annotation(item_id="i2", annotator_id="u2", label="bad"),
        ]
        # No items with multiple annotations -> perfect agreement by default
        kappa = compute_inter_annotator_agreement(annotations)
        assert kappa == 1.0


# ---------------------------------------------------------------------------
# build_collaboration_report
# ---------------------------------------------------------------------------


class TestBuildCollaborationReport:
    def test_basic_report(self):
        annotators = [Annotator(user_id="u1", name="Alice"), Annotator(user_id="u2", name="Bob")]
        annotations = [
            Annotation(item_id="i1", annotator_id="u1", label="good"),
            Annotation(item_id="i1", annotator_id="u2", label="good"),
            Annotation(item_id="i2", annotator_id="u1", label="bad"),
            Annotation(item_id="i2", annotator_id="u2", label="bad"),
        ]
        report = build_collaboration_report(annotators, annotations)
        assert report.total_items == 2
        assert report.annotated_items == 2
        assert report.conflicts == 0
        assert report.agreement_rate == 1.0
        assert len(report.annotators) == 2

    def test_report_with_conflicts(self):
        annotators = [Annotator(user_id="u1"), Annotator(user_id="u2")]
        annotations = [
            Annotation(item_id="i1", annotator_id="u1", label="good"),
            Annotation(item_id="i1", annotator_id="u2", label="bad"),
        ]
        report = build_collaboration_report(annotators, annotations)
        assert report.conflicts == 1

    def test_empty_annotations(self):
        annotators = [Annotator(user_id="u1")]
        report = build_collaboration_report(annotators, [])
        assert report.total_items == 0
        assert report.annotated_items == 0

    def test_all_agree(self):
        annotators = [Annotator(user_id="u1"), Annotator(user_id="u2")]
        annotations = [
            Annotation(item_id="i1", annotator_id="u1", label="ok"),
            Annotation(item_id="i1", annotator_id="u2", label="ok"),
        ]
        report = build_collaboration_report(annotators, annotations)
        assert report.conflicts == 0
        assert report.agreement_rate == 1.0
