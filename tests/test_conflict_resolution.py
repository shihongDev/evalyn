"""Tests for the annotation conflict resolution module."""

from __future__ import annotations

import pytest

from evalyn_sdk.conflict_resolution import (
    AnnotatorLabel,
    Conflict,
    Resolution,
    ResolutionPolicy,
    apply_resolution_policy,
    detect_conflicts,
    format_conflict_view,
    format_resolution_summary,
    resolve_all_conflicts,
    resolve_by_majority_vote,
    resolve_by_senior_override,
)


# ---------------------------------------------------------------------------
# AnnotatorLabel
# ---------------------------------------------------------------------------


class TestAnnotatorLabel:
    def test_create_minimal(self):
        lbl = AnnotatorLabel(item_id="i1", annotator_id="a1", label=0.8)
        assert lbl.confidence == 1.0
        assert lbl.reasoning == ""

    def test_create_full(self):
        lbl = AnnotatorLabel("i1", "a1", 0.8, confidence=0.9, reasoning="looks good")
        assert lbl.confidence == 0.9
        assert lbl.reasoning == "looks good"

    def test_as_dict(self):
        lbl = AnnotatorLabel("i1", "a1", 0.5, 0.7, "reason")
        d = lbl.as_dict()
        assert d["item_id"] == "i1"
        assert d["annotator_id"] == "a1"
        assert d["label"] == 0.5
        assert d["confidence"] == 0.7
        assert d["reasoning"] == "reason"

    def test_from_dict_minimal(self):
        lbl = AnnotatorLabel.from_dict(
            {"item_id": "i1", "annotator_id": "a1", "label": 1.0}
        )
        assert lbl.confidence == 1.0
        assert lbl.reasoning == ""

    def test_from_dict_full(self):
        lbl = AnnotatorLabel.from_dict(
            {
                "item_id": "i1",
                "annotator_id": "a1",
                "label": 0.3,
                "confidence": 0.6,
                "reasoning": "unsure",
            }
        )
        assert lbl.label == 0.3
        assert lbl.reasoning == "unsure"

    def test_roundtrip(self):
        orig = AnnotatorLabel("x", "y", 0.42, 0.99, "reason text")
        restored = AnnotatorLabel.from_dict(orig.as_dict())
        assert restored.item_id == orig.item_id
        assert restored.label == orig.label
        assert restored.reasoning == orig.reasoning


# ---------------------------------------------------------------------------
# Conflict
# ---------------------------------------------------------------------------


class TestConflict:
    def _make_conflict(self):
        labels = [
            AnnotatorLabel("i1", "a1", 0.9),
            AnnotatorLabel("i1", "a2", 0.1),
        ]
        return Conflict(item_id="i1", labels=labels, max_disagreement=0.8, needs_tiebreaker=True)

    def test_create(self):
        c = self._make_conflict()
        assert c.item_id == "i1"
        assert len(c.labels) == 2
        assert c.needs_tiebreaker is True

    def test_as_dict(self):
        d = self._make_conflict().as_dict()
        assert d["item_id"] == "i1"
        assert len(d["labels"]) == 2
        assert d["max_disagreement"] == 0.8

    def test_from_dict(self):
        d = self._make_conflict().as_dict()
        c = Conflict.from_dict(d)
        assert c.item_id == "i1"
        assert c.labels[0].annotator_id == "a1"

    def test_roundtrip(self):
        orig = self._make_conflict()
        restored = Conflict.from_dict(orig.as_dict())
        assert restored.max_disagreement == orig.max_disagreement
        assert len(restored.labels) == len(orig.labels)


# ---------------------------------------------------------------------------
# ResolutionPolicy
# ---------------------------------------------------------------------------


class TestResolutionPolicy:
    def test_create_defaults(self):
        p = ResolutionPolicy(policy_name="majority_vote")
        assert p.senior_annotators == []
        assert p.min_agreement == 0.5

    def test_create_full(self):
        p = ResolutionPolicy("senior_override", ["senior1"], 0.7)
        assert p.senior_annotators == ["senior1"]

    def test_as_dict(self):
        p = ResolutionPolicy("majority_vote", ["s1", "s2"], 0.6)
        d = p.as_dict()
        assert d["policy_name"] == "majority_vote"
        assert d["senior_annotators"] == ["s1", "s2"]

    def test_from_dict_minimal(self):
        p = ResolutionPolicy.from_dict({"policy_name": "discussion_required"})
        assert p.senior_annotators == []
        assert p.min_agreement == 0.5

    def test_roundtrip(self):
        orig = ResolutionPolicy("senior_override", ["s1"], 0.8)
        restored = ResolutionPolicy.from_dict(orig.as_dict())
        assert restored.policy_name == orig.policy_name
        assert restored.senior_annotators == orig.senior_annotators


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


class TestResolution:
    def test_create_defaults(self):
        r = Resolution(item_id="i1", resolved_label=1.0, method="majority_vote")
        assert r.resolved_by == ""
        assert r.original_labels == []

    def test_as_dict(self):
        r = Resolution("i1", 1.0, "majority_vote", "voter", [AnnotatorLabel("i1", "a1", 0.9)])
        d = r.as_dict()
        assert d["resolved_by"] == "voter"
        assert len(d["original_labels"]) == 1

    def test_from_dict(self):
        d = {
            "item_id": "i1",
            "resolved_label": 0.5,
            "method": "senior_override",
            "resolved_by": "senior1",
            "original_labels": [
                {"item_id": "i1", "annotator_id": "a1", "label": 0.9}
            ],
        }
        r = Resolution.from_dict(d)
        assert r.resolved_by == "senior1"
        assert r.original_labels[0].annotator_id == "a1"

    def test_roundtrip(self):
        orig = Resolution("i1", 0.7, "test", "tester", [AnnotatorLabel("i1", "a1", 0.7)])
        restored = Resolution.from_dict(orig.as_dict())
        assert restored.item_id == orig.item_id
        assert restored.resolved_label == orig.resolved_label


# ---------------------------------------------------------------------------
# detect_conflicts
# ---------------------------------------------------------------------------


class TestDetectConflicts:
    def test_no_labels(self):
        assert detect_conflicts([]) == []

    def test_single_annotator_no_conflict(self):
        labels = [AnnotatorLabel("i1", "a1", 0.9)]
        assert detect_conflicts(labels) == []

    def test_agreement_below_threshold(self):
        labels = [
            AnnotatorLabel("i1", "a1", 0.7),
            AnnotatorLabel("i1", "a2", 0.8),
        ]
        # diff = 0.1, threshold = 0.5 -> no conflict
        assert detect_conflicts(labels, threshold=0.5) == []

    def test_disagreement_above_threshold(self):
        labels = [
            AnnotatorLabel("i1", "a1", 0.9),
            AnnotatorLabel("i1", "a2", 0.1),
        ]
        conflicts = detect_conflicts(labels, threshold=0.5)
        assert len(conflicts) == 1
        assert conflicts[0].item_id == "i1"
        assert conflicts[0].max_disagreement == pytest.approx(0.8)

    def test_multiple_items_mixed(self):
        labels = [
            AnnotatorLabel("i1", "a1", 0.9),
            AnnotatorLabel("i1", "a2", 0.1),
            AnnotatorLabel("i2", "a1", 0.5),
            AnnotatorLabel("i2", "a2", 0.5),
        ]
        conflicts = detect_conflicts(labels, threshold=0.5)
        assert len(conflicts) == 1
        assert conflicts[0].item_id == "i1"

    def test_threshold_exact_boundary(self):
        labels = [
            AnnotatorLabel("i1", "a1", 1.0),
            AnnotatorLabel("i1", "a2", 0.5),
        ]
        # diff = 0.5, threshold = 0.5 -> NOT above -> no conflict
        assert detect_conflicts(labels, threshold=0.5) == []

    def test_three_annotators(self):
        labels = [
            AnnotatorLabel("i1", "a1", 1.0),
            AnnotatorLabel("i1", "a2", 0.5),
            AnnotatorLabel("i1", "a3", 0.0),
        ]
        conflicts = detect_conflicts(labels, threshold=0.5)
        assert len(conflicts) == 1
        assert conflicts[0].max_disagreement == pytest.approx(1.0)

    def test_needs_tiebreaker_set(self):
        labels = [
            AnnotatorLabel("i1", "a1", 1.0),
            AnnotatorLabel("i1", "a2", 0.0),
        ]
        conflicts = detect_conflicts(labels)
        assert conflicts[0].needs_tiebreaker is True

    def test_custom_threshold(self):
        labels = [
            AnnotatorLabel("i1", "a1", 0.6),
            AnnotatorLabel("i1", "a2", 0.4),
        ]
        # diff = 0.2
        assert detect_conflicts(labels, threshold=0.1) != []
        assert detect_conflicts(labels, threshold=0.3) == []


# ---------------------------------------------------------------------------
# resolve_by_majority_vote
# ---------------------------------------------------------------------------


class TestResolveMajorityVote:
    def _conflict(self, labels_values):
        labels = [
            AnnotatorLabel("i1", f"a{i}", v) for i, v in enumerate(labels_values)
        ]
        return Conflict("i1", labels, max_disagreement=1.0, needs_tiebreaker=True)

    def test_majority_ones(self):
        c = self._conflict([0.8, 0.9, 0.1])
        r = resolve_by_majority_vote(c)
        assert r.resolved_label == 1.0
        assert r.method == "majority_vote"

    def test_majority_zeros(self):
        c = self._conflict([0.1, 0.2, 0.3])
        r = resolve_by_majority_vote(c)
        assert r.resolved_label == 0.0

    def test_tie_goes_to_one(self):
        c = self._conflict([0.8, 0.2])
        r = resolve_by_majority_vote(c)
        assert r.resolved_label == 1.0

    def test_custom_threshold(self):
        c = self._conflict([0.6, 0.7, 0.3])
        r = resolve_by_majority_vote(c, threshold=0.65)
        # 0.6 < 0.65 -> 0, 0.7 >= 0.65 -> 1, 0.3 < 0.65 -> 0
        assert r.resolved_label == 0.0

    def test_preserves_original_labels(self):
        c = self._conflict([0.8, 0.2])
        r = resolve_by_majority_vote(c)
        assert len(r.original_labels) == 2

    def test_all_above_threshold(self):
        c = self._conflict([0.9, 0.8, 0.7])
        r = resolve_by_majority_vote(c)
        assert r.resolved_label == 1.0


# ---------------------------------------------------------------------------
# resolve_by_senior_override
# ---------------------------------------------------------------------------


class TestResolveSeniorOverride:
    def _conflict(self):
        labels = [
            AnnotatorLabel("i1", "junior1", 0.2),
            AnnotatorLabel("i1", "senior1", 0.9),
            AnnotatorLabel("i1", "junior2", 0.3),
        ]
        return Conflict("i1", labels, max_disagreement=0.7, needs_tiebreaker=True)

    def test_finds_senior(self):
        r = resolve_by_senior_override(self._conflict(), ["senior1"])
        assert r is not None
        assert r.resolved_label == 0.9
        assert r.resolved_by == "senior1"

    def test_no_senior_present(self):
        r = resolve_by_senior_override(self._conflict(), ["senior99"])
        assert r is None

    def test_first_senior_wins(self):
        labels = [
            AnnotatorLabel("i1", "senior2", 0.3),
            AnnotatorLabel("i1", "senior1", 0.9),
        ]
        c = Conflict("i1", labels, 0.6, True)
        r = resolve_by_senior_override(c, ["senior1", "senior2"])
        # senior2 appears first in labels
        assert r is not None
        assert r.resolved_by == "senior2"

    def test_method_name(self):
        r = resolve_by_senior_override(self._conflict(), ["senior1"])
        assert r.method == "senior_override"


# ---------------------------------------------------------------------------
# apply_resolution_policy
# ---------------------------------------------------------------------------


class TestApplyResolutionPolicy:
    def _conflict(self):
        labels = [
            AnnotatorLabel("i1", "a1", 0.9),
            AnnotatorLabel("i1", "a2", 0.1),
        ]
        return Conflict("i1", labels, 0.8, True)

    def test_majority_vote_policy(self):
        policy = ResolutionPolicy("majority_vote")
        r = apply_resolution_policy(self._conflict(), policy)
        assert r is not None
        assert r.method == "majority_vote"

    def test_senior_override_policy(self):
        policy = ResolutionPolicy("senior_override", ["a1"])
        r = apply_resolution_policy(self._conflict(), policy)
        assert r is not None
        assert r.resolved_by == "a1"

    def test_senior_override_no_match(self):
        policy = ResolutionPolicy("senior_override", ["nobody"])
        r = apply_resolution_policy(self._conflict(), policy)
        assert r is None

    def test_discussion_required_returns_none(self):
        policy = ResolutionPolicy("discussion_required")
        r = apply_resolution_policy(self._conflict(), policy)
        assert r is None

    def test_unknown_policy_returns_none(self):
        policy = ResolutionPolicy("unknown_policy")
        r = apply_resolution_policy(self._conflict(), policy)
        assert r is None

    def test_min_agreement_passed_to_majority(self):
        policy = ResolutionPolicy("majority_vote", min_agreement=0.9)
        labels = [
            AnnotatorLabel("i1", "a1", 0.85),
            AnnotatorLabel("i1", "a2", 0.95),
        ]
        c = Conflict("i1", labels, 0.1, True)
        r = apply_resolution_policy(c, policy)
        assert r is not None
        # 0.85 < 0.9 -> 0, 0.95 >= 0.9 -> 1, tie goes to 1
        assert r.resolved_label == 1.0


# ---------------------------------------------------------------------------
# resolve_all_conflicts
# ---------------------------------------------------------------------------


class TestResolveAllConflicts:
    def test_all_resolved(self):
        c1 = Conflict("i1", [AnnotatorLabel("i1", "a1", 0.9), AnnotatorLabel("i1", "a2", 0.1)], 0.8, True)
        c2 = Conflict("i2", [AnnotatorLabel("i2", "a1", 0.8), AnnotatorLabel("i2", "a2", 0.2)], 0.6, True)
        policy = ResolutionPolicy("majority_vote")
        resolved, unresolved = resolve_all_conflicts([c1, c2], policy)
        assert len(resolved) == 2
        assert len(unresolved) == 0

    def test_none_resolved(self):
        c1 = Conflict("i1", [AnnotatorLabel("i1", "a1", 0.9), AnnotatorLabel("i1", "a2", 0.1)], 0.8, True)
        policy = ResolutionPolicy("discussion_required")
        resolved, unresolved = resolve_all_conflicts([c1], policy)
        assert len(resolved) == 0
        assert len(unresolved) == 1

    def test_mixed(self):
        c1 = Conflict("i1", [AnnotatorLabel("i1", "senior1", 0.9), AnnotatorLabel("i1", "a2", 0.1)], 0.8, True)
        c2 = Conflict("i2", [AnnotatorLabel("i2", "a1", 0.8), AnnotatorLabel("i2", "a2", 0.2)], 0.6, True)
        policy = ResolutionPolicy("senior_override", ["senior1"])
        resolved, unresolved = resolve_all_conflicts([c1, c2], policy)
        assert len(resolved) == 1
        assert resolved[0].item_id == "i1"
        assert len(unresolved) == 1
        assert unresolved[0].item_id == "i2"

    def test_empty_input(self):
        resolved, unresolved = resolve_all_conflicts([], ResolutionPolicy("majority_vote"))
        assert resolved == []
        assert unresolved == []


# ---------------------------------------------------------------------------
# format_conflict_view
# ---------------------------------------------------------------------------


class TestFormatConflictView:
    def test_basic_output(self):
        labels = [
            AnnotatorLabel("i1", "alice", 0.9, 0.95, "clearly correct"),
            AnnotatorLabel("i1", "bob", 0.2, 0.60, "seems wrong"),
        ]
        c = Conflict("i1", labels, 0.7, True)
        text = format_conflict_view(c)
        assert "CONFLICT: i1" in text
        assert "alice" in text
        assert "bob" in text
        assert "0.700" in text  # max_disagreement
        assert "clearly correct" in text

    def test_empty_reasoning_shows_dash(self):
        labels = [
            AnnotatorLabel("i1", "a1", 0.5),
            AnnotatorLabel("i1", "a2", 0.1),
        ]
        c = Conflict("i1", labels, 0.4, True)
        text = format_conflict_view(c)
        assert "-" in text

    def test_tiebreaker_flag(self):
        labels = [AnnotatorLabel("i1", "a1", 0.5), AnnotatorLabel("i1", "a2", 0.0)]
        c = Conflict("i1", labels, 0.5, False)
        text = format_conflict_view(c)
        assert "Needs tiebreaker: False" in text


# ---------------------------------------------------------------------------
# format_resolution_summary
# ---------------------------------------------------------------------------


class TestFormatResolutionSummary:
    def test_all_resolved(self):
        resolutions = [
            Resolution("i1", 1.0, "majority_vote", "majority"),
            Resolution("i2", 0.0, "majority_vote", "majority"),
        ]
        text = format_resolution_summary(resolutions, [])
        assert "Total conflicts: 2" in text
        assert "Resolved: 2" in text
        assert "Unresolved: 0" in text
        assert "i1" in text

    def test_all_unresolved(self):
        conflict = Conflict(
            "i1",
            [AnnotatorLabel("i1", "a1", 0.9), AnnotatorLabel("i1", "a2", 0.1)],
            0.8,
            True,
        )
        text = format_resolution_summary([], [conflict])
        assert "Unresolved: 1" in text
        assert "need discussion" in text
        assert "a1, a2" in text

    def test_mixed(self):
        resolutions = [Resolution("i1", 1.0, "majority_vote")]
        unresolved = [
            Conflict("i2", [AnnotatorLabel("i2", "a1", 0.5)], 0.3, True)
        ]
        text = format_resolution_summary(resolutions, unresolved)
        assert "Resolved: 1" in text
        assert "Unresolved: 1" in text

    def test_empty(self):
        text = format_resolution_summary([], [])
        assert "Total conflicts: 0" in text
