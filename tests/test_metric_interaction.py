"""Tests for metric_interaction module."""

from __future__ import annotations

import sys
from pathlib import Path

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

import pytest

from evalyn_sdk.analysis.metric_interaction import (
    InteractionEffect,
    InteractionReport,
    compute_conditional_pass_rate,
    compute_joint_pass_rate,
    detect_all_interactions,
    detect_interaction,
    find_strongest_conflicts,
    find_strongest_synergies,
)


# ---------------------------------------------------------------------------
# compute_joint_pass_rate
# ---------------------------------------------------------------------------


class TestComputeJointPassRate:
    def test_all_pass(self):
        a = [True, True, True]
        b = [True, True, True]
        assert compute_joint_pass_rate(a, b) == pytest.approx(1.0)

    def test_all_fail(self):
        a = [False, False, False]
        b = [False, False, False]
        assert compute_joint_pass_rate(a, b) == pytest.approx(0.0)

    def test_mixed(self):
        a = [True, True, False, False]
        b = [True, False, True, False]
        # Both pass only at index 0
        assert compute_joint_pass_rate(a, b) == pytest.approx(0.25)

    def test_empty(self):
        assert compute_joint_pass_rate([], []) == pytest.approx(0.0)

    def test_single_item_both_pass(self):
        assert compute_joint_pass_rate([True], [True]) == pytest.approx(1.0)

    def test_single_item_one_fails(self):
        assert compute_joint_pass_rate([True], [False]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_conditional_pass_rate
# ---------------------------------------------------------------------------


class TestComputeConditionalPassRate:
    def test_all_pass(self):
        a = [True, True, True]
        b = [True, True, True]
        # P(B|A) = 3/3 = 1.0
        assert compute_conditional_pass_rate(a, b) == pytest.approx(1.0)

    def test_a_never_passes(self):
        a = [False, False, False]
        b = [True, True, True]
        assert compute_conditional_pass_rate(a, b) == pytest.approx(0.0)

    def test_b_fails_when_a_passes(self):
        a = [True, True, False]
        b = [False, False, True]
        # A passes at 0,1. B passes at neither 0 nor 1 -> 0/2
        assert compute_conditional_pass_rate(a, b) == pytest.approx(0.0)

    def test_mixed(self):
        a = [True, True, False, True]
        b = [True, False, True, True]
        # A passes at 0,1,3. B passes at 0,3 -> 2/3
        assert compute_conditional_pass_rate(a, b) == pytest.approx(2.0 / 3.0)

    def test_empty(self):
        assert compute_conditional_pass_rate([], []) == pytest.approx(0.0)

    def test_single_item(self):
        assert compute_conditional_pass_rate([True], [True]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# detect_interaction
# ---------------------------------------------------------------------------


class TestDetectInteraction:
    def test_synergy_detected(self):
        # Both pass together more than expected by independence
        # A passes 50%, B passes 50%, but they always pass together
        a = [True, True, False, False]
        b = [True, True, False, False]
        # rate_a=0.5, rate_b=0.5, expected_ind=0.25, joint=0.5 > 0.25 -> synergy
        e = detect_interaction("ma", a, "mb", b)
        assert e.interaction_type == "synergy"
        assert e.strength == pytest.approx(0.25)

    def test_conflict_detected(self):
        # Passing A makes B less likely to pass
        # A passes at 0,1,2; B passes at 3,4,5
        a = [True, True, True, False, False, False]
        b = [False, False, False, True, True, True]
        # rate_a=0.5, rate_b=0.5, joint=0.0, expected=0.25
        # joint < expected, so not synergy
        # conditional_b_given_a = 0/3 = 0.0 < rate_b=0.5 -> conflict
        e = detect_interaction("ma", a, "mb", b)
        assert e.interaction_type == "conflict"
        assert e.strength == pytest.approx(0.5)

    def test_independent_detected(self):
        # Metrics that are roughly independent
        # A passes 50%, B passes 100% -- joint=0.5, expected=0.5, cond=1.0 >= 1.0
        a = [True, False, True, False]
        b = [True, True, True, True]
        # rate_a=0.5, rate_b=1.0, expected=0.5, joint=0.5
        # joint == expected -> not synergy
        # cond_b_given_a = 2/2 = 1.0 >= rate_b=1.0 -> not conflict
        e = detect_interaction("ma", a, "mb", b)
        assert e.interaction_type == "independent"
        assert e.strength == 0.0

    def test_all_pass(self):
        a = [True, True, True, True]
        b = [True, True, True, True]
        # joint=1.0, expected=1.0 -> not synergy
        # cond=1.0 >= 1.0 -> not conflict
        e = detect_interaction("ma", a, "mb", b)
        assert e.interaction_type == "independent"

    def test_all_fail(self):
        a = [False, False, False]
        b = [False, False, False]
        # joint=0, expected=0 -> not synergy
        # cond=0 (a never passes returns 0), rate_b=0 -> not conflict (0 < 0 is false)
        e = detect_interaction("ma", a, "mb", b)
        assert e.interaction_type == "independent"

    def test_metric_names_preserved(self):
        e = detect_interaction("accuracy", [True], "f1", [True])
        assert e.metric_a == "accuracy"
        assert e.metric_b == "f1"

    def test_description_non_empty_for_synergy(self):
        a = [True, True, False, False]
        b = [True, True, False, False]
        e = detect_interaction("ma", a, "mb", b)
        assert e.description != ""


# ---------------------------------------------------------------------------
# detect_all_interactions
# ---------------------------------------------------------------------------


class TestDetectAllInteractions:
    def test_two_metrics(self):
        scores = {
            "a": [True, True, False, False],
            "b": [True, True, False, False],
        }
        report = detect_all_interactions(scores)
        assert report.total_pairs == 1
        assert len(report.effects) == 1

    def test_three_metrics(self):
        scores = {
            "a": [True, True, False],
            "b": [True, False, True],
            "c": [False, True, True],
        }
        report = detect_all_interactions(scores)
        # 3 choose 2 = 3 pairs
        assert report.total_pairs == 3
        assert len(report.effects) == 3

    def test_single_metric(self):
        scores = {"a": [True, False]}
        report = detect_all_interactions(scores)
        assert report.total_pairs == 0
        assert report.effects == []

    def test_empty_metrics(self):
        report = detect_all_interactions({})
        assert report.total_pairs == 0

    def test_synergy_count(self):
        # All synergy pairs
        scores = {
            "a": [True, True, False, False],
            "b": [True, True, False, False],
            "c": [True, True, False, False],
        }
        report = detect_all_interactions(scores)
        assert report.synergies == 3  # all 3 pairs are synergy


# ---------------------------------------------------------------------------
# find_strongest_synergies / find_strongest_conflicts
# ---------------------------------------------------------------------------


class TestFindStrongest:
    def _make_report(self) -> InteractionReport:
        effects = [
            InteractionEffect("a", "b", "synergy", 0.3),
            InteractionEffect("a", "c", "synergy", 0.1),
            InteractionEffect("b", "c", "conflict", 0.5),
            InteractionEffect("a", "d", "conflict", 0.2),
            InteractionEffect("c", "d", "independent", 0.0),
        ]
        return InteractionReport(
            effects=effects, total_pairs=5, synergies=2, conflicts=2
        )

    def test_top_synergies(self):
        report = self._make_report()
        top = find_strongest_synergies(report, top_n=2)
        assert len(top) == 2
        assert top[0].strength == pytest.approx(0.3)
        assert top[1].strength == pytest.approx(0.1)

    def test_top_conflicts(self):
        report = self._make_report()
        top = find_strongest_conflicts(report, top_n=2)
        assert len(top) == 2
        assert top[0].strength == pytest.approx(0.5)
        assert top[1].strength == pytest.approx(0.2)

    def test_top_n_exceeds_available(self):
        report = self._make_report()
        top = find_strongest_synergies(report, top_n=100)
        assert len(top) == 2  # only 2 synergies exist

    def test_no_synergies(self):
        report = InteractionReport(
            effects=[InteractionEffect("a", "b", "conflict", 0.5)],
            total_pairs=1, synergies=0, conflicts=1,
        )
        assert find_strongest_synergies(report) == []

    def test_no_conflicts(self):
        report = InteractionReport(
            effects=[InteractionEffect("a", "b", "synergy", 0.5)],
            total_pairs=1, synergies=1, conflicts=0,
        )
        assert find_strongest_conflicts(report) == []


# ---------------------------------------------------------------------------
# InteractionReport format_text
# ---------------------------------------------------------------------------


class TestInteractionReportFormat:
    def test_format_text_header(self):
        report = InteractionReport(total_pairs=3, synergies=1, conflicts=1)
        text = report.format_text()
        assert "3 pairs" in text
        assert "Synergies: 1" in text
        assert "Conflicts: 1" in text

    def test_format_text_effects(self):
        effects = [InteractionEffect("a", "b", "synergy", 0.25, "strong link")]
        report = InteractionReport(effects=effects, total_pairs=1, synergies=1, conflicts=0)
        text = report.format_text()
        assert "SYNERGY" in text
        assert "a x b" in text
        assert "strong link" in text


# ---------------------------------------------------------------------------
# InteractionEffect as_dict
# ---------------------------------------------------------------------------


class TestInteractionEffectAsDict:
    def test_as_dict(self):
        e = InteractionEffect("a", "b", "synergy", 0.3, "desc")
        d = e.as_dict()
        assert d["metric_a"] == "a"
        assert d["metric_b"] == "b"
        assert d["interaction_type"] == "synergy"
        assert d["strength"] == 0.3
        assert d["description"] == "desc"


# ---------------------------------------------------------------------------
# InteractionReport as_dict / from_dict roundtrip
# ---------------------------------------------------------------------------


class TestInteractionReportRoundtrip:
    def test_roundtrip(self):
        effects = [
            InteractionEffect("a", "b", "synergy", 0.3, "desc1"),
            InteractionEffect("c", "d", "conflict", 0.1, "desc2"),
        ]
        report = InteractionReport(effects=effects, total_pairs=2, synergies=1, conflicts=1)
        d = report.as_dict()
        report2 = InteractionReport.from_dict(d)
        assert report2.total_pairs == 2
        assert report2.synergies == 1
        assert report2.conflicts == 1
        assert len(report2.effects) == 2
        assert report2.effects[0].metric_a == "a"
        assert report2.effects[1].interaction_type == "conflict"

    def test_empty_report_roundtrip(self):
        report = InteractionReport()
        d = report.as_dict()
        report2 = InteractionReport.from_dict(d)
        assert report2.total_pairs == 0
        assert report2.effects == []
