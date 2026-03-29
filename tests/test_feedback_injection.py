"""Tests for evalyn_sdk.feedback_injection module."""

from __future__ import annotations

import random

import pytest

from evalyn_sdk.feedback_injection import (
    FailureTarget,
    InjectedItem,
    InjectionConfig,
    InjectionReport,
    compute_coverage,
    format_injection_report,
    generate_from_failure_target,
    generate_injection_suite,
    prioritize_targets,
)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def _make_target(
    pid: str = "t1",
    severity: str = "medium",
    frequency: int = 1,
    examples: list[str] | None = None,
) -> FailureTarget:
    return FailureTarget(
        pattern_id=pid,
        description=f"Description for {pid}",
        example_inputs=examples if examples is not None else ["example one", "example two", "example three"],
        severity=severity,
        frequency=frequency,
    )


# -----------------------------------------------------------------------
# FailureTarget
# -----------------------------------------------------------------------


class TestFailureTarget:
    def test_defaults(self):
        t = FailureTarget(
            pattern_id="p", description="d", example_inputs=["a"],
        )
        assert t.severity == "medium"
        assert t.frequency == 1

    def test_custom(self):
        t = FailureTarget(
            pattern_id="p", description="d", example_inputs=["a"],
            severity="critical", frequency=5,
        )
        assert t.severity == "critical"
        assert t.frequency == 5

    def test_as_dict(self):
        t = _make_target()
        d = t.as_dict()
        assert d["pattern_id"] == "t1"
        assert isinstance(d["example_inputs"], list)

    def test_from_dict(self):
        d = {
            "pattern_id": "x", "description": "desc",
            "example_inputs": ["e1"], "severity": "high", "frequency": 3,
        }
        t = FailureTarget.from_dict(d)
        assert t.pattern_id == "x"
        assert t.severity == "high"
        assert t.frequency == 3

    def test_from_dict_defaults(self):
        d = {"pattern_id": "x", "description": "d", "example_inputs": []}
        t = FailureTarget.from_dict(d)
        assert t.severity == "medium"
        assert t.frequency == 1

    def test_roundtrip(self):
        t = _make_target(severity="critical", frequency=10)
        assert FailureTarget.from_dict(t.as_dict()) == t


# -----------------------------------------------------------------------
# InjectionConfig
# -----------------------------------------------------------------------


class TestInjectionConfig:
    def test_defaults(self):
        cfg = InjectionConfig(targets=[_make_target()])
        assert cfg.items_per_target == 3
        assert cfg.include_examples is True

    def test_as_dict(self):
        cfg = InjectionConfig(targets=[_make_target()], items_per_target=5)
        d = cfg.as_dict()
        assert d["items_per_target"] == 5
        assert len(d["targets"]) == 1

    def test_from_dict(self):
        d = {
            "targets": [_make_target().as_dict()],
            "items_per_target": 7,
            "include_examples": False,
        }
        cfg = InjectionConfig.from_dict(d)
        assert cfg.items_per_target == 7
        assert cfg.include_examples is False

    def test_roundtrip(self):
        cfg = InjectionConfig(
            targets=[_make_target("a"), _make_target("b")],
            items_per_target=2,
            include_examples=False,
        )
        assert InjectionConfig.from_dict(cfg.as_dict()) == cfg


# -----------------------------------------------------------------------
# InjectedItem
# -----------------------------------------------------------------------


class TestInjectedItem:
    def test_basic(self):
        item = InjectedItem(
            input_text="hello", target_pattern_id="p1",
            injection_method="rephrase",
        )
        assert item.tags == []

    def test_as_dict(self):
        item = InjectedItem(
            input_text="t", target_pattern_id="p",
            injection_method="m", tags=["a", "b"],
        )
        d = item.as_dict()
        assert d["tags"] == ["a", "b"]

    def test_from_dict(self):
        d = {
            "input_text": "t", "target_pattern_id": "p",
            "injection_method": "m", "tags": ["x"],
        }
        item = InjectedItem.from_dict(d)
        assert item.input_text == "t"

    def test_from_dict_no_tags(self):
        d = {
            "input_text": "t", "target_pattern_id": "p",
            "injection_method": "m",
        }
        item = InjectedItem.from_dict(d)
        assert item.tags == []

    def test_roundtrip(self):
        item = InjectedItem(
            input_text="text", target_pattern_id="pid",
            injection_method="rephrase", tags=["t1"],
        )
        assert InjectedItem.from_dict(item.as_dict()) == item


# -----------------------------------------------------------------------
# InjectionReport
# -----------------------------------------------------------------------


class TestInjectionReport:
    def test_as_dict(self):
        report = InjectionReport(
            items=[], total_items=0, patterns_covered=0,
            total_patterns=0, coverage_rate=0.0,
        )
        d = report.as_dict()
        assert d["total_items"] == 0

    def test_from_dict(self):
        d = {
            "items": [], "total_items": 0,
            "patterns_covered": 0, "total_patterns": 1,
            "coverage_rate": 0.0,
        }
        report = InjectionReport.from_dict(d)
        assert report.total_patterns == 1

    def test_roundtrip(self):
        item = InjectedItem(
            input_text="t", target_pattern_id="p",
            injection_method="m", tags=[],
        )
        report = InjectionReport(
            items=[item], total_items=1,
            patterns_covered=1, total_patterns=2,
            coverage_rate=0.5,
        )
        assert InjectionReport.from_dict(report.as_dict()) == report


# -----------------------------------------------------------------------
# generate_from_failure_target
# -----------------------------------------------------------------------


class TestGenerateFromFailureTarget:
    def test_default_count(self):
        target = _make_target()
        items = generate_from_failure_target(target)
        assert len(items) == 3

    def test_custom_count(self):
        target = _make_target()
        items = generate_from_failure_target(target, n_items=5)
        assert len(items) == 5

    def test_pattern_id_set(self):
        target = _make_target(pid="my_pattern")
        items = generate_from_failure_target(target, n_items=2)
        for item in items:
            assert item.target_pattern_id == "my_pattern"

    def test_methods_cycle(self):
        target = _make_target()
        items = generate_from_failure_target(target, n_items=6, rng=random.Random(0))
        methods = [i.injection_method for i in items]
        assert methods[0] == "rephrase"
        assert methods[1] == "combine"
        assert methods[2] == "perturbation"
        assert methods[3] == "rephrase"
        assert methods[4] == "combine"
        assert methods[5] == "perturbation"

    def test_deterministic_with_rng(self):
        target = _make_target()
        a = generate_from_failure_target(target, n_items=5, rng=random.Random(42))
        b = generate_from_failure_target(target, n_items=5, rng=random.Random(42))
        assert [i.input_text for i in a] == [i.input_text for i in b]

    def test_tags_include_pattern_and_severity(self):
        target = _make_target(pid="p1", severity="high")
        items = generate_from_failure_target(target, n_items=1, rng=random.Random(0))
        assert "p1" in items[0].tags
        assert "high" in items[0].tags

    def test_single_example_no_combine(self):
        """With only one example, combine falls back to perturbation."""
        target = _make_target(examples=["only one"])
        items = generate_from_failure_target(target, n_items=3, rng=random.Random(0))
        # index 1 would normally be combine but needs 2 examples
        assert items[1].injection_method == "perturbation"

    def test_no_examples_fallback(self):
        target = _make_target(examples=[])
        items = generate_from_failure_target(target, n_items=2, rng=random.Random(0))
        for item in items:
            assert item.injection_method == "description_fallback"
            assert target.description in item.input_text

    def test_zero_items(self):
        items = generate_from_failure_target(_make_target(), n_items=0)
        assert items == []


# -----------------------------------------------------------------------
# generate_injection_suite
# -----------------------------------------------------------------------


class TestGenerateInjectionSuite:
    def test_basic(self):
        cfg = InjectionConfig(
            targets=[_make_target("a"), _make_target("b")],
            items_per_target=2,
        )
        report = generate_injection_suite(cfg, rng=random.Random(0))
        assert report.total_items == 4
        assert report.patterns_covered == 2
        assert report.total_patterns == 2
        assert report.coverage_rate == 1.0

    def test_empty_targets(self):
        cfg = InjectionConfig(targets=[], items_per_target=3)
        report = generate_injection_suite(cfg)
        assert report.total_items == 0
        assert report.coverage_rate == 0.0

    def test_deterministic(self):
        cfg = InjectionConfig(targets=[_make_target()], items_per_target=5)
        a = generate_injection_suite(cfg, rng=random.Random(7))
        b = generate_injection_suite(cfg, rng=random.Random(7))
        assert [i.input_text for i in a.items] == [i.input_text for i in b.items]


# -----------------------------------------------------------------------
# compute_coverage
# -----------------------------------------------------------------------


class TestComputeCoverage:
    def test_full_coverage(self):
        targets = [_make_target("a"), _make_target("b")]
        items = [
            InjectedItem(input_text="", target_pattern_id="a", injection_method="m"),
            InjectedItem(input_text="", target_pattern_id="b", injection_method="m"),
        ]
        report = InjectionReport(
            items=items, total_items=2,
            patterns_covered=2, total_patterns=2, coverage_rate=1.0,
        )
        assert compute_coverage(report, targets) == 1.0

    def test_partial_coverage(self):
        targets = [_make_target("a"), _make_target("b")]
        items = [
            InjectedItem(input_text="", target_pattern_id="a", injection_method="m"),
        ]
        report = InjectionReport(
            items=items, total_items=1,
            patterns_covered=1, total_patterns=2, coverage_rate=0.5,
        )
        assert compute_coverage(report, targets) == 0.5

    def test_no_targets(self):
        report = InjectionReport(
            items=[], total_items=0,
            patterns_covered=0, total_patterns=0, coverage_rate=0.0,
        )
        assert compute_coverage(report, []) == 0.0

    def test_extra_items_ignored(self):
        """Items targeting patterns not in all_targets don't count."""
        targets = [_make_target("a")]
        items = [
            InjectedItem(input_text="", target_pattern_id="z", injection_method="m"),
        ]
        report = InjectionReport(
            items=items, total_items=1,
            patterns_covered=1, total_patterns=1, coverage_rate=1.0,
        )
        assert compute_coverage(report, targets) == 0.0


# -----------------------------------------------------------------------
# prioritize_targets
# -----------------------------------------------------------------------


class TestPrioritizeTargets:
    def test_severity_order(self):
        targets = [
            _make_target("low", severity="low"),
            _make_target("crit", severity="critical"),
            _make_target("med", severity="medium"),
            _make_target("high", severity="high"),
        ]
        result = prioritize_targets(targets)
        assert [t.pattern_id for t in result] == ["crit", "high", "med", "low"]

    def test_frequency_tiebreak(self):
        targets = [
            _make_target("a", severity="high", frequency=1),
            _make_target("b", severity="high", frequency=10),
            _make_target("c", severity="high", frequency=5),
        ]
        result = prioritize_targets(targets)
        assert [t.pattern_id for t in result] == ["b", "c", "a"]

    def test_empty(self):
        assert prioritize_targets([]) == []

    def test_single(self):
        t = _make_target("only")
        assert prioritize_targets([t]) == [t]

    def test_unknown_severity(self):
        """Unknown severity sorts after all known ones."""
        targets = [
            _make_target("low", severity="low"),
            _make_target("unk", severity="unknown"),
        ]
        result = prioritize_targets(targets)
        # unknown gets -1 -> negated to 1; low gets 0 -> negated to 0
        # ascending sort: low (0) before unknown (1)
        assert result[0].pattern_id == "low"
        assert result[1].pattern_id == "unk"


# -----------------------------------------------------------------------
# format_injection_report
# -----------------------------------------------------------------------


class TestFormatInjectionReport:
    def test_basic(self):
        item = InjectedItem(
            input_text="hello world", target_pattern_id="p1",
            injection_method="rephrase", tags=[],
        )
        report = InjectionReport(
            items=[item], total_items=1,
            patterns_covered=1, total_patterns=2, coverage_rate=0.5,
        )
        text = format_injection_report(report)
        assert "Total items: 1" in text
        assert "Patterns covered: 1/2" in text
        assert "50.0%" in text
        assert "[p1]" in text
        assert "(rephrase)" in text

    def test_empty_report(self):
        report = InjectionReport(
            items=[], total_items=0,
            patterns_covered=0, total_patterns=0, coverage_rate=0.0,
        )
        text = format_injection_report(report)
        assert "Total items: 0" in text

    def test_multiple_patterns_grouped(self):
        items = [
            InjectedItem(input_text="a", target_pattern_id="p1", injection_method="m"),
            InjectedItem(input_text="b", target_pattern_id="p2", injection_method="m"),
            InjectedItem(input_text="c", target_pattern_id="p1", injection_method="m"),
        ]
        report = InjectionReport(
            items=items, total_items=3,
            patterns_covered=2, total_patterns=2, coverage_rate=1.0,
        )
        text = format_injection_report(report)
        assert "[p1] (2 items)" in text
        assert "[p2] (1 items)" in text
