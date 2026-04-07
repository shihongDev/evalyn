"""Tests for behavior-based test case generation."""

from __future__ import annotations

import random
import sys
from pathlib import Path

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.behavior_test_gen import (
    BehaviorSpec,
    GeneratedTestCase,
    TestGenConfig,
    TestGenReport,
    generate_positive_variants,
    generate_negative_variants,
    generate_edge_cases,
    generate_test_suite,
    format_test_report,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_spec(
    behavior_id: str = "b1",
    description: str = "summarize text",
    expected_outcome: str = "A short summary",
    tags: list | None = None,
) -> BehaviorSpec:
    return BehaviorSpec(
        behavior_id=behavior_id,
        description=description,
        expected_outcome=expected_outcome,
        tags=tags or [],
    )


# ---------------------------------------------------------------------------
# BehaviorSpec
# ---------------------------------------------------------------------------


class TestBehaviorSpec:
    def test_creation(self):
        spec = _make_spec()
        assert spec.behavior_id == "b1"
        assert spec.description == "summarize text"
        assert spec.expected_outcome == "A short summary"
        assert spec.tags == []

    def test_as_dict(self):
        spec = _make_spec(tags=["nlp", "summary"])
        d = spec.as_dict()
        assert d["behavior_id"] == "b1"
        assert d["tags"] == ["nlp", "summary"]

    def test_from_dict(self):
        d = {
            "behavior_id": "b2",
            "description": "translate",
            "expected_outcome": "translated text",
            "tags": ["translation"],
        }
        spec = BehaviorSpec.from_dict(d)
        assert spec.behavior_id == "b2"
        assert spec.tags == ["translation"]

    def test_roundtrip(self):
        spec = _make_spec(tags=["a", "b"])
        assert BehaviorSpec.from_dict(spec.as_dict()).as_dict() == spec.as_dict()

    def test_from_dict_defaults(self):
        d = {
            "behavior_id": "b3",
            "description": "test",
            "expected_outcome": "result",
        }
        spec = BehaviorSpec.from_dict(d)
        assert spec.tags == []


# ---------------------------------------------------------------------------
# GeneratedTestCase
# ---------------------------------------------------------------------------


class TestGeneratedTestCase:
    def test_creation(self):
        tc = GeneratedTestCase(
            test_id="t1",
            input_text="hello",
            expected_output="world",
            behavior_id="b1",
        )
        assert tc.variant == ""
        assert tc.difficulty == "medium"

    def test_as_dict(self):
        tc = GeneratedTestCase(
            test_id="t1",
            input_text="hello",
            expected_output="world",
            behavior_id="b1",
            variant="positive",
            difficulty="easy",
        )
        d = tc.as_dict()
        assert d["test_id"] == "t1"
        assert d["variant"] == "positive"

    def test_from_dict(self):
        d = {
            "test_id": "t2",
            "input_text": "in",
            "expected_output": "out",
            "behavior_id": "b1",
        }
        tc = GeneratedTestCase.from_dict(d)
        assert tc.variant == ""
        assert tc.difficulty == "medium"

    def test_roundtrip(self):
        tc = GeneratedTestCase(
            test_id="t1",
            input_text="hello",
            expected_output="world",
            behavior_id="b1",
            variant="neg",
            difficulty="hard",
        )
        assert (
            GeneratedTestCase.from_dict(tc.as_dict()).as_dict() == tc.as_dict()
        )


# ---------------------------------------------------------------------------
# TestGenConfig
# ---------------------------------------------------------------------------


class TestTestGenConfig:
    def test_defaults(self):
        cfg = TestGenConfig()
        assert cfg.variants_per_behavior == 5
        assert cfg.include_edge_cases is True
        assert cfg.include_negatives is True
        assert cfg.seed is None

    def test_as_dict(self):
        cfg = TestGenConfig(seed=42)
        d = cfg.as_dict()
        assert d["seed"] == 42

    def test_from_dict(self):
        d = {"variants_per_behavior": 3, "include_edge_cases": False}
        cfg = TestGenConfig.from_dict(d)
        assert cfg.variants_per_behavior == 3
        assert cfg.include_edge_cases is False

    def test_roundtrip(self):
        cfg = TestGenConfig(variants_per_behavior=10, seed=7)
        assert TestGenConfig.from_dict(cfg.as_dict()).as_dict() == cfg.as_dict()


# ---------------------------------------------------------------------------
# TestGenReport
# ---------------------------------------------------------------------------


class TestTestGenReport:
    def test_empty(self):
        r = TestGenReport()
        assert r.total_tests == 0
        assert r.test_cases == []

    def test_as_dict(self):
        tc = GeneratedTestCase(
            test_id="t1",
            input_text="in",
            expected_output="out",
            behavior_id="b1",
        )
        r = TestGenReport(
            test_cases=[tc],
            total_behaviors=1,
            total_tests=1,
            coverage={"b1": 1},
        )
        d = r.as_dict()
        assert len(d["test_cases"]) == 1
        assert d["coverage"] == {"b1": 1}

    def test_from_dict(self):
        d = {
            "test_cases": [
                {
                    "test_id": "t1",
                    "input_text": "in",
                    "expected_output": "out",
                    "behavior_id": "b1",
                }
            ],
            "total_behaviors": 1,
            "total_tests": 1,
            "coverage": {"b1": 1},
        }
        r = TestGenReport.from_dict(d)
        assert len(r.test_cases) == 1
        assert r.test_cases[0].test_id == "t1"

    def test_roundtrip(self):
        tc = GeneratedTestCase(
            test_id="t1",
            input_text="in",
            expected_output="out",
            behavior_id="b1",
        )
        r = TestGenReport(
            test_cases=[tc],
            total_behaviors=1,
            total_tests=1,
            coverage={"b1": 1},
        )
        assert TestGenReport.from_dict(r.as_dict()).as_dict() == r.as_dict()


# ---------------------------------------------------------------------------
# generate_positive_variants
# ---------------------------------------------------------------------------


class TestGeneratePositiveVariants:
    def test_returns_requested_count(self):
        spec = _make_spec()
        cases = generate_positive_variants(spec, n=3)
        assert len(cases) == 3

    def test_all_positive_variant(self):
        spec = _make_spec()
        cases = generate_positive_variants(spec, n=3)
        for tc in cases:
            assert tc.variant == "positive"

    def test_contains_description(self):
        spec = _make_spec(description="summarize text")
        cases = generate_positive_variants(spec, n=3)
        for tc in cases:
            assert "summarize text" in tc.input_text

    def test_expected_output_matches(self):
        spec = _make_spec(expected_outcome="Summary here")
        cases = generate_positive_variants(spec, n=2)
        for tc in cases:
            assert tc.expected_output == "Summary here"

    def test_behavior_id_set(self):
        spec = _make_spec(behavior_id="bx")
        cases = generate_positive_variants(spec, n=1)
        assert cases[0].behavior_id == "bx"

    def test_difficulty_is_easy(self):
        spec = _make_spec()
        cases = generate_positive_variants(spec, n=2)
        for tc in cases:
            assert tc.difficulty == "easy"

    def test_unique_inputs(self):
        spec = _make_spec()
        cases = generate_positive_variants(spec, n=5)
        inputs = [tc.input_text for tc in cases]
        assert len(set(inputs)) == len(inputs)

    def test_seeded_rng_deterministic(self):
        spec = _make_spec()
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        cases1 = generate_positive_variants(spec, n=3, rng=rng1)
        cases2 = generate_positive_variants(spec, n=3, rng=rng2)
        for a, b in zip(cases1, cases2):
            assert a.input_text == b.input_text

    def test_n_zero(self):
        spec = _make_spec()
        cases = generate_positive_variants(spec, n=0)
        assert cases == []

    def test_n_exceeds_templates(self):
        """Requesting more than available templates caps at template count."""
        spec = _make_spec()
        cases = generate_positive_variants(spec, n=100)
        # Should get at most the number of templates
        assert len(cases) <= 100
        assert len(cases) > 0


# ---------------------------------------------------------------------------
# generate_negative_variants
# ---------------------------------------------------------------------------


class TestGenerateNegativeVariants:
    def test_returns_requested_count(self):
        spec = _make_spec()
        cases = generate_negative_variants(spec, n=2)
        assert len(cases) == 2

    def test_all_negative_variant(self):
        spec = _make_spec()
        cases = generate_negative_variants(spec, n=2)
        for tc in cases:
            assert tc.variant == "negative"

    def test_difficulty_is_hard(self):
        spec = _make_spec()
        cases = generate_negative_variants(spec, n=2)
        for tc in cases:
            assert tc.difficulty == "hard"

    def test_expected_output_empty(self):
        spec = _make_spec()
        cases = generate_negative_variants(spec, n=2)
        for tc in cases:
            assert tc.expected_output == ""

    def test_contains_description_reference(self):
        spec = _make_spec(description="translate code")
        cases = generate_negative_variants(spec, n=2)
        for tc in cases:
            assert "translate code" in tc.input_text

    def test_n_zero(self):
        spec = _make_spec()
        cases = generate_negative_variants(spec, n=0)
        assert cases == []


# ---------------------------------------------------------------------------
# generate_edge_cases
# ---------------------------------------------------------------------------


class TestGenerateEdgeCases:
    def test_returns_three_cases(self):
        spec = _make_spec()
        cases = generate_edge_cases(spec)
        assert len(cases) == 3

    def test_empty_input_case(self):
        spec = _make_spec()
        cases = generate_edge_cases(spec)
        empty = [tc for tc in cases if tc.variant == "edge_empty"]
        assert len(empty) == 1
        assert empty[0].input_text == ""

    def test_long_input_case(self):
        spec = _make_spec(description="summarize text")
        cases = generate_edge_cases(spec)
        long_cases = [tc for tc in cases if tc.variant == "edge_long"]
        assert len(long_cases) == 1
        assert len(long_cases[0].input_text) > 200

    def test_special_chars_case(self):
        spec = _make_spec()
        # Use a seeded RNG for deterministic results
        import random
        rng = random.Random(42)
        cases = generate_edge_cases(spec, rng=rng)
        special = [tc for tc in cases if tc.variant == "edge_special"]
        assert len(special) == 1
        # Should contain some non-alphanumeric characters from _SPECIAL_CHARS
        text = special[0].input_text
        assert any(not c.isalnum() and not c.isspace() for c in text)

    def test_all_hard_difficulty(self):
        spec = _make_spec()
        cases = generate_edge_cases(spec)
        for tc in cases:
            assert tc.difficulty == "hard"

    def test_behavior_id_propagated(self):
        spec = _make_spec(behavior_id="edge_test")
        cases = generate_edge_cases(spec)
        for tc in cases:
            assert tc.behavior_id == "edge_test"


# ---------------------------------------------------------------------------
# generate_test_suite
# ---------------------------------------------------------------------------


class TestGenerateTestSuite:
    def test_single_behavior_default_config(self):
        spec = _make_spec()
        report = generate_test_suite([spec])
        assert report.total_behaviors == 1
        assert report.total_tests > 0
        assert "b1" in report.coverage

    def test_multiple_behaviors(self):
        specs = [
            _make_spec(behavior_id="b1"),
            _make_spec(behavior_id="b2"),
        ]
        report = generate_test_suite(specs)
        assert report.total_behaviors == 2
        assert "b1" in report.coverage
        assert "b2" in report.coverage

    def test_no_edge_cases(self):
        spec = _make_spec()
        config = TestGenConfig(
            variants_per_behavior=2,
            include_edge_cases=False,
            include_negatives=True,
        )
        report = generate_test_suite([spec], config)
        variants = {tc.variant for tc in report.test_cases}
        assert "edge_empty" not in variants
        assert "edge_long" not in variants
        assert "edge_special" not in variants

    def test_no_negatives(self):
        spec = _make_spec()
        config = TestGenConfig(
            variants_per_behavior=2,
            include_edge_cases=True,
            include_negatives=False,
        )
        report = generate_test_suite([spec], config)
        variants = {tc.variant for tc in report.test_cases}
        assert "negative" not in variants

    def test_seeded_deterministic(self):
        specs = [_make_spec(), _make_spec(behavior_id="b2")]
        config1 = TestGenConfig(seed=99)
        config2 = TestGenConfig(seed=99)
        r1 = generate_test_suite(specs, config1)
        r2 = generate_test_suite(specs, config2)
        for a, b in zip(r1.test_cases, r2.test_cases):
            assert a.input_text == b.input_text

    def test_coverage_counts(self):
        spec = _make_spec()
        config = TestGenConfig(
            variants_per_behavior=3,
            include_edge_cases=True,
            include_negatives=True,
        )
        report = generate_test_suite([spec], config)
        assert report.coverage["b1"] == report.total_tests

    def test_empty_behaviors_list(self):
        report = generate_test_suite([])
        assert report.total_behaviors == 0
        assert report.total_tests == 0
        assert report.test_cases == []

    def test_total_tests_matches_cases(self):
        specs = [_make_spec(behavior_id=f"b{i}") for i in range(3)]
        report = generate_test_suite(specs)
        assert report.total_tests == len(report.test_cases)

    def test_coverage_sums_to_total(self):
        specs = [_make_spec(behavior_id=f"b{i}") for i in range(3)]
        report = generate_test_suite(specs)
        assert sum(report.coverage.values()) == report.total_tests


# ---------------------------------------------------------------------------
# format_test_report
# ---------------------------------------------------------------------------


class TestFormatTestReport:
    def test_contains_header(self):
        report = TestGenReport(total_behaviors=1, total_tests=5)
        text = format_test_report(report)
        assert "Test Generation Report" in text

    def test_contains_counts(self):
        report = TestGenReport(total_behaviors=2, total_tests=10)
        text = format_test_report(report)
        assert "Total behaviors: 2" in text
        assert "Total tests: 10" in text

    def test_contains_coverage(self):
        report = TestGenReport(
            total_behaviors=1,
            total_tests=3,
            coverage={"b1": 3},
        )
        text = format_test_report(report)
        assert "b1: 3 tests" in text

    def test_truncates_long_input(self):
        tc = GeneratedTestCase(
            test_id="t1",
            input_text="x" * 100,
            expected_output="out",
            behavior_id="b1",
        )
        report = TestGenReport(test_cases=[tc], total_tests=1)
        text = format_test_report(report)
        assert "..." in text

    def test_empty_report(self):
        report = TestGenReport()
        text = format_test_report(report)
        assert "Total tests: 0" in text
