"""Tests for the score binning module."""

from __future__ import annotations

import sys
from pathlib import Path

# Add SDK to path
SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.score_binning import (
    BinnedScore,
    BinningConfig,
    DEFAULT_BINNING,
    GradeDistribution,
    GradeMapping,
    LETTER_GRADE_BINNING,
    PASS_FAIL_BINNING,
    bin_scores,
    compute_grade_distribution,
    create_custom_binning,
    format_grade_report,
    score_to_grade,
)


# ---------------------------------------------------------------------------
# GradeMapping
# ---------------------------------------------------------------------------


class TestGradeMapping:
    def test_basic_fields(self):
        m = GradeMapping(grade="A", min_score=0.9, max_score=1.0)
        assert m.grade == "A"
        assert m.min_score == 0.9
        assert m.max_score == 1.0
        assert m.label == ""
        assert m.color_hint == ""

    def test_with_label_and_color(self):
        m = GradeMapping(grade="A", min_score=0.9, max_score=1.0, label="Excellent", color_hint="green")
        assert m.label == "Excellent"
        assert m.color_hint == "green"

    def test_as_dict(self):
        m = GradeMapping(grade="B", min_score=0.7, max_score=0.9, label="Good")
        d = m.as_dict()
        assert d["grade"] == "B"
        assert d["min_score"] == 0.7
        assert d["max_score"] == 0.9
        assert d["label"] == "Good"
        assert d["color_hint"] == ""

    def test_from_dict(self):
        d = {"grade": "C", "min_score": 0.5, "max_score": 0.7, "label": "Fair", "color_hint": "yellow"}
        m = GradeMapping.from_dict(d)
        assert m.grade == "C"
        assert m.label == "Fair"
        assert m.color_hint == "yellow"

    def test_from_dict_defaults(self):
        d = {"grade": "D", "min_score": 0.3, "max_score": 0.5}
        m = GradeMapping.from_dict(d)
        assert m.label == ""
        assert m.color_hint == ""

    def test_roundtrip(self):
        original = GradeMapping(grade="A", min_score=0.9, max_score=1.0, label="Top", color_hint="#00ff00")
        restored = GradeMapping.from_dict(original.as_dict())
        assert restored == original


# ---------------------------------------------------------------------------
# BinningConfig
# ---------------------------------------------------------------------------


class TestBinningConfig:
    def test_empty_config(self):
        cfg = BinningConfig()
        assert cfg.mappings == []
        assert cfg.default_grade == "F"

    def test_custom_default_grade(self):
        cfg = BinningConfig(default_grade="N/A")
        assert cfg.default_grade == "N/A"

    def test_as_dict(self):
        cfg = BinningConfig(
            mappings=[GradeMapping(grade="A", min_score=0.9, max_score=1.0)],
            default_grade="X",
        )
        d = cfg.as_dict()
        assert len(d["mappings"]) == 1
        assert d["default_grade"] == "X"

    def test_from_dict(self):
        d = {
            "mappings": [{"grade": "A", "min_score": 0.9, "max_score": 1.0}],
            "default_grade": "Z",
        }
        cfg = BinningConfig.from_dict(d)
        assert len(cfg.mappings) == 1
        assert cfg.default_grade == "Z"

    def test_from_dict_defaults(self):
        cfg = BinningConfig.from_dict({})
        assert cfg.mappings == []
        assert cfg.default_grade == "F"

    def test_roundtrip(self):
        original = DEFAULT_BINNING
        restored = BinningConfig.from_dict(original.as_dict())
        assert len(restored.mappings) == len(original.mappings)
        assert restored.default_grade == original.default_grade


# ---------------------------------------------------------------------------
# BinnedScore
# ---------------------------------------------------------------------------


class TestBinnedScore:
    def test_basic_fields(self):
        b = BinnedScore(item_id="x", score=0.85, grade="B", label="Good")
        assert b.item_id == "x"
        assert b.score == 0.85
        assert b.grade == "B"
        assert b.label == "Good"

    def test_as_dict(self):
        b = BinnedScore(item_id="y", score=0.4, grade="D", label="")
        d = b.as_dict()
        assert d["item_id"] == "y"
        assert d["score"] == 0.4
        assert d["grade"] == "D"

    def test_from_dict(self):
        d = {"item_id": "z", "score": 0.95, "grade": "A", "label": "Excellent"}
        b = BinnedScore.from_dict(d)
        assert b.item_id == "z"
        assert b.label == "Excellent"

    def test_from_dict_default_label(self):
        d = {"item_id": "w", "score": 0.1, "grade": "F"}
        b = BinnedScore.from_dict(d)
        assert b.label == ""

    def test_roundtrip(self):
        original = BinnedScore(item_id="abc", score=0.77, grade="B", label="Good")
        restored = BinnedScore.from_dict(original.as_dict())
        assert restored == original


# ---------------------------------------------------------------------------
# GradeDistribution
# ---------------------------------------------------------------------------


class TestGradeDistribution:
    def test_empty(self):
        gd = GradeDistribution()
        assert gd.grade_counts == {}
        assert gd.total == 0
        assert gd.most_common == ""

    def test_as_dict(self):
        gd = GradeDistribution(grade_counts={"A": 3, "B": 2}, total=5, most_common="A")
        d = gd.as_dict()
        assert d["total"] == 5
        assert d["most_common"] == "A"
        assert d["grade_counts"]["A"] == 3

    def test_from_dict(self):
        d = {"grade_counts": {"C": 10}, "total": 10, "most_common": "C"}
        gd = GradeDistribution.from_dict(d)
        assert gd.grade_counts == {"C": 10}
        assert gd.total == 10

    def test_from_dict_defaults(self):
        gd = GradeDistribution.from_dict({})
        assert gd.grade_counts == {}
        assert gd.total == 0

    def test_roundtrip(self):
        original = GradeDistribution(grade_counts={"A": 1, "F": 2}, total=3, most_common="F")
        restored = GradeDistribution.from_dict(original.as_dict())
        assert restored == original


# ---------------------------------------------------------------------------
# Built-in Configurations
# ---------------------------------------------------------------------------


class TestBuiltinConfigs:
    def test_default_binning_has_five_grades(self):
        assert len(DEFAULT_BINNING.mappings) == 5

    def test_default_binning_grades(self):
        grades = [m.grade for m in DEFAULT_BINNING.mappings]
        assert grades == ["A", "B", "C", "D", "F"]

    def test_letter_grade_binning_labels(self):
        labels = [m.label for m in LETTER_GRADE_BINNING.mappings]
        assert labels == ["Excellent", "Good", "Fair", "Poor", "Failing"]

    def test_pass_fail_binning_has_two_grades(self):
        assert len(PASS_FAIL_BINNING.mappings) == 2
        grades = [m.grade for m in PASS_FAIL_BINNING.mappings]
        assert "Pass" in grades
        assert "Fail" in grades


# ---------------------------------------------------------------------------
# score_to_grade
# ---------------------------------------------------------------------------


class TestScoreToGrade:
    def test_high_score_is_a(self):
        assert score_to_grade(0.95, DEFAULT_BINNING) == "A"

    def test_boundary_0_9_is_a(self):
        assert score_to_grade(0.9, DEFAULT_BINNING) == "A"

    def test_score_0_85_is_b(self):
        assert score_to_grade(0.85, DEFAULT_BINNING) == "B"

    def test_score_0_5_is_c(self):
        assert score_to_grade(0.5, DEFAULT_BINNING) == "C"

    def test_score_0_3_is_d(self):
        assert score_to_grade(0.3, DEFAULT_BINNING) == "D"

    def test_score_0_1_is_f(self):
        assert score_to_grade(0.1, DEFAULT_BINNING) == "F"

    def test_perfect_score_is_a(self):
        assert score_to_grade(1.0, DEFAULT_BINNING) == "A"

    def test_zero_score_is_f(self):
        assert score_to_grade(0.0, DEFAULT_BINNING) == "F"

    def test_pass_fail_pass(self):
        assert score_to_grade(0.75, PASS_FAIL_BINNING) == "Pass"

    def test_pass_fail_fail(self):
        assert score_to_grade(0.25, PASS_FAIL_BINNING) == "Fail"

    def test_default_grade_for_no_match(self):
        cfg = BinningConfig(
            mappings=[GradeMapping(grade="X", min_score=0.5, max_score=0.6)],
            default_grade="NONE",
        )
        assert score_to_grade(0.9, cfg) == "NONE"

    def test_empty_config_returns_default(self):
        cfg = BinningConfig(mappings=[], default_grade="Z")
        assert score_to_grade(0.5, cfg) == "Z"


# ---------------------------------------------------------------------------
# bin_scores
# ---------------------------------------------------------------------------


class TestBinScores:
    def test_basic_binning(self):
        scores = {"a": 0.95, "b": 0.55, "c": 0.1}
        results = bin_scores(scores)
        grades = {r.item_id: r.grade for r in results}
        assert grades["a"] == "A"
        assert grades["c"] == "F"

    def test_uses_default_config_when_none(self):
        scores = {"x": 0.75}
        results = bin_scores(scores, config=None)
        assert results[0].grade == "B"

    def test_custom_config(self):
        cfg = PASS_FAIL_BINNING
        scores = {"a": 0.8, "b": 0.2}
        results = bin_scores(scores, config=cfg)
        grades = {r.item_id: r.grade for r in results}
        assert grades["a"] == "Pass"
        assert grades["b"] == "Fail"

    def test_labels_populated(self):
        scores = {"a": 0.95}
        results = bin_scores(scores, config=LETTER_GRADE_BINNING)
        assert results[0].label == "Excellent"

    def test_empty_scores(self):
        results = bin_scores({})
        assert results == []

    def test_all_items_present(self):
        scores = {f"item_{i}": i / 10 for i in range(11)}
        results = bin_scores(scores)
        assert len(results) == 11


# ---------------------------------------------------------------------------
# compute_grade_distribution
# ---------------------------------------------------------------------------


class TestComputeGradeDistribution:
    def test_basic_distribution(self):
        binned = [
            BinnedScore(item_id="a", score=0.95, grade="A", label=""),
            BinnedScore(item_id="b", score=0.85, grade="B", label=""),
            BinnedScore(item_id="c", score=0.92, grade="A", label=""),
        ]
        dist = compute_grade_distribution(binned)
        assert dist.total == 3
        assert dist.grade_counts["A"] == 2
        assert dist.grade_counts["B"] == 1
        assert dist.most_common == "A"

    def test_single_item(self):
        binned = [BinnedScore(item_id="x", score=0.5, grade="C", label="")]
        dist = compute_grade_distribution(binned)
        assert dist.total == 1
        assert dist.most_common == "C"

    def test_empty_list(self):
        dist = compute_grade_distribution([])
        assert dist.total == 0
        assert dist.most_common == ""
        assert dist.grade_counts == {}

    def test_all_same_grade(self):
        binned = [
            BinnedScore(item_id=f"i{i}", score=0.1, grade="F", label="")
            for i in range(5)
        ]
        dist = compute_grade_distribution(binned)
        assert dist.most_common == "F"
        assert dist.grade_counts == {"F": 5}


# ---------------------------------------------------------------------------
# format_grade_report
# ---------------------------------------------------------------------------


class TestFormatGradeReport:
    def test_contains_distribution_header(self):
        binned = [BinnedScore(item_id="a", score=0.95, grade="A", label="")]
        dist = compute_grade_distribution(binned)
        report = format_grade_report(binned, dist)
        assert "Grade Distribution" in report

    def test_contains_per_item_header(self):
        binned = [BinnedScore(item_id="a", score=0.95, grade="A", label="")]
        dist = compute_grade_distribution(binned)
        report = format_grade_report(binned, dist)
        assert "Per-Item Grades" in report

    def test_contains_item_id(self):
        binned = [BinnedScore(item_id="my_item", score=0.5, grade="C", label="")]
        dist = compute_grade_distribution(binned)
        report = format_grade_report(binned, dist)
        assert "my_item" in report

    def test_label_in_report(self):
        binned = [BinnedScore(item_id="a", score=0.95, grade="A", label="Excellent")]
        dist = compute_grade_distribution(binned)
        report = format_grade_report(binned, dist)
        assert "(Excellent)" in report

    def test_no_label_no_parens(self):
        binned = [BinnedScore(item_id="a", score=0.5, grade="C", label="")]
        dist = compute_grade_distribution(binned)
        report = format_grade_report(binned, dist)
        # Should not have empty parens
        assert "()" not in report

    def test_total_shown(self):
        binned = [
            BinnedScore(item_id="a", score=0.95, grade="A", label=""),
            BinnedScore(item_id="b", score=0.45, grade="D", label=""),
        ]
        dist = compute_grade_distribution(binned)
        report = format_grade_report(binned, dist)
        assert "Total: 2" in report

    def test_percentage_shown(self):
        binned = [
            BinnedScore(item_id="a", score=0.95, grade="A", label=""),
            BinnedScore(item_id="b", score=0.92, grade="A", label=""),
        ]
        dist = compute_grade_distribution(binned)
        report = format_grade_report(binned, dist)
        assert "100.0%" in report


# ---------------------------------------------------------------------------
# create_custom_binning
# ---------------------------------------------------------------------------


class TestCreateCustomBinning:
    def test_creates_config(self):
        specs = [("High", 0.8, 1.0), ("Low", 0.0, 0.8)]
        cfg = create_custom_binning(specs)
        assert len(cfg.mappings) == 2
        assert cfg.mappings[0].grade == "High"
        assert cfg.mappings[1].grade == "Low"

    def test_empty_specs(self):
        cfg = create_custom_binning([])
        assert cfg.mappings == []

    def test_single_spec(self):
        cfg = create_custom_binning([("Only", 0.0, 1.0)])
        assert len(cfg.mappings) == 1

    def test_custom_config_works_with_score_to_grade(self):
        cfg = create_custom_binning([("Good", 0.5, 1.0), ("Bad", 0.0, 0.5)])
        assert score_to_grade(0.75, cfg) == "Good"
        assert score_to_grade(0.25, cfg) == "Bad"
