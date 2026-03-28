"""Tests for rubric optimization.

All tests use synthetic data - no storage or ML deps required.
"""

from __future__ import annotations

import pytest

from evalyn_sdk.evaluation.rubric_optimization import (
    Rubric,
    RubricCriterion,
    RubricOptimizationResult,
    RubricVariant,
    compare_rubric_variants,
    create_rubric,
    generate_rubric_variant,
    score_rubric_clarity,
    suggest_rubric_improvements,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample_criteria() -> list:
    return [
        {"name": "accuracy", "description": "Response is factually correct and well-supported", "weight": 2.0},
        {"name": "clarity", "description": "Response is clear and easy to understand", "weight": 1.5},
        {"name": "relevance", "description": "Response directly addresses the question asked", "weight": 1.0},
    ]


def _sample_rubric() -> Rubric:
    return create_rubric("helpfulness", _sample_criteria(), preamble="Evaluate the response")


# ---------------------------------------------------------------------------
# RubricCriterion
# ---------------------------------------------------------------------------


class TestRubricCriterion:
    def test_as_dict(self):
        c = RubricCriterion(name="acc", description="Is it accurate", weight=2.0, examples=["ex1"])
        d = c.as_dict()
        assert d["name"] == "acc"
        assert d["weight"] == 2.0
        assert d["examples"] == ["ex1"]

    def test_from_dict(self):
        d = {"name": "clarity", "description": "Is it clear", "weight": 1.5, "examples": ["a", "b"]}
        c = RubricCriterion.from_dict(d)
        assert c.name == "clarity"
        assert c.weight == 1.5
        assert len(c.examples) == 2

    def test_from_dict_defaults(self):
        d = {"name": "test"}
        c = RubricCriterion.from_dict(d)
        assert c.description == ""
        assert c.weight == 1.0
        assert c.examples == []

    def test_roundtrip(self):
        c = RubricCriterion(name="x", description="desc", weight=3.0, examples=["e1"])
        restored = RubricCriterion.from_dict(c.as_dict())
        assert restored.name == c.name
        assert restored.weight == c.weight
        assert restored.examples == c.examples


# ---------------------------------------------------------------------------
# Rubric
# ---------------------------------------------------------------------------


class TestRubric:
    def test_as_dict(self):
        rubric = _sample_rubric()
        d = rubric.as_dict()
        assert d["metric_id"] == "helpfulness"
        assert len(d["criteria"]) == 3
        assert d["preamble"] == "Evaluate the response"

    def test_from_dict(self):
        rubric = _sample_rubric()
        restored = Rubric.from_dict(rubric.as_dict())
        assert restored.metric_id == "helpfulness"
        assert len(restored.criteria) == 3
        assert restored.version == 1

    def test_roundtrip(self):
        rubric = _sample_rubric()
        d = rubric.as_dict()
        restored = Rubric.from_dict(d)
        assert restored.as_dict() == d

    def test_format_text_header(self):
        rubric = _sample_rubric()
        text = rubric.format_text()
        assert "Rubric: helpfulness" in text
        assert "v1" in text
        assert "Preamble:" in text

    def test_format_text_criteria(self):
        rubric = _sample_rubric()
        text = rubric.format_text()
        assert "accuracy" in text
        assert "clarity" in text
        assert "relevance" in text

    def test_format_text_no_preamble(self):
        rubric = create_rubric("test", [{"name": "a", "description": "desc"}])
        text = rubric.format_text()
        assert "Preamble:" not in text

    def test_format_text_with_examples(self):
        rubric = create_rubric("test", [
            {"name": "a", "description": "desc", "examples": ["example one"]},
        ])
        text = rubric.format_text()
        assert "- example one" in text

    def test_empty_criteria(self):
        rubric = Rubric(metric_id="empty")
        assert len(rubric.criteria) == 0
        text = rubric.format_text()
        assert "Criteria: 0" in text


# ---------------------------------------------------------------------------
# RubricVariant
# ---------------------------------------------------------------------------


class TestRubricVariant:
    def test_as_dict(self):
        rubric = _sample_rubric()
        v = RubricVariant(variant_id="v1", rubric=rubric, alignment_score=0.85)
        d = v.as_dict()
        assert d["variant_id"] == "v1"
        assert d["alignment_score"] == 0.85
        assert d["rubric"]["metric_id"] == "helpfulness"


# ---------------------------------------------------------------------------
# RubricOptimizationResult
# ---------------------------------------------------------------------------


class TestRubricOptimizationResult:
    def test_as_dict(self):
        rubric = _sample_rubric()
        v = RubricVariant(variant_id="v1", rubric=rubric, alignment_score=0.9)
        result = RubricOptimizationResult(
            original=rubric, variants=[v],
            best_variant_id="v1", improvement=0.1,
        )
        d = result.as_dict()
        assert d["best_variant_id"] == "v1"
        assert d["improvement"] == 0.1

    def test_from_dict(self):
        rubric = _sample_rubric()
        v = RubricVariant(variant_id="v1", rubric=rubric, alignment_score=0.9)
        result = RubricOptimizationResult(
            original=rubric, variants=[v],
            best_variant_id="v1", improvement=0.1,
        )
        restored = RubricOptimizationResult.from_dict(result.as_dict())
        assert restored.best_variant_id == "v1"
        assert len(restored.variants) == 1
        assert restored.original.metric_id == "helpfulness"

    def test_roundtrip(self):
        rubric = _sample_rubric()
        v = RubricVariant(variant_id="v1", rubric=rubric, alignment_score=0.9)
        result = RubricOptimizationResult(
            original=rubric, variants=[v],
            best_variant_id="v1", improvement=0.1,
        )
        d = result.as_dict()
        restored = RubricOptimizationResult.from_dict(d)
        assert restored.as_dict() == d

    def test_format_text(self):
        rubric = _sample_rubric()
        v = RubricVariant(variant_id="v1", rubric=rubric, alignment_score=0.9)
        result = RubricOptimizationResult(
            original=rubric, variants=[v],
            best_variant_id="v1", improvement=0.1,
        )
        text = result.format_text()
        assert "Rubric Optimization Result" in text
        assert "v1" in text
        assert "Improvement:" in text


# ---------------------------------------------------------------------------
# create_rubric
# ---------------------------------------------------------------------------


class TestCreateRubric:
    def test_from_criteria_dicts(self):
        rubric = create_rubric("test", _sample_criteria())
        assert rubric.metric_id == "test"
        assert len(rubric.criteria) == 3
        assert rubric.criteria[0].name == "accuracy"
        assert rubric.criteria[0].weight == 2.0

    def test_with_preamble(self):
        rubric = create_rubric("test", _sample_criteria(), preamble="Be thorough")
        assert rubric.preamble == "Be thorough"

    def test_empty_criteria(self):
        rubric = create_rubric("empty", [])
        assert len(rubric.criteria) == 0
        assert rubric.version == 1

    def test_single_criterion(self):
        rubric = create_rubric("single", [{"name": "only", "description": "the only one"}])
        assert len(rubric.criteria) == 1
        assert rubric.criteria[0].name == "only"


# ---------------------------------------------------------------------------
# generate_rubric_variant
# ---------------------------------------------------------------------------


class TestGenerateRubricVariant:
    def test_reorder_changes_order(self):
        """Reorder should produce a rubric with the same criteria but potentially different order."""
        rubric = _sample_rubric()
        # Run multiple times to increase chance of seeing a different order
        names_sets = set()
        for _ in range(20):
            variant = generate_rubric_variant(rubric, method="reorder")
            order = tuple(c.name for c in variant.criteria)
            names_sets.add(order)
        # Should have same criteria names regardless of order
        variant = generate_rubric_variant(rubric, method="reorder")
        assert sorted(c.name for c in variant.criteria) == sorted(c.name for c in rubric.criteria)
        assert variant.version == rubric.version + 1

    def test_simplify_removes_lowest(self):
        rubric = _sample_rubric()
        variant = generate_rubric_variant(rubric, method="simplify")
        # Original has weights 2.0, 1.5, 1.0 - should remove weight=1.0 (relevance)
        assert len(variant.criteria) == 2
        names = [c.name for c in variant.criteria]
        assert "relevance" not in names
        assert "accuracy" in names
        assert "clarity" in names

    def test_simplify_single_criterion(self):
        rubric = create_rubric("test", [{"name": "only", "description": "the only one"}])
        variant = generate_rubric_variant(rubric, method="simplify")
        # Should not remove when only one criterion
        assert len(variant.criteria) == 1

    def test_expand_adds_examples(self):
        rubric = _sample_rubric()
        variant = generate_rubric_variant(rubric, method="expand")
        for c in variant.criteria:
            assert len(c.examples) >= 1
            assert any("Example for" in ex for ex in c.examples)

    def test_expand_preserves_existing_examples(self):
        rubric = create_rubric("test", [
            {"name": "a", "description": "desc", "examples": ["existing"]},
        ])
        variant = generate_rubric_variant(rubric, method="expand")
        assert "existing" in variant.criteria[0].examples
        assert len(variant.criteria[0].examples) == 2

    def test_unknown_method_returns_copy(self):
        rubric = _sample_rubric()
        variant = generate_rubric_variant(rubric, method="unknown")
        assert len(variant.criteria) == len(rubric.criteria)
        assert variant.version == rubric.version + 1

    def test_version_incremented(self):
        rubric = _sample_rubric()
        variant = generate_rubric_variant(rubric, method="reorder")
        assert variant.version == 2

    def test_does_not_mutate_original(self):
        rubric = _sample_rubric()
        original_names = [c.name for c in rubric.criteria]
        generate_rubric_variant(rubric, method="simplify")
        assert [c.name for c in rubric.criteria] == original_names

    def test_empty_criteria_simplify(self):
        rubric = create_rubric("test", [])
        variant = generate_rubric_variant(rubric, method="simplify")
        assert len(variant.criteria) == 0

    def test_empty_criteria_expand(self):
        rubric = create_rubric("test", [])
        variant = generate_rubric_variant(rubric, method="expand")
        assert len(variant.criteria) == 0


# ---------------------------------------------------------------------------
# score_rubric_clarity
# ---------------------------------------------------------------------------


class TestScoreRubricClarity:
    def test_long_descriptions(self):
        criteria = [
            {"name": "a", "description": "x" * 200},
            {"name": "b", "description": "y" * 300},
        ]
        rubric = create_rubric("test", criteria)
        score = score_rubric_clarity(rubric)
        assert score == 1.0

    def test_short_descriptions(self):
        criteria = [
            {"name": "a", "description": "ok"},
            {"name": "b", "description": "fine"},
        ]
        rubric = create_rubric("test", criteria)
        score = score_rubric_clarity(rubric)
        assert 0.0 < score < 0.1

    def test_empty_criteria(self):
        rubric = create_rubric("test", [])
        assert score_rubric_clarity(rubric) == 0.0

    def test_moderate_descriptions(self):
        criteria = [
            {"name": "a", "description": "x" * 100},
        ]
        rubric = create_rubric("test", criteria)
        score = score_rubric_clarity(rubric)
        assert abs(score - 0.5) < 1e-9

    def test_mixed_descriptions(self):
        criteria = [
            {"name": "a", "description": "x" * 200},
            {"name": "b", "description": ""},
        ]
        rubric = create_rubric("test", criteria)
        score = score_rubric_clarity(rubric)
        assert abs(score - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# compare_rubric_variants
# ---------------------------------------------------------------------------


class TestCompareRubricVariants:
    def test_finds_best(self):
        rubric = _sample_rubric()
        v1 = RubricVariant(variant_id="v1", rubric=rubric, alignment_score=0.5)
        v2 = RubricVariant(variant_id="v2", rubric=rubric, alignment_score=0.9)
        v3 = RubricVariant(variant_id="v3", rubric=rubric, alignment_score=0.7)
        result = compare_rubric_variants([v1, v2, v3])
        assert result.best_variant_id == "v2"
        assert result.improvement == pytest.approx(0.4, abs=1e-9)

    def test_single_variant(self):
        rubric = _sample_rubric()
        v1 = RubricVariant(variant_id="v1", rubric=rubric, alignment_score=0.8)
        result = compare_rubric_variants([v1])
        assert result.best_variant_id == "v1"
        assert result.improvement == 0.0

    def test_empty_variants(self):
        result = compare_rubric_variants([])
        assert result.best_variant_id == ""
        assert result.original.metric_id == "unknown"

    def test_original_from_first(self):
        rubric = _sample_rubric()
        v1 = RubricVariant(variant_id="v1", rubric=rubric, alignment_score=0.5)
        result = compare_rubric_variants([v1])
        assert result.original.metric_id == "helpfulness"

    def test_negative_improvement(self):
        """If best is the first variant, improvement is 0."""
        rubric = _sample_rubric()
        v1 = RubricVariant(variant_id="v1", rubric=rubric, alignment_score=0.9)
        v2 = RubricVariant(variant_id="v2", rubric=rubric, alignment_score=0.5)
        result = compare_rubric_variants([v1, v2])
        assert result.best_variant_id == "v1"
        assert result.improvement == 0.0


# ---------------------------------------------------------------------------
# suggest_rubric_improvements
# ---------------------------------------------------------------------------


class TestSuggestRubricImprovements:
    def test_suggests_examples(self):
        rubric = _sample_rubric()
        suggestions = suggest_rubric_improvements(rubric)
        example_suggestions = [s for s in suggestions if "Add examples" in s]
        assert len(example_suggestions) == 3  # all three criteria lack examples

    def test_suggests_short_description(self):
        rubric = create_rubric("test", [{"name": "brief", "description": "short"}])
        suggestions = suggest_rubric_improvements(rubric)
        short_suggestions = [s for s in suggestions if "too short" in s]
        assert len(short_suggestions) == 1

    def test_no_suggestions_for_good_rubric(self):
        criteria = [
            {"name": "a", "description": "A very detailed description of criterion a", "examples": ["ex1"]},
        ]
        rubric = create_rubric("test", criteria)
        suggestions = suggest_rubric_improvements(rubric)
        assert len(suggestions) == 0

    def test_empty_criteria(self):
        rubric = create_rubric("test", [])
        suggestions = suggest_rubric_improvements(rubric)
        assert any("no criteria" in s for s in suggestions)

    def test_mixed_criteria(self):
        criteria = [
            {"name": "good", "description": "A sufficiently long description", "examples": ["ex"]},
            {"name": "bad", "description": "short"},
        ]
        rubric = create_rubric("test", criteria)
        suggestions = suggest_rubric_improvements(rubric)
        assert any("bad" in s for s in suggestions)
        assert not any("good" in s and "too short" in s for s in suggestions)
