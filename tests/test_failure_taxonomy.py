"""Tests for the failure taxonomy module."""

from __future__ import annotations

import sys
from pathlib import Path

# Add SDK to path
SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.analysis.failure_taxonomy import (
    FAILURE_CATEGORIES,
    CategorizedFailure,
    FailureCategory,
    TaxonomyReport,
    build_taxonomy,
    classify_failure,
    get_category_distribution,
    render_taxonomy_tree,
    suggest_category_fixes,
)


# ---------------------------------------------------------------------------
# classify_failure
# ---------------------------------------------------------------------------


class TestClassifyFailure:
    def test_empty_output_none(self):
        """None output -> model (refusal)."""
        item = {"item_id": "i1", "output": None, "error": "", "metric_id": "acc"}
        result = classify_failure(item)
        assert result.category == "model"
        assert result.subcategory == "refusal"

    def test_empty_output_empty_string(self):
        """Empty string output -> model (refusal)."""
        item = {"item_id": "i2", "output": "", "error": "", "metric_id": "acc"}
        result = classify_failure(item)
        assert result.category == "model"

    def test_empty_output_whitespace(self):
        """Whitespace-only output -> model (refusal)."""
        item = {"item_id": "i3", "output": "   ", "error": "", "metric_id": "acc"}
        result = classify_failure(item)
        assert result.category == "model"

    def test_tool_error_in_error_field(self):
        """Error containing 'tool' -> tool category."""
        item = {"item_id": "i4", "output": "some output", "error": "Tool execution failed", "metric_id": "acc"}
        result = classify_failure(item)
        assert result.category == "tool"
        assert result.subcategory == "tool_error"

    def test_tool_error_case_insensitive(self):
        """Tool detection is case-insensitive."""
        item = {"item_id": "i5", "output": "some output", "error": "TOOL timeout", "metric_id": "acc"}
        result = classify_failure(item)
        assert result.category == "tool"

    def test_safety_metric(self):
        """metric_id containing 'safety' -> safety category."""
        item = {"item_id": "i6", "output": "long enough output text", "error": "", "metric_id": "toxicity_safety"}
        result = classify_failure(item)
        assert result.category == "safety"
        assert result.subcategory == "policy"

    def test_safety_metric_case_insensitive(self):
        """Safety detection is case-insensitive."""
        item = {"item_id": "i7", "output": "long enough output text", "error": "", "metric_id": "SAFETY_check"}
        result = classify_failure(item)
        assert result.category == "safety"

    def test_short_output_prompt(self):
        """Short output (< 20 chars) -> prompt category."""
        item = {"item_id": "i8", "output": "short", "error": "", "metric_id": "acc"}
        result = classify_failure(item)
        assert result.category == "prompt"
        assert result.subcategory == "insufficient_context"

    def test_other_fallback(self):
        """Long non-matching output -> other."""
        item = {
            "item_id": "i9",
            "output": "This is a sufficiently long output that does not match any pattern",
            "error": "",
            "metric_id": "accuracy",
        }
        result = classify_failure(item)
        assert result.category == "other"

    def test_missing_item_id(self):
        """Missing item_id defaults to 'unknown'."""
        item = {"output": None, "error": "", "metric_id": "acc"}
        result = classify_failure(item)
        assert result.item_id == "unknown"

    def test_missing_error_field(self):
        """Missing error field does not crash."""
        item = {"item_id": "i10", "output": "a long enough output for testing purposes", "metric_id": "acc"}
        result = classify_failure(item)
        assert result.category == "other"

    def test_priority_empty_output_over_tool(self):
        """Empty output takes priority over tool error."""
        item = {"item_id": "i11", "output": None, "error": "Tool failed", "metric_id": "acc"}
        result = classify_failure(item)
        assert result.category == "model"

    def test_priority_tool_over_safety(self):
        """Tool error takes priority over safety metric."""
        item = {"item_id": "i12", "output": "some output", "error": "Tool broke", "metric_id": "safety_check"}
        result = classify_failure(item)
        assert result.category == "tool"

    def test_confidence_model(self):
        """Model failures have confidence 0.8."""
        item = {"item_id": "i13", "output": None, "error": "", "metric_id": "acc"}
        result = classify_failure(item)
        assert result.confidence == 0.8

    def test_confidence_other(self):
        """Other failures have confidence 0.3."""
        item = {
            "item_id": "i14",
            "output": "This output is long enough to not be short",
            "error": "",
            "metric_id": "acc",
        }
        result = classify_failure(item)
        assert result.confidence == 0.3

    def test_evidence_populated(self):
        """Evidence string is not empty."""
        item = {"item_id": "i15", "output": None, "error": "", "metric_id": "acc"}
        result = classify_failure(item)
        assert len(result.evidence) > 0


# ---------------------------------------------------------------------------
# build_taxonomy
# ---------------------------------------------------------------------------


class TestBuildTaxonomy:
    def test_aggregates_categories(self):
        """Multiple failures are grouped by category."""
        failures = [
            {"item_id": "a", "output": None, "error": "", "metric_id": "acc"},
            {"item_id": "b", "output": None, "error": "", "metric_id": "acc"},
            {"item_id": "c", "output": "ok", "error": "Tool error", "metric_id": "acc"},
        ]
        report = build_taxonomy(failures)
        assert report.total_failures == 3
        cat_names = {c.name for c in report.categories}
        assert "model" in cat_names
        assert "tool" in cat_names

    def test_empty_failures(self):
        """Empty list returns empty report."""
        report = build_taxonomy([])
        assert report.total_failures == 0
        assert report.categories == []
        assert report.categorized == []

    def test_all_same_category(self):
        """All failures in one category."""
        failures = [
            {"item_id": f"item_{i}", "output": None, "error": "", "metric_id": "acc"}
            for i in range(5)
        ]
        report = build_taxonomy(failures)
        assert report.total_failures == 5
        assert len(report.categories) == 1
        assert report.categories[0].name == "model"
        assert report.categories[0].item_count == 5

    def test_uncategorized_count(self):
        """Uncategorized count tracks 'other' category."""
        failures = [
            {
                "item_id": "x",
                "output": "This is long enough to not be short at all",
                "error": "",
                "metric_id": "accuracy",
            }
        ]
        report = build_taxonomy(failures)
        assert report.uncategorized_count == 1

    def test_examples_limited(self):
        """Examples list is limited to 5 per category."""
        failures = [
            {"item_id": f"item_{i}", "output": None, "error": "", "metric_id": "acc"}
            for i in range(10)
        ]
        report = build_taxonomy(failures)
        model_cat = [c for c in report.categories if c.name == "model"][0]
        assert len(model_cat.examples) <= 5

    def test_categorized_list(self):
        """All input failures appear in categorized list."""
        failures = [
            {"item_id": "a", "output": None, "error": "", "metric_id": "acc"},
            {"item_id": "b", "output": "ok", "error": "Tool error", "metric_id": "acc"},
        ]
        report = build_taxonomy(failures)
        assert len(report.categorized) == 2
        ids = {c.item_id for c in report.categorized}
        assert ids == {"a", "b"}


# ---------------------------------------------------------------------------
# get_category_distribution
# ---------------------------------------------------------------------------


class TestGetCategoryDistribution:
    def test_distribution(self):
        """Returns counts per category."""
        failures = [
            {"item_id": "a", "output": None, "error": "", "metric_id": "acc"},
            {"item_id": "b", "output": None, "error": "", "metric_id": "acc"},
            {"item_id": "c", "output": "ok", "error": "Tool error", "metric_id": "acc"},
        ]
        report = build_taxonomy(failures)
        dist = get_category_distribution(report)
        assert dist["model"] == 2
        assert dist["tool"] == 1

    def test_empty_report(self):
        """Empty report returns empty dict."""
        report = TaxonomyReport()
        dist = get_category_distribution(report)
        assert dist == {}


# ---------------------------------------------------------------------------
# suggest_category_fixes
# ---------------------------------------------------------------------------


class TestSuggestCategoryFixes:
    def test_known_category(self):
        """Known category returns non-empty list."""
        for cat in FAILURE_CATEGORIES:
            fixes = suggest_category_fixes(cat)
            assert len(fixes) > 0

    def test_unknown_category(self):
        """Unknown category returns 'other' fixes."""
        fixes = suggest_category_fixes("nonexistent")
        other_fixes = suggest_category_fixes("other")
        assert fixes == other_fixes

    def test_prompt_fixes(self):
        """Prompt category has prompt-related suggestions."""
        fixes = suggest_category_fixes("prompt")
        assert any("prompt" in f.lower() or "context" in f.lower() for f in fixes)

    def test_returns_copy(self):
        """Returns a copy, not the original list."""
        fixes1 = suggest_category_fixes("model")
        fixes2 = suggest_category_fixes("model")
        assert fixes1 == fixes2
        fixes1.append("extra")
        assert len(suggest_category_fixes("model")) == len(fixes2)


# ---------------------------------------------------------------------------
# render_taxonomy_tree
# ---------------------------------------------------------------------------


class TestRenderTaxonomyTree:
    def test_tree_contains_total(self):
        """Tree output includes total count."""
        failures = [
            {"item_id": "a", "output": None, "error": "", "metric_id": "acc"},
        ]
        report = build_taxonomy(failures)
        tree = render_taxonomy_tree(report)
        assert "Failures (1)" in tree

    def test_tree_contains_category(self):
        """Tree output includes category names and counts."""
        failures = [
            {"item_id": "a", "output": None, "error": "", "metric_id": "acc"},
            {"item_id": "b", "output": None, "error": "", "metric_id": "acc"},
        ]
        report = build_taxonomy(failures)
        tree = render_taxonomy_tree(report)
        assert "model (2)" in tree

    def test_tree_contains_examples(self):
        """Tree output includes example item IDs."""
        failures = [
            {"item_id": "item_abc", "output": None, "error": "", "metric_id": "acc"},
        ]
        report = build_taxonomy(failures)
        tree = render_taxonomy_tree(report)
        assert "item_abc" in tree

    def test_empty_report_tree(self):
        """Empty report renders without error."""
        report = TaxonomyReport()
        tree = render_taxonomy_tree(report)
        assert "Failures (0)" in tree


# ---------------------------------------------------------------------------
# TaxonomyReport format_text
# ---------------------------------------------------------------------------


class TestTaxonomyReportFormatText:
    def test_format_text_header(self):
        """format_text includes header with total."""
        failures = [
            {"item_id": "a", "output": None, "error": "", "metric_id": "acc"},
        ]
        report = build_taxonomy(failures)
        text = report.format_text()
        assert "1 total failures" in text

    def test_format_text_categories(self):
        """format_text lists categories with counts."""
        failures = [
            {"item_id": "a", "output": None, "error": "", "metric_id": "acc"},
            {"item_id": "b", "output": "ok", "error": "Tool error", "metric_id": "acc"},
        ]
        report = build_taxonomy(failures)
        text = report.format_text()
        assert "model:" in text
        assert "tool:" in text


# ---------------------------------------------------------------------------
# as_dict / from_dict roundtrips
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_failure_category_as_dict(self):
        """FailureCategory.as_dict returns all fields."""
        fc = FailureCategory(name="model", description="desc", item_count=3, examples=["a", "b"])
        d = fc.as_dict()
        assert d["name"] == "model"
        assert d["item_count"] == 3
        assert d["examples"] == ["a", "b"]

    def test_categorized_failure_as_dict(self):
        """CategorizedFailure.as_dict returns all fields."""
        cf = CategorizedFailure(item_id="x", category="tool", subcategory="err", confidence=0.7, evidence="ev")
        d = cf.as_dict()
        assert d["item_id"] == "x"
        assert d["category"] == "tool"
        assert d["confidence"] == 0.7

    def test_taxonomy_report_roundtrip(self):
        """TaxonomyReport survives as_dict -> from_dict roundtrip."""
        failures = [
            {"item_id": "a", "output": None, "error": "", "metric_id": "acc"},
            {"item_id": "b", "output": "ok", "error": "Tool error", "metric_id": "acc"},
        ]
        original = build_taxonomy(failures)
        d = original.as_dict()
        restored = TaxonomyReport.from_dict(d)
        assert restored.total_failures == original.total_failures
        assert restored.uncategorized_count == original.uncategorized_count
        assert len(restored.categories) == len(original.categories)
        assert len(restored.categorized) == len(original.categorized)

    def test_taxonomy_report_from_dict_empty(self):
        """from_dict handles empty dict."""
        report = TaxonomyReport.from_dict({})
        assert report.total_failures == 0
        assert report.categories == []
        assert report.categorized == []
