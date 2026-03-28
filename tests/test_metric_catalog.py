"""Tests for evalyn_sdk.metric_catalog module."""

from __future__ import annotations

import json

import pytest

from evalyn_sdk.metric_catalog import (
    CatalogEntry,
    MetricCatalog,
    build_catalog,
    filter_catalog,
    format_catalog_html,
    format_catalog_markdown,
    format_catalog_plain,
    get_category_summary,
)


# ---------------------------------------------------------------------------
# Sample test data
# ---------------------------------------------------------------------------

SAMPLE_CATEGORIES = {
    "safety": "Content safety and harm prevention",
    "correctness": "Factual accuracy and truthfulness",
    "style": "Writing quality, tone, and clarity",
}

SAMPLE_REGISTRY = [
    {
        "id": "toxicity_safety",
        "type": "subjective",
        "description": "PASS if output is safe.",
        "category": "safety",
        "scope": "llm_call",
        "config": {
            "rubric": [
                "No harassment.",
                "No self-harm instructions.",
                "No illegal content.",
            ],
        },
    },
    {
        "id": "helpfulness_accuracy",
        "type": "subjective",
        "description": "PASS if the output answers accurately.",
        "category": "correctness",
        "scope": "overall",
        "config": {
            "rubric": [
                "Addresses core question.",
                "Accurate information.",
                "Acknowledges limitations.",
                "Actionable content.",
            ],
        },
    },
    {
        "id": "tone_alignment",
        "type": "subjective",
        "description": "PASS if tone matches.",
        "category": "style",
        "scope": "llm_call",
        "config": {
            "rubric": [
                "Matches desired tone.",
                "Consistent tone.",
            ],
        },
    },
    {
        "id": "coherence_clarity",
        "type": "subjective",
        "description": "PASS if response is logically structured.",
        "category": "style",
        "scope": "llm_call",
        "config": {
            "rubric": [
                "Logical flow.",
                "No contradictions.",
                "Good transitions.",
                "Acceptable grammar.",
                "Appropriate terminology.",
            ],
        },
    },
    {
        "id": "factual_check",
        "type": "objective",
        "description": "Checks factual accuracy against reference.",
        "category": "correctness",
        "scope": "overall",
        "config": {},
    },
]

SAMPLE_BUNDLES = {
    "chatbot": ["toxicity_safety", "helpfulness_accuracy", "tone_alignment"],
    "content-writer": ["coherence_clarity", "tone_alignment"],
}


@pytest.fixture
def catalog():
    return build_catalog(SAMPLE_REGISTRY, SAMPLE_CATEGORIES, SAMPLE_BUNDLES)


# ---------------------------------------------------------------------------
# CatalogEntry tests
# ---------------------------------------------------------------------------


class TestCatalogEntry:
    def test_as_dict_roundtrip(self):
        entry = CatalogEntry(
            metric_id="test_metric",
            metric_type="subjective",
            description="A test",
            category="safety",
            scope="llm_call",
            rubric_levels=3,
            bundle_memberships=["chatbot"],
            recommended_use="Per-call evaluation",
        )
        d = entry.as_dict()
        restored = CatalogEntry.from_dict(d)
        assert restored.metric_id == "test_metric"
        assert restored.metric_type == "subjective"
        assert restored.rubric_levels == 3
        assert restored.bundle_memberships == ["chatbot"]
        assert restored.recommended_use == "Per-call evaluation"

    def test_from_dict_defaults(self):
        entry = CatalogEntry.from_dict({"metric_id": "x", "metric_type": "obj"})
        assert entry.description == ""
        assert entry.category == ""
        assert entry.scope == ""
        assert entry.rubric_levels == 0
        assert entry.bundle_memberships == []
        assert entry.recommended_use == ""

    def test_as_dict_keys(self):
        entry = CatalogEntry(
            metric_id="m",
            metric_type="t",
            description="d",
            category="c",
            scope="s",
            rubric_levels=1,
        )
        d = entry.as_dict()
        expected_keys = {
            "metric_id",
            "metric_type",
            "description",
            "category",
            "scope",
            "rubric_levels",
            "bundle_memberships",
            "recommended_use",
        }
        assert set(d.keys()) == expected_keys

    def test_bundle_memberships_is_copy(self):
        entry = CatalogEntry(
            metric_id="m",
            metric_type="t",
            description="d",
            category="c",
            scope="s",
            rubric_levels=0,
            bundle_memberships=["a", "b"],
        )
        d = entry.as_dict()
        d["bundle_memberships"].append("c")
        assert len(entry.bundle_memberships) == 2


# ---------------------------------------------------------------------------
# MetricCatalog tests
# ---------------------------------------------------------------------------


class TestMetricCatalog:
    def test_as_dict_roundtrip(self, catalog):
        d = catalog.as_dict()
        restored = MetricCatalog.from_dict(d)
        assert restored.total_count == catalog.total_count
        assert len(restored.entries) == len(catalog.entries)
        assert restored.categories == catalog.categories
        assert restored.generated_at == catalog.generated_at

    def test_from_dict_defaults(self):
        mc = MetricCatalog.from_dict({})
        assert mc.entries == []
        assert mc.categories == {}
        assert mc.total_count == 0
        assert mc.generated_at == ""

    def test_json_serializable(self, catalog):
        d = catalog.as_dict()
        s = json.dumps(d)
        loaded = json.loads(s)
        restored = MetricCatalog.from_dict(loaded)
        assert restored.total_count == catalog.total_count

    def test_categories_dict_is_copy(self, catalog):
        d = catalog.as_dict()
        d["categories"]["new_cat"] = "blah"
        assert "new_cat" not in catalog.categories


# ---------------------------------------------------------------------------
# build_catalog tests
# ---------------------------------------------------------------------------


class TestBuildCatalog:
    def test_entry_count(self, catalog):
        assert catalog.total_count == 5
        assert len(catalog.entries) == 5

    def test_categories_passed_through(self, catalog):
        assert catalog.categories == SAMPLE_CATEGORIES

    def test_rubric_levels_from_list(self, catalog):
        tox = next(e for e in catalog.entries if e.metric_id == "toxicity_safety")
        assert tox.rubric_levels == 3

    def test_rubric_levels_empty(self, catalog):
        fc = next(e for e in catalog.entries if e.metric_id == "factual_check")
        assert fc.rubric_levels == 0

    def test_bundle_memberships(self, catalog):
        tox = next(e for e in catalog.entries if e.metric_id == "toxicity_safety")
        assert "chatbot" in tox.bundle_memberships

    def test_multi_bundle_membership(self, catalog):
        tone = next(e for e in catalog.entries if e.metric_id == "tone_alignment")
        assert sorted(tone.bundle_memberships) == ["chatbot", "content-writer"]

    def test_no_bundle_membership(self, catalog):
        fc = next(e for e in catalog.entries if e.metric_id == "factual_check")
        assert fc.bundle_memberships == []

    def test_no_bundles_arg(self):
        cat = build_catalog(SAMPLE_REGISTRY, SAMPLE_CATEGORIES)
        for e in cat.entries:
            assert e.bundle_memberships == []

    def test_recommended_use_contains_scope(self, catalog):
        tox = next(e for e in catalog.entries if e.metric_id == "toxicity_safety")
        assert "Per-call" in tox.recommended_use

    def test_recommended_use_overall_scope(self, catalog):
        h = next(e for e in catalog.entries if e.metric_id == "helpfulness_accuracy")
        assert "End-to-end" in h.recommended_use

    def test_generated_at_nonempty(self, catalog):
        assert catalog.generated_at != ""

    def test_empty_registry(self):
        cat = build_catalog([], {})
        assert cat.total_count == 0
        assert cat.entries == []

    def test_metric_type_preserved(self, catalog):
        fc = next(e for e in catalog.entries if e.metric_id == "factual_check")
        assert fc.metric_type == "objective"

    def test_scope_preserved(self, catalog):
        tox = next(e for e in catalog.entries if e.metric_id == "toxicity_safety")
        assert tox.scope == "llm_call"

    def test_rubric_as_dict(self):
        reg = [
            {
                "id": "dict_rubric",
                "type": "subjective",
                "description": "Has dict rubric",
                "category": "safety",
                "scope": "overall",
                "config": {
                    "rubric": {"5": "Excellent", "4": "Good", "3": "Average"},
                },
            }
        ]
        cat = build_catalog(reg, {})
        entry = cat.entries[0]
        assert entry.rubric_levels == 3


# ---------------------------------------------------------------------------
# filter_catalog tests
# ---------------------------------------------------------------------------


class TestFilterCatalog:
    def test_filter_by_category(self, catalog):
        result = filter_catalog(catalog, category="safety")
        assert result.total_count == 1
        assert result.entries[0].metric_id == "toxicity_safety"

    def test_filter_by_scope(self, catalog):
        result = filter_catalog(catalog, scope="llm_call")
        assert result.total_count == 3

    def test_filter_by_search_in_id(self, catalog):
        result = filter_catalog(catalog, search="toxicity")
        assert result.total_count == 1

    def test_filter_by_search_in_description(self, catalog):
        result = filter_catalog(catalog, search="accurately")
        assert result.total_count == 1
        assert result.entries[0].metric_id == "helpfulness_accuracy"

    def test_filter_case_insensitive(self, catalog):
        result = filter_catalog(catalog, search="PASS")
        assert result.total_count >= 1

    def test_filter_combined_category_and_scope(self, catalog):
        result = filter_catalog(catalog, category="style", scope="llm_call")
        assert result.total_count == 2

    def test_filter_combined_category_and_search(self, catalog):
        result = filter_catalog(catalog, category="correctness", search="factual")
        assert result.total_count == 1

    def test_filter_no_match(self, catalog):
        result = filter_catalog(catalog, category="nonexistent")
        assert result.total_count == 0
        assert result.entries == []

    def test_filter_preserves_categories(self, catalog):
        result = filter_catalog(catalog, category="safety")
        assert result.categories == catalog.categories

    def test_filter_preserves_generated_at(self, catalog):
        result = filter_catalog(catalog, search="tone")
        assert result.generated_at == catalog.generated_at

    def test_filter_no_args_returns_all(self, catalog):
        result = filter_catalog(catalog)
        assert result.total_count == catalog.total_count


# ---------------------------------------------------------------------------
# format_catalog_markdown tests
# ---------------------------------------------------------------------------


class TestFormatCatalogMarkdown:
    def test_has_title(self, catalog):
        md = format_catalog_markdown(catalog)
        assert "# Metric Catalog" in md

    def test_has_toc(self, catalog):
        md = format_catalog_markdown(catalog)
        assert "## Table of Contents" in md
        assert "safety" in md
        assert "correctness" in md

    def test_has_metric_entries(self, catalog):
        md = format_catalog_markdown(catalog)
        assert "### toxicity_safety" in md
        assert "### helpfulness_accuracy" in md

    def test_has_description(self, catalog):
        md = format_catalog_markdown(catalog)
        assert "PASS if output is safe." in md

    def test_has_scope(self, catalog):
        md = format_catalog_markdown(catalog)
        assert "llm_call" in md

    def test_has_rubric_levels(self, catalog):
        md = format_catalog_markdown(catalog)
        assert "Rubric levels" in md

    def test_has_bundles(self, catalog):
        md = format_catalog_markdown(catalog)
        assert "chatbot" in md

    def test_empty_catalog(self):
        cat = MetricCatalog(generated_at="2026-01-01T00:00:00+00:00")
        md = format_catalog_markdown(cat)
        assert "Total metrics: 0" in md


# ---------------------------------------------------------------------------
# format_catalog_html tests
# ---------------------------------------------------------------------------


class TestFormatCatalogHtml:
    def test_is_valid_html_structure(self, catalog):
        html_str = format_catalog_html(catalog)
        assert "<!DOCTYPE html>" in html_str
        assert "<html" in html_str
        assert "</html>" in html_str
        assert "<table>" in html_str

    def test_has_inline_css(self, catalog):
        html_str = format_catalog_html(catalog)
        assert "<style>" in html_str
        assert "border-collapse" in html_str

    def test_has_filter_links(self, catalog):
        html_str = format_catalog_html(catalog)
        assert "filter-links" in html_str
        assert "safety" in html_str

    def test_has_search_hint(self, catalog):
        html_str = format_catalog_html(catalog)
        assert "search-hint" in html_str
        assert "Ctrl+F" in html_str

    def test_has_metric_rows(self, catalog):
        html_str = format_catalog_html(catalog)
        assert "toxicity_safety" in html_str
        assert "helpfulness_accuracy" in html_str

    def test_html_escapes_content(self):
        """Ensure user content with special chars is escaped."""
        reg = [
            {
                "id": "xss_test",
                "type": "subjective",
                "description": '<script>alert("xss")</script>',
                "category": "safety",
                "scope": "overall",
                "config": {"rubric": ["a"]},
            }
        ]
        cat = build_catalog(reg, {"safety": "test <b>bold</b>"})
        html_str = format_catalog_html(cat)
        assert "<script>" not in html_str
        assert "&lt;script&gt;" in html_str

    def test_empty_catalog_html(self):
        cat = MetricCatalog(generated_at="2026-01-01T00:00:00+00:00")
        html_str = format_catalog_html(cat)
        assert "<table>" in html_str
        assert "(0 metrics)" in html_str


# ---------------------------------------------------------------------------
# format_catalog_plain tests
# ---------------------------------------------------------------------------


class TestFormatCatalogPlain:
    def test_has_header(self, catalog):
        txt = format_catalog_plain(catalog)
        assert "Metric Catalog" in txt
        assert str(catalog.total_count) in txt

    def test_has_column_headers(self, catalog):
        txt = format_catalog_plain(catalog)
        assert "ID" in txt
        assert "Type" in txt
        assert "Category" in txt

    def test_has_metrics(self, catalog):
        txt = format_catalog_plain(catalog)
        assert "toxicity_safety" in txt
        assert "helpfulness_accuracy" in txt

    def test_has_separator(self, catalog):
        txt = format_catalog_plain(catalog)
        assert "---" in txt

    def test_empty_catalog_plain(self):
        cat = MetricCatalog(total_count=0, generated_at="now")
        txt = format_catalog_plain(cat)
        assert "0 metrics" in txt


# ---------------------------------------------------------------------------
# get_category_summary tests
# ---------------------------------------------------------------------------


class TestGetCategorySummary:
    def test_counts(self, catalog):
        summary = get_category_summary(catalog)
        assert summary["safety"] == 1
        assert summary["correctness"] == 2
        assert summary["style"] == 2

    def test_all_categories_present(self, catalog):
        summary = get_category_summary(catalog)
        assert set(summary.keys()) == {"safety", "correctness", "style"}

    def test_empty_catalog_summary(self):
        cat = MetricCatalog()
        summary = get_category_summary(cat)
        assert summary == {}

    def test_sum_equals_total(self, catalog):
        summary = get_category_summary(catalog)
        assert sum(summary.values()) == catalog.total_count
