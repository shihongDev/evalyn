"""Tests for guidelines_generator module."""

from __future__ import annotations

import pytest

from evalyn_sdk.guidelines_generator import (
    AnnotationGuideline,
    GuidelineSection,
    extract_edge_cases,
    format_guideline_html,
    format_guideline_markdown,
    format_guidelines_index,
    generate_batch_guidelines,
    generate_guideline,
)


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

SAMPLE_RUBRIC = {
    "1": "Poor quality. The response is irrelevant or harmful.",
    "2": "Below average. The response may sometimes be on topic but lacks depth.",
    "3": "Average. The response is borderline acceptable.",
    "4": "Good. The response is relevant and helpful.",
    "5": "Excellent. The response is comprehensive unless ambiguous input is given.",
}

SAMPLE_EXAMPLES = [
    {
        "input": "What is 2+2?",
        "output": "4",
        "label": "5",
        "reasoning": "Correct and concise.",
    },
    {
        "input": "Explain quantum physics",
        "output": "It is complicated.",
        "label": "1",
        "reasoning": "Not helpful at all.",
    },
]


# ---------------------------------------------------------------------------
# GuidelineSection
# ---------------------------------------------------------------------------


class TestGuidelineSection:
    def test_create(self):
        s = GuidelineSection(title="Overview", content="Some content")
        assert s.title == "Overview"
        assert s.examples == []

    def test_as_dict(self):
        s = GuidelineSection(
            title="T",
            content="C",
            examples=[{"input": "x", "output": "y", "label": "1", "reasoning": "r"}],
        )
        d = s.as_dict()
        assert d["title"] == "T"
        assert len(d["examples"]) == 1

    def test_from_dict(self):
        d = {"title": "T", "content": "C", "examples": [{"input": "x"}]}
        s = GuidelineSection.from_dict(d)
        assert s.title == "T"
        assert len(s.examples) == 1

    def test_from_dict_defaults(self):
        d = {"title": "T", "content": "C"}
        s = GuidelineSection.from_dict(d)
        assert s.examples == []

    def test_roundtrip(self):
        s = GuidelineSection(
            title="Test",
            content="Content here",
            examples=[{"input": "a", "output": "b", "label": "1", "reasoning": "r"}],
        )
        s2 = GuidelineSection.from_dict(s.as_dict())
        assert s2.title == s.title
        assert len(s2.examples) == len(s.examples)


# ---------------------------------------------------------------------------
# AnnotationGuideline
# ---------------------------------------------------------------------------


class TestAnnotationGuideline:
    def test_create(self):
        g = AnnotationGuideline(metric_id="helpfulness", metric_description="Desc")
        assert g.sections == []
        assert g.edge_cases == []

    def test_as_dict(self):
        g = AnnotationGuideline(
            metric_id="m1",
            metric_description="d",
            sections=[GuidelineSection("t", "c")],
            edge_cases=["edge1"],
            generated_at="2026-01-01",
        )
        d = g.as_dict()
        assert d["metric_id"] == "m1"
        assert len(d["sections"]) == 1
        assert d["edge_cases"] == ["edge1"]

    def test_from_dict(self):
        d = {
            "metric_id": "m1",
            "metric_description": "d",
            "sections": [{"title": "t", "content": "c"}],
            "scale_description": "1-5",
            "edge_cases": ["e1"],
            "generated_at": "2026-01-01",
        }
        g = AnnotationGuideline.from_dict(d)
        assert g.metric_id == "m1"
        assert len(g.sections) == 1

    def test_from_dict_defaults(self):
        d = {"metric_id": "m1", "metric_description": "d"}
        g = AnnotationGuideline.from_dict(d)
        assert g.sections == []
        assert g.edge_cases == []

    def test_roundtrip(self):
        g = AnnotationGuideline(
            metric_id="m1",
            metric_description="d",
            sections=[GuidelineSection("t", "c", [{"input": "x", "output": "y", "label": "1", "reasoning": "r"}])],
            scale_description="1 to 5",
            edge_cases=["case1"],
            generated_at="2026-01-01",
        )
        g2 = AnnotationGuideline.from_dict(g.as_dict())
        assert g2.metric_id == g.metric_id
        assert len(g2.sections) == 1
        assert g2.edge_cases == ["case1"]


# ---------------------------------------------------------------------------
# extract_edge_cases
# ---------------------------------------------------------------------------


class TestExtractEdgeCases:
    def test_finds_borderline(self):
        rubric = {"3": "The response is borderline acceptable."}
        cases = extract_edge_cases(rubric)
        assert len(cases) >= 1
        assert any("borderline" in c.lower() for c in cases)

    def test_finds_ambiguous(self):
        rubric = {"5": "Great unless ambiguous input is given."}
        cases = extract_edge_cases(rubric)
        assert any("ambiguous" in c.lower() for c in cases)

    def test_finds_may(self):
        rubric = {"2": "The response may be off-topic."}
        cases = extract_edge_cases(rubric)
        assert any("may" in c.lower() for c in cases)

    def test_finds_sometimes(self):
        rubric = {"2": "The answer is sometimes correct."}
        cases = extract_edge_cases(rubric)
        assert any("sometimes" in c.lower() for c in cases)

    def test_finds_unless(self):
        rubric = {"4": "Good response unless the question is vague."}
        cases = extract_edge_cases(rubric)
        assert any("unless" in c.lower() for c in cases)

    def test_no_keywords(self):
        rubric = {"1": "Bad.", "5": "Great."}
        cases = extract_edge_cases(rubric)
        assert cases == []

    def test_deduplicates(self):
        rubric = {
            "2": "May be borderline. May be borderline.",
        }
        cases = extract_edge_cases(rubric)
        # "May be borderline." appears twice but should be deduped
        texts = [c for c in cases if "borderline" in c.lower()]
        assert len(texts) == 1

    def test_multiple_sentences(self):
        rubric = {"3": "Borderline case. Sometimes off. Always check."}
        cases = extract_edge_cases(rubric)
        # "Borderline case" and "Sometimes off" - two matches
        assert len(cases) == 2

    def test_includes_score_prefix(self):
        rubric = {"3": "This is borderline."}
        cases = extract_edge_cases(rubric)
        assert cases[0].startswith("Score 3:")

    def test_sample_rubric(self):
        cases = extract_edge_cases(SAMPLE_RUBRIC)
        # Should find: "may sometimes" in score 2, "borderline" in score 3,
        # "unless ambiguous" in score 5
        assert len(cases) >= 3


# ---------------------------------------------------------------------------
# generate_guideline
# ---------------------------------------------------------------------------


class TestGenerateGuideline:
    def test_basic_generation(self):
        g = generate_guideline("helpfulness", "How helpful is the response", SAMPLE_RUBRIC)
        assert g.metric_id == "helpfulness"
        assert g.metric_description == "How helpful is the response"
        assert len(g.sections) == 5  # Overview, Scale, Pass/Fail, Examples, Edge Cases

    def test_sections_titles(self):
        g = generate_guideline("m1", "desc", SAMPLE_RUBRIC)
        titles = [s.title for s in g.sections]
        assert "Overview" in titles
        assert "Scoring Scale" in titles
        assert "Pass/Fail Criteria" in titles
        assert "Examples" in titles
        assert "Edge Cases" in titles

    def test_overview_contains_metric_id(self):
        g = generate_guideline("my_metric", "desc", {"1": "bad", "5": "good"})
        overview = [s for s in g.sections if s.title == "Overview"][0]
        assert "my_metric" in overview.content

    def test_scale_section_lists_scores(self):
        rubric = {"1": "Bad", "2": "OK", "3": "Good"}
        g = generate_guideline("m", "d", rubric)
        scale = [s for s in g.sections if s.title == "Scoring Scale"][0]
        assert "1: Bad" in scale.content
        assert "3: Good" in scale.content

    def test_pass_fail_numeric(self):
        rubric = {"1": "Bad", "2": "OK", "3": "Average", "4": "Good", "5": "Great"}
        g = generate_guideline("m", "d", rubric)
        pf = [s for s in g.sections if s.title == "Pass/Fail Criteria"][0]
        # scores >= 5/2=2.5 pass, so 3,4,5 pass; 1,2 fail
        assert "3" in pf.content
        assert "Passing" in pf.content

    def test_examples_section_with_examples(self):
        g = generate_guideline("m", "d", {"1": "bad"}, examples=SAMPLE_EXAMPLES)
        ex_section = [s for s in g.sections if s.title == "Examples"][0]
        assert len(ex_section.examples) == 2

    def test_examples_section_without_examples(self):
        g = generate_guideline("m", "d", {"1": "bad"})
        ex_section = [s for s in g.sections if s.title == "Examples"][0]
        assert "No examples" in ex_section.content

    def test_edge_cases_populated(self):
        g = generate_guideline("m", "d", SAMPLE_RUBRIC)
        assert len(g.edge_cases) >= 1

    def test_scale_description(self):
        rubric = {"1": "Bad", "5": "Good"}
        g = generate_guideline("m", "d", rubric)
        assert "1: Bad" in g.scale_description
        assert "5: Good" in g.scale_description

    def test_generated_at_set(self):
        g = generate_guideline("m", "d", {"1": "x"})
        assert g.generated_at != ""


# ---------------------------------------------------------------------------
# generate_batch_guidelines
# ---------------------------------------------------------------------------


class TestGenerateBatchGuidelines:
    def test_batch(self):
        metrics = [
            {"id": "m1", "description": "d1", "rubric": {"1": "bad", "5": "good"}},
            {"id": "m2", "description": "d2", "rubric": {"1": "x", "3": "y"}},
        ]
        results = generate_batch_guidelines(metrics)
        assert len(results) == 2
        assert results[0].metric_id == "m1"
        assert results[1].metric_id == "m2"

    def test_batch_with_examples(self):
        metrics = [
            {
                "id": "m1",
                "description": "d1",
                "rubric": {"1": "bad"},
                "examples": [{"input": "a", "output": "b", "label": "1", "reasoning": "r"}],
            }
        ]
        results = generate_batch_guidelines(metrics)
        ex_section = [s for s in results[0].sections if s.title == "Examples"][0]
        assert len(ex_section.examples) == 1

    def test_batch_empty(self):
        assert generate_batch_guidelines([]) == []


# ---------------------------------------------------------------------------
# format_guideline_markdown
# ---------------------------------------------------------------------------


class TestFormatGuidelineMarkdown:
    def test_contains_metric_id_header(self):
        g = generate_guideline("helpfulness", "desc", SAMPLE_RUBRIC)
        md = format_guideline_markdown(g)
        assert md.startswith("# helpfulness")

    def test_contains_description(self):
        g = generate_guideline("m1", "A great metric", {"1": "bad"})
        md = format_guideline_markdown(g)
        assert "A great metric" in md

    def test_contains_section_headers(self):
        g = generate_guideline("m1", "d", SAMPLE_RUBRIC)
        md = format_guideline_markdown(g)
        assert "## Overview" in md
        assert "## Scoring Scale" in md

    def test_contains_examples(self):
        g = generate_guideline("m1", "d", {"1": "bad"}, examples=SAMPLE_EXAMPLES)
        md = format_guideline_markdown(g)
        assert "### Example 1" in md
        assert "What is 2+2?" in md

    def test_contains_edge_cases_summary(self):
        g = generate_guideline("m1", "d", SAMPLE_RUBRIC)
        md = format_guideline_markdown(g)
        assert "## Edge Cases Summary" in md


# ---------------------------------------------------------------------------
# format_guideline_html
# ---------------------------------------------------------------------------


class TestFormatGuidelineHtml:
    def test_valid_html_structure(self):
        g = generate_guideline("m1", "d", {"1": "bad"})
        html_out = format_guideline_html(g)
        assert "<!DOCTYPE html>" in html_out
        assert "<html" in html_out
        assert "</html>" in html_out

    def test_escapes_special_chars(self):
        g = generate_guideline("m<1>", "desc & more", {"1": 'bad "quote"'})
        html_out = format_guideline_html(g)
        assert "m&lt;1&gt;" in html_out
        assert "desc &amp; more" in html_out

    def test_contains_style(self):
        g = generate_guideline("m1", "d", {"1": "bad"})
        html_out = format_guideline_html(g)
        assert "<style>" in html_out

    def test_contains_sections(self):
        g = generate_guideline("m1", "d", SAMPLE_RUBRIC)
        html_out = format_guideline_html(g)
        assert "<h2>Overview</h2>" in html_out
        assert "<h2>Scoring Scale</h2>" in html_out

    def test_examples_in_html(self):
        g = generate_guideline("m1", "d", {"1": "bad"}, examples=SAMPLE_EXAMPLES)
        html_out = format_guideline_html(g)
        assert "Example 1" in html_out
        assert "class=\"example\"" in html_out

    def test_edge_cases_in_html(self):
        g = generate_guideline("m1", "d", SAMPLE_RUBRIC)
        html_out = format_guideline_html(g)
        assert "class=\"edge-case\"" in html_out

    def test_footer_timestamp(self):
        g = generate_guideline("m1", "d", {"1": "x"})
        html_out = format_guideline_html(g)
        assert "Generated at" in html_out


# ---------------------------------------------------------------------------
# format_guidelines_index
# ---------------------------------------------------------------------------


class TestFormatGuidelinesIndex:
    def test_index_header(self):
        index = format_guidelines_index([])
        assert "# Annotation Guidelines Index" in index

    def test_empty_index(self):
        index = format_guidelines_index([])
        assert "No guidelines generated" in index

    def test_index_with_guidelines(self):
        gs = generate_batch_guidelines([
            {"id": "m1", "description": "d1", "rubric": {"1": "bad"}},
            {"id": "m2", "description": "d2", "rubric": {"1": "x", "3": "y"}},
        ])
        index = format_guidelines_index(gs)
        assert "m1" in index
        assert "m2" in index
        assert "| 1 |" in index
        assert "| 2 |" in index

    def test_index_table_format(self):
        gs = [generate_guideline("m1", "d1", {"1": "bad"})]
        index = format_guidelines_index(gs)
        assert "| # | Metric" in index
        assert "|---|" in index
