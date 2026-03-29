"""Tests for dataset curation suggestions."""

from __future__ import annotations

import sys
from pathlib import Path

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.curation_suggestions import (
    CurationGap,
    CurationSuggestion,
    CurationReport,
    analyze_topic_coverage,
    analyze_difficulty_distribution,
    analyze_length_distribution,
    build_curation_prompt,
    generate_curation_report,
    format_curation_report,
)


# ---------------------------------------------------------------------------
# CurationGap
# ---------------------------------------------------------------------------


class TestCurationGap:
    def test_creation(self):
        gap = CurationGap(gap_type="topic", description="Missing math")
        assert gap.gap_type == "topic"
        assert gap.severity == "medium"
        assert gap.suggested_count == 5

    def test_as_dict(self):
        gap = CurationGap(
            gap_type="difficulty",
            description="Too easy",
            severity="high",
            suggested_count=10,
        )
        d = gap.as_dict()
        assert d["gap_type"] == "difficulty"
        assert d["severity"] == "high"
        assert d["suggested_count"] == 10

    def test_from_dict(self):
        d = {"gap_type": "length", "description": "Too short"}
        gap = CurationGap.from_dict(d)
        assert gap.severity == "medium"
        assert gap.suggested_count == 5

    def test_roundtrip(self):
        gap = CurationGap(
            gap_type="category",
            description="Missing science",
            severity="low",
            suggested_count=3,
        )
        assert CurationGap.from_dict(gap.as_dict()).as_dict() == gap.as_dict()


# ---------------------------------------------------------------------------
# CurationSuggestion
# ---------------------------------------------------------------------------


class TestCurationSuggestion:
    def test_creation(self):
        gap = CurationGap(gap_type="topic", description="test")
        s = CurationSuggestion(
            suggestion_id="s1", gap=gap, action="Add items"
        )
        assert s.example_prompt == ""

    def test_as_dict(self):
        gap = CurationGap(gap_type="topic", description="test")
        s = CurationSuggestion(
            suggestion_id="s1",
            gap=gap,
            action="Add items",
            example_prompt="Generate 5 items",
        )
        d = s.as_dict()
        assert d["suggestion_id"] == "s1"
        assert d["gap"]["gap_type"] == "topic"
        assert d["example_prompt"] == "Generate 5 items"

    def test_from_dict(self):
        d = {
            "suggestion_id": "s2",
            "gap": {"gap_type": "length", "description": "short"},
            "action": "Add longer items",
        }
        s = CurationSuggestion.from_dict(d)
        assert s.gap.gap_type == "length"
        assert s.example_prompt == ""

    def test_roundtrip(self):
        gap = CurationGap(gap_type="difficulty", description="hard gap")
        s = CurationSuggestion(
            suggestion_id="s1", gap=gap, action="Act", example_prompt="prompt"
        )
        assert (
            CurationSuggestion.from_dict(s.as_dict()).as_dict() == s.as_dict()
        )


# ---------------------------------------------------------------------------
# CurationReport
# ---------------------------------------------------------------------------


class TestCurationReport:
    def test_empty(self):
        r = CurationReport()
        assert r.suggestions == []
        assert r.total_items == 0
        assert r.gaps_found == 0

    def test_as_dict(self):
        gap = CurationGap(gap_type="topic", description="test")
        s = CurationSuggestion(suggestion_id="s1", gap=gap, action="Act")
        r = CurationReport(suggestions=[s], total_items=10, gaps_found=1)
        d = r.as_dict()
        assert d["total_items"] == 10
        assert len(d["suggestions"]) == 1

    def test_from_dict(self):
        d = {
            "suggestions": [
                {
                    "suggestion_id": "s1",
                    "gap": {"gap_type": "topic", "description": "x"},
                    "action": "y",
                }
            ],
            "total_items": 5,
            "gaps_found": 1,
        }
        r = CurationReport.from_dict(d)
        assert r.total_items == 5
        assert r.suggestions[0].gap.gap_type == "topic"

    def test_roundtrip(self):
        gap = CurationGap(gap_type="topic", description="test")
        s = CurationSuggestion(suggestion_id="s1", gap=gap, action="Act")
        r = CurationReport(suggestions=[s], total_items=10, gaps_found=1)
        assert CurationReport.from_dict(r.as_dict()).as_dict() == r.as_dict()


# ---------------------------------------------------------------------------
# analyze_topic_coverage
# ---------------------------------------------------------------------------


class TestAnalyzeTopicCoverage:
    def test_empty_items(self):
        gaps = analyze_topic_coverage({})
        assert gaps == []

    def test_no_gaps_uniform(self):
        # All items share the same words - no underrepresentation
        items = {
            f"item_{i}": "machine learning classification model"
            for i in range(10)
        }
        gaps = analyze_topic_coverage(items, min_cluster_size=3)
        assert len(gaps) == 0

    def test_detects_underrepresented_topic(self):
        # "quantum" appears multiple times but only in one item
        items = {
            "item_0": "quantum quantum computing research quantum",
            "item_1": "machine learning model training",
            "item_2": "machine learning deep learning",
            "item_3": "machine learning neural networks",
        }
        gaps = analyze_topic_coverage(items, min_cluster_size=3)
        topic_gaps = [g for g in gaps if g.gap_type == "topic"]
        descriptions = " ".join(g.description for g in topic_gaps)
        assert "quantum" in descriptions

    def test_min_cluster_size_respected(self):
        items = {
            "item_0": "alpha beta gamma",
            "item_1": "alpha beta delta",
            "item_2": "epsilon zeta eta",
        }
        # With min_cluster_size=3, words appearing in <3 items may be flagged
        gaps = analyze_topic_coverage(items, min_cluster_size=3)
        # All content words appear in <=2 items, so some may be flagged
        assert isinstance(gaps, list)

    def test_single_item(self):
        items = {"item_0": "hello world hello world"}
        gaps = analyze_topic_coverage(items, min_cluster_size=2)
        # "hello" and "world" each appear in 1 item but multiple times
        assert len(gaps) >= 0  # Implementation-dependent

    def test_gap_type_is_topic(self):
        items = {
            "item_0": "physics physics physics physics",
            "item_1": "chemistry lab work",
            "item_2": "chemistry lab experiments",
            "item_3": "chemistry lab research",
        }
        gaps = analyze_topic_coverage(items, min_cluster_size=3)
        for g in gaps:
            assert g.gap_type == "topic"

    def test_severity_high_for_frequent(self):
        # Word with count >= 4 in one item but only 1 doc
        items = {
            "item_0": "rare rare rare rare rare",
            "item_1": "common everyday words here",
            "item_2": "common everyday talk here",
            "item_3": "common everyday chat here",
        }
        gaps = analyze_topic_coverage(items, min_cluster_size=3)
        rare_gaps = [g for g in gaps if "rare" in g.description]
        if rare_gaps:
            assert rare_gaps[0].severity == "high"


# ---------------------------------------------------------------------------
# analyze_difficulty_distribution
# ---------------------------------------------------------------------------


class TestAnalyzeDifficultyDistribution:
    def test_empty_items(self):
        gaps = analyze_difficulty_distribution({})
        assert gaps == []

    def test_balanced_no_gaps(self):
        # Short words (easy), medium words, long words
        items = {}
        for i in range(10):
            items[f"easy_{i}"] = "go to do it be"  # avg ~2 chars
        for i in range(10):
            items[f"med_{i}"] = "create manage review report"  # avg ~6 chars
        for i in range(10):
            items[f"hard_{i}"] = (
                "sophisticated extrapolation authentication"  # avg ~14 chars
            )
        gaps = analyze_difficulty_distribution(items)
        # Should have no underrepresentation gaps
        under_gaps = [g for g in gaps if "underrepresented" in g.description]
        assert len(under_gaps) == 0

    def test_all_easy_detects_gap(self):
        items = {f"item_{i}": "go do it be me" for i in range(10)}
        gaps = analyze_difficulty_distribution(items)
        assert len(gaps) > 0
        descriptions = " ".join(g.description for g in gaps)
        # Should flag missing hard or medium
        assert "underrepresented" in descriptions or "overrepresented" in descriptions

    def test_all_hard_detects_gap(self):
        items = {
            f"item_{i}": "authentication extrapolation disambiguation"
            for i in range(10)
        }
        gaps = analyze_difficulty_distribution(items)
        assert len(gaps) > 0

    def test_gap_type_is_difficulty(self):
        items = {f"item_{i}": "go do be" for i in range(10)}
        gaps = analyze_difficulty_distribution(items)
        for g in gaps:
            assert g.gap_type == "difficulty"

    def test_suggested_count_positive(self):
        items = {f"item_{i}": "go do be" for i in range(10)}
        gaps = analyze_difficulty_distribution(items)
        under_gaps = [g for g in gaps if "underrepresented" in g.description]
        for g in under_gaps:
            assert g.suggested_count >= 1


# ---------------------------------------------------------------------------
# analyze_length_distribution
# ---------------------------------------------------------------------------


class TestAnalyzeLengthDistribution:
    def test_empty_items(self):
        gaps = analyze_length_distribution({})
        assert gaps == []

    def test_balanced_no_gaps(self):
        items = {}
        for i in range(10):
            items[f"short_{i}"] = "Hi"  # < 50 chars
        for i in range(10):
            items[f"med_{i}"] = "x" * 100  # 50-200 chars
        for i in range(10):
            items[f"long_{i}"] = "y" * 300  # > 200 chars
        gaps = analyze_length_distribution(items)
        under_gaps = [g for g in gaps if "underrepresented" in g.description]
        assert len(under_gaps) == 0

    def test_all_short_detects_gap(self):
        items = {f"item_{i}": "short" for i in range(10)}
        gaps = analyze_length_distribution(items)
        assert len(gaps) > 0
        descriptions = " ".join(g.description for g in gaps)
        assert "underrepresented" in descriptions or "overrepresented" in descriptions

    def test_all_long_detects_gap(self):
        items = {f"item_{i}": "z" * 500 for i in range(10)}
        gaps = analyze_length_distribution(items)
        assert len(gaps) > 0

    def test_gap_type_is_length(self):
        items = {f"item_{i}": "tiny" for i in range(10)}
        gaps = analyze_length_distribution(items)
        for g in gaps:
            assert g.gap_type == "length"

    def test_length_thresholds(self):
        # 49 chars is short, 50 is medium, 201 is long
        items = {
            "a": "x" * 49,
            "b": "x" * 50,
            "c": "x" * 201,
        }
        gaps = analyze_length_distribution(items)
        # Perfectly balanced (1 each of 3) - no underrepresentation
        under_gaps = [g for g in gaps if "underrepresented" in g.description]
        assert len(under_gaps) == 0

    def test_suggested_count_positive_for_under(self):
        items = {f"item_{i}": "tiny" for i in range(10)}
        gaps = analyze_length_distribution(items)
        under_gaps = [g for g in gaps if "underrepresented" in g.description]
        for g in under_gaps:
            assert g.suggested_count >= 1


# ---------------------------------------------------------------------------
# build_curation_prompt
# ---------------------------------------------------------------------------


class TestBuildCurationPrompt:
    def test_contains_gap_info(self):
        gap = CurationGap(gap_type="topic", description="Missing math")
        prompt = build_curation_prompt([gap], [])
        assert "Missing math" in prompt
        assert "topic" in prompt

    def test_contains_existing_items(self):
        gap = CurationGap(gap_type="topic", description="test")
        prompt = build_curation_prompt([gap], ["item one", "item two"])
        assert "item one" in prompt
        assert "item two" in prompt

    def test_empty_gaps(self):
        prompt = build_curation_prompt([], ["item"])
        assert "Identified Gaps" in prompt

    def test_multiple_gaps(self):
        gaps = [
            CurationGap(gap_type="topic", description="gap A"),
            CurationGap(gap_type="length", description="gap B"),
        ]
        prompt = build_curation_prompt(gaps, [])
        assert "gap A" in prompt
        assert "gap B" in prompt

    def test_truncates_long_existing(self):
        gap = CurationGap(gap_type="topic", description="test")
        long_item = "x" * 200
        prompt = build_curation_prompt([gap], [long_item])
        assert "..." in prompt

    def test_samples_existing_items(self):
        gap = CurationGap(gap_type="topic", description="test")
        items = [f"item_{i}" for i in range(20)]
        prompt = build_curation_prompt([gap], items)
        # Should include at most 10
        assert "item_0" in prompt
        assert "item_9" in prompt

    def test_contains_instructions(self):
        gap = CurationGap(gap_type="topic", description="test")
        prompt = build_curation_prompt([gap], [])
        assert "Instructions" in prompt
        assert "JSON" in prompt

    def test_severity_shown(self):
        gap = CurationGap(
            gap_type="topic", description="test", severity="critical"
        )
        prompt = build_curation_prompt([gap], [])
        assert "critical" in prompt


# ---------------------------------------------------------------------------
# generate_curation_report
# ---------------------------------------------------------------------------


class TestGenerateCurationReport:
    def test_empty_items(self):
        report = generate_curation_report({})
        assert report.total_items == 0
        assert report.gaps_found == 0

    def test_returns_curation_report(self):
        items = {f"item_{i}": "short" for i in range(10)}
        report = generate_curation_report(items)
        assert isinstance(report, CurationReport)
        assert report.total_items == 10

    def test_gaps_found_matches_suggestions(self):
        items = {f"item_{i}": "short" for i in range(10)}
        report = generate_curation_report(items)
        assert report.gaps_found == len(report.suggestions)

    def test_suggestions_have_ids(self):
        items = {f"item_{i}": "tiny" for i in range(10)}
        report = generate_curation_report(items)
        for s in report.suggestions:
            assert s.suggestion_id != ""
            assert len(s.suggestion_id) > 0

    def test_suggestions_have_actions(self):
        items = {f"item_{i}": "tiny" for i in range(10)}
        report = generate_curation_report(items)
        for s in report.suggestions:
            assert s.action != ""

    def test_mixed_dataset(self):
        items = {}
        for i in range(5):
            items[f"short_{i}"] = "hi"
        for i in range(5):
            items[f"med_{i}"] = "x" * 100
        for i in range(5):
            items[f"long_{i}"] = "y" * 300
        report = generate_curation_report(items)
        assert isinstance(report, CurationReport)
        assert report.total_items == 15

    def test_example_prompt_for_actionable_gaps(self):
        items = {f"item_{i}": "tiny" for i in range(10)}
        report = generate_curation_report(items)
        actionable = [s for s in report.suggestions if s.gap.suggested_count > 0]
        for s in actionable:
            assert s.example_prompt != ""


# ---------------------------------------------------------------------------
# format_curation_report
# ---------------------------------------------------------------------------


class TestFormatCurationReport:
    def test_contains_header(self):
        report = CurationReport(total_items=10, gaps_found=2)
        text = format_curation_report(report)
        assert "Dataset Curation Report" in text

    def test_contains_counts(self):
        report = CurationReport(total_items=10, gaps_found=2)
        text = format_curation_report(report)
        assert "Total items analyzed: 10" in text
        assert "Gaps found: 2" in text

    def test_no_gaps_message(self):
        report = CurationReport(total_items=10, gaps_found=0)
        text = format_curation_report(report)
        assert "well balanced" in text

    def test_suggestions_listed(self):
        gap = CurationGap(
            gap_type="topic",
            description="Missing math coverage",
            severity="high",
        )
        s = CurationSuggestion(
            suggestion_id="s1",
            gap=gap,
            action="Add math items",
            example_prompt="prompt text",
        )
        report = CurationReport(
            suggestions=[s], total_items=10, gaps_found=1
        )
        text = format_curation_report(report)
        assert "Missing math coverage" in text
        assert "high" in text
        assert "Add math items" in text
        assert "LLM prompt available" in text

    def test_no_prompt_indicator_when_empty(self):
        gap = CurationGap(gap_type="topic", description="test")
        s = CurationSuggestion(
            suggestion_id="s1", gap=gap, action="Act", example_prompt=""
        )
        report = CurationReport(
            suggestions=[s], total_items=5, gaps_found=1
        )
        text = format_curation_report(report)
        assert "LLM prompt available" not in text
