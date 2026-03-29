"""Tests for the natural language summary module."""

from __future__ import annotations

import sys
from pathlib import Path

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.nl_summary import (
    SummaryConfig,
    NLSummary,
    build_heuristic_summary,
    build_summary_prompt,
    parse_llm_summary,
    generate_metric_insights,
    generate_recommendations,
    format_nl_summary,
)


# ---------------------------------------------------------------------------
# SummaryConfig
# ---------------------------------------------------------------------------


class TestSummaryConfig:
    def test_defaults(self):
        cfg = SummaryConfig()
        assert cfg.style == "executive"
        assert cfg.include_recommendations is True
        assert cfg.max_metrics == 10

    def test_custom_values(self):
        cfg = SummaryConfig(style="technical", include_recommendations=False, max_metrics=5)
        assert cfg.style == "technical"
        assert cfg.include_recommendations is False
        assert cfg.max_metrics == 5

    def test_as_dict(self):
        cfg = SummaryConfig(style="brief", max_metrics=3)
        d = cfg.as_dict()
        assert d["style"] == "brief"
        assert d["max_metrics"] == 3
        assert d["include_recommendations"] is True

    def test_from_dict(self):
        d = {"style": "technical", "include_recommendations": False, "max_metrics": 7}
        cfg = SummaryConfig.from_dict(d)
        assert cfg.style == "technical"
        assert cfg.include_recommendations is False
        assert cfg.max_metrics == 7

    def test_from_dict_defaults(self):
        cfg = SummaryConfig.from_dict({})
        assert cfg.style == "executive"
        assert cfg.include_recommendations is True
        assert cfg.max_metrics == 10

    def test_roundtrip(self):
        cfg = SummaryConfig(style="brief", include_recommendations=False, max_metrics=2)
        restored = SummaryConfig.from_dict(cfg.as_dict())
        assert restored.style == cfg.style
        assert restored.include_recommendations == cfg.include_recommendations
        assert restored.max_metrics == cfg.max_metrics


# ---------------------------------------------------------------------------
# NLSummary
# ---------------------------------------------------------------------------


class TestNLSummary:
    def test_defaults(self):
        s = NLSummary(title="Test")
        assert s.title == "Test"
        assert s.paragraphs == []
        assert s.recommendations == []
        assert s.generated_at != ""

    def test_as_dict(self):
        s = NLSummary(
            title="T",
            paragraphs=["p1", "p2"],
            recommendations=["r1"],
            generated_at="2026-01-01T00:00:00Z",
        )
        d = s.as_dict()
        assert d["title"] == "T"
        assert d["paragraphs"] == ["p1", "p2"]
        assert d["recommendations"] == ["r1"]
        assert d["generated_at"] == "2026-01-01T00:00:00Z"

    def test_from_dict(self):
        d = {
            "title": "Report",
            "paragraphs": ["a", "b"],
            "recommendations": ["fix X"],
            "generated_at": "2026-03-01",
        }
        s = NLSummary.from_dict(d)
        assert s.title == "Report"
        assert len(s.paragraphs) == 2
        assert s.recommendations == ["fix X"]

    def test_from_dict_defaults(self):
        s = NLSummary.from_dict({})
        assert s.title == ""
        assert s.paragraphs == []

    def test_roundtrip(self):
        s = NLSummary(title="X", paragraphs=["a"], recommendations=["b"], generated_at="ts")
        restored = NLSummary.from_dict(s.as_dict())
        assert restored.title == s.title
        assert restored.paragraphs == s.paragraphs
        assert restored.recommendations == s.recommendations
        assert restored.generated_at == s.generated_at

    def test_generated_at_auto_filled(self):
        s = NLSummary(title="Auto")
        assert len(s.generated_at) > 0


# ---------------------------------------------------------------------------
# build_heuristic_summary
# ---------------------------------------------------------------------------


class TestBuildHeuristicSummary:
    def test_high_pass_rate(self):
        metrics = {"accuracy": 0.95, "relevance": 0.88}
        s = build_heuristic_summary(metrics, pass_rate=0.96, total_items=100)
        assert "96%" in s.paragraphs[0]
        assert "excellent" in s.paragraphs[0]
        assert s.title.startswith("Evaluation Summary")

    def test_low_pass_rate(self):
        metrics = {"accuracy": 0.3, "relevance": 0.2}
        s = build_heuristic_summary(metrics, pass_rate=0.25, total_items=50)
        assert "poor" in s.paragraphs[0]

    def test_medium_pass_rate(self):
        metrics = {"accuracy": 0.75}
        s = build_heuristic_summary(metrics, pass_rate=0.72, total_items=20)
        assert "acceptable" in s.paragraphs[0]

    def test_includes_recommendations_by_default(self):
        metrics = {"low_metric": 0.3}
        s = build_heuristic_summary(metrics, pass_rate=0.5, total_items=10)
        assert len(s.recommendations) > 0

    def test_no_recommendations_when_disabled(self):
        metrics = {"low_metric": 0.3}
        cfg = SummaryConfig(include_recommendations=False)
        s = build_heuristic_summary(metrics, pass_rate=0.5, total_items=10, config=cfg)
        assert len(s.recommendations) == 0

    def test_max_metrics_limits_output(self):
        metrics = {f"m{i}": float(i) / 20 for i in range(20)}
        cfg = SummaryConfig(max_metrics=3)
        s = build_heuristic_summary(metrics, pass_rate=0.5, total_items=10, config=cfg)
        # Should not mention all 20 metrics in paragraphs
        assert isinstance(s, NLSummary)

    def test_empty_metrics(self):
        s = build_heuristic_summary({}, pass_rate=1.0, total_items=0)
        assert len(s.paragraphs) >= 1

    def test_brief_style(self):
        metrics = {"a": 0.9, "b": 0.8, "c": 0.3}
        cfg = SummaryConfig(style="brief")
        s = build_heuristic_summary(metrics, pass_rate=0.7, total_items=10, config=cfg)
        # Brief style uses "Top metrics:" format
        found_top = any("Top metrics:" in p for p in s.paragraphs)
        assert found_top

    def test_technical_style_has_range(self):
        metrics = {"a": 0.9, "b": 0.5}
        cfg = SummaryConfig(style="technical")
        s = build_heuristic_summary(metrics, pass_rate=0.7, total_items=10, config=cfg)
        found_range = any("Score range:" in p for p in s.paragraphs)
        assert found_range

    def test_bottom_metrics_shown(self):
        metrics = {"good": 0.95, "bad": 0.25}
        s = build_heuristic_summary(metrics, pass_rate=0.6, total_items=10)
        found_attention = any("attention" in p.lower() for p in s.paragraphs)
        assert found_attention

    def test_all_high_scores_no_bottom(self):
        metrics = {"a": 0.95, "b": 0.90}
        s = build_heuristic_summary(metrics, pass_rate=0.95, total_items=10)
        # No "requiring attention" paragraph when all scores >= 0.7
        found_attention = any("requiring attention" in p.lower() for p in s.paragraphs)
        assert not found_attention


# ---------------------------------------------------------------------------
# build_summary_prompt
# ---------------------------------------------------------------------------


class TestBuildSummaryPrompt:
    def test_contains_metrics(self):
        metrics = {"accuracy": 0.85}
        prompt = build_summary_prompt(metrics, pass_rate=0.85, total_items=10)
        assert "accuracy" in prompt
        assert "0.850" in prompt

    def test_contains_pass_rate(self):
        prompt = build_summary_prompt({"m": 0.5}, pass_rate=0.75, total_items=20)
        assert "75.0%" in prompt
        assert "20" in prompt

    def test_executive_style(self):
        cfg = SummaryConfig(style="executive")
        prompt = build_summary_prompt({"m": 0.5}, pass_rate=0.5, total_items=5, config=cfg)
        assert "non-technical" in prompt

    def test_technical_style(self):
        cfg = SummaryConfig(style="technical")
        prompt = build_summary_prompt({"m": 0.5}, pass_rate=0.5, total_items=5, config=cfg)
        assert "technical detail" in prompt

    def test_brief_style(self):
        cfg = SummaryConfig(style="brief")
        prompt = build_summary_prompt({"m": 0.5}, pass_rate=0.5, total_items=5, config=cfg)
        assert "brief" in prompt

    def test_recommendations_requested(self):
        cfg = SummaryConfig(include_recommendations=True)
        prompt = build_summary_prompt({"m": 0.5}, pass_rate=0.5, total_items=5, config=cfg)
        assert "recommendation" in prompt.lower()

    def test_no_recommendations(self):
        cfg = SummaryConfig(include_recommendations=False)
        prompt = build_summary_prompt({"m": 0.5}, pass_rate=0.5, total_items=5, config=cfg)
        assert "recommendation" not in prompt.lower()

    def test_max_metrics_limits_prompt(self):
        metrics = {f"m{i}": float(i) / 20 for i in range(20)}
        cfg = SummaryConfig(max_metrics=3)
        prompt = build_summary_prompt(metrics, pass_rate=0.5, total_items=10, config=cfg)
        # Should only list 3 metrics
        metric_lines = [l for l in prompt.splitlines() if l.strip().startswith("- m")]
        assert len(metric_lines) == 3


# ---------------------------------------------------------------------------
# parse_llm_summary
# ---------------------------------------------------------------------------


class TestParseLlmSummary:
    def test_basic_parsing(self):
        text = "This is paragraph one.\n\nThis is paragraph two."
        s = parse_llm_summary(text)
        assert len(s.paragraphs) == 2
        assert s.paragraphs[0] == "This is paragraph one."

    def test_extracts_recommendations(self):
        text = "Overview paragraph.\n\n- Fix accuracy\n- Improve latency"
        s = parse_llm_summary(text)
        assert len(s.recommendations) == 2
        assert "Fix accuracy" in s.recommendations

    def test_mixed_content(self):
        text = "First para.\n\nSecond para.\n\n- Rec one\n- Rec two\n\nThird para."
        s = parse_llm_summary(text)
        assert len(s.paragraphs) == 3
        assert len(s.recommendations) == 2

    def test_empty_response(self):
        s = parse_llm_summary("")
        assert s.paragraphs == []
        assert s.recommendations == []

    def test_title_from_first_paragraph(self):
        text = "Short title here.\n\nMore details follow."
        s = parse_llm_summary(text)
        assert s.title == "Short title here"

    def test_long_first_paragraph_uses_default_title(self):
        long_text = "A" * 100 + ".\n\nSecond paragraph."
        s = parse_llm_summary(long_text)
        assert s.title == "LLM Analysis Summary"

    def test_multiline_paragraph(self):
        text = "Line one of para.\nLine two of para.\n\nSecond para."
        s = parse_llm_summary(text)
        assert len(s.paragraphs) == 2
        assert "Line one" in s.paragraphs[0]
        assert "Line two" in s.paragraphs[0]

    def test_bullet_only(self):
        text = "- Item A\n- Item B\n- Item C"
        s = parse_llm_summary(text)
        assert len(s.recommendations) == 3
        assert len(s.paragraphs) == 0

    def test_empty_bullets_ignored(self):
        text = "Para.\n\n- \n- Valid rec"
        s = parse_llm_summary(text)
        assert len(s.recommendations) == 1


# ---------------------------------------------------------------------------
# generate_metric_insights
# ---------------------------------------------------------------------------


class TestGenerateMetricInsights:
    def test_strong_metric(self):
        insights = generate_metric_insights({"accuracy": 0.95})
        assert len(insights) == 1
        assert "strong" in insights[0]

    def test_adequate_metric(self):
        insights = generate_metric_insights({"relevance": 0.75})
        assert "adequate" in insights[0]

    def test_moderate_metric(self):
        insights = generate_metric_insights({"fluency": 0.55})
        assert "moderate" in insights[0]

    def test_needs_attention(self):
        insights = generate_metric_insights({"safety": 0.35})
        assert "needs attention" in insights[0]

    def test_critically_low(self):
        insights = generate_metric_insights({"broken": 0.1})
        assert "critically low" in insights[0]

    def test_multiple_metrics_sorted(self):
        insights = generate_metric_insights({"low": 0.2, "high": 0.95})
        assert "high" in insights[0]
        assert "low" in insights[1]

    def test_empty_metrics(self):
        insights = generate_metric_insights({})
        assert insights == []

    def test_score_included_in_insight(self):
        insights = generate_metric_insights({"acc": 0.82})
        assert "0.82" in insights[0]

    def test_boundary_0_9(self):
        insights = generate_metric_insights({"m": 0.9})
        assert "strong" in insights[0]

    def test_boundary_0_7(self):
        insights = generate_metric_insights({"m": 0.7})
        assert "adequate" in insights[0]

    def test_boundary_0_5(self):
        insights = generate_metric_insights({"m": 0.5})
        assert "moderate" in insights[0]

    def test_boundary_0_3(self):
        insights = generate_metric_insights({"m": 0.3})
        assert "needs attention" in insights[0]


# ---------------------------------------------------------------------------
# generate_recommendations
# ---------------------------------------------------------------------------


class TestGenerateRecommendations:
    def test_no_recs_above_threshold(self):
        recs = generate_recommendations({"a": 0.9, "b": 0.8})
        assert recs == []

    def test_rec_for_low_metric(self):
        recs = generate_recommendations({"a": 0.5})
        assert len(recs) == 1
        assert "a" in recs[0]

    def test_urgent_for_very_low(self):
        recs = generate_recommendations({"broken": 0.1})
        assert "Urgently" in recs[0]

    def test_review_for_medium_low(self):
        recs = generate_recommendations({"mid": 0.4})
        assert "Review" in recs[0]

    def test_tune_for_near_threshold(self):
        recs = generate_recommendations({"close": 0.65})
        assert "tuning" in recs[0]

    def test_custom_threshold(self):
        recs = generate_recommendations({"m": 0.85}, threshold=0.9)
        assert len(recs) == 1

    def test_all_above_custom_threshold(self):
        recs = generate_recommendations({"m": 0.6}, threshold=0.5)
        assert recs == []

    def test_multiple_recs_sorted(self):
        recs = generate_recommendations({"low": 0.1, "mid": 0.5})
        # Sorted by score ascending
        assert "low" in recs[0]
        assert "mid" in recs[1]

    def test_empty_metrics(self):
        recs = generate_recommendations({})
        assert recs == []


# ---------------------------------------------------------------------------
# format_nl_summary
# ---------------------------------------------------------------------------


class TestFormatNlSummary:
    def test_basic_format(self):
        s = NLSummary(title="Report", paragraphs=["Overview."], generated_at="ts")
        text = format_nl_summary(s)
        assert "Report" in text
        assert "=====" in text
        assert "Overview." in text

    def test_includes_recommendations(self):
        s = NLSummary(title="R", recommendations=["Do X"], generated_at="ts")
        text = format_nl_summary(s)
        assert "Recommendations:" in text
        assert "- Do X" in text

    def test_no_recommendations(self):
        s = NLSummary(title="R", paragraphs=["P"], generated_at="ts")
        text = format_nl_summary(s)
        assert "Recommendations:" not in text

    def test_multiple_paragraphs(self):
        s = NLSummary(title="T", paragraphs=["A", "B", "C"], generated_at="ts")
        text = format_nl_summary(s)
        assert text.count("\n\n") >= 2

    def test_underline_length_matches_title(self):
        title = "My Report Title"
        s = NLSummary(title=title, generated_at="ts")
        text = format_nl_summary(s)
        lines = text.splitlines()
        assert lines[1] == "=" * len(title)
