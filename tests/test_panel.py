"""Tests for the expert panel module."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

# Add SDK to path
SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from conftest import make_run_analysis
from evalyn_sdk.analysis.core import RunAnalysis, MetricStats, ItemStats
from evalyn_sdk.analysis.insights import InsightsReport, CorrelationResult, RegressionAlert
from evalyn_sdk.analysis.panel import (
    ExpertOpinion,
    PanelDiscussion,
    EXPERT_ROLES,
    prepare_panel_context,
    build_expert_prompt,
    build_moderator_prompt,
    parse_expert_response,
    create_api_client,
    run_expert_panel,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sample_analysis():
    """Build a sample RunAnalysis with mixed results."""
    items = {}
    for i in range(10):
        items[f"item_{i}"] = {
            "helpfulness": {
                "passed": i < 7,
                "score": 0.9 if i < 7 else 0.3,
                "reason": None if i < 7 else "Output not helpful",
            },
            "accuracy": {
                "passed": i < 8,
                "score": 0.8 if i < 8 else 0.2,
                "reason": None if i < 8 else "Inaccurate response",
            },
        }
    return make_run_analysis(items)


def _make_sample_report():
    """Build a sample InsightsReport."""
    return InsightsReport(
        correlations=[
            CorrelationResult(
                metric_a="helpfulness", metric_b="accuracy",
                pearson=0.85, relationship="redundant",
            ),
        ],
        regressions=[
            RegressionAlert(
                metric_id="helpfulness",
                previous_pass_rate=0.9,
                current_pass_rate=0.7,
                delta=-0.2,
                severity="critical",
            ),
        ],
    )


@dataclass
class MockGenerateResult:
    text: str
    input_tokens: int = 100
    output_tokens: int = 50
    model: str = "mock-model"


class MockClient:
    """Mock API client for testing."""

    def __init__(self, responses: list[str] | None = None):
        self._responses = responses or []
        self._call_count = 0
        self.prompts: list[str] = []

    def generate_with_usage(self, prompt: str, temperature=None):
        self.prompts.append(prompt)
        if self._call_count < len(self._responses):
            text = self._responses[self._call_count]
        else:
            text = json.dumps({
                "summary": f"Mock analysis #{self._call_count}",
                "findings": ["Finding 1"],
                "concerns": ["Concern 1"],
                "suggestions": ["Suggestion 1"],
                "confidence": 0.7,
            })
        self._call_count += 1
        return MockGenerateResult(text=text)


# ---------------------------------------------------------------------------
# Data Model Tests
# ---------------------------------------------------------------------------


class TestExpertOpinion:
    def test_as_dict_roundtrip(self):
        opinion = ExpertOpinion(
            role="quality_analyst",
            summary="Test summary",
            findings=["f1", "f2"],
            concerns=["c1"],
            suggestions=["s1"],
            confidence=0.8,
        )
        d = opinion.as_dict()
        restored = ExpertOpinion.from_dict(d)
        assert restored.role == "quality_analyst"
        assert restored.summary == "Test summary"
        assert restored.findings == ["f1", "f2"]
        assert restored.confidence == 0.8

    def test_from_dict_defaults(self):
        opinion = ExpertOpinion.from_dict({})
        assert opinion.role == "unknown"
        assert opinion.summary == ""
        assert opinion.findings == []
        assert opinion.confidence == 0.0


class TestPanelDiscussion:
    def test_as_dict_roundtrip(self):
        discussion = PanelDiscussion(
            experts=[
                ExpertOpinion(role="quality_analyst", summary="Test"),
            ],
            synthesis="Overall analysis",
            action_plan=["Step 1", "Step 2"],
            dissenting_views=["View 1"],
            total_input_tokens=500,
            total_output_tokens=300,
        )
        d = discussion.as_dict()
        restored = PanelDiscussion.from_dict(d)
        assert len(restored.experts) == 1
        assert restored.synthesis == "Overall analysis"
        assert restored.action_plan == ["Step 1", "Step 2"]
        assert restored.total_input_tokens == 500


# ---------------------------------------------------------------------------
# Context Preparation Tests
# ---------------------------------------------------------------------------


class TestPrepareContext:
    def test_includes_metric_stats(self):
        analysis = _make_sample_analysis()
        report = InsightsReport()
        context = prepare_panel_context(analysis, report)
        assert "helpfulness" in context
        assert "accuracy" in context
        assert "METRIC STATS" in context

    def test_includes_failed_items(self):
        analysis = _make_sample_analysis()
        report = InsightsReport()
        context = prepare_panel_context(analysis, report)
        assert "FAILED ITEMS" in context

    def test_includes_correlations(self):
        analysis = _make_sample_analysis()
        report = _make_sample_report()
        context = prepare_panel_context(analysis, report)
        assert "CORRELATIONS" in context
        assert "helpfulness" in context

    def test_includes_regressions(self):
        analysis = _make_sample_analysis()
        report = _make_sample_report()
        context = prepare_panel_context(analysis, report)
        assert "REGRESSIONS" in context
        assert "critical" in context

    def test_sample_failures_limit(self):
        analysis = _make_sample_analysis()
        report = InsightsReport()
        context = prepare_panel_context(analysis, report, sample_failures=2)
        # Should show at most 2 failed items
        failed_lines = [l for l in context.split("\n") if "... failed:" in l]
        assert len(failed_lines) <= 2

    def test_empty_report(self):
        analysis = make_run_analysis({"item_0": {"m": {"passed": True, "score": 1.0}}})
        report = InsightsReport()
        context = prepare_panel_context(analysis, report)
        assert "METRIC STATS" in context
        assert "CORRELATIONS" not in context


# ---------------------------------------------------------------------------
# Prompt Builder Tests
# ---------------------------------------------------------------------------


class TestBuildExpertPrompt:
    def test_contains_role_instruction(self):
        prompt = build_expert_prompt("quality_analyst", "test context")
        assert "Quality Analyst" in prompt

    def test_contains_context(self):
        prompt = build_expert_prompt("quality_analyst", "EVALUATION DATA HERE")
        assert "EVALUATION DATA HERE" in prompt

    def test_contains_output_schema(self):
        prompt = build_expert_prompt("quality_analyst", "ctx")
        assert "JSON" in prompt
        assert "findings" in prompt
        assert "confidence" in prompt

    def test_includes_prior_opinions(self):
        prior = [ExpertOpinion(role="metric_critic", summary="Prior analysis")]
        prompt = build_expert_prompt("data_scientist", "ctx", prior_opinions=prior)
        assert "PRIOR EXPERT" in prompt
        assert "Prior analysis" in prompt

    def test_no_prior_opinions(self):
        prompt = build_expert_prompt("quality_analyst", "ctx", prior_opinions=None)
        assert "PRIOR EXPERT" not in prompt

    def test_all_roles_have_instructions(self):
        for role in EXPERT_ROLES:
            prompt = build_expert_prompt(role, "ctx")
            assert len(prompt) > 100


class TestBuildModeratorPrompt:
    def test_contains_all_opinions(self):
        opinions = [
            ExpertOpinion(role="quality_analyst", summary="QA says X"),
            ExpertOpinion(role="strategist", summary="Strat says Y"),
        ]
        prompt = build_moderator_prompt("ctx", opinions)
        assert "QA says X" in prompt
        assert "Strat says Y" in prompt
        assert "Moderator" in prompt

    def test_output_schema(self):
        prompt = build_moderator_prompt("ctx", [])
        assert "synthesis" in prompt
        assert "action_plan" in prompt
        assert "dissenting_views" in prompt


# ---------------------------------------------------------------------------
# Response Parsing Tests
# ---------------------------------------------------------------------------


class TestParseExpertResponse:
    def test_parse_valid_json(self):
        raw = json.dumps({
            "summary": "Analysis complete",
            "findings": ["f1", "f2", "f3"],
            "concerns": ["c1"],
            "suggestions": ["s1", "s2"],
            "confidence": 0.85,
        })
        result = parse_expert_response("quality_analyst", raw)
        assert result.role == "quality_analyst"
        assert result.summary == "Analysis complete"
        assert len(result.findings) == 3
        assert result.confidence == 0.85

    def test_parse_json_in_markdown(self):
        raw = '```json\n{"summary": "test", "findings": [], "confidence": 0.9}\n```'
        result = parse_expert_response("metric_critic", raw)
        assert result.summary == "test"
        assert result.confidence == 0.9

    def test_parse_json_with_surrounding_text(self):
        raw = 'Here is my analysis: {"summary": "embedded", "findings": ["f1"], "confidence": 0.7} end'
        result = parse_expert_response("data_scientist", raw)
        assert result.summary == "embedded"

    def test_parse_invalid_json_fallback(self):
        raw = "This is not JSON at all"
        result = parse_expert_response("strategist", raw)
        assert result.role == "strategist"
        assert result.confidence == 0.3  # fallback confidence
        assert "This is not JSON" in result.summary

    def test_truncates_lists(self):
        raw = json.dumps({
            "summary": "test",
            "findings": [f"f{i}" for i in range(10)],
            "concerns": [f"c{i}" for i in range(10)],
            "suggestions": [f"s{i}" for i in range(10)],
            "confidence": 0.5,
        })
        result = parse_expert_response("quality_analyst", raw)
        assert len(result.findings) <= 5
        assert len(result.concerns) <= 3
        assert len(result.suggestions) <= 3

    def test_clamps_confidence(self):
        raw = json.dumps({"summary": "test", "confidence": 1.5})
        result = parse_expert_response("quality_analyst", raw)
        assert result.confidence == 1.0

        raw = json.dumps({"summary": "test", "confidence": -0.5})
        result = parse_expert_response("quality_analyst", raw)
        assert result.confidence == 0.0


# ---------------------------------------------------------------------------
# Expert Panel Orchestration Tests
# ---------------------------------------------------------------------------


class TestRunExpertPanel:
    def test_runs_all_experts_by_default(self):
        analysis = _make_sample_analysis()
        report = _make_sample_report()
        client = MockClient()
        discussion = run_expert_panel(analysis, report, client)
        # 4 experts + 1 moderator = 5 calls
        assert len(client.prompts) == 5
        assert len(discussion.experts) == 4

    def test_expert_subset(self):
        analysis = _make_sample_analysis()
        report = InsightsReport()
        client = MockClient()
        discussion = run_expert_panel(
            analysis, report, client,
            experts=["quality_analyst", "strategist"],
        )
        # 2 experts + 1 moderator = 3 calls
        assert len(client.prompts) == 3
        assert len(discussion.experts) == 2
        assert discussion.experts[0].role == "quality_analyst"
        assert discussion.experts[1].role == "strategist"

    def test_invalid_experts_fallback(self):
        analysis = _make_sample_analysis()
        report = InsightsReport()
        client = MockClient()
        discussion = run_expert_panel(
            analysis, report, client,
            experts=["nonexistent_role"],
        )
        # Falls back to all experts
        assert len(discussion.experts) == 4

    def test_token_tracking(self):
        analysis = _make_sample_analysis()
        report = InsightsReport()
        client = MockClient()
        discussion = run_expert_panel(analysis, report, client)
        # 5 calls x 100 input + 5 calls x 50 output
        assert discussion.total_input_tokens == 500
        assert discussion.total_output_tokens == 250

    def test_sequential_opinion_passing(self):
        """Later experts should receive prior experts' opinions in their prompts."""
        analysis = _make_sample_analysis()
        report = InsightsReport()
        client = MockClient()
        run_expert_panel(analysis, report, client)

        # First expert should NOT have prior opinions
        assert "PRIOR EXPERT" not in client.prompts[0]
        # Second expert should have first expert's opinion
        assert "PRIOR EXPERT" in client.prompts[1]
        # Third expert should reference prior opinions
        assert "PRIOR EXPERT" in client.prompts[2]

    def test_moderator_receives_all_opinions(self):
        analysis = _make_sample_analysis()
        report = InsightsReport()
        responses = [
            json.dumps({"summary": f"Expert {i} says something", "findings": [], "confidence": 0.8})
            for i in range(4)
        ]
        # Add moderator response
        responses.append(json.dumps({
            "synthesis": "All experts agree",
            "action_plan": ["Do X", "Do Y"],
            "dissenting_views": ["Expert 1 disagrees"],
        }))
        client = MockClient(responses=responses)
        discussion = run_expert_panel(analysis, report, client)
        assert discussion.synthesis == "All experts agree"
        assert discussion.action_plan == ["Do X", "Do Y"]
        assert len(discussion.dissenting_views) == 1

    def test_as_dict_roundtrip(self):
        """PanelDiscussion should survive serialization/deserialization."""
        analysis = _make_sample_analysis()
        report = InsightsReport()
        client = MockClient()
        discussion = run_expert_panel(analysis, report, client)
        d = discussion.as_dict()
        restored = PanelDiscussion.from_dict(d)
        assert len(restored.experts) == len(discussion.experts)
        assert restored.total_input_tokens == discussion.total_input_tokens


# ---------------------------------------------------------------------------
# Client Creation Tests
# ---------------------------------------------------------------------------


class TestCreateApiClient:
    def test_gemini_default(self):
        client = create_api_client(provider="gemini")
        assert client.model == "gemini-2.5-flash"

    def test_model_override(self):
        client = create_api_client(provider="gemini", model="gemini-2.5-flash-lite")
        assert client.model == "gemini-2.5-flash-lite"

    def test_openai_client(self):
        client = create_api_client(provider="openai")
        assert client.model == "gpt-4o-mini"

    def test_ollama_client(self):
        client = create_api_client(provider="ollama")
        assert client.model == "llama3.2"
