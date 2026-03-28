"""Tests for multi-agent communication scoring.

All tests use synthetic data - no storage or ML deps required.
"""

from __future__ import annotations

import pytest

from evalyn_sdk.evaluation.multiagent_scoring import (
    AgentMessage,
    CommunicationReport,
    MessageScore,
    compute_collaborative_efficiency,
    evaluate_communication,
    score_clarity,
    score_information_density,
    score_message,
    score_relevance,
)


# ---------------------------------------------------------------------------
# score_relevance
# ---------------------------------------------------------------------------


class TestScoreRelevance:
    def test_high_overlap(self):
        msg = AgentMessage(agent_id="a", content="the quick brown fox")
        context = [AgentMessage(agent_id="b", content="the quick brown dog")]
        rel = score_relevance(msg, context)
        # 3 of 4 words overlap
        assert abs(rel - 0.75) < 1e-9

    def test_no_overlap(self):
        msg = AgentMessage(agent_id="a", content="alpha beta")
        context = [AgentMessage(agent_id="b", content="gamma delta")]
        rel = score_relevance(msg, context)
        assert rel == 0.0

    def test_full_overlap(self):
        msg = AgentMessage(agent_id="a", content="hello world")
        context = [AgentMessage(agent_id="b", content="hello world again")]
        rel = score_relevance(msg, context)
        assert rel == 1.0

    def test_empty_context(self):
        msg = AgentMessage(agent_id="a", content="hello")
        assert score_relevance(msg, []) == 0.0

    def test_empty_message(self):
        msg = AgentMessage(agent_id="a", content="")
        context = [AgentMessage(agent_id="b", content="hello")]
        assert score_relevance(msg, context) == 0.0


# ---------------------------------------------------------------------------
# score_clarity
# ---------------------------------------------------------------------------


class TestScoreClarity:
    def test_short_message(self):
        msg = AgentMessage(agent_id="a", content="hi")
        score = score_clarity(msg)
        assert score < 0.5  # very short, penalized

    def test_optimal_length(self):
        msg = AgentMessage(agent_id="a", content="x" * 200)
        score = score_clarity(msg)
        assert score == 1.0

    def test_very_long(self):
        msg = AgentMessage(agent_id="a", content="x" * 3000)
        score = score_clarity(msg)
        assert score < 0.5

    def test_empty(self):
        msg = AgentMessage(agent_id="a", content="")
        assert score_clarity(msg) == 0.0

    def test_medium_length(self):
        # 100 chars should score close to 1.0
        msg = AgentMessage(agent_id="a", content="x" * 100)
        score = score_clarity(msg)
        assert score >= 0.99

    def test_500_chars(self):
        msg = AgentMessage(agent_id="a", content="x" * 500)
        assert score_clarity(msg) == 1.0

    def test_1000_chars(self):
        msg = AgentMessage(agent_id="a", content="x" * 1000)
        score = score_clarity(msg)
        assert 0.5 < score < 1.0


# ---------------------------------------------------------------------------
# score_information_density
# ---------------------------------------------------------------------------


class TestScoreInformationDensity:
    def test_unique(self):
        msg = AgentMessage(agent_id="a", content="alpha beta gamma delta")
        score = score_information_density(msg)
        assert score == 1.0

    def test_repetitive(self):
        msg = AgentMessage(agent_id="a", content="the the the the")
        score = score_information_density(msg)
        assert abs(score - 0.25) < 1e-9

    def test_half_repeated(self):
        msg = AgentMessage(agent_id="a", content="hello hello world world")
        score = score_information_density(msg)
        assert abs(score - 0.5) < 1e-9

    def test_empty(self):
        msg = AgentMessage(agent_id="a", content="")
        assert score_information_density(msg) == 0.0

    def test_single_word(self):
        msg = AgentMessage(agent_id="a", content="word")
        assert score_information_density(msg) == 1.0


# ---------------------------------------------------------------------------
# score_message
# ---------------------------------------------------------------------------


class TestScoreMessage:
    def test_composite_is_average(self):
        msg = AgentMessage(agent_id="a", content="the quick brown fox jumps")
        context = [AgentMessage(agent_id="b", content="the quick brown dog jumps")]
        s = score_message(msg, context)
        expected = (s.relevance + s.clarity + s.information_density) / 3.0
        assert abs(s.composite - expected) < 1e-9
        assert s.agent_id == "a"


# ---------------------------------------------------------------------------
# compute_collaborative_efficiency
# ---------------------------------------------------------------------------


class TestCollaborativeEfficiency:
    def test_all_substantive(self):
        msgs = [
            AgentMessage(agent_id="a", content="This is a substantive message with detail"),
            AgentMessage(agent_id="b", content="Another meaningful contribution here"),
        ]
        eff = compute_collaborative_efficiency(msgs)
        assert eff == 1.0

    def test_trivial_messages(self):
        msgs = [
            AgentMessage(agent_id="a", content="ok"),
            AgentMessage(agent_id="b", content="yes"),
            AgentMessage(agent_id="c", content="done"),
        ]
        eff = compute_collaborative_efficiency(msgs)
        assert eff == 0.0

    def test_mixed(self):
        msgs = [
            AgentMessage(agent_id="a", content="This is a detailed analysis of the problem"),
            AgentMessage(agent_id="b", content="ok"),
        ]
        eff = compute_collaborative_efficiency(msgs)
        assert abs(eff - 0.5) < 1e-9

    def test_empty(self):
        assert compute_collaborative_efficiency([]) == 0.0

    def test_short_but_not_trivial(self):
        # < 20 chars, not trivial keyword but still not substantive
        msgs = [AgentMessage(agent_id="a", content="hmm maybe?")]
        eff = compute_collaborative_efficiency(msgs)
        assert eff == 0.0


# ---------------------------------------------------------------------------
# evaluate_communication
# ---------------------------------------------------------------------------


class TestEvaluateCommunication:
    def test_full(self):
        msgs = [
            AgentMessage(agent_id="a", content="We need to solve the parsing problem in module X"),
            AgentMessage(agent_id="b", content="I agree about the parsing problem, let me investigate"),
            AgentMessage(agent_id="a", content="The parsing issue is in the tokenizer function"),
        ]
        report = evaluate_communication(msgs)
        assert report.total_messages == 3
        assert report.unique_agents == 2
        assert len(report.scores) == 3
        assert 0.0 <= report.avg_composite <= 1.0
        assert 0.0 <= report.collaborative_efficiency <= 1.0

    def test_single_message(self):
        msgs = [AgentMessage(agent_id="a", content="Hello world, this is a test message")]
        report = evaluate_communication(msgs)
        assert report.total_messages == 1
        assert report.unique_agents == 1
        # First message has no context, so relevance = 0
        assert report.scores[0].relevance == 0.0

    def test_empty(self):
        report = evaluate_communication([])
        assert report.total_messages == 0
        assert report.unique_agents == 0
        assert report.avg_composite == 0.0
        assert report.scores == []


# ---------------------------------------------------------------------------
# format_text
# ---------------------------------------------------------------------------


class TestCommunicationReportFormatText:
    def test_contains_header(self):
        report = CommunicationReport(
            total_messages=5, unique_agents=2,
            avg_composite=0.75, collaborative_efficiency=0.8,
        )
        text = report.format_text()
        assert "Communication Report" in text
        assert "5" in text
        assert "2" in text

    def test_includes_scores(self):
        msgs = [
            AgentMessage(agent_id="agent1", content="Let us discuss the analysis results"),
            AgentMessage(agent_id="agent2", content="The analysis results show improvement"),
        ]
        report = evaluate_communication(msgs)
        text = report.format_text()
        assert "agent1" in text
        assert "agent2" in text


# ---------------------------------------------------------------------------
# as_dict / from_dict roundtrips
# ---------------------------------------------------------------------------


class TestRoundtrips:
    def test_agent_message_as_dict(self):
        m = AgentMessage(agent_id="x", content="hi", timestamp_ms=100.0, message_type="tool")
        d = m.as_dict()
        assert d["agent_id"] == "x"
        assert d["content"] == "hi"
        assert d["timestamp_ms"] == 100.0
        assert d["message_type"] == "tool"

    def test_message_score_as_dict(self):
        s = MessageScore(agent_id="a", relevance=0.5, clarity=0.8, information_density=0.6, composite=0.633)
        d = s.as_dict()
        assert d["agent_id"] == "a"
        assert d["relevance"] == 0.5
        assert d["clarity"] == 0.8

    def test_communication_report_roundtrip(self):
        msgs = [
            AgentMessage(agent_id="a", content="Test message with some content"),
            AgentMessage(agent_id="b", content="Another test message responding"),
        ]
        report = evaluate_communication(msgs)
        d = report.as_dict()
        restored = CommunicationReport.from_dict(d)
        assert restored.total_messages == report.total_messages
        assert restored.unique_agents == report.unique_agents
        assert len(restored.messages) == 2
        assert len(restored.scores) == 2
        assert restored.messages[0].agent_id == "a"

    def test_empty_report_roundtrip(self):
        report = CommunicationReport()
        d = report.as_dict()
        restored = CommunicationReport.from_dict(d)
        assert restored.total_messages == 0
        assert restored.messages == []
        assert restored.scores == []
