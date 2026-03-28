"""Tests for reasoning evaluation."""

import pytest
from evalyn_sdk.evaluation.reasoning_eval import (
    ReasoningStep,
    ReasoningEvalResult,
    evaluate_faithfulness,
    evaluate_logical_consistency,
    evaluate_evidence_usage,
    evaluate_conclusion_validity,
    evaluate_reasoning,
    extract_reasoning_steps,
    _word_overlap,
)


# -- ReasoningStep as_dict / from_dict --


class TestReasoningStepSerialization:
    def test_as_dict(self):
        step = ReasoningStep(text="the sky is blue", step_type="premise", referenced_evidence=["source1"])
        d = step.as_dict()
        assert d == {
            "text": "the sky is blue",
            "step_type": "premise",
            "referenced_evidence": ["source1"],
        }

    def test_from_dict(self):
        d = {"text": "therefore X", "step_type": "conclusion", "referenced_evidence": ["a", "b"]}
        step = ReasoningStep.from_dict(d)
        assert step.text == "therefore X"
        assert step.step_type == "conclusion"
        assert step.referenced_evidence == ["a", "b"]

    def test_roundtrip(self):
        step = ReasoningStep(text="test", step_type="evidence", referenced_evidence=["ref"])
        assert ReasoningStep.from_dict(step.as_dict()).as_dict() == step.as_dict()

    def test_from_dict_defaults(self):
        d = {"text": "hello"}
        step = ReasoningStep.from_dict(d)
        assert step.step_type == "reasoning"
        assert step.referenced_evidence == []


# -- ReasoningEvalResult --


class TestReasoningEvalResultSerialization:
    def test_as_dict(self):
        r = ReasoningEvalResult(
            faithfulness=0.9,
            logical_consistency=0.8,
            evidence_usage=0.7,
            conclusion_validity=0.6,
            overall=0.75,
            step_count=4,
        )
        d = r.as_dict()
        assert d["faithfulness"] == 0.9
        assert d["step_count"] == 4
        assert d["details"] == {}

    def test_from_dict(self):
        d = {
            "faithfulness": 0.5,
            "logical_consistency": 0.6,
            "evidence_usage": 0.7,
            "conclusion_validity": 0.8,
            "overall": 0.65,
            "step_count": 3,
            "details": {"note": "test"},
        }
        r = ReasoningEvalResult.from_dict(d)
        assert r.faithfulness == 0.5
        assert r.step_count == 3
        assert r.details == {"note": "test"}

    def test_format_text(self):
        r = ReasoningEvalResult(
            faithfulness=1.0,
            logical_consistency=1.0,
            evidence_usage=1.0,
            conclusion_validity=1.0,
            overall=1.0,
            step_count=5,
        )
        text = r.format_text()
        assert "Reasoning Evaluation Result" in text
        assert "faithfulness" in text
        assert "step_count" in text
        assert "1.000" in text


# -- evaluate_faithfulness --


class TestEvaluateFaithfulness:
    def test_high_overlap(self):
        source = "the database stores user records and handles queries efficiently"
        steps = [
            ReasoningStep("the database stores records for user queries", step_type="reasoning"),
            ReasoningStep("queries are handled by the database efficiently", step_type="reasoning"),
        ]
        score = evaluate_faithfulness(steps, source)
        assert score == 1.0  # both should have > 0.1 overlap

    def test_no_overlap(self):
        source = "quantum mechanics explains particle behavior"
        steps = [
            ReasoningStep("the weather is sunny today", step_type="reasoning"),
            ReasoningStep("cats enjoy sleeping in warm places", step_type="reasoning"),
        ]
        score = evaluate_faithfulness(steps, source)
        assert score == 0.0

    def test_partial_overlap(self):
        source = "the database handles user queries"
        steps = [
            ReasoningStep("the database handles queries well", step_type="reasoning"),
            ReasoningStep("completely unrelated topic xyz", step_type="reasoning"),
        ]
        score = evaluate_faithfulness(steps, source)
        assert score == 0.5

    def test_only_reasoning_steps_counted(self):
        source = "the database handles queries"
        steps = [
            ReasoningStep("totally unrelated text", step_type="conclusion"),
            ReasoningStep("the database handles queries well", step_type="reasoning"),
        ]
        score = evaluate_faithfulness(steps, source)
        assert score == 1.0  # only 1 reasoning step, and it overlaps

    def test_no_reasoning_steps(self):
        steps = [ReasoningStep("conclusion text", step_type="conclusion")]
        assert evaluate_faithfulness(steps, "anything") == 1.0


# -- evaluate_logical_consistency --


class TestEvaluateLogicalConsistency:
    def test_connected_steps(self):
        steps = [
            ReasoningStep("the database stores user records", step_type="reasoning"),
            ReasoningStep("user records in the database are indexed", step_type="reasoning"),
            ReasoningStep("indexed user records enable fast database queries", step_type="reasoning"),
        ]
        score = evaluate_logical_consistency(steps)
        assert score > 0.2  # should have meaningful overlap

    def test_disconnected_steps(self):
        steps = [
            ReasoningStep("quantum physics explains subatomic particles", step_type="reasoning"),
            ReasoningStep("chocolate cake tastes delicious with vanilla", step_type="reasoning"),
        ]
        score = evaluate_logical_consistency(steps)
        assert score < 0.1  # very little word overlap

    def test_single_step(self):
        steps = [ReasoningStep("only one step", step_type="reasoning")]
        assert evaluate_logical_consistency(steps) == 1.0

    def test_non_reasoning_steps_excluded(self):
        steps = [
            ReasoningStep("quantum physics explains particles", step_type="reasoning"),
            ReasoningStep("connecting premise middle", step_type="premise"),
            ReasoningStep("chocolate cake tastes delicious", step_type="reasoning"),
        ]
        score = evaluate_logical_consistency(steps)
        # Only reasoning steps are compared: no shared words between them
        assert score < 0.1


# -- evaluate_evidence_usage --


class TestEvaluateEvidenceUsage:
    def test_all_have_evidence(self):
        steps = [
            ReasoningStep("step1", referenced_evidence=["ref1"]),
            ReasoningStep("step2", referenced_evidence=["ref2"]),
        ]
        assert evaluate_evidence_usage(steps) == 1.0

    def test_none_have_evidence(self):
        steps = [
            ReasoningStep("step1"),
            ReasoningStep("step2"),
        ]
        assert evaluate_evidence_usage(steps) == 0.0

    def test_partial_evidence(self):
        steps = [
            ReasoningStep("step1", referenced_evidence=["ref1"]),
            ReasoningStep("step2"),
        ]
        assert evaluate_evidence_usage(steps) == 0.5

    def test_empty_steps(self):
        assert evaluate_evidence_usage([]) == 1.0


# -- evaluate_conclusion_validity --


class TestEvaluateConclusionValidity:
    def test_references_reasoning(self):
        steps = [
            ReasoningStep("the database stores user records efficiently", step_type="reasoning"),
            ReasoningStep("the database stores user records efficiently and enables fast queries", step_type="conclusion"),
        ]
        score = evaluate_conclusion_validity(steps)
        assert score > 0.5  # significant overlap

    def test_no_conclusions(self):
        steps = [ReasoningStep("just reasoning", step_type="reasoning")]
        assert evaluate_conclusion_validity(steps) == 1.0

    def test_no_reasoning_steps(self):
        steps = [ReasoningStep("a conclusion with no basis", step_type="conclusion")]
        assert evaluate_conclusion_validity(steps) == 0.0

    def test_unrelated_conclusion(self):
        steps = [
            ReasoningStep("quantum physics explains particle behavior", step_type="reasoning"),
            ReasoningStep("chocolate cake is delicious with vanilla", step_type="conclusion"),
        ]
        score = evaluate_conclusion_validity(steps)
        assert score < 0.1


# -- evaluate_reasoning (full pipeline) --


class TestEvaluateReasoning:
    def test_full_pipeline(self):
        source = "databases store records and handle queries"
        steps = [
            ReasoningStep(
                "databases store records",
                step_type="reasoning",
                referenced_evidence=["doc1"],
            ),
            ReasoningStep(
                "records in databases are queried",
                step_type="reasoning",
                referenced_evidence=["doc2"],
            ),
            ReasoningStep(
                "therefore databases store and query records efficiently",
                step_type="conclusion",
                referenced_evidence=["doc3"],
            ),
        ]
        result = evaluate_reasoning(steps, source)
        assert result.step_count == 3
        assert result.faithfulness > 0.0
        assert result.evidence_usage == 1.0
        assert 0.0 <= result.overall <= 1.0

    def test_overall_is_average(self):
        steps = [
            ReasoningStep("test step", step_type="reasoning", referenced_evidence=["r"]),
        ]
        result = evaluate_reasoning(steps, "test step")
        # faithfulness: overlap > 0.1 so 1.0
        # consistency: single step so 1.0
        # evidence: 1/1 = 1.0
        # conclusion: no conclusions so 1.0
        expected = (1.0 + 1.0 + 1.0 + 1.0) / 4.0
        assert abs(result.overall - expected) < 1e-9

    def test_details_populated(self):
        steps = [
            ReasoningStep("reasoning text", step_type="reasoning"),
            ReasoningStep("conclusion text", step_type="conclusion"),
        ]
        result = evaluate_reasoning(steps, "some source")
        assert result.details["reasoning_step_count"] == 1
        assert result.details["conclusion_step_count"] == 1
        assert result.details["source_text_length"] == len("some source")

    def test_empty_steps(self):
        result = evaluate_reasoning([], "source")
        assert result.step_count == 0
        assert result.overall > 0.0  # all metrics return 1.0 for empty


# -- extract_reasoning_steps --


class TestExtractReasoningSteps:
    def test_numbered_pattern(self):
        text = "1. First do this 2. Then do that 3. In conclusion we finish"
        steps = extract_reasoning_steps(text)
        assert len(steps) == 3
        assert "First do this" in steps[0].text
        assert steps[2].step_type == "conclusion"

    def test_keyword_splitting(self):
        text = "The sky is blue because light scatters. However clouds block it. Therefore it rains."
        steps = extract_reasoning_steps(text)
        assert len(steps) >= 3
        # Should have at least one premise (because) and one conclusion (therefore)
        types = [s.step_type for s in steps]
        assert "premise" in types
        assert "conclusion" in types

    def test_empty_text(self):
        assert extract_reasoning_steps("") == []
        assert extract_reasoning_steps("   ") == []

    def test_single_sentence(self):
        steps = extract_reasoning_steps("Just one sentence.")
        assert len(steps) == 1
        assert steps[0].step_type == "reasoning"

    def test_numbered_with_premise(self):
        text = "1. Because we need data 2. We build a pipeline 3. In conclusion it works"
        steps = extract_reasoning_steps(text)
        assert steps[0].step_type == "premise"
        assert steps[2].step_type == "conclusion"


# -- word overlap helper --


class TestWordOverlap:
    def test_identical(self):
        assert _word_overlap("hello world", "hello world") == 1.0

    def test_no_overlap(self):
        assert _word_overlap("alpha beta", "gamma delta") == 0.0

    def test_partial(self):
        score = _word_overlap("hello world foo", "hello bar baz")
        # intersection: {hello}, union: {hello, world, foo, bar, baz} = 5
        assert abs(score - 1.0 / 5.0) < 1e-9

    def test_empty_strings(self):
        assert _word_overlap("", "") == 0.0
        assert _word_overlap("hello", "") == 0.0


# -- Edge cases --


class TestEdgeCases:
    def test_single_reasoning_step_faithfulness(self):
        step = ReasoningStep("the cat sat on the mat", step_type="reasoning")
        # High overlap with source
        assert evaluate_faithfulness([step], "the cat sat on the mat") == 1.0

    def test_single_step_consistency(self):
        step = ReasoningStep("only step", step_type="reasoning")
        assert evaluate_logical_consistency([step]) == 1.0

    def test_empty_steps_all_metrics(self):
        assert evaluate_faithfulness([], "source") == 1.0
        assert evaluate_logical_consistency([]) == 1.0
        assert evaluate_evidence_usage([]) == 1.0
        assert evaluate_conclusion_validity([]) == 1.0
