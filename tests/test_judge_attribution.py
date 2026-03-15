"""Tests for span attribution in LLM judge prompt and response parsing."""
import json

from evalyn_sdk.judges.llm_judge import LLMJudge


def test_parse_span_attribution():
    """Test that span_attribution is extracted from judge response."""
    judge = LLMJudge.__new__(LLMJudge)
    judge.rubric = None

    raw = json.dumps(
        {
            "passed": True,
            "reason": "Good output",
            "score": 0.9,
            "span_attribution": [
                {"span_id": "span-1", "relevance": 0.8, "reason": "main LLM call"},
                {"span_id": "span-2", "relevance": 0.3, "reason": "tool call"},
            ],
        }
    )

    score, passed, reason, parsed = judge._parse_response(raw)
    assert passed is True
    assert "span_attribution" in parsed
    assert len(parsed["span_attribution"]) == 2
    assert parsed["span_attribution"][0]["span_id"] == "span-1"
    assert parsed["span_attribution"][0]["relevance"] == 0.8


def test_parse_response_without_attribution():
    """Responses without span_attribution should parse normally."""
    judge = LLMJudge.__new__(LLMJudge)
    judge.rubric = None

    raw = json.dumps({"passed": False, "reason": "Bad output", "score": 0.2})

    score, passed, reason, parsed = judge._parse_response(raw)
    assert passed is False
    assert score == 0.2
    assert "span_attribution" not in parsed


def test_prompt_includes_span_attribution_instruction():
    """Verify the prompt mentions span_attribution in return format."""
    judge = LLMJudge.__new__(LLMJudge)
    judge.prompt = "Test prompt"
    judge.rubric = None

    from evalyn_sdk.models import FunctionCall, DatasetItem

    call = FunctionCall.from_dict(
        {
            "id": "call-1",
            "function_name": "test_fn",
            "session_id": "s1",
            "started_at": "2026-01-01T00:00:00",
            "ended_at": "2026-01-01T00:00:01",
            "duration_ms": 1000,
            "inputs": {},
            "output": "test output",
            "error": None,
            "trace": [],
            "metadata": {},
        }
    )

    item = DatasetItem(id="item-1", input="test input", output="test output")

    prompt = judge._build_evaluation_prompt(call, item)
    assert "span_attribution" in prompt
    assert "relevance" in prompt


def test_prompt_includes_span_ids_when_present():
    """When call has spans, the prompt should include span info."""
    judge = LLMJudge.__new__(LLMJudge)
    judge.prompt = "Test prompt"
    judge.rubric = None

    from evalyn_sdk.models import FunctionCall, DatasetItem, Span

    span = Span.new(name="llm_call", span_type="llm_call")

    call = FunctionCall.from_dict(
        {
            "id": "call-1",
            "function_name": "test_fn",
            "session_id": "s1",
            "started_at": "2026-01-01T00:00:00",
            "ended_at": "2026-01-01T00:00:01",
            "duration_ms": 1000,
            "inputs": {},
            "output": "test output",
            "error": None,
            "trace": [],
            "metadata": {},
            "spans": [span.as_dict()],
        }
    )

    item = DatasetItem(id="item-1", input="test input", output="test output")

    prompt = judge._build_evaluation_prompt(call, item)
    assert span.id in prompt
    assert "llm_call" in prompt
