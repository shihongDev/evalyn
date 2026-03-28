from __future__ import annotations

from datetime import datetime, timezone

import pytest

from evalyn_sdk.models import Span
from evalyn_sdk.trace.anthropic_thinking import (
    ThinkingBlock,
    ThinkingAnalysis,
    extract_thinking_blocks,
    analyze_thinking,
    inject_thinking_into_span,
    extract_thinking_from_span,
    has_thinking,
    compute_thinking_overhead,
)


def _span(sid: str = "s1", **attrs) -> Span:
    return Span(
        id=sid,
        name="test",
        span_type="llm_call",
        parent_id=None,
        start_time=datetime.now(timezone.utc),
        attributes=attrs,
    )


# -- ThinkingBlock -------------------------------------------------------------

class TestThinkingBlock:
    def test_basic(self):
        b = ThinkingBlock(text="Let me think...", token_count=10, duration_ms=50.0)
        assert b.text == "Let me think..."
        assert b.token_count == 10
        assert b.duration_ms == 50.0

    def test_defaults(self):
        b = ThinkingBlock(text="hello")
        assert b.token_count == 0
        assert b.duration_ms == 0.0

    def test_as_dict(self):
        b = ThinkingBlock(text="abc", token_count=5, duration_ms=1.2)
        d = b.as_dict()
        assert d == {"text": "abc", "token_count": 5, "duration_ms": 1.2}

    def test_from_dict(self):
        b = ThinkingBlock.from_dict({"text": "x", "token_count": 3, "duration_ms": 0.5})
        assert b.text == "x"
        assert b.token_count == 3
        assert b.duration_ms == 0.5

    def test_from_dict_defaults(self):
        b = ThinkingBlock.from_dict({})
        assert b.text == ""
        assert b.token_count == 0
        assert b.duration_ms == 0.0

    def test_roundtrip(self):
        original = ThinkingBlock(text="deep thought", token_count=42, duration_ms=100.0)
        rebuilt = ThinkingBlock.from_dict(original.as_dict())
        assert rebuilt.text == original.text
        assert rebuilt.token_count == original.token_count
        assert rebuilt.duration_ms == original.duration_ms


# -- ThinkingAnalysis ----------------------------------------------------------

class TestThinkingAnalysis:
    def test_defaults(self):
        a = ThinkingAnalysis()
        assert a.blocks == []
        assert a.total_tokens == 0
        assert a.total_blocks == 0
        assert a.avg_block_length == 0.0
        assert a.has_thinking is False

    def test_as_dict(self):
        a = ThinkingAnalysis(
            blocks=[ThinkingBlock(text="hi", token_count=2)],
            total_tokens=2,
            total_blocks=1,
            avg_block_length=2.0,
            has_thinking=True,
        )
        d = a.as_dict()
        assert d["total_tokens"] == 2
        assert d["total_blocks"] == 1
        assert d["has_thinking"] is True
        assert len(d["blocks"]) == 1

    def test_from_dict(self):
        data = {
            "blocks": [{"text": "yo", "token_count": 1}],
            "total_tokens": 1,
            "total_blocks": 1,
            "avg_block_length": 2.0,
            "has_thinking": True,
        }
        a = ThinkingAnalysis.from_dict(data)
        assert a.total_tokens == 1
        assert len(a.blocks) == 1
        assert a.blocks[0].text == "yo"

    def test_roundtrip(self):
        original = ThinkingAnalysis(
            blocks=[ThinkingBlock(text="block1", token_count=10, duration_ms=5.0)],
            total_tokens=10,
            total_blocks=1,
            avg_block_length=6.0,
            has_thinking=True,
        )
        rebuilt = ThinkingAnalysis.from_dict(original.as_dict())
        assert rebuilt.total_tokens == original.total_tokens
        assert rebuilt.total_blocks == original.total_blocks
        assert rebuilt.has_thinking == original.has_thinking
        assert len(rebuilt.blocks) == len(original.blocks)

    def test_format_text_no_thinking(self):
        a = ThinkingAnalysis()
        assert a.format_text() == "No thinking blocks found."

    def test_format_text_with_blocks(self):
        a = ThinkingAnalysis(
            blocks=[
                ThinkingBlock(text="short thought", token_count=5),
                ThinkingBlock(text="another one", token_count=3),
            ],
            total_tokens=8,
            total_blocks=2,
            avg_block_length=12.0,
            has_thinking=True,
        )
        text = a.format_text()
        assert "2 block(s)" in text
        assert "8 tokens" in text
        assert "[1]" in text
        assert "[2]" in text

    def test_format_text_long_block_truncated(self):
        long_text = "x" * 200
        a = ThinkingAnalysis(
            blocks=[ThinkingBlock(text=long_text, token_count=100)],
            total_tokens=100,
            total_blocks=1,
            avg_block_length=200.0,
            has_thinking=True,
        )
        text = a.format_text()
        assert "..." in text


# -- extract_thinking_blocks ---------------------------------------------------

class TestExtractThinkingBlocks:
    def test_valid_response(self):
        response = {
            "content": [
                {"type": "thinking", "thinking": "I should analyze this", "token_count": 15},
                {"type": "text", "text": "Here is my answer"},
                {"type": "thinking", "thinking": "Let me verify", "token_count": 8},
            ]
        }
        blocks = extract_thinking_blocks(response)
        assert len(blocks) == 2
        assert blocks[0].text == "I should analyze this"
        assert blocks[0].token_count == 15
        assert blocks[1].text == "Let me verify"

    def test_no_thinking_blocks(self):
        response = {
            "content": [
                {"type": "text", "text": "Just a regular response"},
            ]
        }
        blocks = extract_thinking_blocks(response)
        assert blocks == []

    def test_empty_content(self):
        blocks = extract_thinking_blocks({"content": []})
        assert blocks == []

    def test_no_content_key(self):
        blocks = extract_thinking_blocks({})
        assert blocks == []

    def test_empty_response(self):
        blocks = extract_thinking_blocks({})
        assert blocks == []

    def test_non_dict_content_items(self):
        response = {"content": ["string_item", 42, None]}
        blocks = extract_thinking_blocks(response)
        assert blocks == []

    def test_text_field_fallback(self):
        """If 'thinking' key is missing, fall back to 'text' key."""
        response = {
            "content": [
                {"type": "thinking", "text": "fallback text", "token_count": 5},
            ]
        }
        blocks = extract_thinking_blocks(response)
        assert len(blocks) == 1
        assert blocks[0].text == "fallback text"

    def test_duration_ms_captured(self):
        response = {
            "content": [
                {"type": "thinking", "thinking": "hmm", "duration_ms": 42.5},
            ]
        }
        blocks = extract_thinking_blocks(response)
        assert blocks[0].duration_ms == 42.5


# -- analyze_thinking ----------------------------------------------------------

class TestAnalyzeThinking:
    def test_empty_list(self):
        a = analyze_thinking([])
        assert a.has_thinking is False
        assert a.total_tokens == 0
        assert a.total_blocks == 0

    def test_single_block(self):
        blocks = [ThinkingBlock(text="hello", token_count=10)]
        a = analyze_thinking(blocks)
        assert a.has_thinking is True
        assert a.total_tokens == 10
        assert a.total_blocks == 1
        assert a.avg_block_length == 5.0

    def test_multiple_blocks(self):
        blocks = [
            ThinkingBlock(text="ab", token_count=3),
            ThinkingBlock(text="cdef", token_count=7),
        ]
        a = analyze_thinking(blocks)
        assert a.total_tokens == 10
        assert a.total_blocks == 2
        assert a.avg_block_length == 3.0  # (2 + 4) / 2


# -- inject_thinking_into_span ------------------------------------------------

class TestInjectThinkingIntoSpan:
    def test_returns_new_span(self):
        s = _span()
        blocks = [ThinkingBlock(text="t", token_count=1)]
        new_s = inject_thinking_into_span(s, blocks)
        assert new_s is not s

    def test_original_unchanged(self):
        s = _span()
        blocks = [ThinkingBlock(text="t", token_count=1)]
        inject_thinking_into_span(s, blocks)
        assert "anthropic.thinking" not in s.attributes

    def test_thinking_data_present(self):
        s = _span()
        blocks = [ThinkingBlock(text="deep", token_count=5)]
        new_s = inject_thinking_into_span(s, blocks)
        data = new_s.attributes["anthropic.thinking"]
        assert data["has_thinking"] is True
        assert data["total_tokens"] == 5

    def test_preserves_other_attributes(self):
        s = _span(model="claude-3-opus")
        blocks = [ThinkingBlock(text="x", token_count=1)]
        new_s = inject_thinking_into_span(s, blocks)
        assert new_s.attributes["model"] == "claude-3-opus"


# -- extract_thinking_from_span ------------------------------------------------

class TestExtractThinkingFromSpan:
    def test_roundtrip(self):
        s = _span()
        blocks = [ThinkingBlock(text="thought", token_count=20, duration_ms=10.0)]
        new_s = inject_thinking_into_span(s, blocks)
        analysis = extract_thinking_from_span(new_s)
        assert analysis is not None
        assert analysis.has_thinking is True
        assert analysis.total_tokens == 20
        assert len(analysis.blocks) == 1
        assert analysis.blocks[0].text == "thought"

    def test_no_thinking_returns_none(self):
        s = _span()
        assert extract_thinking_from_span(s) is None

    def test_with_explicit_data(self):
        s = _span(**{
            "anthropic.thinking": {
                "blocks": [{"text": "ok", "token_count": 2}],
                "total_tokens": 2,
                "total_blocks": 1,
                "avg_block_length": 2.0,
                "has_thinking": True,
            }
        })
        analysis = extract_thinking_from_span(s)
        assert analysis is not None
        assert analysis.total_tokens == 2


# -- has_thinking --------------------------------------------------------------

class TestHasThinking:
    def test_true(self):
        s = _span()
        blocks = [ThinkingBlock(text="t", token_count=1)]
        new_s = inject_thinking_into_span(s, blocks)
        assert has_thinking(new_s) is True

    def test_false_no_attr(self):
        s = _span()
        assert has_thinking(s) is False

    def test_false_empty_analysis(self):
        s = _span(**{
            "anthropic.thinking": {
                "blocks": [],
                "total_tokens": 0,
                "total_blocks": 0,
                "avg_block_length": 0.0,
                "has_thinking": False,
            }
        })
        assert has_thinking(s) is False


# -- compute_thinking_overhead -------------------------------------------------

class TestComputeThinkingOverhead:
    def test_no_spans(self):
        result = compute_thinking_overhead([])
        assert result["total_thinking_tokens"] == 0
        assert result["total_spans"] == 0
        assert result["spans_with_thinking"] == 0

    def test_mixed_spans(self):
        s1 = _span("s1", output_tokens=100)
        blocks1 = [ThinkingBlock(text="a", token_count=50)]
        s1 = inject_thinking_into_span(s1, blocks1)

        s2 = _span("s2", output_tokens=200)
        # s2 has no thinking

        s3 = _span("s3", output_tokens=100)
        blocks3 = [ThinkingBlock(text="b", token_count=30)]
        s3 = inject_thinking_into_span(s3, blocks3)

        result = compute_thinking_overhead([s1, s2, s3])
        assert result["total_thinking_tokens"] == 80
        assert result["spans_with_thinking"] == 2
        assert result["total_spans"] == 3
        assert result["avg_thinking_tokens_per_span"] == 40.0
        assert result["thinking_to_output_ratio"] == 80 / 400

    def test_no_thinking_at_all(self):
        spans = [_span("a", output_tokens=50), _span("b", output_tokens=50)]
        result = compute_thinking_overhead(spans)
        assert result["total_thinking_tokens"] == 0
        assert result["avg_thinking_tokens_per_span"] == 0.0
        assert result["thinking_to_output_ratio"] == 0.0

    def test_no_output_tokens(self):
        s = _span("s1")
        blocks = [ThinkingBlock(text="t", token_count=10)]
        s = inject_thinking_into_span(s, blocks)
        result = compute_thinking_overhead([s])
        assert result["total_thinking_tokens"] == 10
        assert result["thinking_to_output_ratio"] == 0.0
