from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..models import Span


@dataclass
class ThinkingBlock:
    """A single thinking/reasoning block from a Claude response."""

    text: str
    token_count: int = 0
    duration_ms: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "token_count": self.token_count,
            "duration_ms": self.duration_ms,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ThinkingBlock:
        return cls(
            text=data.get("text", ""),
            token_count=data.get("token_count", 0),
            duration_ms=data.get("duration_ms", 0.0),
        )


@dataclass
class ThinkingAnalysis:
    """Aggregate analysis of thinking blocks."""

    blocks: List[ThinkingBlock] = field(default_factory=list)
    total_tokens: int = 0
    total_blocks: int = 0
    avg_block_length: float = 0.0
    has_thinking: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "blocks": [b.as_dict() for b in self.blocks],
            "total_tokens": self.total_tokens,
            "total_blocks": self.total_blocks,
            "avg_block_length": self.avg_block_length,
            "has_thinking": self.has_thinking,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ThinkingAnalysis:
        return cls(
            blocks=[ThinkingBlock.from_dict(b) for b in data.get("blocks", [])],
            total_tokens=data.get("total_tokens", 0),
            total_blocks=data.get("total_blocks", 0),
            avg_block_length=data.get("avg_block_length", 0.0),
            has_thinking=data.get("has_thinking", False),
        )

    def format_text(self) -> str:
        """Format a human-readable summary of the thinking analysis."""
        if not self.has_thinking:
            return "No thinking blocks found."
        lines = [
            f"Thinking: {self.total_blocks} block(s), "
            f"{self.total_tokens} tokens, "
            f"avg length {self.avg_block_length:.1f} chars",
        ]
        for i, block in enumerate(self.blocks):
            preview = block.text[:80]
            if len(block.text) > 80:
                preview += "..."
            lines.append(f"  [{i + 1}] ({block.token_count} tok) {preview}")
        return "\n".join(lines)


def extract_thinking_blocks(response_data: Dict[str, Any]) -> List[ThinkingBlock]:
    """Extract thinking blocks from a Claude API response.

    Looks for content items with type == "thinking" in response_data["content"].
    """
    content = response_data.get("content", [])
    blocks: List[ThinkingBlock] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "thinking":
            continue
        text = item.get("thinking", item.get("text", ""))
        token_count = item.get("token_count", 0)
        duration_ms = item.get("duration_ms", 0.0)
        blocks.append(ThinkingBlock(
            text=text,
            token_count=token_count,
            duration_ms=duration_ms,
        ))
    return blocks


def analyze_thinking(blocks: List[ThinkingBlock]) -> ThinkingAnalysis:
    """Compute aggregate statistics from a list of thinking blocks."""
    if not blocks:
        return ThinkingAnalysis()
    total_tokens = sum(b.token_count for b in blocks)
    total_blocks = len(blocks)
    total_length = sum(len(b.text) for b in blocks)
    avg_length = total_length / total_blocks if total_blocks else 0.0
    return ThinkingAnalysis(
        blocks=list(blocks),
        total_tokens=total_tokens,
        total_blocks=total_blocks,
        avg_block_length=avg_length,
        has_thinking=True,
    )


def inject_thinking_into_span(
    span: Span, blocks: List[ThinkingBlock]
) -> Span:
    """Add thinking blocks to span attributes under 'anthropic.thinking'.

    Returns a new Span - does not mutate the original.
    """
    new_span = copy.deepcopy(span)
    analysis = analyze_thinking(blocks)
    new_span.attributes["anthropic.thinking"] = analysis.as_dict()
    return new_span


def extract_thinking_from_span(span: Span) -> Optional[ThinkingAnalysis]:
    """Extract ThinkingAnalysis from a span's attributes.

    Returns None if the span has no thinking data.
    """
    raw = span.attributes.get("anthropic.thinking")
    if raw is None:
        return None
    return ThinkingAnalysis.from_dict(raw)


def has_thinking(span: Span) -> bool:
    """Quick check whether a span contains thinking data."""
    raw = span.attributes.get("anthropic.thinking")
    if raw is None:
        return False
    return raw.get("has_thinking", False)


def compute_thinking_overhead(spans: List[Span]) -> Dict[str, Any]:
    """Aggregate thinking statistics across multiple spans.

    Returns a dict with:
    - total_thinking_tokens: sum of thinking tokens across all spans
    - avg_thinking_tokens_per_span: average thinking tokens per span with thinking
    - thinking_to_output_ratio: ratio of thinking tokens to output tokens
    - spans_with_thinking: count of spans that have thinking
    - total_spans: total number of spans examined
    """
    total_thinking_tokens = 0
    spans_with_thinking = 0
    total_output_tokens = 0

    for s in spans:
        analysis = extract_thinking_from_span(s)
        if analysis is not None and analysis.has_thinking:
            total_thinking_tokens += analysis.total_tokens
            spans_with_thinking += 1
        # Output tokens from span attributes (common convention)
        total_output_tokens += s.attributes.get("output_tokens", 0)

    avg_per_span = (
        total_thinking_tokens / spans_with_thinking
        if spans_with_thinking
        else 0.0
    )
    ratio = (
        total_thinking_tokens / total_output_tokens
        if total_output_tokens
        else 0.0
    )

    return {
        "total_thinking_tokens": total_thinking_tokens,
        "avg_thinking_tokens_per_span": avg_per_span,
        "thinking_to_output_ratio": ratio,
        "spans_with_thinking": spans_with_thinking,
        "total_spans": len(spans),
    }
