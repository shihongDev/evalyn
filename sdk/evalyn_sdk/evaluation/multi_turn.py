from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..models import Span


@dataclass
class Turn:
    index: int
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Turn:
        ts = data.get("timestamp")
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        return cls(
            index=data["index"],
            role=data["role"],
            content=data["content"],
            timestamp=ts,
        )


@dataclass
class TurnScore:
    turn_index: int
    score: float
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "score": self.score,
            "passed": self.passed,
            "details": self.details,
        }


@dataclass
class ConversationMetrics:
    turn_count: int
    avg_turn_score: float
    min_turn_score: float
    max_turn_score: float
    context_consistency: float  # 0-1
    topic_drift_score: float  # 0-1

    def as_dict(self) -> dict[str, Any]:
        return {
            "turn_count": self.turn_count,
            "avg_turn_score": self.avg_turn_score,
            "min_turn_score": self.min_turn_score,
            "max_turn_score": self.max_turn_score,
            "context_consistency": self.context_consistency,
            "topic_drift_score": self.topic_drift_score,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConversationMetrics:
        return cls(
            turn_count=data["turn_count"],
            avg_turn_score=data["avg_turn_score"],
            min_turn_score=data["min_turn_score"],
            max_turn_score=data["max_turn_score"],
            context_consistency=data["context_consistency"],
            topic_drift_score=data["topic_drift_score"],
        )


@dataclass
class MultiTurnReport:
    conversation_id: str
    turns: list[Turn]
    turn_scores: list[TurnScore]
    metrics: ConversationMetrics

    def as_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "turns": [t.as_dict() for t in self.turns],
            "turn_scores": [s.as_dict() for s in self.turn_scores],
            "metrics": self.metrics.as_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MultiTurnReport:
        return cls(
            conversation_id=data["conversation_id"],
            turns=[Turn.from_dict(t) for t in data["turns"]],
            turn_scores=[
                TurnScore(
                    turn_index=s["turn_index"],
                    score=s["score"],
                    passed=s["passed"],
                    details=s.get("details", {}),
                )
                for s in data["turn_scores"]
            ],
            metrics=ConversationMetrics.from_dict(data["metrics"]),
        )

    def format_text(self) -> str:
        lines: list[str] = []
        lines.append(f"Conversation: {self.conversation_id}")
        lines.append(f"Turns: {self.metrics.turn_count}")
        lines.append(
            f"Avg score: {self.metrics.avg_turn_score:.2f}  "
            f"Min: {self.metrics.min_turn_score:.2f}  "
            f"Max: {self.metrics.max_turn_score:.2f}"
        )
        lines.append(f"Context consistency: {self.metrics.context_consistency:.2f}")
        lines.append(f"Topic drift: {self.metrics.topic_drift_score:.2f}")
        lines.append("")
        for ts in self.turn_scores:
            status = "PASS" if ts.passed else "FAIL"
            lines.append(f"  Turn {ts.turn_index}: {ts.score:.2f} [{status}]")
        return "\n".join(lines)


def _tokenize(text: str) -> set:
    """Simple word tokenizer for overlap calculations."""
    return set(text.lower().split())


def extract_turns(spans: list[Span]) -> list[Turn]:
    """Extract turns from span attributes.

    Looks for attributes with "role" and "content" keys, or
    "input"/"output" pairs. Orders by timestamp.
    """
    turns: list[Turn] = []
    idx = 0

    # Sort spans by start_time for consistent ordering
    sorted_spans = sorted(spans, key=lambda s: s.start_time)

    for span in sorted_spans:
        attrs = span.attributes

        if "role" in attrs and "content" in attrs:
            turns.append(
                Turn(
                    index=idx,
                    role=attrs["role"],
                    content=str(attrs["content"]),
                    timestamp=span.start_time,
                )
            )
            idx += 1
        elif "input" in attrs or "output" in attrs:
            if "input" in attrs:
                turns.append(
                    Turn(
                        index=idx,
                        role="user",
                        content=str(attrs["input"]),
                        timestamp=span.start_time,
                    )
                )
                idx += 1
            if "output" in attrs:
                turns.append(
                    Turn(
                        index=idx,
                        role="assistant",
                        content=str(attrs["output"]),
                        timestamp=span.start_time,
                    )
                )
                idx += 1

    return turns


def score_turn_quality(turn: Turn, context: list[Turn]) -> TurnScore:
    """Score a single turn using simple heuristics.

    - Length-based quality: content must be >= 10 chars to be non-trivial
    - Context reference: check if turn references prior context words
    """
    checks: dict[str, bool] = {}

    # Length check
    is_nontrivial = len(turn.content.strip()) >= 10
    checks["nontrivial_length"] = is_nontrivial

    # Context reference check
    references_context = False
    if context:
        turn_words = _tokenize(turn.content)
        for prev in context:
            prev_words = _tokenize(prev.content)
            overlap = turn_words & prev_words
            # Filter out very common short words
            meaningful_overlap = {w for w in overlap if len(w) > 3}
            if meaningful_overlap:
                references_context = True
                break
    checks["references_context"] = references_context

    # Score: average of checks (with context reference only counted if there is context)
    if context:
        score = (float(is_nontrivial) + float(references_context)) / 2.0
    else:
        score = float(is_nontrivial)

    passed = score >= 0.5
    return TurnScore(
        turn_index=turn.index,
        score=score,
        passed=passed,
        details={"checks": checks},
    )


def evaluate_context_consistency(turns: list[Turn]) -> float:
    """Measure how well later turns reference/build on earlier ones.

    Uses word overlap ratio between consecutive assistant turns.
    Returns the average overlap ratio (0-1).
    """
    assistant_turns = [t for t in turns if t.role == "assistant"]
    if len(assistant_turns) < 2:
        return 1.0  # Single or no assistant turns: fully consistent by default

    overlaps: list[float] = []
    for i in range(1, len(assistant_turns)):
        prev_words = _tokenize(assistant_turns[i - 1].content)
        curr_words = _tokenize(assistant_turns[i].content)
        if not prev_words or not curr_words:
            overlaps.append(0.0)
            continue
        union = prev_words | curr_words
        intersection = prev_words & curr_words
        overlaps.append(len(intersection) / len(union) if union else 0.0)

    return sum(overlaps) / len(overlaps)


def evaluate_topic_drift(turns: list[Turn]) -> float:
    """Measure how much the topic shifts over the conversation.

    Compares first turn to last turn word overlap.
    Higher overlap means lower drift.
    Returns 0-1 where 0 = no drift, 1 = complete drift.
    """
    if len(turns) < 2:
        return 0.0  # No drift possible with fewer than 2 turns

    first_words = _tokenize(turns[0].content)
    last_words = _tokenize(turns[-1].content)

    if not first_words or not last_words:
        return 1.0  # Empty content: maximum drift

    union = first_words | last_words
    intersection = first_words & last_words
    overlap = len(intersection) / len(union) if union else 0.0

    # Invert: high overlap = low drift
    return 1.0 - overlap


def evaluate_conversation(
    conversation_id: str, turns: list[Turn]
) -> MultiTurnReport:
    """Full multi-turn conversation evaluation."""
    turn_scores: list[TurnScore] = []
    for i, turn in enumerate(turns):
        context = turns[:i]
        ts = score_turn_quality(turn, context)
        turn_scores.append(ts)

    scores = [ts.score for ts in turn_scores]
    if scores:
        avg_score = sum(scores) / len(scores)
        min_score = min(scores)
        max_score = max(scores)
    else:
        avg_score = 0.0
        min_score = 0.0
        max_score = 0.0

    context_consistency = evaluate_context_consistency(turns)
    topic_drift = evaluate_topic_drift(turns)

    metrics = ConversationMetrics(
        turn_count=len(turns),
        avg_turn_score=avg_score,
        min_turn_score=min_score,
        max_turn_score=max_score,
        context_consistency=context_consistency,
        topic_drift_score=topic_drift,
    )

    return MultiTurnReport(
        conversation_id=conversation_id,
        turns=turns,
        turn_scores=turn_scores,
        metrics=metrics,
    )
