from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..models import Span


@dataclass
class SessionTurn:
    """A single turn in a session conversation."""

    index: int
    role: str
    content: str
    model: str = ""
    span_id: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "role": self.role,
            "content": self.content,
            "model": self.model,
            "span_id": self.span_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionTurn:
        return cls(
            index=data["index"],
            role=data["role"],
            content=data["content"],
            model=data.get("model", ""),
            span_id=data.get("span_id", ""),
        )


@dataclass
class SessionReplayPlan:
    """Plan for replaying a full session against a target model."""

    session_id: str
    turns: list[SessionTurn] = field(default_factory=list)
    target_model: str = ""
    total_turns: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turns": [t.as_dict() for t in self.turns],
            "target_model": self.target_model,
            "total_turns": self.total_turns,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionReplayPlan:
        return cls(
            session_id=data["session_id"],
            turns=[SessionTurn.from_dict(t) for t in data.get("turns", [])],
            target_model=data.get("target_model", ""),
            total_turns=data.get("total_turns", 0),
        )

    def format_text(self) -> str:
        lines: list[str] = []
        lines.append(
            f"Session replay plan: {self.session_id} - "
            f"{self.total_turns} turns to {self.target_model}"
        )
        for turn in self.turns:
            model_str = f" ({turn.model})" if turn.model else ""
            content_preview = turn.content[:60] + "..." if len(turn.content) > 60 else turn.content
            lines.append(f"  [{turn.index}] {turn.role}{model_str}: {content_preview}")
        return "\n".join(lines)


@dataclass
class TurnComparison:
    """Comparison of a single turn between original and replayed session."""

    turn_index: int
    original_content: str
    replayed_content: str
    similarity: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "original_content": self.original_content,
            "replayed_content": self.replayed_content,
            "similarity": self.similarity,
        }


@dataclass
class SessionReplayReport:
    """Aggregated report from replaying a full session."""

    session_id: str
    comparisons: list[TurnComparison] = field(default_factory=list)
    avg_similarity: float = 0.0
    total_turns: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "comparisons": [c.as_dict() for c in self.comparisons],
            "avg_similarity": self.avg_similarity,
            "total_turns": self.total_turns,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionReplayReport:
        comparisons = [
            TurnComparison(**c) if isinstance(c, dict) and "turn_index" in c else c
            for c in data.get("comparisons", [])
        ]
        return cls(
            session_id=data["session_id"],
            comparisons=comparisons,
            avg_similarity=data.get("avg_similarity", 0.0),
            total_turns=data.get("total_turns", 0),
        )

    def format_text(self) -> str:
        lines: list[str] = []
        lines.append(f"Session replay report: {self.session_id}")
        lines.append(f"  Total turns: {self.total_turns}")
        lines.append(f"  Average similarity: {self.avg_similarity:.2%}")
        lines.append("")
        for comp in self.comparisons:
            lines.append(f"  Turn {comp.turn_index}: similarity={comp.similarity:.2%}")
            orig_preview = (
                comp.original_content[:50] + "..."
                if len(comp.original_content) > 50
                else comp.original_content
            )
            replay_preview = (
                comp.replayed_content[:50] + "..."
                if len(comp.replayed_content) > 50
                else comp.replayed_content
            )
            lines.append(f"    Original: {orig_preview}")
            lines.append(f"    Replayed: {replay_preview}")
        return "\n".join(lines)


def _compute_word_similarity(a: str, b: str) -> float:
    """Compute word-overlap similarity (Jaccard index) between two texts."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a and not words_b:
        return 1.0
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def extract_session_turns(spans: list[Span], session_id: str = "") -> list[SessionTurn]:
    """Extract session turns from spans, ordered by timestamp.

    Looks for 'role' and 'content' or 'input'/'output' in span attributes.
    """
    turns: list[SessionTurn] = []

    # Sort spans by start_time for chronological ordering
    sorted_spans = sorted(spans, key=lambda s: s.start_time)

    for span in sorted_spans:
        attrs = span.attributes
        model = str(attrs.get("model", ""))

        # Try role + content pattern
        if "role" in attrs and "content" in attrs:
            turns.append(
                SessionTurn(
                    index=len(turns),
                    role=str(attrs["role"]),
                    content=str(attrs["content"]),
                    model=model,
                    span_id=span.id,
                )
            )
        # Try input/output pattern - creates two turns (user + assistant)
        elif "input" in attrs or "output" in attrs:
            if "input" in attrs:
                turns.append(
                    SessionTurn(
                        index=len(turns),
                        role="user",
                        content=str(attrs["input"]),
                        model=model,
                        span_id=span.id,
                    )
                )
            if "output" in attrs:
                turns.append(
                    SessionTurn(
                        index=len(turns),
                        role="assistant",
                        content=str(attrs["output"]),
                        model=model,
                        span_id=span.id,
                    )
                )

    return turns


def build_session_replay_plan(turns: list[SessionTurn], target_model: str) -> SessionReplayPlan:
    """Build a session replay plan from extracted turns."""
    session_id = ""
    if turns:
        session_id = turns[0].span_id or "unknown"
    return SessionReplayPlan(
        session_id=session_id,
        turns=list(turns),
        target_model=target_model,
        total_turns=len(turns),
    )


def compare_turn(original: SessionTurn, replayed_content: str) -> TurnComparison:
    """Compare a single turn between original and replayed content."""
    similarity = _compute_word_similarity(original.content, replayed_content)
    return TurnComparison(
        turn_index=original.index,
        original_content=original.content,
        replayed_content=replayed_content,
        similarity=similarity,
    )


def build_session_replay_report(
    session_id: str, comparisons: list[TurnComparison]
) -> SessionReplayReport:
    """Aggregate turn comparisons into a session replay report."""
    if not comparisons:
        return SessionReplayReport(session_id=session_id)

    avg_sim = sum(c.similarity for c in comparisons) / len(comparisons)

    return SessionReplayReport(
        session_id=session_id,
        comparisons=list(comparisons),
        avg_similarity=avg_sim,
        total_turns=len(comparisons),
    )
