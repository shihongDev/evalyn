"""
Playground session model and scoring pipeline.

Pure-Python module for interactive evaluation sessions. Provides
dataclasses for session/turn management and functions for scoring,
comparison, formatting, and export - with no external dependencies
and no actual LLM calls.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class PlaygroundConfig:
    """Configuration for a playground session."""

    metrics: list[str] = field(default_factory=list)
    show_scores: bool = True
    history_limit: int = 50

    def as_dict(self) -> dict[str, Any]:
        return {
            "metrics": list(self.metrics),
            "show_scores": self.show_scores,
            "history_limit": self.history_limit,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlaygroundConfig:
        return cls(
            metrics=list(data.get("metrics", [])),
            show_scores=data.get("show_scores", True),
            history_limit=data.get("history_limit", 50),
        )


@dataclass
class PlaygroundTurn:
    """A single input/output turn with scores."""

    turn_id: int
    input_text: str
    output_text: str
    scores: dict[str, float] = field(default_factory=dict)
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = _now_iso()

    def as_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "input_text": self.input_text,
            "output_text": self.output_text,
            "scores": dict(self.scores),
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlaygroundTurn:
        return cls(
            turn_id=data["turn_id"],
            input_text=data["input_text"],
            output_text=data["output_text"],
            scores=dict(data.get("scores", {})),
            timestamp=data.get("timestamp", ""),
        )


@dataclass
class PlaygroundSession:
    """A playground session containing turns and configuration."""

    session_id: str
    turns: list[PlaygroundTurn] = field(default_factory=list)
    config: PlaygroundConfig = field(default_factory=PlaygroundConfig)
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = _now_iso()

    def as_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turns": [t.as_dict() for t in self.turns],
            "config": self.config.as_dict(),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlaygroundSession:
        config_data = data.get("config", {})
        config = PlaygroundConfig.from_dict(config_data) if config_data else PlaygroundConfig()
        return cls(
            session_id=data["session_id"],
            turns=[PlaygroundTurn.from_dict(t) for t in data.get("turns", [])],
            config=config,
            created_at=data.get("created_at", ""),
        )


# ---------------------------------------------------------------------------
# Session Management
# ---------------------------------------------------------------------------


def create_session(config: PlaygroundConfig | None = None) -> PlaygroundSession:
    """Create a new playground session with a UUID and timestamp."""
    return PlaygroundSession(
        session_id=str(uuid.uuid4()),
        config=config if config is not None else PlaygroundConfig(),
    )


def add_turn(
    session: PlaygroundSession,
    input_text: str,
    output_text: str,
    scores: dict[str, float],
) -> PlaygroundSession:
    """Return a new session with the turn appended (immutable)."""
    new_turn = PlaygroundTurn(
        turn_id=len(session.turns),
        input_text=input_text,
        output_text=output_text,
        scores=dict(scores),
    )
    new_turns = list(session.turns) + [new_turn]
    return PlaygroundSession(
        session_id=session.session_id,
        turns=new_turns,
        config=session.config,
        created_at=session.created_at,
    )


def get_turn_history(
    session: PlaygroundSession,
    limit: int = 0,
) -> list[PlaygroundTurn]:
    """Get recent turns from the session.

    If limit is 0, return all turns. Otherwise return the last *limit* turns.
    """
    if limit <= 0:
        return list(session.turns)
    return list(session.turns[-limit:])


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def compute_session_stats(session: PlaygroundSession) -> dict[str, Any]:
    """Compute aggregate statistics for a session.

    Returns a dict with:
      total_turns  - number of turns
      avg_scores   - per-metric average score
      best_turn    - turn_id with the highest average score (or None)
      worst_turn   - turn_id with the lowest average score (or None)
    """
    total = len(session.turns)
    if total == 0:
        return {
            "total_turns": 0,
            "avg_scores": {},
            "best_turn": None,
            "worst_turn": None,
        }

    # Accumulate scores per metric
    metric_sums: dict[str, float] = {}
    metric_counts: dict[str, int] = {}
    for turn in session.turns:
        for metric, score in turn.scores.items():
            metric_sums[metric] = metric_sums.get(metric, 0.0) + score
            metric_counts[metric] = metric_counts.get(metric, 0) + 1

    avg_scores = {
        m: metric_sums[m] / metric_counts[m] for m in metric_sums
    }

    # Best/worst by average across all metrics for that turn
    best_id: int | None = None
    worst_id: int | None = None
    best_avg = float("-inf")
    worst_avg = float("inf")

    for turn in session.turns:
        if turn.scores:
            turn_avg = sum(turn.scores.values()) / len(turn.scores)
        else:
            turn_avg = 0.0
        if turn_avg >= best_avg:
            best_avg = turn_avg
            best_id = turn.turn_id
        if turn_avg <= worst_avg:
            worst_avg = turn_avg
            worst_id = turn.turn_id

    return {
        "total_turns": total,
        "avg_scores": avg_scores,
        "best_turn": best_id,
        "worst_turn": worst_id,
    }


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_turn_display(turn: PlaygroundTurn) -> str:
    """Format a single turn for display.

    Example output:
      Turn 0
        Input:  What is 2+2?
        Output: 4
        Scores: accuracy=1.000 relevance=0.900
    """
    lines = [f"Turn {turn.turn_id}"]
    lines.append(f"  Input:  {turn.input_text}")
    lines.append(f"  Output: {turn.output_text}")
    if turn.scores:
        score_parts = [f"{m}={s:.3f}" for m, s in sorted(turn.scores.items())]
        lines.append(f"  Scores: {' '.join(score_parts)}")
    return "\n".join(lines)


def format_session_summary(session: PlaygroundSession) -> str:
    """Format a session overview with stats.

    Example output:
      Session <id>
        Created: <timestamp>
        Turns: 5
        Avg scores: accuracy=0.850 relevance=0.780
    """
    stats = compute_session_stats(session)
    lines = [f"Session {session.session_id}"]
    lines.append(f"  Created: {session.created_at}")
    lines.append(f"  Turns: {stats['total_turns']}")
    if stats["avg_scores"]:
        score_parts = [f"{m}={s:.3f}" for m, s in sorted(stats["avg_scores"].items())]
        lines.append(f"  Avg scores: {' '.join(score_parts)}")
    if stats["best_turn"] is not None:
        lines.append(f"  Best turn: {stats['best_turn']}")
    if stats["worst_turn"] is not None:
        lines.append(f"  Worst turn: {stats['worst_turn']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Comparison and Export
# ---------------------------------------------------------------------------


def compare_turns(
    turn_a: PlaygroundTurn,
    turn_b: PlaygroundTurn,
) -> dict[str, Any]:
    """Per-metric delta between two turns.

    Returns a dict mapping metric names to the difference (b - a).
    Only metrics present in both turns are compared.
    """
    all_metrics = set(turn_a.scores) | set(turn_b.scores)
    deltas: dict[str, float] = {}
    for m in sorted(all_metrics):
        score_a = turn_a.scores.get(m, 0.0)
        score_b = turn_b.scores.get(m, 0.0)
        deltas[m] = score_b - score_a
    return deltas


def export_session(session: PlaygroundSession) -> str:
    """Export the full session as a JSON string."""
    return json.dumps(session.as_dict(), indent=2)
