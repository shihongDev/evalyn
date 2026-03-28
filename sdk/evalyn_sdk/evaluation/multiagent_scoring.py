"""Multi-agent communication scoring.

Evaluate quality of inter-agent communication by scoring relevance,
clarity, and information density of messages exchanged between agents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class AgentMessage:
    """Single message in a multi-agent conversation."""

    agent_id: str
    content: str
    timestamp_ms: float = 0.0
    message_type: str = "text"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "content": self.content,
            "timestamp_ms": self.timestamp_ms,
            "message_type": self.message_type,
        }


@dataclass
class MessageScore:
    """Quality scores for a single message."""

    agent_id: str
    relevance: float  # 0-1
    clarity: float  # 0-1
    information_density: float  # 0-1
    composite: float  # 0-1

    def as_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "relevance": round(self.relevance, 4),
            "clarity": round(self.clarity, 4),
            "information_density": round(self.information_density, 4),
            "composite": round(self.composite, 4),
        }


@dataclass
class CommunicationReport:
    """Full communication quality report."""

    messages: List[AgentMessage] = field(default_factory=list)
    scores: List[MessageScore] = field(default_factory=list)
    avg_composite: float = 0.0
    collaborative_efficiency: float = 0.0
    total_messages: int = 0
    unique_agents: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "messages": [m.as_dict() for m in self.messages],
            "scores": [s.as_dict() for s in self.scores],
            "avg_composite": round(self.avg_composite, 4),
            "collaborative_efficiency": round(self.collaborative_efficiency, 4),
            "total_messages": self.total_messages,
            "unique_agents": self.unique_agents,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CommunicationReport:
        messages = [
            AgentMessage(
                agent_id=m["agent_id"],
                content=m["content"],
                timestamp_ms=m.get("timestamp_ms", 0.0),
                message_type=m.get("message_type", "text"),
            )
            for m in data.get("messages", [])
        ]
        scores = [
            MessageScore(
                agent_id=s["agent_id"],
                relevance=s["relevance"],
                clarity=s["clarity"],
                information_density=s["information_density"],
                composite=s["composite"],
            )
            for s in data.get("scores", [])
        ]
        return cls(
            messages=messages,
            scores=scores,
            avg_composite=data.get("avg_composite", 0.0),
            collaborative_efficiency=data.get("collaborative_efficiency", 0.0),
            total_messages=data.get("total_messages", 0),
            unique_agents=data.get("unique_agents", 0),
        )

    def format_text(self) -> str:
        lines = [
            "Communication Report",
            f"  Total messages:          {self.total_messages}",
            f"  Unique agents:           {self.unique_agents}",
            f"  Avg composite score:     {self.avg_composite:.1%}",
            f"  Collaborative efficiency:{self.collaborative_efficiency:.1%}",
        ]
        for s in self.scores:
            lines.append(
                f"  {s.agent_id}: rel={s.relevance:.2f}"
                f"  clar={s.clarity:.2f}"
                f"  dens={s.information_density:.2f}"
                f"  comp={s.composite:.2f}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

_TRIVIAL_MESSAGES = {"ok", "yes", "no", "done", "thanks", "sure", "agreed", "ack"}


def _tokenize(text: str) -> List[str]:
    """Simple whitespace tokenizer, lowercased."""
    return text.lower().split()


def score_relevance(message: AgentMessage, context: List[AgentMessage]) -> float:
    """Word overlap between message and recent context messages.

    Higher overlap means more relevant. Returns 0.0 when context is empty.
    """
    if not context:
        return 0.0
    msg_words = set(_tokenize(message.content))
    if not msg_words:
        return 0.0
    context_words: set[str] = set()
    for m in context:
        context_words.update(_tokenize(m.content))
    if not context_words:
        return 0.0
    overlap = len(msg_words & context_words)
    return min(overlap / len(msg_words), 1.0)


def score_clarity(message: AgentMessage) -> float:
    """Heuristic clarity score based on message length.

    Very short (< 10 chars) and very long (> 2000 chars) messages score
    lower. Score peaks around 100-500 chars.
    """
    length = len(message.content)
    if length == 0:
        return 0.0
    if length < 10:
        return length / 10.0
    if length <= 100:
        # Ramp from 0.8 to 1.0
        return 0.8 + 0.2 * ((length - 10) / 90.0)
    if length <= 500:
        return 1.0
    if length <= 2000:
        # Gradual decline from 1.0 to 0.5
        return 1.0 - 0.5 * ((length - 500) / 1500.0)
    # Beyond 2000: further decline, floor at 0.2
    excess = length - 2000
    score = 0.5 - 0.3 * min(excess / 3000.0, 1.0)
    return max(score, 0.2)


def score_information_density(message: AgentMessage) -> float:
    """Unique words / total words ratio.

    Higher ratio means more information-dense. Returns 0.0 for empty.
    """
    words = _tokenize(message.content)
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def score_message(message: AgentMessage, context: List[AgentMessage]) -> MessageScore:
    """Score a single message. Composite = average of relevance, clarity, density."""
    rel = score_relevance(message, context)
    clar = score_clarity(message)
    dens = score_information_density(message)
    composite = (rel + clar + dens) / 3.0
    return MessageScore(
        agent_id=message.agent_id,
        relevance=rel,
        clarity=clar,
        information_density=dens,
        composite=composite,
    )


def compute_collaborative_efficiency(messages: List[AgentMessage]) -> float:
    """Fraction of messages that are substantive.

    Substantive: > 20 chars and not a trivial message like "ok" or "yes".
    Returns 0.0 for empty input.
    """
    if not messages:
        return 0.0
    substantive = 0
    for m in messages:
        stripped = m.content.strip().lower()
        if len(m.content) > 20 and stripped not in _TRIVIAL_MESSAGES:
            substantive += 1
    return substantive / len(messages)


def evaluate_communication(messages: List[AgentMessage]) -> CommunicationReport:
    """Full evaluation of multi-agent communication."""
    if not messages:
        return CommunicationReport()

    scores: List[MessageScore] = []
    for i, msg in enumerate(messages):
        context = messages[:i]  # all prior messages
        scores.append(score_message(msg, context))

    avg_composite = sum(s.composite for s in scores) / len(scores) if scores else 0.0
    collab_eff = compute_collaborative_efficiency(messages)
    unique_agents = len({m.agent_id for m in messages})

    return CommunicationReport(
        messages=list(messages),
        scores=scores,
        avg_composite=avg_composite,
        collaborative_efficiency=collab_eff,
        total_messages=len(messages),
        unique_agents=unique_agents,
    )
