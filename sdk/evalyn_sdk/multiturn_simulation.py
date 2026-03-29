"""
Multi-turn conversation simulation: generate full conversations for evaluation.

Provides a framework for generating multi-turn conversations with configurable
flow patterns (clarification, topic shift, error recovery, deepening).
Pure Python - no external deps, no LLM calls.
"""

from __future__ import annotations

import random
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""

    role: str  # "user" / "assistant"
    content: str
    turn_number: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "turn_number": self.turn_number,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ConversationTurn:
        return cls(
            role=data["role"],
            content=data["content"],
            turn_number=data["turn_number"],
        )


@dataclass
class ConversationFlow:
    """A conversation flow pattern that governs turn progression."""

    flow_id: str
    pattern: str  # "clarification" / "topic_shift" / "error_recovery" / "deepening"
    description: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "flow_id": self.flow_id,
            "pattern": self.pattern,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ConversationFlow:
        return cls(
            flow_id=data["flow_id"],
            pattern=data["pattern"],
            description=data["description"],
        )


@dataclass
class ConversationConfig:
    """Configuration for generating a conversation."""

    min_turns: int = 2
    max_turns: int = 6
    flow_pattern: str = "clarification"
    topic: str = ""
    seed_message: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "min_turns": self.min_turns,
            "max_turns": self.max_turns,
            "flow_pattern": self.flow_pattern,
            "topic": self.topic,
            "seed_message": self.seed_message,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ConversationConfig:
        return cls(
            min_turns=data.get("min_turns", 2),
            max_turns=data.get("max_turns", 6),
            flow_pattern=data.get("flow_pattern", "clarification"),
            topic=data.get("topic", ""),
            seed_message=data.get("seed_message", ""),
        )


@dataclass
class SimulatedConversation:
    """A generated multi-turn conversation."""

    conversation_id: str
    turns: List[ConversationTurn]
    flow_pattern: str
    topic: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "turns": [t.as_dict() for t in self.turns],
            "flow_pattern": self.flow_pattern,
            "topic": self.topic,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SimulatedConversation:
        return cls(
            conversation_id=data["conversation_id"],
            turns=[ConversationTurn.from_dict(t) for t in data.get("turns", [])],
            flow_pattern=data.get("flow_pattern", ""),
            topic=data.get("topic", ""),
            metadata=data.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# Flow Patterns
# ---------------------------------------------------------------------------


FLOW_PATTERNS: Dict[str, ConversationFlow] = {
    "clarification": ConversationFlow(
        flow_id="clarification",
        pattern="clarification",
        description="User asks for clarification on the assistant's response",
    ),
    "topic_shift": ConversationFlow(
        flow_id="topic_shift",
        pattern="topic_shift",
        description="User shifts to a related but different topic",
    ),
    "error_recovery": ConversationFlow(
        flow_id="error_recovery",
        pattern="error_recovery",
        description="User corrects a misunderstanding or error in the response",
    ),
    "deepening": ConversationFlow(
        flow_id="deepening",
        pattern="deepening",
        description="User asks to go deeper on a specific aspect",
    ),
}


# ---------------------------------------------------------------------------
# Template banks for generating varied content
# ---------------------------------------------------------------------------

_CLARIFICATION_TEMPLATES = [
    "Can you explain more about {aspect}?",
    "What do you mean by {aspect}?",
    "Could you clarify the part about {aspect}?",
    "I'm not sure I understand {aspect}. Can you elaborate?",
    "Can you break down {aspect} further?",
]

_TOPIC_SHIFT_TEMPLATES = [
    "That's helpful. Now, about {aspect} -",
    "Thanks. Switching gears, what about {aspect}?",
    "Got it. Moving on to {aspect} -",
    "Understood. How does {aspect} relate to this?",
    "OK. Let's talk about {aspect} instead.",
]

_ERROR_RECOVERY_TEMPLATES = [
    "That doesn't seem right. I meant {aspect}.",
    "Actually, I was asking about {aspect}, not that.",
    "I think there's a misunderstanding. I need {aspect}.",
    "Not quite - let me rephrase. I want {aspect}.",
    "That's not what I meant. I was referring to {aspect}.",
]

_DEEPENING_TEMPLATES = [
    "Can you go deeper on {aspect}?",
    "Tell me more details about {aspect}.",
    "What are the specifics of {aspect}?",
    "Dive deeper into {aspect} please.",
    "Expand on {aspect} - I want the details.",
]

_TEMPLATE_BANKS = {
    "clarification": _CLARIFICATION_TEMPLATES,
    "topic_shift": _TOPIC_SHIFT_TEMPLATES,
    "error_recovery": _ERROR_RECOVERY_TEMPLATES,
    "deepening": _DEEPENING_TEMPLATES,
}

_DEFAULT_TOPICS = [
    "the main concept",
    "the implementation details",
    "the trade-offs",
    "how this works in practice",
    "the edge cases",
    "performance considerations",
    "the underlying mechanism",
]

_ASSISTANT_RESPONSES = [
    "That's a great question. Here's what I can tell you about {topic}: "
    "the key points involve understanding the fundamentals and applying them.",
    "Let me explain. When it comes to {topic}, there are several factors to consider. "
    "The most important aspect is consistency in approach.",
    "Sure. Regarding {topic}, the main idea is to break it down into smaller parts "
    "and address each one systematically.",
    "Of course. {topic} is best understood by looking at concrete examples. "
    "Consider a scenario where you need to handle multiple cases.",
    "Good question. {topic} requires careful consideration of context. "
    "The approach depends on your specific requirements.",
]

_SEED_MESSAGES = [
    "Can you help me understand {topic}?",
    "I have a question about {topic}.",
    "What can you tell me about {topic}?",
    "I need help with {topic}.",
    "Explain {topic} to me.",
]


# ---------------------------------------------------------------------------
# Core Functions
# ---------------------------------------------------------------------------


def generate_follow_up(
    turn: ConversationTurn, flow: ConversationFlow, rng: random.Random
) -> str:
    """Generate a follow-up user message based on the flow pattern.

    Uses the previous assistant turn's content to pick a relevant aspect,
    then applies the flow pattern template.
    """
    templates = _TEMPLATE_BANKS.get(flow.pattern, _CLARIFICATION_TEMPLATES)
    template = rng.choice(templates)

    # Extract an aspect from the previous turn content
    words = turn.content.split()
    if len(words) > 4:
        start = rng.randint(0, max(0, len(words) - 4))
        aspect = " ".join(words[start : start + 3])
    else:
        aspect = rng.choice(_DEFAULT_TOPICS)

    return template.format(aspect=aspect)


def generate_conversation(
    config: ConversationConfig, rng: random.Random | None = None
) -> SimulatedConversation:
    """Generate a full simulated conversation.

    Creates alternating user/assistant turns following the configured
    flow pattern. The conversation length is randomly chosen between
    min_turns and max_turns.
    """
    if rng is None:
        rng = random.Random()

    flow = FLOW_PATTERNS.get(config.flow_pattern)
    if flow is None:
        flow = FLOW_PATTERNS["clarification"]

    topic = config.topic or rng.choice(_DEFAULT_TOPICS)
    num_turns = rng.randint(config.min_turns, config.max_turns)
    # Ensure even number of turns (user + assistant pairs)
    if num_turns % 2 != 0:
        num_turns += 1

    turns: List[ConversationTurn] = []

    # First user turn
    if config.seed_message:
        first_msg = config.seed_message
    else:
        first_msg = rng.choice(_SEED_MESSAGES).format(topic=topic)

    turns.append(ConversationTurn(role="user", content=first_msg, turn_number=0))

    for i in range(1, num_turns):
        if i % 2 == 1:
            # Assistant turn
            response = rng.choice(_ASSISTANT_RESPONSES).format(topic=topic)
            turns.append(
                ConversationTurn(role="assistant", content=response, turn_number=i)
            )
        else:
            # User follow-up based on flow pattern
            prev_turn = turns[-1]
            follow_up = generate_follow_up(prev_turn, flow, rng)
            turns.append(
                ConversationTurn(role="user", content=follow_up, turn_number=i)
            )

    conversation_id = str(uuid.uuid4())
    return SimulatedConversation(
        conversation_id=conversation_id,
        turns=turns,
        flow_pattern=config.flow_pattern,
        topic=topic,
        metadata={
            "num_turns": len(turns),
            "min_turns": config.min_turns,
            "max_turns": config.max_turns,
        },
    )


def generate_batch(
    configs: List[ConversationConfig],
) -> List[SimulatedConversation]:
    """Batch generate conversations from a list of configs."""
    return [generate_conversation(cfg) for cfg in configs]


def conversation_to_dataset_items(conv: SimulatedConversation) -> List[Dict[str, Any]]:
    """Convert a conversation to evalyn dataset items (one per turn pair).

    Each dataset item represents one user-assistant exchange, with the
    conversation history up to that point included as context.
    """
    items: List[Dict[str, Any]] = []
    history: List[Dict[str, str]] = []

    for i in range(0, len(conv.turns) - 1, 2):
        user_turn = conv.turns[i]
        if i + 1 >= len(conv.turns):
            break
        assistant_turn = conv.turns[i + 1]

        item = {
            "input": user_turn.content,
            "expected_output": assistant_turn.content,
            "conversation_id": conv.conversation_id,
            "turn_pair": i // 2,
            "flow_pattern": conv.flow_pattern,
            "topic": conv.topic,
            "history": list(history),
        }
        items.append(item)

        history.append({"role": "user", "content": user_turn.content})
        history.append({"role": "assistant", "content": assistant_turn.content})

    return items


def format_conversation(conv: SimulatedConversation) -> str:
    """Format a conversation as a human-readable string."""
    lines = [
        f"Conversation: {conv.conversation_id}",
        f"Pattern: {conv.flow_pattern}",
        f"Topic: {conv.topic}",
        f"Turns: {len(conv.turns)}",
        "",
    ]
    for turn in conv.turns:
        role_label = "User" if turn.role == "user" else "Assistant"
        lines.append(f"[{turn.turn_number}] {role_label}: {turn.content}")
    return "\n".join(lines)


def get_conversation_stats(
    conversations: List[SimulatedConversation],
) -> Dict[str, Any]:
    """Compute statistics over a batch of conversations.

    Returns avg turns, turn distribution, and pattern distribution.
    """
    if not conversations:
        return {
            "count": 0,
            "avg_turns": 0.0,
            "turn_distribution": {},
            "pattern_distribution": {},
        }

    turn_counts = [len(c.turns) for c in conversations]
    pattern_counter: Counter[str] = Counter(c.flow_pattern for c in conversations)
    turn_counter: Counter[int] = Counter(turn_counts)

    return {
        "count": len(conversations),
        "avg_turns": sum(turn_counts) / len(turn_counts),
        "turn_distribution": dict(turn_counter),
        "pattern_distribution": dict(pattern_counter),
    }
