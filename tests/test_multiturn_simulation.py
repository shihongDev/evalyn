"""Tests for multi-turn conversation simulation module."""

from __future__ import annotations

import random

from evalyn_sdk.multiturn_simulation import (
    FLOW_PATTERNS,
    ConversationConfig,
    ConversationFlow,
    ConversationTurn,
    SimulatedConversation,
    conversation_to_dataset_items,
    format_conversation,
    generate_batch,
    generate_conversation,
    generate_follow_up,
    get_conversation_stats,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_turn(**overrides) -> ConversationTurn:
    defaults = {"role": "assistant", "content": "Some response text here", "turn_number": 1}
    defaults.update(overrides)
    return ConversationTurn(**defaults)


def _make_flow(**overrides) -> ConversationFlow:
    defaults = {
        "flow_id": "clarification",
        "pattern": "clarification",
        "description": "Ask for clarification",
    }
    defaults.update(overrides)
    return ConversationFlow(**defaults)


def _make_config(**overrides) -> ConversationConfig:
    defaults = {
        "min_turns": 2,
        "max_turns": 6,
        "flow_pattern": "clarification",
        "topic": "testing",
        "seed_message": "",
    }
    defaults.update(overrides)
    return ConversationConfig(**defaults)


def _fixed_rng(seed: int = 42) -> random.Random:
    return random.Random(seed)


# ---------------------------------------------------------------------------
# ConversationTurn
# ---------------------------------------------------------------------------


class TestConversationTurn:
    def test_as_dict_roundtrip(self):
        t = _make_turn()
        d = t.as_dict()
        t2 = ConversationTurn.from_dict(d)
        assert t2.role == t.role
        assert t2.content == t.content
        assert t2.turn_number == t.turn_number

    def test_as_dict_fields(self):
        t = _make_turn(role="user", content="hello", turn_number=0)
        d = t.as_dict()
        assert d["role"] == "user"
        assert d["content"] == "hello"
        assert d["turn_number"] == 0

    def test_from_dict_values(self):
        t = ConversationTurn.from_dict(
            {"role": "user", "content": "test", "turn_number": 3}
        )
        assert t.role == "user"
        assert t.content == "test"
        assert t.turn_number == 3

    def test_user_role(self):
        t = _make_turn(role="user")
        assert t.role == "user"

    def test_assistant_role(self):
        t = _make_turn(role="assistant")
        assert t.role == "assistant"


# ---------------------------------------------------------------------------
# ConversationFlow
# ---------------------------------------------------------------------------


class TestConversationFlow:
    def test_as_dict_roundtrip(self):
        f = _make_flow()
        d = f.as_dict()
        f2 = ConversationFlow.from_dict(d)
        assert f2.flow_id == f.flow_id
        assert f2.pattern == f.pattern
        assert f2.description == f.description

    def test_as_dict_fields(self):
        f = _make_flow(flow_id="x", pattern="deepening", description="Go deeper")
        d = f.as_dict()
        assert d["flow_id"] == "x"
        assert d["pattern"] == "deepening"

    def test_from_dict(self):
        f = ConversationFlow.from_dict(
            {"flow_id": "a", "pattern": "topic_shift", "description": "shift"}
        )
        assert f.pattern == "topic_shift"


# ---------------------------------------------------------------------------
# ConversationConfig
# ---------------------------------------------------------------------------


class TestConversationConfig:
    def test_defaults(self):
        c = ConversationConfig()
        assert c.min_turns == 2
        assert c.max_turns == 6
        assert c.flow_pattern == "clarification"
        assert c.topic == ""
        assert c.seed_message == ""

    def test_as_dict_roundtrip(self):
        c = _make_config(topic="math", seed_message="What is 2+2?")
        d = c.as_dict()
        c2 = ConversationConfig.from_dict(d)
        assert c2.topic == "math"
        assert c2.seed_message == "What is 2+2?"
        assert c2.min_turns == c.min_turns

    def test_from_dict_defaults(self):
        c = ConversationConfig.from_dict({})
        assert c.min_turns == 2
        assert c.max_turns == 6

    def test_custom_values(self):
        c = _make_config(min_turns=4, max_turns=10)
        assert c.min_turns == 4
        assert c.max_turns == 10


# ---------------------------------------------------------------------------
# SimulatedConversation
# ---------------------------------------------------------------------------


class TestSimulatedConversation:
    def test_as_dict_roundtrip(self):
        turns = [
            ConversationTurn(role="user", content="hi", turn_number=0),
            ConversationTurn(role="assistant", content="hello", turn_number=1),
        ]
        conv = SimulatedConversation(
            conversation_id="abc",
            turns=turns,
            flow_pattern="clarification",
            topic="greetings",
            metadata={"key": "value"},
        )
        d = conv.as_dict()
        conv2 = SimulatedConversation.from_dict(d)
        assert conv2.conversation_id == "abc"
        assert len(conv2.turns) == 2
        assert conv2.turns[0].role == "user"
        assert conv2.flow_pattern == "clarification"
        assert conv2.metadata == {"key": "value"}

    def test_as_dict_turns_serialized(self):
        turns = [ConversationTurn(role="user", content="q", turn_number=0)]
        conv = SimulatedConversation(
            conversation_id="x",
            turns=turns,
            flow_pattern="deepening",
            topic="test",
        )
        d = conv.as_dict()
        assert isinstance(d["turns"], list)
        assert d["turns"][0]["role"] == "user"

    def test_from_dict_empty_turns(self):
        conv = SimulatedConversation.from_dict(
            {"conversation_id": "x", "turns": [], "flow_pattern": "clarification"}
        )
        assert conv.turns == []

    def test_metadata_default(self):
        conv = SimulatedConversation(
            conversation_id="x",
            turns=[],
            flow_pattern="clarification",
            topic="test",
        )
        assert conv.metadata == {}


# ---------------------------------------------------------------------------
# FLOW_PATTERNS
# ---------------------------------------------------------------------------


class TestFlowPatterns:
    def test_four_patterns_exist(self):
        assert len(FLOW_PATTERNS) == 4

    def test_clarification_exists(self):
        assert "clarification" in FLOW_PATTERNS
        assert FLOW_PATTERNS["clarification"].pattern == "clarification"

    def test_topic_shift_exists(self):
        assert "topic_shift" in FLOW_PATTERNS

    def test_error_recovery_exists(self):
        assert "error_recovery" in FLOW_PATTERNS

    def test_deepening_exists(self):
        assert "deepening" in FLOW_PATTERNS

    def test_all_have_descriptions(self):
        for flow in FLOW_PATTERNS.values():
            assert flow.description


# ---------------------------------------------------------------------------
# generate_follow_up
# ---------------------------------------------------------------------------


class TestGenerateFollowUp:
    def test_clarification_pattern(self):
        turn = _make_turn(content="The key concept is understanding variables")
        flow = FLOW_PATTERNS["clarification"]
        result = generate_follow_up(turn, flow, _fixed_rng())
        assert isinstance(result, str)
        assert len(result) > 0

    def test_topic_shift_pattern(self):
        turn = _make_turn(content="Here is the explanation of functions")
        flow = FLOW_PATTERNS["topic_shift"]
        result = generate_follow_up(turn, flow, _fixed_rng())
        assert isinstance(result, str)

    def test_error_recovery_pattern(self):
        turn = _make_turn(content="The answer is definitely X based on Y")
        flow = FLOW_PATTERNS["error_recovery"]
        result = generate_follow_up(turn, flow, _fixed_rng())
        assert isinstance(result, str)

    def test_deepening_pattern(self):
        turn = _make_turn(content="The surface level explanation is as follows")
        flow = FLOW_PATTERNS["deepening"]
        result = generate_follow_up(turn, flow, _fixed_rng())
        assert isinstance(result, str)

    def test_short_content_fallback(self):
        turn = _make_turn(content="ok")
        flow = FLOW_PATTERNS["clarification"]
        result = generate_follow_up(turn, flow, _fixed_rng())
        assert isinstance(result, str)
        assert len(result) > 0

    def test_deterministic_with_seed(self):
        turn = _make_turn(content="A longer piece of content for testing purposes")
        flow = FLOW_PATTERNS["clarification"]
        r1 = generate_follow_up(turn, flow, _fixed_rng(99))
        r2 = generate_follow_up(turn, flow, _fixed_rng(99))
        assert r1 == r2


# ---------------------------------------------------------------------------
# generate_conversation
# ---------------------------------------------------------------------------


class TestGenerateConversation:
    def test_basic_generation(self):
        cfg = _make_config()
        conv = generate_conversation(cfg, rng=_fixed_rng())
        assert isinstance(conv, SimulatedConversation)
        assert len(conv.turns) >= 2

    def test_alternating_roles(self):
        cfg = _make_config(min_turns=4, max_turns=4)
        conv = generate_conversation(cfg, rng=_fixed_rng())
        for i, turn in enumerate(conv.turns):
            expected_role = "user" if i % 2 == 0 else "assistant"
            assert turn.role == expected_role, f"Turn {i} should be {expected_role}"

    def test_turn_count_within_bounds(self):
        cfg = _make_config(min_turns=2, max_turns=8)
        conv = generate_conversation(cfg, rng=_fixed_rng())
        assert len(conv.turns) >= 2
        # +1 for rounding to even
        assert len(conv.turns) <= 9

    def test_seed_message_used(self):
        cfg = _make_config(seed_message="What is Python?")
        conv = generate_conversation(cfg, rng=_fixed_rng())
        assert conv.turns[0].content == "What is Python?"

    def test_topic_preserved(self):
        cfg = _make_config(topic="machine learning")
        conv = generate_conversation(cfg, rng=_fixed_rng())
        assert conv.topic == "machine learning"

    def test_flow_pattern_preserved(self):
        cfg = _make_config(flow_pattern="deepening")
        conv = generate_conversation(cfg, rng=_fixed_rng())
        assert conv.flow_pattern == "deepening"

    def test_conversation_id_generated(self):
        cfg = _make_config()
        conv = generate_conversation(cfg, rng=_fixed_rng())
        assert conv.conversation_id
        assert len(conv.conversation_id) > 0

    def test_metadata_populated(self):
        cfg = _make_config()
        conv = generate_conversation(cfg, rng=_fixed_rng())
        assert "num_turns" in conv.metadata

    def test_default_rng_works(self):
        cfg = _make_config()
        conv = generate_conversation(cfg)
        assert len(conv.turns) >= 2

    def test_unknown_flow_falls_back(self):
        cfg = _make_config(flow_pattern="nonexistent")
        conv = generate_conversation(cfg, rng=_fixed_rng())
        assert len(conv.turns) >= 2

    def test_turn_numbers_sequential(self):
        cfg = _make_config(min_turns=4, max_turns=6)
        conv = generate_conversation(cfg, rng=_fixed_rng())
        for i, turn in enumerate(conv.turns):
            assert turn.turn_number == i

    def test_even_turn_count(self):
        # Conversations should have even number of turns (pairs)
        for seed in range(10):
            cfg = _make_config(min_turns=2, max_turns=8)
            conv = generate_conversation(cfg, rng=_fixed_rng(seed))
            assert len(conv.turns) % 2 == 0


# ---------------------------------------------------------------------------
# generate_batch
# ---------------------------------------------------------------------------


class TestGenerateBatch:
    def test_empty_batch(self):
        result = generate_batch([])
        assert result == []

    def test_single_config(self):
        configs = [_make_config()]
        result = generate_batch(configs)
        assert len(result) == 1
        assert isinstance(result[0], SimulatedConversation)

    def test_multiple_configs(self):
        configs = [
            _make_config(flow_pattern="clarification"),
            _make_config(flow_pattern="deepening"),
            _make_config(flow_pattern="topic_shift"),
        ]
        result = generate_batch(configs)
        assert len(result) == 3

    def test_unique_ids(self):
        configs = [_make_config() for _ in range(5)]
        result = generate_batch(configs)
        ids = [c.conversation_id for c in result]
        assert len(set(ids)) == 5


# ---------------------------------------------------------------------------
# conversation_to_dataset_items
# ---------------------------------------------------------------------------


class TestConversationToDatasetItems:
    def test_basic_conversion(self):
        turns = [
            ConversationTurn(role="user", content="Q1", turn_number=0),
            ConversationTurn(role="assistant", content="A1", turn_number=1),
        ]
        conv = SimulatedConversation(
            conversation_id="c1",
            turns=turns,
            flow_pattern="clarification",
            topic="test",
        )
        items = conversation_to_dataset_items(conv)
        assert len(items) == 1
        assert items[0]["input"] == "Q1"
        assert items[0]["expected_output"] == "A1"

    def test_multiple_pairs(self):
        turns = [
            ConversationTurn(role="user", content="Q1", turn_number=0),
            ConversationTurn(role="assistant", content="A1", turn_number=1),
            ConversationTurn(role="user", content="Q2", turn_number=2),
            ConversationTurn(role="assistant", content="A2", turn_number=3),
        ]
        conv = SimulatedConversation(
            conversation_id="c1",
            turns=turns,
            flow_pattern="clarification",
            topic="test",
        )
        items = conversation_to_dataset_items(conv)
        assert len(items) == 2
        assert items[1]["input"] == "Q2"
        assert items[1]["turn_pair"] == 1

    def test_history_accumulates(self):
        turns = [
            ConversationTurn(role="user", content="Q1", turn_number=0),
            ConversationTurn(role="assistant", content="A1", turn_number=1),
            ConversationTurn(role="user", content="Q2", turn_number=2),
            ConversationTurn(role="assistant", content="A2", turn_number=3),
        ]
        conv = SimulatedConversation(
            conversation_id="c1",
            turns=turns,
            flow_pattern="clarification",
            topic="test",
        )
        items = conversation_to_dataset_items(conv)
        assert items[0]["history"] == []
        assert len(items[1]["history"]) == 2

    def test_metadata_included(self):
        turns = [
            ConversationTurn(role="user", content="Q", turn_number=0),
            ConversationTurn(role="assistant", content="A", turn_number=1),
        ]
        conv = SimulatedConversation(
            conversation_id="cid",
            turns=turns,
            flow_pattern="deepening",
            topic="test",
        )
        items = conversation_to_dataset_items(conv)
        assert items[0]["conversation_id"] == "cid"
        assert items[0]["flow_pattern"] == "deepening"

    def test_empty_conversation(self):
        conv = SimulatedConversation(
            conversation_id="empty",
            turns=[],
            flow_pattern="clarification",
            topic="test",
        )
        items = conversation_to_dataset_items(conv)
        assert items == []

    def test_odd_turns_drops_last(self):
        turns = [
            ConversationTurn(role="user", content="Q1", turn_number=0),
            ConversationTurn(role="assistant", content="A1", turn_number=1),
            ConversationTurn(role="user", content="Q2", turn_number=2),
        ]
        conv = SimulatedConversation(
            conversation_id="c1",
            turns=turns,
            flow_pattern="clarification",
            topic="test",
        )
        items = conversation_to_dataset_items(conv)
        assert len(items) == 1


# ---------------------------------------------------------------------------
# format_conversation
# ---------------------------------------------------------------------------


class TestFormatConversation:
    def test_basic_format(self):
        turns = [
            ConversationTurn(role="user", content="Hello", turn_number=0),
            ConversationTurn(role="assistant", content="Hi there", turn_number=1),
        ]
        conv = SimulatedConversation(
            conversation_id="fmt-test",
            turns=turns,
            flow_pattern="clarification",
            topic="greetings",
        )
        result = format_conversation(conv)
        assert "fmt-test" in result
        assert "clarification" in result
        assert "greetings" in result
        assert "[0] User: Hello" in result
        assert "[1] Assistant: Hi there" in result

    def test_empty_conversation(self):
        conv = SimulatedConversation(
            conversation_id="empty",
            turns=[],
            flow_pattern="clarification",
            topic="nothing",
        )
        result = format_conversation(conv)
        assert "empty" in result
        assert "Turns: 0" in result


# ---------------------------------------------------------------------------
# get_conversation_stats
# ---------------------------------------------------------------------------


class TestGetConversationStats:
    def test_empty_list(self):
        stats = get_conversation_stats([])
        assert stats["count"] == 0
        assert stats["avg_turns"] == 0.0

    def test_single_conversation(self):
        turns = [
            ConversationTurn(role="user", content="Q", turn_number=0),
            ConversationTurn(role="assistant", content="A", turn_number=1),
        ]
        conv = SimulatedConversation(
            conversation_id="s1",
            turns=turns,
            flow_pattern="clarification",
            topic="test",
        )
        stats = get_conversation_stats([conv])
        assert stats["count"] == 1
        assert stats["avg_turns"] == 2.0
        assert stats["turn_distribution"] == {2: 1}
        assert stats["pattern_distribution"] == {"clarification": 1}

    def test_multiple_conversations(self):
        convs = []
        for pattern in ["clarification", "deepening", "clarification"]:
            turns = [
                ConversationTurn(role="user", content="Q", turn_number=0),
                ConversationTurn(role="assistant", content="A", turn_number=1),
            ]
            convs.append(
                SimulatedConversation(
                    conversation_id=f"c-{pattern}",
                    turns=turns,
                    flow_pattern=pattern,
                    topic="test",
                )
            )
        stats = get_conversation_stats(convs)
        assert stats["count"] == 3
        assert stats["pattern_distribution"]["clarification"] == 2
        assert stats["pattern_distribution"]["deepening"] == 1

    def test_avg_turns_calculation(self):
        conv2 = SimulatedConversation(
            conversation_id="a",
            turns=[_make_turn(turn_number=i) for i in range(2)],
            flow_pattern="clarification",
            topic="test",
        )
        conv4 = SimulatedConversation(
            conversation_id="b",
            turns=[_make_turn(turn_number=i) for i in range(4)],
            flow_pattern="clarification",
            topic="test",
        )
        stats = get_conversation_stats([conv2, conv4])
        assert stats["avg_turns"] == 3.0

    def test_turn_distribution(self):
        conv2 = SimulatedConversation(
            conversation_id="a",
            turns=[_make_turn(turn_number=i) for i in range(2)],
            flow_pattern="clarification",
            topic="test",
        )
        conv2b = SimulatedConversation(
            conversation_id="b",
            turns=[_make_turn(turn_number=i) for i in range(2)],
            flow_pattern="clarification",
            topic="test",
        )
        stats = get_conversation_stats([conv2, conv2b])
        assert stats["turn_distribution"] == {2: 2}
