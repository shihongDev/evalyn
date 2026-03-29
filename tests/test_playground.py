"""Tests for the playground session model and scoring pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.playground import (
    PlaygroundConfig,
    PlaygroundTurn,
    PlaygroundSession,
    create_session,
    add_turn,
    get_turn_history,
    compute_session_stats,
    format_turn_display,
    format_session_summary,
    compare_turns,
    export_session,
)


# ---------------------------------------------------------------------------
# PlaygroundConfig
# ---------------------------------------------------------------------------


class TestPlaygroundConfig:
    def test_defaults(self):
        cfg = PlaygroundConfig()
        assert cfg.metrics == []
        assert cfg.show_scores is True
        assert cfg.history_limit == 50

    def test_custom_values(self):
        cfg = PlaygroundConfig(metrics=["accuracy", "relevance"], show_scores=False, history_limit=10)
        assert cfg.metrics == ["accuracy", "relevance"]
        assert cfg.show_scores is False
        assert cfg.history_limit == 10

    def test_as_dict(self):
        cfg = PlaygroundConfig(metrics=["a"])
        d = cfg.as_dict()
        assert d["metrics"] == ["a"]
        assert d["show_scores"] is True
        assert d["history_limit"] == 50

    def test_from_dict(self):
        d = {"metrics": ["x", "y"], "show_scores": False, "history_limit": 5}
        cfg = PlaygroundConfig.from_dict(d)
        assert cfg.metrics == ["x", "y"]
        assert cfg.show_scores is False

    def test_from_dict_defaults(self):
        cfg = PlaygroundConfig.from_dict({})
        assert cfg.metrics == []
        assert cfg.show_scores is True

    def test_roundtrip(self):
        cfg = PlaygroundConfig(metrics=["m1", "m2"], show_scores=False, history_limit=20)
        restored = PlaygroundConfig.from_dict(cfg.as_dict())
        assert restored.metrics == cfg.metrics
        assert restored.show_scores == cfg.show_scores
        assert restored.history_limit == cfg.history_limit


# ---------------------------------------------------------------------------
# PlaygroundTurn
# ---------------------------------------------------------------------------


class TestPlaygroundTurn:
    def test_basic_creation(self):
        t = PlaygroundTurn(turn_id=0, input_text="hi", output_text="hello")
        assert t.turn_id == 0
        assert t.input_text == "hi"
        assert t.output_text == "hello"
        assert t.scores == {}
        assert t.timestamp != ""

    def test_with_scores(self):
        t = PlaygroundTurn(turn_id=1, input_text="a", output_text="b", scores={"acc": 0.9})
        assert t.scores["acc"] == 0.9

    def test_as_dict(self):
        t = PlaygroundTurn(turn_id=2, input_text="in", output_text="out", scores={"x": 0.5})
        d = t.as_dict()
        assert d["turn_id"] == 2
        assert d["input_text"] == "in"
        assert d["scores"] == {"x": 0.5}
        assert "timestamp" in d

    def test_from_dict(self):
        d = {
            "turn_id": 3,
            "input_text": "q",
            "output_text": "a",
            "scores": {"m": 0.7},
            "timestamp": "2026-01-01T00:00:00",
        }
        t = PlaygroundTurn.from_dict(d)
        assert t.turn_id == 3
        assert t.timestamp == "2026-01-01T00:00:00"

    def test_roundtrip(self):
        t = PlaygroundTurn(turn_id=4, input_text="q", output_text="a", scores={"s": 0.8})
        restored = PlaygroundTurn.from_dict(t.as_dict())
        assert restored.turn_id == t.turn_id
        assert restored.input_text == t.input_text
        assert restored.scores == t.scores

    def test_auto_timestamp(self):
        t = PlaygroundTurn(turn_id=0, input_text="", output_text="")
        assert len(t.timestamp) > 0


# ---------------------------------------------------------------------------
# PlaygroundSession
# ---------------------------------------------------------------------------


class TestPlaygroundSession:
    def test_basic_creation(self):
        s = PlaygroundSession(session_id="s1")
        assert s.session_id == "s1"
        assert s.turns == []
        assert isinstance(s.config, PlaygroundConfig)
        assert s.created_at != ""

    def test_as_dict(self):
        s = PlaygroundSession(session_id="s2")
        d = s.as_dict()
        assert d["session_id"] == "s2"
        assert d["turns"] == []
        assert "config" in d
        assert "created_at" in d

    def test_from_dict(self):
        d = {
            "session_id": "s3",
            "turns": [
                {"turn_id": 0, "input_text": "a", "output_text": "b", "scores": {}, "timestamp": "t"}
            ],
            "config": {"metrics": ["m1"]},
            "created_at": "2026-01-01T00:00:00",
        }
        s = PlaygroundSession.from_dict(d)
        assert s.session_id == "s3"
        assert len(s.turns) == 1
        assert s.config.metrics == ["m1"]

    def test_roundtrip(self):
        s = PlaygroundSession(
            session_id="s4",
            config=PlaygroundConfig(metrics=["a"]),
        )
        restored = PlaygroundSession.from_dict(s.as_dict())
        assert restored.session_id == s.session_id
        assert restored.config.metrics == s.config.metrics


# ---------------------------------------------------------------------------
# create_session
# ---------------------------------------------------------------------------


class TestCreateSession:
    def test_default_config(self):
        s = create_session()
        assert len(s.session_id) > 0
        assert s.turns == []
        assert s.created_at != ""
        assert s.config.metrics == []

    def test_custom_config(self):
        cfg = PlaygroundConfig(metrics=["accuracy"], history_limit=10)
        s = create_session(config=cfg)
        assert s.config.metrics == ["accuracy"]
        assert s.config.history_limit == 10

    def test_unique_ids(self):
        s1 = create_session()
        s2 = create_session()
        assert s1.session_id != s2.session_id


# ---------------------------------------------------------------------------
# add_turn
# ---------------------------------------------------------------------------


class TestAddTurn:
    def test_adds_turn(self):
        s = create_session()
        s2 = add_turn(s, "hello", "hi", {"acc": 0.9})
        assert len(s2.turns) == 1
        assert s2.turns[0].input_text == "hello"
        assert s2.turns[0].output_text == "hi"
        assert s2.turns[0].scores == {"acc": 0.9}

    def test_immutable(self):
        s = create_session()
        s2 = add_turn(s, "a", "b", {})
        assert len(s.turns) == 0
        assert len(s2.turns) == 1

    def test_preserves_session_id(self):
        s = create_session()
        s2 = add_turn(s, "a", "b", {})
        assert s2.session_id == s.session_id

    def test_incremental_turn_ids(self):
        s = create_session()
        s = add_turn(s, "a", "b", {})
        s = add_turn(s, "c", "d", {})
        s = add_turn(s, "e", "f", {})
        assert [t.turn_id for t in s.turns] == [0, 1, 2]

    def test_preserves_created_at(self):
        s = create_session()
        original_ts = s.created_at
        s2 = add_turn(s, "a", "b", {})
        assert s2.created_at == original_ts


# ---------------------------------------------------------------------------
# get_turn_history
# ---------------------------------------------------------------------------


class TestGetTurnHistory:
    def _session_with_turns(self, n: int) -> PlaygroundSession:
        s = create_session()
        for i in range(n):
            s = add_turn(s, f"in_{i}", f"out_{i}", {"score": float(i)})
        return s

    def test_all_turns(self):
        s = self._session_with_turns(5)
        history = get_turn_history(s)
        assert len(history) == 5

    def test_limit_zero_returns_all(self):
        s = self._session_with_turns(3)
        history = get_turn_history(s, limit=0)
        assert len(history) == 3

    def test_limit_returns_recent(self):
        s = self._session_with_turns(5)
        history = get_turn_history(s, limit=2)
        assert len(history) == 2
        assert history[0].turn_id == 3
        assert history[1].turn_id == 4

    def test_limit_exceeds_total(self):
        s = self._session_with_turns(2)
        history = get_turn_history(s, limit=10)
        assert len(history) == 2

    def test_empty_session(self):
        s = create_session()
        history = get_turn_history(s)
        assert history == []


# ---------------------------------------------------------------------------
# compute_session_stats
# ---------------------------------------------------------------------------


class TestComputeSessionStats:
    def test_empty_session(self):
        s = create_session()
        stats = compute_session_stats(s)
        assert stats["total_turns"] == 0
        assert stats["avg_scores"] == {}
        assert stats["best_turn"] is None
        assert stats["worst_turn"] is None

    def test_single_turn(self):
        s = create_session()
        s = add_turn(s, "a", "b", {"accuracy": 0.8})
        stats = compute_session_stats(s)
        assert stats["total_turns"] == 1
        assert abs(stats["avg_scores"]["accuracy"] - 0.8) < 1e-9
        assert stats["best_turn"] == 0
        assert stats["worst_turn"] == 0

    def test_multiple_turns(self):
        s = create_session()
        s = add_turn(s, "a", "b", {"accuracy": 0.6})
        s = add_turn(s, "c", "d", {"accuracy": 1.0})
        s = add_turn(s, "e", "f", {"accuracy": 0.2})
        stats = compute_session_stats(s)
        assert stats["total_turns"] == 3
        assert abs(stats["avg_scores"]["accuracy"] - 0.6) < 1e-9
        assert stats["best_turn"] == 1
        assert stats["worst_turn"] == 2

    def test_multiple_metrics(self):
        s = create_session()
        s = add_turn(s, "a", "b", {"accuracy": 0.8, "relevance": 0.6})
        s = add_turn(s, "c", "d", {"accuracy": 0.4, "relevance": 1.0})
        stats = compute_session_stats(s)
        assert abs(stats["avg_scores"]["accuracy"] - 0.6) < 1e-9
        assert abs(stats["avg_scores"]["relevance"] - 0.8) < 1e-9

    def test_best_worst_by_average(self):
        s = create_session()
        s = add_turn(s, "a", "b", {"m1": 1.0, "m2": 0.0})  # avg 0.5
        s = add_turn(s, "c", "d", {"m1": 0.8, "m2": 0.8})  # avg 0.8
        stats = compute_session_stats(s)
        assert stats["best_turn"] == 1
        assert stats["worst_turn"] == 0


# ---------------------------------------------------------------------------
# format_turn_display
# ---------------------------------------------------------------------------


class TestFormatTurnDisplay:
    def test_basic_format(self):
        t = PlaygroundTurn(turn_id=0, input_text="hello", output_text="world", scores={"acc": 0.9})
        result = format_turn_display(t)
        assert "Turn 0" in result
        assert "hello" in result
        assert "world" in result
        assert "acc=0.900" in result

    def test_no_scores(self):
        t = PlaygroundTurn(turn_id=1, input_text="a", output_text="b")
        result = format_turn_display(t)
        assert "Turn 1" in result
        assert "Scores" not in result

    def test_multiple_scores(self):
        t = PlaygroundTurn(turn_id=0, input_text="q", output_text="a", scores={"x": 0.5, "y": 0.8})
        result = format_turn_display(t)
        assert "x=0.500" in result
        assert "y=0.800" in result


# ---------------------------------------------------------------------------
# format_session_summary
# ---------------------------------------------------------------------------


class TestFormatSessionSummary:
    def test_empty_session(self):
        s = create_session()
        result = format_session_summary(s)
        assert "Session" in result
        assert "Turns: 0" in result

    def test_with_turns(self):
        s = create_session()
        s = add_turn(s, "a", "b", {"accuracy": 0.8})
        s = add_turn(s, "c", "d", {"accuracy": 0.6})
        result = format_session_summary(s)
        assert "Turns: 2" in result
        assert "accuracy=" in result

    def test_includes_session_id(self):
        s = create_session()
        result = format_session_summary(s)
        assert s.session_id in result

    def test_includes_best_worst(self):
        s = create_session()
        s = add_turn(s, "a", "b", {"m": 1.0})
        s = add_turn(s, "c", "d", {"m": 0.0})
        result = format_session_summary(s)
        assert "Best turn" in result
        assert "Worst turn" in result


# ---------------------------------------------------------------------------
# compare_turns
# ---------------------------------------------------------------------------


class TestCompareTurns:
    def test_same_metrics(self):
        a = PlaygroundTurn(turn_id=0, input_text="", output_text="", scores={"acc": 0.5})
        b = PlaygroundTurn(turn_id=1, input_text="", output_text="", scores={"acc": 0.8})
        deltas = compare_turns(a, b)
        assert abs(deltas["acc"] - 0.3) < 1e-9

    def test_different_metrics(self):
        a = PlaygroundTurn(turn_id=0, input_text="", output_text="", scores={"x": 0.5})
        b = PlaygroundTurn(turn_id=1, input_text="", output_text="", scores={"y": 0.8})
        deltas = compare_turns(a, b)
        # x only in a: b-a = 0.0 - 0.5 = -0.5
        assert abs(deltas["x"] - (-0.5)) < 1e-9
        # y only in b: b-a = 0.8 - 0.0 = 0.8
        assert abs(deltas["y"] - 0.8) < 1e-9

    def test_overlapping_metrics(self):
        a = PlaygroundTurn(turn_id=0, input_text="", output_text="", scores={"x": 0.3, "y": 0.7})
        b = PlaygroundTurn(turn_id=1, input_text="", output_text="", scores={"y": 0.9, "z": 0.4})
        deltas = compare_turns(a, b)
        assert abs(deltas["y"] - 0.2) < 1e-9
        assert "x" in deltas
        assert "z" in deltas

    def test_empty_scores(self):
        a = PlaygroundTurn(turn_id=0, input_text="", output_text="", scores={})
        b = PlaygroundTurn(turn_id=1, input_text="", output_text="", scores={})
        deltas = compare_turns(a, b)
        assert deltas == {}

    def test_negative_delta(self):
        a = PlaygroundTurn(turn_id=0, input_text="", output_text="", scores={"acc": 0.9})
        b = PlaygroundTurn(turn_id=1, input_text="", output_text="", scores={"acc": 0.3})
        deltas = compare_turns(a, b)
        assert deltas["acc"] < 0


# ---------------------------------------------------------------------------
# export_session
# ---------------------------------------------------------------------------


class TestExportSession:
    def test_valid_json(self):
        s = create_session()
        s = add_turn(s, "hello", "world", {"acc": 0.9})
        exported = export_session(s)
        data = json.loads(exported)
        assert data["session_id"] == s.session_id
        assert len(data["turns"]) == 1

    def test_roundtrip_via_json(self):
        s = create_session()
        s = add_turn(s, "a", "b", {"x": 0.5})
        exported = export_session(s)
        data = json.loads(exported)
        restored = PlaygroundSession.from_dict(data)
        assert restored.session_id == s.session_id
        assert len(restored.turns) == 1
        assert restored.turns[0].scores == {"x": 0.5}

    def test_empty_session_export(self):
        s = create_session()
        exported = export_session(s)
        data = json.loads(exported)
        assert data["turns"] == []

    def test_export_preserves_config(self):
        cfg = PlaygroundConfig(metrics=["m1", "m2"], show_scores=False)
        s = create_session(config=cfg)
        exported = export_session(s)
        data = json.loads(exported)
        assert data["config"]["metrics"] == ["m1", "m2"]
        assert data["config"]["show_scores"] is False
