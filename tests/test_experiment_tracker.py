"""Tests for the experiment tracker integration module."""

from __future__ import annotations

import sys
from pathlib import Path

# Add SDK to path
SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.experiment_tracker import (
    ConsoleTracker,
    ExperimentTracker,
    TrackerConfig,
    TrackerEvent,
    TrackerLog,
    create_tracker,
    format_tracker_log,
    log_eval_results,
)


# ---------------------------------------------------------------------------
# TrackerConfig
# ---------------------------------------------------------------------------


class TestTrackerConfig:
    def test_defaults(self):
        cfg = TrackerConfig(tracker_type="console")
        assert cfg.tracker_type == "console"
        assert cfg.project_name == "evalyn"
        assert cfg.run_name == ""
        assert cfg.tags == []

    def test_custom_values(self):
        cfg = TrackerConfig(
            tracker_type="wandb",
            project_name="my-project",
            run_name="run-1",
            tags=["nightly", "v2"],
        )
        assert cfg.tracker_type == "wandb"
        assert cfg.project_name == "my-project"
        assert cfg.run_name == "run-1"
        assert cfg.tags == ["nightly", "v2"]

    def test_as_dict(self):
        cfg = TrackerConfig(tracker_type="mlflow", tags=["a"])
        d = cfg.as_dict()
        assert d["tracker_type"] == "mlflow"
        assert d["project_name"] == "evalyn"
        assert d["run_name"] == ""
        assert d["tags"] == ["a"]

    def test_from_dict_minimal(self):
        cfg = TrackerConfig.from_dict({"tracker_type": "neptune"})
        assert cfg.tracker_type == "neptune"
        assert cfg.project_name == "evalyn"
        assert cfg.run_name == ""
        assert cfg.tags == []

    def test_from_dict_full(self):
        data = {
            "tracker_type": "wandb",
            "project_name": "p",
            "run_name": "r",
            "tags": ["x", "y"],
        }
        cfg = TrackerConfig.from_dict(data)
        assert cfg.tracker_type == "wandb"
        assert cfg.project_name == "p"
        assert cfg.run_name == "r"
        assert cfg.tags == ["x", "y"]

    def test_roundtrip(self):
        cfg = TrackerConfig(
            tracker_type="console",
            project_name="proj",
            run_name="run",
            tags=["t1", "t2"],
        )
        assert TrackerConfig.from_dict(cfg.as_dict()) == cfg

    def test_tags_are_copied(self):
        """Tags list should be a copy, not a reference."""
        original = ["a"]
        cfg = TrackerConfig.from_dict({"tracker_type": "console", "tags": original})
        original.append("b")
        assert cfg.tags == ["a"]


# ---------------------------------------------------------------------------
# TrackerEvent
# ---------------------------------------------------------------------------


class TestTrackerEvent:
    def test_metric_event(self):
        ev = TrackerEvent(event_type="metric", key="accuracy", value=0.95, step=1)
        assert ev.event_type == "metric"
        assert ev.key == "accuracy"
        assert ev.value == 0.95
        assert ev.step == 1

    def test_default_step(self):
        ev = TrackerEvent(event_type="param", key="lr", value=0.01)
        assert ev.step == 0

    def test_as_dict(self):
        ev = TrackerEvent(event_type="artifact", key="model", value="/tmp/m.bin", step=3)
        d = ev.as_dict()
        assert d == {
            "event_type": "artifact",
            "key": "model",
            "value": "/tmp/m.bin",
            "step": 3,
        }

    def test_from_dict_minimal(self):
        ev = TrackerEvent.from_dict(
            {"event_type": "metric", "key": "loss", "value": 0.1}
        )
        assert ev.step == 0

    def test_from_dict_full(self):
        data = {"event_type": "param", "key": "epochs", "value": 10, "step": 5}
        ev = TrackerEvent.from_dict(data)
        assert ev.event_type == "param"
        assert ev.key == "epochs"
        assert ev.value == 10
        assert ev.step == 5

    def test_roundtrip(self):
        ev = TrackerEvent(event_type="metric", key="f1", value=0.88, step=2)
        assert TrackerEvent.from_dict(ev.as_dict()) == ev

    def test_value_can_be_string(self):
        ev = TrackerEvent(event_type="param", key="model", value="gpt-4o")
        assert ev.value == "gpt-4o"

    def test_value_can_be_dict(self):
        ev = TrackerEvent(event_type="param", key="cfg", value={"a": 1})
        assert ev.value == {"a": 1}


# ---------------------------------------------------------------------------
# TrackerLog
# ---------------------------------------------------------------------------


class TestTrackerLog:
    def test_empty_log(self):
        log = TrackerLog()
        assert log.events == []
        assert log.config.tracker_type == "console"

    def test_as_dict(self):
        ev = TrackerEvent(event_type="metric", key="k", value=1.0)
        cfg = TrackerConfig(tracker_type="wandb")
        log = TrackerLog(events=[ev], config=cfg)
        d = log.as_dict()
        assert len(d["events"]) == 1
        assert d["config"]["tracker_type"] == "wandb"

    def test_from_dict(self):
        data = {
            "events": [
                {"event_type": "metric", "key": "a", "value": 1.0},
                {"event_type": "param", "key": "b", "value": "x"},
            ],
            "config": {"tracker_type": "mlflow"},
        }
        log = TrackerLog.from_dict(data)
        assert len(log.events) == 2
        assert log.events[0].key == "a"
        assert log.config.tracker_type == "mlflow"

    def test_roundtrip(self):
        ev = TrackerEvent(event_type="artifact", key="f", value="/p", step=0)
        cfg = TrackerConfig(tracker_type="console", tags=["tag"])
        log = TrackerLog(events=[ev], config=cfg)
        restored = TrackerLog.from_dict(log.as_dict())
        assert restored.events == log.events
        assert restored.config == log.config


# ---------------------------------------------------------------------------
# ConsoleTracker
# ---------------------------------------------------------------------------


class TestConsoleTracker:
    def _make_tracker(self, **kwargs) -> ConsoleTracker:
        cfg = TrackerConfig(tracker_type="console", **kwargs)
        return ConsoleTracker(cfg)

    def test_name(self):
        t = self._make_tracker()
        assert t.name == "console"

    def test_log_metric(self):
        t = self._make_tracker()
        ev = t.log_metric("acc", 0.9, step=1)
        assert ev.event_type == "metric"
        assert ev.key == "acc"
        assert ev.value == 0.9
        assert ev.step == 1

    def test_log_metric_default_step(self):
        t = self._make_tracker()
        ev = t.log_metric("loss", 0.1)
        assert ev.step == 0

    def test_log_param(self):
        t = self._make_tracker()
        ev = t.log_param("lr", 0.001)
        assert ev.event_type == "param"
        assert ev.key == "lr"
        assert ev.value == 0.001

    def test_log_artifact(self):
        t = self._make_tracker()
        ev = t.log_artifact("weights", "/tmp/w.pt")
        assert ev.event_type == "artifact"
        assert ev.key == "weights"
        assert ev.value == "/tmp/w.pt"

    def test_get_log_accumulates(self):
        t = self._make_tracker()
        t.log_metric("a", 1.0)
        t.log_param("b", "v")
        t.log_artifact("c", "/p")
        log = t.get_log()
        assert len(log.events) == 3
        assert log.config.tracker_type == "console"

    def test_get_log_returns_copy(self):
        t = self._make_tracker()
        t.log_metric("x", 1.0)
        log1 = t.get_log()
        t.log_metric("y", 2.0)
        log2 = t.get_log()
        assert len(log1.events) == 1
        assert len(log2.events) == 2

    def test_is_experiment_tracker(self):
        t = self._make_tracker()
        assert isinstance(t, ExperimentTracker)

    def test_config_preserved(self):
        t = self._make_tracker(project_name="proj", run_name="r1", tags=["a"])
        log = t.get_log()
        assert log.config.project_name == "proj"
        assert log.config.run_name == "r1"
        assert log.config.tags == ["a"]


# ---------------------------------------------------------------------------
# create_tracker
# ---------------------------------------------------------------------------


class TestCreateTracker:
    def test_console_type(self):
        cfg = TrackerConfig(tracker_type="console")
        t = create_tracker(cfg)
        assert isinstance(t, ConsoleTracker)
        assert t.name == "console"

    def test_unsupported_type_raises(self):
        cfg = TrackerConfig(tracker_type="wandb")
        try:
            create_tracker(cfg)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "wandb" in str(e)

    def test_unsupported_mlflow(self):
        cfg = TrackerConfig(tracker_type="mlflow")
        try:
            create_tracker(cfg)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "mlflow" in str(e)

    def test_unsupported_neptune(self):
        cfg = TrackerConfig(tracker_type="neptune")
        try:
            create_tracker(cfg)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "neptune" in str(e)

    def test_unknown_type(self):
        cfg = TrackerConfig(tracker_type="unknown_tracker_xyz")
        try:
            create_tracker(cfg)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "unknown_tracker_xyz" in str(e)


# ---------------------------------------------------------------------------
# log_eval_results
# ---------------------------------------------------------------------------


class TestLogEvalResults:
    def _make_tracker(self) -> ConsoleTracker:
        return ConsoleTracker(TrackerConfig(tracker_type="console"))

    def test_metrics_only(self):
        t = self._make_tracker()
        events = log_eval_results(t, {"acc": 0.9, "f1": 0.85})
        assert len(events) == 2
        assert all(e.event_type == "metric" for e in events)

    def test_metrics_and_params(self):
        t = self._make_tracker()
        events = log_eval_results(
            t,
            results={"acc": 0.9},
            params={"model": "gpt-4o", "temp": 0.7},
        )
        param_events = [e for e in events if e.event_type == "param"]
        metric_events = [e for e in events if e.event_type == "metric"]
        assert len(param_events) == 2
        assert len(metric_events) == 1

    def test_params_logged_before_metrics(self):
        t = self._make_tracker()
        events = log_eval_results(t, {"m": 1.0}, params={"p": "v"})
        assert events[0].event_type == "param"
        assert events[1].event_type == "metric"

    def test_empty_results(self):
        t = self._make_tracker()
        events = log_eval_results(t, {})
        assert events == []

    def test_none_params(self):
        t = self._make_tracker()
        events = log_eval_results(t, {"x": 1.0}, params=None)
        assert len(events) == 1

    def test_events_in_tracker_log(self):
        t = self._make_tracker()
        log_eval_results(t, {"a": 1.0, "b": 2.0}, params={"c": "d"})
        log = t.get_log()
        assert len(log.events) == 3

    def test_sorted_keys(self):
        """Results and params should be logged in sorted key order."""
        t = self._make_tracker()
        events = log_eval_results(
            t,
            results={"z_metric": 1.0, "a_metric": 2.0},
            params={"z_param": "a", "a_param": "b"},
        )
        keys = [e.key for e in events]
        assert keys == ["a_param", "z_param", "a_metric", "z_metric"]


# ---------------------------------------------------------------------------
# format_tracker_log
# ---------------------------------------------------------------------------


class TestFormatTrackerLog:
    def test_empty_log(self):
        log = TrackerLog(
            events=[],
            config=TrackerConfig(tracker_type="console"),
        )
        text = format_tracker_log(log)
        assert "Tracker: console" in text
        assert "Events (0):" in text

    def test_metric_format(self):
        ev = TrackerEvent(event_type="metric", key="acc", value=0.95, step=2)
        log = TrackerLog(
            events=[ev],
            config=TrackerConfig(tracker_type="console"),
        )
        text = format_tracker_log(log)
        assert "[metric] acc = 0.95 (step 2)" in text

    def test_param_format(self):
        ev = TrackerEvent(event_type="param", key="lr", value=0.01)
        log = TrackerLog(events=[ev], config=TrackerConfig(tracker_type="console"))
        text = format_tracker_log(log)
        assert "[param]  lr = 0.01" in text

    def test_artifact_format(self):
        ev = TrackerEvent(event_type="artifact", key="model", value="/tmp/m.bin")
        log = TrackerLog(events=[ev], config=TrackerConfig(tracker_type="console"))
        text = format_tracker_log(log)
        assert "[artifact] model -> /tmp/m.bin" in text

    def test_run_name_shown(self):
        log = TrackerLog(
            events=[],
            config=TrackerConfig(tracker_type="console", run_name="my-run"),
        )
        text = format_tracker_log(log)
        assert "Run: my-run" in text

    def test_tags_shown(self):
        log = TrackerLog(
            events=[],
            config=TrackerConfig(tracker_type="console", tags=["a", "b"]),
        )
        text = format_tracker_log(log)
        assert "Tags: a, b" in text

    def test_no_run_name_hidden(self):
        log = TrackerLog(
            events=[],
            config=TrackerConfig(tracker_type="console"),
        )
        text = format_tracker_log(log)
        assert "Run:" not in text

    def test_no_tags_hidden(self):
        log = TrackerLog(
            events=[],
            config=TrackerConfig(tracker_type="console"),
        )
        text = format_tracker_log(log)
        assert "Tags:" not in text

    def test_project_shown(self):
        log = TrackerLog(
            events=[],
            config=TrackerConfig(tracker_type="console", project_name="my-proj"),
        )
        text = format_tracker_log(log)
        assert "Project: my-proj" in text

    def test_multiple_events(self):
        events = [
            TrackerEvent(event_type="param", key="a", value=1),
            TrackerEvent(event_type="metric", key="b", value=2.0, step=1),
            TrackerEvent(event_type="artifact", key="c", value="/p"),
        ]
        log = TrackerLog(events=events, config=TrackerConfig(tracker_type="console"))
        text = format_tracker_log(log)
        assert "Events (3):" in text
        assert "[param]" in text
        assert "[metric]" in text
        assert "[artifact]" in text
