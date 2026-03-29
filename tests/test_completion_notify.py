"""Tests for the completion notification module."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import patch

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.completion_notify import (
    NotifyConfig,
    NotifyEvent,
    should_notify,
    build_notification_message,
    detect_notification_method,
    format_terminal_bell,
    format_osascript_command,
    format_notify_send_command,
    create_notify_event,
    format_notify_summary,
)


# ---------------------------------------------------------------------------
# NotifyConfig
# ---------------------------------------------------------------------------


class TestNotifyConfig:
    def test_defaults(self):
        cfg = NotifyConfig()
        assert cfg.enabled is True
        assert cfg.min_duration_seconds == 30.0
        assert cfg.sound is False
        assert cfg.method == "auto"

    def test_custom_values(self):
        cfg = NotifyConfig(enabled=False, min_duration_seconds=10.0, sound=True, method="osascript")
        assert cfg.enabled is False
        assert cfg.min_duration_seconds == 10.0
        assert cfg.sound is True
        assert cfg.method == "osascript"

    def test_as_dict(self):
        cfg = NotifyConfig(enabled=False, min_duration_seconds=5.0, sound=True, method="notify_send")
        d = cfg.as_dict()
        assert d == {
            "enabled": False,
            "min_duration_seconds": 5.0,
            "sound": True,
            "method": "notify_send",
        }

    def test_from_dict(self):
        d = {"enabled": False, "min_duration_seconds": 15.0, "sound": True, "method": "terminal_bell"}
        cfg = NotifyConfig.from_dict(d)
        assert cfg.enabled is False
        assert cfg.min_duration_seconds == 15.0
        assert cfg.sound is True
        assert cfg.method == "terminal_bell"

    def test_from_dict_defaults(self):
        cfg = NotifyConfig.from_dict({})
        assert cfg.enabled is True
        assert cfg.min_duration_seconds == 30.0
        assert cfg.sound is False
        assert cfg.method == "auto"

    def test_roundtrip(self):
        original = NotifyConfig(enabled=False, min_duration_seconds=60.0, sound=True, method="osascript")
        restored = NotifyConfig.from_dict(original.as_dict())
        assert restored.enabled == original.enabled
        assert restored.min_duration_seconds == original.min_duration_seconds
        assert restored.sound == original.sound
        assert restored.method == original.method


# ---------------------------------------------------------------------------
# NotifyEvent
# ---------------------------------------------------------------------------


class TestNotifyEvent:
    def test_fields(self):
        event = NotifyEvent(
            command="analyze",
            duration_seconds=45.0,
            exit_code=0,
            message="evalyn analyze completed in 45.0s (success)",
            timestamp="2026-03-29T10:00:00",
        )
        assert event.command == "analyze"
        assert event.duration_seconds == 45.0
        assert event.exit_code == 0

    def test_as_dict(self):
        event = NotifyEvent(
            command="run",
            duration_seconds=120.5,
            exit_code=1,
            message="evalyn run completed in 120.5s (failed)",
            timestamp="2026-03-29T12:00:00",
        )
        d = event.as_dict()
        assert d["command"] == "run"
        assert d["duration_seconds"] == 120.5
        assert d["exit_code"] == 1
        assert d["message"] == "evalyn run completed in 120.5s (failed)"
        assert d["timestamp"] == "2026-03-29T12:00:00"

    def test_from_dict(self):
        d = {
            "command": "compare",
            "duration_seconds": 33.0,
            "exit_code": 0,
            "message": "ok",
            "timestamp": "2026-01-01T00:00:00",
        }
        event = NotifyEvent.from_dict(d)
        assert event.command == "compare"
        assert event.duration_seconds == 33.0
        assert event.exit_code == 0

    def test_roundtrip(self):
        original = NotifyEvent(
            command="evaluate",
            duration_seconds=99.9,
            exit_code=2,
            message="evalyn evaluate completed in 99.9s (failed)",
            timestamp="2026-06-15T08:30:00",
        )
        restored = NotifyEvent.from_dict(original.as_dict())
        assert restored.command == original.command
        assert restored.duration_seconds == original.duration_seconds
        assert restored.exit_code == original.exit_code
        assert restored.message == original.message
        assert restored.timestamp == original.timestamp


# ---------------------------------------------------------------------------
# should_notify
# ---------------------------------------------------------------------------


class TestShouldNotify:
    def test_above_threshold(self):
        cfg = NotifyConfig(min_duration_seconds=30.0)
        assert should_notify(30.0, cfg) is True
        assert should_notify(31.0, cfg) is True

    def test_below_threshold(self):
        cfg = NotifyConfig(min_duration_seconds=30.0)
        assert should_notify(29.9, cfg) is False

    def test_zero_duration(self):
        cfg = NotifyConfig(min_duration_seconds=0.0)
        assert should_notify(0.0, cfg) is True

    def test_disabled(self):
        cfg = NotifyConfig(enabled=False, min_duration_seconds=0.0)
        assert should_notify(100.0, cfg) is False

    def test_exact_threshold(self):
        cfg = NotifyConfig(min_duration_seconds=10.0)
        assert should_notify(10.0, cfg) is True

    def test_custom_threshold(self):
        cfg = NotifyConfig(min_duration_seconds=5.0)
        assert should_notify(4.9, cfg) is False
        assert should_notify(5.0, cfg) is True
        assert should_notify(5.1, cfg) is True


# ---------------------------------------------------------------------------
# build_notification_message
# ---------------------------------------------------------------------------


class TestBuildNotificationMessage:
    def test_success(self):
        msg = build_notification_message("analyze", 45.0, 0)
        assert msg == "evalyn analyze completed in 45.0s (success)"

    def test_failure(self):
        msg = build_notification_message("run", 120.3, 1)
        assert msg == "evalyn run completed in 120.3s (failed)"

    def test_non_zero_exit_code(self):
        msg = build_notification_message("evaluate", 30.0, 2)
        assert "failed" in msg

    def test_zero_duration(self):
        msg = build_notification_message("quick", 0.0, 0)
        assert msg == "evalyn quick completed in 0.0s (success)"

    def test_long_duration(self):
        msg = build_notification_message("batch", 3600.0, 0)
        assert "3600.0s" in msg

    def test_command_name_preserved(self):
        msg = build_notification_message("my-special-command", 10.0, 0)
        assert "my-special-command" in msg


# ---------------------------------------------------------------------------
# detect_notification_method
# ---------------------------------------------------------------------------


class TestDetectNotificationMethod:
    def test_osascript_available(self):
        with patch("evalyn_sdk.completion_notify.shutil.which") as mock_which:
            mock_which.side_effect = lambda cmd: "/usr/bin/osascript" if cmd == "osascript" else None
            assert detect_notification_method() == "osascript"

    def test_notify_send_available(self):
        with patch("evalyn_sdk.completion_notify.shutil.which") as mock_which:
            mock_which.side_effect = lambda cmd: "/usr/bin/notify-send" if cmd == "notify-send" else None
            assert detect_notification_method() == "notify_send"

    def test_fallback_to_terminal_bell(self):
        with patch("evalyn_sdk.completion_notify.shutil.which", return_value=None):
            assert detect_notification_method() == "terminal_bell"

    def test_osascript_preferred_over_notify_send(self):
        with patch("evalyn_sdk.completion_notify.shutil.which") as mock_which:
            mock_which.side_effect = lambda cmd: f"/usr/bin/{cmd}"
            assert detect_notification_method() == "osascript"


# ---------------------------------------------------------------------------
# format_terminal_bell
# ---------------------------------------------------------------------------


class TestFormatTerminalBell:
    def test_returns_bell_char(self):
        assert format_terminal_bell() == "\a"

    def test_length(self):
        assert len(format_terminal_bell()) == 1


# ---------------------------------------------------------------------------
# format_osascript_command
# ---------------------------------------------------------------------------


class TestFormatOsascriptCommand:
    def test_default_title(self):
        cmd = format_osascript_command("hello world")
        assert cmd[0] == "osascript"
        assert cmd[1] == "-e"
        assert "hello world" in cmd[2]
        assert "evalyn" in cmd[2]

    def test_custom_title(self):
        cmd = format_osascript_command("test msg", title="MyApp")
        assert "MyApp" in cmd[2]
        assert "test msg" in cmd[2]

    def test_returns_list(self):
        cmd = format_osascript_command("msg")
        assert isinstance(cmd, list)
        assert len(cmd) == 3

    def test_display_notification_syntax(self):
        cmd = format_osascript_command("msg", "t")
        assert "display notification" in cmd[2]
        assert "with title" in cmd[2]


# ---------------------------------------------------------------------------
# format_notify_send_command
# ---------------------------------------------------------------------------


class TestFormatNotifySendCommand:
    def test_default_title(self):
        cmd = format_notify_send_command("hello")
        assert cmd == ["notify-send", "evalyn", "hello"]

    def test_custom_title(self):
        cmd = format_notify_send_command("hello", title="MyApp")
        assert cmd == ["notify-send", "MyApp", "hello"]

    def test_returns_list(self):
        cmd = format_notify_send_command("x")
        assert isinstance(cmd, list)
        assert len(cmd) == 3


# ---------------------------------------------------------------------------
# create_notify_event
# ---------------------------------------------------------------------------


class TestCreateNotifyEvent:
    def test_success_event(self):
        event = create_notify_event("analyze", 45.0, 0)
        assert event.command == "analyze"
        assert event.duration_seconds == 45.0
        assert event.exit_code == 0
        assert "success" in event.message
        assert event.timestamp  # non-empty

    def test_failure_event(self):
        event = create_notify_event("run", 120.0, 1)
        assert event.exit_code == 1
        assert "failed" in event.message

    def test_timestamp_is_set(self):
        event = create_notify_event("cmd", 1.0, 0)
        assert len(event.timestamp) > 0
        assert "T" in event.timestamp  # ISO-ish format

    def test_message_matches_build(self):
        event = create_notify_event("test", 50.0, 0)
        expected = build_notification_message("test", 50.0, 0)
        assert event.message == expected


# ---------------------------------------------------------------------------
# format_notify_summary
# ---------------------------------------------------------------------------


class TestFormatNotifySummary:
    def test_success_summary(self):
        event = NotifyEvent(
            command="analyze",
            duration_seconds=45.0,
            exit_code=0,
            message="ok",
            timestamp="2026-03-29T10:00:00",
        )
        summary = format_notify_summary(event)
        assert "analyze" in summary
        assert "success" in summary
        assert "45.0s" in summary
        assert "2026-03-29T10:00:00" in summary

    def test_failure_summary(self):
        event = NotifyEvent(
            command="run",
            duration_seconds=120.0,
            exit_code=1,
            message="err",
            timestamp="2026-01-01T00:00:00",
        )
        summary = format_notify_summary(event)
        assert "failed" in summary
        assert "run" in summary

    def test_summary_has_all_fields(self):
        event = NotifyEvent(
            command="eval",
            duration_seconds=10.0,
            exit_code=0,
            message="m",
            timestamp="2026-06-15T08:30:00",
        )
        summary = format_notify_summary(event)
        assert "Command:" in summary
        assert "Status:" in summary
        assert "Duration:" in summary
        assert "Time:" in summary

    def test_summary_multiline(self):
        event = NotifyEvent(
            command="x",
            duration_seconds=1.0,
            exit_code=0,
            message="m",
            timestamp="t",
        )
        summary = format_notify_summary(event)
        assert "\n" in summary
        lines = summary.strip().split("\n")
        assert len(lines) == 4
