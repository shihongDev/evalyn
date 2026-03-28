"""Tests for evalyn_sdk.storage.read_only_mode.

Run with: uv run pytest tests/test_read_only_mode.py -v
"""

from __future__ import annotations

import pytest

from evalyn_sdk.storage.read_only_mode import (
    ReadOnlyStatus,
    check_write_allowed,
    disable_read_only,
    enable_read_only,
    get_history,
    get_status,
    is_read_only,
    read_only_context,
    reset_read_only,
)


@pytest.fixture(autouse=True)
def _clean_state():
    """Reset read-only state before each test."""
    reset_read_only()
    yield
    reset_read_only()


# ---------------------------------------------------------------------------
# enable / disable / is_read_only
# ---------------------------------------------------------------------------


class TestEnableDisable:
    def test_enable_sets_flag(self):
        enable_read_only()
        assert is_read_only() is True

    def test_disable_clears_flag(self):
        enable_read_only()
        disable_read_only()
        assert is_read_only() is False

    def test_is_read_only_default_false(self):
        assert is_read_only() is False

    def test_enable_returns_status(self):
        status = enable_read_only(reason="analysis")
        assert isinstance(status, ReadOnlyStatus)
        assert status.enabled is True
        assert status.reason == "analysis"
        assert status.since != ""

    def test_disable_returns_status(self):
        enable_read_only()
        status = disable_read_only()
        assert status.enabled is False

    def test_enable_already_enabled(self):
        enable_read_only(reason="first")
        status = enable_read_only(reason="second")
        assert status.enabled is True
        assert status.reason == "second"
        assert is_read_only() is True

    def test_disable_already_disabled(self):
        status = disable_read_only()
        assert status.enabled is False
        assert is_read_only() is False


# ---------------------------------------------------------------------------
# get_status
# ---------------------------------------------------------------------------


class TestGetStatus:
    def test_status_when_writable(self):
        status = get_status()
        assert status.enabled is False

    def test_status_when_read_only(self):
        enable_read_only(reason="testing")
        status = get_status()
        assert status.enabled is True
        assert status.reason == "testing"
        assert status.since != ""

    def test_status_reflects_latest_enable(self):
        enable_read_only(reason="first")
        enable_read_only(reason="second")
        status = get_status()
        assert status.reason == "second"


# ---------------------------------------------------------------------------
# check_write_allowed
# ---------------------------------------------------------------------------


class TestCheckWriteAllowed:
    def test_allowed_when_writable(self):
        allowed, reason = check_write_allowed()
        assert allowed is True
        assert reason == ""

    def test_blocked_when_read_only(self):
        enable_read_only(reason="safety")
        allowed, reason = check_write_allowed()
        assert allowed is False
        assert "read-only" in reason
        assert "safety" in reason

    def test_allowed_after_disable(self):
        enable_read_only()
        disable_read_only()
        allowed, _ = check_write_allowed()
        assert allowed is True


# ---------------------------------------------------------------------------
# read_only_context
# ---------------------------------------------------------------------------


class TestReadOnlyContext:
    def test_context_enables(self):
        assert is_read_only() is False
        with read_only_context(reason="ctx"):
            assert is_read_only() is True

    def test_context_restores_writable(self):
        with read_only_context():
            pass
        assert is_read_only() is False

    def test_context_restores_read_only(self):
        enable_read_only(reason="outer")
        with read_only_context(reason="inner"):
            assert is_read_only() is True
        assert is_read_only() is True

    def test_nested_contexts(self):
        with read_only_context(reason="outer"):
            assert is_read_only() is True
            with read_only_context(reason="inner"):
                assert is_read_only() is True
            assert is_read_only() is True
        assert is_read_only() is False

    def test_context_restores_on_exception(self):
        try:
            with read_only_context():
                raise ValueError("boom")
        except ValueError:
            pass
        assert is_read_only() is False


# ---------------------------------------------------------------------------
# get_history
# ---------------------------------------------------------------------------


class TestHistory:
    def test_empty_history(self):
        assert get_history() == []

    def test_history_records_enable(self):
        enable_read_only(reason="test")
        history = get_history()
        assert len(history) == 1
        assert history[0]["action"] == "enable"
        assert history[0]["reason"] == "test"

    def test_history_records_disable(self):
        enable_read_only()
        disable_read_only()
        history = get_history()
        assert len(history) == 2
        assert history[1]["action"] == "disable"

    def test_history_has_timestamps(self):
        enable_read_only()
        for entry in get_history():
            assert "timestamp" in entry

    def test_history_is_copy(self):
        enable_read_only()
        h = get_history()
        h.clear()
        assert len(get_history()) == 1


# ---------------------------------------------------------------------------
# reset_read_only
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_flag(self):
        enable_read_only()
        reset_read_only()
        assert is_read_only() is False

    def test_reset_clears_history(self):
        enable_read_only()
        disable_read_only()
        reset_read_only()
        assert get_history() == []


# ---------------------------------------------------------------------------
# ReadOnlyStatus dataclass
# ---------------------------------------------------------------------------


class TestReadOnlyStatus:
    def test_as_dict(self):
        s = ReadOnlyStatus(enabled=True, since="2026-01-01T00:00:00", reason="r")
        d = s.as_dict()
        assert d == {
            "enabled": True,
            "since": "2026-01-01T00:00:00",
            "reason": "r",
        }

    def test_as_dict_defaults(self):
        s = ReadOnlyStatus(enabled=False)
        d = s.as_dict()
        assert d["since"] == ""
        assert d["reason"] == ""
