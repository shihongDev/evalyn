"""Tests for instrumentation_toggle module.

All tests are pure - no external deps required.
"""

from __future__ import annotations

import pytest

from evalyn_sdk.trace.instrumentation_toggle import (
    clear_toggle_history,
    get_toggle_history,
    instrumentation_paused,
    is_instrumentation_enabled,
    pause_instrumentation,
    reset_instrumentation,
    resume_instrumentation,
    toggle_instrumentation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_state():
    """Reset global state before each test."""
    reset_instrumentation()
    yield
    reset_instrumentation()


# ---------------------------------------------------------------------------
# Default state
# ---------------------------------------------------------------------------


class TestDefaultState:
    def test_default_is_enabled(self):
        assert is_instrumentation_enabled() is True

    def test_history_empty_after_reset(self):
        assert get_toggle_history() == []


# ---------------------------------------------------------------------------
# toggle_instrumentation
# ---------------------------------------------------------------------------


class TestToggle:
    def test_toggle_to_disabled(self):
        toggle_instrumentation(False)
        assert is_instrumentation_enabled() is False

    def test_toggle_returns_previous_state(self):
        prev = toggle_instrumentation(False)
        assert prev is True

    def test_toggle_returns_previous_state_when_already_disabled(self):
        toggle_instrumentation(False)
        prev = toggle_instrumentation(True)
        assert prev is False

    def test_toggle_to_same_state(self):
        prev = toggle_instrumentation(True)
        assert prev is True
        assert is_instrumentation_enabled() is True

    def test_multiple_toggles(self):
        toggle_instrumentation(False)
        toggle_instrumentation(True)
        toggle_instrumentation(False)
        assert is_instrumentation_enabled() is False

    def test_toggle_default_param_enables(self):
        toggle_instrumentation(False)
        toggle_instrumentation()
        assert is_instrumentation_enabled() is True


# ---------------------------------------------------------------------------
# is_instrumentation_enabled
# ---------------------------------------------------------------------------


class TestIsEnabled:
    def test_reflects_toggle(self):
        assert is_instrumentation_enabled() is True
        toggle_instrumentation(False)
        assert is_instrumentation_enabled() is False
        toggle_instrumentation(True)
        assert is_instrumentation_enabled() is True


# ---------------------------------------------------------------------------
# pause / resume shortcuts
# ---------------------------------------------------------------------------


class TestPauseResume:
    def test_pause_disables(self):
        pause_instrumentation()
        assert is_instrumentation_enabled() is False

    def test_resume_enables(self):
        pause_instrumentation()
        resume_instrumentation()
        assert is_instrumentation_enabled() is True

    def test_pause_returns_previous(self):
        prev = pause_instrumentation()
        assert prev is True

    def test_resume_returns_previous(self):
        pause_instrumentation()
        prev = resume_instrumentation()
        assert prev is False


# ---------------------------------------------------------------------------
# History tracking
# ---------------------------------------------------------------------------


class TestHistory:
    def test_toggle_records_history(self):
        toggle_instrumentation(False)
        history = get_toggle_history()
        assert len(history) == 1
        assert history[0]["enabled"] is False
        assert history[0]["previous"] is True
        assert "timestamp" in history[0]

    def test_multiple_entries(self):
        toggle_instrumentation(False)
        toggle_instrumentation(True)
        toggle_instrumentation(False)
        history = get_toggle_history()
        assert len(history) == 3

    def test_history_returns_copy(self):
        toggle_instrumentation(False)
        h1 = get_toggle_history()
        h1.clear()
        assert len(get_toggle_history()) == 1

    def test_clear_toggle_history(self):
        toggle_instrumentation(False)
        toggle_instrumentation(True)
        clear_toggle_history()
        assert get_toggle_history() == []

    def test_clear_does_not_change_state(self):
        toggle_instrumentation(False)
        clear_toggle_history()
        assert is_instrumentation_enabled() is False

    def test_history_timestamp_is_iso(self):
        toggle_instrumentation(False)
        ts = get_toggle_history()[0]["timestamp"]
        # Should be parseable as ISO format
        from datetime import datetime
        datetime.fromisoformat(ts)


# ---------------------------------------------------------------------------
# reset_instrumentation
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_enables(self):
        toggle_instrumentation(False)
        reset_instrumentation()
        assert is_instrumentation_enabled() is True

    def test_reset_clears_history(self):
        toggle_instrumentation(False)
        toggle_instrumentation(True)
        reset_instrumentation()
        assert get_toggle_history() == []


# ---------------------------------------------------------------------------
# instrumentation_paused context manager
# ---------------------------------------------------------------------------


class TestContextManager:
    def test_pauses_inside(self):
        with instrumentation_paused():
            assert is_instrumentation_enabled() is False

    def test_restores_after(self):
        with instrumentation_paused():
            pass
        assert is_instrumentation_enabled() is True

    def test_restores_on_exception(self):
        with pytest.raises(ValueError):
            with instrumentation_paused():
                raise ValueError("boom")
        assert is_instrumentation_enabled() is True

    def test_restores_previous_disabled_state(self):
        toggle_instrumentation(False)
        with instrumentation_paused():
            assert is_instrumentation_enabled() is False
        # Should restore to False (the state before entering)
        assert is_instrumentation_enabled() is False

    def test_nested_context_managers(self):
        assert is_instrumentation_enabled() is True
        with instrumentation_paused():
            assert is_instrumentation_enabled() is False
            with instrumentation_paused():
                assert is_instrumentation_enabled() is False
            # Inner restores to False (paused by outer)
            assert is_instrumentation_enabled() is False
        # Outer restores to True (original state)
        assert is_instrumentation_enabled() is True

    def test_context_manager_records_history(self):
        with instrumentation_paused():
            pass
        history = get_toggle_history()
        # pause on enter + restore on exit
        assert len(history) >= 2
