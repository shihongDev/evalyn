"""Tests for the progress dashboard module."""

from __future__ import annotations

import sys
import time
from pathlib import Path

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.progress_dashboard import (
    TaskProgress,
    DashboardConfig,
    DashboardState,
    create_dashboard,
    update_task,
    render_progress_bar,
    render_task_line,
    render_dashboard,
    get_overall_progress,
    is_complete,
)


# ---------------------------------------------------------------------------
# TaskProgress
# ---------------------------------------------------------------------------


class TestTaskProgress:
    def test_defaults(self):
        tp = TaskProgress(task_id="t0", name="test", progress=0.0, status="pending")
        assert tp.message == ""
        assert tp.items_done == 0
        assert tp.items_total == 0

    def test_custom_values(self):
        tp = TaskProgress(
            task_id="t1", name="eval", progress=0.5,
            status="running", message="halfway", items_done=25, items_total=50,
        )
        assert tp.progress == 0.5
        assert tp.items_done == 25

    def test_as_dict(self):
        tp = TaskProgress(task_id="t0", name="x", progress=0.3, status="running")
        d = tp.as_dict()
        assert d["task_id"] == "t0"
        assert d["progress"] == 0.3
        assert d["status"] == "running"
        assert d["message"] == ""

    def test_from_dict(self):
        d = {
            "task_id": "t1",
            "name": "eval",
            "progress": 0.75,
            "status": "complete",
            "message": "done",
            "items_done": 30,
            "items_total": 40,
        }
        tp = TaskProgress.from_dict(d)
        assert tp.task_id == "t1"
        assert tp.progress == 0.75
        assert tp.items_done == 30

    def test_from_dict_defaults(self):
        tp = TaskProgress.from_dict({"task_id": "t0", "name": "x"})
        assert tp.progress == 0.0
        assert tp.status == "pending"
        assert tp.message == ""

    def test_roundtrip(self):
        original = TaskProgress(
            task_id="t2", name="build", progress=1.0,
            status="complete", message="ok", items_done=10, items_total=10,
        )
        restored = TaskProgress.from_dict(original.as_dict())
        assert restored.task_id == original.task_id
        assert restored.progress == original.progress
        assert restored.items_total == original.items_total


# ---------------------------------------------------------------------------
# DashboardConfig
# ---------------------------------------------------------------------------


class TestDashboardConfig:
    def test_defaults(self):
        cfg = DashboardConfig()
        assert cfg.width == 60
        assert cfg.show_elapsed is True
        assert cfg.show_rate is True

    def test_custom(self):
        cfg = DashboardConfig(width=80, show_elapsed=False, show_rate=False)
        assert cfg.width == 80

    def test_as_dict(self):
        cfg = DashboardConfig(width=40)
        d = cfg.as_dict()
        assert d["width"] == 40

    def test_from_dict(self):
        cfg = DashboardConfig.from_dict({"width": 100, "show_elapsed": False})
        assert cfg.width == 100
        assert cfg.show_elapsed is False
        assert cfg.show_rate is True  # default

    def test_from_dict_defaults(self):
        cfg = DashboardConfig.from_dict({})
        assert cfg.width == 60

    def test_roundtrip(self):
        original = DashboardConfig(width=90, show_elapsed=False, show_rate=False)
        restored = DashboardConfig.from_dict(original.as_dict())
        assert restored.width == original.width
        assert restored.show_elapsed == original.show_elapsed


# ---------------------------------------------------------------------------
# DashboardState
# ---------------------------------------------------------------------------


class TestDashboardState:
    def test_defaults(self):
        state = DashboardState()
        assert state.tasks == []
        assert state.start_time == 0.0
        assert state.elapsed_seconds == 0.0

    def test_as_dict(self):
        tp = TaskProgress(task_id="t0", name="x", progress=0.0, status="pending")
        state = DashboardState(tasks=[tp], start_time=1000.0)
        d = state.as_dict()
        assert len(d["tasks"]) == 1
        assert d["start_time"] == 1000.0

    def test_from_dict(self):
        d = {
            "tasks": [
                {"task_id": "t0", "name": "a", "progress": 0.5, "status": "running"},
            ],
            "start_time": 500.0,
            "elapsed_seconds": 10.0,
        }
        state = DashboardState.from_dict(d)
        assert len(state.tasks) == 1
        assert state.tasks[0].name == "a"
        assert state.elapsed_seconds == 10.0

    def test_roundtrip(self):
        tp = TaskProgress(task_id="t0", name="test", progress=1.0, status="complete")
        original = DashboardState(tasks=[tp], start_time=100.0, elapsed_seconds=5.0)
        restored = DashboardState.from_dict(original.as_dict())
        assert len(restored.tasks) == len(original.tasks)
        assert restored.start_time == original.start_time


# ---------------------------------------------------------------------------
# create_dashboard
# ---------------------------------------------------------------------------


class TestCreateDashboard:
    def test_creates_pending_tasks(self):
        state = create_dashboard(["A", "B", "C"])
        assert len(state.tasks) == 3
        for t in state.tasks:
            assert t.status == "pending"
            assert t.progress == 0.0

    def test_task_ids(self):
        state = create_dashboard(["X", "Y"])
        assert state.tasks[0].task_id == "task_0"
        assert state.tasks[1].task_id == "task_1"

    def test_task_names(self):
        state = create_dashboard(["alpha", "beta"])
        assert state.tasks[0].name == "alpha"
        assert state.tasks[1].name == "beta"

    def test_start_time_set(self):
        before = time.time()
        state = create_dashboard(["t"])
        after = time.time()
        assert before <= state.start_time <= after

    def test_empty_task_list(self):
        state = create_dashboard([])
        assert state.tasks == []


# ---------------------------------------------------------------------------
# update_task
# ---------------------------------------------------------------------------


class TestUpdateTask:
    def test_updates_progress(self):
        state = create_dashboard(["A", "B"])
        updated = update_task(state, "task_0", progress=0.5)
        assert updated.tasks[0].progress == 0.5
        assert updated.tasks[1].progress == 0.0  # unchanged

    def test_does_not_mutate_original(self):
        state = create_dashboard(["A"])
        _ = update_task(state, "task_0", progress=0.5, status="running")
        assert state.tasks[0].progress == 0.0
        assert state.tasks[0].status == "pending"

    def test_updates_status(self):
        state = create_dashboard(["A"])
        updated = update_task(state, "task_0", progress=1.0, status="complete")
        assert updated.tasks[0].status == "complete"

    def test_updates_message(self):
        state = create_dashboard(["A"])
        updated = update_task(state, "task_0", progress=0.3, message="loading")
        assert updated.tasks[0].message == "loading"

    def test_updates_items_done(self):
        state = create_dashboard(["A"])
        updated = update_task(state, "task_0", progress=0.5, items_done=25)
        assert updated.tasks[0].items_done == 25

    def test_unknown_task_id(self):
        state = create_dashboard(["A"])
        updated = update_task(state, "nonexistent", progress=0.5)
        assert updated.tasks[0].progress == 0.0  # unchanged

    def test_elapsed_updates(self):
        state = create_dashboard(["A"])
        # Force start_time slightly in the past
        state = DashboardState(
            tasks=state.tasks,
            start_time=time.time() - 2.0,
            elapsed_seconds=0.0,
        )
        updated = update_task(state, "task_0", progress=0.5)
        assert updated.elapsed_seconds >= 1.5  # at least close to 2s


# ---------------------------------------------------------------------------
# render_progress_bar
# ---------------------------------------------------------------------------


class TestRenderProgressBar:
    def test_zero_progress(self):
        bar = render_progress_bar(0.0, width=10)
        assert "[" in bar
        assert "]" in bar
        assert "0%" in bar

    def test_full_progress(self):
        bar = render_progress_bar(1.0, width=10)
        assert "100%" in bar
        assert "==========" in bar

    def test_half_progress(self):
        bar = render_progress_bar(0.5, width=10)
        assert "50%" in bar
        assert ">" in bar

    def test_clamps_above_one(self):
        bar = render_progress_bar(1.5, width=10)
        assert "100%" in bar

    def test_clamps_below_zero(self):
        bar = render_progress_bar(-0.5, width=10)
        assert "0%" in bar

    def test_custom_width(self):
        bar = render_progress_bar(0.5, width=20)
        # Bar content between brackets is 20 chars
        inner = bar.split("[")[1].split("]")[0]
        assert len(inner) == 20


# ---------------------------------------------------------------------------
# render_task_line
# ---------------------------------------------------------------------------


class TestRenderTaskLine:
    def test_pending_task(self):
        tp = TaskProgress(task_id="t0", name="Build", progress=0.0, status="pending")
        line = render_task_line(tp)
        assert "[..]" in line
        assert "Build" in line

    def test_complete_task(self):
        tp = TaskProgress(task_id="t0", name="Test", progress=1.0, status="complete")
        line = render_task_line(tp)
        assert "[OK]" in line
        assert "100%" in line

    def test_failed_task(self):
        tp = TaskProgress(task_id="t0", name="Deploy", progress=0.3, status="failed")
        line = render_task_line(tp)
        assert "[!!]" in line

    def test_running_task(self):
        tp = TaskProgress(task_id="t0", name="Run", progress=0.5, status="running")
        line = render_task_line(tp)
        assert "[>>]" in line

    def test_items_shown(self):
        tp = TaskProgress(
            task_id="t0", name="Eval", progress=0.5,
            status="running", items_done=25, items_total=50,
        )
        line = render_task_line(tp)
        assert "25/50 items" in line

    def test_no_items_when_total_zero(self):
        tp = TaskProgress(task_id="t0", name="X", progress=0.0, status="pending")
        line = render_task_line(tp)
        assert "items" not in line

    def test_message_shown(self):
        tp = TaskProgress(
            task_id="t0", name="X", progress=0.5,
            status="running", message="processing",
        )
        line = render_task_line(tp)
        assert "processing" in line


# ---------------------------------------------------------------------------
# render_dashboard
# ---------------------------------------------------------------------------


class TestRenderDashboard:
    def test_renders_header(self):
        state = create_dashboard(["A", "B"])
        output = render_dashboard(state)
        assert "Progress:" in output

    def test_renders_all_tasks(self):
        state = create_dashboard(["Alpha", "Beta", "Gamma"])
        output = render_dashboard(state)
        assert "Alpha" in output
        assert "Beta" in output
        assert "Gamma" in output

    def test_renders_summary(self):
        state = create_dashboard(["A"])
        state = update_task(state, "task_0", progress=1.0, status="complete")
        output = render_dashboard(state)
        assert "1/1 tasks done" in output

    def test_with_config(self):
        state = create_dashboard(["A"])
        cfg = DashboardConfig(width=40, show_elapsed=False)
        output = render_dashboard(state, config=cfg)
        assert "elapsed" not in output

    def test_elapsed_shown_by_default(self):
        state = create_dashboard(["A"])
        output = render_dashboard(state)
        assert "elapsed" in output

    def test_empty_dashboard(self):
        state = create_dashboard([])
        output = render_dashboard(state)
        assert "Progress:" in output
        assert "0/0 tasks done" in output

    def test_multiline_output(self):
        state = create_dashboard(["A", "B"])
        output = render_dashboard(state)
        lines = output.strip().split("\n")
        # header + separator + 2 tasks + separator + summary = 6
        assert len(lines) >= 6


# ---------------------------------------------------------------------------
# get_overall_progress
# ---------------------------------------------------------------------------


class TestGetOverallProgress:
    def test_no_tasks(self):
        state = DashboardState()
        assert get_overall_progress(state) == 0.0

    def test_all_zero(self):
        state = create_dashboard(["A", "B"])
        assert get_overall_progress(state) == 0.0

    def test_all_complete(self):
        state = create_dashboard(["A", "B"])
        state = update_task(state, "task_0", progress=1.0, status="complete")
        state = update_task(state, "task_1", progress=1.0, status="complete")
        assert get_overall_progress(state) == 1.0

    def test_mixed_progress(self):
        state = create_dashboard(["A", "B"])
        state = update_task(state, "task_0", progress=0.5)
        state = update_task(state, "task_1", progress=1.0, status="complete")
        assert abs(get_overall_progress(state) - 0.75) < 0.01

    def test_single_task(self):
        state = create_dashboard(["A"])
        state = update_task(state, "task_0", progress=0.3)
        assert abs(get_overall_progress(state) - 0.3) < 0.01


# ---------------------------------------------------------------------------
# is_complete
# ---------------------------------------------------------------------------


class TestIsComplete:
    def test_empty(self):
        assert is_complete(DashboardState()) is True

    def test_all_pending(self):
        state = create_dashboard(["A", "B"])
        assert is_complete(state) is False

    def test_all_complete(self):
        state = create_dashboard(["A", "B"])
        state = update_task(state, "task_0", progress=1.0, status="complete")
        state = update_task(state, "task_1", progress=1.0, status="complete")
        assert is_complete(state) is True

    def test_mixed_complete_and_failed(self):
        state = create_dashboard(["A", "B"])
        state = update_task(state, "task_0", progress=1.0, status="complete")
        state = update_task(state, "task_1", progress=0.5, status="failed")
        assert is_complete(state) is True

    def test_one_still_running(self):
        state = create_dashboard(["A", "B"])
        state = update_task(state, "task_0", progress=1.0, status="complete")
        state = update_task(state, "task_1", progress=0.8, status="running")
        assert is_complete(state) is False

    def test_all_failed(self):
        state = create_dashboard(["A"])
        state = update_task(state, "task_0", progress=0.0, status="failed")
        assert is_complete(state) is True
