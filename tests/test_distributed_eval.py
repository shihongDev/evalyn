"""Tests for evalyn_sdk.evaluation.distributed module."""

from __future__ import annotations

import pytest

from evalyn_sdk.evaluation.distributed import (
    DistributedConfig,
    DistributedReport,
    TaskMessage,
    TaskQueue,
    WorkerStatus,
    create_task_messages,
    estimate_distributed_time,
    partition_tasks,
    simulate_distributed_run,
)


# ---------------------------------------------------------------------------
# TaskMessage
# ---------------------------------------------------------------------------


class TestTaskMessage:
    def test_as_dict(self):
        t = TaskMessage(task_id="t1", item_id="i1", metric_id="m1")
        d = t.as_dict()
        assert d["task_id"] == "t1"
        assert d["item_id"] == "i1"
        assert d["metric_id"] == "m1"
        assert d["payload"] == {}
        assert d["priority"] == 0

    def test_from_dict(self):
        d = {"task_id": "t2", "item_id": "i2", "metric_id": "m2", "priority": 5}
        t = TaskMessage.from_dict(d)
        assert t.task_id == "t2"
        assert t.priority == 5

    def test_roundtrip(self):
        original = TaskMessage(
            task_id="t3", item_id="i3", metric_id="m3", payload={"k": "v"}, priority=2
        )
        restored = TaskMessage.from_dict(original.as_dict())
        assert restored.task_id == original.task_id
        assert restored.payload == original.payload
        assert restored.priority == original.priority

    def test_from_dict_defaults(self):
        t = TaskMessage.from_dict({})
        assert t.task_id == ""
        assert t.item_id == ""
        assert t.metric_id == ""
        assert t.payload == {}
        assert t.priority == 0


# ---------------------------------------------------------------------------
# WorkerStatus
# ---------------------------------------------------------------------------


class TestWorkerStatus:
    def test_as_dict(self):
        ws = WorkerStatus(worker_id="w1", status="busy", tasks_completed=3, current_task="t1")
        d = ws.as_dict()
        assert d["worker_id"] == "w1"
        assert d["status"] == "busy"
        assert d["tasks_completed"] == 3
        assert d["current_task"] == "t1"

    def test_defaults(self):
        ws = WorkerStatus(worker_id="w2")
        assert ws.status == "idle"
        assert ws.tasks_completed == 0
        assert ws.current_task == ""


# ---------------------------------------------------------------------------
# DistributedConfig
# ---------------------------------------------------------------------------


class TestDistributedConfig:
    def test_as_dict(self):
        cfg = DistributedConfig(num_workers=8, queue_type="redis")
        d = cfg.as_dict()
        assert d["num_workers"] == 8
        assert d["queue_type"] == "redis"

    def test_from_dict(self):
        cfg = DistributedConfig.from_dict({"num_workers": 16, "timeout_seconds": 30.0})
        assert cfg.num_workers == 16
        assert cfg.timeout_seconds == 30.0
        assert cfg.queue_type == "memory"

    def test_roundtrip(self):
        original = DistributedConfig(num_workers=2, queue_type="sqs", retry_failed=False, timeout_seconds=120.0)
        restored = DistributedConfig.from_dict(original.as_dict())
        assert restored.num_workers == original.num_workers
        assert restored.queue_type == original.queue_type
        assert restored.retry_failed == original.retry_failed
        assert restored.timeout_seconds == original.timeout_seconds


# ---------------------------------------------------------------------------
# DistributedReport
# ---------------------------------------------------------------------------


class TestDistributedReport:
    def test_as_dict(self):
        r = DistributedReport(total_tasks=10, completed=8, failed=2, avg_task_duration_ms=50.0, workers_used=4)
        d = r.as_dict()
        assert d["total_tasks"] == 10
        assert d["completed"] == 8
        assert d["failed"] == 2

    def test_from_dict(self):
        r = DistributedReport.from_dict({"total_tasks": 5, "completed": 5})
        assert r.total_tasks == 5
        assert r.failed == 0

    def test_roundtrip(self):
        original = DistributedReport(total_tasks=20, completed=18, failed=2, avg_task_duration_ms=75.5, workers_used=3)
        restored = DistributedReport.from_dict(original.as_dict())
        assert restored.total_tasks == original.total_tasks
        assert restored.avg_task_duration_ms == original.avg_task_duration_ms

    def test_format_text(self):
        r = DistributedReport(total_tasks=10, completed=9, failed=1, avg_task_duration_ms=42.5, workers_used=2)
        text = r.format_text()
        assert "Distributed Evaluation Report" in text
        assert "Total tasks: 10" in text
        assert "Completed: 9" in text
        assert "Failed: 1" in text
        assert "42.5" in text
        assert "Workers used: 2" in text


# ---------------------------------------------------------------------------
# TaskQueue
# ---------------------------------------------------------------------------


class TestTaskQueue:
    def test_enqueue_dequeue(self):
        q = TaskQueue()
        t = TaskMessage(task_id="t1", item_id="i1", metric_id="m1")
        q.enqueue(t)
        assert q.size() == 1
        result = q.dequeue()
        assert result is not None
        assert result.task_id == "t1"
        assert q.size() == 0

    def test_dequeue_empty(self):
        q = TaskQueue()
        assert q.dequeue() is None

    def test_is_empty(self):
        q = TaskQueue()
        assert q.is_empty()
        q.enqueue(TaskMessage(task_id="t1", item_id="i1", metric_id="m1"))
        assert not q.is_empty()

    def test_peek(self):
        q = TaskQueue()
        assert q.peek() is None
        t = TaskMessage(task_id="t1", item_id="i1", metric_id="m1")
        q.enqueue(t)
        peeked = q.peek()
        assert peeked is not None
        assert peeked.task_id == "t1"
        # peek should not remove the item
        assert q.size() == 1

    def test_fifo_order(self):
        q = TaskQueue()
        q.enqueue(TaskMessage(task_id="first", item_id="i1", metric_id="m1"))
        q.enqueue(TaskMessage(task_id="second", item_id="i2", metric_id="m2"))
        assert q.dequeue().task_id == "first"
        assert q.dequeue().task_id == "second"

    def test_size_multiple(self):
        q = TaskQueue()
        for i in range(5):
            q.enqueue(TaskMessage(task_id=f"t{i}", item_id="i", metric_id="m"))
        assert q.size() == 5


# ---------------------------------------------------------------------------
# create_task_messages
# ---------------------------------------------------------------------------


class TestCreateTaskMessages:
    def test_correct_count(self):
        tasks = create_task_messages(["i1", "i2"], ["m1", "m2", "m3"])
        assert len(tasks) == 6

    def test_single_item_single_metric(self):
        tasks = create_task_messages(["i1"], ["m1"])
        assert len(tasks) == 1
        assert tasks[0].item_id == "i1"
        assert tasks[0].metric_id == "m1"

    def test_empty_items(self):
        tasks = create_task_messages([], ["m1"])
        assert len(tasks) == 0

    def test_empty_metrics(self):
        tasks = create_task_messages(["i1"], [])
        assert len(tasks) == 0

    def test_unique_task_ids(self):
        tasks = create_task_messages(["i1", "i2"], ["m1", "m2"])
        ids = [t.task_id for t in tasks]
        assert len(set(ids)) == len(ids)


# ---------------------------------------------------------------------------
# partition_tasks
# ---------------------------------------------------------------------------


class TestPartitionTasks:
    def test_round_robin(self):
        tasks = create_task_messages(["i1", "i2", "i3"], ["m1"])
        parts = partition_tasks(tasks, 2)
        assert len(parts) == 2
        assert len(parts[0]) == 2  # tasks 0 and 2
        assert len(parts[1]) == 1  # task 1

    def test_single_worker(self):
        tasks = create_task_messages(["i1", "i2"], ["m1"])
        parts = partition_tasks(tasks, 1)
        assert len(parts) == 1
        assert len(parts[0]) == 2

    def test_more_workers_than_tasks(self):
        tasks = create_task_messages(["i1"], ["m1"])
        parts = partition_tasks(tasks, 4)
        assert len(parts) == 4
        non_empty = [p for p in parts if len(p) > 0]
        assert len(non_empty) == 1

    def test_empty_tasks(self):
        parts = partition_tasks([], 3)
        assert len(parts) == 3
        assert all(len(p) == 0 for p in parts)

    def test_zero_workers(self):
        parts = partition_tasks([TaskMessage(task_id="t", item_id="i", metric_id="m")], 0)
        assert parts == []


# ---------------------------------------------------------------------------
# simulate_distributed_run
# ---------------------------------------------------------------------------


class TestSimulateDistributedRun:
    def test_all_succeed(self):
        tasks = create_task_messages(["i1", "i2"], ["m1"])

        def noop(task):
            pass

        report = simulate_distributed_run(tasks, 2, noop)
        assert report.total_tasks == 2
        assert report.completed == 2
        assert report.failed == 0
        assert report.workers_used == 2

    def test_some_fail(self):
        tasks = create_task_messages(["i1", "i2", "i3"], ["m1"])
        call_count = {"n": 0}

        def maybe_fail(task):
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise ValueError("simulated failure")

        report = simulate_distributed_run(tasks, 1, maybe_fail)
        assert report.total_tasks == 3
        assert report.completed == 2
        assert report.failed == 1

    def test_empty_tasks(self):
        report = simulate_distributed_run([], 4, lambda t: None)
        assert report.total_tasks == 0
        assert report.completed == 0

    def test_workers_capped_by_tasks(self):
        tasks = create_task_messages(["i1"], ["m1"])
        report = simulate_distributed_run(tasks, 10, lambda t: None)
        assert report.workers_used == 1


# ---------------------------------------------------------------------------
# estimate_distributed_time
# ---------------------------------------------------------------------------


class TestEstimateDistributedTime:
    def test_basic(self):
        # 10 tasks, 2 workers, 100ms each -> ceil(10/2)*100 = 500
        result = estimate_distributed_time(10, 2, 100.0)
        assert result == 500.0

    def test_uneven_split(self):
        # 7 tasks, 3 workers -> ceil(7/3)=3 tasks per worker -> 3*100 = 300
        result = estimate_distributed_time(7, 3, 100.0)
        assert result == 300.0

    def test_single_worker(self):
        result = estimate_distributed_time(5, 1, 200.0)
        assert result == 1000.0

    def test_zero_workers(self):
        result = estimate_distributed_time(10, 0)
        assert result == 0.0

    def test_zero_tasks(self):
        result = estimate_distributed_time(0, 4)
        assert result == 0.0

    def test_default_avg(self):
        # Default avg_task_ms is 100.0
        result = estimate_distributed_time(4, 2)
        assert result == 200.0
