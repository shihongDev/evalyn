import time

import pytest

from evalyn_sdk import (
    EvalRunner,
    EvalTracer,
    DatasetItem,
    latency_metric,
    exact_match_metric,
    eval,
)
from evalyn_sdk.storage.sqlite import SQLiteStorage


def test_eval_decorator_records_call(tmp_path):
    storage = SQLiteStorage(tmp_path / "calls.sqlite")
    tracer = EvalTracer(storage=storage)

    @eval(tracer=tracer)
    def add(a: int, b: int) -> int:
        return a + b

    result = add(2, 3)
    assert result == 5

    calls = storage.list_calls()
    assert len(calls) == 1
    call = calls[0]
    assert call.function_name == "add"
    assert call.output == 5
    assert call.duration_ms is not None
    assert "code" in call.metadata
    assert call.metadata["code"]["source_hash"]
    assert "def add" in (call.metadata["code"]["source"] or "")


def test_runner_caches_by_inputs(tmp_path):
    storage = SQLiteStorage(tmp_path / "runner.sqlite")
    tracer = EvalTracer(storage=storage)

    calls = {"count": 0}

    @eval(tracer=tracer)
    def echo(user_input: str) -> str:
        calls["count"] += 1
        # simulate tiny delay
        time.sleep(0.01)
        return f"echo:{user_input}"

    dataset = [
        DatasetItem(id="1", inputs={"user_input": "hi"}, expected="echo:hi"),
        DatasetItem(id="2", inputs={"user_input": "hi"}, expected="echo:hi"),
    ]

    runner = EvalRunner(
        target_fn=echo,
        metrics=[latency_metric(), exact_match_metric()],
        tracer=tracer,
        storage=storage,
        dataset_name="cache-test",
        cache_enabled=True,
        instrument=False,  # already decorated
    )

    run = runner.run_dataset(dataset)

    # only one actual invocation should have happened due to cache
    assert calls["count"] == 1
    assert run.summary["metrics"]["exact_match"]["pass_rate"] == 1.0
    assert len(run.metric_results) == 4  # 2 metrics x 2 items
