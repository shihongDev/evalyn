from evalyn_sdk.models import EvalRun, MetricResult, MetricSpec, now_utc
from evalyn_sdk.storage.sqlite import SQLiteStorage


def test_store_and_load_eval_run(tmp_path):
    storage = SQLiteStorage(tmp_path / "runs.sqlite")
    metric_spec = MetricSpec(id="m1", name="Exact", type="objective")
    result = MetricResult(
        metric_id="m1",
        item_id="item-1",
        call_id="call-1",
        score=1.0,
        passed=True,
        details={},
        raw_judge=None,
    )
    run = EvalRun(
        id="run-1",
        dataset_name="demo",
        created_at=now_utc(),
        metric_results=[result],
        metrics=[metric_spec],
        summary={"metrics": {"m1": {"count": 1, "avg_score": 1.0, "pass_rate": 1.0}}},
    )

    storage.store_eval_run(run)

    loaded = storage.get_eval_run("run-1")
    assert loaded is not None
    assert loaded.id == "run-1"
    assert loaded.metric_results[0].metric_id == "m1"

    runs = storage.list_eval_runs(limit=5)
    assert any(r.id == "run-1" for r in runs)
