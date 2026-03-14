"""Tests for CLI analysis, runs, and trend commands using temp data."""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from conftest import run_cli, TEST_DATA_DIR


# ---------------------------------------------------------------------------
# Test data builders
# ---------------------------------------------------------------------------


def _make_eval_run_data(
    *,
    run_id=None,
    dataset_name="test-project",
    num_items=5,
    metric_ids=("helpfulness", "safety"),
    pass_rates=None,
    created_at=None,
):
    """Build a realistic eval run results.json dict.

    pass_rates: dict of metric_id -> float (0-1) controlling pass probability.
    """
    if run_id is None:
        run_id = str(uuid4())
    if created_at is None:
        created_at = datetime.now(timezone.utc).isoformat()
    if pass_rates is None:
        pass_rates = {m: 0.8 for m in metric_ids}

    metric_results = []
    items = [f"item-{i:03d}" for i in range(num_items)]

    for item_id in items:
        for metric_id in metric_ids:
            pr = pass_rates.get(metric_id, 0.5)
            # Deterministic pass/fail based on item index and pass rate
            item_idx = int(item_id.split("-")[1])
            passed = item_idx < int(pr * num_items)
            score = 0.95 if passed else 0.35
            metric_results.append({
                "metric_id": metric_id,
                "item_id": item_id,
                "call_id": f"call-{item_id}",
                "score": score,
                "passed": passed,
                "details": {"reason": "good" if passed else "needs improvement"},
            })

    metrics = [
        {"id": m, "name": m.title(), "type": "subjective", "description": f"Test {m}"}
        for m in metric_ids
    ]

    return {
        "id": run_id,
        "dataset_name": dataset_name,
        "created_at": created_at,
        "metric_results": metric_results,
        "metrics": metrics,
        "judge_configs": [],
        "summary": {},
        "usage_summary": {},
    }


def _create_dataset_with_runs(
    base_dir: Path,
    *,
    dataset_name="test-project",
    num_runs=2,
    num_items=5,
    metric_ids=("helpfulness", "safety"),
):
    """Create a complete dataset dir with eval_runs for testing."""
    dataset_dir = base_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Create dataset.jsonl
    dataset_file = dataset_dir / "dataset.jsonl"
    with open(dataset_file, "w") as f:
        for i in range(num_items):
            item = {
                "id": f"item-{i:03d}",
                "input": {"query": f"Question {i}"},
                "output": f"Answer to question {i}",
                "metadata": {"source": "test"},
            }
            f.write(json.dumps(item) + "\n")

    # Create meta.json
    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project": dataset_name,
        "version": "v1",
        "item_count": num_items,
    }
    with open(dataset_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Create metrics dir
    metrics_dir = dataset_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    metrics = [
        {"id": m, "name": m.title(), "type": "subjective", "description": f"Test {m}"}
        for m in metric_ids
    ]
    with open(metrics_dir / "default.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Create eval_runs with varying pass rates to simulate improvement
    eval_runs_dir = dataset_dir / "eval_runs"
    eval_runs_dir.mkdir(exist_ok=True)

    base_time = datetime(2026, 3, 1, tzinfo=timezone.utc)
    run_ids = []
    for run_idx in range(num_runs):
        run_id = str(uuid4())
        run_ids.append(run_id)
        created = (base_time + timedelta(days=run_idx)).isoformat()

        # Gradually improving pass rates
        pass_rates = {}
        for m in metric_ids:
            base_rate = 0.6 + (run_idx * 0.1)  # 0.6, 0.7, 0.8, ...
            pass_rates[m] = min(base_rate, 1.0)

        run_data = _make_eval_run_data(
            run_id=run_id,
            dataset_name=dataset_name,
            num_items=num_items,
            metric_ids=metric_ids,
            pass_rates=pass_rates,
            created_at=created,
        )

        run_dir = eval_runs_dir / run_id
        run_dir.mkdir()
        with open(run_dir / "results.json", "w") as f:
            json.dump(run_data, f, indent=2)

    return dataset_dir, run_ids


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def test_dir():
    path = TEST_DATA_DIR / f"cli_commands_{uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(scope="module")
def dataset_with_runs(test_dir):
    """Create a dataset with 3 eval runs."""
    dataset_dir, run_ids = _create_dataset_with_runs(
        test_dir,
        dataset_name="cmd-test-project",
        num_runs=3,
        num_items=5,
    )
    return dataset_dir, run_ids


@pytest.fixture(scope="module")
def storage_with_runs(test_dir, dataset_with_runs):
    """Populate a temp SQLite with eval runs for trend testing."""
    from evalyn_sdk.storage.sqlite import SQLiteStorage
    from evalyn_sdk.models import EvalRun

    db_path = test_dir / "cmd_test.sqlite"
    storage = SQLiteStorage(path=db_path)
    dataset_dir, run_ids = dataset_with_runs

    # Load each run from disk and store in SQLite
    eval_runs_dir = dataset_dir / "eval_runs"
    for run_id in run_ids:
        results_file = eval_runs_dir / run_id / "results.json"
        with open(results_file) as f:
            run_data = json.load(f)
        run = EvalRun.from_dict(run_data)
        storage.store_eval_run(run)

    yield storage, db_path
    storage.close()


# ---------------------------------------------------------------------------
# cmd_analyze tests
# ---------------------------------------------------------------------------


class TestCmdAnalyze:
    """Test the analyze command with prepared dataset."""

    def test_analyze_latest_run(self, dataset_with_runs):
        dataset_dir, run_ids = dataset_with_runs
        result = run_cli("analyze", "--dataset", str(dataset_dir))
        result.assert_success()
        # Should show EVAL RUN ANALYSIS header or pass rates
        assert "ANALYSIS" in result.stdout or "Pass Rate" in result.stdout or "pass_rate" in result.stdout

    def test_analyze_specific_run_via_storage(self, storage_with_runs, dataset_with_runs):
        _, db_path = storage_with_runs
        _, run_ids = dataset_with_runs
        result = run_cli(
            "analyze", "--run", run_ids[0],
            env={"EVALYN_DB": str(db_path)},
        )
        result.assert_success()

    def test_analyze_json_format(self, dataset_with_runs):
        dataset_dir, run_ids = dataset_with_runs
        result = run_cli(
            "analyze", "--dataset", str(dataset_dir), "--format", "json"
        )
        result.assert_success()
        data = json.loads(result.stdout)
        assert "run_id" in data or "id" in data
        assert "metrics" in data or "metric_stats" in data


# ---------------------------------------------------------------------------
# cmd_compare tests
# ---------------------------------------------------------------------------


class TestCmdCompare:
    """Test the compare command with two runs."""

    def test_compare_latest_two_runs(self, dataset_with_runs):
        dataset_dir, run_ids = dataset_with_runs
        result = run_cli(
            "compare",
            "--dataset", str(dataset_dir),
            "--latest",
        )
        result.assert_success()
        assert "COMPARISON" in result.stdout or "Run" in result.stdout or "compare" in result.stdout.lower()

    def test_compare_via_storage(self, storage_with_runs, dataset_with_runs):
        _, db_path = storage_with_runs
        _, run_ids = dataset_with_runs
        result = run_cli(
            "compare",
            "--run1", run_ids[0],
            "--run2", run_ids[1],
            env={"EVALYN_DB": str(db_path)},
        )
        result.assert_success()

    def test_compare_nonexistent_run_fails(self, storage_with_runs):
        _, db_path = storage_with_runs
        result = run_cli(
            "compare",
            "--run1", "nonexistent-run-id",
            "--run2", "also-nonexistent",
            env={"EVALYN_DB": str(db_path)},
        )
        # Should fail or show error
        assert not result.success or "error" in result.stdout.lower() or "not found" in result.stdout.lower()


# ---------------------------------------------------------------------------
# cmd_trend tests
# ---------------------------------------------------------------------------


class TestCmdTrend:
    """Test the trend command using prepared SQLite data."""

    def test_trend_with_project(self, storage_with_runs):
        _, db_path = storage_with_runs
        result = run_cli(
            "trend",
            "--project", "cmd-test-project",
            env={"EVALYN_DB": str(db_path)},
        )
        result.assert_success()
        assert "TREND" in result.stdout or "trend" in result.stdout.lower() or "Pass Rate" in result.stdout

    def test_trend_json_format(self, storage_with_runs):
        _, db_path = storage_with_runs
        result = run_cli(
            "trend",
            "--project", "cmd-test-project",
            "--format", "json",
            env={"EVALYN_DB": str(db_path)},
        )
        result.assert_success()
        data = json.loads(result.stdout)
        assert "project" in data
        assert data["project"] == "cmd-test-project"
        assert "runs_analyzed" in data
        assert data["runs_analyzed"] == 3
        assert "trends" in data
        assert "overall_delta" in data["trends"]

    def test_trend_empty_project(self, storage_with_runs):
        _, db_path = storage_with_runs
        result = run_cli(
            "trend",
            "--project", "nonexistent-project",
            env={"EVALYN_DB": str(db_path)},
        )
        # Should fail with "no runs found"
        assert not result.success or "no eval runs" in result.stdout.lower()

    def test_trend_missing_project_arg(self):
        result = run_cli("trend")
        assert not result.success or "error" in result.stdout.lower()


# ---------------------------------------------------------------------------
# cmd_list_runs tests
# ---------------------------------------------------------------------------


class TestCmdListRuns:
    """Test the list-runs command."""

    def test_list_runs_table(self, storage_with_runs):
        _, db_path = storage_with_runs
        result = run_cli(
            "list-runs",
            env={"EVALYN_DB": str(db_path)},
        )
        result.assert_success()
        # Should show at least one run
        assert "cmd-test-project" in result.stdout or "id" in result.stdout.lower()

    def test_list_runs_json(self, storage_with_runs):
        _, db_path = storage_with_runs
        result = run_cli(
            "list-runs",
            "--format", "json",
            env={"EVALYN_DB": str(db_path)},
        )
        result.assert_success()
        data = json.loads(result.stdout)
        assert isinstance(data, list)
        assert len(data) >= 3
        for run in data:
            assert "id" in run
            assert "dataset_name" in run

    def test_list_runs_with_limit(self, storage_with_runs):
        _, db_path = storage_with_runs
        result = run_cli(
            "list-runs",
            "--limit", "1",
            "--format", "json",
            env={"EVALYN_DB": str(db_path)},
        )
        result.assert_success()
        data = json.loads(result.stdout)
        assert len(data) == 1


# ---------------------------------------------------------------------------
# cmd_show_run tests
# ---------------------------------------------------------------------------


class TestCmdShowRun:
    """Test the show-run command."""

    def test_show_run_last(self, storage_with_runs):
        _, db_path = storage_with_runs
        result = run_cli(
            "show-run",
            "--last",
            env={"EVALYN_DB": str(db_path)},
        )
        result.assert_success()
        assert "Eval Run" in result.stdout or "cmd-test-project" in result.stdout

    def test_show_run_last_json(self, storage_with_runs):
        _, db_path = storage_with_runs
        result = run_cli(
            "show-run",
            "--last",
            "--format", "json",
            env={"EVALYN_DB": str(db_path)},
        )
        result.assert_success()
        data = json.loads(result.stdout)
        assert "id" in data
        assert "dataset_name" in data
        assert "metric_results" in data
        assert isinstance(data["metric_results"], list)
        assert len(data["metric_results"]) > 0

    def test_show_run_by_id(self, storage_with_runs, dataset_with_runs):
        _, db_path = storage_with_runs
        _, run_ids = dataset_with_runs
        result = run_cli(
            "show-run",
            "--id", run_ids[0],
            "--format", "json",
            env={"EVALYN_DB": str(db_path)},
        )
        result.assert_success()
        data = json.loads(result.stdout)
        assert data["id"] == run_ids[0]

    def test_show_run_by_short_id(self, storage_with_runs, dataset_with_runs):
        _, db_path = storage_with_runs
        _, run_ids = dataset_with_runs
        short_id = run_ids[0][:8]
        result = run_cli(
            "show-run",
            "--id", short_id,
            "--format", "json",
            env={"EVALYN_DB": str(db_path)},
        )
        result.assert_success()
        data = json.loads(result.stdout)
        assert data["id"] == run_ids[0]

    def test_show_run_missing_id(self):
        result = run_cli("show-run")
        assert not result.success or "error" in result.stdout.lower()


# ---------------------------------------------------------------------------
# cmd_status tests
# ---------------------------------------------------------------------------


class TestCmdStatus:
    """Test the status command."""

    def test_status_with_dataset(self, dataset_with_runs):
        dataset_dir, _ = dataset_with_runs
        result = run_cli("status", "--dataset", str(dataset_dir))
        result.assert_success()
        assert "DATASET" in result.stdout
        assert "METRICS" in result.stdout
        assert "EVAL RUNS" in result.stdout
