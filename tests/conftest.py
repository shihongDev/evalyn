"""
Pytest fixtures and helpers for evalyn_sdk tests.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List
from uuid import uuid4

import pytest


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
SDK_ROOT = PROJECT_ROOT / "sdk"
EVALYN_SDK_PATH = SDK_ROOT / "evalyn_sdk"
TEST_DATA_DIR = PROJECT_ROOT / "data" / "test-output"
# Per-worker DB path for pytest-xdist parallel safety
_worker_id = os.environ.get("PYTEST_XDIST_WORKER", "gw0")
TEST_DB_PATH = PROJECT_ROOT / "data" / "test" / f"traces_{_worker_id}.sqlite"

# Add SDK to sys.path so all test files can import evalyn_sdk directly
sys.path.insert(0, str(SDK_ROOT))

# ---------------------------------------------------------------------------
# Shared timestamp constants for test data builders
# ---------------------------------------------------------------------------

T0 = datetime(2026, 3, 14, 10, 0, 0, tzinfo=timezone.utc)
T1 = datetime(2026, 3, 14, 10, 0, 1, 500000, tzinfo=timezone.utc)  # +1.5s
T2 = datetime(2026, 3, 14, 10, 0, 3, tzinfo=timezone.utc)  # +3s
T3 = datetime(2026, 3, 14, 10, 1, 0, tzinfo=timezone.utc)  # +1min

# Ensure test directories exist
TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
TEST_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# Set test database for all tests
os.environ["EVALYN_DB"] = str(TEST_DB_PATH)


# ---------------------------------------------------------------------------
# Shared RunAnalysis builder
# ---------------------------------------------------------------------------

from evalyn_sdk.analysis.core import RunAnalysis, MetricStats, ItemStats


def make_run_analysis(
    items: dict[str, dict[str, dict]],
    run_id: str = "test-run",
    dataset_name: str = "test-dataset",
    metric_ids: list[str] | None = None,
) -> RunAnalysis:
    """Build a RunAnalysis from {item_id: {metric_id: {passed, score, reason?}}} dict.

    Args:
        items: Mapping of item IDs to metric results.
        run_id: Run identifier (default "test-run").
        dataset_name: Dataset name (default "test-dataset").
        metric_ids: Unused - kept for API compatibility. Metrics are
            inferred from the *items* dict.

    Returns:
        A fully populated RunAnalysis instance.
    """
    item_stats = {}
    metric_stats_map: dict[str, MetricStats] = {}

    for item_id, metrics in items.items():
        ist = ItemStats(item_id=item_id)
        for metric_id, result in metrics.items():
            ist.metric_results[metric_id] = {
                "passed": result.get("passed", True),
                "score": result.get("score", 1.0),
                "reason": result.get("reason"),
                "details": {},
            }
            if result.get("passed", True):
                ist.metrics_passed += 1
            else:
                ist.metrics_failed += 1

            if metric_id not in metric_stats_map:
                metric_stats_map[metric_id] = MetricStats(
                    metric_id=metric_id, metric_type="objective"
                )
            ms = metric_stats_map[metric_id]
            ms.count += 1
            score = result.get("score", 1.0)
            if score is not None:
                ms.scores.append(score)
            if result.get("passed", True):
                ms.passed += 1
            else:
                ms.failed += 1
            ms.has_pass_fail = True

        item_stats[item_id] = ist

    failed_items = [iid for iid, ist in item_stats.items() if not ist.all_passed]

    return RunAnalysis(
        run_id=run_id,
        dataset_name=dataset_name,
        created_at="2026-01-01",
        total_items=len(item_stats),
        total_metrics=len(metric_stats_map),
        metric_stats=metric_stats_map,
        item_stats=item_stats,
        failed_items=failed_items,
    )


# ---------------------------------------------------------------------------
# CLI runner helper
# ---------------------------------------------------------------------------

class CLIResult:
    """Result of running a CLI command."""

    def __init__(self, returncode: int, stdout: str, stderr: str, command: List[str]):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.command = command
        self.success = returncode == 0

    def __repr__(self) -> str:
        return f"CLIResult(returncode={self.returncode}, stdout={len(self.stdout)} chars)"

    def assert_success(self) -> "CLIResult":
        """Assert the command succeeded."""
        if not self.success:
            raise AssertionError(
                f"Command failed: {' '.join(self.command)}\n"
                f"Return code: {self.returncode}\n"
                f"Stdout: {self.stdout}\n"
                f"Stderr: {self.stderr}"
            )
        return self

    def assert_failure(self) -> "CLIResult":
        """Assert the command failed."""
        if self.success:
            raise AssertionError(
                f"Command unexpectedly succeeded: {' '.join(self.command)}\n"
                f"Stdout: {self.stdout}"
            )
        return self

    def assert_output_contains(self, text: str) -> "CLIResult":
        """Assert stdout contains text."""
        if text not in self.stdout:
            raise AssertionError(
                f"Output does not contain '{text}'\n"
                f"Stdout: {self.stdout}"
            )
        return self

    def assert_stderr_contains(self, text: str) -> "CLIResult":
        """Assert stderr contains text."""
        if text not in self.stderr:
            raise AssertionError(
                f"Stderr does not contain '{text}'\n"
                f"Stderr: {self.stderr}"
            )
        return self


def run_cli(*args: str, timeout: int = 60, env: Dict[str, str] | None = None) -> CLIResult:
    """
    Run evalyn CLI command and return result.

    Args:
        *args: CLI arguments (e.g., "list-calls", "--limit", "5")
        timeout: Command timeout in seconds
        env: Additional environment variables

    Returns:
        CLIResult with stdout, stderr, returncode
    """
    cmd = ["python", "-m", "evalyn_sdk.cli"] + list(args)

    run_env = os.environ.copy()
    # Add sdk to PYTHONPATH so evalyn_sdk is importable
    run_env["PYTHONPATH"] = str(SDK_ROOT)
    if env:
        run_env.update(env)

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(PROJECT_ROOT),
        env=run_env,
    )

    return CLIResult(
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        command=cmd,
    )


# ---------------------------------------------------------------------------
# Test data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test data within test_data/."""
    path = TEST_DATA_DIR / f"tmp_{uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def temp_db(temp_dir: Path) -> Generator[Path, None, None]:
    """Create a temporary SQLite database path."""
    db_path = temp_dir / "test_evalyn.sqlite"
    yield db_path


@pytest.fixture(scope="session")
def persistent_test_dir() -> Path:
    """
    Get or create a persistent test data directory.

    This directory persists across test runs and can be used for:
    - Pre-generated test datasets
    - Cached test fixtures
    - Manual test data
    """
    path = TEST_DATA_DIR / "fixtures"
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture
def sample_dataset(temp_dir: Path) -> Generator[Path, None, None]:
    """Create a sample dataset for testing."""
    dataset_dir = temp_dir / "test-dataset"
    dataset_dir.mkdir(parents=True)

    # Create dataset.jsonl
    items = [
        {
            "id": str(uuid4()),
            "inputs": {"query": "What is Python?"},
            "expected": "Python is a programming language.",
            "metadata": {"source": "test"},
        },
        {
            "id": str(uuid4()),
            "inputs": {"query": "What is JavaScript?"},
            "expected": "JavaScript is a scripting language.",
            "metadata": {"source": "test"},
        },
        {
            "id": str(uuid4()),
            "inputs": {"query": "What is Rust?"},
            "expected": "Rust is a systems programming language.",
            "metadata": {"source": "test"},
        },
    ]

    dataset_file = dataset_dir / "dataset.jsonl"
    with open(dataset_file, "w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")

    # Create meta.json
    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project": "test",
        "version": "v1",
        "item_count": len(items),
        "filters": {},
    }

    meta_file = dataset_dir / "meta.json"
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)

    # Create metrics directory
    metrics_dir = dataset_dir / "metrics"
    metrics_dir.mkdir()

    # Create a basic metrics file
    metrics = [
        {
            "id": "output_nonempty",
            "type": "objective",
            "name": "Output Non-Empty",
            "description": "Check output is not empty",
        },
        {
            "id": "latency_ms",
            "type": "objective",
            "name": "Latency",
            "description": "Measure execution latency",
        },
    ]

    metrics_file = metrics_dir / "basic.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    yield dataset_dir


@pytest.fixture
def sample_metrics_file(temp_dir: Path) -> Generator[Path, None, None]:
    """Create a sample metrics file for testing."""
    metrics = [
        {
            "id": "output_nonempty",
            "type": "objective",
            "name": "Output Non-Empty",
            "description": "Check output is not empty",
        },
        {
            "id": "json_valid",
            "type": "objective",
            "name": "JSON Valid",
            "description": "Check output is valid JSON",
        },
    ]

    metrics_file = temp_dir / "test_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    yield metrics_file


@pytest.fixture
def sample_annotations_file(temp_dir: Path) -> Generator[Path, None, None]:
    """Create a sample annotations file for testing."""
    annotations = [
        {
            "id": str(uuid4()),
            "target_id": str(uuid4()),
            "label": True,
            "rationale": "Looks good",
            "annotator": "test_user",
            "source": "human",
            "confidence": 5,
        },
        {
            "id": str(uuid4()),
            "target_id": str(uuid4()),
            "label": False,
            "rationale": "Incorrect output",
            "annotator": "test_user",
            "source": "human",
            "confidence": 4,
        },
    ]

    annotations_file = temp_dir / "test_annotations.jsonl"
    with open(annotations_file, "w") as f:
        for ann in annotations:
            f.write(json.dumps(ann) + "\n")

    yield annotations_file


@pytest.fixture
def traced_function():
    """Create and run a traced function to generate test data."""
    from evalyn_sdk import eval

    @eval(project="test_project", version="v1")
    def sample_function(query: str) -> str:
        """A sample function for testing."""
        return f"Response to: {query}"

    # Run it once to create a trace
    sample_function("test query")

    return sample_function


# ---------------------------------------------------------------------------
# Environment fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clean_env(temp_dir: Path) -> Generator[Dict[str, str], None, None]:
    """Provide a clean environment with isolated storage."""
    env = {
        "EVALYN_DB": str(temp_dir / "evalyn.sqlite"),
        "EVALYN_DATA_DIR": str(temp_dir / "data"),
    }
    yield env


# ---------------------------------------------------------------------------
# Realistic test data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def realistic_eval_run_data() -> Dict[str, Any]:
    """Realistic eval run data: 10 items, 3 metrics, mixed scores, failures with reasons."""
    items = [f"item-{i:03d}" for i in range(10)]
    metric_results = []
    for item_id in items:
        idx = int(item_id.split("-")[1])

        # helpfulness: most pass, a few fail
        passed = idx not in (3, 7)
        metric_results.append({
            "metric_id": "helpfulness_accuracy",
            "item_id": item_id,
            "call_id": f"call-{item_id}",
            "score": 0.92 if passed else 0.28,
            "passed": passed,
            "details": {"reason": "Addresses query well" if passed else "Missed key points"},
            "input_tokens": 150 + idx * 10,
            "output_tokens": 80 + idx * 5,
            "model": "gemini-2.5-flash",
        })

        # safety: almost all pass
        passed = idx != 7
        metric_results.append({
            "metric_id": "toxicity_safety",
            "item_id": item_id,
            "call_id": f"call-{item_id}",
            "score": 0.98 if passed else 0.15,
            "passed": passed,
            "details": {"reason": "Content safe" if passed else "Borderline harmful suggestion"},
            "input_tokens": 120,
            "output_tokens": 60,
            "model": "gemini-2.5-flash",
        })

        # latency: score-only, no pass/fail
        metric_results.append({
            "metric_id": "latency_ms",
            "item_id": item_id,
            "call_id": f"call-{item_id}",
            "score": 250 + idx * 30,
            "passed": None,
            "details": {},
        })

    return {
        "id": str(uuid4()),
        "dataset_name": "realistic-test-project",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metric_results": metric_results,
        "metrics": [
            {"id": "helpfulness_accuracy", "name": "Helpfulness", "type": "subjective", "description": "Accuracy judge"},
            {"id": "toxicity_safety", "name": "Safety", "type": "subjective", "description": "Safety judge"},
            {"id": "latency_ms", "name": "Latency", "type": "objective", "description": "Execution time"},
        ],
        "judge_configs": [],
        "summary": {},
        "usage_summary": {
            "total_input_tokens": 2700,
            "total_output_tokens": 1350,
            "total_tokens": 4050,
            "total_cost_usd": 0.0082,
            "models_used": ["gemini-2.5-flash"],
        },
    }


@pytest.fixture
def realistic_dataset_with_runs(temp_dir: Path) -> Generator[Path, None, None]:
    """Full dataset dir: dataset.jsonl, meta.json, metrics/, eval_runs/ with 2 runs."""
    from datetime import timedelta

    dataset_dir = temp_dir / "realistic-dataset"
    dataset_dir.mkdir(parents=True)

    # dataset.jsonl: 10 items
    dataset_file = dataset_dir / "dataset.jsonl"
    with open(dataset_file, "w") as f:
        for i in range(10):
            item = {
                "id": f"item-{i:03d}",
                "input": {"query": f"Explain concept {i} in machine learning"},
                "output": f"Machine learning concept {i} involves processing data through algorithms to find patterns.",
                "metadata": {"source": "synthetic", "difficulty": "medium" if i < 5 else "hard"},
            }
            f.write(json.dumps(item) + "\n")

    # meta.json
    with open(dataset_dir / "meta.json", "w") as f:
        json.dump({
            "created_at": datetime.now(timezone.utc).isoformat(),
            "project": "realistic-test",
            "version": "v1",
            "item_count": 10,
        }, f, indent=2)

    # metrics/
    metrics_dir = dataset_dir / "metrics"
    metrics_dir.mkdir()
    with open(metrics_dir / "default.json", "w") as f:
        json.dump([
            {"id": "helpfulness_accuracy", "name": "Helpfulness", "type": "subjective", "description": "Quality"},
            {"id": "output_nonempty", "name": "Non-Empty", "type": "objective", "description": "Not empty"},
        ], f, indent=2)

    # eval_runs/ with 2 runs showing improvement
    eval_runs_dir = dataset_dir / "eval_runs"
    eval_runs_dir.mkdir()
    base_time = datetime(2026, 3, 10, tzinfo=timezone.utc)

    for run_idx in range(2):
        run_id = str(uuid4())
        created = (base_time + timedelta(days=run_idx)).isoformat()
        pass_rate = 0.6 + run_idx * 0.2  # 0.6 then 0.8

        metric_results = []
        for i in range(10):
            item_id = f"item-{i:03d}"
            passed = i < int(pass_rate * 10)
            metric_results.append({
                "metric_id": "helpfulness_accuracy",
                "item_id": item_id,
                "call_id": f"call-{item_id}",
                "score": 0.9 if passed else 0.3,
                "passed": passed,
                "details": {},
            })
            metric_results.append({
                "metric_id": "output_nonempty",
                "item_id": item_id,
                "call_id": f"call-{item_id}",
                "score": 1.0,
                "passed": True,
                "details": {},
            })

        run_data = {
            "id": run_id,
            "dataset_name": "realistic-test",
            "created_at": created,
            "metric_results": metric_results,
            "metrics": [
                {"id": "helpfulness_accuracy", "name": "Helpfulness", "type": "subjective", "description": "Quality"},
                {"id": "output_nonempty", "name": "Non-Empty", "type": "objective", "description": "Not empty"},
            ],
            "judge_configs": [],
            "summary": {},
        }

        run_dir = eval_runs_dir / run_id
        run_dir.mkdir()
        with open(run_dir / "results.json", "w") as f:
            json.dump(run_data, f, indent=2)

    yield dataset_dir


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "requires_api: marks tests that require API keys")
    config.addinivalue_line("markers", "integration: marks integration tests")
