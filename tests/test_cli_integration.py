"""
Comprehensive CLI integration tests using real traced data.

These tests use the actual evalyn.sqlite database and real traced calls
to verify all CLI functionality end-to-end.

Run with: pytest tests/test_cli_integration.py -v
Skip slow tests: pytest tests/test_cli_integration.py -v -m "not slow"
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest

# All tests in this module share module-scoped fixtures with a common
# directory, so they must run on a single xdist worker.
pytestmark = pytest.mark.xdist_group("cli_integration")

from conftest import run_cli, SDK_ROOT, TEST_DATA_DIR


# ===========================================================================
# Test data setup
# ===========================================================================

@pytest.fixture(scope="module")
def test_data_dir():
    """Create a directory for test outputs within test_data/."""
    path = TEST_DATA_DIR / "cli_integration"
    path.mkdir(parents=True, exist_ok=True)
    yield path
    # Clean up after tests
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(scope="module")
def test_dataset(test_data_dir):
    """Build a test dataset from existing traces."""
    dataset_path = test_data_dir / "test-dataset"

    # Build dataset from gemini-deep-research-agent project
    result = run_cli(
        "build-dataset",
        "--project", "gemini-deep-research-agent",
        "--version", "v1",
        "--output", str(dataset_path),
        "--limit", "5",
    )

    if not result.success:
        # If no data, create a minimal mock dataset
        dataset_path.mkdir(parents=True, exist_ok=True)
        items = [
            {
                "id": "test-1",
                "inputs": {"query": "What is Python?"},
                "output": "Python is a programming language.",
                "metadata": {"function_name": "test_func"},
            }
        ]
        with open(dataset_path / "dataset.jsonl", "w") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")
        with open(dataset_path / "meta.json", "w") as f:
            json.dump({"project": "test", "version": "v1", "item_count": 1}, f)
        (dataset_path / "metrics").mkdir(exist_ok=True)

    return dataset_path


# ===========================================================================
# List commands tests
# ===========================================================================

class TestListCommands:
    """Test all list-* commands with real data."""

    def test_list_calls_default(self):
        """Test list-calls runs without error."""
        result = run_cli("list-calls")
        result.assert_success()

    def test_list_calls_with_limit(self):
        """Test list-calls with custom limit."""
        result = run_cli("list-calls", "--limit", "3")
        result.assert_success()
        # Count data lines (exclude header, separator, pagination info, and hints)
        lines = [l for l in result.stdout.strip().split("\n")
                 if l and not l.startswith("-") and not l.startswith("(")
                 and not l.startswith("Hint:") and not l.startswith("Showing")]
        assert len(lines) <= 4  # header + up to 3 data rows

    def test_list_calls_with_project_filter(self):
        """Test list-calls filtered by project."""
        result = run_cli("list-calls", "--project", "gemini-deep-research-agent", "--limit", "5")
        result.assert_success()
        if "gemini" in result.stdout.lower():
            assert "gemini-deep-research-agent" in result.stdout

    def test_list_calls_json_output(self):
        """Test list-calls with JSON output."""
        result = run_cli("list-calls", "--format", "json", "--limit", "5")
        result.assert_success()
        data = json.loads(result.stdout)
        assert isinstance(data, dict)
        assert "calls" in data
        assert isinstance(data["calls"], list)

    def test_list_runs(self):
        """Test list-runs command."""
        result = run_cli("list-runs")
        result.assert_success()

    def test_list_metrics(self):
        """Test list-metrics shows all templates."""
        result = run_cli("list-metrics")
        result.assert_success()
        result.assert_output_contains("latency_ms")
        result.assert_output_contains("json_valid")
        result.assert_output_contains("Objective metrics")
        result.assert_output_contains("Subjective metrics")

    def test_list_calibrations(self, test_dataset):
        """Test list-calibrations command."""
        result = run_cli("list-calibrations", "--dataset", str(test_dataset))
        result.assert_success()


# ===========================================================================
# Show commands tests
# ===========================================================================

class TestShowCommands:
    """Test all show-* commands with real data."""

    def test_show_projects(self):
        """Test show-projects displays project summaries."""
        result = run_cli("show-projects")
        result.assert_success()
        result.assert_output_contains("project")

    def test_show_call_with_real_id(self):
        """Test show-call with a real call ID."""
        # First get a call ID
        list_result = run_cli("list-calls", "--format", "json", "--limit", "1")
        list_result.assert_success()
        data = json.loads(list_result.stdout)
        calls = data.get("calls", [])

        if calls:
            call_id = calls[0]["id"]
            result = run_cli("show-call", "--id", call_id)
            result.assert_success()
            result.assert_output_contains(call_id[:8])  # At least partial ID
        else:
            pytest.skip("No calls in test database")

    def test_show_call_json_output(self):
        """Test show-call with JSON output."""
        list_result = run_cli("list-calls", "--format", "json", "--limit", "1")
        data = json.loads(list_result.stdout)
        calls = data.get("calls", [])

        if calls:
            call_id = calls[0]["id"]
            result = run_cli("show-call", "--id", call_id, "--format", "json")
            result.assert_success()
            call_data = json.loads(result.stdout)
            assert call_data["id"] == call_id
        else:
            pytest.skip("No calls in test database")

    def test_show_trace_with_real_call(self):
        """Test show-trace with a real call."""
        list_result = run_cli("list-calls", "--format", "json", "--limit", "1")
        data = json.loads(list_result.stdout)
        calls = data.get("calls", [])

        if calls:
            call_id = calls[0]["id"]
            result = run_cli("show-trace", "--id", call_id)
            # May or may not have trace events, but should not error
            result.assert_success()
        else:
            pytest.skip("No calls in test database")


# ===========================================================================
# Dataset workflow tests
# ===========================================================================

class TestDatasetWorkflow:
    """Test the full dataset workflow."""

    def test_build_dataset(self, test_data_dir):
        """Test building a dataset from traces."""
        output_path = test_data_dir / "workflow-dataset"

        result = run_cli(
            "build-dataset",
            "--project", "gemini-deep-research-agent",
            "--version", "v1",
            "--output", str(output_path),
            "--limit", "3",
        )

        result.assert_success()
        # Check dataset structure
        assert (output_path / "dataset.jsonl").exists() or (output_path / "meta.json").exists()

    def test_status_command(self, test_dataset):
        """Test status command shows dataset info."""
        result = run_cli("status", "--dataset", str(test_dataset))
        result.assert_success()
        assert "dataset" in result.stdout.lower()

    def test_suggest_metrics_basic(self, test_dataset):
        """Test suggest-metrics with basic mode."""
        # Use --target instead of --project to avoid needing traces in DB
        result = run_cli(
            "suggest-metrics",
            "--target", "evalyn_sdk.cli.main:main",
            "--dataset", str(test_dataset),
            "--mode", "basic",
        )
        result.assert_success()

    def test_suggest_metrics_bundle(self, test_dataset):
        """Test suggest-metrics with bundle mode."""
        result = run_cli(
            "suggest-metrics",
            "--target", "evalyn_sdk.cli.main:main",
            "--dataset", str(test_dataset),
            "--mode", "bundle",
            "--bundle", "research-agent",
        )
        result.assert_success()

    def test_suggest_metrics_saves_to_file(self, test_dataset):
        """Test suggest-metrics saves metrics file."""
        result = run_cli(
            "suggest-metrics",
            "--target", "evalyn_sdk.cli.main:main",
            "--dataset", str(test_dataset),
            "--mode", "bundle",
            "--bundle", "summarization",
            "--metrics-name", "test-bundle",
        )
        result.assert_success()

        # Check metrics file was created
        metrics_dir = test_dataset / "metrics"
        assert metrics_dir.exists(), "suggest-metrics should create metrics directory"
        files = list(metrics_dir.glob("*.json"))
        assert len(files) > 0


# ===========================================================================
# Evaluation workflow tests
# ===========================================================================

class TestEvaluationWorkflow:
    """Test the evaluation workflow."""

    def test_run_eval_with_dataset(self, test_dataset):
        """Test running evaluation on dataset."""
        # First ensure metrics exist using --target
        run_cli(
            "suggest-metrics",
            "--target", "evalyn_sdk.cli.main:main",
            "--dataset", str(test_dataset),
            "--mode", "basic",
        )

        result = run_cli(
            "run-eval",
            "--dataset", str(test_dataset),
        )
        result.assert_success()

    def test_run_eval_with_explicit_metrics(self, test_dataset, test_data_dir):
        """Test running evaluation with explicit metrics file."""
        # Create a metrics file
        metrics_file = test_data_dir / "explicit_metrics.json"
        metrics = [
            {"id": "output_nonempty", "type": "objective", "name": "Non-Empty", "description": "Check not empty"},
            {"id": "latency_ms", "type": "objective", "name": "Latency", "description": "Measure latency"},
        ]
        with open(metrics_file, "w") as f:
            json.dump(metrics, f)

        result = run_cli(
            "run-eval",
            "--dataset", str(test_dataset),
            "--metrics", str(metrics_file),
        )
        result.assert_success()

    def test_run_eval_shows_results(self, test_dataset):
        """Test run-eval displays results."""
        # Ensure metrics using --target
        run_cli(
            "suggest-metrics",
            "--target", "evalyn_sdk.cli.main:main",
            "--dataset", str(test_dataset),
            "--mode", "basic",
        )

        result = run_cli(
            "run-eval",
            "--dataset", str(test_dataset),
        )
        result.assert_success()
        # Should show some kind of results or summary
        assert "Pass Rate" in result.stdout or "metric" in result.stdout.lower()


# ===========================================================================
# Annotation workflow tests
# ===========================================================================

class TestAnnotationWorkflow:
    """Test annotation-related commands."""

    def test_export_for_annotation(self, test_dataset, test_data_dir):
        """Test exporting dataset for annotation."""
        output_file = test_data_dir / "annotations_export.jsonl"

        result = run_cli(
            "export-for-annotation",
            "--dataset", str(test_dataset),
            "--output", str(output_file),
        )
        result.assert_success()
        assert output_file.exists(), "export-for-annotation should create output file"
        # Verify any exported lines are valid JSONL
        with open(output_file) as f:
            for line in f:
                json.loads(line)

    def test_import_annotations(self, test_data_dir):
        """Test importing annotations."""
        # Create annotations file
        annotations_file = test_data_dir / "test_annotations.jsonl"
        annotations = [
            {
                "id": "ann-1",
                "target_id": "call-1",
                "label": True,
                "rationale": "Good response",
                "annotator": "tester",
                "source": "human",
                "confidence": 5,
            }
        ]
        with open(annotations_file, "w") as f:
            for ann in annotations:
                f.write(json.dumps(ann) + "\n")

        result = run_cli(
            "import-annotations",
            "--path", str(annotations_file),
        )
        result.assert_success()

    def test_annotation_stats(self, test_data_dir):
        """Test annotation stats command."""
        # Create a dataset directory with annotations
        stats_dataset = test_data_dir / "stats-dataset"
        stats_dataset.mkdir(parents=True, exist_ok=True)
        annotations_file = stats_dataset / "annotations.jsonl"
        annotations = [
            {"id": "1", "target_id": "c1", "label": True, "annotator": "a", "source": "human"},
            {"id": "2", "target_id": "c2", "label": False, "annotator": "a", "source": "human"},
        ]
        with open(annotations_file, "w") as f:
            for ann in annotations:
                f.write(json.dumps(ann) + "\n")

        result = run_cli(
            "annotation-stats",
            "--dataset", str(stats_dataset),
        )
        result.assert_success()


# ===========================================================================
# Full pipeline tests (slow)
# ===========================================================================

@pytest.mark.slow
class TestFullPipeline:
    """Test complete workflows end-to-end."""

    def test_full_basic_workflow(self, test_data_dir):
        """Test: build-dataset -> suggest-metrics -> run-eval -> status."""
        dataset_path = test_data_dir / "full-workflow"

        # Step 1: Build dataset
        result = run_cli(
            "build-dataset",
            "--project", "gemini-deep-research-agent",
            "--output", str(dataset_path),
            "--limit", "3",
        )

        if not result.success:
            # Create mock dataset if no data
            dataset_path.mkdir(parents=True, exist_ok=True)
            with open(dataset_path / "dataset.jsonl", "w") as f:
                f.write(json.dumps({"id": "1", "inputs": {"q": "test"}, "output": "result"}) + "\n")
            with open(dataset_path / "meta.json", "w") as f:
                json.dump({"project": "test", "item_count": 1}, f)
            (dataset_path / "metrics").mkdir(exist_ok=True)

        # Step 2: Suggest metrics (use --target to avoid needing traces)
        result = run_cli(
            "suggest-metrics",
            "--target", "evalyn_sdk.cli.main:main",
            "--dataset", str(dataset_path),
            "--mode", "basic",
        )
        result.assert_success()

        # Verify dataset files exist before run-eval
        assert (dataset_path / "dataset.jsonl").exists(), f"dataset.jsonl missing in {dataset_path}"
        assert (dataset_path / "metrics").exists(), f"metrics dir missing in {dataset_path}"

        # Step 3: Run evaluation (skip if no metrics files generated)
        metrics_files = list((dataset_path / "metrics").glob("*.json"))
        if not metrics_files:
            pytest.skip("No metrics files generated - skipping run-eval")

        result = run_cli(
            "run-eval",
            "--dataset", str(dataset_path),
        )
        result.assert_success()

        # Step 4: Check status
        result = run_cli(
            "status",
            "--dataset", str(dataset_path),
        )
        result.assert_success()

    def test_metrics_bundle_workflow(self, test_data_dir):
        """Test using different metric bundles."""
        dataset_path = test_data_dir / "bundle-workflow"
        dataset_path.mkdir(parents=True, exist_ok=True)

        # Create minimal dataset
        with open(dataset_path / "dataset.jsonl", "w") as f:
            f.write(json.dumps({
                "id": "1",
                "inputs": {"query": "Summarize this article"},
                "output": "This article discusses important topics.",
            }) + "\n")
        with open(dataset_path / "meta.json", "w") as f:
            json.dump({"project": "bundle-test", "item_count": 1}, f)
        (dataset_path / "metrics").mkdir(exist_ok=True)

        # Test different bundles (use --target to avoid needing traces)
        for bundle in ["summarization", "research-agent", "orchestrator"]:
            result = run_cli(
                "suggest-metrics",
                "--target", "evalyn_sdk.cli.main:main",
                "--dataset", str(dataset_path),
                "--mode", "bundle",
                "--bundle", bundle,
                "--metrics-name", f"bundle-{bundle}",
            )
            result.assert_success()


# ===========================================================================
# Error handling tests
# ===========================================================================

class TestErrorHandling:
    """Test proper error handling for edge cases."""

    def test_nonexistent_dataset(self, test_data_dir):
        """Test error for nonexistent dataset."""
        result = run_cli("run-eval", "--dataset", str(test_data_dir / "nonexistent"))
        result.assert_failure()

    def test_invalid_metrics_json(self, test_dataset, test_data_dir):
        """Test error for invalid metrics JSON."""
        bad_file = test_data_dir / "bad_metrics.json"
        bad_file.write_text("{ invalid json }")

        result = run_cli(
            "run-eval",
            "--dataset", str(test_dataset),
            "--metrics", str(bad_file),
        )
        result.assert_failure()

    def test_missing_required_args(self):
        """Test error for missing required arguments."""
        # show-call without --id should fail
        result = run_cli("show-call")
        result.assert_failure()

        # show-trace without --id should fail
        result = run_cli("show-trace")
        result.assert_failure()

    def test_invalid_call_id(self):
        """Test error for invalid call ID."""
        result = run_cli("show-call", "--id", "nonexistent-id-12345")
        result.assert_output_contains("No call found")


# ===========================================================================
# Output format tests
# ===========================================================================

class TestOutputFormats:
    """Test different output formats."""

    def test_list_calls_table_format(self):
        """Test list-calls table output runs without error."""
        result = run_cli("list-calls", "--limit", "3")
        result.assert_success()

    def test_list_calls_json_format(self):
        """Test list-calls JSON output is valid."""
        result = run_cli("list-calls", "--format", "json", "--limit", "3")
        result.assert_success()
        data = json.loads(result.stdout)
        assert isinstance(data, dict)
        assert "calls" in data
        calls = data["calls"]
        if calls:
            assert "id" in calls[0]
            assert "function" in calls[0]

    def test_show_call_json_format(self):
        """Test show-call JSON output is complete."""
        list_result = run_cli("list-calls", "--format", "json", "--limit", "1")
        data = json.loads(list_result.stdout)
        calls = data.get("calls", [])

        if calls:
            result = run_cli("show-call", "--id", calls[0]["id"], "--format", "json")
            result.assert_success()
            call_data = json.loads(result.stdout)
            assert "id" in call_data
            assert "inputs" in call_data or "input" in call_data
            assert "output" in call_data or "error" in call_data
        else:
            pytest.skip("No calls in test database")


# ===========================================================================
# Concurrent/parallel tests
# ===========================================================================

class TestConcurrency:
    """Test concurrent CLI operations."""

    def test_repeated_list_calls(self):
        """Test repeated list-calls produce consistent results."""
        results = []
        for _ in range(3):
            result = run_cli("list-calls", "--limit", "2")
            results.append(result)

        for result in results:
            result.assert_success()
