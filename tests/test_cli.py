"""
Comprehensive test suite for evalyn CLI commands.

Run with: pytest tests/test_cli.py -v
Fast tests only: pytest tests/test_cli.py -v -m "not slow"
"""

from __future__ import annotations

import json

import pytest

from conftest import run_cli


# ===========================================================================
# Basic CLI tests
# ===========================================================================


class TestCLIBasic:
    """Basic CLI functionality tests."""

    def test_help(self):
        """Test --help flag."""
        result = run_cli("--help")
        result.assert_success()
        result.assert_output_contains("Evalyn CLI")
        result.assert_output_contains("list-calls")
        result.assert_output_contains("run-eval")
        result.assert_output_contains("delete-traces")

    def test_version(self):
        """Test --version flag."""
        result = run_cli("--version")
        result.assert_success()
        result.assert_output_contains("evalyn")

    def test_help_command(self):
        """Test 'help' subcommand."""
        result = run_cli("help")
        result.assert_success()
        # Help command shows ASCII banner and usage with subcommands
        result.assert_output_contains("Evalyn CLI")

    def test_unknown_command(self):
        """Test unknown command gives helpful error."""
        result = run_cli("nonexistent-command")
        result.assert_failure()

    def test_insights_help(self):
        """Test insights --help flag."""
        result = run_cli("insights", "--help")
        result.assert_success()
        result.assert_output_contains("--format")
        result.assert_output_contains("--latest")

    def test_insights_in_help(self):
        """Test insights command appears in main help."""
        result = run_cli("--help")
        result.assert_success()
        result.assert_output_contains("insights")


# ===========================================================================
# list-calls tests
# ===========================================================================


class TestListCalls:
    """Tests for 'evalyn list-calls' command."""

    def test_list_calls_help(self):
        """Test list-calls --help."""
        result = run_cli("list-calls", "--help")
        result.assert_success()
        result.assert_output_contains("--limit")

    def test_list_calls_default(self):
        """Test list-calls with default options."""
        result = run_cli("list-calls")
        result.assert_success()

    def test_list_calls_with_limit(self):
        """Test list-calls with --limit."""
        result = run_cli("list-calls", "--limit", "5")
        result.assert_success()

    def test_list_calls_with_project_filter(self):
        """Test list-calls with --project filter."""
        result = run_cli("list-calls", "--project", "nonexistent_project")
        result.assert_success()

    def test_list_calls_json_output(self):
        """Test list-calls with --json output."""
        result = run_cli("list-calls", "--format", "json", "--limit", "5")
        result.assert_success()
        # Should be valid JSON with pagination info
        try:
            data = json.loads(result.stdout)
            assert isinstance(data, dict)
            assert "calls" in data
            assert isinstance(data["calls"], list)
            assert "total" in data
            assert "showing" in data
            assert "offset" in data
        except json.JSONDecodeError:
            pytest.fail(f"Invalid JSON output: {result.stdout}")


# ===========================================================================
# show-call tests
# ===========================================================================


class TestShowCall:
    """Tests for 'evalyn show-call' command."""

    def test_show_call_help(self):
        """Test show-call --help."""
        result = run_cli("show-call", "--help")
        result.assert_success()
        result.assert_output_contains("--id")

    def test_show_call_missing_id(self):
        """Test show-call without --id fails."""
        result = run_cli("show-call")
        result.assert_failure()

    def test_show_call_nonexistent_id(self):
        """Test show-call with nonexistent ID."""
        result = run_cli("show-call", "--id", "nonexistent-id-12345")
        # Should show "No call found" message
        assert "no call found" in result.stdout.lower()


# ===========================================================================
# show-trace tests
# ===========================================================================


class TestShowTrace:
    """Tests for 'evalyn show-trace' command."""

    def test_show_trace_help(self):
        """Test show-trace --help."""
        result = run_cli("show-trace", "--help")
        result.assert_success()
        result.assert_output_contains("--id")

    def test_show_trace_missing_call(self):
        """Test show-trace without --id fails."""
        result = run_cli("show-trace")
        result.assert_failure()


# ===========================================================================
# show-projects tests
# ===========================================================================


class TestShowProjects:
    """Tests for 'evalyn show-projects' command."""

    def test_show_projects_help(self):
        """Test show-projects --help."""
        result = run_cli("show-projects", "--help")
        result.assert_success()

    def test_show_projects_default(self):
        """Test show-projects with default options."""
        result = run_cli("show-projects")
        result.assert_success()


# ===========================================================================
# list-metrics tests
# ===========================================================================


class TestListMetrics:
    """Tests for 'evalyn list-metrics' command."""

    def test_list_metrics_help(self):
        """Test list-metrics --help."""
        result = run_cli("list-metrics", "--help")
        result.assert_success()

    def test_list_metrics_default(self):
        """Test list-metrics shows both objective and subjective."""
        result = run_cli("list-metrics")
        result.assert_success()
        result.assert_output_contains("Objective metrics")
        result.assert_output_contains("Subjective metrics")

    def test_list_metrics_has_common_metrics(self):
        """Test list-metrics includes common metrics."""
        result = run_cli("list-metrics")
        result.assert_success()
        result.assert_output_contains("latency_ms")
        result.assert_output_contains("json_valid")


# ===========================================================================
# build-dataset tests
# ===========================================================================


class TestBuildDataset:
    """Tests for 'evalyn build-dataset' command."""

    def test_build_dataset_help(self):
        """Test build-dataset --help."""
        result = run_cli("build-dataset", "--help")
        result.assert_success()
        result.assert_output_contains("--project")

    def test_build_dataset_missing_project(self):
        """Test build-dataset without --project succeeds if zero or one project exists."""
        result = run_cli("build-dataset")
        # With empty/single-project DB, the command succeeds (writes 0 items).
        # With multiple projects it would fail, asking user to specify --project.
        result.assert_success()

    def test_build_dataset_help_shows_sampling_flags(self):
        """Test build-dataset --help shows sampling flags."""
        result = run_cli("build-dataset", "--help")
        result.assert_success()
        result.assert_output_contains("--mode")
        result.assert_output_contains("--deduplicate")
        result.assert_output_contains("--dedup-threshold")
        result.assert_output_contains("--n-clusters")
        result.assert_output_contains("--seed")

    def test_build_dataset_mode_choices(self):
        """Test build-dataset --mode accepts valid modes and rejects invalid ones."""
        # Valid mode should not fail on argument parsing
        result = run_cli("build-dataset", "--help")
        result.assert_success()
        result.assert_output_contains("all")
        result.assert_output_contains("random")
        result.assert_output_contains("diverse")
        result.assert_output_contains("stratified")
        result.assert_output_contains("clustered")

    def test_build_dataset_invalid_mode(self):
        """Test build-dataset with invalid --mode fails."""
        result = run_cli("build-dataset", "--mode", "invalid_mode")
        result.assert_failure()


# ===========================================================================
# suggest-metrics tests
# ===========================================================================


class TestSuggestMetrics:
    """Tests for 'evalyn suggest-metrics' command."""

    def test_suggest_metrics_help(self):
        """Test suggest-metrics --help."""
        result = run_cli("suggest-metrics", "--help")
        result.assert_success()
        result.assert_output_contains("--mode")
        result.assert_output_contains("--target")

    def test_suggest_metrics_bundle_mode(self, sample_dataset):
        """Test suggest-metrics with bundle mode (no LLM needed)."""
        # Bundle mode requires --project or --target
        result = run_cli(
            "suggest-metrics",
            "--target",
            "evalyn_sdk.cli.main:main",
            "--dataset",
            str(sample_dataset),
            "--mode",
            "bundle",
            "--bundle",
            "research-agent",
        )
        result.assert_success()

    def test_suggest_metrics_basic_mode(self, sample_dataset):
        """Test suggest-metrics with basic heuristic mode."""
        # Basic mode requires --project or --target
        result = run_cli(
            "suggest-metrics",
            "--target",
            "evalyn_sdk.cli.main:main",
            "--dataset",
            str(sample_dataset),
            "--mode",
            "basic",
        )
        result.assert_success()


# ===========================================================================
# run-eval tests
# ===========================================================================


class TestRunEval:
    """Tests for 'evalyn run-eval' command."""

    def test_run_eval_help(self):
        """Test run-eval --help."""
        result = run_cli("run-eval", "--help")
        result.assert_success()
        result.assert_output_contains("--dataset")
        result.assert_output_contains("--metrics")

    def test_run_eval_missing_dataset(self):
        """Test run-eval without dataset fails."""
        result = run_cli("run-eval")
        result.assert_failure()

    def test_run_eval_with_dataset(self, sample_dataset, sample_metrics_file):
        """Test run-eval with dataset directory and explicit metrics."""
        result = run_cli(
            "run-eval",
            "--dataset",
            str(sample_dataset),
            "--metrics",
            str(sample_metrics_file),
        )
        result.assert_success()

    def test_run_eval_nonexistent_dataset(self, temp_dir):
        """Test run-eval with nonexistent dataset."""
        result = run_cli(
            "run-eval",
            "--dataset",
            str(temp_dir / "nonexistent"),
        )
        result.assert_failure()


# ===========================================================================
# list-runs tests
# ===========================================================================


class TestListRuns:
    """Tests for 'evalyn list-runs' command."""

    def test_list_runs_help(self):
        """Test list-runs --help."""
        result = run_cli("list-runs", "--help")
        result.assert_success()

    def test_list_runs_default(self):
        """Test list-runs with default options."""
        result = run_cli("list-runs")
        result.assert_success()


# ===========================================================================
# show-run tests
# ===========================================================================


class TestShowRun:
    """Tests for 'evalyn show-run' command."""

    def test_show_run_help(self):
        """Test show-run --help."""
        result = run_cli("show-run", "--help")
        result.assert_success()
        result.assert_output_contains("--id")


# ===========================================================================
# annotation commands tests
# ===========================================================================


class TestAnnotationCommands:
    """Tests for annotation-related commands."""

    def test_export_for_annotation_help(self):
        """Test export-for-annotation --help."""
        result = run_cli("export-for-annotation", "--help")
        result.assert_success()
        result.assert_output_contains("--dataset")

    def test_import_annotations_help(self):
        """Test import-annotations --help."""
        result = run_cli("import-annotations", "--help")
        result.assert_success()
        result.assert_output_contains("--path")

    def test_annotation_stats_help(self):
        """Test annotation-stats --help."""
        result = run_cli("annotation-stats", "--help")
        result.assert_success()

    def test_annotate_help(self):
        """Test annotate --help."""
        result = run_cli("annotate", "--help")
        result.assert_success()
        result.assert_output_contains("--dataset")

    def test_import_annotations(self, sample_annotations_file):
        """Test importing annotations."""
        result = run_cli(
            "import-annotations",
            "--path",
            str(sample_annotations_file),
        )
        result.assert_success()


# ===========================================================================
# calibrate tests
# ===========================================================================


class TestCalibrate:
    """Tests for 'evalyn calibrate' command."""

    def test_calibrate_help(self):
        """Test calibrate --help."""
        result = run_cli("calibrate", "--help")
        result.assert_success()
        result.assert_output_contains("--metric-id")
        result.assert_output_contains("--annotations")


# ===========================================================================
# simulate tests
# ===========================================================================


class TestSimulate:
    """Tests for 'evalyn simulate' command."""

    def test_simulate_help(self):
        """Test simulate --help."""
        result = run_cli("simulate", "--help")
        result.assert_success()
        result.assert_output_contains("--dataset")
        result.assert_output_contains("--modes")

    def test_simulate_missing_dataset(self):
        """Test simulate without --dataset fails."""
        result = run_cli("simulate")
        result.assert_failure()


# ===========================================================================
# status tests
# ===========================================================================


class TestStatus:
    """Tests for 'evalyn status' command."""

    def test_status_help(self):
        """Test status --help."""
        result = run_cli("status", "--help")
        result.assert_success()

    def test_status_with_dataset(self, sample_dataset):
        """Test status with a dataset."""
        result = run_cli("status", "--dataset", str(sample_dataset))
        result.assert_success()
        result.assert_output_contains("DATASET")


# ===========================================================================
# list-calibrations tests
# ===========================================================================


class TestListCalibrations:
    """Tests for 'evalyn list-calibrations' command."""

    def test_list_calibrations_help(self):
        """Test list-calibrations --help."""
        result = run_cli("list-calibrations", "--help")
        result.assert_success()


# ===========================================================================
# init tests
# ===========================================================================


class TestInit:
    """Tests for 'evalyn init' command."""

    def test_init_help(self):
        """Test init --help."""
        result = run_cli("init", "--help")
        result.assert_success()


# ===========================================================================
# one-click tests
# ===========================================================================


class TestOneClick:
    """Tests for 'evalyn one-click' command."""

    def test_one_click_help(self):
        """Test one-click --help."""
        result = run_cli("one-click", "--help")
        result.assert_success()
        result.assert_output_contains("--project")


# ===========================================================================
# Integration tests (slower)
# ===========================================================================


@pytest.mark.slow
@pytest.mark.integration
class TestCLIIntegration:
    """Integration tests that run full workflows."""

    def test_full_workflow_basic_metrics(self, sample_dataset):
        """Test basic workflow: suggest metrics -> run eval."""
        # 1. Suggest metrics using --target (avoids needing traces in storage)
        result = run_cli(
            "suggest-metrics",
            "--target",
            "evalyn_sdk.cli.main:main",
            "--dataset",
            str(sample_dataset),
            "--mode",
            "basic",
            "--metrics-name",
            "test-basic",
        )
        result.assert_success()

        # 2. Run evaluation
        result = run_cli(
            "run-eval",
            "--dataset",
            str(sample_dataset),
        )
        result.assert_success()

        # 3. Check status
        result = run_cli("status", "--dataset", str(sample_dataset))
        result.assert_success()

    def test_dataset_with_bundle_metrics(self, sample_dataset):
        """Test using bundled metrics."""
        # Suggest metrics with --target (avoids needing traces in storage)
        result = run_cli(
            "suggest-metrics",
            "--target",
            "evalyn_sdk.cli.main:main",
            "--dataset",
            str(sample_dataset),
            "--mode",
            "bundle",
            "--bundle",
            "summarization",
        )
        result.assert_success()

        # Run eval
        result = run_cli(
            "run-eval",
            "--dataset",
            str(sample_dataset),
        )
        result.assert_success()


# ===========================================================================
# Error handling tests
# ===========================================================================


class TestErrorHandling:
    """Tests for proper error handling."""

    def test_invalid_json_metrics_file(self, temp_dir, sample_dataset):
        """Test handling of invalid JSON metrics file."""
        bad_file = temp_dir / "bad_metrics.json"
        bad_file.write_text("{ invalid json }")

        result = run_cli(
            "run-eval",
            "--dataset",
            str(sample_dataset),
            "--metrics",
            str(bad_file),
        )
        result.assert_failure()

    def test_empty_dataset(self, temp_dir):
        """Test handling of empty dataset directory."""
        empty_dir = temp_dir / "empty_dataset"
        empty_dir.mkdir()

        result = run_cli("run-eval", "--dataset", str(empty_dir))
        result.assert_failure()

    def test_missing_required_args(self):
        """Test commands with missing required arguments."""
        # show-call requires --id
        result = run_cli("show-call")
        result.assert_failure()

        # calibrate requires --metric-id and --annotations
        result = run_cli("calibrate")
        result.assert_failure()


# ===========================================================================
# Validate command tests
# ===========================================================================


class TestValidate:
    """Tests for 'evalyn validate' command."""

    def test_validate_help(self):
        """Test validate --help."""
        result = run_cli("validate", "--help")
        result.assert_success()
        result.assert_output_contains("--dataset")
        result.assert_output_contains("--latest")

    def test_validate_with_sample_dataset(self, sample_dataset):
        """Test validate on sample dataset."""
        result = run_cli("validate", "--dataset", str(sample_dataset))
        # Should succeed (exit 0) for valid dataset
        result.assert_success()
        result.assert_output_contains("Total items")
        result.assert_output_contains("Dataset is valid")

    def test_validate_nonexistent_dataset(self, temp_dir):
        """Test validate on nonexistent dataset."""
        result = run_cli("validate", "--dataset", str(temp_dir / "nonexistent"))
        result.assert_failure()

    def test_validate_no_dataset_specified(self):
        """Test validate without --dataset or --latest."""
        result = run_cli("validate")
        result.assert_failure()
        result.assert_output_contains("No dataset specified")


# ===========================================================================
# Analyze command tests
# ===========================================================================


class TestAnalyze:
    """Tests for 'evalyn analyze' command."""

    def test_analyze_help(self):
        """Test analyze --help."""
        result = run_cli("analyze", "--help")
        result.assert_success()
        result.assert_output_contains("--run")
        result.assert_output_contains("--dataset")

    def test_analyze_no_args(self):
        """Test analyze without args."""
        result = run_cli("analyze")
        result.assert_failure()
        result.assert_output_contains("Specify --run")


# ===========================================================================
# Compare command tests
# ===========================================================================


class TestCompare:
    """Tests for 'evalyn compare' command."""

    def test_compare_help(self):
        """Test compare --help."""
        result = run_cli("compare", "--help")
        result.assert_success()
        result.assert_output_contains("--run1")
        result.assert_output_contains("--run2")

    def test_compare_missing_run1(self):
        """Test compare without --run1."""
        result = run_cli("compare", "--run2", "some-id")
        result.assert_failure()

    def test_compare_missing_run2(self):
        """Test compare without --run2."""
        result = run_cli("compare", "--run1", "some-id")
        result.assert_failure()


# ===========================================================================
# Export command tests
# ===========================================================================


class TestExport:
    """Tests for 'evalyn export' command."""

    def test_export_help(self):
        """Test export --help."""
        result = run_cli("export", "--help")
        result.assert_success()
        result.assert_output_contains("--format")
        result.assert_output_contains("--output")
        result.assert_output_contains("json")
        result.assert_output_contains("csv")
        result.assert_output_contains("markdown")
        result.assert_output_contains("html")

    def test_export_no_args(self):
        """Test export without args."""
        result = run_cli("export")
        result.assert_failure()
        result.assert_output_contains("No eval run found")


# ===========================================================================
# Delete traces tests
# ===========================================================================


class TestDeleteTraces:
    """Tests for 'evalyn delete-traces' command."""

    def test_delete_traces_help(self):
        """Test delete-traces --help."""
        result = run_cli("delete-traces", "--help")
        result.assert_success()
        result.assert_output_contains("--id")
        result.assert_output_contains("-n")
        result.assert_output_contains("--db")
        result.assert_output_contains("--yes")

    def test_delete_traces_defaults_to_test_db(self):
        """Test delete-traces defaults to test database."""
        result = run_cli("delete-traces", "--help")
        result.assert_success()
        result.assert_output_contains("default: test")

    def test_delete_traces_no_traces(self):
        """Test delete-traces with no traces in test db (after deletion)."""
        # This should gracefully handle empty database
        result = run_cli("delete-traces", "-n", "0", "--db", "test", "--yes")
        result.assert_success()

    def test_delete_traces_with_nonexistent_id(self):
        """Test delete-traces with nonexistent ID shows warning."""
        result = run_cli(
            "delete-traces", "--id", "nonexistent-id-12345", "--db", "test", "--yes"
        )
        # Should show warning about not finding the ID
        assert (
            "warning" in result.stdout.lower()
            or "no traces found" in result.stdout.lower()
        )


# ===========================================================================
# Insights CLI flag tests
# ===========================================================================


class TestInsightsFlags:
    """Tests for insights command flags (--deep, --format html, --provider, etc)."""

    def test_insights_help_shows_new_flags(self):
        """Insights help should show --deep, --provider, --model, --experts, --format html."""
        result = run_cli("insights", "--help")
        result.assert_success()
        result.assert_output_contains("--deep")
        result.assert_output_contains("--provider")
        result.assert_output_contains("--model")
        result.assert_output_contains("--experts")

    def test_insights_format_html_in_help(self):
        """HTML should be listed as a format choice."""
        result = run_cli("insights", "--help")
        result.assert_success()
        result.assert_output_contains("html")

    def test_insights_provider_choices(self):
        """Provider arg should accept gemini, openai, ollama."""
        result = run_cli("insights", "--help")
        result.assert_success()
        result.assert_output_contains("gemini")
        result.assert_output_contains("openai")
        result.assert_output_contains("ollama")

    def test_insights_invalid_format(self):
        """Invalid format should fail."""
        result = run_cli("insights", "--format", "csv")
        result.assert_failure()


# ===========================================================================
# Quickstart tests
# ===========================================================================


class TestQuickstart:
    """Tests for quickstart pure functions and CLI flags."""

    def test_scan_file_for_frameworks_openai(self, tmp_path):
        """Detect openai import in a Python file."""
        from evalyn_sdk.cli.commands.quickstart import _scan_file_for_frameworks

        agent_file = tmp_path / "agent.py"
        agent_file.write_text("import openai\nclient = openai.OpenAI()\n")
        result = _scan_file_for_frameworks(agent_file)
        assert result == ["openai"]

    def test_scan_file_for_frameworks_anthropic(self, tmp_path):
        """Detect anthropic import in a Python file."""
        from evalyn_sdk.cli.commands.quickstart import _scan_file_for_frameworks

        agent_file = tmp_path / "agent.py"
        agent_file.write_text("import anthropic\nclient = anthropic.Anthropic()\n")
        result = _scan_file_for_frameworks(agent_file)
        assert result == ["anthropic"]

    def test_scan_file_for_frameworks_multiple(self, tmp_path):
        """Detect multiple frameworks (openai and langchain) in one file."""
        from evalyn_sdk.cli.commands.quickstart import _scan_file_for_frameworks

        agent_file = tmp_path / "agent.py"
        agent_file.write_text(
            "import openai\nfrom langchain import chains\n"
        )
        result = _scan_file_for_frameworks(agent_file)
        assert "openai" in result
        assert "langchain" in result

    def test_scan_file_for_frameworks_none(self, tmp_path):
        """No framework imports yields empty list."""
        from evalyn_sdk.cli.commands.quickstart import _scan_file_for_frameworks

        agent_file = tmp_path / "agent.py"
        agent_file.write_text("import os\nimport sys\nprint('hello')\n")
        result = _scan_file_for_frameworks(agent_file)
        assert result == []

    def test_scan_file_for_frameworks_only_top_50_lines(self, tmp_path):
        """Import on line 60 should NOT be detected (only first 50 lines scanned)."""
        from evalyn_sdk.cli.commands.quickstart import _scan_file_for_frameworks

        agent_file = tmp_path / "agent.py"
        lines = ["# filler\n"] * 59 + ["import openai\n"]
        agent_file.write_text("".join(lines))
        result = _scan_file_for_frameworks(agent_file)
        assert result == []

    def test_check_instrumentation_present_true(self, tmp_path):
        """File containing 'import evalyn_sdk' should return True."""
        from evalyn_sdk.cli.commands.quickstart import _check_instrumentation_present

        agent_file = tmp_path / "agent.py"
        agent_file.write_text("import evalyn_sdk\nimport openai\n")
        assert _check_instrumentation_present(str(agent_file)) is True

    def test_check_instrumentation_present_false(self, tmp_path):
        """File without evalyn_sdk import should return False."""
        from evalyn_sdk.cli.commands.quickstart import _check_instrumentation_present

        agent_file = tmp_path / "agent.py"
        agent_file.write_text("import openai\n")
        assert _check_instrumentation_present(str(agent_file)) is False

    def test_check_instrumentation_present_nonexistent_file(self):
        """Nonexistent file should return False."""
        from evalyn_sdk.cli.commands.quickstart import _check_instrumentation_present

        assert _check_instrumentation_present("/tmp/no_such_file_xyz.py") is False

    def test_quickstart_help(self):
        """quickstart --help shows key flags."""
        result = run_cli("quickstart", "--help")
        result.assert_success()
        result.assert_output_contains("--agent-file")
        result.assert_output_contains("--run")
        result.assert_output_contains("--timeout")


# ===========================================================================
# Dashboard tests
# ===========================================================================


class TestDashboard:
    """Tests for dashboard command."""

    def test_dashboard_help(self):
        """dashboard --help shows key flags."""
        result = run_cli("dashboard", "--help")
        result.assert_success()
        result.assert_output_contains("--run")
        result.assert_output_contains("--dataset")
        result.assert_output_contains("--output")

    def test_dashboard_no_data(self):
        """dashboard without data fails with appropriate message."""
        result = run_cli("dashboard")
        result.assert_failure()


# ===========================================================================
# _open_in_browser tests
# ===========================================================================


class TestOpenInBrowser:
    """Tests for the _open_in_browser helper."""

    def test_open_in_browser_returns_bool(self, tmp_path):
        """_open_in_browser should return a bool regardless of success."""
        from evalyn_sdk.cli.commands.dashboard import _open_in_browser

        fake_file = tmp_path / "test.html"
        fake_file.write_text("<html></html>")
        result = _open_in_browser(fake_file)
        assert isinstance(result, bool)

    def test_open_in_browser_success(self, tmp_path):
        """Successful subprocess call returns True."""
        from unittest.mock import patch
        from evalyn_sdk.cli.commands.dashboard import _open_in_browser

        fake_file = tmp_path / "test.html"
        fake_file.write_text("<html></html>")
        with patch("evalyn_sdk.cli.commands.dashboard.subprocess.run"):
            assert _open_in_browser(fake_file) is True

    def test_open_in_browser_failure(self, tmp_path):
        """Failed subprocess call returns False."""
        from unittest.mock import patch
        import subprocess
        from evalyn_sdk.cli.commands.dashboard import _open_in_browser

        fake_file = tmp_path / "test.html"
        fake_file.write_text("<html></html>")
        with patch(
            "evalyn_sdk.cli.commands.dashboard.subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "open"),
        ):
            assert _open_in_browser(fake_file) is False
