"""Tests for structured logging and Python evaluation API."""

import json
import logging
import os
import pytest
from pathlib import Path
from unittest.mock import patch
from evalyn_sdk.logging_config import (
    configure_logging, get_logger, JSONFormatter, TextFormatter,
)
from evalyn_sdk.models import DatasetItem
from evalyn_sdk.api import evaluate, analyze, EvalResult, _resolve_dataset


# =============================================================================
# Structured logging tests
# =============================================================================

class TestJSONFormatter:
    def test_produces_valid_json(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="test message", args=(), exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["message"] == "test message"
        assert parsed["level"] == "INFO"
        assert "timestamp" in parsed

    def test_includes_logger_name(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="evalyn_sdk.runner", level=logging.WARNING, pathname="", lineno=0,
            msg="warning", args=(), exc_info=None,
        )
        output = json.loads(formatter.format(record))
        assert output["logger"] == "evalyn_sdk.runner"


class TestTextFormatter:
    def test_produces_readable_output(self):
        formatter = TextFormatter()
        record = logging.LogRecord(
            name="test", level=logging.ERROR, pathname="", lineno=0,
            msg="error msg", args=(), exc_info=None,
        )
        output = formatter.format(record)
        assert "ERROR" in output
        assert "error msg" in output


class TestConfigureLogging:
    def test_default_config(self):
        configure_logging()
        logger = logging.getLogger("evalyn_sdk")
        assert logger.level == logging.WARNING

    def test_debug_level(self):
        configure_logging(level="debug")
        logger = logging.getLogger("evalyn_sdk")
        assert logger.level == logging.DEBUG

    def test_env_var_level(self):
        with patch.dict(os.environ, {"EVALYN_LOG_LEVEL": "info"}):
            configure_logging()
            logger = logging.getLogger("evalyn_sdk")
            assert logger.level == logging.INFO

    def test_json_format(self):
        configure_logging(json_format=True)
        logger = logging.getLogger("evalyn_sdk")
        assert any(isinstance(h.formatter, JSONFormatter) for h in logger.handlers)

    def test_file_output(self, tmp_path):
        log_file = str(tmp_path / "test.log")
        configure_logging(level="info", log_file=log_file)
        logger = logging.getLogger("evalyn_sdk")
        logger.info("test message")
        # Flush handlers
        for h in logger.handlers:
            h.flush()
        assert Path(log_file).exists()

    def test_no_duplicate_handlers(self):
        configure_logging()
        configure_logging()  # call twice
        logger = logging.getLogger("evalyn_sdk")
        # Should have exactly 1 stderr handler (not 2)
        stderr_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler)]
        assert len(stderr_handlers) == 1

    def teardown_method(self):
        # Reset logging after each test
        configure_logging(level="warning")


class TestGetLogger:
    def test_namespaced(self):
        logger = get_logger("runner")
        assert logger.name == "evalyn_sdk.runner"

    def test_different_names(self):
        l1 = get_logger("a")
        l2 = get_logger("b")
        assert l1.name != l2.name


# =============================================================================
# Python API tests
# =============================================================================

def _make_items(n=5):
    return [DatasetItem(
        id=f"item-{i}",
        input={"query": f"question {i}"},
        output=f"answer {i}",
    ) for i in range(n)]


class TestResolveDataset:
    def test_from_list(self):
        items = _make_items(3)
        resolved = _resolve_dataset(items)
        assert len(resolved) == 3

    def test_from_dicts(self):
        dicts = [{"id": "1", "input": "q", "output": "a"}]
        resolved = _resolve_dataset(dicts)
        assert len(resolved) == 1
        assert resolved[0].id == "1"

    def test_from_file(self, tmp_path):
        # Write JSONL dataset
        dataset_file = tmp_path / "dataset.jsonl"
        import json
        with open(dataset_file, "w") as f:
            for i in range(3):
                f.write(json.dumps({"id": f"i{i}", "input": f"q{i}", "output": f"a{i}"}) + "\n")
        resolved = _resolve_dataset(str(dataset_file))
        assert len(resolved) == 3

    def test_from_directory(self, tmp_path):
        dataset_file = tmp_path / "dataset.jsonl"
        import json
        with open(dataset_file, "w") as f:
            f.write(json.dumps({"id": "1", "input": "q", "output": "a"}) + "\n")
        resolved = _resolve_dataset(str(tmp_path))
        assert len(resolved) == 1

    def test_missing_path_raises(self):
        with pytest.raises(FileNotFoundError):
            _resolve_dataset("/nonexistent/path")


class TestEvaluateAPI:
    def test_basic_objective(self):
        items = _make_items(5)
        result = evaluate(items, metrics=["json_valid", "nonempty_output"])
        assert isinstance(result, EvalResult)
        assert result.total_items == 5
        assert 0.0 <= result.overall_pass_rate <= 1.0

    def test_with_name_and_tags(self):
        items = _make_items(3)
        result = evaluate(items, metrics=["nonempty_output"], name="test-run", tags=["ci"])
        assert result.run.name == "test-run"
        assert result.run.tags == ["ci"]

    def test_metric_summary(self):
        items = _make_items(3)
        result = evaluate(items, metrics=["nonempty_output"])
        summary = result.metric_summary
        assert "nonempty_output" in summary
        assert "pass_rate" in summary["nonempty_output"]

    def test_as_dict(self):
        items = _make_items(3)
        result = evaluate(items, metrics=["nonempty_output"])
        d = result.as_dict()
        assert "run_id" in d
        assert "overall_pass_rate" in d
        assert "metrics" in d

    def test_empty_dataset_raises(self):
        with pytest.raises(ValueError, match="empty"):
            evaluate([], metrics=["json_valid"])

    def test_no_metrics_raises(self):
        with pytest.raises(ValueError, match="No metrics"):
            evaluate(_make_items(3), metrics=[])


class TestAnalyzeAPI:
    def test_analyze_run(self):
        items = _make_items(3)
        result = evaluate(items, metrics=["nonempty_output"])
        analysis = analyze(result.run)
        assert analysis.total_items == 3
        assert isinstance(analysis, type(result.analysis))

    def test_analyze_dict(self):
        items = _make_items(3)
        result = evaluate(items, metrics=["nonempty_output"])
        analysis = analyze(result.run.as_dict())
        assert analysis.total_items == 3
