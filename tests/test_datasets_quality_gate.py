"""Tests for dataset quality gate."""

import pytest
from evalyn_sdk.datasets_quality_gate import (
    QualityCheck,
    QualityGateConfig,
    QualityGateResult,
    check_avg_input_length,
    check_duplicate_ids,
    check_empty_inputs,
    check_has_outputs,
    check_max_items,
    check_min_items,
    run_quality_gate,
    should_proceed,
)


def _items(n: int, input_text: str = "some test input text") -> list:
    return [{"id": str(i), "input": input_text, "output": f"output-{i}"} for i in range(n)]


class TestCheckMinItems:
    def test_pass(self):
        result = check_min_items(_items(10), QualityGateConfig(min_items=5))
        assert result.passed is True

    def test_fail(self):
        result = check_min_items(_items(2), QualityGateConfig(min_items=5))
        assert result.passed is False
        assert result.severity == "error"

    def test_exact_boundary(self):
        result = check_min_items(_items(5), QualityGateConfig(min_items=5))
        assert result.passed is True

    def test_empty(self):
        result = check_min_items([], QualityGateConfig(min_items=1))
        assert result.passed is False


class TestCheckMaxItems:
    def test_pass(self):
        result = check_max_items(_items(10), QualityGateConfig(max_items=100))
        assert result.passed is True

    def test_fail(self):
        result = check_max_items(_items(200), QualityGateConfig(max_items=100))
        assert result.passed is False
        assert result.severity == "error"

    def test_exact_boundary(self):
        result = check_max_items(_items(100), QualityGateConfig(max_items=100))
        assert result.passed is True


class TestCheckEmptyInputs:
    def test_pass(self):
        items = _items(10, "valid input")
        result = check_empty_inputs(items, QualityGateConfig(max_empty_inputs_pct=0.1))
        assert result.passed is True

    def test_fail(self):
        items = [{"id": str(i), "input": ""} for i in range(5)]
        items += _items(5, "valid")
        config = QualityGateConfig(max_empty_inputs_pct=0.1)
        result = check_empty_inputs(items, config)
        assert result.passed is False

    def test_no_input_field(self):
        items = [{"id": "0"}, {"id": "1", "input": "hello"}]
        result = check_empty_inputs(items, QualityGateConfig(max_empty_inputs_pct=0.1))
        assert result.passed is False  # 1 out of 2 = 50%

    def test_empty_items(self):
        result = check_empty_inputs([], QualityGateConfig())
        assert result.passed is True


class TestCheckAvgInputLength:
    def test_pass(self):
        items = _items(5, "a long enough input string")
        result = check_avg_input_length(items, QualityGateConfig(min_avg_input_length=10))
        assert result.passed is True

    def test_fail(self):
        items = _items(5, "hi")
        result = check_avg_input_length(items, QualityGateConfig(min_avg_input_length=10))
        assert result.passed is False

    def test_empty_items(self):
        result = check_avg_input_length([], QualityGateConfig())
        assert result.passed is True


class TestCheckHasOutputs:
    def test_not_required(self):
        items = [{"id": "0", "input": "test"}]
        result = check_has_outputs(items, QualityGateConfig(require_outputs=False))
        assert result.passed is True
        assert result.severity == "info"

    def test_required_and_present(self):
        items = [{"id": "0", "input": "test", "output": "answer"}]
        result = check_has_outputs(items, QualityGateConfig(require_outputs=True))
        assert result.passed is True

    def test_required_and_missing(self):
        items = [{"id": "0", "input": "test"}, {"id": "1", "input": "test2", "output": "ok"}]
        result = check_has_outputs(items, QualityGateConfig(require_outputs=True))
        assert result.passed is False
        assert result.severity == "error"

    def test_all_missing(self):
        items = [{"id": str(i), "input": f"q{i}"} for i in range(3)]
        result = check_has_outputs(items, QualityGateConfig(require_outputs=True))
        assert result.passed is False
        assert "3/3" in result.message


class TestCheckDuplicateIds:
    def test_no_duplicates(self):
        items = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
        result = check_duplicate_ids(items)
        assert result.passed is True

    def test_with_duplicates(self):
        items = [{"id": "a"}, {"id": "a"}, {"id": "b"}]
        result = check_duplicate_ids(items)
        assert result.passed is False
        assert "1 duplicate" in result.message

    def test_no_id_fields(self):
        items = [{"input": "x"}, {"input": "y"}]
        result = check_duplicate_ids(items)
        assert result.passed is True

    def test_empty(self):
        result = check_duplicate_ids([])
        assert result.passed is True


class TestQualityGateConfig:
    def test_defaults(self):
        cfg = QualityGateConfig()
        assert cfg.min_items == 1
        assert cfg.max_items == 100000
        assert cfg.fail_on_error is True

    def test_as_dict(self):
        cfg = QualityGateConfig(min_items=5, require_outputs=True)
        d = cfg.as_dict()
        assert d["min_items"] == 5
        assert d["require_outputs"] is True

    def test_from_dict(self):
        cfg = QualityGateConfig.from_dict({"min_items": 10, "fail_on_error": False})
        assert cfg.min_items == 10
        assert cfg.fail_on_error is False

    def test_roundtrip(self):
        cfg = QualityGateConfig(min_items=3, max_items=500, require_outputs=True)
        restored = QualityGateConfig.from_dict(cfg.as_dict())
        assert restored.min_items == cfg.min_items
        assert restored.max_items == cfg.max_items
        assert restored.require_outputs == cfg.require_outputs


class TestQualityGateResult:
    def test_as_dict_from_dict_roundtrip(self):
        result = QualityGateResult(
            passed=True,
            checks=[QualityCheck(name="test", passed=True, severity="info", message="ok")],
            errors=0,
            warnings=0,
            blocked=False,
        )
        restored = QualityGateResult.from_dict(result.as_dict())
        assert restored.passed is True
        assert len(restored.checks) == 1
        assert restored.checks[0].name == "test"

    def test_format_text_passed(self):
        result = QualityGateResult(
            passed=True,
            checks=[QualityCheck(name="min_items", passed=True, message="10 items")],
        )
        text = result.format_text()
        assert "PASSED" in text
        assert "min_items" in text

    def test_format_text_blocked(self):
        result = QualityGateResult(
            passed=False,
            checks=[QualityCheck(name="min_items", passed=False, severity="error", message="0 items")],
            errors=1,
            blocked=True,
        )
        text = result.format_text()
        assert "BLOCKED" in text
        assert "FAIL" in text


class TestRunQualityGate:
    def test_all_pass(self):
        items = _items(10, "some reasonably long input text here")
        result = run_quality_gate(items)
        assert result.passed is True
        assert result.blocked is False
        assert result.errors == 0

    def test_blocked_on_error(self):
        result = run_quality_gate([], QualityGateConfig(min_items=5, fail_on_error=True))
        assert result.passed is False
        assert result.blocked is True
        assert result.errors >= 1

    def test_not_blocked_when_fail_on_error_false(self):
        result = run_quality_gate([], QualityGateConfig(min_items=5, fail_on_error=False))
        assert result.blocked is False

    def test_default_config(self):
        items = _items(5, "enough text for a valid input")
        result = run_quality_gate(items)
        assert result.passed is True

    def test_multiple_failures(self):
        # Short input, too few items
        items = [{"id": "0", "input": "x"}]
        config = QualityGateConfig(min_items=10, min_avg_input_length=50)
        result = run_quality_gate(items, config)
        assert result.errors >= 1
        assert result.blocked is True

    def test_warnings_do_not_block(self):
        # Short avg input is a warning, not error
        items = _items(5, "hi")
        config = QualityGateConfig(min_avg_input_length=100)
        result = run_quality_gate(items, config)
        assert result.warnings >= 1
        # Should not be blocked by warnings alone
        assert result.blocked is False


class TestShouldProceed:
    def test_true_when_passed(self):
        result = QualityGateResult(passed=True, blocked=False)
        assert should_proceed(result) is True

    def test_false_when_blocked(self):
        result = QualityGateResult(passed=False, blocked=True)
        assert should_proceed(result) is False

    def test_true_when_warnings_only(self):
        result = QualityGateResult(passed=False, warnings=2, blocked=False)
        assert should_proceed(result) is True
