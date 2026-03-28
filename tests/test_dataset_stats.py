"""Tests for dataset statistics and health check."""

import pytest
from evalyn_sdk.models import DatasetItem
from evalyn_sdk.analysis.dataset_stats import (
    compute_dataset_stats,
    check_dataset_health,
    DatasetStats,
    HealthCheckResult,
)


def _item(item_id: str, input_text: str = "question", output_text: str = "answer",
          human_label: dict = None, metadata: dict = None) -> DatasetItem:
    return DatasetItem(
        id=item_id,
        input=input_text,
        output=output_text,
        human_label=human_label,
        metadata=metadata or {},
    )


class TestComputeStats:
    def test_basic(self):
        items = [_item("1", "hello", "world"), _item("2", "foo bar", "baz")]
        stats = compute_dataset_stats(items)
        assert stats.item_count == 2
        assert len(stats.input_lengths) == 2
        assert len(stats.output_lengths) == 2

    def test_avg_lengths(self):
        items = [_item("1", "aa", "bbbb"), _item("2", "cccc", "dd")]
        stats = compute_dataset_stats(items)
        assert stats.avg_input_length == 3.0  # (2+4)/2
        assert stats.avg_output_length == 3.0  # (4+2)/2

    def test_max_lengths(self):
        items = [_item("1", "short", "x"), _item("2", "a very long input", "y")]
        stats = compute_dataset_stats(items)
        assert stats.max_input_length == len("a very long input")
        assert stats.max_output_length == 1

    def test_empty_dataset(self):
        stats = compute_dataset_stats([])
        assert stats.item_count == 0
        assert stats.avg_input_length == 0.0
        assert stats.reference_coverage == 0.0

    def test_reference_coverage(self):
        items = [
            _item("1", human_label={"passed": True}),
            _item("2"),
            _item("3", human_label={"passed": False}),
        ]
        stats = compute_dataset_stats(items)
        assert stats.has_reference_count == 2
        assert stats.reference_coverage == pytest.approx(2/3)

    def test_duplicate_detection(self):
        items = [
            _item("1", "same input", "out1"),
            _item("2", "same input", "out2"),
            _item("3", "different", "out3"),
        ]
        stats = compute_dataset_stats(items)
        assert stats.duplicate_input_count == 1
        assert stats.duplicate_rate == pytest.approx(1/3)

    def test_empty_inputs(self):
        items = [_item("1", "", "out"), _item("2", "  ", "out"), _item("3", "ok", "out")]
        stats = compute_dataset_stats(items)
        assert stats.empty_input_count == 2

    def test_empty_outputs(self):
        items = [_item("1", "in", ""), _item("2", "in", None)]
        stats = compute_dataset_stats(items)
        assert stats.empty_output_count == 2

    def test_metadata_fields(self):
        items = [
            _item("1", metadata={"source": "prod", "tag": "v1"}),
            _item("2", metadata={"source": "test"}),
        ]
        stats = compute_dataset_stats(items)
        assert stats.metadata_fields["source"] == 2
        assert stats.metadata_fields["tag"] == 1

    def test_dict_input(self):
        items = [_item("1", input_text={"query": "hello", "context": "world"})]
        stats = compute_dataset_stats(items)
        assert stats.input_lengths[0] > 0  # JSON serialized length

    def test_as_dict(self):
        items = [_item("1")]
        stats = compute_dataset_stats(items)
        d = stats.as_dict()
        assert "item_count" in d
        assert "avg_input_length" in d
        assert "duplicate_rate" in d

    def test_format_text(self):
        items = [_item("1"), _item("2")]
        stats = compute_dataset_stats(items)
        text = stats.format_text()
        assert "Dataset Statistics" in text
        assert "Items:" in text


class TestHealthCheck:
    def test_healthy_dataset(self):
        items = [_item(str(i)) for i in range(10)]
        result = check_dataset_health(items)
        assert result.passed is True
        assert result.status == "WARN"  # no human labels warning

    def test_empty_dataset_fails(self):
        result = check_dataset_health([])
        assert result.passed is False
        assert result.status == "FAIL"
        assert any("empty" in e.lower() for e in result.errors)

    def test_too_few_items(self):
        items = [_item("1"), _item("2")]
        result = check_dataset_health(items, min_items=5)
        assert result.passed is False
        assert any("too few" in e.lower() for e in result.errors)

    def test_high_duplicate_rate(self):
        items = [
            _item("1", "same"),
            _item("2", "same"),
            _item("3", "same"),
            _item("4", "same"),
            _item("5", "different"),
        ]
        result = check_dataset_health(items, max_duplicate_rate=0.2)
        assert any("duplicate" in w.lower() for w in result.warnings)

    def test_many_empty_outputs(self):
        items = [_item(str(i), "input", "") for i in range(10)]
        result = check_dataset_health(items, max_empty_rate=0.05)
        assert any("empty output" in w.lower() for w in result.warnings)

    def test_no_references_warning(self):
        items = [_item(str(i)) for i in range(10)]
        result = check_dataset_health(items)
        assert any("human label" in w.lower() for w in result.warnings)

    def test_with_references_no_warning(self):
        items = [_item(str(i), human_label={"passed": True}) for i in range(10)]
        result = check_dataset_health(items)
        assert not any("human label" in w.lower() for w in result.warnings)

    def test_custom_thresholds(self):
        items = [_item(str(i)) for i in range(3)]
        result = check_dataset_health(items, min_items=2)
        assert result.passed is True

    def test_result_as_dict(self):
        result = check_dataset_health([_item("1")])
        d = result.as_dict()
        assert "passed" in d
        assert "status" in d
        assert "warnings" in d
        assert "errors" in d

    def test_format_text_pass(self):
        items = [_item(str(i), input_text=f"unique input {i}", human_label={"p": True}) for i in range(10)]
        result = check_dataset_health(items)
        text = result.format_text()
        assert "PASS" in text

    def test_format_text_fail(self):
        result = check_dataset_health([])
        text = result.format_text()
        assert "FAIL" in text
        assert "ERROR" in text
