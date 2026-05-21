"""Tests for dataset filtering DSL."""

import pytest
from evalyn_sdk.models import DatasetItem
from evalyn_sdk.datasets.filter import (
    parse_filter,
    filter_items,
    FilterExpression,
    CompoundFilter,
)


def _item(item_id: str, input_text: str = "test", output_text: str = "answer",
          metadata: dict = None, human_label: dict = None) -> DatasetItem:
    return DatasetItem(
        id=item_id, input=input_text, output=output_text,
        metadata=metadata or {}, human_label=human_label,
    )


class TestFilterExpression:
    """Test individual filter expressions."""

    def test_equals(self):
        expr = FilterExpression("id", "=", "item-1")
        item = _item("item-1")
        assert expr.evaluate(item) is True

    def test_not_equals(self):
        expr = FilterExpression("id", "!=", "item-1")
        assert expr.evaluate(_item("item-2")) is True
        assert expr.evaluate(_item("item-1")) is False

    def test_contains(self):
        expr = FilterExpression("input", "contains", "refund")
        assert expr.evaluate(_item("1", "I want a refund")) is True
        assert expr.evaluate(_item("2", "hello world")) is False

    def test_contains_case_insensitive(self):
        expr = FilterExpression("input", "contains", "REFUND")
        assert expr.evaluate(_item("1", "i want a refund")) is True

    def test_not_contains(self):
        expr = FilterExpression("output", "not_contains", "error")
        assert expr.evaluate(_item("1", output_text="success")) is True
        assert expr.evaluate(_item("2", output_text="error occurred")) is False

    def test_greater_than(self):
        expr = FilterExpression("input_length", ">", "5")
        assert expr.evaluate(_item("1", "hello world")) is True
        assert expr.evaluate(_item("2", "hi")) is False

    def test_less_than(self):
        expr = FilterExpression("output_length", "<", "10")
        assert expr.evaluate(_item("1", output_text="short")) is True
        assert expr.evaluate(_item("2", output_text="this is a longer output text")) is False

    def test_metadata_field(self):
        expr = FilterExpression("metadata.source", "=", "production")
        assert expr.evaluate(_item("1", metadata={"source": "production"})) is True
        assert expr.evaluate(_item("2", metadata={"source": "test"})) is False
        assert expr.evaluate(_item("3", metadata={})) is False

    def test_matches_regex(self):
        expr = FilterExpression("output", "matches", r"^\d+$")
        assert expr.evaluate(_item("1", output_text="42")) is True
        assert expr.evaluate(_item("2", output_text="not a number")) is False

    def test_is_null(self):
        expr = FilterExpression("metadata.tag", "is_null", "")
        assert expr.evaluate(_item("1", metadata={})) is True
        assert expr.evaluate(_item("2", metadata={"tag": "v1"})) is False

    def test_has_reference(self):
        expr = FilterExpression("has_reference", "=", "True")
        assert expr.evaluate(_item("1", human_label={"passed": True})) is True
        assert expr.evaluate(_item("2")) is False


class TestParseFilter:
    """Test query parsing."""

    def test_simple_equals(self):
        f = parse_filter("id = item-1")
        assert len(f.expressions) == 1
        assert f.expressions[0].field == "id"

    def test_contains(self):
        f = parse_filter("input contains refund")
        assert f.expressions[0].operator == "contains"
        assert f.expressions[0].value == "refund"

    def test_numeric(self):
        f = parse_filter("output_length > 500")
        assert f.expressions[0].operator == ">"
        assert f.expressions[0].value == "500"

    def test_and(self):
        f = parse_filter("output_length > 100 and metadata.tag = production")
        assert len(f.expressions) == 2
        assert f.logic == "and"

    def test_or(self):
        f = parse_filter("input contains refund or input contains return")
        assert len(f.expressions) == 2
        assert f.logic == "or"

    def test_metadata_dot(self):
        f = parse_filter("metadata.source = prod")
        assert f.expressions[0].field == "metadata.source"

    def test_empty_query(self):
        f = parse_filter("")
        assert len(f.expressions) == 0

    def test_invalid_query_raises(self):
        with pytest.raises(ValueError):
            parse_filter("not a valid filter ??? !!!")


class TestFilterItems:
    """Test end-to-end item filtering."""

    def test_filter_by_metadata(self):
        items = [
            _item("1", metadata={"source": "prod"}),
            _item("2", metadata={"source": "test"}),
            _item("3", metadata={"source": "prod"}),
        ]
        result = filter_items(items, "metadata.source = prod")
        assert len(result) == 2
        assert all(r.metadata["source"] == "prod" for r in result)

    def test_filter_by_length(self):
        items = [
            _item("1", output_text="short"),
            _item("2", output_text="this is a much longer output text here"),
        ]
        result = filter_items(items, "output_length > 10")
        assert len(result) == 1
        assert result[0].id == "2"

    def test_filter_by_content(self):
        items = [
            _item("1", "How do I get a refund?"),
            _item("2", "What is the weather?"),
            _item("3", "I want to return this item"),
        ]
        result = filter_items(items, "input contains refund or input contains return")
        assert len(result) == 2

    def test_compound_and(self):
        items = [
            _item("1", "short", "response", metadata={"tag": "prod"}),
            _item("2", "a much longer input here", "resp", metadata={"tag": "prod"}),
            _item("3", "a much longer input here", "resp", metadata={"tag": "test"}),
        ]
        result = filter_items(items, "input_length > 10 and metadata.tag = prod")
        assert len(result) == 1
        assert result[0].id == "2"

    def test_empty_filter_returns_all(self):
        items = [_item("1"), _item("2")]
        result = filter_items(items, "")
        assert len(result) == 2

    def test_no_matches(self):
        items = [_item("1", metadata={"tag": "test"})]
        result = filter_items(items, "metadata.tag = prod")
        assert len(result) == 0
