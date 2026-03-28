"""Tests for analysis.comparison_template module."""

from __future__ import annotations

import pytest

from evalyn_sdk.analysis.comparison_template import (
    ComparisonField,
    ComparisonOutput,
    ComparisonTemplate,
    EXECUTIVE_TEMPLATE,
    QA_TEMPLATE,
    TECHNICAL_TEMPLATE,
    create_custom_template,
    format_value,
    get_comparison_template,
    list_comparison_templates,
    render_comparison,
)


# ---------------------------------------------------------------------------
# ComparisonField
# ---------------------------------------------------------------------------


class TestComparisonField:
    def test_defaults(self):
        f = ComparisonField("x")
        assert f.label == ""
        assert f.format_fn == "default"

    def test_as_dict(self):
        f = ComparisonField("rate", "Rate", "percentage")
        d = f.as_dict()
        assert d == {"name": "rate", "label": "Rate", "format_fn": "percentage"}

    def test_from_dict(self):
        d = {"name": "cost", "label": "Cost", "format_fn": "currency"}
        f = ComparisonField.from_dict(d)
        assert f.name == "cost"
        assert f.label == "Cost"
        assert f.format_fn == "currency"

    def test_from_dict_defaults(self):
        f = ComparisonField.from_dict({"name": "x"})
        assert f.label == ""
        assert f.format_fn == "default"

    def test_roundtrip(self):
        original = ComparisonField("lat", "Latency", "duration")
        restored = ComparisonField.from_dict(original.as_dict())
        assert restored.name == original.name
        assert restored.label == original.label
        assert restored.format_fn == original.format_fn


# ---------------------------------------------------------------------------
# ComparisonTemplate
# ---------------------------------------------------------------------------


class TestComparisonTemplate:
    def test_defaults(self):
        t = ComparisonTemplate("t")
        assert t.description == ""
        assert t.fields == []
        assert t.show_delta is True
        assert t.show_chart is False
        assert t.audience == "technical"

    def test_as_dict(self):
        t = ComparisonTemplate("t", "desc", [ComparisonField("a")], audience="qa")
        d = t.as_dict()
        assert d["name"] == "t"
        assert d["description"] == "desc"
        assert d["audience"] == "qa"
        assert len(d["fields"]) == 1

    def test_from_dict(self):
        d = {
            "name": "custom",
            "description": "d",
            "fields": [{"name": "x"}],
            "show_delta": False,
            "show_chart": True,
            "audience": "executive",
        }
        t = ComparisonTemplate.from_dict(d)
        assert t.name == "custom"
        assert t.show_delta is False
        assert t.show_chart is True
        assert t.audience == "executive"
        assert len(t.fields) == 1

    def test_roundtrip(self):
        original = ComparisonTemplate(
            "rt",
            "roundtrip",
            [ComparisonField("a", "A", "percentage")],
            show_delta=False,
            show_chart=True,
            audience="executive",
        )
        restored = ComparisonTemplate.from_dict(original.as_dict())
        assert restored.name == original.name
        assert restored.show_delta == original.show_delta
        assert restored.show_chart == original.show_chart
        assert restored.audience == original.audience
        assert len(restored.fields) == len(original.fields)


# ---------------------------------------------------------------------------
# ComparisonOutput
# ---------------------------------------------------------------------------


class TestComparisonOutput:
    def test_as_dict(self):
        o = ComparisonOutput("t", [{"field": "A", "a": "1", "b": "2"}], "summary")
        d = o.as_dict()
        assert d["template_name"] == "t"
        assert len(d["rows"]) == 1
        assert d["summary"] == "summary"

    def test_format_text_basic(self):
        o = ComparisonOutput("test", [{"f": "v1", "g": "v2"}], "done")
        text = o.format_text()
        assert "Comparison: test" in text
        assert "done" in text

    def test_format_text_no_summary(self):
        o = ComparisonOutput("t", [])
        text = o.format_text()
        assert "Comparison: t" in text

    def test_format_text_empty_rows(self):
        o = ComparisonOutput("empty", [], "no data")
        text = o.format_text()
        assert "no data" in text


# ---------------------------------------------------------------------------
# Built-in Templates
# ---------------------------------------------------------------------------


class TestBuiltinTemplates:
    def test_executive_template(self):
        t = get_comparison_template("executive")
        assert t.name == "executive"
        assert t.audience == "executive"
        assert len(t.fields) == 2

    def test_technical_template(self):
        t = get_comparison_template("technical")
        assert t.name == "technical"
        assert t.audience == "technical"
        assert len(t.fields) == 4

    def test_qa_template(self):
        t = get_comparison_template("qa")
        assert t.name == "qa"
        assert t.audience == "qa"
        assert len(t.fields) == 3

    def test_unknown_template_raises(self):
        with pytest.raises(KeyError):
            get_comparison_template("nonexistent")

    def test_list_has_three(self):
        templates = list_comparison_templates()
        assert len(templates) == 3
        names = {t.name for t in templates}
        assert names == {"executive", "technical", "qa"}


# ---------------------------------------------------------------------------
# format_value
# ---------------------------------------------------------------------------


class TestFormatValue:
    def test_percentage(self):
        assert format_value(85, "percentage") == "85.0%"

    def test_percentage_float(self):
        assert format_value(92.35, "percentage") == "92.3%"

    def test_currency(self):
        assert format_value(0.05, "currency") == "$0.05"

    def test_currency_large(self):
        assert format_value(12.5, "currency") == "$12.50"

    def test_duration(self):
        assert format_value(150, "duration") == "150ms"

    def test_default(self):
        assert format_value(42, "default") == "42"

    def test_default_string(self):
        assert format_value("hello", "default") == "hello"

    def test_none_value(self):
        assert format_value(None, "percentage") == "N/A"


# ---------------------------------------------------------------------------
# render_comparison
# ---------------------------------------------------------------------------


class TestRenderComparison:
    def test_basic(self):
        data_a = {"pass_rate": 80.0, "cost_usd": 0.10}
        data_b = {"pass_rate": 90.0, "cost_usd": 0.12}
        output = render_comparison(data_a, data_b, EXECUTIVE_TEMPLATE)
        assert output.template_name == "executive"
        assert len(output.rows) == 2
        assert output.rows[0]["field"] == "Pass Rate"
        assert output.rows[0]["a"] == "80.0%"
        assert output.rows[0]["b"] == "90.0%"

    def test_delta_computed(self):
        data_a = {"pass_rate": 80.0}
        data_b = {"pass_rate": 90.0}
        template = ComparisonTemplate(
            "t", fields=[ComparisonField("pass_rate", "PR", "percentage")]
        )
        output = render_comparison(data_a, data_b, template)
        assert output.rows[0]["delta"] == pytest.approx(10.0)

    def test_delta_disabled(self):
        data_a = {"pass_rate": 80.0}
        data_b = {"pass_rate": 90.0}
        template = ComparisonTemplate(
            "t",
            fields=[ComparisonField("pass_rate", "PR")],
            show_delta=False,
        )
        output = render_comparison(data_a, data_b, template)
        assert "delta" not in output.rows[0]

    def test_empty_data(self):
        output = render_comparison({}, {}, EXECUTIVE_TEMPLATE)
        assert len(output.rows) == 2
        assert output.rows[0]["a"] == "N/A"

    def test_missing_field(self):
        data_a = {"pass_rate": 50.0}
        data_b = {}
        output = render_comparison(data_a, data_b, EXECUTIVE_TEMPLATE)
        assert output.rows[0]["a"] == "50.0%"
        assert output.rows[0]["b"] == "N/A"

    def test_summary_no_changes(self):
        data = {"pass_rate": 80.0, "cost_usd": 0.10}
        output = render_comparison(data, data, EXECUTIVE_TEMPLATE)
        assert output.summary == "No changes detected"


# ---------------------------------------------------------------------------
# create_custom_template
# ---------------------------------------------------------------------------


class TestCreateCustomTemplate:
    def test_basic(self):
        t = create_custom_template(
            "mine",
            [{"name": "acc", "label": "Accuracy", "format_fn": "percentage"}],
            audience="qa",
        )
        assert t.name == "mine"
        assert t.audience == "qa"
        assert len(t.fields) == 1
        assert t.fields[0].name == "acc"

    def test_minimal_fields(self):
        t = create_custom_template("m", [{"name": "x"}])
        assert t.audience == "technical"
        assert t.fields[0].format_fn == "default"
