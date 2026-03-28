"""Tests for metric versioning and version comparison."""

import pytest
from evalyn_sdk.models import MetricSpec
from evalyn_sdk.analysis.versioning import (
    compare_metric_versions,
    VersionComparisonReport,
    MetricVersionChange,
)


class TestMetricSpecVersionHash:
    """Test version hash computation on MetricSpec."""

    def test_hash_is_deterministic(self):
        spec = MetricSpec(id="help", name="help", type="subjective", config={"threshold": 0.5})
        assert spec.version_hash == spec.version_hash

    def test_same_config_same_hash(self):
        a = MetricSpec(id="help", name="help", type="subjective", config={"threshold": 0.5})
        b = MetricSpec(id="help", name="help", type="subjective", config={"threshold": 0.5})
        assert a.version_hash == b.version_hash

    def test_different_config_different_hash(self):
        a = MetricSpec(id="help", name="help", type="subjective", config={"threshold": 0.5})
        b = MetricSpec(id="help", name="help", type="subjective", config={"threshold": 0.7})
        assert a.version_hash != b.version_hash

    def test_different_type_different_hash(self):
        a = MetricSpec(id="check", name="check", type="objective")
        b = MetricSpec(id="check", name="check", type="subjective")
        assert a.version_hash != b.version_hash

    def test_description_does_not_affect_hash(self):
        """Description is cosmetic, shouldn't change the version."""
        a = MetricSpec(id="help", name="help", type="subjective", description="v1")
        b = MetricSpec(id="help", name="help", type="subjective", description="v2")
        assert a.version_hash == b.version_hash

    def test_name_does_not_affect_hash(self):
        """Display name is cosmetic, shouldn't change the version."""
        a = MetricSpec(id="help", name="Helpfulness v1", type="subjective")
        b = MetricSpec(id="help", name="Helpfulness v2", type="subjective")
        assert a.version_hash == b.version_hash

    def test_hash_is_16_chars(self):
        spec = MetricSpec(id="test", name="test", type="objective")
        assert len(spec.version_hash) == 16

    def test_hash_in_as_dict(self):
        spec = MetricSpec(id="test", name="test", type="objective")
        d = spec.as_dict()
        assert "version_hash" in d
        assert d["version_hash"] == spec.version_hash

    def test_prompt_change_changes_hash(self):
        """Changing the prompt in config should change the version."""
        a = MetricSpec(id="help", name="help", type="subjective", config={"prompt": "Rate quality"})
        b = MetricSpec(id="help", name="help", type="subjective", config={"prompt": "Rate accuracy"})
        assert a.version_hash != b.version_hash

    def test_rubric_change_changes_hash(self):
        a = MetricSpec(id="help", name="help", type="subjective", config={"rubric": ["criteria 1"]})
        b = MetricSpec(id="help", name="help", type="subjective", config={"rubric": ["criteria 2"]})
        assert a.version_hash != b.version_hash


class TestCompareMetricVersions:
    """Test version comparison between runs."""

    def test_no_changes(self):
        baseline = [MetricSpec(id="a", name="a", type="objective")]
        current = [MetricSpec(id="a", name="a", type="objective")]
        report = compare_metric_versions(baseline, current)
        assert not report.has_changes
        assert report.unchanged_metrics == ["a"]

    def test_modified_metric(self):
        baseline = [MetricSpec(id="help", name="help", type="subjective", config={"threshold": 0.5})]
        current = [MetricSpec(id="help", name="help", type="subjective", config={"threshold": 0.8})]
        report = compare_metric_versions(baseline, current)
        assert report.has_changes
        assert report.modified_count == 1
        assert report.changes[0].change_type == "modified"
        assert report.changes[0].metric_id == "help"

    def test_added_metric(self):
        baseline = [MetricSpec(id="a", name="a", type="objective")]
        current = [
            MetricSpec(id="a", name="a", type="objective"),
            MetricSpec(id="b", name="b", type="subjective"),
        ]
        report = compare_metric_versions(baseline, current)
        assert report.added_count == 1
        assert report.changes[0].metric_id == "b"
        assert report.changes[0].change_type == "added"

    def test_removed_metric(self):
        baseline = [
            MetricSpec(id="a", name="a", type="objective"),
            MetricSpec(id="b", name="b", type="subjective"),
        ]
        current = [MetricSpec(id="a", name="a", type="objective")]
        report = compare_metric_versions(baseline, current)
        assert report.removed_count == 1
        assert report.changes[0].metric_id == "b"
        assert report.changes[0].change_type == "removed"

    def test_complex_changes(self):
        baseline = [
            MetricSpec(id="a", name="a", type="objective"),
            MetricSpec(id="b", name="b", type="subjective", config={"v": 1}),
            MetricSpec(id="c", name="c", type="objective"),
        ]
        current = [
            MetricSpec(id="a", name="a", type="objective"),
            MetricSpec(id="b", name="b", type="subjective", config={"v": 2}),
            MetricSpec(id="d", name="d", type="subjective"),
        ]
        report = compare_metric_versions(baseline, current)
        assert report.modified_count == 1  # b changed
        assert report.removed_count == 1   # c removed
        assert report.added_count == 1     # d added
        assert "a" in report.unchanged_metrics

    def test_empty_runs(self):
        report = compare_metric_versions([], [])
        assert not report.has_changes
        assert report.unchanged_metrics == []


class TestVersionComparisonReport:
    """Test report formatting."""

    def test_format_warnings_modified(self):
        baseline = [MetricSpec(id="help", name="help", type="subjective", config={"v": 1})]
        current = [MetricSpec(id="help", name="help", type="subjective", config={"v": 2})]
        report = compare_metric_versions(baseline, current)
        warnings = report.format_warnings()
        assert len(warnings) == 1
        assert "help" in warnings[0]
        assert "changed" in warnings[0]

    def test_format_warnings_added(self):
        report = compare_metric_versions([], [MetricSpec(id="new", name="new", type="objective")])
        warnings = report.format_warnings()
        assert len(warnings) == 1
        assert "new" in warnings[0]
        assert "new" in warnings[0].lower()

    def test_format_warnings_empty(self):
        report = compare_metric_versions(
            [MetricSpec(id="a", name="a", type="objective")],
            [MetricSpec(id="a", name="a", type="objective")],
        )
        assert report.format_warnings() == []
