"""Tests for code diff correlation module."""

from __future__ import annotations

import sys
from pathlib import Path

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.integration.code_diff_correlation import (
    CodeChange,
    CodeMetricReport,
    CorrelationEntry,
    MetricChange,
    build_correlation_report,
    correlate_change,
    identify_impactful_changes,
    render_correlation_timeline,
    suggest_root_causes,
)


# ---------------------------------------------------------------------------
# correlate_change
# ---------------------------------------------------------------------------


class TestCorrelateChange:
    def test_basic(self):
        cc = CodeChange(commit_hash="abc123", description="fix prompt")
        mc = [MetricChange(metric_id="accuracy", delta=0.05)]
        entry = correlate_change(cc, mc)
        assert entry.correlation_strength == 0.05
        assert len(entry.metric_changes) == 1

    def test_no_metric_changes(self):
        cc = CodeChange(commit_hash="abc123")
        entry = correlate_change(cc, [])
        assert entry.correlation_strength == 0.0
        assert entry.metric_changes == []

    def test_multiple_metrics(self):
        cc = CodeChange(commit_hash="abc123")
        mc = [
            MetricChange(metric_id="a", delta=0.1),
            MetricChange(metric_id="b", delta=-0.3),
        ]
        entry = correlate_change(cc, mc)
        # avg of abs(0.1), abs(-0.3) = (0.1 + 0.3) / 2 = 0.2
        assert abs(entry.correlation_strength - 0.2) < 1e-9

    def test_negative_deltas(self):
        cc = CodeChange(commit_hash="x")
        mc = [MetricChange(metric_id="m", delta=-0.5)]
        entry = correlate_change(cc, mc)
        assert entry.correlation_strength == 0.5

    def test_zero_deltas(self):
        cc = CodeChange(commit_hash="x")
        mc = [MetricChange(metric_id="m", delta=0.0)]
        entry = correlate_change(cc, mc)
        assert entry.correlation_strength == 0.0


# ---------------------------------------------------------------------------
# build_correlation_report
# ---------------------------------------------------------------------------


class TestBuildCorrelationReport:
    def test_finds_strongest(self):
        e1 = CorrelationEntry(
            code_change=CodeChange(description="small change"),
            correlation_strength=0.05,
        )
        e2 = CorrelationEntry(
            code_change=CodeChange(description="big change"),
            correlation_strength=0.5,
        )
        report = build_correlation_report([e1, e2])
        assert report.strongest_correlation == "big change"
        assert report.total_changes == 2

    def test_empty_entries(self):
        report = build_correlation_report([])
        assert report.strongest_correlation is None
        assert report.total_changes == 0

    def test_single_entry(self):
        e = CorrelationEntry(
            code_change=CodeChange(description="only"),
            correlation_strength=0.1,
        )
        report = build_correlation_report([e])
        assert report.strongest_correlation == "only"
        assert report.total_changes == 1

    def test_strongest_uses_commit_hash_if_no_description(self):
        e = CorrelationEntry(
            code_change=CodeChange(commit_hash="abc123"),
            correlation_strength=0.2,
        )
        report = build_correlation_report([e])
        assert report.strongest_correlation == "abc123"

    def test_all_zero_strength(self):
        e = CorrelationEntry(
            code_change=CodeChange(description="no-op"),
            correlation_strength=0.0,
        )
        report = build_correlation_report([e])
        assert report.strongest_correlation is None


# ---------------------------------------------------------------------------
# identify_impactful_changes
# ---------------------------------------------------------------------------


class TestIdentifyImpactfulChanges:
    def test_filters_by_threshold(self):
        e1 = CorrelationEntry(correlation_strength=0.05)
        e2 = CorrelationEntry(correlation_strength=0.2)
        report = CodeMetricReport(entries=[e1, e2], total_changes=2)
        impactful = identify_impactful_changes(report, threshold=0.1)
        assert len(impactful) == 1
        assert impactful[0].correlation_strength == 0.2

    def test_default_threshold(self):
        e = CorrelationEntry(correlation_strength=0.15)
        report = CodeMetricReport(entries=[e], total_changes=1)
        assert len(identify_impactful_changes(report)) == 1

    def test_no_impactful(self):
        e = CorrelationEntry(correlation_strength=0.01)
        report = CodeMetricReport(entries=[e], total_changes=1)
        assert identify_impactful_changes(report, threshold=0.1) == []

    def test_empty_report(self):
        report = CodeMetricReport()
        assert identify_impactful_changes(report) == []


# ---------------------------------------------------------------------------
# render_correlation_timeline
# ---------------------------------------------------------------------------


class TestRenderCorrelationTimeline:
    def test_basic_output(self):
        cc = CodeChange(
            commit_hash="abc",
            description="update prompt",
            timestamp="2026-01-01T00:00:00",
            files_changed=["prompt.txt"],
            lines_added=5,
            lines_removed=2,
        )
        mc = [MetricChange(metric_id="acc", delta=0.1)]
        entry = CorrelationEntry(
            code_change=cc, metric_changes=mc, correlation_strength=0.1
        )
        text = render_correlation_timeline([entry])
        assert "Timeline" in text
        assert "update prompt" in text
        assert "prompt.txt" in text
        assert "+5" in text
        assert "-2" in text

    def test_empty_entries(self):
        text = render_correlation_timeline([])
        assert "No entries" in text

    def test_multiple_entries(self):
        entries = [
            CorrelationEntry(
                code_change=CodeChange(description=f"change {i}"),
                correlation_strength=float(i) / 10,
            )
            for i in range(3)
        ]
        text = render_correlation_timeline(entries)
        assert "[1]" in text
        assert "[2]" in text
        assert "[3]" in text


# ---------------------------------------------------------------------------
# suggest_root_causes
# ---------------------------------------------------------------------------


class TestSuggestRootCauses:
    def test_prompt_file(self):
        entry = CorrelationEntry(
            code_change=CodeChange(files_changed=["prompt.txt"])
        )
        suggestions = suggest_root_causes(entry)
        assert any("prompt-related" in s for s in suggestions)

    def test_model_file(self):
        entry = CorrelationEntry(
            code_change=CodeChange(files_changed=["model_config.py"])
        )
        suggestions = suggest_root_causes(entry)
        assert any("model configuration" in s for s in suggestions)

    def test_config_file(self):
        entry = CorrelationEntry(
            code_change=CodeChange(files_changed=["settings.yaml"])
        )
        suggestions = suggest_root_causes(entry)
        assert any("configuration" in s for s in suggestions)

    def test_no_files(self):
        entry = CorrelationEntry(code_change=CodeChange(files_changed=[]))
        suggestions = suggest_root_causes(entry)
        assert any("unable to suggest" in s.lower() for s in suggestions)

    def test_generic_file(self):
        entry = CorrelationEntry(
            code_change=CodeChange(files_changed=["utils.py"])
        )
        suggestions = suggest_root_causes(entry)
        assert any("review for potential impact" in s for s in suggestions)

    def test_multiple_files(self):
        entry = CorrelationEntry(
            code_change=CodeChange(
                files_changed=["prompt.txt", "config.yaml", "utils.py"]
            )
        )
        suggestions = suggest_root_causes(entry)
        assert len(suggestions) == 3

    def test_eval_file(self):
        entry = CorrelationEntry(
            code_change=CodeChange(files_changed=["eval_suite.py"])
        )
        suggestions = suggest_root_causes(entry)
        assert any("evaluation logic" in s for s in suggestions)

    def test_test_file(self):
        entry = CorrelationEntry(
            code_change=CodeChange(files_changed=["test_main.py"])
        )
        suggestions = suggest_root_causes(entry)
        assert any("test-related" in s for s in suggestions)

    def test_dataset_file(self):
        entry = CorrelationEntry(
            code_change=CodeChange(files_changed=["dataset_loader.py"])
        )
        suggestions = suggest_root_causes(entry)
        assert any("data pipeline" in s for s in suggestions)


# ---------------------------------------------------------------------------
# Serialization round-trips
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_code_change_as_dict(self):
        cc = CodeChange(
            commit_hash="abc",
            files_changed=["a.py"],
            lines_added=10,
            lines_removed=3,
            description="fix",
            timestamp="2026-01-01",
        )
        d = cc.as_dict()
        assert d["commit_hash"] == "abc"
        assert d["files_changed"] == ["a.py"]

    def test_code_change_from_dict_roundtrip(self):
        cc = CodeChange(commit_hash="x", files_changed=["b.py"], lines_added=1)
        restored = CodeChange.from_dict(cc.as_dict())
        assert restored.commit_hash == cc.commit_hash
        assert restored.files_changed == cc.files_changed
        assert restored.lines_added == cc.lines_added

    def test_metric_change_as_dict(self):
        mc = MetricChange(metric_id="acc", before_score=0.8, after_score=0.9, delta=0.1)
        d = mc.as_dict()
        assert d["metric_id"] == "acc"
        assert d["delta"] == 0.1

    def test_correlation_entry_as_dict(self):
        entry = CorrelationEntry(
            code_change=CodeChange(commit_hash="x"),
            metric_changes=[MetricChange(metric_id="m", delta=0.1)],
            correlation_strength=0.1,
        )
        d = entry.as_dict()
        assert d["code_change"]["commit_hash"] == "x"
        assert len(d["metric_changes"]) == 1

    def test_report_roundtrip(self):
        entry = CorrelationEntry(
            code_change=CodeChange(commit_hash="abc", description="test"),
            metric_changes=[
                MetricChange(metric_id="m", before_score=0.5, after_score=0.7, delta=0.2)
            ],
            correlation_strength=0.2,
        )
        report = CodeMetricReport(
            entries=[entry], strongest_correlation="test", total_changes=1
        )
        d = report.as_dict()
        restored = CodeMetricReport.from_dict(d)
        assert restored.strongest_correlation == "test"
        assert restored.total_changes == 1
        assert len(restored.entries) == 1
        assert restored.entries[0].code_change.commit_hash == "abc"
        assert restored.entries[0].metric_changes[0].delta == 0.2

    def test_report_format_text(self):
        entry = CorrelationEntry(
            code_change=CodeChange(description="update", commit_hash="abc"),
            metric_changes=[
                MetricChange(metric_id="acc", before_score=0.8, after_score=0.9, delta=0.1)
            ],
            correlation_strength=0.1,
        )
        report = CodeMetricReport(
            entries=[entry], strongest_correlation="update", total_changes=1
        )
        text = report.format_text()
        assert "Code-Metric Correlation Report" in text
        assert "total_changes: 1" in text
        assert "update" in text
        assert "acc" in text

    def test_report_format_text_empty(self):
        report = CodeMetricReport()
        text = report.format_text()
        assert "total_changes: 0" in text
        assert "none" in text
