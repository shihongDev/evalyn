"""Tests for the compliance_report module."""

from __future__ import annotations

import uuid

import pytest

from evalyn_sdk.compliance_report import (
    BenchmarkResult,
    ComplianceReport,
    ComplianceSection,
    EvalMethodology,
    build_eu_ai_act_sections,
    build_nist_rmf_sections,
    create_compliance_report,
    format_report_html,
    format_report_markdown,
    validate_report,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def methodology():
    return EvalMethodology(
        method_name="Automated Eval Pipeline",
        description="Runs a suite of metrics against test dataset.",
        metrics_used=["helpfulness", "safety", "coherence"],
        dataset_description="500 curated QA pairs",
        limitations="Limited to English-only inputs.",
    )


@pytest.fixture
def benchmarks():
    return [
        BenchmarkResult(
            benchmark_name="Helpfulness",
            score=0.85,
            threshold=0.80,
            passed=True,
            notes="Above target.",
        ),
        BenchmarkResult(
            benchmark_name="Safety",
            score=0.95,
            threshold=0.90,
            passed=True,
        ),
        BenchmarkResult(
            benchmark_name="Coherence",
            score=0.60,
            threshold=0.70,
            passed=False,
            notes="Below threshold.",
        ),
    ]


@pytest.fixture
def failing_benchmarks():
    return [
        BenchmarkResult(
            benchmark_name="Accuracy",
            score=0.40,
            threshold=0.80,
            passed=False,
        ),
    ]


@pytest.fixture
def report(methodology, benchmarks):
    return create_compliance_report(
        system_name="TestBot",
        system_version="1.0.0",
        methodology=methodology,
        benchmarks=benchmarks,
        frameworks=["eu_ai_act", "nist_rmf"],
        author="Test Author",
    )


# ---------------------------------------------------------------------------
# EvalMethodology
# ---------------------------------------------------------------------------


class TestEvalMethodology:
    def test_as_dict(self, methodology):
        d = methodology.as_dict()
        assert d["method_name"] == "Automated Eval Pipeline"
        assert d["metrics_used"] == ["helpfulness", "safety", "coherence"]
        assert d["limitations"] == "Limited to English-only inputs."

    def test_from_dict_roundtrip(self, methodology):
        d = methodology.as_dict()
        restored = EvalMethodology.from_dict(d)
        assert restored.method_name == methodology.method_name
        assert restored.metrics_used == methodology.metrics_used
        assert restored.limitations == methodology.limitations

    def test_from_dict_defaults(self):
        m = EvalMethodology.from_dict({
            "method_name": "basic",
            "description": "desc",
        })
        assert m.metrics_used == []
        assert m.dataset_description == ""
        assert m.limitations == ""

    def test_default_limitations_empty(self):
        m = EvalMethodology(
            method_name="m",
            description="d",
            metrics_used=[],
            dataset_description="ds",
        )
        assert m.limitations == ""


# ---------------------------------------------------------------------------
# BenchmarkResult
# ---------------------------------------------------------------------------


class TestBenchmarkResult:
    def test_as_dict(self, benchmarks):
        d = benchmarks[0].as_dict()
        assert d["benchmark_name"] == "Helpfulness"
        assert d["score"] == 0.85
        assert d["passed"] is True

    def test_from_dict_roundtrip(self, benchmarks):
        for b in benchmarks:
            restored = BenchmarkResult.from_dict(b.as_dict())
            assert restored.benchmark_name == b.benchmark_name
            assert restored.score == b.score
            assert restored.passed == b.passed

    def test_from_dict_default_notes(self):
        b = BenchmarkResult.from_dict({
            "benchmark_name": "test",
            "score": 0.5,
            "threshold": 0.5,
            "passed": True,
        })
        assert b.notes == ""


# ---------------------------------------------------------------------------
# ComplianceSection
# ---------------------------------------------------------------------------


class TestComplianceSection:
    def test_as_dict(self):
        s = ComplianceSection(
            section_id="s1",
            title="Title",
            content="Body text",
            references=["ref1"],
            framework="eu_ai_act",
        )
        d = s.as_dict()
        assert d["section_id"] == "s1"
        assert d["framework"] == "eu_ai_act"

    def test_from_dict_roundtrip(self):
        s = ComplianceSection(
            section_id="s2",
            title="T",
            content="C",
            references=["r1", "r2"],
            framework="nist_rmf",
        )
        restored = ComplianceSection.from_dict(s.as_dict())
        assert restored.section_id == s.section_id
        assert restored.references == s.references
        assert restored.framework == s.framework

    def test_from_dict_defaults(self):
        s = ComplianceSection.from_dict({
            "section_id": "x",
            "title": "t",
        })
        assert s.content == ""
        assert s.references == []
        assert s.framework == "eu_ai_act"


# ---------------------------------------------------------------------------
# ComplianceReport
# ---------------------------------------------------------------------------


class TestComplianceReport:
    def test_as_dict(self, report):
        d = report.as_dict()
        assert d["system_name"] == "TestBot"
        assert d["system_version"] == "1.0.0"
        assert d["author"] == "Test Author"
        assert isinstance(d["sections"], list)
        assert isinstance(d["benchmarks"], list)
        assert isinstance(d["methodology"], dict)

    def test_from_dict_roundtrip(self, report):
        d = report.as_dict()
        restored = ComplianceReport.from_dict(d)
        assert restored.report_id == report.report_id
        assert restored.system_name == report.system_name
        assert len(restored.sections) == len(report.sections)
        assert len(restored.benchmarks) == len(report.benchmarks)
        assert restored.methodology.method_name == report.methodology.method_name

    def test_report_id_is_uuid(self, report):
        # Should be a valid UUID
        uuid.UUID(report.report_id)

    def test_generated_at_set(self, report):
        assert report.generated_at != ""


# ---------------------------------------------------------------------------
# build_eu_ai_act_sections
# ---------------------------------------------------------------------------


class TestBuildEuAiActSections:
    def test_produces_seven_sections(self, methodology, benchmarks):
        sections = build_eu_ai_act_sections("Sys", methodology, benchmarks)
        assert len(sections) == 7

    def test_section_ids(self, methodology, benchmarks):
        sections = build_eu_ai_act_sections("Sys", methodology, benchmarks)
        ids = {s.section_id for s in sections}
        assert "eu_1_system_description" in ids
        assert "eu_2_risk_classification" in ids
        assert "eu_3_evaluation_methodology" in ids
        assert "eu_4_benchmark_results" in ids
        assert "eu_5_limitations" in ids
        assert "eu_6_data_governance" in ids
        assert "eu_7_human_oversight" in ids

    def test_all_sections_are_eu_framework(self, methodology, benchmarks):
        sections = build_eu_ai_act_sections("Sys", methodology, benchmarks)
        for s in sections:
            assert s.framework == "eu_ai_act"

    def test_system_name_in_description(self, methodology, benchmarks):
        sections = build_eu_ai_act_sections("MySystem", methodology, benchmarks)
        desc = next(s for s in sections if s.section_id == "eu_1_system_description")
        assert "MySystem" in desc.content

    def test_limitations_section_includes_failures(self, methodology, benchmarks):
        sections = build_eu_ai_act_sections("Sys", methodology, benchmarks)
        lim = next(s for s in sections if s.section_id == "eu_5_limitations")
        assert "Coherence" in lim.content

    def test_limitations_when_no_failures(self, methodology):
        b = [BenchmarkResult("A", 0.9, 0.8, True)]
        sections = build_eu_ai_act_sections("Sys", methodology, b)
        lim = next(s for s in sections if s.section_id == "eu_5_limitations")
        assert "Failed benchmarks" not in lim.content

    def test_benchmark_results_section_content(self, methodology, benchmarks):
        sections = build_eu_ai_act_sections("Sys", methodology, benchmarks)
        bench = next(s for s in sections if s.section_id == "eu_4_benchmark_results")
        assert "PASSED" in bench.content
        assert "FAILED" in bench.content

    def test_references_populated(self, methodology, benchmarks):
        sections = build_eu_ai_act_sections("Sys", methodology, benchmarks)
        for s in sections:
            assert len(s.references) >= 1


# ---------------------------------------------------------------------------
# build_nist_rmf_sections
# ---------------------------------------------------------------------------


class TestBuildNistRmfSections:
    def test_produces_four_sections(self, methodology, benchmarks):
        sections = build_nist_rmf_sections("Sys", methodology, benchmarks)
        assert len(sections) == 4

    def test_section_ids(self, methodology, benchmarks):
        sections = build_nist_rmf_sections("Sys", methodology, benchmarks)
        ids = {s.section_id for s in sections}
        assert ids == {"nist_1_map", "nist_2_measure", "nist_3_manage", "nist_4_govern"}

    def test_all_sections_are_nist_framework(self, methodology, benchmarks):
        sections = build_nist_rmf_sections("Sys", methodology, benchmarks)
        for s in sections:
            assert s.framework == "nist_rmf"

    def test_manage_section_includes_failures(self, methodology, benchmarks):
        sections = build_nist_rmf_sections("Sys", methodology, benchmarks)
        manage = next(s for s in sections if s.section_id == "nist_3_manage")
        assert "Coherence" in manage.content

    def test_manage_section_all_pass(self, methodology):
        b = [BenchmarkResult("A", 0.9, 0.8, True)]
        m = EvalMethodology("m", "d", ["x"], "ds", limitations="")
        sections = build_nist_rmf_sections("Sys", m, b)
        manage = next(s for s in sections if s.section_id == "nist_3_manage")
        assert "All benchmarks passed" in manage.content


# ---------------------------------------------------------------------------
# create_compliance_report
# ---------------------------------------------------------------------------


class TestCreateComplianceReport:
    def test_creates_report_with_eu_sections(self, methodology, benchmarks):
        r = create_compliance_report("S", "1.0", methodology, benchmarks, ["eu_ai_act"])
        assert len(r.sections) == 7

    def test_creates_report_with_nist_sections(self, methodology, benchmarks):
        r = create_compliance_report("S", "1.0", methodology, benchmarks, ["nist_rmf"])
        assert len(r.sections) == 4

    def test_creates_report_with_both_frameworks(self, methodology, benchmarks):
        r = create_compliance_report(
            "S", "1.0", methodology, benchmarks, ["eu_ai_act", "nist_rmf"]
        )
        assert len(r.sections) == 11

    def test_title_includes_system_name(self, methodology, benchmarks):
        r = create_compliance_report("MyBot", "2.0", methodology, benchmarks, ["eu_ai_act"])
        assert "MyBot" in r.title

    def test_title_includes_framework_name(self, methodology, benchmarks):
        r = create_compliance_report("S", "1.0", methodology, benchmarks, ["eu_ai_act"])
        assert "EU AI Act" in r.title

    def test_unknown_framework_produces_no_sections(self, methodology, benchmarks):
        r = create_compliance_report("S", "1.0", methodology, benchmarks, ["unknown_fw"])
        assert len(r.sections) == 0

    def test_author_default_empty(self, methodology, benchmarks):
        r = create_compliance_report("S", "1.0", methodology, benchmarks, ["eu_ai_act"])
        assert r.author == ""


# ---------------------------------------------------------------------------
# format_report_markdown
# ---------------------------------------------------------------------------


class TestFormatReportMarkdown:
    def test_starts_with_title(self, report):
        md = format_report_markdown(report)
        assert md.startswith("# Compliance Report:")

    def test_includes_report_id(self, report):
        md = format_report_markdown(report)
        assert report.report_id in md

    def test_includes_system_info(self, report):
        md = format_report_markdown(report)
        assert "TestBot" in md
        assert "1.0.0" in md

    def test_includes_author(self, report):
        md = format_report_markdown(report)
        assert "Test Author" in md

    def test_includes_benchmark_table(self, report):
        md = format_report_markdown(report)
        assert "| Benchmark |" in md
        assert "PASS" in md
        assert "FAIL" in md

    def test_includes_methodology(self, report):
        md = format_report_markdown(report)
        assert "Automated Eval Pipeline" in md

    def test_includes_section_titles(self, report):
        md = format_report_markdown(report)
        assert "## System Description" in md
        assert "## Map" in md

    def test_no_author_line_when_empty(self, methodology, benchmarks):
        r = create_compliance_report("S", "1.0", methodology, benchmarks, ["eu_ai_act"])
        md = format_report_markdown(r)
        assert "**Author:**" not in md

    def test_no_benchmarks_message(self, methodology):
        r = create_compliance_report("S", "1.0", methodology, [], ["eu_ai_act"])
        md = format_report_markdown(r)
        assert "No benchmark results available." in md


# ---------------------------------------------------------------------------
# format_report_html
# ---------------------------------------------------------------------------


class TestFormatReportHtml:
    def test_starts_with_doctype(self, report):
        h = format_report_html(report)
        assert h.startswith("<!DOCTYPE html>")

    def test_contains_title_tag(self, report):
        h = format_report_html(report)
        assert "<title>" in h

    def test_contains_style_tag(self, report):
        h = format_report_html(report)
        assert "<style>" in h

    def test_escapes_html_in_system_name(self, methodology, benchmarks):
        r = create_compliance_report(
            "<script>alert(1)</script>", "1.0",
            methodology, benchmarks, ["eu_ai_act"],
        )
        h = format_report_html(r)
        assert "<script>" not in h
        assert "&lt;script&gt;" in h

    def test_contains_benchmark_table(self, report):
        h = format_report_html(report)
        assert "<table>" in h
        assert "PASS" in h
        assert "FAIL" in h

    def test_pass_fail_css_classes(self, report):
        h = format_report_html(report)
        assert 'class="pass"' in h
        assert 'class="fail"' in h

    def test_no_benchmarks_html(self, methodology):
        r = create_compliance_report("S", "1.0", methodology, [], ["eu_ai_act"])
        h = format_report_html(r)
        assert "No benchmark results available." in h

    def test_contains_section_divs(self, report):
        h = format_report_html(report)
        assert 'class="section"' in h

    def test_author_shown_when_present(self, report):
        h = format_report_html(report)
        assert "Test Author" in h


# ---------------------------------------------------------------------------
# validate_report
# ---------------------------------------------------------------------------


class TestValidateReport:
    def test_valid_report_has_only_benchmark_warnings(self, report):
        errors = validate_report(report)
        # Only error should be the failed benchmark
        benchmark_errors = [e for e in errors if "Benchmark" in e]
        non_benchmark = [e for e in errors if "Benchmark" not in e]
        assert len(benchmark_errors) == 1
        assert len(non_benchmark) == 0

    def test_missing_system_name(self, report):
        report.system_name = ""
        errors = validate_report(report)
        assert any("system_name" in e for e in errors)

    def test_missing_system_version(self, report):
        report.system_version = ""
        errors = validate_report(report)
        assert any("system_version" in e for e in errors)

    def test_missing_methodology_name(self, report):
        report.methodology.method_name = ""
        errors = validate_report(report)
        assert any("method_name" in e for e in errors)

    def test_missing_methodology_description(self, report):
        report.methodology.description = ""
        errors = validate_report(report)
        assert any("description" in e for e in errors)

    def test_no_metrics_in_methodology(self, report):
        report.methodology.metrics_used = []
        errors = validate_report(report)
        assert any("No metrics" in e for e in errors)

    def test_missing_report_id(self, report):
        report.report_id = ""
        errors = validate_report(report)
        assert any("report_id" in e for e in errors)

    def test_missing_generated_at(self, report):
        report.generated_at = ""
        errors = validate_report(report)
        assert any("generated_at" in e for e in errors)

    def test_failed_benchmark_flagged(self, report):
        errors = validate_report(report)
        assert any("Coherence" in e and "failed" in e for e in errors)

    def test_all_pass_no_benchmark_errors(self, methodology):
        r = create_compliance_report(
            "S", "1.0", methodology,
            [BenchmarkResult("A", 0.9, 0.8, True)],
            ["eu_ai_act"],
        )
        errors = validate_report(r)
        assert not any("failed" in e for e in errors)

    def test_missing_eu_section_detected(self, methodology, benchmarks):
        r = create_compliance_report(
            "S", "1.0", methodology, benchmarks, ["eu_ai_act"]
        )
        # Remove one section
        r.sections = [s for s in r.sections if s.section_id != "eu_7_human_oversight"]
        errors = validate_report(r)
        assert any("eu_7_human_oversight" in e for e in errors)

    def test_missing_nist_section_detected(self, methodology, benchmarks):
        r = create_compliance_report(
            "S", "1.0", methodology, benchmarks, ["nist_rmf"]
        )
        r.sections = [s for s in r.sections if s.section_id != "nist_4_govern"]
        errors = validate_report(r)
        assert any("nist_4_govern" in e for e in errors)

    def test_empty_section_content_detected(self, methodology, benchmarks):
        r = create_compliance_report(
            "S", "1.0", methodology, benchmarks, ["eu_ai_act"]
        )
        r.sections[0].content = "   "
        errors = validate_report(r)
        assert any("empty content" in e for e in errors)
