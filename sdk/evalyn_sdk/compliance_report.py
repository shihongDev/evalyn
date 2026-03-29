"""
Compliance report generation for regulatory frameworks.

Auto-generates evaluation documentation for EU AI Act, NIST AI RMF,
and ISO 42001 compliance. Pure Python with no external dependencies.
"""

from __future__ import annotations

import html
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class EvalMethodology:
    """Describes the evaluation methodology used."""

    method_name: str
    description: str
    metrics_used: List[str]
    dataset_description: str
    limitations: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "method_name": self.method_name,
            "description": self.description,
            "metrics_used": list(self.metrics_used),
            "dataset_description": self.dataset_description,
            "limitations": self.limitations,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EvalMethodology:
        return cls(
            method_name=data["method_name"],
            description=data["description"],
            metrics_used=data.get("metrics_used", []),
            dataset_description=data.get("dataset_description", ""),
            limitations=data.get("limitations", ""),
        )


@dataclass
class BenchmarkResult:
    """Result of a single benchmark evaluation."""

    benchmark_name: str
    score: float
    threshold: float
    passed: bool
    notes: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "benchmark_name": self.benchmark_name,
            "score": self.score,
            "threshold": self.threshold,
            "passed": self.passed,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BenchmarkResult:
        return cls(
            benchmark_name=data["benchmark_name"],
            score=data["score"],
            threshold=data["threshold"],
            passed=data["passed"],
            notes=data.get("notes", ""),
        )


@dataclass
class ComplianceSection:
    """A section within a compliance report."""

    section_id: str
    title: str
    content: str
    references: List[str]
    framework: str  # "eu_ai_act", "nist_rmf", "iso_42001"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "section_id": self.section_id,
            "title": self.title,
            "content": self.content,
            "references": list(self.references),
            "framework": self.framework,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ComplianceSection:
        return cls(
            section_id=data["section_id"],
            title=data["title"],
            content=data.get("content", ""),
            references=data.get("references", []),
            framework=data.get("framework", "eu_ai_act"),
        )


@dataclass
class ComplianceReport:
    """Full compliance report aggregating sections, methodology, and benchmarks."""

    report_id: str
    title: str
    system_name: str
    system_version: str
    sections: List[ComplianceSection]
    methodology: EvalMethodology
    benchmarks: List[BenchmarkResult]
    generated_at: str
    author: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "title": self.title,
            "system_name": self.system_name,
            "system_version": self.system_version,
            "sections": [s.as_dict() for s in self.sections],
            "methodology": self.methodology.as_dict(),
            "benchmarks": [b.as_dict() for b in self.benchmarks],
            "generated_at": self.generated_at,
            "author": self.author,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ComplianceReport:
        return cls(
            report_id=data["report_id"],
            title=data["title"],
            system_name=data["system_name"],
            system_version=data["system_version"],
            sections=[ComplianceSection.from_dict(s) for s in data.get("sections", [])],
            methodology=EvalMethodology.from_dict(data["methodology"]),
            benchmarks=[BenchmarkResult.from_dict(b) for b in data.get("benchmarks", [])],
            generated_at=data.get("generated_at", ""),
            author=data.get("author", ""),
        )


# ---------------------------------------------------------------------------
# Section Builders
# ---------------------------------------------------------------------------


def _benchmark_summary(benchmarks: List[BenchmarkResult]) -> str:
    """Format benchmark results into a readable summary."""
    if not benchmarks:
        return "No benchmark results available."
    lines = []
    for b in benchmarks:
        status = "PASSED" if b.passed else "FAILED"
        line = f"- {b.benchmark_name}: {b.score:.3f} (threshold: {b.threshold:.3f}) [{status}]"
        if b.notes:
            line += f" - {b.notes}"
        lines.append(line)
    passed = sum(1 for b in benchmarks if b.passed)
    lines.append(f"\nOverall: {passed}/{len(benchmarks)} benchmarks passed.")
    return "\n".join(lines)


def _metrics_summary(methodology: EvalMethodology) -> str:
    """Format metrics list into a readable summary."""
    if not methodology.metrics_used:
        return "No metrics specified."
    return "Metrics evaluated: " + ", ".join(methodology.metrics_used) + "."


def build_eu_ai_act_sections(
    system_name: str,
    methodology: EvalMethodology,
    benchmarks: List[BenchmarkResult],
) -> List[ComplianceSection]:
    """Generate standard EU AI Act compliance sections.

    Produces sections covering: System Description, Risk Classification,
    Evaluation Methodology, Benchmark Results, Limitations and Risks,
    Data Governance, and Human Oversight.
    """
    fw = "eu_ai_act"
    sections = []

    sections.append(ComplianceSection(
        section_id="eu_1_system_description",
        title="System Description",
        content=(
            f"{system_name} is an AI system evaluated under the EU AI Act framework. "
            f"The evaluation methodology used is: {methodology.method_name}. "
            f"{methodology.description}"
        ),
        references=["EU AI Act, Article 9 - Risk Management System"],
        framework=fw,
    ))

    sections.append(ComplianceSection(
        section_id="eu_2_risk_classification",
        title="Risk Classification",
        content=(
            f"Risk classification for {system_name} should be determined based on "
            "the intended use case and deployment context as specified in Annex III "
            "of the EU AI Act. This report provides evaluation evidence to support "
            "that classification."
        ),
        references=[
            "EU AI Act, Article 6 - Classification Rules",
            "EU AI Act, Annex III - High-Risk AI Systems",
        ],
        framework=fw,
    ))

    sections.append(ComplianceSection(
        section_id="eu_3_evaluation_methodology",
        title="Evaluation Methodology",
        content=(
            f"Method: {methodology.method_name}\n\n"
            f"{methodology.description}\n\n"
            f"Dataset: {methodology.dataset_description}\n\n"
            f"{_metrics_summary(methodology)}"
        ),
        references=["EU AI Act, Article 9(7) - Testing Procedures"],
        framework=fw,
    ))

    sections.append(ComplianceSection(
        section_id="eu_4_benchmark_results",
        title="Benchmark Results",
        content=_benchmark_summary(benchmarks),
        references=["EU AI Act, Article 9(8) - Performance Metrics"],
        framework=fw,
    ))

    limitations_content = (
        f"Known limitations: {methodology.limitations}"
        if methodology.limitations
        else "No specific limitations documented."
    )
    failed = [b for b in benchmarks if not b.passed]
    if failed:
        limitations_content += (
            "\n\nFailed benchmarks requiring attention:\n"
            + "\n".join(f"- {b.benchmark_name}: scored {b.score:.3f} below threshold {b.threshold:.3f}" for b in failed)
        )

    sections.append(ComplianceSection(
        section_id="eu_5_limitations",
        title="Limitations and Risks",
        content=limitations_content,
        references=[
            "EU AI Act, Article 13 - Transparency",
            "EU AI Act, Article 9(4) - Risk Identification",
        ],
        framework=fw,
    ))

    sections.append(ComplianceSection(
        section_id="eu_6_data_governance",
        title="Data Governance",
        content=(
            f"Evaluation dataset: {methodology.dataset_description}\n\n"
            "Data governance practices should be documented separately covering "
            "data collection, labeling, preprocessing, and bias assessment."
        ),
        references=["EU AI Act, Article 10 - Data and Data Governance"],
        framework=fw,
    ))

    sections.append(ComplianceSection(
        section_id="eu_7_human_oversight",
        title="Human Oversight",
        content=(
            "Human oversight mechanisms should be documented for production deployment. "
            "This evaluation report provides baseline performance evidence to inform "
            "oversight requirements."
        ),
        references=["EU AI Act, Article 14 - Human Oversight"],
        framework=fw,
    ))

    return sections


def build_nist_rmf_sections(
    system_name: str,
    methodology: EvalMethodology,
    benchmarks: List[BenchmarkResult],
) -> List[ComplianceSection]:
    """Generate NIST AI RMF compliance sections.

    Produces sections covering: Map, Measure, Manage, and Govern.
    """
    fw = "nist_rmf"
    sections = []

    sections.append(ComplianceSection(
        section_id="nist_1_map",
        title="Map",
        content=(
            f"System: {system_name}\n\n"
            f"Evaluation scope: {methodology.method_name}\n\n"
            f"{methodology.description}\n\n"
            "Context mapping should identify intended users, deployment environment, "
            "and potential impacts of the AI system."
        ),
        references=["NIST AI RMF 1.0, Map Function"],
        framework=fw,
    ))

    sections.append(ComplianceSection(
        section_id="nist_2_measure",
        title="Measure",
        content=(
            f"{_metrics_summary(methodology)}\n\n"
            f"Dataset: {methodology.dataset_description}\n\n"
            f"{_benchmark_summary(benchmarks)}"
        ),
        references=["NIST AI RMF 1.0, Measure Function"],
        framework=fw,
    ))

    manage_items = []
    failed = [b for b in benchmarks if not b.passed]
    if failed:
        manage_items.append(
            "Action items for failed benchmarks:\n"
            + "\n".join(f"- Address {b.benchmark_name} (gap: {b.threshold - b.score:.3f})" for b in failed)
        )
    if methodology.limitations:
        manage_items.append(f"Documented limitations: {methodology.limitations}")
    if not manage_items:
        manage_items.append("All benchmarks passed. Continue monitoring for drift.")

    sections.append(ComplianceSection(
        section_id="nist_3_manage",
        title="Manage",
        content="\n\n".join(manage_items),
        references=["NIST AI RMF 1.0, Manage Function"],
        framework=fw,
    ))

    sections.append(ComplianceSection(
        section_id="nist_4_govern",
        title="Govern",
        content=(
            "Governance policies should define roles, responsibilities, and processes "
            "for ongoing AI risk management. This evaluation provides evidence for "
            "governance review and decision-making."
        ),
        references=["NIST AI RMF 1.0, Govern Function"],
        framework=fw,
    ))

    return sections


# ---------------------------------------------------------------------------
# Report Construction
# ---------------------------------------------------------------------------


def create_compliance_report(
    system_name: str,
    system_version: str,
    methodology: EvalMethodology,
    benchmarks: List[BenchmarkResult],
    frameworks: List[str],
    author: str = "",
) -> ComplianceReport:
    """Build a full compliance report with UUID and timestamp.

    Args:
        system_name: Name of the AI system being evaluated.
        system_version: Version string for the system.
        methodology: Evaluation methodology details.
        benchmarks: List of benchmark results.
        frameworks: Frameworks to include ("eu_ai_act", "nist_rmf").
        author: Report author name.

    Returns:
        A ComplianceReport with sections for all requested frameworks.
    """
    sections: List[ComplianceSection] = []
    builders = {
        "eu_ai_act": build_eu_ai_act_sections,
        "nist_rmf": build_nist_rmf_sections,
    }

    for fw in frameworks:
        builder = builders.get(fw)
        if builder:
            sections.extend(builder(system_name, methodology, benchmarks))

    framework_names = {
        "eu_ai_act": "EU AI Act",
        "nist_rmf": "NIST AI RMF",
    }
    title_parts = [framework_names.get(fw, fw) for fw in frameworks]
    title = f"Compliance Report: {system_name} - " + ", ".join(title_parts)

    return ComplianceReport(
        report_id=str(uuid.uuid4()),
        title=title,
        system_name=system_name,
        system_version=system_version,
        sections=sections,
        methodology=methodology,
        benchmarks=benchmarks,
        generated_at=datetime.now(timezone.utc).isoformat(),
        author=author,
    )


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


def format_report_markdown(report: ComplianceReport) -> str:
    """Format a compliance report as Markdown."""
    lines: List[str] = []
    lines.append(f"# {report.title}")
    lines.append("")
    lines.append(f"**Report ID:** {report.report_id}")
    lines.append(f"**System:** {report.system_name} v{report.system_version}")
    lines.append(f"**Generated:** {report.generated_at}")
    if report.author:
        lines.append(f"**Author:** {report.author}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Methodology
    lines.append("## Evaluation Methodology")
    lines.append("")
    lines.append(f"**Method:** {report.methodology.method_name}")
    lines.append("")
    lines.append(report.methodology.description)
    lines.append("")
    lines.append(f"**Dataset:** {report.methodology.dataset_description}")
    lines.append("")
    if report.methodology.metrics_used:
        lines.append("**Metrics:** " + ", ".join(report.methodology.metrics_used))
        lines.append("")
    if report.methodology.limitations:
        lines.append(f"**Limitations:** {report.methodology.limitations}")
        lines.append("")

    # Benchmarks
    lines.append("## Benchmark Results")
    lines.append("")
    if report.benchmarks:
        lines.append("| Benchmark | Score | Threshold | Status |")
        lines.append("|-----------|-------|-----------|--------|")
        for b in report.benchmarks:
            status = "PASS" if b.passed else "FAIL"
            lines.append(f"| {b.benchmark_name} | {b.score:.3f} | {b.threshold:.3f} | {status} |")
        lines.append("")
    else:
        lines.append("No benchmark results available.")
        lines.append("")

    # Sections
    for section in report.sections:
        lines.append(f"## {section.title}")
        lines.append("")
        lines.append(section.content)
        lines.append("")
        if section.references:
            lines.append("**References:**")
            for ref in section.references:
                lines.append(f"- {ref}")
            lines.append("")

    return "\n".join(lines)


def format_report_html(report: ComplianceReport) -> str:
    """Format a compliance report as self-contained HTML with inline CSS.

    All dynamic content is escaped via html.escape to prevent XSS.
    """
    e = html.escape

    style = (
        "body { font-family: sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; "
        "color: #333; line-height: 1.6; }"
        "h1 { color: #1a365d; border-bottom: 2px solid #1a365d; padding-bottom: 8px; }"
        "h2 { color: #2c5282; margin-top: 24px; }"
        "table { border-collapse: collapse; width: 100%; margin: 16px 0; }"
        "th, td { border: 1px solid #ccc; padding: 8px 12px; text-align: left; }"
        "th { background: #edf2f7; }"
        ".pass { color: #276749; font-weight: bold; }"
        ".fail { color: #c53030; font-weight: bold; }"
        ".meta { color: #718096; font-size: 0.9em; }"
        ".ref { font-size: 0.85em; color: #4a5568; }"
        ".section { margin-bottom: 24px; padding: 16px; background: #f7fafc; "
        "border-left: 3px solid #2c5282; }"
    )

    parts: List[str] = []
    parts.append("<!DOCTYPE html>")
    parts.append("<html lang=\"en\">")
    parts.append("<head>")
    parts.append(f"<meta charset=\"utf-8\"><title>{e(report.title)}</title>")
    parts.append(f"<style>{style}</style>")
    parts.append("</head>")
    parts.append("<body>")

    parts.append(f"<h1>{e(report.title)}</h1>")
    parts.append("<div class=\"meta\">")
    parts.append(f"<p>Report ID: {e(report.report_id)}</p>")
    parts.append(f"<p>System: {e(report.system_name)} v{e(report.system_version)}</p>")
    parts.append(f"<p>Generated: {e(report.generated_at)}</p>")
    if report.author:
        parts.append(f"<p>Author: {e(report.author)}</p>")
    parts.append("</div>")

    # Methodology
    parts.append("<h2>Evaluation Methodology</h2>")
    parts.append(f"<p><strong>Method:</strong> {e(report.methodology.method_name)}</p>")
    parts.append(f"<p>{e(report.methodology.description)}</p>")
    parts.append(f"<p><strong>Dataset:</strong> {e(report.methodology.dataset_description)}</p>")
    if report.methodology.metrics_used:
        parts.append(
            "<p><strong>Metrics:</strong> "
            + e(", ".join(report.methodology.metrics_used))
            + "</p>"
        )
    if report.methodology.limitations:
        parts.append(
            f"<p><strong>Limitations:</strong> {e(report.methodology.limitations)}</p>"
        )

    # Benchmarks
    parts.append("<h2>Benchmark Results</h2>")
    if report.benchmarks:
        parts.append("<table>")
        parts.append("<tr><th>Benchmark</th><th>Score</th><th>Threshold</th><th>Status</th></tr>")
        for b in report.benchmarks:
            css_class = "pass" if b.passed else "fail"
            status = "PASS" if b.passed else "FAIL"
            parts.append(
                f"<tr><td>{e(b.benchmark_name)}</td>"
                f"<td>{b.score:.3f}</td>"
                f"<td>{b.threshold:.3f}</td>"
                f"<td class=\"{css_class}\">{status}</td></tr>"
            )
        parts.append("</table>")
    else:
        parts.append("<p>No benchmark results available.</p>")

    # Sections
    for section in report.sections:
        parts.append("<div class=\"section\">")
        parts.append(f"<h2>{e(section.title)}</h2>")
        # Preserve line breaks in content
        parts.append(f"<p>{e(section.content).replace(chr(10), '<br>')}</p>")
        if section.references:
            parts.append("<div class=\"ref\"><strong>References:</strong><ul>")
            for ref in section.references:
                parts.append(f"<li>{e(ref)}</li>")
            parts.append("</ul></div>")
        parts.append("</div>")

    parts.append("</body>")
    parts.append("</html>")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


_REQUIRED_EU_SECTION_IDS = {
    "eu_1_system_description",
    "eu_2_risk_classification",
    "eu_3_evaluation_methodology",
    "eu_4_benchmark_results",
    "eu_5_limitations",
    "eu_6_data_governance",
    "eu_7_human_oversight",
}

_REQUIRED_NIST_SECTION_IDS = {
    "nist_1_map",
    "nist_2_measure",
    "nist_3_manage",
    "nist_4_govern",
}


def validate_report(report: ComplianceReport) -> List[str]:
    """Validate a compliance report for completeness.

    Returns a list of validation errors. An empty list means the report
    is valid.
    """
    errors: List[str] = []

    if not report.report_id:
        errors.append("Missing report_id.")
    if not report.system_name:
        errors.append("Missing system_name.")
    if not report.system_version:
        errors.append("Missing system_version.")
    if not report.generated_at:
        errors.append("Missing generated_at timestamp.")
    if not report.methodology.method_name:
        errors.append("Missing methodology method_name.")
    if not report.methodology.description:
        errors.append("Missing methodology description.")
    if not report.methodology.metrics_used:
        errors.append("No metrics specified in methodology.")

    # Check framework-specific required sections
    section_ids = {s.section_id for s in report.sections}
    frameworks_present = {s.framework for s in report.sections}

    if "eu_ai_act" in frameworks_present:
        missing = _REQUIRED_EU_SECTION_IDS - section_ids
        for sid in sorted(missing):
            errors.append(f"Missing required EU AI Act section: {sid}")

    if "nist_rmf" in frameworks_present:
        missing = _REQUIRED_NIST_SECTION_IDS - section_ids
        for sid in sorted(missing):
            errors.append(f"Missing required NIST RMF section: {sid}")

    # Check for empty section content
    for section in report.sections:
        if not section.content.strip():
            errors.append(f"Section '{section.section_id}' has empty content.")

    # Flag failed benchmarks
    failed = [b for b in report.benchmarks if not b.passed]
    for b in failed:
        errors.append(
            f"Benchmark '{b.benchmark_name}' failed: "
            f"{b.score:.3f} < {b.threshold:.3f}."
        )

    return errors
