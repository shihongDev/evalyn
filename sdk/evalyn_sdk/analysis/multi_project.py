"""Multi-project dashboard: view and compare metrics across multiple projects.

Aggregates per-project summaries, computes pairwise comparisons, and
detects regressions where pass rates fall below a threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ProjectSummary:
    """High-level summary of a single project."""

    project_name: str
    dataset_count: int = 0
    run_count: int = 0
    latest_pass_rate: float | None = None
    total_cost_usd: float = 0.0
    latest_run_date: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "project_name": self.project_name,
            "dataset_count": self.dataset_count,
            "run_count": self.run_count,
            "latest_pass_rate": self.latest_pass_rate,
            "total_cost_usd": round(self.total_cost_usd, 4),
            "latest_run_date": self.latest_run_date,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProjectSummary:
        return cls(
            project_name=data["project_name"],
            dataset_count=data.get("dataset_count", 0),
            run_count=data.get("run_count", 0),
            latest_pass_rate=data.get("latest_pass_rate"),
            total_cost_usd=data.get("total_cost_usd", 0.0),
            latest_run_date=data.get("latest_run_date", ""),
        )


@dataclass
class ProjectComparison:
    """Pairwise comparison between two projects."""

    project_a: str
    project_b: str
    pass_rate_delta: float | None = None
    cost_delta: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "project_a": self.project_a,
            "project_b": self.project_b,
            "pass_rate_delta": (
                round(self.pass_rate_delta, 4)
                if self.pass_rate_delta is not None
                else None
            ),
            "cost_delta": round(self.cost_delta, 4),
        }

    def format_text(self) -> str:
        pr_str = (
            f"{self.pass_rate_delta:+.1%}"
            if self.pass_rate_delta is not None
            else "N/A"
        )
        lines = [
            f"Comparison: {self.project_a} vs {self.project_b}",
            f"  Pass rate delta:  {pr_str}",
            f"  Cost delta:       ${self.cost_delta:+.4f}",
        ]
        return "\n".join(lines)


@dataclass
class MultiProjectReport:
    """Full multi-project dashboard report."""

    projects: list[ProjectSummary] = field(default_factory=list)
    comparisons: list[ProjectComparison] = field(default_factory=list)
    total_projects: int = 0
    total_cost_usd: float = 0.0
    regressions: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "projects": [p.as_dict() for p in self.projects],
            "comparisons": [c.as_dict() for c in self.comparisons],
            "total_projects": self.total_projects,
            "total_cost_usd": round(self.total_cost_usd, 4),
            "regressions": list(self.regressions),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MultiProjectReport:
        projects = [
            ProjectSummary.from_dict(p) for p in data.get("projects", [])
        ]
        comparisons = [
            ProjectComparison(**c) for c in data.get("comparisons", [])
        ]
        return cls(
            projects=projects,
            comparisons=comparisons,
            total_projects=data.get("total_projects", 0),
            total_cost_usd=data.get("total_cost_usd", 0.0),
            regressions=data.get("regressions", []),
        )

    def format_text(self) -> str:
        lines = [
            "Multi-Project Report",
            f"  Total projects:  {self.total_projects}",
            f"  Total cost:      ${self.total_cost_usd:.4f}",
        ]
        if self.regressions:
            lines.append(f"  Regressions:     {', '.join(self.regressions)}")
        for p in self.projects:
            pr_str = (
                f"{p.latest_pass_rate:.1%}"
                if p.latest_pass_rate is not None
                else "N/A"
            )
            lines.append(
                f"  - {p.project_name}: pass_rate={pr_str}, "
                f"cost=${p.total_cost_usd:.4f}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def create_project_summary(
    name: str,
    datasets: int = 0,
    runs: int = 0,
    pass_rate: float | None = None,
    cost: float = 0.0,
    latest_date: str = "",
) -> ProjectSummary:
    """Factory for creating a ProjectSummary."""
    return ProjectSummary(
        project_name=name,
        dataset_count=datasets,
        run_count=runs,
        latest_pass_rate=pass_rate,
        total_cost_usd=cost,
        latest_run_date=latest_date,
    )


def compare_projects(
    a: ProjectSummary,
    b: ProjectSummary,
) -> ProjectComparison:
    """Compute deltas between two project summaries.

    pass_rate_delta = b.latest_pass_rate - a.latest_pass_rate (None if either is None).
    cost_delta = b.total_cost_usd - a.total_cost_usd.
    """
    if a.latest_pass_rate is not None and b.latest_pass_rate is not None:
        pr_delta: float | None = b.latest_pass_rate - a.latest_pass_rate
    else:
        pr_delta = None
    return ProjectComparison(
        project_a=a.project_name,
        project_b=b.project_name,
        pass_rate_delta=pr_delta,
        cost_delta=b.total_cost_usd - a.total_cost_usd,
    )


def detect_regressions(
    projects: list[ProjectSummary],
    min_pass_rate: float = 0.8,
) -> list[str]:
    """Return names of projects whose pass rate is below min_pass_rate."""
    return [
        p.project_name
        for p in projects
        if p.latest_pass_rate is not None and p.latest_pass_rate < min_pass_rate
    ]


def build_multi_project_report(
    projects: list[ProjectSummary],
) -> MultiProjectReport:
    """Build a full report with pairwise comparisons and regression detection."""
    comparisons: list[ProjectComparison] = []
    for i, a in enumerate(projects):
        for b in projects[i + 1 :]:
            comparisons.append(compare_projects(a, b))

    regressions = detect_regressions(projects)
    total_cost = sum(p.total_cost_usd for p in projects)

    return MultiProjectReport(
        projects=list(projects),
        comparisons=comparisons,
        total_projects=len(projects),
        total_cost_usd=total_cost,
        regressions=regressions,
    )


def rank_projects(
    projects: list[ProjectSummary],
    by: str = "pass_rate",
) -> list[ProjectSummary]:
    """Sort projects by pass_rate (descending) or cost (ascending).

    Projects with None pass_rate are sorted to the end when sorting by pass_rate.
    """
    if by == "cost":
        return sorted(projects, key=lambda p: p.total_cost_usd)
    # Default: pass_rate descending
    return sorted(
        projects,
        key=lambda p: p.latest_pass_rate if p.latest_pass_rate is not None else -1.0,
        reverse=True,
    )
