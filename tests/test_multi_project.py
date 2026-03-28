"""Tests for multi-project dashboard."""

import pytest
from evalyn_sdk.analysis.multi_project import (
    ProjectSummary,
    ProjectComparison,
    MultiProjectReport,
    create_project_summary,
    compare_projects,
    detect_regressions,
    build_multi_project_report,
    rank_projects,
)


# =============================================================================
# ProjectSummary
# =============================================================================


class TestProjectSummary:
    def test_as_dict(self):
        p = ProjectSummary("proj-a", 3, 10, 0.85, 12.50, "2026-03-01")
        d = p.as_dict()
        assert d["project_name"] == "proj-a"
        assert d["dataset_count"] == 3
        assert d["latest_pass_rate"] == 0.85
        assert d["total_cost_usd"] == 12.50

    def test_from_dict(self):
        d = {
            "project_name": "proj-a",
            "dataset_count": 3,
            "run_count": 10,
            "latest_pass_rate": 0.85,
            "total_cost_usd": 12.50,
            "latest_run_date": "2026-03-01",
        }
        p = ProjectSummary.from_dict(d)
        assert p.project_name == "proj-a"
        assert p.dataset_count == 3

    def test_from_dict_roundtrip(self):
        p = ProjectSummary("proj-a", 3, 10, 0.85, 12.50, "2026-03-01")
        restored = ProjectSummary.from_dict(p.as_dict())
        assert restored.project_name == p.project_name
        assert restored.latest_pass_rate == p.latest_pass_rate
        assert restored.total_cost_usd == p.total_cost_usd

    def test_defaults(self):
        p = ProjectSummary("proj-a")
        assert p.dataset_count == 0
        assert p.latest_pass_rate is None
        assert p.total_cost_usd == 0.0
        assert p.latest_run_date == ""


# =============================================================================
# ProjectComparison
# =============================================================================


class TestProjectComparison:
    def test_as_dict(self):
        c = ProjectComparison("proj-a", "proj-b", 0.1, -5.0)
        d = c.as_dict()
        assert d["project_a"] == "proj-a"
        assert d["pass_rate_delta"] == 0.1
        assert d["cost_delta"] == -5.0

    def test_as_dict_none_pass_rate(self):
        c = ProjectComparison("proj-a", "proj-b", None, 2.0)
        d = c.as_dict()
        assert d["pass_rate_delta"] is None

    def test_format_text(self):
        c = ProjectComparison("proj-a", "proj-b", 0.1, -5.0)
        text = c.format_text()
        assert "proj-a" in text
        assert "proj-b" in text
        assert "Pass rate delta" in text

    def test_format_text_none_pass_rate(self):
        c = ProjectComparison("proj-a", "proj-b", None, 2.0)
        text = c.format_text()
        assert "N/A" in text


# =============================================================================
# MultiProjectReport
# =============================================================================


class TestMultiProjectReport:
    def test_as_dict(self):
        report = MultiProjectReport(
            projects=[ProjectSummary("proj-a", 1, 2, 0.9, 5.0)],
            total_projects=1,
            total_cost_usd=5.0,
        )
        d = report.as_dict()
        assert d["total_projects"] == 1
        assert len(d["projects"]) == 1

    def test_from_dict(self):
        d = {
            "projects": [
                {"project_name": "proj-a", "dataset_count": 1, "run_count": 2,
                 "latest_pass_rate": 0.9, "total_cost_usd": 5.0, "latest_run_date": ""}
            ],
            "comparisons": [],
            "total_projects": 1,
            "total_cost_usd": 5.0,
            "regressions": [],
        }
        report = MultiProjectReport.from_dict(d)
        assert report.total_projects == 1
        assert report.projects[0].project_name == "proj-a"

    def test_from_dict_roundtrip(self):
        report = MultiProjectReport(
            projects=[ProjectSummary("proj-a", 1, 2, 0.9, 5.0)],
            total_projects=1,
            total_cost_usd=5.0,
            regressions=["proj-b"],
        )
        restored = MultiProjectReport.from_dict(report.as_dict())
        assert restored.total_projects == report.total_projects
        assert restored.regressions == ["proj-b"]

    def test_format_text(self):
        report = MultiProjectReport(
            projects=[ProjectSummary("proj-a", 1, 2, 0.9, 5.0)],
            total_projects=1,
            total_cost_usd=5.0,
            regressions=["proj-b"],
        )
        text = report.format_text()
        assert "Multi-Project Report" in text
        assert "proj-a" in text
        assert "proj-b" in text

    def test_format_text_no_regressions(self):
        report = MultiProjectReport(total_projects=0, total_cost_usd=0.0)
        text = report.format_text()
        assert "Regressions" not in text


# =============================================================================
# create_project_summary
# =============================================================================


class TestCreateProjectSummary:
    def test_factory(self):
        p = create_project_summary("proj-a", datasets=3, runs=10, pass_rate=0.85, cost=12.50)
        assert p.project_name == "proj-a"
        assert p.dataset_count == 3
        assert p.run_count == 10
        assert p.latest_pass_rate == 0.85
        assert p.total_cost_usd == 12.50

    def test_defaults(self):
        p = create_project_summary("proj-a")
        assert p.dataset_count == 0
        assert p.latest_pass_rate is None
        assert p.total_cost_usd == 0.0


# =============================================================================
# compare_projects
# =============================================================================


class TestCompareProjects:
    def test_pass_rate_delta(self):
        a = ProjectSummary("a", latest_pass_rate=0.8)
        b = ProjectSummary("b", latest_pass_rate=0.9)
        c = compare_projects(a, b)
        assert c.pass_rate_delta == pytest.approx(0.1)

    def test_cost_delta(self):
        a = ProjectSummary("a", total_cost_usd=10.0)
        b = ProjectSummary("b", total_cost_usd=5.0)
        c = compare_projects(a, b)
        assert c.cost_delta == pytest.approx(-5.0)

    def test_none_pass_rate(self):
        a = ProjectSummary("a", latest_pass_rate=None)
        b = ProjectSummary("b", latest_pass_rate=0.9)
        c = compare_projects(a, b)
        assert c.pass_rate_delta is None

    def test_both_none_pass_rate(self):
        a = ProjectSummary("a", latest_pass_rate=None)
        b = ProjectSummary("b", latest_pass_rate=None)
        c = compare_projects(a, b)
        assert c.pass_rate_delta is None


# =============================================================================
# detect_regressions
# =============================================================================


class TestDetectRegressions:
    def test_finds_below_threshold(self):
        projects = [
            ProjectSummary("a", latest_pass_rate=0.9),
            ProjectSummary("b", latest_pass_rate=0.6),
            ProjectSummary("c", latest_pass_rate=0.75),
        ]
        result = detect_regressions(projects, min_pass_rate=0.8)
        assert set(result) == {"b", "c"}

    def test_none_below(self):
        projects = [
            ProjectSummary("a", latest_pass_rate=0.9),
            ProjectSummary("b", latest_pass_rate=0.85),
        ]
        result = detect_regressions(projects, min_pass_rate=0.8)
        assert result == []

    def test_none_pass_rate_excluded(self):
        projects = [
            ProjectSummary("a", latest_pass_rate=None),
            ProjectSummary("b", latest_pass_rate=0.9),
        ]
        result = detect_regressions(projects, min_pass_rate=0.8)
        assert result == []

    def test_empty_list(self):
        result = detect_regressions([], min_pass_rate=0.8)
        assert result == []


# =============================================================================
# build_multi_project_report
# =============================================================================


class TestBuildMultiProjectReport:
    def test_full_report(self):
        projects = [
            ProjectSummary("a", latest_pass_rate=0.9, total_cost_usd=10.0),
            ProjectSummary("b", latest_pass_rate=0.6, total_cost_usd=5.0),
        ]
        report = build_multi_project_report(projects)
        assert report.total_projects == 2
        assert report.total_cost_usd == pytest.approx(15.0)
        assert len(report.comparisons) == 1
        assert "b" in report.regressions

    def test_single_project(self):
        projects = [ProjectSummary("a", latest_pass_rate=0.9, total_cost_usd=10.0)]
        report = build_multi_project_report(projects)
        assert report.total_projects == 1
        assert len(report.comparisons) == 0

    def test_empty_list(self):
        report = build_multi_project_report([])
        assert report.total_projects == 0
        assert report.comparisons == []
        assert report.regressions == []

    def test_three_projects_pairwise(self):
        projects = [
            ProjectSummary("a", latest_pass_rate=0.9),
            ProjectSummary("b", latest_pass_rate=0.8),
            ProjectSummary("c", latest_pass_rate=0.7),
        ]
        report = build_multi_project_report(projects)
        # 3 choose 2 = 3 comparisons
        assert len(report.comparisons) == 3


# =============================================================================
# rank_projects
# =============================================================================


class TestRankProjects:
    def test_by_pass_rate(self):
        projects = [
            ProjectSummary("a", latest_pass_rate=0.7),
            ProjectSummary("b", latest_pass_rate=0.9),
            ProjectSummary("c", latest_pass_rate=0.8),
        ]
        ranked = rank_projects(projects, by="pass_rate")
        assert [p.project_name for p in ranked] == ["b", "c", "a"]

    def test_by_cost(self):
        projects = [
            ProjectSummary("a", total_cost_usd=20.0),
            ProjectSummary("b", total_cost_usd=5.0),
            ProjectSummary("c", total_cost_usd=10.0),
        ]
        ranked = rank_projects(projects, by="cost")
        assert [p.project_name for p in ranked] == ["b", "c", "a"]

    def test_none_pass_rate_sorted_last(self):
        projects = [
            ProjectSummary("a", latest_pass_rate=None),
            ProjectSummary("b", latest_pass_rate=0.9),
        ]
        ranked = rank_projects(projects, by="pass_rate")
        assert ranked[0].project_name == "b"
        assert ranked[1].project_name == "a"

    def test_empty_list(self):
        ranked = rank_projects([], by="pass_rate")
        assert ranked == []
