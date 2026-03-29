"""Tests for simulation coverage report module."""

from __future__ import annotations

import sys
from pathlib import Path

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.coverage_report import (
    CoverageRegion,
    CoverageResult,
    extract_keywords,
    compute_vocabulary_coverage,
    identify_coverage_gaps,
    build_coverage_report,
    suggest_simulation_targets,
    format_coverage_report,
)


# ---------------------------------------------------------------------------
# CoverageRegion dataclass
# ---------------------------------------------------------------------------


class TestCoverageRegion:
    def test_basic_construction(self):
        r = CoverageRegion(
            region_id="r1", keywords=["alpha", "bravo"], simulated_count=5,
            production_count=20, gap=4.0,
        )
        assert r.region_id == "r1"
        assert r.gap == 4.0

    def test_as_dict(self):
        r = CoverageRegion(
            region_id="r1", keywords=["kw1"], simulated_count=1,
            production_count=10, gap=10.0,
        )
        d = r.as_dict()
        assert d["region_id"] == "r1"
        assert d["keywords"] == ["kw1"]
        assert d["gap"] == 10.0

    def test_from_dict(self):
        d = {
            "region_id": "r2", "keywords": ["x", "y"],
            "simulated_count": 3, "production_count": 9, "gap": 3.0,
        }
        r = CoverageRegion.from_dict(d)
        assert r.region_id == "r2"
        assert r.keywords == ["x", "y"]

    def test_roundtrip(self):
        original = CoverageRegion(
            region_id="r1", keywords=["a", "b", "c"],
            simulated_count=2, production_count=8, gap=4.0,
        )
        restored = CoverageRegion.from_dict(original.as_dict())
        assert restored.region_id == original.region_id
        assert restored.keywords == original.keywords
        assert restored.gap == original.gap

    def test_gap_zero_simulated(self):
        r = CoverageRegion(
            region_id="r1", keywords=["kw"], simulated_count=0,
            production_count=5, gap=5.0,
        )
        assert r.gap == 5.0


# ---------------------------------------------------------------------------
# CoverageResult dataclass
# ---------------------------------------------------------------------------


class TestCoverageResult:
    def test_basic_construction(self):
        r = CoverageResult(
            overlap_score=0.75, total_sim_items=100, total_prod_items=200,
            uncovered_regions=[], recommendations=["do something"],
        )
        assert r.overlap_score == 0.75
        assert r.total_sim_items == 100

    def test_as_dict(self):
        region = CoverageRegion("r1", ["kw"], 1, 5, 5.0)
        r = CoverageResult(
            overlap_score=0.5, total_sim_items=10, total_prod_items=20,
            uncovered_regions=[region], recommendations=["rec1"],
        )
        d = r.as_dict()
        assert d["overlap_score"] == 0.5
        assert len(d["uncovered_regions"]) == 1
        assert d["recommendations"] == ["rec1"]

    def test_from_dict(self):
        d = {
            "overlap_score": 0.6,
            "total_sim_items": 50,
            "total_prod_items": 100,
            "uncovered_regions": [
                {"region_id": "r1", "keywords": ["k"], "simulated_count": 0,
                 "production_count": 10, "gap": 10.0},
            ],
            "recommendations": ["fix gaps"],
        }
        r = CoverageResult.from_dict(d)
        assert r.overlap_score == 0.6
        assert len(r.uncovered_regions) == 1
        assert r.uncovered_regions[0].region_id == "r1"

    def test_roundtrip(self):
        region = CoverageRegion("r1", ["kw1", "kw2"], 2, 10, 5.0)
        original = CoverageResult(
            overlap_score=0.8, total_sim_items=50, total_prod_items=100,
            uncovered_regions=[region], recommendations=["rec"],
        )
        restored = CoverageResult.from_dict(original.as_dict())
        assert restored.overlap_score == original.overlap_score
        assert len(restored.uncovered_regions) == 1
        assert restored.recommendations == original.recommendations


# ---------------------------------------------------------------------------
# extract_keywords
# ---------------------------------------------------------------------------


class TestExtractKeywords:
    def test_empty_texts(self):
        assert extract_keywords([]) == []

    def test_basic_extraction(self):
        texts = ["python programming language", "python data science"]
        keywords = extract_keywords(texts, top_n=10)
        assert "python" in keywords

    def test_stop_words_filtered(self):
        texts = ["the and but or for with"]
        keywords = extract_keywords(texts, top_n=10)
        assert keywords == []

    def test_short_words_filtered(self):
        texts = ["ab cd ef gh ij"]
        keywords = extract_keywords(texts, top_n=10)
        assert keywords == []

    def test_top_n_limit(self):
        texts = ["alpha bravo charlie delta echo foxtrot golf hotel"]
        keywords = extract_keywords(texts, top_n=3)
        assert len(keywords) <= 3

    def test_frequency_ordering(self):
        texts = [
            "python python python java java rust",
            "python python java",
        ]
        keywords = extract_keywords(texts, top_n=3)
        assert keywords[0] == "python"  # most frequent

    def test_lowercase(self):
        texts = ["Python PYTHON python"]
        keywords = extract_keywords(texts, top_n=5)
        assert "python" in keywords

    def test_multiple_texts(self):
        texts = [
            "machine learning algorithms",
            "deep learning networks",
            "learning optimization techniques",
        ]
        keywords = extract_keywords(texts, top_n=5)
        assert "learning" in keywords


# ---------------------------------------------------------------------------
# compute_vocabulary_coverage
# ---------------------------------------------------------------------------


class TestComputeVocabularyCoverage:
    def test_empty_production(self):
        assert compute_vocabulary_coverage(["some text"], []) == 0.0

    def test_empty_simulation(self):
        coverage = compute_vocabulary_coverage([], ["some text here"])
        assert coverage == 0.0

    def test_both_empty(self):
        assert compute_vocabulary_coverage([], []) == 0.0

    def test_perfect_coverage(self):
        texts = ["alpha bravo charlie delta"]
        coverage = compute_vocabulary_coverage(texts, texts)
        assert coverage == 1.0

    def test_no_coverage(self):
        sim = ["alpha bravo charlie"]
        prod = ["delta echo foxtrot"]
        coverage = compute_vocabulary_coverage(sim, prod)
        assert coverage == 0.0

    def test_partial_coverage(self):
        sim = ["alpha bravo charlie"]
        prod = ["alpha bravo delta"]
        coverage = compute_vocabulary_coverage(sim, prod)
        assert 0.0 < coverage < 1.0

    def test_coverage_between_zero_and_one(self):
        sim = ["python programming"]
        prod = ["python data science programming"]
        coverage = compute_vocabulary_coverage(sim, prod)
        assert 0.0 <= coverage <= 1.0

    def test_simulation_superset(self):
        sim = ["alpha bravo charlie delta echo"]
        prod = ["alpha bravo charlie"]
        coverage = compute_vocabulary_coverage(sim, prod)
        assert coverage == 1.0


# ---------------------------------------------------------------------------
# identify_coverage_gaps
# ---------------------------------------------------------------------------


class TestIdentifyCoverageGaps:
    def test_no_gaps_when_identical(self):
        texts = ["alpha bravo charlie delta"]
        gaps = identify_coverage_gaps(texts, texts)
        assert gaps == []

    def test_gaps_found_when_different(self):
        sim = ["alpha bravo charlie"]
        prod = ["delta echo foxtrot golf hotel"]
        gaps = identify_coverage_gaps(sim, prod)
        assert len(gaps) > 0

    def test_n_regions_respected(self):
        sim = ["alpha"]
        prod = ["bravo charlie delta echo foxtrot golf hotel india juliet kilo"]
        gaps = identify_coverage_gaps(sim, prod, n_regions=3)
        assert len(gaps) <= 3

    def test_gap_score_calculation(self):
        sim = ["alpha bravo"]
        prod = ["charlie delta echo foxtrot golf"]
        gaps = identify_coverage_gaps(sim, prod)
        for gap in gaps:
            assert gap.gap >= 0
            expected = gap.production_count / max(1, gap.simulated_count)
            assert abs(gap.gap - expected) < 0.01

    def test_region_has_keywords(self):
        sim = ["alpha"]
        prod = ["bravo charlie delta echo"]
        gaps = identify_coverage_gaps(sim, prod)
        for gap in gaps:
            assert len(gap.keywords) > 0

    def test_region_ids_unique(self):
        sim = ["alpha"]
        prod = ["bravo charlie delta echo foxtrot golf hotel"]
        gaps = identify_coverage_gaps(sim, prod, n_regions=5)
        ids = [g.region_id for g in gaps]
        assert len(ids) == len(set(ids))

    def test_empty_production(self):
        gaps = identify_coverage_gaps(["some text"], [])
        assert gaps == []

    def test_empty_simulation(self):
        gaps = identify_coverage_gaps([], ["alpha bravo charlie"])
        assert len(gaps) > 0

    def test_sorted_by_gap_descending(self):
        sim = ["alpha"]
        prod = [
            "bravo bravo bravo",
            "charlie charlie",
            "delta",
        ]
        gaps = identify_coverage_gaps(sim, prod, n_regions=3)
        if len(gaps) >= 2:
            for i in range(len(gaps) - 1):
                assert gaps[i].gap >= gaps[i + 1].gap


# ---------------------------------------------------------------------------
# build_coverage_report
# ---------------------------------------------------------------------------


class TestBuildCoverageReport:
    def test_basic_report(self):
        sim = ["alpha bravo charlie"]
        prod = ["alpha delta echo"]
        result = build_coverage_report(sim, prod)
        assert isinstance(result, CoverageResult)
        assert result.total_sim_items == 1
        assert result.total_prod_items == 1

    def test_overlap_score_present(self):
        sim = ["alpha bravo charlie"]
        prod = ["alpha bravo delta"]
        result = build_coverage_report(sim, prod)
        assert 0.0 <= result.overlap_score <= 1.0

    def test_recommendations_present(self):
        sim = ["alpha bravo charlie"]
        prod = ["delta echo foxtrot"]
        result = build_coverage_report(sim, prod)
        assert len(result.recommendations) > 0

    def test_empty_inputs(self):
        result = build_coverage_report([], [])
        assert result.overlap_score == 0.0
        assert result.total_sim_items == 0
        assert result.total_prod_items == 0

    def test_perfect_coverage_report(self):
        texts = ["alpha bravo charlie delta echo foxtrot"]
        result = build_coverage_report(texts, texts)
        assert result.overlap_score == 1.0
        assert len(result.uncovered_regions) == 0

    def test_low_coverage_gets_critical_recommendation(self):
        sim = ["zzz yyy xxx"]
        prod = [
            "alpha bravo charlie delta echo foxtrot golf hotel india juliet",
            "kilo lima mike november oscar papa quebec romeo sierra tango",
        ]
        result = build_coverage_report(sim, prod)
        assert any("30%" in r or "critical" in r.lower() or "Critical" in r for r in result.recommendations)


# ---------------------------------------------------------------------------
# suggest_simulation_targets
# ---------------------------------------------------------------------------


class TestSuggestSimulationTargets:
    def test_empty_gaps(self):
        assert suggest_simulation_targets([]) == []

    def test_one_gap(self):
        gap = CoverageRegion("r1", ["alpha", "bravo"], 0, 10, 10.0)
        suggestions = suggest_simulation_targets([gap])
        assert len(suggestions) == 1
        assert "alpha" in suggestions[0]

    def test_multiple_gaps(self):
        gaps = [
            CoverageRegion("r1", ["kw1"], 0, 10, 10.0),
            CoverageRegion("r2", ["kw2"], 1, 5, 5.0),
        ]
        suggestions = suggest_simulation_targets(gaps)
        assert len(suggestions) == 2

    def test_ordered_by_gap(self):
        gaps = [
            CoverageRegion("r1", ["small"], 5, 10, 2.0),
            CoverageRegion("r2", ["big"], 0, 20, 20.0),
        ]
        suggestions = suggest_simulation_targets(gaps)
        assert "big" in suggestions[0]
        assert "small" in suggestions[1]

    def test_suggestion_contains_keywords(self):
        gap = CoverageRegion("r1", ["python", "machine", "learning"], 0, 15, 15.0)
        suggestions = suggest_simulation_targets([gap])
        assert "python" in suggestions[0]
        assert "machine" in suggestions[0]

    def test_gap_score_in_suggestion(self):
        gap = CoverageRegion("r1", ["keyword"], 0, 10, 10.0)
        suggestions = suggest_simulation_targets([gap])
        assert "10.0" in suggestions[0]


# ---------------------------------------------------------------------------
# format_coverage_report
# ---------------------------------------------------------------------------


class TestFormatCoverageReport:
    def test_empty_result(self):
        result = CoverageResult(
            overlap_score=0.0, total_sim_items=0, total_prod_items=0,
            uncovered_regions=[], recommendations=[],
        )
        report = format_coverage_report(result)
        assert "SIMULATION COVERAGE REPORT" in report

    def test_includes_overlap(self):
        result = CoverageResult(
            overlap_score=0.75, total_sim_items=10, total_prod_items=20,
            uncovered_regions=[], recommendations=[],
        )
        report = format_coverage_report(result)
        assert "75.0%" in report

    def test_includes_item_counts(self):
        result = CoverageResult(
            overlap_score=0.5, total_sim_items=42, total_prod_items=99,
            uncovered_regions=[], recommendations=[],
        )
        report = format_coverage_report(result)
        assert "42" in report
        assert "99" in report

    def test_includes_regions(self):
        region = CoverageRegion("r1", ["keyword"], 0, 10, 10.0)
        result = CoverageResult(
            overlap_score=0.5, total_sim_items=10, total_prod_items=20,
            uncovered_regions=[region], recommendations=[],
        )
        report = format_coverage_report(result)
        assert "r1" in report
        assert "keyword" in report

    def test_includes_recommendations(self):
        result = CoverageResult(
            overlap_score=0.5, total_sim_items=10, total_prod_items=20,
            uncovered_regions=[], recommendations=["Improve coverage now"],
        )
        report = format_coverage_report(result)
        assert "Improve coverage now" in report

    def test_report_is_string(self):
        result = CoverageResult(
            overlap_score=0.0, total_sim_items=0, total_prod_items=0,
            uncovered_regions=[], recommendations=[],
        )
        assert isinstance(format_coverage_report(result), str)
