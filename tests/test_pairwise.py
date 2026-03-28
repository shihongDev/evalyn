"""Tests for the pairwise comparison module."""

from __future__ import annotations

import sys
from pathlib import Path

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.evaluation.pairwise import (
    PairwiseResult,
    EloRating,
    PairwiseReport,
    compare_pair,
    compute_elo_ratings,
    build_pairwise_report,
    expected_score,
)


# ---------------------------------------------------------------------------
# PairwiseResult
# ---------------------------------------------------------------------------


class TestPairwiseResult:
    def test_as_dict(self):
        r = PairwiseResult("i1", "A", "B", "a", 0.8, "longer output")
        d = r.as_dict()
        assert d["item_id"] == "i1"
        assert d["winner"] == "a"
        assert d["confidence"] == 0.8

    def test_from_dict(self):
        d = {"item_id": "i2", "model_a": "X", "model_b": "Y", "winner": "b"}
        r = PairwiseResult.from_dict(d)
        assert r.model_a == "X"
        assert r.confidence == 0.0
        assert r.reason == ""

    def test_roundtrip(self):
        r = PairwiseResult("i1", "A", "B", "tie", 0.5, "close call")
        r2 = PairwiseResult.from_dict(r.as_dict())
        assert r2.item_id == r.item_id
        assert r2.winner == r.winner
        assert r2.reason == r.reason


# ---------------------------------------------------------------------------
# EloRating
# ---------------------------------------------------------------------------


class TestEloRating:
    def test_defaults(self):
        e = EloRating(model="gpt4")
        assert e.rating == 1500.0
        assert e.wins == 0
        assert e.games == 0

    def test_win_rate_no_games(self):
        e = EloRating(model="m")
        assert e.win_rate == 0.0

    def test_win_rate_with_games(self):
        e = EloRating(model="m", wins=3, losses=1, ties=1, games=5)
        assert abs(e.win_rate - 0.6) < 1e-9

    def test_as_dict_from_dict_roundtrip(self):
        e = EloRating(model="m", rating=1600.0, wins=5, losses=2, ties=1, games=8)
        e2 = EloRating.from_dict(e.as_dict())
        assert e2.model == e.model
        assert e2.rating == 1600.0
        assert e2.wins == 5


# ---------------------------------------------------------------------------
# compare_pair
# ---------------------------------------------------------------------------


class TestComparePair:
    def test_a_wins_longer(self):
        r = compare_pair("i1", "a long detailed answer here", "short", "A", "B")
        assert r.winner == "a"
        assert r.confidence > 0.0

    def test_b_wins_longer(self):
        r = compare_pair("i1", "short", "a much longer and more detailed response")
        assert r.winner == "b"

    def test_tie_equal_length(self):
        r = compare_pair("i1", "hello", "world")
        assert r.winner == "tie"

    def test_tie_both_empty(self):
        r = compare_pair("i1", "", "")
        assert r.winner == "tie"
        assert r.confidence == 1.0

    def test_tie_whitespace_only(self):
        r = compare_pair("i1", "   ", "   ")
        assert r.winner == "tie"

    def test_model_names_carried(self):
        r = compare_pair("i1", "aaa", "b", "gpt4", "claude")
        assert r.model_a == "gpt4"
        assert r.model_b == "claude"

    def test_criterion_in_reason(self):
        r = compare_pair("i1", "longer output text", "x", criterion="helpfulness")
        assert "helpfulness" in r.reason

    def test_confidence_bounded(self):
        r = compare_pair("i1", "x" * 1000, "y")
        assert 0.0 <= r.confidence <= 1.0


# ---------------------------------------------------------------------------
# expected_score
# ---------------------------------------------------------------------------


class TestExpectedScore:
    def test_equal_ratings(self):
        assert abs(expected_score(1500.0, 1500.0) - 0.5) < 1e-9

    def test_higher_rating_favored(self):
        es = expected_score(1700.0, 1500.0)
        assert es > 0.5

    def test_lower_rating_disadvantaged(self):
        es = expected_score(1300.0, 1500.0)
        assert es < 0.5

    def test_symmetric(self):
        ea = expected_score(1600.0, 1400.0)
        eb = expected_score(1400.0, 1600.0)
        assert abs(ea + eb - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# compute_elo_ratings
# ---------------------------------------------------------------------------


class TestComputeEloRatings:
    def test_initialization(self):
        results = [PairwiseResult("i1", "A", "B", "a")]
        ratings = compute_elo_ratings(results)
        assert "A" in ratings
        assert "B" in ratings

    def test_winner_gains_rating(self):
        results = [PairwiseResult(f"i{i}", "A", "B", "a") for i in range(10)]
        ratings = compute_elo_ratings(results)
        assert ratings["A"].rating > 1500.0
        assert ratings["B"].rating < 1500.0

    def test_wins_losses_counted(self):
        results = [
            PairwiseResult("i1", "A", "B", "a"),
            PairwiseResult("i2", "A", "B", "b"),
            PairwiseResult("i3", "A", "B", "tie"),
        ]
        ratings = compute_elo_ratings(results)
        assert ratings["A"].wins == 1
        assert ratings["A"].losses == 1
        assert ratings["A"].ties == 1
        assert ratings["A"].games == 3

    def test_ratings_converge(self):
        """Many games between equal players: ratings stay near initial."""
        results = []
        for i in range(100):
            w = "a" if i % 2 == 0 else "b"
            results.append(PairwiseResult(f"i{i}", "A", "B", w))
        ratings = compute_elo_ratings(results)
        assert abs(ratings["A"].rating - 1500.0) < 50.0
        assert abs(ratings["B"].rating - 1500.0) < 50.0

    def test_custom_initial_rating(self):
        results = [PairwiseResult("i1", "A", "B", "tie")]
        ratings = compute_elo_ratings(results, initial_rating=1000.0)
        # After a tie between equally rated players, ratings should stay near 1000
        assert abs(ratings["A"].rating - 1000.0) < 1.0

    def test_k_factor_sensitivity(self):
        """Higher k_factor should produce larger rating changes."""
        results = [PairwiseResult("i1", "A", "B", "a")]
        low_k = compute_elo_ratings(results, k_factor=16.0)
        high_k = compute_elo_ratings(results, k_factor=64.0)
        delta_low = abs(low_k["A"].rating - 1500.0)
        delta_high = abs(high_k["A"].rating - 1500.0)
        assert delta_high > delta_low

    def test_three_models(self):
        results = [
            PairwiseResult("i1", "A", "B", "a"),
            PairwiseResult("i2", "B", "C", "b"),
            PairwiseResult("i3", "A", "C", "a"),
        ]
        ratings = compute_elo_ratings(results)
        assert len(ratings) == 3
        # A beats B and C, so A should be highest
        assert ratings["A"].rating > ratings["C"].rating

    def test_empty_results(self):
        ratings = compute_elo_ratings([])
        assert len(ratings) == 0


# ---------------------------------------------------------------------------
# PairwiseReport
# ---------------------------------------------------------------------------


class TestPairwiseReport:
    def _sample_results(self) -> list:
        return [
            PairwiseResult("i1", "A", "B", "a"),
            PairwiseResult("i2", "A", "B", "b"),
            PairwiseResult("i3", "A", "B", "a"),
            PairwiseResult("i4", "B", "C", "b"),
        ]

    def test_win_loss_matrix(self):
        report = build_pairwise_report(self._sample_results())
        m = report.win_loss_matrix
        assert m["A"]["B"] == 2  # A beat B twice
        assert m["B"]["A"] == 1  # B beat A once

    def test_win_loss_matrix_includes_all_models(self):
        report = build_pairwise_report(self._sample_results())
        m = report.win_loss_matrix
        assert "C" in m

    def test_format_text(self):
        report = build_pairwise_report(self._sample_results())
        text = report.format_text()
        assert "Pairwise Report" in text
        assert "Elo Ratings" in text

    def test_total_comparisons(self):
        results = self._sample_results()
        report = build_pairwise_report(results)
        assert report.total_comparisons == 4

    def test_as_dict_from_dict_roundtrip(self):
        report = build_pairwise_report(self._sample_results())
        d = report.as_dict()
        report2 = PairwiseReport.from_dict(d)
        assert report2.total_comparisons == report.total_comparisons
        assert len(report2.results) == len(report.results)
        assert set(report2.ratings.keys()) == set(report.ratings.keys())

    def test_empty_report(self):
        report = build_pairwise_report([])
        assert report.total_comparisons == 0
        assert report.win_loss_matrix == {}
