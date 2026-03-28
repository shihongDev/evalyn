"""Tests for hybrid confidence scoring."""

import pytest
from evalyn_sdk.evaluation.hybrid_confidence import (
    MethodScore,
    HybridScore,
    compute_hybrid_score,
    compute_with_bayesian_update,
    DEFAULT_WEIGHTS,
)


# =============================================================================
# MethodScore
# =============================================================================

class TestMethodScore:
    def test_as_dict(self):
        ms = MethodScore(method="logprobs", score=0.85, available=True, weight=0.35)
        d = ms.as_dict()
        assert d["method"] == "logprobs"
        assert d["score"] == 0.85


# =============================================================================
# HybridScore
# =============================================================================

class TestHybridScore:
    def test_as_dict(self):
        h = HybridScore(combined_score=0.8, methods_used=2, methods_unavailable=["deepconf"])
        d = h.as_dict()
        assert d["combined_score"] == 0.8
        assert d["methods_used"] == 2
        assert "deepconf" in d["methods_unavailable"]

    def test_from_dict(self):
        d = {
            "combined_score": 0.75,
            "methods_used": 1,
            "methods_unavailable": ["x"],
            "method_scores": [
                {"method": "a", "score": 0.75, "available": True, "weight": 1.0},
            ],
        }
        h = HybridScore.from_dict(d)
        assert h.combined_score == 0.75
        assert len(h.method_scores) == 1

    def test_format_text(self):
        h = HybridScore(
            combined_score=0.8,
            method_scores=[MethodScore("logprobs", 0.9, True, 0.35)],
            methods_unavailable=["deepconf"],
        )
        text = h.format_text()
        assert "0.8" in text
        assert "logprobs" in text
        assert "Fallbacks" in text


# =============================================================================
# compute_hybrid_score
# =============================================================================

class TestComputeHybridScore:
    def test_single_method(self):
        h = compute_hybrid_score({"logprobs": 0.9})
        assert h.combined_score == pytest.approx(0.9)
        assert h.methods_used == 1

    def test_two_equal_methods(self):
        h = compute_hybrid_score(
            {"a": 0.8, "b": 0.6},
            weights={"a": 1.0, "b": 1.0},
        )
        assert h.combined_score == pytest.approx(0.7)

    def test_weighted_combination(self):
        h = compute_hybrid_score(
            {"a": 1.0, "b": 0.0},
            weights={"a": 3.0, "b": 1.0},
        )
        assert h.combined_score == pytest.approx(0.75)

    def test_unavailable_method_excluded(self):
        h = compute_hybrid_score(
            {"a": 0.8, "b": None},
            weights={"a": 1.0, "b": 1.0},
        )
        assert h.combined_score == pytest.approx(0.8)
        assert h.methods_used == 1
        assert "b" in h.methods_unavailable

    def test_all_unavailable(self):
        h = compute_hybrid_score({"a": None, "b": None})
        assert h.combined_score == 0.0
        assert h.methods_used == 0

    def test_default_weights_used(self):
        h = compute_hybrid_score({"logprobs": 0.9, "consistency": 0.7})
        # Should use DEFAULT_WEIGHTS
        assert h.methods_used == 2
        # Weighted average with 0.35 and 0.30
        expected = (0.35 * 0.9 + 0.30 * 0.7) / (0.35 + 0.30)
        assert h.combined_score == pytest.approx(expected, abs=0.01)

    def test_renormalization(self):
        # When one method is unavailable, remaining weight renormalized to 1.0
        h = compute_hybrid_score(
            {"logprobs": 0.9, "consistency": None, "verbalized": 0.7},
        )
        assert h.methods_used == 2
        assert 0.7 <= h.combined_score <= 0.9

    def test_all_methods(self):
        h = compute_hybrid_score({
            "logprobs": 0.9,
            "consistency": 0.8,
            "verbalized": 0.7,
            "deepconf": 0.6,
        })
        assert h.methods_used == 4
        assert 0.6 <= h.combined_score <= 0.9


# =============================================================================
# compute_with_bayesian_update
# =============================================================================

class TestBayesianUpdate:
    def test_single_high_confidence(self):
        h = compute_with_bayesian_update({"a": 0.9})
        assert h.combined_score > 0.5

    def test_single_low_confidence(self):
        h = compute_with_bayesian_update({"a": 0.1})
        assert h.combined_score < 0.5

    def test_prior_50_with_neutral_evidence(self):
        h = compute_with_bayesian_update({"a": 0.5}, prior=0.5)
        assert h.combined_score == pytest.approx(0.5, abs=0.01)

    def test_multiple_agreeing_methods_strengthen(self):
        single = compute_with_bayesian_update({"a": 0.9})
        double = compute_with_bayesian_update({"a": 0.9, "b": 0.9})
        assert double.combined_score > single.combined_score

    def test_unavailable_methods_skipped(self):
        h = compute_with_bayesian_update({"a": 0.9, "b": None})
        assert h.methods_used == 1
        assert "b" in h.methods_unavailable
        assert h.combined_score > 0.5

    def test_all_unavailable(self):
        h = compute_with_bayesian_update({"a": None}, prior=0.5)
        assert h.combined_score == 0.5  # falls back to prior

    def test_custom_reliability(self):
        # High reliability method should shift posterior more
        h_high = compute_with_bayesian_update(
            {"a": 0.9}, method_reliability={"a": 0.99},
        )
        h_low = compute_with_bayesian_update(
            {"a": 0.9}, method_reliability={"a": 0.5},
        )
        assert h_high.combined_score > h_low.combined_score

    def test_conflicting_methods(self):
        h = compute_with_bayesian_update({"a": 0.9, "b": 0.1})
        # Conflicting evidence should keep posterior near prior
        assert 0.3 <= h.combined_score <= 0.7

    def test_score_in_valid_range(self):
        h = compute_with_bayesian_update({
            "a": 0.99, "b": 0.99, "c": 0.99,
        })
        assert 0.0 <= h.combined_score <= 1.0
