"""Tests for progressive sampling module."""

from __future__ import annotations

import math

import pytest

from evalyn_sdk.progressive_sampling import (
    ProgressiveConfig,
    ProgressiveResult,
    ProgressiveRound,
    compute_ci_width,
    expand_sample,
    format_progressive_report,
    is_sufficient,
    run_progressive_sampling,
)


# ---- ProgressiveConfig ----


class TestProgressiveConfig:
    def test_defaults(self):
        c = ProgressiveConfig()
        assert c.initial_size == 20
        assert c.expansion_factor == 2.0
        assert c.max_size == 500
        assert c.ci_width_threshold == 0.1
        assert c.max_rounds == 5
        assert c.seed is None

    def test_custom_values(self):
        c = ProgressiveConfig(
            initial_size=10,
            expansion_factor=1.5,
            max_size=100,
            ci_width_threshold=0.05,
            max_rounds=3,
            seed=7,
        )
        assert c.initial_size == 10
        assert c.seed == 7

    def test_as_dict(self):
        c = ProgressiveConfig(seed=42)
        d = c.as_dict()
        assert d == {
            "initial_size": 20,
            "expansion_factor": 2.0,
            "max_size": 500,
            "ci_width_threshold": 0.1,
            "max_rounds": 5,
            "seed": 42,
        }

    def test_from_dict_full(self):
        d = {
            "initial_size": 15,
            "expansion_factor": 3.0,
            "max_size": 200,
            "ci_width_threshold": 0.05,
            "max_rounds": 10,
            "seed": 99,
        }
        c = ProgressiveConfig.from_dict(d)
        assert c.initial_size == 15
        assert c.expansion_factor == 3.0
        assert c.seed == 99

    def test_from_dict_defaults(self):
        c = ProgressiveConfig.from_dict({})
        assert c.initial_size == 20
        assert c.seed is None

    def test_roundtrip(self):
        c = ProgressiveConfig(
            initial_size=30,
            expansion_factor=1.5,
            max_size=300,
            ci_width_threshold=0.08,
            max_rounds=4,
            seed=11,
        )
        assert ProgressiveConfig.from_dict(c.as_dict()) == c


# ---- ProgressiveRound ----


class TestProgressiveRound:
    def test_creation(self):
        r = ProgressiveRound(round_number=1, sample_size=20, ci_width=0.15, sufficient=False)
        assert r.round_number == 1
        assert r.sufficient is False

    def test_as_dict(self):
        r = ProgressiveRound(round_number=2, sample_size=40, ci_width=0.08, sufficient=True)
        d = r.as_dict()
        assert d == {
            "round_number": 2,
            "sample_size": 40,
            "ci_width": 0.08,
            "sufficient": True,
        }

    def test_from_dict(self):
        d = {"round_number": 3, "sample_size": 80, "ci_width": 0.05, "sufficient": True}
        r = ProgressiveRound.from_dict(d)
        assert r.round_number == 3
        assert r.ci_width == 0.05

    def test_roundtrip(self):
        r = ProgressiveRound(round_number=1, sample_size=20, ci_width=0.12, sufficient=False)
        assert ProgressiveRound.from_dict(r.as_dict()) == r


# ---- ProgressiveResult ----


class TestProgressiveResult:
    def test_defaults(self):
        r = ProgressiveResult()
        assert r.final_sample_ids == []
        assert r.rounds == []
        assert r.converged is False

    def test_as_dict(self):
        r = ProgressiveResult(
            final_sample_ids=["a", "b"],
            rounds=[ProgressiveRound(1, 20, 0.1, True)],
            total_rounds=1,
            final_ci_width=0.1,
            converged=True,
        )
        d = r.as_dict()
        assert d["final_sample_ids"] == ["a", "b"]
        assert len(d["rounds"]) == 1
        assert d["converged"] is True

    def test_from_dict(self):
        d = {
            "final_sample_ids": ["x"],
            "rounds": [{"round_number": 1, "sample_size": 10, "ci_width": 0.2, "sufficient": False}],
            "total_rounds": 1,
            "final_ci_width": 0.2,
            "converged": False,
        }
        r = ProgressiveResult.from_dict(d)
        assert r.final_sample_ids == ["x"]
        assert r.rounds[0].ci_width == 0.2

    def test_from_dict_empty(self):
        r = ProgressiveResult.from_dict({})
        assert r.final_sample_ids == []
        assert r.converged is False

    def test_roundtrip(self):
        r = ProgressiveResult(
            final_sample_ids=["a"],
            rounds=[ProgressiveRound(1, 20, 0.08, True)],
            total_rounds=1,
            final_ci_width=0.08,
            converged=True,
        )
        r2 = ProgressiveResult.from_dict(r.as_dict())
        assert r2.final_sample_ids == r.final_sample_ids
        assert r2.converged == r.converged


# ---- compute_ci_width ----


class TestComputeCiWidth:
    def test_constant_scores(self):
        scores = [0.5] * 20
        ci = compute_ci_width(scores)
        assert ci == 0.0

    def test_known_values(self):
        # std of [0, 1] with ddof=1 is sqrt(0.5) ~ 0.7071
        # ci = 1.96 * 0.7071 / sqrt(2) ~ 0.98
        ci = compute_ci_width([0.0, 1.0])
        assert abs(ci - 1.96 * (1.0 / math.sqrt(2)) / math.sqrt(2)) < 0.001

    def test_fewer_than_two_scores(self):
        assert compute_ci_width([]) == 0.0
        assert compute_ci_width([0.5]) == 0.0

    def test_wider_for_high_variance(self):
        low_var = compute_ci_width([0.49, 0.50, 0.51, 0.50, 0.49])
        high_var = compute_ci_width([0.0, 1.0, 0.0, 1.0, 0.0])
        assert high_var > low_var

    def test_narrower_with_more_samples(self):
        few = compute_ci_width([0.3, 0.7, 0.3, 0.7])
        many = compute_ci_width([0.3, 0.7] * 50)
        assert many < few

    def test_non_negative(self):
        import random as _rng
        r = _rng.Random(42)
        scores = [r.random() for _ in range(100)]
        assert compute_ci_width(scores) >= 0.0


# ---- is_sufficient ----


class TestIsSufficient:
    def test_below_threshold(self):
        assert is_sufficient(0.05, 0.1) is True

    def test_at_threshold(self):
        assert is_sufficient(0.1, 0.1) is True

    def test_above_threshold(self):
        assert is_sufficient(0.15, 0.1) is False

    def test_zero_width(self):
        assert is_sufficient(0.0, 0.1) is True

    def test_zero_threshold(self):
        assert is_sufficient(0.0, 0.0) is True
        assert is_sufficient(0.01, 0.0) is False


# ---- expand_sample ----


class TestExpandSample:
    def test_basic_expansion(self):
        current = ["a", "b"]
        all_ids = ["a", "b", "c", "d", "e", "f"]
        expanded = expand_sample(current, all_ids, expansion_factor=2.0, max_size=100, seed=42)
        assert len(expanded) == 4  # 2 * 2.0
        assert expanded[:2] == ["a", "b"]
        for item in expanded[2:]:
            assert item not in ["a", "b"]

    def test_respects_max_size(self):
        current = ["a", "b", "c"]
        all_ids = [f"x{i}" for i in range(100)] + ["a", "b", "c"]
        expanded = expand_sample(current, all_ids, expansion_factor=10.0, max_size=5, seed=42)
        assert len(expanded) == 5

    def test_pool_exhausted(self):
        current = ["a", "b"]
        all_ids = ["a", "b", "c"]
        expanded = expand_sample(current, all_ids, expansion_factor=10.0, max_size=100, seed=42)
        assert len(expanded) == 3
        assert "c" in expanded

    def test_no_expansion_when_at_max(self):
        current = ["a", "b", "c"]
        all_ids = ["a", "b", "c", "d"]
        expanded = expand_sample(current, all_ids, expansion_factor=1.0, max_size=3, seed=42)
        # target = min(3 * 1.0, 3) = 3, no new items to add
        assert expanded == ["a", "b", "c"]

    def test_deterministic_with_seed(self):
        current = ["a"]
        all_ids = list("abcdefghij")
        a = expand_sample(current, all_ids, 3.0, 100, seed=42)
        b = expand_sample(current, all_ids, 3.0, 100, seed=42)
        assert a == b

    def test_empty_current(self):
        # expansion_factor * 0 = 0, so nothing to add
        expanded = expand_sample([], ["a", "b"], 2.0, 100, seed=42)
        assert expanded == []

    def test_empty_pool(self):
        expanded = expand_sample(["a"], [], 2.0, 100, seed=42)
        assert expanded == ["a"]


# ---- run_progressive_sampling ----


class TestRunProgressiveSampling:
    def _constant_evaluator(self, ids: list[str]) -> list[float]:
        """All scores 0.5 - very tight CI."""
        return [0.5] * len(ids)

    def _noisy_evaluator(self, ids: list[str]) -> list[float]:
        """Alternating 0/1 - wide CI that shrinks with more samples."""
        return [float(i % 2) for i in range(len(ids))]

    def test_converges_quickly_for_constant(self):
        all_ids = [f"item_{i}" for i in range(100)]
        result = run_progressive_sampling(all_ids, self._constant_evaluator)
        assert result.converged is True
        assert result.total_rounds == 1

    def test_expands_for_noisy_data(self):
        all_ids = [f"item_{i}" for i in range(500)]
        config = ProgressiveConfig(
            initial_size=5,
            expansion_factor=2.0,
            max_size=500,
            ci_width_threshold=0.05,
            max_rounds=10,
            seed=42,
        )
        result = run_progressive_sampling(all_ids, self._noisy_evaluator, config)
        assert result.total_rounds > 1

    def test_max_rounds_respected(self):
        all_ids = [f"item_{i}" for i in range(1000)]
        config = ProgressiveConfig(
            initial_size=5,
            max_rounds=3,
            ci_width_threshold=0.001,  # very tight
            seed=42,
        )
        result = run_progressive_sampling(all_ids, self._noisy_evaluator, config)
        assert result.total_rounds <= 3

    def test_max_size_respected(self):
        all_ids = [f"item_{i}" for i in range(1000)]
        config = ProgressiveConfig(
            initial_size=10,
            expansion_factor=2.0,
            max_size=30,
            ci_width_threshold=0.001,
            max_rounds=10,
            seed=42,
        )
        result = run_progressive_sampling(all_ids, self._noisy_evaluator, config)
        assert len(result.final_sample_ids) <= 30

    def test_empty_pool(self):
        result = run_progressive_sampling([], self._constant_evaluator)
        assert result.converged is True
        assert result.final_sample_ids == []
        assert result.total_rounds == 0

    def test_pool_smaller_than_initial(self):
        all_ids = ["a", "b", "c"]
        config = ProgressiveConfig(initial_size=10, seed=42)
        result = run_progressive_sampling(all_ids, self._constant_evaluator, config)
        assert len(result.final_sample_ids) == 3

    def test_default_config_used(self):
        all_ids = [f"item_{i}" for i in range(100)]
        result = run_progressive_sampling(all_ids, self._constant_evaluator)
        assert isinstance(result, ProgressiveResult)

    def test_rounds_recorded(self):
        all_ids = [f"item_{i}" for i in range(200)]
        config = ProgressiveConfig(
            initial_size=10,
            ci_width_threshold=0.001,
            max_rounds=3,
            seed=42,
        )
        result = run_progressive_sampling(all_ids, self._noisy_evaluator, config)
        assert len(result.rounds) == result.total_rounds
        for r in result.rounds:
            assert r.sample_size > 0
            assert r.ci_width >= 0

    def test_sample_sizes_increase(self):
        all_ids = [f"item_{i}" for i in range(500)]
        config = ProgressiveConfig(
            initial_size=10,
            expansion_factor=2.0,
            ci_width_threshold=0.001,
            max_rounds=5,
            seed=42,
        )
        result = run_progressive_sampling(all_ids, self._noisy_evaluator, config)
        sizes = [r.sample_size for r in result.rounds]
        for i in range(1, len(sizes)):
            assert sizes[i] >= sizes[i - 1]

    def test_final_ci_width_matches_last_round(self):
        all_ids = [f"item_{i}" for i in range(100)]
        config = ProgressiveConfig(initial_size=10, max_rounds=3, seed=42)
        result = run_progressive_sampling(all_ids, self._noisy_evaluator, config)
        if result.rounds:
            assert result.final_ci_width == result.rounds[-1].ci_width

    def test_seed_determinism(self):
        all_ids = [f"item_{i}" for i in range(100)]
        config = ProgressiveConfig(initial_size=10, max_rounds=3, seed=42)
        r1 = run_progressive_sampling(all_ids, self._noisy_evaluator, config)
        r2 = run_progressive_sampling(all_ids, self._noisy_evaluator, config)
        assert r1.final_sample_ids == r2.final_sample_ids
        assert r1.total_rounds == r2.total_rounds

    def test_converged_flag_when_sufficient(self):
        all_ids = [f"item_{i}" for i in range(100)]
        result = run_progressive_sampling(all_ids, self._constant_evaluator)
        assert result.converged is True
        if result.rounds:
            assert result.rounds[-1].sufficient is True

    def test_not_converged_flag(self):
        all_ids = [f"item_{i}" for i in range(50)]
        config = ProgressiveConfig(
            initial_size=5,
            max_rounds=2,
            ci_width_threshold=0.001,
            max_size=10,
            seed=42,
        )
        result = run_progressive_sampling(all_ids, self._noisy_evaluator, config)
        # With noisy data and tight threshold, unlikely to converge in 2 rounds
        # with max 10 samples
        assert result.converged is False or result.final_ci_width <= 0.001


# ---- format_progressive_report ----


class TestFormatProgressiveReport:
    def test_basic_format(self):
        result = ProgressiveResult(
            final_sample_ids=["a", "b", "c"],
            rounds=[
                ProgressiveRound(1, 2, 0.15, False),
                ProgressiveRound(2, 4, 0.08, True),
            ],
            total_rounds=2,
            final_ci_width=0.08,
            converged=True,
        )
        report = format_progressive_report(result)
        assert "Progressive Sampling Report" in report
        assert "Total rounds: 2" in report
        assert "Final sample size: 3" in report
        assert "Converged: True" in report

    def test_empty_result(self):
        result = ProgressiveResult()
        report = format_progressive_report(result)
        assert "Progressive Sampling Report" in report
        assert "Total rounds: 0" in report

    def test_round_details_in_report(self):
        result = ProgressiveResult(
            final_sample_ids=["a"],
            rounds=[ProgressiveRound(1, 10, 0.12, False)],
            total_rounds=1,
            final_ci_width=0.12,
            converged=False,
        )
        report = format_progressive_report(result)
        assert "0.12" in report
        assert "no" in report.lower()

    def test_ids_in_report(self):
        result = ProgressiveResult(
            final_sample_ids=["item_42", "item_99"],
            rounds=[],
            total_rounds=0,
            final_ci_width=0.0,
            converged=True,
        )
        report = format_progressive_report(result)
        assert "item_42" in report
        assert "item_99" in report
