"""Tests for the simulation evaluation loop module."""

from __future__ import annotations

from evalyn_sdk.eval_loop import (
    LoopConfig,
    LoopResult,
    RoundResult,
    check_convergence,
    compute_improvement_rate,
    extract_round_failure_patterns,
    format_loop_report,
    run_eval_loop,
    should_stop,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_round(round_number: int, pass_rate: float) -> RoundResult:
    return RoundResult(
        round_number=round_number,
        items_generated=10,
        items_evaluated=10,
        pass_rate=pass_rate,
    )


def _simple_generate(n: int, patterns: list[str]) -> list[dict]:
    """Generate n items, each with an 'input' field."""
    return [{"input": f"item_{i}"} for i in range(n)]


def _all_pass_evaluate(items: list[dict]) -> list[dict]:
    """Evaluate all items as passing."""
    return [{**item, "score": 1.0} for item in items]


def _all_fail_evaluate(items: list[dict]) -> list[dict]:
    """Evaluate all items as failing."""
    return [{**item, "score": 0.0} for item in items]


def _half_pass_evaluate(items: list[dict]) -> list[dict]:
    """Evaluate half items as passing, half as failing."""
    results = []
    for i, item in enumerate(items):
        score = 1.0 if i % 2 == 0 else 0.0
        results.append({**item, "score": score})
    return results


# ---------------------------------------------------------------------------
# LoopConfig tests
# ---------------------------------------------------------------------------


class TestLoopConfig:
    def test_defaults(self):
        cfg = LoopConfig()
        assert cfg.max_rounds == 3
        assert cfg.items_per_round == 10
        assert cfg.convergence_threshold == 0.02
        assert cfg.target_pass_rate == 0.9

    def test_custom_values(self):
        cfg = LoopConfig(max_rounds=5, items_per_round=20)
        assert cfg.max_rounds == 5
        assert cfg.items_per_round == 20

    def test_as_dict(self):
        cfg = LoopConfig(max_rounds=2, target_pass_rate=0.8)
        d = cfg.as_dict()
        assert d["max_rounds"] == 2
        assert d["target_pass_rate"] == 0.8
        assert "items_per_round" in d

    def test_from_dict(self):
        d = {"max_rounds": 7, "items_per_round": 50, "convergence_threshold": 0.05, "target_pass_rate": 0.95}
        cfg = LoopConfig.from_dict(d)
        assert cfg.max_rounds == 7
        assert cfg.items_per_round == 50

    def test_from_dict_defaults(self):
        cfg = LoopConfig.from_dict({})
        assert cfg.max_rounds == 3
        assert cfg.convergence_threshold == 0.02

    def test_roundtrip(self):
        cfg = LoopConfig(max_rounds=4, items_per_round=15, convergence_threshold=0.01, target_pass_rate=0.85)
        restored = LoopConfig.from_dict(cfg.as_dict())
        assert restored.max_rounds == cfg.max_rounds
        assert restored.items_per_round == cfg.items_per_round
        assert restored.convergence_threshold == cfg.convergence_threshold
        assert restored.target_pass_rate == cfg.target_pass_rate


# ---------------------------------------------------------------------------
# RoundResult tests
# ---------------------------------------------------------------------------


class TestRoundResult:
    def test_basic(self):
        rr = RoundResult(round_number=1, items_generated=10, items_evaluated=10, pass_rate=0.8)
        assert rr.round_number == 1
        assert rr.improved_from_previous is False

    def test_with_failure_patterns(self):
        rr = RoundResult(
            round_number=2, items_generated=5, items_evaluated=5,
            pass_rate=0.6, failure_patterns=["pattern_a", "pattern_b"],
            improved_from_previous=True,
        )
        assert len(rr.failure_patterns) == 2
        assert rr.improved_from_previous is True

    def test_as_dict(self):
        rr = _make_round(1, 0.75)
        d = rr.as_dict()
        assert d["round_number"] == 1
        assert d["pass_rate"] == 0.75
        assert d["failure_patterns"] == []

    def test_from_dict(self):
        d = {
            "round_number": 3, "items_generated": 20,
            "items_evaluated": 18, "pass_rate": 0.55,
            "failure_patterns": ["x"],
        }
        rr = RoundResult.from_dict(d)
        assert rr.round_number == 3
        assert rr.items_evaluated == 18
        assert rr.failure_patterns == ["x"]

    def test_roundtrip(self):
        rr = RoundResult(
            round_number=2, items_generated=10, items_evaluated=8,
            pass_rate=0.625, failure_patterns=["a", "b"],
            improved_from_previous=True,
        )
        restored = RoundResult.from_dict(rr.as_dict())
        assert restored.round_number == rr.round_number
        assert restored.pass_rate == rr.pass_rate
        assert restored.failure_patterns == rr.failure_patterns
        assert restored.improved_from_previous == rr.improved_from_previous


# ---------------------------------------------------------------------------
# LoopResult tests
# ---------------------------------------------------------------------------


class TestLoopResult:
    def test_empty(self):
        lr = LoopResult()
        assert lr.rounds == []
        assert lr.final_pass_rate == 0.0
        assert lr.converged is False

    def test_as_dict(self):
        lr = LoopResult(
            rounds=[_make_round(1, 0.5), _make_round(2, 0.7)],
            final_pass_rate=0.7, converged=False,
            total_items=20, total_rounds=2,
        )
        d = lr.as_dict()
        assert len(d["rounds"]) == 2
        assert d["final_pass_rate"] == 0.7

    def test_from_dict(self):
        d = {
            "rounds": [
                {"round_number": 1, "items_generated": 10, "items_evaluated": 10, "pass_rate": 0.5},
            ],
            "final_pass_rate": 0.5, "converged": False,
            "total_items": 10, "total_rounds": 1,
        }
        lr = LoopResult.from_dict(d)
        assert lr.total_rounds == 1
        assert len(lr.rounds) == 1
        assert lr.rounds[0].pass_rate == 0.5

    def test_roundtrip(self):
        lr = LoopResult(
            rounds=[_make_round(1, 0.4), _make_round(2, 0.6)],
            final_pass_rate=0.6, converged=True,
            total_items=20, total_rounds=2,
        )
        restored = LoopResult.from_dict(lr.as_dict())
        assert restored.converged == lr.converged
        assert restored.total_items == lr.total_items
        assert len(restored.rounds) == 2


# ---------------------------------------------------------------------------
# check_convergence tests
# ---------------------------------------------------------------------------


class TestCheckConvergence:
    def test_empty_rounds(self):
        assert check_convergence([]) is False

    def test_single_round(self):
        assert check_convergence([_make_round(1, 0.5)]) is False

    def test_converged(self):
        rounds = [_make_round(1, 0.80), _make_round(2, 0.81)]
        assert check_convergence(rounds, threshold=0.02) is True

    def test_not_converged(self):
        rounds = [_make_round(1, 0.50), _make_round(2, 0.70)]
        assert check_convergence(rounds, threshold=0.02) is False

    def test_exact_threshold(self):
        rounds = [_make_round(1, 0.80), _make_round(2, 0.82)]
        assert check_convergence(rounds, threshold=0.02) is True

    def test_uses_last_two_only(self):
        rounds = [_make_round(1, 0.3), _make_round(2, 0.8), _make_round(3, 0.81)]
        assert check_convergence(rounds, threshold=0.02) is True


# ---------------------------------------------------------------------------
# extract_round_failure_patterns tests
# ---------------------------------------------------------------------------


class TestExtractFailurePatterns:
    def test_empty(self):
        assert extract_round_failure_patterns([]) == []

    def test_all_passing(self):
        results = [{"input": "hello", "score": 0.9}]
        assert extract_round_failure_patterns(results) == []

    def test_all_failing(self):
        results = [
            {"input": "bad item 1", "score": 0.1},
            {"input": "bad item 2", "score": 0.3},
        ]
        patterns = extract_round_failure_patterns(results)
        assert len(patterns) == 2
        assert "bad item 1" in patterns[0]

    def test_truncation(self):
        long_input = "x" * 100
        results = [{"input": long_input, "score": 0.0}]
        patterns = extract_round_failure_patterns(results)
        assert len(patterns[0]) == 50

    def test_custom_score_field(self):
        results = [{"input": "item", "quality": 0.1}]
        patterns = extract_round_failure_patterns(results, score_field="quality")
        assert len(patterns) == 1

    def test_custom_threshold(self):
        results = [{"input": "item", "score": 0.6}]
        # With default threshold=0.5, this passes. With 0.8, it fails.
        assert extract_round_failure_patterns(results, threshold=0.5) == []
        patterns = extract_round_failure_patterns(results, threshold=0.8)
        assert len(patterns) == 1

    def test_missing_input_field(self):
        results = [{"score": 0.1}]
        patterns = extract_round_failure_patterns(results)
        assert patterns == [""]

    def test_missing_score_defaults_to_zero(self):
        results = [{"input": "no score"}]
        patterns = extract_round_failure_patterns(results)
        assert len(patterns) == 1


# ---------------------------------------------------------------------------
# should_stop tests
# ---------------------------------------------------------------------------


class TestShouldStop:
    def test_empty_rounds(self):
        cfg = LoopConfig(max_rounds=3)
        assert should_stop([], cfg) is False

    def test_max_rounds_reached(self):
        cfg = LoopConfig(max_rounds=2)
        rounds = [_make_round(1, 0.5), _make_round(2, 0.6)]
        assert should_stop(rounds, cfg) is True

    def test_target_pass_rate_achieved(self):
        cfg = LoopConfig(max_rounds=5, target_pass_rate=0.9)
        rounds = [_make_round(1, 0.95)]
        assert should_stop(rounds, cfg) is True

    def test_converged(self):
        cfg = LoopConfig(max_rounds=10, convergence_threshold=0.01)
        rounds = [_make_round(1, 0.7), _make_round(2, 0.705)]
        assert should_stop(rounds, cfg) is True

    def test_not_stopped(self):
        cfg = LoopConfig(max_rounds=5, target_pass_rate=0.9, convergence_threshold=0.01)
        rounds = [_make_round(1, 0.5)]
        assert should_stop(rounds, cfg) is False


# ---------------------------------------------------------------------------
# run_eval_loop tests
# ---------------------------------------------------------------------------


class TestRunEvalLoop:
    def test_all_pass_stops_at_round_one(self):
        result = run_eval_loop(_simple_generate, _all_pass_evaluate)
        assert result.final_pass_rate == 1.0
        assert result.total_rounds == 1

    def test_all_fail_converges_at_two(self):
        """All-fail evaluator produces 0.0 pass rate every round; converges after 2."""
        cfg = LoopConfig(max_rounds=5)
        result = run_eval_loop(_simple_generate, _all_fail_evaluate, config=cfg)
        assert result.total_rounds == 2
        assert result.final_pass_rate == 0.0
        assert result.converged is True

    def test_default_config_used(self):
        result = run_eval_loop(_simple_generate, _all_pass_evaluate)
        assert result.total_rounds >= 1

    def test_custom_items_per_round(self):
        cfg = LoopConfig(items_per_round=5, max_rounds=1)
        result = run_eval_loop(_simple_generate, _all_pass_evaluate, config=cfg)
        assert result.rounds[0].items_generated == 5

    def test_failure_patterns_passed_to_generator(self):
        """Check that the generator receives failure patterns from previous rounds."""
        received_patterns: list[list[str]] = []

        def tracking_generate(n: int, patterns: list[str]) -> list[dict]:
            received_patterns.append(list(patterns))
            return [{"input": f"item_{i}"} for i in range(n)]

        cfg = LoopConfig(max_rounds=3, target_pass_rate=1.0)
        run_eval_loop(tracking_generate, _all_fail_evaluate, config=cfg)
        # First round should have empty patterns
        assert received_patterns[0] == []
        # Subsequent rounds should have patterns
        assert len(received_patterns[1]) > 0

    def test_improved_flag(self):
        call_count = [0]

        def improving_evaluate(items: list[dict]) -> list[dict]:
            call_count[0] += 1
            score = 0.3 * call_count[0]
            return [{**item, "score": min(score, 1.0)} for item in items]

        cfg = LoopConfig(max_rounds=3, target_pass_rate=1.0)
        result = run_eval_loop(_simple_generate, improving_evaluate, config=cfg)
        # Round 2 should improve from round 1
        if len(result.rounds) >= 2:
            assert result.rounds[1].improved_from_previous is True

    def test_convergence_stops_early(self):
        """Stable evaluator produces constant pass_rate; converges after 2 rounds."""
        call_count = [0]

        def stable_evaluate(items: list[dict]) -> list[dict]:
            call_count[0] += 1
            # Score 0.45 means below 0.5 threshold, so pass_rate stays ~0.0
            # But we want non-trivial pass rate: use alternating pass/fail
            # to get 0.5 pass rate every round.
            results = []
            for i, item in enumerate(items):
                score = 0.8 if i % 2 == 0 else 0.2
                results.append({**item, "score": score})
            return results

        cfg = LoopConfig(max_rounds=10, target_pass_rate=0.99, convergence_threshold=0.05)
        result = run_eval_loop(_simple_generate, stable_evaluate, config=cfg)
        # Should stop after 2 rounds due to convergence (pass_rate is 0.5 both times)
        assert result.total_rounds == 2
        assert result.converged is True

    def test_total_items(self):
        cfg = LoopConfig(max_rounds=1, items_per_round=7)
        result = run_eval_loop(_simple_generate, _half_pass_evaluate, config=cfg)
        assert result.total_items == 7

    def test_half_pass(self):
        cfg = LoopConfig(max_rounds=1)
        result = run_eval_loop(_simple_generate, _half_pass_evaluate, config=cfg)
        assert result.final_pass_rate == 0.5


# ---------------------------------------------------------------------------
# format_loop_report tests
# ---------------------------------------------------------------------------


class TestFormatLoopReport:
    def test_basic_report(self):
        lr = LoopResult(
            rounds=[_make_round(1, 0.5), _make_round(2, 0.7)],
            final_pass_rate=0.7, converged=False,
            total_items=20, total_rounds=2,
        )
        report = format_loop_report(lr)
        assert "Evaluation Loop Report" in report
        assert "50.00%" in report
        assert "70.00%" in report
        assert "Total rounds: 2" in report

    def test_empty_result(self):
        lr = LoopResult()
        report = format_loop_report(lr)
        assert "Total rounds: 0" in report

    def test_improved_marker(self):
        rr = RoundResult(
            round_number=2, items_generated=10, items_evaluated=10,
            pass_rate=0.8, improved_from_previous=True,
        )
        lr = LoopResult(
            rounds=[_make_round(1, 0.5), rr],
            final_pass_rate=0.8, converged=False,
            total_items=20, total_rounds=2,
        )
        report = format_loop_report(lr)
        assert "[improved]" in report

    def test_converged_flag(self):
        lr = LoopResult(converged=True, total_rounds=2, total_items=20, final_pass_rate=0.9)
        report = format_loop_report(lr)
        assert "Converged: True" in report


# ---------------------------------------------------------------------------
# compute_improvement_rate tests
# ---------------------------------------------------------------------------


class TestComputeImprovementRate:
    def test_empty(self):
        assert compute_improvement_rate([]) == 0.0

    def test_single_round(self):
        assert compute_improvement_rate([_make_round(1, 0.5)]) == 0.0

    def test_improving(self):
        rounds = [_make_round(1, 0.4), _make_round(2, 0.8)]
        rate = compute_improvement_rate(rounds)
        assert abs(rate - 0.2) < 1e-9  # (0.8 - 0.4) / 2

    def test_declining(self):
        rounds = [_make_round(1, 0.8), _make_round(2, 0.4)]
        rate = compute_improvement_rate(rounds)
        assert rate < 0

    def test_three_rounds(self):
        rounds = [_make_round(1, 0.3), _make_round(2, 0.5), _make_round(3, 0.6)]
        rate = compute_improvement_rate(rounds)
        assert abs(rate - 0.1) < 1e-9  # (0.6 - 0.3) / 3

    def test_no_change(self):
        rounds = [_make_round(1, 0.5), _make_round(2, 0.5)]
        rate = compute_improvement_rate(rounds)
        assert rate == 0.0
