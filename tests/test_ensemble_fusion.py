"""Tests for calibration ensemble fusion."""

from __future__ import annotations

import pytest

from evalyn_sdk.calibration.ensemble_fusion import (
    FusionConfig,
    OptimizerCandidate,
    TournamentResult,
    best_of_select,
    run_ensemble_fusion,
    tournament_select,
    weighted_merge_prompts,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_candidates(n: int = 5) -> list[OptimizerCandidate]:
    return [
        OptimizerCandidate(
            optimizer_name=f"opt_{i}",
            prompt=f"prompt {i}",
            alignment_score=0.1 * (i + 1),
            cost_tokens=100 * (i + 1),
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# OptimizerCandidate
# ---------------------------------------------------------------------------


class TestOptimizerCandidate:
    def test_as_dict(self):
        c = OptimizerCandidate("opro", "do X", 0.85, 500)
        d = c.as_dict()
        assert d["optimizer_name"] == "opro"
        assert d["prompt"] == "do X"
        assert d["alignment_score"] == 0.85
        assert d["cost_tokens"] == 500

    def test_default_cost_tokens(self):
        c = OptimizerCandidate("opro", "do X", 0.85)
        assert c.cost_tokens == 0


# ---------------------------------------------------------------------------
# TournamentResult
# ---------------------------------------------------------------------------


class TestTournamentResult:
    def test_as_dict_roundtrip(self):
        winner = OptimizerCandidate("a", "p1", 0.9, 10)
        tr = TournamentResult(
            winner=winner,
            participants=[winner],
            rounds=2,
            method="tournament",
        )
        d = tr.as_dict()
        restored = TournamentResult.from_dict(d)
        assert restored.winner.optimizer_name == "a"
        assert restored.winner.alignment_score == 0.9
        assert restored.rounds == 2
        assert restored.method == "tournament"
        assert len(restored.participants) == 1

    def test_from_dict_defaults(self):
        d = {
            "winner": {"optimizer_name": "x", "prompt": "p", "alignment_score": 0.5},
            "participants": [],
        }
        tr = TournamentResult.from_dict(d)
        assert tr.winner.cost_tokens == 0
        assert tr.rounds == 0
        assert tr.method == "tournament"

    def test_format_text_contains_key_info(self):
        winner = OptimizerCandidate("best", "prompt", 0.95, 0)
        other = OptimizerCandidate("other", "prompt2", 0.70, 0)
        tr = TournamentResult(winner=winner, participants=[winner, other], rounds=3)
        text = tr.format_text()
        assert "best" in text
        assert "0.9500" in text
        assert "Rounds: 3" in text
        assert "Participants: 2" in text

    def test_format_text_method_label(self):
        c = OptimizerCandidate("a", "p", 0.5)
        tr = TournamentResult(winner=c, participants=[c], rounds=0, method="best_of")
        assert "best_of" in tr.format_text()


# ---------------------------------------------------------------------------
# FusionConfig
# ---------------------------------------------------------------------------


class TestFusionConfig:
    def test_defaults(self):
        cfg = FusionConfig()
        assert cfg.method == "tournament"
        assert cfg.tournament_rounds == 3

    def test_as_dict(self):
        cfg = FusionConfig(method="best_of", tournament_rounds=5)
        d = cfg.as_dict()
        assert d["method"] == "best_of"
        assert d["tournament_rounds"] == 5

    def test_from_dict(self):
        cfg = FusionConfig.from_dict({"method": "weighted_merge", "tournament_rounds": 1})
        assert cfg.method == "weighted_merge"
        assert cfg.tournament_rounds == 1

    def test_from_dict_defaults(self):
        cfg = FusionConfig.from_dict({})
        assert cfg.method == "tournament"
        assert cfg.tournament_rounds == 3

    def test_roundtrip(self):
        cfg = FusionConfig(method="best_of", tournament_rounds=7)
        restored = FusionConfig.from_dict(cfg.as_dict())
        assert restored.method == cfg.method
        assert restored.tournament_rounds == cfg.tournament_rounds


# ---------------------------------------------------------------------------
# tournament_select
# ---------------------------------------------------------------------------


class TestTournamentSelect:
    def test_picks_best_on_average(self):
        candidates = _make_candidates(5)
        best = max(candidates, key=lambda c: c.alignment_score)
        # With enough rounds the best should usually win
        wins = 0
        for seed in range(50):
            result = tournament_select(candidates, rounds=5, seed=seed)
            if result.winner.optimizer_name == best.optimizer_name:
                wins += 1
        assert wins > 25, f"Best candidate should win majority of tournaments, got {wins}/50"

    def test_deterministic_with_seed(self):
        candidates = _make_candidates(6)
        r1 = tournament_select(candidates, rounds=3, seed=42)
        r2 = tournament_select(candidates, rounds=3, seed=42)
        assert r1.winner.optimizer_name == r2.winner.optimizer_name

    def test_single_candidate(self):
        c = OptimizerCandidate("solo", "p", 0.5)
        result = tournament_select([c])
        assert result.winner.optimizer_name == "solo"
        assert result.rounds == 0
        assert len(result.participants) == 1

    def test_two_candidates(self):
        a = OptimizerCandidate("a", "p1", 0.9)
        b = OptimizerCandidate("b", "p2", 0.1)
        result = tournament_select([a, b], rounds=1, seed=0)
        assert result.winner.alignment_score == 0.9

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="zero candidates"):
            tournament_select([])

    def test_method_is_tournament(self):
        result = tournament_select(_make_candidates(3), rounds=1, seed=0)
        assert result.method == "tournament"

    def test_participants_preserved(self):
        candidates = _make_candidates(4)
        result = tournament_select(candidates, rounds=2, seed=0)
        assert len(result.participants) == 4


# ---------------------------------------------------------------------------
# best_of_select
# ---------------------------------------------------------------------------


class TestBestOfSelect:
    def test_picks_highest(self):
        candidates = _make_candidates(5)
        result = best_of_select(candidates)
        assert result.winner.optimizer_name == "opt_4"
        assert result.winner.alignment_score == pytest.approx(0.5)

    def test_method_is_best_of(self):
        result = best_of_select(_make_candidates(3))
        assert result.method == "best_of"

    def test_rounds_is_zero(self):
        result = best_of_select(_make_candidates(2))
        assert result.rounds == 0

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="zero candidates"):
            best_of_select([])

    def test_single_candidate(self):
        c = OptimizerCandidate("only", "p", 0.42)
        result = best_of_select([c])
        assert result.winner.optimizer_name == "only"


# ---------------------------------------------------------------------------
# weighted_merge_prompts
# ---------------------------------------------------------------------------


class TestWeightedMergePrompts:
    def test_returns_best_prompt(self):
        candidates = _make_candidates(4)
        merged = weighted_merge_prompts(candidates)
        assert merged == "prompt 3"

    def test_empty_returns_empty(self):
        assert weighted_merge_prompts([]) == ""

    def test_single_candidate(self):
        c = OptimizerCandidate("x", "my prompt", 0.5)
        assert weighted_merge_prompts([c]) == "my prompt"


# ---------------------------------------------------------------------------
# run_ensemble_fusion
# ---------------------------------------------------------------------------


class TestRunEnsembleFusion:
    def test_routes_tournament(self):
        result = run_ensemble_fusion(_make_candidates(4))
        assert result.method == "tournament"

    def test_routes_best_of(self):
        cfg = FusionConfig(method="best_of")
        result = run_ensemble_fusion(_make_candidates(4), config=cfg)
        assert result.method == "best_of"

    def test_routes_weighted_merge(self):
        cfg = FusionConfig(method="weighted_merge")
        result = run_ensemble_fusion(_make_candidates(4), config=cfg)
        assert result.method == "weighted_merge"

    def test_weighted_merge_winner_has_best_prompt(self):
        candidates = _make_candidates(3)
        cfg = FusionConfig(method="weighted_merge")
        result = run_ensemble_fusion(candidates, config=cfg)
        assert result.winner.prompt == "prompt 2"

    def test_unknown_method_raises(self):
        cfg = FusionConfig(method="bogus")
        with pytest.raises(ValueError, match="Unknown fusion method"):
            run_ensemble_fusion(_make_candidates(2), config=cfg)

    def test_default_config_applied(self):
        # Should not raise when config is None
        result = run_ensemble_fusion(_make_candidates(3))
        assert result.winner is not None
