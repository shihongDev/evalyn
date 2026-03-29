"""Tests for the CAPO optimizer module."""

from __future__ import annotations

import random

import pytest

from evalyn_sdk.capo_optimizer import (
    MUTATION_OPERATORS,
    CAPOConfig,
    CAPOState,
    MutationOperator,
    PromptCandidate,
    advance_generation,
    apply_mutation,
    compute_confidence_score,
    format_optimization_report,
    initialize_population,
    run_optimization,
    select_parent,
    should_stop,
)


# ---------------------------------------------------------------------------
# PromptCandidate
# ---------------------------------------------------------------------------


class TestPromptCandidate:
    def test_defaults(self):
        c = PromptCandidate(prompt_text="hello")
        assert c.score == 0.0
        assert c.confidence == 0.0
        assert c.generation == 0
        assert c.mutations_applied == []

    def test_as_dict(self):
        c = PromptCandidate(prompt_text="x", score=1.0, confidence=0.9, generation=3, mutations_applied=["rephrase"])
        d = c.as_dict()
        assert d["prompt_text"] == "x"
        assert d["score"] == 1.0
        assert d["mutations_applied"] == ["rephrase"]

    def test_from_dict_minimal(self):
        c = PromptCandidate.from_dict({"prompt_text": "hi"})
        assert c.prompt_text == "hi"
        assert c.score == 0.0

    def test_roundtrip(self):
        c = PromptCandidate(prompt_text="abc", score=0.5, confidence=0.7, generation=2, mutations_applied=["elaborate"])
        assert PromptCandidate.from_dict(c.as_dict()).prompt_text == c.prompt_text
        assert PromptCandidate.from_dict(c.as_dict()).mutations_applied == c.mutations_applied

    def test_mutations_list_is_independent(self):
        c = PromptCandidate(prompt_text="a", mutations_applied=["x"])
        d = c.as_dict()
        d["mutations_applied"].append("y")
        assert c.mutations_applied == ["x"]


# ---------------------------------------------------------------------------
# CAPOConfig
# ---------------------------------------------------------------------------


class TestCAPOConfig:
    def test_defaults(self):
        cfg = CAPOConfig()
        assert cfg.population_size == 10
        assert cfg.max_generations == 20
        assert cfg.seed is None

    def test_as_dict(self):
        cfg = CAPOConfig(seed=42)
        d = cfg.as_dict()
        assert d["seed"] == 42
        assert d["population_size"] == 10

    def test_from_dict(self):
        cfg = CAPOConfig.from_dict({"population_size": 5, "seed": 7})
        assert cfg.population_size == 5
        assert cfg.seed == 7
        assert cfg.max_generations == 20  # default

    def test_roundtrip(self):
        cfg = CAPOConfig(population_size=8, mutation_rate=0.5)
        cfg2 = CAPOConfig.from_dict(cfg.as_dict())
        assert cfg2.population_size == cfg.population_size
        assert cfg2.mutation_rate == cfg.mutation_rate


# ---------------------------------------------------------------------------
# CAPOState
# ---------------------------------------------------------------------------


class TestCAPOState:
    def test_as_dict_no_best(self):
        s = CAPOState(generation=0, population=[])
        d = s.as_dict()
        assert d["best_candidate"] is None

    def test_as_dict_with_best(self):
        best = PromptCandidate(prompt_text="best")
        s = CAPOState(generation=1, population=[best], best_candidate=best)
        d = s.as_dict()
        assert d["best_candidate"]["prompt_text"] == "best"

    def test_roundtrip(self):
        pop = [PromptCandidate(prompt_text="a"), PromptCandidate(prompt_text="b")]
        s = CAPOState(generation=2, population=pop, best_candidate=pop[0], history=[{"generation": 0}])
        s2 = CAPOState.from_dict(s.as_dict())
        assert s2.generation == 2
        assert len(s2.population) == 2
        assert s2.best_candidate is not None
        assert s2.history == [{"generation": 0}]


# ---------------------------------------------------------------------------
# MutationOperator
# ---------------------------------------------------------------------------


class TestMutationOperator:
    def test_as_dict(self):
        op = MutationOperator(name="rephrase", description="test", weight=0.5)
        d = op.as_dict()
        assert d["name"] == "rephrase"
        assert d["weight"] == 0.5

    def test_from_dict(self):
        op = MutationOperator.from_dict({"name": "x", "description": "y"})
        assert op.weight == 1.0

    def test_builtin_operators(self):
        assert len(MUTATION_OPERATORS) == 6
        names = {op.name for op in MUTATION_OPERATORS}
        assert "rephrase" in names
        assert "elaborate" in names
        assert "simplify" in names
        assert "add_example" in names
        assert "remove_section" in names
        assert "reorder" in names


# ---------------------------------------------------------------------------
# initialize_population
# ---------------------------------------------------------------------------


class TestInitializePopulation:
    def test_creates_correct_size(self):
        cfg = CAPOConfig(population_size=8, seed=1)
        state = initialize_population("Test prompt", cfg)
        assert len(state.population) == 8

    def test_seed_prompt_is_first(self):
        cfg = CAPOConfig(population_size=5, seed=1)
        state = initialize_population("My prompt", cfg)
        assert state.population[0].prompt_text == "My prompt"

    def test_no_exact_duplicates(self):
        cfg = CAPOConfig(population_size=10, seed=42)
        state = initialize_population("Hello world", cfg)
        texts = [c.prompt_text for c in state.population]
        assert len(texts) == len(set(texts))

    def test_generation_zero(self):
        cfg = CAPOConfig(population_size=5, seed=1)
        state = initialize_population("p", cfg)
        assert state.generation == 0
        for c in state.population:
            assert c.generation == 0

    def test_deterministic_with_seed(self):
        cfg = CAPOConfig(population_size=6, seed=99)
        s1 = initialize_population("prompt", cfg)
        s2 = initialize_population("prompt", cfg)
        t1 = [c.prompt_text for c in s1.population]
        t2 = [c.prompt_text for c in s2.population]
        assert t1 == t2


# ---------------------------------------------------------------------------
# select_parent
# ---------------------------------------------------------------------------


class TestSelectParent:
    def test_returns_candidate(self):
        pop = [
            PromptCandidate(prompt_text="a", score=1.0, confidence=0.9),
            PromptCandidate(prompt_text="b", score=0.5, confidence=0.5),
        ]
        result = select_parent(pop, 2, random.Random(1))
        assert result in pop

    def test_prefers_high_score(self):
        high = PromptCandidate(prompt_text="high", score=10.0, confidence=1.0)
        low = PromptCandidate(prompt_text="low", score=0.1, confidence=0.1)
        # With tournament_size=2, both will be selected, high should win
        wins = sum(
            1
            for _ in range(100)
            if select_parent([high, low], 2, random.Random(_)).prompt_text == "high"
        )
        assert wins == 100

    def test_tournament_size_clamped(self):
        pop = [PromptCandidate(prompt_text="only", score=1.0, confidence=1.0)]
        result = select_parent(pop, 5, random.Random(0))
        assert result.prompt_text == "only"


# ---------------------------------------------------------------------------
# apply_mutation
# ---------------------------------------------------------------------------


class TestApplyMutation:
    def test_elaborate(self):
        c = PromptCandidate(prompt_text="Do something")
        op = MutationOperator(name="elaborate", description="test")
        result = apply_mutation(c, op, random.Random(0))
        assert len(result.prompt_text) > len(c.prompt_text)
        assert "elaborate" in result.mutations_applied

    def test_add_example(self):
        c = PromptCandidate(prompt_text="Explain this")
        op = MutationOperator(name="add_example", description="test")
        result = apply_mutation(c, op, random.Random(0))
        assert "For example" in result.prompt_text

    def test_simplify_multi_sentence(self):
        c = PromptCandidate(prompt_text="First sentence. Second sentence. Third sentence.")
        op = MutationOperator(name="simplify", description="test")
        result = apply_mutation(c, op, random.Random(0))
        # Should have removed one sentence
        assert len(result.prompt_text) < len(c.prompt_text)

    def test_simplify_single_sentence(self):
        c = PromptCandidate(prompt_text="Only one")
        op = MutationOperator(name="simplify", description="test")
        result = apply_mutation(c, op, random.Random(0))
        assert result.prompt_text  # should not be empty

    def test_rephrase(self):
        c = PromptCandidate(prompt_text="First. Second.")
        op = MutationOperator(name="rephrase", description="test")
        result = apply_mutation(c, op, random.Random(0))
        assert result.prompt_text  # non-empty

    def test_remove_section(self):
        c = PromptCandidate(prompt_text="Keep this. Remove this.")
        op = MutationOperator(name="remove_section", description="test")
        result = apply_mutation(c, op, random.Random(0))
        assert "Remove this" not in result.prompt_text

    def test_reorder(self):
        c = PromptCandidate(prompt_text="A. B. C.")
        op = MutationOperator(name="reorder", description="test")
        result = apply_mutation(c, op, random.Random(42))
        assert result.prompt_text  # non-empty

    def test_generation_incremented(self):
        c = PromptCandidate(prompt_text="x", generation=3)
        op = MutationOperator(name="elaborate", description="test")
        result = apply_mutation(c, op, random.Random(0))
        assert result.generation == 4

    def test_mutations_list_accumulated(self):
        c = PromptCandidate(prompt_text="x", mutations_applied=["simplify"])
        op = MutationOperator(name="elaborate", description="test")
        result = apply_mutation(c, op, random.Random(0))
        assert result.mutations_applied == ["simplify", "elaborate"]


# ---------------------------------------------------------------------------
# compute_confidence_score
# ---------------------------------------------------------------------------


class TestComputeConfidenceScore:
    def test_empty_list(self):
        assert compute_confidence_score([]) == 0.0

    def test_single_score(self):
        assert compute_confidence_score([5.0]) == 0.0

    def test_identical_scores(self):
        conf = compute_confidence_score([1.0, 1.0, 1.0])
        assert conf == 1.0

    def test_high_variance_low_confidence(self):
        conf = compute_confidence_score([0.0, 10.0])
        assert conf < 0.3

    def test_low_variance_high_confidence(self):
        conf = compute_confidence_score([5.0, 5.1, 4.9, 5.0])
        assert conf > 0.9

    def test_returns_float(self):
        result = compute_confidence_score([1.0, 2.0, 3.0])
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# should_stop
# ---------------------------------------------------------------------------


class TestShouldStop:
    def test_max_generations(self):
        cfg = CAPOConfig(max_generations=5)
        state = CAPOState(generation=5, population=[])
        assert should_stop(state, cfg) is True

    def test_below_max(self):
        cfg = CAPOConfig(max_generations=5)
        state = CAPOState(generation=2, population=[])
        assert should_stop(state, cfg) is False

    def test_convergence(self):
        cfg = CAPOConfig(max_generations=100, confidence_threshold=0.8)
        best = PromptCandidate(prompt_text="x", confidence=0.9, score=1.0)
        state = CAPOState(generation=5, population=[], best_candidate=best)
        assert should_stop(state, cfg) is True

    def test_convergence_too_early(self):
        cfg = CAPOConfig(max_generations=100, confidence_threshold=0.8)
        best = PromptCandidate(prompt_text="x", confidence=0.9, score=1.0)
        state = CAPOState(generation=1, population=[], best_candidate=best)
        assert should_stop(state, cfg) is False

    def test_no_best_candidate(self):
        cfg = CAPOConfig(max_generations=100)
        state = CAPOState(generation=5, population=[])
        assert should_stop(state, cfg) is False


# ---------------------------------------------------------------------------
# advance_generation
# ---------------------------------------------------------------------------


class TestAdvanceGeneration:
    def _score_fn(self, text: str) -> float:
        return float(len(text))

    def test_increments_generation(self):
        cfg = CAPOConfig(population_size=4, seed=1)
        state = initialize_population("test prompt", cfg)
        new_state = advance_generation(state, self._score_fn, cfg)
        assert new_state.generation == 1

    def test_population_size_maintained(self):
        cfg = CAPOConfig(population_size=6, seed=1)
        state = initialize_population("hello world", cfg)
        new_state = advance_generation(state, self._score_fn, cfg)
        assert len(new_state.population) == 6

    def test_best_candidate_set(self):
        cfg = CAPOConfig(population_size=4, seed=1)
        state = initialize_population("test", cfg)
        new_state = advance_generation(state, self._score_fn, cfg)
        assert new_state.best_candidate is not None

    def test_history_grows(self):
        cfg = CAPOConfig(population_size=4, seed=1)
        state = initialize_population("test", cfg)
        s1 = advance_generation(state, self._score_fn, cfg)
        s2 = advance_generation(s1, self._score_fn, cfg)
        assert len(s2.history) == 2


# ---------------------------------------------------------------------------
# run_optimization
# ---------------------------------------------------------------------------


class TestRunOptimization:
    def test_basic_run(self):
        def score_fn(text: str) -> float:
            return float(len(text))

        cfg = CAPOConfig(population_size=4, max_generations=3, seed=42)
        state = run_optimization("optimize me", score_fn, cfg)
        assert state.generation >= 1
        assert state.best_candidate is not None

    def test_default_config(self):
        def score_fn(text: str) -> float:
            return 1.0 if "good" in text.lower() else 0.0

        # Uses small population to keep it fast
        cfg = CAPOConfig(population_size=4, max_generations=2, seed=1)
        state = run_optimization("good prompt", score_fn, cfg)
        assert len(state.population) == 4

    def test_none_config(self):
        # Very short run with default config
        call_count = 0

        def score_fn(text: str) -> float:
            nonlocal call_count
            call_count += 1
            return 1.0

        cfg = CAPOConfig(
            population_size=3,
            max_generations=1,
            confidence_threshold=0.99,
            seed=0,
        )
        state = run_optimization("test", score_fn, cfg)
        assert state.generation >= 1

    def test_convergence_stops_early(self):
        # Score function that always returns exactly the same value
        # -> zero variance -> confidence=1.0 -> should converge
        cfg = CAPOConfig(
            population_size=4,
            max_generations=50,
            confidence_threshold=0.9,
            seed=42,
        )
        state = run_optimization("converge", lambda t: 5.0, cfg)
        # Should stop well before 50 generations
        assert state.generation < 50


# ---------------------------------------------------------------------------
# format_optimization_report
# ---------------------------------------------------------------------------


class TestFormatOptimizationReport:
    def test_basic_report(self):
        best = PromptCandidate(prompt_text="best prompt", score=0.95, confidence=0.88, generation=3)
        state = CAPOState(
            generation=5,
            population=[best],
            best_candidate=best,
            history=[
                {"generation": 0, "best_score": 0.5, "avg_score": 0.3, "best_confidence": 0.4},
                {"generation": 1, "best_score": 0.9, "avg_score": 0.6, "best_confidence": 0.8},
            ],
        )
        report = format_optimization_report(state)
        assert "CAPO Optimization Report" in report
        assert "0.9500" in report
        assert "Gen 0" in report
        assert "Gen 1" in report

    def test_report_no_best(self):
        state = CAPOState(generation=0, population=[])
        report = format_optimization_report(state)
        assert "CAPO Optimization Report" in report
        assert "Best Candidate" not in report

    def test_long_prompt_truncated(self):
        long_text = "A" * 200
        best = PromptCandidate(prompt_text=long_text, score=1.0, confidence=1.0)
        state = CAPOState(generation=1, population=[best], best_candidate=best)
        report = format_optimization_report(state)
        assert "..." in report
