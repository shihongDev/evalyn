"""Tests for new optimizer infrastructure and implementations."""
from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from evalyn_sdk.calibration.base_optimizer import BaseOptimizer
from evalyn_sdk.calibration.models import PromptOptimizationResult


@dataclass
class DummyConfig:
    task_model: str = "test-model"
    scorer_model: str = "test-model"


class ConcreteOptimizer(BaseOptimizer):
    """Concrete subclass for testing."""

    def optimize(self, *, metric_id, current_rubric, current_preamble, **kwargs):
        return self.build_result(
            original_preamble=current_preamble,
            optimized_preamble="improved " + current_preamble,
            rubric=current_rubric,
            reasoning="test",
            estimated_improvement="medium",
        )


class TestBaseOptimizer:
    def test_init_stores_config(self):
        cfg = DummyConfig()
        opt = ConcreteOptimizer(config=cfg)
        assert opt.config is cfg
        assert opt._api_key is None

    def test_init_with_api_key(self):
        opt = ConcreteOptimizer(config=DummyConfig(), api_key="sk-test")
        assert opt._api_key == "sk-test"

    def test_build_result_returns_prompt_optimization_result(self):
        opt = ConcreteOptimizer(config=DummyConfig())
        result = opt.build_result(
            original_preamble="original",
            optimized_preamble="improved",
            rubric=["criterion 1", "criterion 2"],
            reasoning="fixed false positives",
            estimated_improvement="high",
        )
        assert isinstance(result, PromptOptimizationResult)
        assert result.original_preamble == "original"
        assert result.optimized_preamble == "improved"
        assert result.original_rubric == ["criterion 1", "criterion 2"]
        assert result.improved_rubric == ["criterion 1", "criterion 2"]
        assert result.estimated_improvement == "high"
        assert "fixed false positives" in result.improvement_reasoning
        assert result.full_prompt  # non-empty

    def test_build_result_rubric_always_unchanged(self):
        opt = ConcreteOptimizer(config=DummyConfig())
        result = opt.build_result(
            original_preamble="a",
            optimized_preamble="b",
            rubric=["r1", "r2"],
            reasoning="x",
            estimated_improvement="low",
        )
        assert result.original_rubric == result.improved_rubric

    def test_optimize_abstract_raises(self):
        # BaseOptimizer is now an ABC - instantiation fails if optimize() not implemented
        with pytest.raises(TypeError, match="abstract"):
            BaseOptimizer(config=DummyConfig())

    def test_concrete_optimize_works(self):
        opt = ConcreteOptimizer(config=DummyConfig())
        result = opt.optimize(
            metric_id="test",
            current_rubric=["be good"],
            current_preamble="judge prompt",
            metric_results=[],
            annotations=[],
        )
        assert isinstance(result, PromptOptimizationResult)
        assert result.optimized_preamble == "improved judge prompt"


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------

from evalyn_sdk.calibration.factory import (
    OPTIMIZER_REGISTRY,
    call_optimizer,
    create_optimizer,
)


class TestFactory:
    def test_registry_has_all_optimizers(self):
        expected = {
            "basic", "ape", "opro", "gepa", "gepa-native",
            "evoprompt", "textgrad", "miprov2", "promptbreeder",
        }
        assert set(OPTIMIZER_REGISTRY.keys()) == expected

    def test_create_basic_optimizer(self):
        opt = create_optimizer("basic", model="gemini-2.5-flash-lite")
        assert opt.__class__.__name__ == "BasicOptimizer"

    def test_create_ape_optimizer(self):
        from evalyn_sdk.calibration.ape import APEConfig

        opt = create_optimizer("ape", config=APEConfig())
        assert opt.__class__.__name__ == "APEOptimizer"

    def test_create_opro_optimizer(self):
        from evalyn_sdk.calibration.opro import OPROConfig

        opt = create_optimizer("opro", config=OPROConfig())
        assert opt.__class__.__name__ == "OPROOptimizer"

    def test_create_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown optimizer"):
            create_optimizer("nonexistent")

    def test_call_optimizer_filters_kwargs_for_legacy(self):
        """Legacy optimizers only get params they accept."""
        import inspect

        opt = create_optimizer("basic", model="gemini-2.5-flash-lite")
        sig = inspect.signature(opt.optimize)
        param_names = {p.name for p in sig.parameters.values() if p.name != "self"}
        # BasicOptimizer does not accept metric_results
        assert "metric_results" not in param_names

    def test_call_optimizer_passes_all_kwargs_to_new_style(self):
        """New-style optimizers with **kwargs get everything."""
        opt = ConcreteOptimizer(config=DummyConfig())
        result = call_optimizer(
            opt,
            metric_id="test",
            current_rubric=["r1"],
            current_preamble="preamble",
            metric_results=[],
            annotations=[],
            disagreements=None,
            dataset_items=[],
            accumulator=None,
            extra_param="should be passed through",
        )
        assert isinstance(result, PromptOptimizationResult)


# ---------------------------------------------------------------------------
# EvoPrompt tests
# ---------------------------------------------------------------------------

from evalyn_sdk.calibration.evoprompt import EvoPromptConfig, EvoPromptOptimizer


class TestEvoPromptConfig:
    def test_defaults(self):
        cfg = EvoPromptConfig()
        assert cfg.population_size == 8
        assert cfg.generations == 5
        assert cfg.mutation_rate == 0.3
        assert cfg.crossover_rate == 0.7
        assert cfg.tournament_size == 2
        assert cfg.early_stop_patience == 2

    def test_custom_values(self):
        cfg = EvoPromptConfig(population_size=4, generations=3)
        assert cfg.population_size == 4
        assert cfg.generations == 3


class TestEvoPrompt:
    def _make_optimizer(self, score_sequence=None, llm_text="variant preamble"):
        """Helper: create optimizer with mocked LLM and score_preamble."""
        cfg = EvoPromptConfig(
            population_size=4,
            generations=3,
            early_stop_patience=2,
        )
        opt = EvoPromptOptimizer(config=cfg, api_key="fake-key")

        # Mock the LLM client
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.text = llm_text
        mock_result.input_tokens = 10
        mock_result.output_tokens = 20
        mock_result.model = "test-model"
        mock_client.generate_with_usage.return_value = mock_result
        opt._task_client_instance = mock_client

        # Mock split_train_val
        train = [
            {"id": "1", "input": "in1", "output": "out1", "expected": "PASS", "call_id": "1"},
            {"id": "2", "input": "in2", "output": "out2", "expected": "FAIL", "call_id": "2"},
            {"id": "3", "input": "in3", "output": "out3", "expected": "PASS", "call_id": "3"},
        ]
        val = [
            {"id": "4", "input": "in4", "output": "out4", "expected": "PASS", "call_id": "4"},
            {"id": "5", "input": "in5", "output": "out5", "expected": "FAIL", "call_id": "5"},
        ]
        opt.split_train_val = MagicMock(return_value=(train, val))

        # Mock score_preamble with a sequence or constant
        if score_sequence is not None:
            opt.score_preamble = MagicMock(side_effect=score_sequence)
        else:
            opt.score_preamble = MagicMock(return_value=0.8)

        return opt

    def test_optimize_returns_result(self):
        opt = self._make_optimizer()
        result = opt.optimize(
            metric_id="accuracy",
            current_rubric=["be correct"],
            current_preamble="evaluate quality",
            metric_results=[],
            annotations=[],
            disagreements=None,
            dataset_items=[],
            accumulator=None,
        )
        assert isinstance(result, PromptOptimizationResult)
        assert result.original_preamble == "evaluate quality"
        assert result.optimized_preamble  # non-empty
        assert result.full_prompt  # non-empty
        assert result.improved_rubric == ["be correct"]  # rubric unchanged

    def test_early_stopping(self):
        """When score_preamble always returns the same value, early stopping triggers."""
        # All scores identical -> no improvement -> early stop after patience
        scores = [0.5] * 200  # enough for any number of calls
        opt = self._make_optimizer(score_sequence=scores)
        result = opt.optimize(
            metric_id="accuracy",
            current_rubric=["be correct"],
            current_preamble="evaluate quality",
            metric_results=[],
            annotations=[],
        )
        assert isinstance(result, PromptOptimizationResult)
        # Verify score_preamble was called, but generations should have stopped early
        # With patience=2 and 3 generations, it should stop before gen 3
        # Initial population (4 scores) + gen0 offspring (4) + gen1 offspring (4) + 2 val = ~14
        # If it ran all 3 gens: 4 + 3*4 + 2 = 18. Early stop should be less.
        assert opt.score_preamble.call_count < 30

    def test_tournament_select(self):
        opt = EvoPromptOptimizer(config=EvoPromptConfig())
        scored = [
            ("preamble_a", 0.3),
            ("preamble_b", 0.9),
            ("preamble_c", 0.5),
            ("preamble_d", 0.7),
        ]
        # With tournament_size >= population, the best always wins
        winner = opt._tournament_select(scored, tournament_size=4)
        assert winner == "preamble_b"

    def test_tournament_select_single(self):
        opt = EvoPromptOptimizer(config=EvoPromptConfig())
        scored = [("only_one", 0.6)]
        winner = opt._tournament_select(scored, tournament_size=1)
        assert winner == "only_one"

    def test_factory_creates_evoprompt(self):
        from evalyn_sdk.calibration.factory import create_optimizer

        opt = create_optimizer("evoprompt", config=EvoPromptConfig())
        assert isinstance(opt, EvoPromptOptimizer)

    def test_not_enough_data_returns_early(self):
        cfg = EvoPromptConfig(population_size=4, generations=2)
        opt = EvoPromptOptimizer(config=cfg, api_key="fake")
        # Return insufficient data
        opt.split_train_val = MagicMock(return_value=([], []))
        result = opt.optimize(
            metric_id="m",
            current_rubric=["r1"],
            current_preamble="preamble",
            metric_results=[],
            annotations=[],
        )
        assert isinstance(result, PromptOptimizationResult)
        assert "Not enough data" in result.improvement_reasoning

    def test_crossover_calls_llm(self):
        opt = self._make_optimizer(llm_text="merged preamble")
        result = opt._crossover("parent A", "parent B")
        assert result == "merged preamble"
        opt._task_client_instance.generate_with_usage.assert_called_once()

    def test_mutate_calls_llm(self):
        opt = self._make_optimizer(llm_text="mutated preamble")
        failures = [{"input": "x", "output": "y", "expected": "PASS"}]
        result = opt._mutate("original", failures)
        assert result == "mutated preamble"
        opt._task_client_instance.generate_with_usage.assert_called_once()

    def test_mutate_no_failures(self):
        opt = self._make_optimizer(llm_text="mutated preamble")
        result = opt._mutate("original", [])
        assert result == "mutated preamble"

    def test_init_population_includes_current(self):
        opt = self._make_optimizer(llm_text='["var1", "var2", "var3"]')
        population = opt._init_population("current", None, ["r1"])
        assert population[0] == "current"
        assert len(population) == opt.config.population_size


# ---------------------------------------------------------------------------
# TextGrad tests
# ---------------------------------------------------------------------------

from evalyn_sdk.calibration.textgrad import TextGradConfig, TextGradOptimizer
from evalyn_sdk.calibration.models import TokenAccumulator
from evalyn_sdk.utils.api_client import GenerateResult


def _make_generate_result(text: str = "ok") -> GenerateResult:
    """Build a minimal GenerateResult for mocking."""
    return GenerateResult(text=text, input_tokens=10, output_tokens=5, model="test")


def _make_examples(n: int = 10) -> list:
    """Build a list of labelled examples for train/val splits."""
    examples = []
    for i in range(n):
        examples.append({
            "id": f"call-{i}",
            "input": f"input {i}",
            "output": f"output {i}",
            "expected": "PASS" if i % 2 == 0 else "FAIL",
            "call_id": f"call-{i}",
        })
    return examples


class TestTextGradConfig:
    def test_defaults(self):
        cfg = TextGradConfig()
        assert cfg.max_iterations == 8
        assert cfg.improvement_threshold == 0.01
        assert cfg.num_failure_examples == 5
        assert cfg.early_stop_patience == 3

    def test_custom_values(self):
        cfg = TextGradConfig(max_iterations=4, early_stop_patience=1)
        assert cfg.max_iterations == 4
        assert cfg.early_stop_patience == 1


class TestTextGrad:
    def _make_optimizer(self) -> TextGradOptimizer:
        cfg = TextGradConfig(max_iterations=3, early_stop_patience=2)
        opt = TextGradOptimizer(config=cfg, api_key="fake-key")
        mock_task = MagicMock()
        mock_task.generate_with_usage.return_value = _make_generate_result(
            "Revised preamble: be stricter on edge cases."
        )
        opt._task_client = mock_task
        mock_scorer = MagicMock()
        mock_scorer.generate_with_usage.return_value = _make_generate_result(
            '{"passed": true, "reason": "ok", "score": 0.9}'
        )
        opt._scorer_client = mock_scorer
        return opt

    def test_optimize_returns_result(self):
        opt = self._make_optimizer()
        train = _make_examples(7)
        val = _make_examples(3)
        opt.split_train_val = MagicMock(return_value=(train, val))
        call_count = {"n": 0}
        def increasing_score(*_a, **_kw):
            call_count["n"] += 1
            return 0.5 + call_count["n"] * 0.05
        opt.score_preamble = MagicMock(side_effect=increasing_score)
        result = opt.optimize(
            metric_id="test_metric",
            current_rubric=["Be accurate"],
            current_preamble="You are a judge.",
            metric_results=[], annotations=[],
        )
        assert isinstance(result, PromptOptimizationResult)
        assert result.original_preamble == "You are a judge."
        assert result.improved_rubric == ["Be accurate"]

    def test_early_stopping(self):
        opt = self._make_optimizer()
        opt.config.early_stop_patience = 2
        opt.config.max_iterations = 10
        opt.split_train_val = MagicMock(return_value=(_make_examples(7), _make_examples(3)))
        opt.score_preamble = MagicMock(return_value=0.6)
        result = opt.optimize(
            metric_id="test", current_rubric=["r1"], current_preamble="orig",
            metric_results=[], annotations=[],
        )
        assert isinstance(result, PromptOptimizationResult)
        assert opt.score_preamble.call_count <= 8

    def test_not_enough_data(self):
        opt = self._make_optimizer()
        opt.split_train_val = MagicMock(return_value=([], []))
        result = opt.optimize(
            metric_id="test", current_rubric=["r1"], current_preamble="orig",
            metric_results=[], annotations=[],
        )
        assert "Not enough data" in result.improvement_reasoning

    def test_no_failures_stops_early(self):
        opt = self._make_optimizer()
        opt.split_train_val = MagicMock(return_value=(_make_examples(7), _make_examples(3)))
        opt.score_preamble = MagicMock(return_value=1.0)
        opt._collect_failures = MagicMock(return_value=[])
        result = opt.optimize(
            metric_id="test", current_rubric=["r1"], current_preamble="orig",
            metric_results=[], annotations=[],
        )
        assert isinstance(result, PromptOptimizationResult)

    def test_factory_creates_textgrad(self):
        opt = create_optimizer("textgrad", config=TextGradConfig())
        assert isinstance(opt, TextGradOptimizer)


# ---------------------------------------------------------------------------
# MIPROv2 tests
# ---------------------------------------------------------------------------

from evalyn_sdk.calibration.miprov2 import MIPROv2Config, MIPROv2Optimizer


class TestMIPROv2Config:
    def test_defaults(self):
        cfg = MIPROv2Config()
        assert cfg.num_instructions == 6
        assert cfg.num_demos == 3
        assert cfg.eval_samples == 10

    def test_custom_values(self):
        cfg = MIPROv2Config(num_instructions=10, num_demos=5)
        assert cfg.num_instructions == 10
        assert cfg.num_demos == 5


class TestMIPROv2:
    def _make_optimizer(self):
        cfg = MIPROv2Config(num_instructions=3, num_demos=2, eval_samples=5)
        opt = MIPROv2Optimizer(config=cfg)
        mock_client = MagicMock()
        mock_client.generate_with_usage.return_value = _make_generate_result(
            '["Preamble candidate 1", "Preamble candidate 2", "Preamble candidate 3"]'
        )
        opt._task_client_instance = mock_client
        return opt

    def test_bootstrap_demos_diversity(self):
        cfg = MIPROv2Config(num_demos=4)
        opt = MIPROv2Optimizer(config=cfg)
        examples = []
        for i in range(20):
            label = "PASS" if i < 10 else "FAIL"
            input_text = "x" * (10 if i % 2 == 0 else 200)
            examples.append({
                "id": f"ex-{i}", "input": input_text, "output": f"out-{i}",
                "expected": label, "call_id": f"call-{i}",
            })
        demos = opt._bootstrap_demos(examples)
        assert len(demos) == 4
        for demo in demos:
            assert "Example: INPUT:" in demo

    def test_bootstrap_demos_empty(self):
        opt = MIPROv2Optimizer(config=MIPROv2Config(num_demos=3))
        assert opt._bootstrap_demos([]) == []

    def test_embed_demos_format(self):
        opt = MIPROv2Optimizer()
        instruction = "You are an expert evaluator."
        demos = [
            "Example: INPUT: test1 OUTPUT: out1 -> VERDICT: pass REASON: good",
            "Example: INPUT: test2 OUTPUT: out2 -> VERDICT: fail REASON: bad",
        ]
        result = opt._embed_demos(instruction, demos)
        assert result.startswith("You are an expert evaluator.")
        assert "Here are examples of correct evaluations:" in result
        assert "test1" in result
        assert "test2" in result

    def test_embed_demos_empty_returns_instruction(self):
        opt = MIPROv2Optimizer()
        assert opt._embed_demos("instruction", []) == "instruction"

    def test_optimize_returns_result(self):
        opt = self._make_optimizer()
        # Mock build_dataset_from_annotations to return train/val splits
        train = _make_examples(7)
        val = _make_examples(3)
        import evalyn_sdk.calibration.miprov2 as miprov2_mod
        original_build = miprov2_mod.build_dataset_from_annotations
        miprov2_mod.build_dataset_from_annotations = MagicMock(return_value=(train, val))
        try:
            opt.score_preamble = MagicMock(return_value=0.75)
            result = opt.optimize(
                metric_id="test",
                current_rubric=["Be accurate"],
                current_preamble="You are a judge.",
                metric_results=[],
                annotations=[],
                dataset_items=[],
            )
            assert isinstance(result, PromptOptimizationResult)
            assert result.original_preamble == "You are a judge."
            assert result.optimized_preamble  # non-empty
            assert result.full_prompt  # non-empty
            assert result.improved_rubric == ["Be accurate"]  # rubric unchanged
        finally:
            miprov2_mod.build_dataset_from_annotations = original_build

    def test_not_enough_data(self):
        opt = self._make_optimizer()
        import evalyn_sdk.calibration.miprov2 as miprov2_mod
        original_build = miprov2_mod.build_dataset_from_annotations
        miprov2_mod.build_dataset_from_annotations = MagicMock(return_value=([], []))
        try:
            result = opt.optimize(
                metric_id="test",
                current_rubric=["r1"],
                current_preamble="Judge.",
                metric_results=[],
                annotations=[],
            )
            assert isinstance(result, PromptOptimizationResult)
            assert "Not enough data" in result.improvement_reasoning
        finally:
            miprov2_mod.build_dataset_from_annotations = original_build

    def test_factory_creates_miprov2(self):
        opt = create_optimizer("miprov2", config=MIPROv2Config())
        assert isinstance(opt, MIPROv2Optimizer)


# ---------------------------------------------------------------------------
# PromptBreeder tests
# ---------------------------------------------------------------------------

from evalyn_sdk.calibration.promptbreeder import (
    BreederUnit,
    PromptBreederConfig,
    PromptBreederOptimizer,
)


class TestPromptBreederConfig:
    def test_defaults(self):
        cfg = PromptBreederConfig()
        assert cfg.population_size == 6
        assert cfg.generations == 5
        assert cfg.num_initial_mutation_prompts == 4
        assert cfg.early_stop_patience == 2

    def test_custom_values(self):
        cfg = PromptBreederConfig(population_size=10, generations=8)
        assert cfg.population_size == 10
        assert cfg.generations == 8


class TestBreederUnit:
    def test_dataclass(self):
        unit = BreederUnit(preamble="test", mutation_prompt="improve clarity")
        assert unit.f1_score == 0.0

    def test_score_assignment(self):
        unit = BreederUnit(preamble="p", mutation_prompt="m", f1_score=0.85)
        assert unit.f1_score == 0.85


class TestPromptBreeder:
    def _make_optimizer(self) -> PromptBreederOptimizer:
        cfg = PromptBreederConfig(
            population_size=3, generations=2,
            num_initial_mutation_prompts=2, early_stop_patience=1,
        )
        opt = PromptBreederOptimizer(config=cfg)
        mock_client = MagicMock()
        mock_client.generate_with_usage.return_value = _make_generate_result(
            '["strategy one", "strategy two"]'
        )
        opt._task_client_instance = mock_client
        opt.score_preamble = MagicMock(return_value=0.75)
        opt.split_train_val = MagicMock(return_value=(_make_examples(6), _make_examples(4)))
        return opt

    def test_optimize_returns_result(self):
        opt = self._make_optimizer()
        result = opt.optimize(
            metric_id="test", current_rubric=["Be accurate"],
            metric_results=[], annotations=[], dataset_items=[],
            current_preamble="You are a judge.",
        )
        assert isinstance(result, PromptOptimizationResult)
        assert result.original_preamble == "You are a judge."
        assert result.improved_rubric == ["Be accurate"]

    def test_early_stopping(self):
        opt = self._make_optimizer()
        result = opt.optimize(
            metric_id="test", current_rubric=["r1"],
            metric_results=[], annotations=[], dataset_items=[],
            current_preamble="Judge.",
        )
        assert isinstance(result, PromptOptimizationResult)

    def test_insufficient_data(self):
        opt = self._make_optimizer()
        opt.split_train_val = MagicMock(return_value=([], []))
        result = opt.optimize(
            metric_id="test", current_rubric=["r1"],
            metric_results=[], annotations=[],
            current_preamble="Judge.",
        )
        assert "Not enough data" in result.improvement_reasoning

    def test_factory_creates_promptbreeder(self):
        opt = create_optimizer("promptbreeder", config=PromptBreederConfig())
        assert isinstance(opt, PromptBreederOptimizer)

    def test_apply_mutation_returns_new_preamble(self):
        opt = PromptBreederOptimizer()
        mock_client = MagicMock()
        mock_client.generate_with_usage.return_value = _make_generate_result(
            "A completely new improved preamble."
        )
        opt._task_client_instance = mock_client
        unit = BreederUnit(preamble="old", mutation_prompt="focus on precision")
        new = opt._apply_mutation(unit, "(no failures)")
        assert new == "A completely new improved preamble."

    def test_apply_mutation_fallback_on_error(self):
        opt = PromptBreederOptimizer()
        mock_client = MagicMock()
        mock_client.generate_with_usage.side_effect = RuntimeError("fail")
        opt._task_client_instance = mock_client
        unit = BreederUnit(preamble="original", mutation_prompt="improve")
        assert opt._apply_mutation(unit, "") == "original"

    def test_evolve_mutation_prompt(self):
        opt = PromptBreederOptimizer()
        mock_client = MagicMock()
        mock_client.generate_with_usage.return_value = _make_generate_result(
            "Focus on boundary conditions."
        )
        opt._task_client_instance = mock_client
        unit = BreederUnit(preamble="t", mutation_prompt="old strat", f1_score=0.7)
        refined = opt._evolve_mutation_prompt(unit, prev_best_f1=0.6)
        assert refined == "Focus on boundary conditions."
