# New Optimizers Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 4 new calibration optimizers (EvoPrompt, TextGrad, MIPROv2, PromptBreeder) with shared infrastructure (BaseOptimizer, factory).

**Architecture:** Each optimizer is a single file inheriting from BaseOptimizer. A factory replaces the if/elif dispatch in engine.py. Config flows from CLI flags through per-optimizer Config dataclasses to the factory. Existing optimizers are not modified.

**Tech Stack:** Python 3.10+, GeminiClient for LLM calls, pytest for tests.

**Spec:** `docs/plans/2026-03-16-new-optimizers-design.md`

---

## Chunk 1: Infrastructure (BaseOptimizer + Factory)

### Task 1: BaseOptimizer

**Files:**
- Create: `sdk/evalyn_sdk/calibration/base_optimizer.py`
- Test: `tests/test_optimizers.py`

- [ ] **Step 1: Write failing tests for BaseOptimizer**

Create `tests/test_optimizers.py` with tests for the base class:

```python
"""Tests for new optimizer infrastructure and implementations."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

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
        assert result.improved_rubric == ["criterion 1", "criterion 2"]  # rubric unchanged
        assert result.estimated_improvement == "high"
        assert "fixed false positives" in result.improvement_reasoning
        assert result.full_prompt  # should be non-empty

    def test_split_train_val_ratio(self):
        opt = ConcreteOptimizer(config=DummyConfig())
        # Create 10 mock metric_results and annotations with matching call_ids
        # build_dataset_from_annotations matches on res.call_id == ann.target_id
        metric_results = [MagicMock(call_id=f"call-{i}", item_id=f"item-{i}", passed=i % 2 == 0) for i in range(10)]
        annotations = [MagicMock(target_id=f"call-{i}", label=(i % 2 == 0)) for i in range(10)]
        dataset_items = []
        for i in range(10):
            item = MagicMock()
            item.id = f"item-{i}"
            item.input = {"query": f"q{i}"}
            item.output = f"answer {i}"
            item.metadata = {"call_id": f"call-{i}"}
            dataset_items.append(item)
        train, val = opt.split_train_val(metric_results, annotations, dataset_items, train_ratio=0.7)
        assert len(train) + len(val) <= 10
        assert len(train) >= 5  # roughly 70%
        # build_dataset_from_annotations returns "PASS"/"FAIL" strings for expected
        for ex in train + val:
            assert ex["expected"] in ("PASS", "FAIL")

    def test_optimize_abstract_raises(self):
        with pytest.raises(NotImplementedError):
            BaseOptimizer(config=DummyConfig()).optimize(
                metric_id="test", current_rubric=[], current_preamble="", disagreements=None,
                metric_results=[], annotations=[], dataset_items=[], accumulator=None,
            )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_optimizers.py -v`
Expected: ImportError (base_optimizer module doesn't exist yet)

- [ ] **Step 3: Implement BaseOptimizer**

Create `sdk/evalyn_sdk/calibration/base_optimizer.py`:

```python
"""Base class for new preamble optimizers.

Provides shared utilities: train/val split, candidate scoring, result building.
Existing optimizers (Basic, APE, OPRO, GEPANative, GEPA) do NOT need to
subclass this - they continue working as-is via the factory adapter.
"""
from __future__ import annotations

from typing import Any, List, Optional

from .models import PromptOptimizationResult, TokenAccumulator
from .utils import build_dataset_from_annotations, build_full_prompt, parse_judge_response


class BaseOptimizer:
    """Common foundation for new preamble optimizers."""

    def __init__(self, config: Any, api_key: str | None = None):
        self.config = config
        self._api_key = api_key

    def split_train_val(
        self,
        metric_results,
        annotations,
        dataset_items,
        train_ratio: float = 0.7,
    ) -> tuple[list, list]:
        """Split annotated examples into train/val sets.

        Delegates to build_dataset_from_annotations from utils.py.
        Returns (train_examples, val_examples) where each example is a dict
        with keys: id, input, output, expected (bool), call_id.
        """
        return build_dataset_from_annotations(
            metric_results, annotations, dataset_items, train_split=train_ratio
        )

    def score_preamble(
        self,
        preamble: str,
        rubric: List[str],
        examples: list,
        accumulator: Optional[TokenAccumulator] = None,
    ) -> float:
        """Score a candidate preamble on labeled examples. Returns F1.

        Uses GeminiClient directly (same pattern as APE._score_candidate and
        OPRO._evaluate_prompt). Builds prompt from preamble+rubric, sends
        each example through, parses pass/fail, computes alignment.

        Note: build_dataset_from_annotations returns "PASS"/"FAIL" strings
        for the 'expected' field, not booleans.
        """
        from ..utils.api_client import GeminiClient
        from .models import AlignmentMetrics

        full_prompt = build_full_prompt(preamble, rubric)
        scorer_model = getattr(self.config, "scorer_model", None)
        client = GeminiClient(model=scorer_model, api_key=self._api_key)

        metrics = AlignmentMetrics()
        for ex in examples:
            try:
                eval_input = f"INPUT: {ex.get('input', '')}\nOUTPUT: {ex.get('output', '')}"
                result = client.generate(full_prompt + "\n\n" + eval_input)
                predicted = parse_judge_response(result.text)
                actual = ex.get("expected") == "PASS"  # convert string to bool
                metrics.record(predicted, actual)
                if accumulator:
                    accumulator.add(result)
            except Exception:
                pass  # skip scoring errors
        return metrics.f1

    def build_result(
        self,
        original_preamble: str,
        optimized_preamble: str,
        rubric: List[str],
        reasoning: str,
        estimated_improvement: str,
    ) -> PromptOptimizationResult:
        """Construct a standard PromptOptimizationResult."""
        return PromptOptimizationResult(
            original_rubric=list(rubric),
            improved_rubric=list(rubric),  # rubric always stays fixed
            improvement_reasoning=reasoning,
            suggested_additions=[],
            suggested_removals=[],
            estimated_improvement=estimated_improvement,
            original_preamble=original_preamble,
            optimized_preamble=optimized_preamble,
            full_prompt=build_full_prompt(optimized_preamble, rubric),
        )

    def optimize(
        self,
        *,
        metric_id: str,
        current_rubric: List[str],
        current_preamble: str,
        metric_results: list,
        annotations: list,
        disagreements: Any = None,
        dataset_items: list | None = None,
        accumulator: TokenAccumulator | None = None,
        **kwargs,
    ) -> PromptOptimizationResult:
        raise NotImplementedError
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_optimizers.py::TestBaseOptimizer -v`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add -f sdk/evalyn_sdk/calibration/base_optimizer.py tests/test_optimizers.py
git commit -m "feat(calibration): add BaseOptimizer base class with shared utilities"
```

---

### Task 2: Factory + call_optimizer

**Files:**
- Create: `sdk/evalyn_sdk/calibration/factory.py`
- Modify: `tests/test_optimizers.py`

- [ ] **Step 1: Write failing tests for factory**

Append to `tests/test_optimizers.py`:

```python
from evalyn_sdk.calibration.factory import create_optimizer, call_optimizer


class TestFactory:
    def test_create_basic_optimizer(self):
        opt = create_optimizer("basic", model="test-model")
        assert opt.__class__.__name__ == "BasicOptimizer"

    def test_create_ape_optimizer(self):
        from evalyn_sdk.calibration.ape import APEConfig
        opt = create_optimizer("ape", config=APEConfig())
        assert opt.__class__.__name__ == "APEOptimizer"

    def test_create_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown optimizer"):
            create_optimizer("nonexistent")

    def test_call_optimizer_filters_kwargs(self):
        """Legacy optimizers only get params they accept."""
        opt = create_optimizer("basic", model="test-model")
        # BasicOptimizer.optimize() does not accept metric_results or annotations
        # call_optimizer should filter those out
        import inspect
        sig = inspect.signature(opt.optimize)
        param_names = {p.name for p in sig.parameters.values() if p.name != "self"}
        assert "metric_results" not in param_names  # verify basic doesn't take it

    def test_call_optimizer_passes_kwargs_to_new_style(self):
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
        )
        assert isinstance(result, PromptOptimizationResult)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_optimizers.py::TestFactory -v`
Expected: ImportError

- [ ] **Step 3: Implement factory**

Create `sdk/evalyn_sdk/calibration/factory.py`:

```python
"""Optimizer factory: lazy-import, instantiate, and call optimizers by name.

Replaces the if/elif dispatch chain in engine.py. Handles both legacy
optimizers (with inconsistent signatures) and new BaseOptimizer subclasses.
"""
from __future__ import annotations

import inspect
from typing import Any, Optional

from .models import PromptOptimizationResult

# Registry: name -> (module_path, class_name)
# Module paths are relative to evalyn_sdk package
OPTIMIZER_REGISTRY: dict[str, tuple[str, str]] = {
    "basic": ("evalyn_sdk.calibration.basic", "BasicOptimizer"),
    "ape": ("evalyn_sdk.calibration.ape", "APEOptimizer"),
    "opro": ("evalyn_sdk.calibration.opro", "OPROOptimizer"),
    "gepa": ("evalyn_sdk.calibration.gepa", "GEPAOptimizer"),
    "gepa-native": ("evalyn_sdk.calibration.gepa_native", "GEPANativeOptimizer"),
    "evoprompt": ("evalyn_sdk.calibration.evoprompt", "EvoPromptOptimizer"),
    "textgrad": ("evalyn_sdk.calibration.textgrad", "TextGradOptimizer"),
    "miprov2": ("evalyn_sdk.calibration.miprov2", "MIPROv2Optimizer"),
    "promptbreeder": ("evalyn_sdk.calibration.promptbreeder", "PromptBreederOptimizer"),
}


def create_optimizer(
    name: str,
    config: Any = None,
    api_key: str | None = None,
    **legacy_kwargs,
) -> Any:
    """Lazy-import and instantiate optimizer by name.

    Args:
        name: optimizer name (must be in OPTIMIZER_REGISTRY)
        config: optimizer-specific Config dataclass
        api_key: optional API key
        **legacy_kwargs: for BasicOptimizer compatibility (model=..., etc.)
    """
    if name not in OPTIMIZER_REGISTRY:
        raise ValueError(
            f"Unknown optimizer: '{name}'. "
            f"Available: {', '.join(sorted(OPTIMIZER_REGISTRY))}"
        )

    module_path, class_name = OPTIMIZER_REGISTRY[name]

    # Special handling for GEPA (external library)
    if name == "gepa":
        from .gepa import GEPA_AVAILABLE
        if not GEPA_AVAILABLE:
            raise ImportError(
                "GEPA optimizer requires the 'gepa' package. "
                "Install with: pip install gepa"
            )

    # Lazy import
    import importlib
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)

    # Instantiate based on constructor signature
    init_sig = inspect.signature(cls.__init__)
    init_params = {p.name for p in init_sig.parameters.values() if p.name != "self"}

    if "config" in init_params and config is not None:
        kwargs = {"config": config}
        if "api_key" in init_params:
            kwargs["api_key"] = api_key
        return cls(**kwargs)
    elif "model" in init_params:
        # BasicOptimizer takes model, api_key
        kwargs = {}
        if legacy_kwargs.get("model"):
            kwargs["model"] = legacy_kwargs["model"]
        if "api_key" in init_params and api_key:
            kwargs["api_key"] = api_key
        return cls(**kwargs)
    else:
        # Fallback: try config + api_key, then just config
        try:
            return cls(config=config, api_key=api_key)
        except TypeError:
            try:
                return cls(config=config)
            except TypeError:
                return cls()


def call_optimizer(optimizer: Any, **kwargs) -> PromptOptimizationResult:
    """Call optimizer.optimize() with signature-aware kwarg filtering.

    New optimizers (with **kwargs in optimize()) get all params.
    Legacy optimizers get only the params their signature accepts.
    """
    sig = inspect.signature(optimizer.optimize)
    params = list(sig.parameters.values())

    # If optimize() accepts **kwargs, pass everything
    if any(p.kind == p.VAR_KEYWORD for p in params):
        return optimizer.optimize(**kwargs)

    # Otherwise filter to accepted params only
    accepted = {p.name for p in params if p.name != "self"}
    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    return optimizer.optimize(**filtered)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_optimizers.py -v`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add -f sdk/evalyn_sdk/calibration/factory.py tests/test_optimizers.py
git commit -m "feat(calibration): add optimizer factory with signature-aware dispatch"
```

---

### Task 3: Engine integration

**Files:**
- Modify: `sdk/evalyn_sdk/calibration/engine.py`
- Modify: `sdk/evalyn_sdk/calibration/__init__.py`

- [ ] **Step 1: Update engine to use factory**

In `sdk/evalyn_sdk/calibration/engine.py`, make these changes:

**1a. Update `CalibrationConfig` dataclass** - add `optimizer_config` field:
```python
@dataclass
class CalibrationConfig:
    # ... existing fields ...
    optimizer_config: Any = None      # NEW: generic optimizer config (any Config dataclass)
    optimizer_api_key: str | None = None  # NEW: API key for optimizer
```

**1b. Update `CalibrationEngine.__init__`** - add `optimizer_config` + backward compat:
```python
def __init__(self, ..., optimizer_config=None, optimizer_api_key=None,
             # Backward compat (deprecated):
             gepa_config=None, opro_config=None, ape_config=None, gepa_native_config=None):
    # ...
    self.optimizer_config = optimizer_config or gepa_config or opro_config or ape_config or gepa_native_config
    self.optimizer_api_key = optimizer_api_key  # new field
```

**1c. Replace if/elif dispatch in `calibrate()`** with factory call:
```python
from .factory import create_optimizer, call_optimizer

optimizer = create_optimizer(
    self.optimizer_type,
    config=self.optimizer_config,
    api_key=self.optimizer_api_key,
    model=self.optimizer_model,  # for BasicOptimizer compat
)
# Note: engine variable is `alignment` but BasicOptimizer param is `alignment_metrics`
prompt_optimization = call_optimizer(
    optimizer,
    metric_id=self.judge_name,
    current_rubric=self.current_rubric,
    metric_results=metric_results,
    annotations=annotations,
    disagreements=disagreement_analysis,
    alignment_metrics=alignment,  # engine var is `alignment`, param name is `alignment_metrics`
    dataset_items=dataset_items,
    current_preamble=self.current_preamble,
    accumulator=accumulator,
)
```

**1d. Update config serialization** in the adjustments dict (around line 600):
```python
# Replace optimizer-specific config serialization with generic:
if self.optimizer_config and hasattr(self.optimizer_config, "__dataclass_fields__"):
    adjustments["optimizer_config"] = {
        k: getattr(self.optimizer_config, k)
        for k in self.optimizer_config.__dataclass_fields__
    }
adjustments["optimizer_type"] = self.optimizer_type
```

**1e. Update `_run_calibration_with_spinner`** in CLI (calibration.py ~line 316):
Add new optimizer names to the spinner message:
```python
# Add to the method that builds spinner text:
# The spinner already handles "basic", "gepa", "opro", "ape", "gepa-native"
# Add: "evoprompt", "textgrad", "miprov2", "promptbreeder"
# These are all long-running, so they all get the spinner.
```

- [ ] **Step 2: Update __init__.py exports**

Add to `sdk/evalyn_sdk/calibration/__init__.py`:
```python
from .base_optimizer import BaseOptimizer
from .factory import OPTIMIZER_REGISTRY, call_optimizer, create_optimizer
```

And add to `__all__`: `"BaseOptimizer"`, `"create_optimizer"`, `"call_optimizer"`, `"OPTIMIZER_REGISTRY"`.

- [ ] **Step 3: Run existing tests to verify nothing broke**

Run: `uv run pytest tests/test_ape.py tests/test_optimizers.py -v`
Expected: all pass (existing APE tests still work, new tests still pass)

- [ ] **Step 4: Commit**

```bash
git add sdk/evalyn_sdk/calibration/engine.py sdk/evalyn_sdk/calibration/__init__.py
git commit -m "refactor(calibration): replace if/elif dispatch with factory in engine"
```

---

## Chunk 2: TextGrad Optimizer (simplest new optimizer - good first)

### Task 4: TextGrad implementation

**Files:**
- Create: `sdk/evalyn_sdk/calibration/textgrad.py`
- Modify: `tests/test_optimizers.py`

- [ ] **Step 1: Write failing tests for TextGrad**

Append to `tests/test_optimizers.py`:

```python
from evalyn_sdk.calibration.textgrad import TextGradConfig, TextGradOptimizer


class TestTextGradConfig:
    def test_defaults(self):
        cfg = TextGradConfig()
        assert cfg.max_iterations == 8
        assert cfg.improvement_threshold == 0.01
        assert cfg.num_failure_examples == 5
        assert cfg.early_stop_patience == 3

    def test_custom_values(self):
        cfg = TextGradConfig(max_iterations=3, improvement_threshold=0.05)
        assert cfg.max_iterations == 3
        assert cfg.improvement_threshold == 0.05


class TestTextGrad:
    def test_optimize_returns_result(self):
        """TextGrad with mocked LLM calls should return PromptOptimizationResult."""
        cfg = TextGradConfig(max_iterations=2, num_failure_examples=2)
        opt = TextGradOptimizer(config=cfg)

        # Mock the LLM client to return predictable responses
        mock_client = MagicMock()
        # Critique response
        mock_client.generate.side_effect = [
            MagicMock(text="The preamble is too vague about edge cases.", input_tokens=100, output_tokens=50, model="test"),
            MagicMock(text="You are an improved judge that handles edge cases precisely.", input_tokens=100, output_tokens=80, model="test"),
            MagicMock(text="Minor issue with ambiguous thresholds.", input_tokens=100, output_tokens=50, model="test"),
            MagicMock(text="You are a refined judge with clear thresholds.", input_tokens=100, output_tokens=80, model="test"),
        ]
        opt._task_client = mock_client

        # Mock score_preamble to return improving F1 scores
        scores = iter([0.6, 0.75, 0.8])
        opt.score_preamble = MagicMock(side_effect=lambda *a, **kw: next(scores))

        # Mock split_train_val
        train = [{"id": "1", "input": "q", "output": "a", "expected": True}]
        val = [{"id": "2", "input": "q2", "output": "a2", "expected": False}]
        opt.split_train_val = MagicMock(return_value=(train, val))

        result = opt.optimize(
            metric_id="helpfulness",
            current_rubric=["Be helpful"],
            current_preamble="You are a judge.",
            metric_results=[], annotations=[], dataset_items=[],
            accumulator=None, disagreements=None,
        )
        assert isinstance(result, PromptOptimizationResult)
        assert result.optimized_preamble != result.original_preamble

    def test_early_stopping(self):
        """TextGrad should stop early when no improvement."""
        cfg = TextGradConfig(max_iterations=10, early_stop_patience=2)
        opt = TextGradOptimizer(config=cfg)

        mock_client = MagicMock()
        mock_client.generate.return_value = MagicMock(
            text="No changes needed.", input_tokens=50, output_tokens=30, model="test"
        )
        opt._task_client = mock_client

        # score_preamble always returns same value -> should trigger early stop
        opt.score_preamble = MagicMock(return_value=0.7)
        opt.split_train_val = MagicMock(return_value=(
            [{"id": "1", "input": "q", "output": "a", "expected": True}], []
        ))

        result = opt.optimize(
            metric_id="test", current_rubric=["r1"], current_preamble="original",
            metric_results=[], annotations=[], dataset_items=[],
            accumulator=None, disagreements=None,
        )
        # Should have stopped early, not run all 10 iterations
        assert mock_client.generate.call_count < 20  # 2 calls per iter * 10 = 20 max

    def test_factory_creates_textgrad(self):
        cfg = TextGradConfig()
        opt = create_optimizer("textgrad", config=cfg)
        assert isinstance(opt, TextGradOptimizer)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_optimizers.py::TestTextGrad -v`
Expected: ImportError

- [ ] **Step 3: Implement TextGrad**

Create `sdk/evalyn_sdk/calibration/textgrad.py` implementing the algorithm from the spec: critique step -> revision step -> score -> greedy-best tracking -> early stopping.

Key structure (follow existing APE/OPRO pattern for GeminiClient initialization):
```python
from ..utils.api_client import GeminiClient

DEFAULT_EVAL_MODEL = "gemini-2.5-flash-lite"

@dataclass
class TextGradConfig:
    max_iterations: int = 8
    improvement_threshold: float = 0.01
    num_failure_examples: int = 5
    early_stop_patience: int = 3
    task_model: str = DEFAULT_EVAL_MODEL
    scorer_model: str = DEFAULT_EVAL_MODEL

class TextGradOptimizer(BaseOptimizer):
    # Lazy GeminiClient initialization (same pattern as APE/OPRO):
    @property
    def _task_client(self):
        if not hasattr(self, "_task_client_instance"):
            self._task_client_instance = GeminiClient(
                model=self.config.task_model, api_key=self._api_key
            )
        return self._task_client_instance

    def optimize(self, *, metric_id, current_rubric, current_preamble, ..., **kwargs):
        train, val = self.split_train_val(...)
        best_preamble, best_f1 = current_preamble, self.score_preamble(...)
        no_improve_count = 0
        for i in range(self.config.max_iterations):
            failures = self._collect_failures(best_preamble, rubric, train, ...)
            critique = self._critique(best_preamble, failures)  # uses self._task_client
            revised = self._revise(best_preamble, critique)     # uses self._task_client
            f1 = self.score_preamble(revised, rubric, train, ...)
            if f1 > best_f1 + self.config.improvement_threshold:
                best_preamble, best_f1 = revised, f1
                no_improve_count = 0
            else:
                no_improve_count += 1
            if no_improve_count >= self.config.early_stop_patience:
                break
        return self.build_result(current_preamble, best_preamble, rubric, ...)
```

All 4 new optimizers follow this same lazy property pattern for GeminiClient.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_optimizers.py -v`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add -f sdk/evalyn_sdk/calibration/textgrad.py tests/test_optimizers.py
git commit -m "feat(calibration): add TextGrad optimizer - iterative critique-based refinement"
```

---

## Chunk 3: EvoPrompt Optimizer

### Task 5: EvoPrompt implementation

**Files:**
- Create: `sdk/evalyn_sdk/calibration/evoprompt.py`
- Modify: `tests/test_optimizers.py`

- [ ] **Step 1: Write failing tests for EvoPrompt**

Test config defaults, population initialization, crossover, mutation, selection, early stopping, factory integration. Use mocked LLM client.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_optimizers.py::TestEvoPrompt -v`

- [ ] **Step 3: Implement EvoPrompt**

Create `sdk/evalyn_sdk/calibration/evoprompt.py` implementing: population init -> tournament selection -> crossover (probabilistic) -> mutation (probabilistic) -> scoring -> elitist selection -> early stopping.

Key structure:
```python
@dataclass
class EvoPromptConfig:
    population_size: int = 8
    generations: int = 5
    mutation_rate: float = 0.3
    crossover_rate: float = 0.7
    tournament_size: int = 2
    early_stop_patience: int = 2
    task_model: str = DEFAULT_EVAL_MODEL
    scorer_model: str = DEFAULT_EVAL_MODEL

class EvoPromptOptimizer(BaseOptimizer):
    def optimize(self, *, metric_id, current_rubric, current_preamble, disagreements, ..., **kwargs):
        train, val = self.split_train_val(...)
        population = self._init_population(current_preamble, disagreements, ...)
        # Score initial population
        scored = [(p, self.score_preamble(p, rubric, train, ...)) for p in population]
        best_f1 = max(s for _, s in scored)
        no_improve = 0
        for gen in range(self.config.generations):
            offspring = []
            for _ in range(self.config.population_size // 2):
                p1, p2 = self._tournament_select(scored), self._tournament_select(scored)
                child = self._crossover(p1, p2) if random() < self.config.crossover_rate else better(p1, p2)
                if random() < self.config.mutation_rate:
                    child = self._mutate(child, failures, ...)
                offspring.append(child)
            # Score offspring, merge with parents, select top-K
            ...
            if new_best <= best_f1: no_improve += 1
            if no_improve >= self.config.early_stop_patience: break
        return self.build_result(...)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_optimizers.py -v`

- [ ] **Step 5: Commit**

```bash
git add -f sdk/evalyn_sdk/calibration/evoprompt.py tests/test_optimizers.py
git commit -m "feat(calibration): add EvoPrompt optimizer - evolutionary population-based search"
```

---

## Chunk 4: MIPROv2 Optimizer

### Task 6: MIPROv2 implementation

**Files:**
- Create: `sdk/evalyn_sdk/calibration/miprov2.py`
- Modify: `tests/test_optimizers.py`

- [ ] **Step 1: Write failing tests for MIPROv2**

Test config defaults, instruction generation, demo bootstrap (bucketing logic), joint selection (greedy demo addition), demo embedding format, factory integration.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_optimizers.py::TestMIPROv2 -v`

- [ ] **Step 3: Implement MIPROv2**

Create `sdk/evalyn_sdk/calibration/miprov2.py` implementing: Stage 1 (instruction generation from disagreements) -> Stage 2 (demo bootstrap with label/length bucketing) -> Stage 3 (greedy instruction+demo selection).

Key structure:
```python
@dataclass
class MIPROv2Config:
    num_instructions: int = 6
    num_demos: int = 3
    eval_samples: int = 10
    task_model: str = DEFAULT_EVAL_MODEL
    scorer_model: str = DEFAULT_EVAL_MODEL

class MIPROv2Optimizer(BaseOptimizer):
    def optimize(self, *, metric_id, current_rubric, current_preamble, disagreements, ..., **kwargs):
        train, val = self.split_train_val(...)
        # Stage 1: generate diverse instructions
        instructions = self._generate_instructions(current_preamble, disagreements, rubric, ...)
        # Stage 2: bootstrap demos from correct examples
        demos = self._bootstrap_demos(train, ...)
        # Stage 3: joint selection
        best_instruction = self._select_best_instruction(instructions, rubric, train, ...)
        best_demos = self._greedy_demo_selection(best_instruction, demos, rubric, train, ...)
        # Build final preamble with embedded demos
        final = self._embed_demos(best_instruction, best_demos)
        return self.build_result(current_preamble, final, rubric, ...)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_optimizers.py -v`

- [ ] **Step 5: Commit**

```bash
git add -f sdk/evalyn_sdk/calibration/miprov2.py tests/test_optimizers.py
git commit -m "feat(calibration): add MIPROv2 optimizer - joint instruction+demo optimization"
```

---

## Chunk 5: PromptBreeder Optimizer

### Task 7: PromptBreeder implementation

**Files:**
- Create: `sdk/evalyn_sdk/calibration/promptbreeder.py`
- Modify: `tests/test_optimizers.py`

- [ ] **Step 1: Write failing tests for PromptBreeder**

Test config defaults, BreederUnit dataclass, population initialization, mutation (apply mutation_prompt to preamble), selection, mutation prompt evolution, early stopping, factory integration.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_optimizers.py::TestPromptBreeder -v`

- [ ] **Step 3: Implement PromptBreeder**

Create `sdk/evalyn_sdk/calibration/promptbreeder.py` implementing: BreederUnit pairs -> init population -> mutate preambles via mutation_prompts -> score -> select top-K -> evolve mutation_prompts -> early stopping.

Key structure:
```python
@dataclass
class BreederUnit:
    preamble: str
    mutation_prompt: str
    f1_score: float = 0.0

@dataclass
class PromptBreederConfig:
    population_size: int = 6
    generations: int = 5
    num_initial_mutation_prompts: int = 4
    early_stop_patience: int = 2
    task_model: str = DEFAULT_EVAL_MODEL
    scorer_model: str = DEFAULT_EVAL_MODEL

class PromptBreederOptimizer(BaseOptimizer):
    def optimize(self, *, metric_id, current_rubric, current_preamble, ..., **kwargs):
        train, val = self.split_train_val(...)
        population = self._init_population(current_preamble, ...)
        # Score initial population
        for unit in population:
            unit.f1_score = self.score_preamble(unit.preamble, rubric, train, ...)
        best_f1 = max(u.f1_score for u in population)
        no_improve = 0
        for gen in range(self.config.generations):
            # Mutate preambles
            for unit in population:
                unit.preamble = self._apply_mutation(unit, failures, ...)
                unit.f1_score = self.score_preamble(unit.preamble, rubric, train, ...)
            # Select top-K
            population = sorted(population, key=lambda u: u.f1_score, reverse=True)[:self.config.population_size]
            # Evolve mutation prompts
            for unit in population:
                unit.mutation_prompt = self._evolve_mutation_prompt(unit, best_f1, ...)
            ...
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_optimizers.py -v`

- [ ] **Step 5: Commit**

```bash
git add -f sdk/evalyn_sdk/calibration/promptbreeder.py tests/test_optimizers.py
git commit -m "feat(calibration): add PromptBreeder optimizer - self-referential prompt evolution"
```

---

## Chunk 6: CLI Integration + Docs

### Task 8: CLI integration

**Files:**
- Modify: `sdk/evalyn_sdk/cli/commands/calibration.py`

- [ ] **Step 1: Add new optimizer choices and flags**

In `register_commands()`, add to `--optimizer` choices: `"evoprompt"`, `"textgrad"`, `"miprov2"`, `"promptbreeder"`.

Add argument groups for each new optimizer:
```python
# EvoPrompt
p.add_argument("--evo-population", type=int, default=8)
p.add_argument("--evo-generations", type=int, default=5)
p.add_argument("--evo-mutation-rate", type=float, default=0.3)
# TextGrad
p.add_argument("--textgrad-iterations", type=int, default=8)
p.add_argument("--textgrad-threshold", type=float, default=0.01)
# MIPROv2
p.add_argument("--mipro-instructions", type=int, default=6)
p.add_argument("--mipro-demos", type=int, default=3)
p.add_argument("--mipro-eval-samples", type=int, default=10)
# PromptBreeder
p.add_argument("--pb-population", type=int, default=6)
p.add_argument("--pb-generations", type=int, default=5)
```

- [ ] **Step 2: Update _build_calibration_optimizer_configs**

Currently returns a dict with named config keys. For new optimizers, add branches that
set a single `optimizer_config` key. For existing optimizers, keep the old keys for compat:

```python
# Add new branches:
if args.optimizer == "evoprompt":
    from ...calibration.evoprompt import EvoPromptConfig
    configs["optimizer_config"] = EvoPromptConfig(
        population_size=getattr(args, "evo_population", 8),
        generations=getattr(args, "evo_generations", 5),
        mutation_rate=getattr(args, "evo_mutation_rate", 0.3),
    )
elif args.optimizer == "textgrad":
    from ...calibration.textgrad import TextGradConfig
    configs["optimizer_config"] = TextGradConfig(
        max_iterations=getattr(args, "textgrad_iterations", 8),
        improvement_threshold=getattr(args, "textgrad_threshold", 0.01),
    )
elif args.optimizer == "miprov2":
    from ...calibration.miprov2 import MIPROv2Config
    configs["optimizer_config"] = MIPROv2Config(
        num_instructions=getattr(args, "mipro_instructions", 6),
        num_demos=getattr(args, "mipro_demos", 3),
        eval_samples=getattr(args, "mipro_eval_samples", 10),
    )
elif args.optimizer == "promptbreeder":
    from ...calibration.promptbreeder import PromptBreederConfig
    configs["optimizer_config"] = PromptBreederConfig(
        population_size=getattr(args, "pb_population", 6),
        generations=getattr(args, "pb_generations", 5),
    )
# Existing branches for ape, opro, gepa, gepa-native unchanged
```

- [ ] **Step 3: Update _build_calibration_engine to pass optimizer_config**

In `_build_calibration_engine`, extract `optimizer_config` from the configs dict and pass to engine:
```python
optimizer_config = configs.pop("optimizer_config", None)
# ... existing unpacking of gepa_config, opro_config, etc. ...
engine = CalibrationEngine(
    ...,
    optimizer_config=optimizer_config,
    # keep deprecated params for existing optimizers:
    gepa_config=configs.get("gepa_config"),
    opro_config=configs.get("opro_config"),
    ape_config=configs.get("ape_config"),
    gepa_native_config=configs.get("gepa_native_config"),
)
```

- [ ] **Step 4: Test CLI help shows new optimizers**

Run: `uv run python -m evalyn_sdk.cli calibrate --help`
Expected: `--optimizer` choices include evoprompt, textgrad, miprov2, promptbreeder

- [ ] **Step 5: Commit**

```bash
git add sdk/evalyn_sdk/cli/commands/calibration.py
git commit -m "feat(cli): add evoprompt, textgrad, miprov2, promptbreeder to calibrate command"
```

---

### Task 9: Documentation

**Files:**
- Create: `docs/optimizers/evoprompt.md`
- Create: `docs/optimizers/textgrad.md`
- Create: `docs/optimizers/miprov2.md`
- Create: `docs/optimizers/promptbreeder.md`
- Modify: `docs/optimizers/README.md`
- Modify: `ROADMAP.md`

- [ ] **Step 1: Write optimizer docs**

Each doc should follow the pattern in existing `docs/optimizers/ape.md`: algorithm overview, when to use, config options, CLI example, cost estimate.

- [ ] **Step 2: Update README comparison table**

Add rows for new optimizers to the comparison table in `docs/optimizers/README.md`.

- [ ] **Step 3: Update ROADMAP.md**

Mark "More Optimizers" sub-items as complete:
```markdown
- [x] DSPy MIPROv2 - Multi-stage instruction optimization
- [x] TextGrad - Gradient-based prompt optimization
- [x] EvoPrompt - Evolutionary prompt optimization
- [x] PromptBreeder - Self-referential prompt evolution
```

- [ ] **Step 4: Commit**

```bash
git add docs/optimizers/ ROADMAP.md
git commit -m "docs: add documentation for 4 new optimizers"
```

---

### Task 10: Final verification

- [ ] **Step 1: Run full test suite**

```bash
uv run pytest tests/test_optimizers.py tests/test_ape.py -v
```
Expected: all pass

- [ ] **Step 2: Run CLI smoke test**

```bash
uv run python -m evalyn_sdk.cli calibrate --help
```
Verify all 9 optimizer choices listed.

- [ ] **Step 3: Import smoke test**

```bash
uv run python -c "
from evalyn_sdk.calibration import (
    create_optimizer, call_optimizer, BaseOptimizer, OPTIMIZER_REGISTRY,
)
from evalyn_sdk.calibration.textgrad import TextGradOptimizer, TextGradConfig
from evalyn_sdk.calibration.evoprompt import EvoPromptOptimizer, EvoPromptConfig
from evalyn_sdk.calibration.miprov2 import MIPROv2Optimizer, MIPROv2Config
from evalyn_sdk.calibration.promptbreeder import PromptBreederOptimizer, PromptBreederConfig
print(f'Registry: {len(OPTIMIZER_REGISTRY)} optimizers')
print('All imports OK')
"
```

- [ ] **Step 4: Final commit and cleanup**

```bash
git status
# Ensure no uncommitted changes
```
