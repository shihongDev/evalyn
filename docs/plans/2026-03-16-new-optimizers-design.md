# New Optimizers Design: EvoPrompt, TextGrad, MIPROv2, PromptBreeder

## Context

Evalyn's calibration system optimizes LLM judge preambles (system prompts) to align with human annotations. Five optimizers exist: Basic (single-shot), APE (search + UCB), OPRO (trajectory-based), GEPA (external library), GEPANative (evolutionary with Pareto). All share a common interface: `optimize() -> PromptOptimizationResult`. All keep the rubric fixed and only modify the preamble.

This design adds 4 new optimizers adapted from published prompt optimization research, plus shared infrastructure (base class, factory) to reduce duplication.

## Design Decisions

- **Evalyn-adapted**: each algorithm's core idea is adapted to evalyn's preamble optimization pattern rather than faithful paper reimplementation
- **One file per optimizer**: follows existing pattern (`ape.py`, `opro.py`, etc.)
- **BaseOptimizer base class**: extracts common utilities (train/val split, candidate scoring, token tracking)
- **Factory function**: replaces growing if/elif dispatch in engine.py
- **Unconstrained cost**: each algorithm's natural complexity determines token usage; documented clearly

## Architecture

### BaseOptimizer (`calibration/base_optimizer.py`)

Shared base class for new optimizers. Existing optimizers do NOT need to adopt it - they continue working as-is. New optimizers inherit from it.

```python
class BaseOptimizer:
    """Common foundation for new preamble optimizers.

    Existing optimizers (Basic, APE, OPRO, GEPANative, GEPA) are NOT
    required to subclass this. The factory handles both styles.
    """

    def __init__(self, config: Any, api_key: str | None = None):
        """Accept optimizer-specific config dataclass + optional API key."""
        self.config = config
        self._api_key = api_key

    # --- Shared utilities (delegate to calibration/utils.py) ---
    def split_train_val(self, metric_results, annotations, dataset_items, train_ratio=0.7) -> tuple[list, list]:
        """Delegates to utils.build_dataset_from_annotations."""

    def score_preamble(self, preamble, rubric, examples, accumulator) -> float:
        """Score a candidate preamble against labeled examples. Returns F1.
        Uses LLMJudge internally with scorer_model from config."""

    def build_result(self, original_preamble, optimized_preamble, rubric,
                     reasoning, estimated_improvement) -> PromptOptimizationResult:
        """Construct PromptOptimizationResult with standard field population."""

    # --- Abstract ---
    def optimize(self, *, metric_id, current_rubric, metric_results, annotations,
                 disagreements, dataset_items, current_preamble, accumulator,
                 **kwargs) -> PromptOptimizationResult:
        """All params passed as kwargs. Optimizer uses what it needs, ignores the rest."""
        raise NotImplementedError
```

**Key design choices for interface reconciliation** (addresses review issue #1, #4):

The existing optimizers have inconsistent signatures:
- BasicOptimizer takes `disagreements`, `alignment_metrics` but not `metric_results`
- APEOptimizer takes `disagreements`, `metric_results`, `annotations`
- OPROOptimizer takes `metric_results`, `annotations` but not `disagreements`
- GEPAOptimizer takes no `accumulator` or `disagreements`

Resolution: the factory calls ALL optimizers with keyword arguments. Each optimizer's `optimize()` accepts `**kwargs` for params it doesn't use. The engine passes the full set:

```python
optimizer.optimize(
    metric_id=..., current_rubric=..., metric_results=..., annotations=...,
    disagreements=..., dataset_items=..., current_preamble=...,
    accumulator=..., alignment_metrics=...,
)
```

Existing optimizers are NOT modified. Their `optimize()` methods accept only the params they need. Python's function call semantics mean extra keyword args to positional-or-keyword params simply won't match - so we wrap existing optimizer calls in a thin adapter within the factory:

```python
def _call_optimizer(optimizer, **all_kwargs):
    """Inspect optimizer.optimize signature, pass only matching params."""
    sig = inspect.signature(optimizer.optimize)
    accepted = {p.name for p in sig.parameters.values() if p.name != 'self'}
    if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
        return optimizer.optimize(**all_kwargs)  # accepts **kwargs
    filtered = {k: v for k, v in all_kwargs.items() if k in accepted}
    return optimizer.optimize(**filtered)
```

### Factory (`calibration/factory.py`)

```python
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

def create_optimizer(name: str, config: Any = None, api_key: str | None = None, **legacy_kwargs):
    """Lazy-import and instantiate optimizer by name.

    For new optimizers: passes config dataclass + api_key.
    For legacy optimizers: passes config=<specific_config> or model=<model_name>.
    For 'gepa' (external library): checks GEPA_AVAILABLE, raises helpful error if missing.
    """

def call_optimizer(optimizer, **kwargs) -> PromptOptimizationResult:
    """Signature-aware call. Inspects optimize() params, passes only matching ones."""
```

### Config plumbing (addresses review issue #3, #11)

Instead of adding 4 more config fields to `CalibrationEngine.__init__`, use a single generic field:

```python
class CalibrationEngine:
    def __init__(self, ..., optimizer_type="basic", optimizer_config=None, optimizer_api_key=None):
        # optimizer_config: the optimizer-specific Config dataclass (any type)
        # Replaces: gepa_config, opro_config, ape_config, gepa_native_config
```

The CLI builds the appropriate config dataclass and passes it:

```python
# In _build_calibration_optimizer_configs():
if args.optimizer == "evoprompt":
    return EvoPromptConfig(population_size=args.evo_population, ...)
elif args.optimizer == "textgrad":
    return TextGradConfig(max_iterations=args.textgrad_iterations, ...)
# ... existing ones unchanged ...
```

The engine passes `optimizer_config` to the factory:
```python
optimizer = create_optimizer(self.optimizer_type, config=self.optimizer_config, api_key=self.optimizer_api_key)
```

Existing optimizer-specific fields (`gepa_config`, etc.) remain as deprecated aliases for backward compatibility. New code uses `optimizer_config`.

## New Optimizers

### 1. EvoPrompt (`calibration/evoprompt.py`)

**Paper**: "EvoPrompt: Language Models for Code-Level Prompt Optimization" (adapted)

**Core idea**: Maintain a population of preambles. Each generation, apply crossover and mutation to produce offspring. Score on training examples. Select survivors by F1.

**Algorithm**:
1. Split data into train/val sets (70/30)
2. Initialize population: current preamble + (N-1) LLM-generated variants from disagreement analysis
3. Score initial population on training set
4. For each generation:
   a. Select parent pairs via tournament selection (size=`tournament_size`)
   b. For each pair, with probability `crossover_rate`: ask LLM to combine strengths of two parents into one child. Otherwise: clone the better parent.
   c. For each child, with probability `mutation_rate`: ask LLM to improve a section based on sampled failure examples
   d. Score all offspring on training set (F1)
   e. Select top-K survivors from parents + offspring (elitist selection)
   f. **Early stopping**: if best F1 unchanged for 2 consecutive generations, stop
5. Return best preamble by F1

**Config**:
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
```

**CLI flags**: `--evo-population`, `--evo-generations`, `--evo-mutation-rate`

### 2. TextGrad (`calibration/textgrad.py`)

**Paper**: "TextGrad: Automatic Differentiation via Text" (adapted)

**Core idea**: Iterative refinement using "text critiques." The LLM produces a natural-language analysis of what's wrong with the current preamble (the "critique"), then applies it to produce an improved version.

**Algorithm**:
1. Split data into train/val sets (70/30)
2. Start with current preamble. Set `best_preamble = current`, `best_f1 = score(current)`
3. For each iteration:
   a. Score current preamble on training set, collect up to `num_failure_examples` failure cases
   b. **Critique step**: ask LLM: "Given this preamble and these failures, what specific weaknesses in the preamble caused these errors? Be precise about which phrases or instructions are problematic."
   c. **Revision step**: ask LLM: "Revise this preamble to address the following critique: [critique]. Keep the same general structure but fix the identified weaknesses."
   d. Score revised preamble on training set
   e. If revised F1 > best_f1: accept, update best. Otherwise: keep best (greedy-best tracking).
   f. **Early stopping**: if no improvement for 3 consecutive iterations, or improvement < `improvement_threshold`, stop
4. Return `best_preamble`

Note: "momentum" from the original spec was misleading. This is greedy-best tracking - always keep the best preamble seen so far, even if the current iteration regresses.

**Config**:
```python
@dataclass
class TextGradConfig:
    max_iterations: int = 8
    improvement_threshold: float = 0.01
    num_failure_examples: int = 5
    early_stop_patience: int = 3
    task_model: str = DEFAULT_EVAL_MODEL
    scorer_model: str = DEFAULT_EVAL_MODEL
```

**CLI flags**: `--textgrad-iterations`, `--textgrad-threshold`

### 3. MIPROv2 (`calibration/miprov2.py`)

**Paper**: "Optimizing Instructions and Demonstrations Jointly for Multi-Stage Language Model Programs" (adapted)

**Core idea**: Multi-stage optimization: generate diverse instructions, bootstrap few-shot demos from correct evaluations, jointly select best instruction+demo combination.

**Algorithm**:
1. Split data into train/val sets (70/30)
2. **Stage 1 - Instruction Generation**: Ask LLM to generate N diverse preamble candidates. The prompt includes: disagreement summary (false positive count, false negative count, example failure reasons), current preamble, rubric. Each candidate should target a different failure pattern.
3. **Stage 2 - Demo Bootstrap**: From training examples where judge verdict == human label, select K diverse demonstrations. Diversity is achieved by simple heuristic: bucket by (label=pass, label=fail) and (input length < median, >= median), then sample evenly from buckets. Each demo is formatted as:
   ```
   Example: INPUT: {input} OUTPUT: {output} -> VERDICT: {pass/fail} REASON: {reason}
   ```
4. **Stage 3 - Joint Selection**:
   a. Score each instruction alone (no demos) on training set. Take top 3.
   b. For each top instruction, greedily add demos one at a time. Score after each addition. Keep demo if F1 improves.
   c. Select best (instruction, demo_set) by F1.
5. Build final preamble by concatenating: instruction text + "\n\nHere are examples of correct evaluations:\n" + demo_text
6. Return final preamble

**Demo embedding format** (addresses review issue #5):
Demos are appended to the preamble text before the rubric. The existing `build_full_prompt()` in `utils.py` concatenates `preamble + rubric + format_instructions`. Since demos are part of the preamble string, no changes to `build_full_prompt()` are needed.

**Config**:
```python
@dataclass
class MIPROv2Config:
    num_instructions: int = 6
    num_demos: int = 3
    eval_samples: int = 10
    task_model: str = DEFAULT_EVAL_MODEL
    scorer_model: str = DEFAULT_EVAL_MODEL
```

**CLI flags**: `--mipro-instructions`, `--mipro-demos`, `--mipro-eval-samples`

### 4. PromptBreeder (`calibration/promptbreeder.py`)

**Paper**: "PromptBreeder: Self-Referential Self-Improvement via Prompt Evolution" (adapted)

**Core idea**: Co-evolve preambles alongside "mutation prompts" - meta-instructions that describe how to improve preambles. The mutation prompt itself is evolved, creating a self-referential improvement loop.

**Data structure**:
```python
@dataclass
class BreederUnit:
    preamble: str
    mutation_prompt: str
    f1_score: float = 0.0
```

Population is a list of `BreederUnit`. Each unit pairs a preamble with the mutation prompt that produced it.

**Algorithm**:
1. Split data into train/val sets (70/30)
2. Initialize population:
   a. Create `population_size` BreederUnits
   b. Preamble[0] = current preamble. Preamble[1..N-1] = LLM-generated variants.
   c. Mutation prompts: diverse initial strategies generated by asking LLM: "Generate N distinct strategies for improving an LLM judge's evaluation prompt. Each strategy should focus on a different aspect (e.g., precision, recall, edge cases, clarity, specificity)."
   d. Pair: unit[i] gets preamble[i] and mutation_prompt[i % num_mutation_prompts]
3. Score initial population
4. For each generation:
   a. **Mutate preambles**: for each unit, apply its mutation_prompt to its preamble:
      - Prompt: "You are improving an LLM judge's preamble. Strategy: {mutation_prompt}. Current preamble: {preamble}. Failure examples: {sampled_failures}. Apply the strategy to produce an improved preamble."
      - Result: new_preamble
   b. **Score**: evaluate each new_preamble on training set
   c. **Select**: keep top-K units by F1 (elitist). Discard the rest.
   d. **Evolve mutation prompts** (for surviving units only):
      - Prompt: "This mutation strategy was used: '{mutation_prompt}'. It produced a preamble scoring {f1_score}. The previous best scored {prev_best}. How should this strategy be refined to produce even better preambles? Return only the improved strategy."
      - Result: updated mutation_prompt for that unit
   e. **Early stopping**: if best F1 unchanged for 2 consecutive generations, stop
5. Return best preamble from final population

**Config**:
```python
@dataclass
class PromptBreederConfig:
    population_size: int = 6
    generations: int = 5
    num_initial_mutation_prompts: int = 4
    early_stop_patience: int = 2
    task_model: str = DEFAULT_EVAL_MODEL
    scorer_model: str = DEFAULT_EVAL_MODEL
```

**CLI flags**: `--pb-population`, `--pb-generations`

## Integration Changes

### Engine (`calibration/engine.py`)

Replace the if/elif optimizer dispatch with factory call. Add `optimizer_config` generic field. Keep backward-compat aliases for existing config fields.

```python
class CalibrationEngine:
    def __init__(self, ..., optimizer_type="basic", optimizer_config=None, optimizer_api_key=None,
                 # Backward compat (deprecated - use optimizer_config instead):
                 gepa_config=None, opro_config=None, ape_config=None, gepa_native_config=None):
        self.optimizer_config = optimizer_config or gepa_config or opro_config or ape_config or gepa_native_config

    def calibrate(self, ...):
        # ... alignment, disagreements, threshold as before ...
        if self.optimize_prompts:
            optimizer = create_optimizer(self.optimizer_type, config=self.optimizer_config, api_key=self.optimizer_api_key)
            prompt_optimization = call_optimizer(
                optimizer,
                metric_id=self.judge_name,
                current_rubric=self.current_rubric,
                metric_results=metric_results,
                annotations=annotations,
                disagreements=disagreement_analysis,
                alignment_metrics=alignment_metrics,
                dataset_items=dataset_items,
                current_preamble=self.current_preamble,
                accumulator=accumulator,
            )
```

### CLI (`cli/commands/calibration.py`)

Add to `--optimizer` choices: `evoprompt`, `textgrad`, `miprov2`, `promptbreeder`.

Add optimizer-specific flags:
- `--evo-population` (int, default 8), `--evo-generations` (int, default 5), `--evo-mutation-rate` (float, default 0.3)
- `--textgrad-iterations` (int, default 8), `--textgrad-threshold` (float, default 0.01)
- `--mipro-instructions` (int, default 6), `--mipro-demos` (int, default 3), `--mipro-eval-samples` (int, default 10)
- `--pb-population` (int, default 6), `--pb-generations` (int, default 5)

`_build_calibration_optimizer_configs()` returns the appropriate Config dataclass based on `args.optimizer`.

### Documentation

Add to `docs/optimizers/`:
- `evoprompt.md` - algorithm, config, when to use, cost estimate
- `textgrad.md` - algorithm, config, when to use, cost estimate
- `miprov2.md` - algorithm, config, when to use, cost estimate
- `promptbreeder.md` - algorithm, config, when to use, cost estimate

Update `docs/optimizers/README.md` with comparison table including new optimizers.

## Testing

`tests/test_optimizers.py` (~400 lines):

For each new optimizer:
- Config dataclass defaults
- Population/candidate generation (mock LLM calls via monkeypatch on GeminiClient)
- Selection/scoring logic with known inputs
- Full optimize() with mocked LLM returning predictable JSON responses
- Edge cases: empty disagreements, single training example, early stopping triggers

For shared infrastructure:
- BaseOptimizer.split_train_val with known data
- BaseOptimizer.score_preamble with mock judge
- BaseOptimizer.build_result produces valid PromptOptimizationResult
- Factory: create_optimizer returns correct class for each name
- Factory: unknown name raises ValueError
- Factory: 'gepa' with missing library raises helpful ImportError
- call_optimizer: correctly filters kwargs for legacy optimizers

## File Summary

New files:
- `sdk/evalyn_sdk/calibration/base_optimizer.py`
- `sdk/evalyn_sdk/calibration/factory.py`
- `sdk/evalyn_sdk/calibration/evoprompt.py`
- `sdk/evalyn_sdk/calibration/textgrad.py`
- `sdk/evalyn_sdk/calibration/miprov2.py`
- `sdk/evalyn_sdk/calibration/promptbreeder.py`
- `tests/test_optimizers.py`
- `docs/optimizers/evoprompt.md`
- `docs/optimizers/textgrad.md`
- `docs/optimizers/miprov2.md`
- `docs/optimizers/promptbreeder.md`

Modified files:
- `sdk/evalyn_sdk/calibration/engine.py` (factory dispatch, generic optimizer_config)
- `sdk/evalyn_sdk/calibration/__init__.py` (export new classes)
- `sdk/evalyn_sdk/cli/commands/calibration.py` (add choices + flags + config builders)
- `docs/optimizers/README.md` (update comparison table)
- `ROADMAP.md` (mark items complete)
