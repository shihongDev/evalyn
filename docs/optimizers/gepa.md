# GEPA Optimizer

Generative Evolution of Prompts Algorithm - evolutionary prompt optimization using LLM-based reflection.

## Overview

GEPA treats prompt optimization as an evolutionary process. It maintains a population of prompts, evaluates them against the calibration dataset, and uses an LLM to reflect on failures and generate improved variants.

**Paper**: GEPA: Generative Evolution of Prompts Algorithm

## Two Implementations

| Implementation | CLI Flag | Dependencies | Token Tracking |
|----------------|----------|--------------|----------------|
| External GEPA | `--optimizer gepa` | Requires `pip install gepa` | No |
| Native GEPA | `--optimizer gepa-native` | None (built-in) | Yes |

**Recommendation**: Use `gepa-native` for most use cases - it provides full token tracking and cost reporting without external dependencies.

## Algorithm

```
┌─────────────────────────────────────────────────────────────────────┐
│                        GEPA OPTIMIZER                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    INITIALIZATION                             │  │
│  │                                                               │  │
│  │  1. Create seed prompt from current rubric                    │  │
│  │  2. Build calibration dataset from annotations                │  │
│  │  3. Split into train/validation sets                         │  │
│  └───────────────────────────┬──────────────────────────────────┘  │
│                              │                                      │
│                              ▼                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                  EVOLUTIONARY LOOP                            │  │
│  │                                                               │  │
│  │    ┌─────────────┐                                            │  │
│  │    │  Population │ (prompts)                                  │  │
│  │    └──────┬──────┘                                            │  │
│  │           │                                                   │  │
│  │           ▼                                                   │  │
│  │    ┌─────────────┐     ┌──────────────────────────────┐      │  │
│  │    │  Evaluate   │────▶│ Score each prompt against    │      │  │
│  │    │  (Task LM)  │     │ train set using Task LM      │      │  │
│  │    └─────────────┘     └──────────────────────────────┘      │  │
│  │           │                                                   │  │
│  │           ▼                                                   │  │
│  │    ┌─────────────┐     ┌──────────────────────────────┐      │  │
│  │    │   Reflect   │────▶│ Strong LLM analyzes failures │      │  │
│  │    │(Reflect LM) │     │ and suggests improvements    │      │  │
│  │    └─────────────┘     └──────────────────────────────┘      │  │
│  │           │                                                   │  │
│  │           ▼                                                   │  │
│  │    ┌─────────────┐     ┌──────────────────────────────┐      │  │
│  │    │   Mutate    │────▶│ Generate new prompt variants │      │  │
│  │    │             │     │ based on reflections         │      │  │
│  │    └─────────────┘     └──────────────────────────────┘      │  │
│  │           │                                                   │  │
│  │           ▼                                                   │  │
│  │    ┌─────────────┐                                            │  │
│  │    │   Select    │ (keep best, discard worst)                │  │
│  │    └──────┬──────┘                                            │  │
│  │           │                                                   │  │
│  │           └──────────▶ Repeat until budget exhausted         │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│                              ▼                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                     VALIDATION                                │  │
│  │                                                               │  │
│  │  Final prompt evaluated on held-out validation set            │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## How It Uses the Metric

```
┌────────────────────────────────────────────────────────────────────┐
│                    METRIC COMPONENTS                               │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  System Prompt ───────▶ Combined with rubric into full prompt     │
│    "Evaluate the        that GEPA optimizes as a whole            │
│     response..."                                                   │
│                              │                                     │
│                              ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐     │
│  │              FULL PROMPT (optimized by GEPA)             │     │
│  │                                                          │     │
│  │  {system_prompt}                                         │     │
│  │                                                          │     │
│  │  RUBRIC:                                                 │     │
│  │  {rubric_items}                                          │     │
│  │                                                          │     │
│  │  INPUT: {input}                                          │     │
│  │  OUTPUT: {output}                                        │     │
│  │                                                          │     │
│  │  Evaluate and return PASS or FAIL.                       │     │
│  └──────────────────────────────────────────────────────────┘     │
│                                                                    │
│  Rubric ──────────────▶ OPTIMIZED (embedded in full prompt)       │
│                                                                    │
│  Threshold ───────────▶ Not used (GEPA optimizes for binary       │
│                          PASS/FAIL accuracy)                       │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

## Configuration

### GEPA-Native (Recommended)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--gepa-native-task-model` | gemini-2.5-flash | Model for evaluation (judge) |
| `--gepa-native-reflection-model` | gemini-2.5-flash | Model for generating mutations |
| `--gepa-native-max-calls` | 150 | Budget for metric evaluations |
| `--gepa-native-initial-candidates` | 5 | Number of initial candidate prompts |
| `--gepa-native-batch-size` | 5 | Mini-batch size for feedback |

**Config Class**: `GEPANativeConfig`
```python
@dataclass
class GEPANativeConfig:
    task_model: str = "gemini-2.5-flash"
    reflection_model: str = "gemini-2.5-flash"
    max_metric_calls: int = 150
    num_initial_candidates: int = 5
    mini_batch_size: int = 5
    pareto_set_size: int = 10
    exploit_prob: float = 0.9
    train_split: float = 0.7
```

### External GEPA (Legacy)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--gepa-task-lm` | gemini-2.5-flash | Model being optimized (judge model) |
| `--gepa-reflection-lm` | gemini-2.5-flash | Strong model for reflection |
| `--gepa-max-calls` | 150 | Budget for metric evaluations |

**Config Class**: `GEPAConfig`
```python
@dataclass
class GEPAConfig:
    task_lm: str = "gemini/gemini-2.5-flash"
    reflection_lm: str = "gemini/gemini-2.5-flash"
    max_metric_calls: int = 150
    train_split: float = 0.7
```

## Cost & Performance

| Config | Estimated Tokens | Estimated Cost | LLM Calls |
|--------|------------------|----------------|-----------|
| 150 calls | ~1-2M | ~$0.50-1.00 | ~150+ |

Formula: `calls = max_metric_calls + reflection_calls`

**Token Tracking**:
- `gepa-native`: Full token tracking with cost breakdown displayed after calibration
- `gepa` (external): No token tracking (external library is a black box)

## Important Notes

**Suggestions only**: Like all optimizers, GEPA does NOT automatically apply changes. It:
1. Evolves a population of candidate prompts
2. Uses reflection to improve failing prompts
3. Returns the best-performing prompt in `prompts/*.txt` files

You must manually review and apply the suggested changes.

## When to Use

**Best for:**
- Complex rubrics needing significant restructuring
- When you want diverse prompt variations explored
- Larger annotation datasets (> 50 samples)
- When you can afford higher API costs

**Not ideal for:**
- Quick iterations
- Small datasets
- Cost-sensitive scenarios

## Example

```bash
# GEPA-Native optimization (recommended - includes token tracking)
evalyn calibrate --optimizer gepa-native \
  --metric-id helpfulness \
  --annotations annotations.jsonl \
  --dataset data/my-dataset

# Custom models and budget (gepa-native)
evalyn calibrate --optimizer gepa-native \
  --metric-id helpfulness \
  --annotations annotations.jsonl \
  --dataset data/my-dataset \
  --gepa-native-task-model gemini-2.5-flash-lite \
  --gepa-native-reflection-model gemini-2.5-flash \
  --gepa-native-max-calls 100 \
  --gepa-native-initial-candidates 3

# External GEPA (requires: pip install gepa)
evalyn calibrate --optimizer gepa \
  --metric-id helpfulness \
  --annotations annotations.jsonl \
  --dataset data/my-dataset \
  --gepa-task-lm gemini-2.5-flash-lite \
  --gepa-reflection-lm gemini-2.5-flash \
  --gepa-max-calls 100
```

## Output

```
--- GEPA OPTIMIZATION ---
Task LM: gemini-2.5-flash-lite
Reflection LM: gemini-2.5-flash
Budget: 150 metric calls

Generation 1: Best accuracy 72%
Generation 2: Best accuracy 78%
Generation 3: Best accuracy 82%
...
Final: Best accuracy 85%

Validation accuracy: 83%
```

## See Also

- [LLM Optimizer](llm.md) - Simpler single-shot approach
- [OPRO Optimizer](opro.md) - Trajectory-based optimization
- [APE Optimizer](ape.md) - UCB-based search optimization
