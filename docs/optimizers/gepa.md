# GEPA Optimizer

Generative Evolution of Prompts Algorithm - evolutionary prompt optimization using LLM-based reflection.

## Overview

GEPA treats prompt optimization as an evolutionary process. It maintains a population of prompts, evaluates them against the calibration dataset, and uses an LLM to reflect on failures and generate improved variants.

**Paper**: GEPA: Generative Evolution of Prompts Algorithm

**Dependency**: Requires the `gepa` package (`pip install gepa`)

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
# Basic GEPA optimization
evalyn calibrate --optimizer gepa \
  --metric-id helpfulness \
  --annotations annotations.jsonl \
  --dataset data/my-dataset

# Custom models and budget
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
