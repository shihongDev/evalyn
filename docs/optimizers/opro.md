# OPRO Optimizer

Optimization by PROmpting - trajectory-based prompt optimization that learns from past solutions.

## Overview

OPRO maintains a trajectory of previously tried prompts and their scores. At each iteration, it shows the LLM this history and asks it to generate improved candidates. The best candidates are evaluated and added to the trajectory, creating a feedback loop that converges toward optimal prompts.

**Paper**: [Large Language Models as Optimizers](https://arxiv.org/abs/2309.03409) (Yang et al., 2023)

## Algorithm

```
┌─────────────────────────────────────────────────────────────────────┐
│                        OPRO OPTIMIZER                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    INITIALIZATION                             │  │
│  │                                                               │  │
│  │  1. Build dataset from annotations (train/val split)         │  │
│  │  2. Create seed prompt from current rubric                    │  │
│  │  3. Evaluate seed prompt, add to trajectory                   │  │
│  │  4. Initialize trajectory = [(seed_prompt, score)]            │  │
│  └───────────────────────────┬──────────────────────────────────┘  │
│                              │                                      │
│                              ▼                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                  OPTIMIZATION LOOP                            │  │
│  │                     (max_iterations)                          │  │
│  │                                                               │  │
│  │   ┌──────────────────────────────────────────────────────┐   │  │
│  │   │                   META-PROMPT                         │   │  │
│  │   │                                                       │   │  │
│  │   │  "Here are previous prompts and their scores:        │   │  │
│  │   │                                                       │   │  │
│  │   │   Prompt: {prompt_1}                                  │   │  │
│  │   │   Score: 72%                                          │   │  │
│  │   │                                                       │   │  │
│  │   │   Prompt: {prompt_2}                                  │   │  │
│  │   │   Score: 78%                                          │   │  │
│  │   │                                                       │   │  │
│  │   │   ... (up to trajectory_length examples)              │   │  │
│  │   │                                                       │   │  │
│  │   │  Generate {candidates_per_step} new prompts that      │   │  │
│  │   │  might achieve higher scores."                        │   │  │
│  │   └───────────────────────────┬──────────────────────────┘   │  │
│  │                               │                               │  │
│  │                               ▼                               │  │
│  │                    ┌───────────────────┐                      │  │
│  │                    │  Optimizer LLM    │                      │  │
│  │                    │  (generate new    │                      │  │
│  │                    │   candidates)     │                      │  │
│  │                    └─────────┬─────────┘                      │  │
│  │                              │                                │  │
│  │                              ▼                                │  │
│  │            ┌─────────────────────────────────┐                │  │
│  │            │     New Candidate Prompts       │                │  │
│  │            │  [candidate_1, candidate_2, ...]│                │  │
│  │            └────────────────┬────────────────┘                │  │
│  │                             │                                 │  │
│  │                             ▼                                 │  │
│  │                 For each candidate:                           │  │
│  │                    ┌───────────────────┐                      │  │
│  │                    │   Scorer LLM      │                      │  │
│  │                    │  (evaluate on     │                      │  │
│  │                    │   train set)      │                      │  │
│  │                    └─────────┬─────────┘                      │  │
│  │                              │                                │  │
│  │                              ▼                                │  │
│  │            ┌─────────────────────────────────┐                │  │
│  │            │  Add (prompt, score) to         │                │  │
│  │            │  trajectory, keep best          │                │  │
│  │            └─────────────────────────────────┘                │  │
│  │                              │                                │  │
│  │         ┌────────────────────┴────────────────────┐          │  │
│  │         │                                         │          │  │
│  │         ▼                                         ▼          │  │
│  │    No improvement for              Best score improved       │  │
│  │    {early_stop_patience}                  │                  │  │
│  │    iterations?                            │                  │  │
│  │         │                                 │                  │  │
│  │         ▼                                 ▼                  │  │
│  │    Early stop                    Continue loop               │  │
│  │                                                              │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│                              ▼                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                     VALIDATION                                │  │
│  │                                                               │  │
│  │  Best prompt from trajectory evaluated on validation set      │  │
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
│  System Prompt ───────▶ Combined with rubric into evaluation      │
│    "Evaluate the        prompt that OPRO optimizes                │
│     response..."                                                   │
│                              │                                     │
│                              ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐     │
│  │        FULL PROMPT (what OPRO generates/optimizes)       │     │
│  │                                                          │     │
│  │  You are evaluating responses for {metric_id}.          │     │
│  │                                                          │     │
│  │  CRITERIA:                                               │     │
│  │  {rubric - this is what gets optimized}                 │     │
│  │                                                          │     │
│  │  INPUT: {input}                                          │     │
│  │  OUTPUT: {output}                                        │     │
│  │                                                          │     │
│  │  Return JSON: {"passed": true/false, "reason": "..."}   │     │
│  └──────────────────────────────────────────────────────────┘     │
│                                                                    │
│  Rubric ──────────────▶ OPTIMIZED                                 │
│    - "criterion 1"      OPRO iteratively improves the rubric      │
│    - "criterion 2"      by learning from trajectory of attempts   │
│                                                                    │
│  Threshold ───────────▶ Not used during optimization              │
│                          (binary PASS/FAIL comparison)             │
│                                                                    │
│  Scoring: Agreement with human annotations                        │
│    score = (correct predictions) / (total samples)                │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--opro-iterations` | 10 | Max optimization iterations |
| `--opro-candidates` | 4 | Candidate prompts per iteration |
| `--opro-optimizer-model` | gemini-2.5-flash | Model for generating candidates |
| `--opro-scorer-model` | gemini-2.5-flash-lite | Model for scoring (judge) |

**Config Class**: `OPROConfig`
```python
@dataclass
class OPROConfig:
    optimizer_model: str = "gemini-2.5-flash"
    scorer_model: str = "gemini-2.5-flash-lite"
    max_iterations: int = 10
    candidates_per_step: int = 4
    trajectory_length: int = 20
    train_split: float = 0.7
    temperature: float = 0.7
    early_stop_patience: int = 3
    timeout: int = 120
```

## When to Use

**Best for:**
- Iterative refinement of existing prompts
- Finding local optima through gradient-like search
- Medium-sized datasets (20-100 samples)
- When you want to leverage past optimization attempts

**Not ideal for:**
- Very small datasets (trajectory provides little signal)
- When you need completely new prompt structures

## Example

```bash
# Basic OPRO optimization
evalyn calibrate --optimizer opro \
  --metric-id helpfulness \
  --annotations annotations.jsonl \
  --dataset data/my-dataset

# Custom configuration
evalyn calibrate --optimizer opro \
  --metric-id helpfulness \
  --annotations annotations.jsonl \
  --dataset data/my-dataset \
  --opro-iterations 15 \
  --opro-candidates 6 \
  --opro-optimizer-model gemini-2.5-flash \
  --opro-scorer-model gemini-2.5-flash-lite
```

## Output

```
--- OPRO OPTIMIZATION ---
Optimizer: gemini-2.5-flash
Scorer: gemini-2.5-flash-lite
Max iterations: 10

Iteration 1: Generated 4 candidates
  Best candidate score: 74% (improved from 70%)
Iteration 2: Generated 4 candidates
  Best candidate score: 78% (improved from 74%)
Iteration 3: Generated 4 candidates
  Best candidate score: 78% (no improvement)
...
Early stopping at iteration 6 (no improvement for 3 iterations)

Best training accuracy: 82%
Validation accuracy: 79%
```

## See Also

- [LLM Optimizer](llm.md) - Simpler single-shot approach
- [APE Optimizer](ape.md) - UCB-based search optimization
- [GEPA Optimizer](gepa.md) - Evolutionary optimization
