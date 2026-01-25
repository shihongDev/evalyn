# APE Optimizer

Automatic Prompt Engineer - search-based optimization using Upper Confidence Bound (UCB) selection.

## Overview

APE treats prompt optimization as a multi-armed bandit problem. It first generates a pool of candidate prompts, then uses UCB to balance exploration (trying uncertain candidates) and exploitation (refining promising ones). This approach efficiently finds good prompts without exhaustively evaluating all candidates.

**Paper**: [Large Language Models Are Human-Level Prompt Engineers](https://arxiv.org/abs/2211.01910) (Zhou et al., 2022)

## Algorithm

```
┌─────────────────────────────────────────────────────────────────────┐
│                        APE OPTIMIZER                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │               PHASE 1: CANDIDATE GENERATION                   │  │
│  │                                                               │  │
│  │   ┌──────────────────────────────────────────────────────┐   │  │
│  │   │              PROPOSAL PROMPT                          │   │  │
│  │   │                                                       │   │  │
│  │   │  "Generate {num_candidates} diverse evaluation       │   │  │
│  │   │   criteria for judging {metric_id}.                  │   │  │
│  │   │                                                       │   │  │
│  │   │   Here are example (input, output, label) tuples:    │   │  │
│  │   │     Input: {input_1}, Output: {output_1}, Pass: Yes  │   │  │
│  │   │     Input: {input_2}, Output: {output_2}, Pass: No   │   │  │
│  │   │     ...                                               │   │  │
│  │   │                                                       │   │  │
│  │   │   Generate varied criteria that could distinguish    │   │  │
│  │   │   good outputs from bad ones."                       │   │  │
│  │   └───────────────────────────┬──────────────────────────┘   │  │
│  │                               │                               │  │
│  │                               ▼                               │  │
│  │                    ┌───────────────────┐                      │  │
│  │                    │   Generator LLM   │                      │  │
│  │                    └─────────┬─────────┘                      │  │
│  │                              │                                │  │
│  │                              ▼                                │  │
│  │            ┌─────────────────────────────────┐                │  │
│  │            │   Candidate Pool (N prompts)    │                │  │
│  │            │   [c1, c2, c3, ... cN]          │                │  │
│  │            │   Each with: mean=0, count=0    │                │  │
│  │            └─────────────────────────────────┘                │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│                              ▼                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │               PHASE 2: UCB EVALUATION                         │  │
│  │                        (eval_rounds)                          │  │
│  │                                                               │  │
│  │   For each round:                                             │  │
│  │                                                               │  │
│  │   ┌──────────────────────────────────────────────────────┐   │  │
│  │   │                UCB SELECTION                          │   │  │
│  │   │                                                       │   │  │
│  │   │  UCB(candidate) = mean_score + c * sqrt(ln(N) / n)   │   │  │
│  │   │                                                       │   │  │
│  │   │  where:                                               │   │  │
│  │   │    mean_score = average accuracy so far               │   │  │
│  │   │    c = exploration_weight (default 1.0)              │   │  │
│  │   │    N = total evaluations across all candidates       │   │  │
│  │   │    n = evaluations for this candidate                │   │  │
│  │   │                                                       │   │  │
│  │   │  Select candidate with highest UCB score             │   │  │
│  │   └───────────────────────────┬──────────────────────────┘   │  │
│  │                               │                               │  │
│  │                               ▼                               │  │
│  │   ┌──────────────────────────────────────────────────────┐   │  │
│  │   │              EVALUATE SELECTED                        │   │  │
│  │   │                                                       │   │  │
│  │   │  Sample {eval_samples_per_round} items from train    │   │  │
│  │   │  Evaluate candidate prompt using Scorer LLM          │   │  │
│  │   │  Update candidate's mean_score and count             │   │  │
│  │   └───────────────────────────┬──────────────────────────┘   │  │
│  │                               │                               │  │
│  │                               ▼                               │  │
│  │                    Repeat for eval_rounds                     │  │
│  │                                                               │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│                              ▼                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                     VALIDATION                                │  │
│  │                                                               │  │
│  │  Best candidate (highest mean) evaluated on validation set    │  │
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
│  Current Rubric ──────▶ Used as context for candidate generation  │
│    - "criterion 1"      The generator sees the current rubric     │
│    - "criterion 2"      and example data to generate alternatives │
│                                                                    │
│                              │                                     │
│                              ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐     │
│  │      CANDIDATE PROMPTS (generated by APE)                │     │
│  │                                                          │     │
│  │  Candidate 1: "Evaluate based on accuracy and clarity"  │     │
│  │  Candidate 2: "Check for completeness and relevance"    │     │
│  │  Candidate 3: "Assess helpfulness and detail level"     │     │
│  │  ...                                                     │     │
│  └──────────────────────────────────────────────────────────┘     │
│                                                                    │
│  System Prompt ───────▶ Combined with each candidate for scoring  │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────┐     │
│  │          SCORING PROMPT (for each candidate)             │     │
│  │                                                          │     │
│  │  {system_prompt}                                         │     │
│  │                                                          │     │
│  │  CRITERIA:                                               │     │
│  │  {candidate_rubric}                                      │     │
│  │                                                          │     │
│  │  INPUT: {input}                                          │     │
│  │  OUTPUT: {output}                                        │     │
│  │                                                          │     │
│  │  Return JSON: {"passed": true/false, "reason": "..."}   │     │
│  └──────────────────────────────────────────────────────────┘     │
│                                                                    │
│  Threshold ───────────▶ Not used during optimization              │
│                          (binary PASS/FAIL comparison)             │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

## UCB Intuition

```
┌─────────────────────────────────────────────────────────────────┐
│                    UCB SELECTION                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Candidate A: mean=80%, evaluated 10 times                      │
│  Candidate B: mean=75%, evaluated 2 times                       │
│  Candidate C: mean=82%, evaluated 8 times                       │
│                                                                 │
│  UCB scores (with c=1.0):                                       │
│    A: 80% + 1.0 * sqrt(ln(20)/10) = 80% + 0.55 = 80.55%        │
│    B: 75% + 1.0 * sqrt(ln(20)/2)  = 75% + 1.22 = 76.22%  <--   │
│    C: 82% + 1.0 * sqrt(ln(20)/8)  = 82% + 0.61 = 82.61%  <--   │
│                                                                 │
│  C has highest UCB: exploit its high mean                       │
│  B has high uncertainty bonus: explore it more                  │
│                                                                 │
│  This balance prevents:                                         │
│    - Premature convergence on suboptimal prompts               │
│    - Wasting evaluations on clearly bad prompts                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--ape-candidates` | 10 | Number of candidate prompts to generate |
| `--ape-rounds` | 5 | UCB evaluation rounds |
| `--ape-samples` | 5 | Samples per candidate per round |

**Config Class**: `APEConfig`
```python
@dataclass
class APEConfig:
    num_candidates: int = 10
    eval_rounds: int = 5
    eval_samples_per_round: int = 5
    generator_model: str = "gemini-2.5-flash"
    scorer_model: str = "gemini-2.5-flash-lite"
    exploration_weight: float = 1.0
    train_split: float = 0.7
    temperature: float = 0.8
    timeout: int = 120
```

## When to Use

**Best for:**
- Exploring diverse prompt variations efficiently
- When you have enough data to evaluate multiple candidates
- Balancing exploration and exploitation automatically
- Medium to large datasets (30+ samples)

**Not ideal for:**
- Very small datasets (UCB needs multiple evaluations per candidate)
- When you want iterative refinement (use OPRO instead)

## Example

```bash
# Basic APE optimization
evalyn calibrate --optimizer ape \
  --metric-id helpfulness \
  --annotations annotations.jsonl \
  --dataset data/my-dataset

# Custom configuration
evalyn calibrate --optimizer ape \
  --metric-id helpfulness \
  --annotations annotations.jsonl \
  --dataset data/my-dataset \
  --ape-candidates 15 \
  --ape-rounds 8 \
  --ape-samples 10
```

## Output

```
--- APE OPTIMIZATION ---
Generating 10 candidate prompts...
Running 5 UCB evaluation rounds (5 samples each)...

Round 1:
  Selected candidate 3 (UCB: inf, mean: n/a)
  Accuracy: 72%
Round 2:
  Selected candidate 7 (UCB: inf, mean: n/a)
  Accuracy: 68%
...
Round 5:
  Selected candidate 3 (UCB: 82.3%, mean: 80%)
  Accuracy: 84%

Best candidate (#3): mean accuracy 80%
Validation accuracy: 78%
```

## See Also

- [LLM Optimizer](llm.md) - Simpler single-shot approach
- [OPRO Optimizer](opro.md) - Trajectory-based optimization
- [GEPA Optimizer](gepa.md) - Evolutionary optimization
