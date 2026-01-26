# Prompt Optimizers

Evalyn provides multiple prompt optimization algorithms for calibrating LLM judges. Each optimizer analyzes disagreements between human annotations and judge results, then suggests improvements to the metric's rubric.

## Overview

```
                      ┌─────────────────┐
                      │   Human Labels  │
                      │  (ground truth) │
                      └────────┬────────┘
                               │
                               ▼
┌─────────────┐      ┌─────────────────┐      ┌──────────────┐
│   Dataset   │─────▶│   Calibration   │◀─────│ Judge Results│
│  (samples)  │      │     Engine      │      │   (metric)   │
└─────────────┘      └────────┬────────┘      └──────────────┘
                               │
                    ┌──────────┴──────────┐
                    │  Compute Alignment  │
                    │  Analyze Disagrees  │
                    └──────────┬──────────┘
                               │
                               ▼
              ┌────────────────────────────────┐
              │         OPTIMIZER              │
              │  ┌────┐ ┌────┐ ┌────┐ ┌────┐  │
              │  │LLM │ │GEPA│ │OPRO│ │APE │  │
              │  └────┘ └────┘ └────┘ └────┘  │
              └────────────────┬───────────────┘
                               │
                               ▼
                    ┌──────────────────┐
                    │ Optimized Rubric │
                    └──────────────────┘
```

## Comparison

| Optimizer | Approach | Cost | Token Tracking | Best For |
|-----------|----------|------|----------------|----------|
| [LLM](llm.md) | Single-shot analysis | Low (1 API call) | Yes | Quick iterations, small disagreement sets |
| [GEPA-Native](gepa.md) | Pareto-based evolution | Medium-High | Yes | Complex rubrics, when diversity matters |
| [OPRO](opro.md) | Trajectory-based search | Medium | Yes | Iterative refinement, finding local optima |
| [APE](ape.md) | UCB bandit search | Medium | Yes | Exploration vs exploitation tradeoff |
| GEPA (external) | Evolutionary reflection | Medium-High | No | Legacy - use gepa-native instead |

## Cost & Performance

Based on benchmarks with 100 samples using gemini-2.5-flash:

| Optimizer | Tokens | Cost | LLM Calls |
|-----------|--------|------|-----------|
| LLM | ~2-5k | ~$0.002 | 1 |
| OPRO (5 iter, 3 cand) | ~550k | ~$0.30 | ~50-100 |
| APE (5 cand, 3 rounds) | ~800k | ~$0.18 | ~50-80 |

Note: GEPA requires external dependencies and is not benchmarked here.

**Token usage scaling:**
- LLM: Fixed cost, regardless of dataset size
- OPRO: `O(iterations x candidates x (samples + context))`
- APE: `O(candidates x rounds x samples)

## Metric Structure

All optimizers work with the same metric structure:

```
┌─────────────────────────────────────────────────────────────┐
│                         METRIC                              │
├─────────────────────────────────────────────────────────────┤
│  System Prompt:  "Evaluate the response for helpfulness..." │
│                                                             │
│  Rubric:                                                    │
│    - "Response addresses the user's question"               │
│    - "Response is accurate and factual"                     │
│    - "Response provides sufficient detail"                  │
│                                                             │
│  Threshold: 0.5 (pass if score >= threshold)               │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
                    ┌──────────────────┐
                    │   LLM Judge      │
                    │   (evaluates     │
                    │    responses)    │
                    └──────────────────┘
```

## Usage

```bash
# LLM optimizer (default)
evalyn calibrate --metric-id helpfulness --annotations data/annotations.jsonl

# GEPA-Native optimizer (recommended for GEPA - includes token tracking)
evalyn calibrate --optimizer gepa-native --metric-id helpfulness --annotations data/annotations.jsonl

# OPRO optimizer
evalyn calibrate --optimizer opro --metric-id helpfulness --annotations data/annotations.jsonl

# APE optimizer
evalyn calibrate --optimizer ape --metric-id helpfulness --annotations data/annotations.jsonl

# GEPA external (legacy - requires: pip install gepa)
evalyn calibrate --optimizer gepa --metric-id helpfulness --annotations data/annotations.jsonl
```

## See Also

- [calibrate CLI](../clis/calibrate.md) - Full CLI reference
- [annotate CLI](../clis/annotate.md) - Create annotations for calibration
