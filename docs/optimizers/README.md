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

| Optimizer | Approach | Cost | Best For |
|-----------|----------|------|----------|
| [LLM](llm.md) | Single-shot analysis | Low (1 API call) | Quick iterations, small disagreement sets |
| [GEPA](gepa.md) | Evolutionary reflection | Medium-High | Complex rubrics, when diversity matters |
| [OPRO](opro.md) | Trajectory-based search | Medium | Iterative refinement, finding local optima |
| [APE](ape.md) | UCB bandit search | Medium | Exploration vs exploitation tradeoff |

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

# GEPA optimizer
evalyn calibrate --optimizer gepa --metric-id helpfulness --annotations data/annotations.jsonl

# OPRO optimizer
evalyn calibrate --optimizer opro --metric-id helpfulness --annotations data/annotations.jsonl

# APE optimizer
evalyn calibrate --optimizer ape --metric-id helpfulness --annotations data/annotations.jsonl
```

## See Also

- [calibrate CLI](../clis/calibrate.md) - Full CLI reference
- [annotate CLI](../clis/annotate.md) - Create annotations for calibration
