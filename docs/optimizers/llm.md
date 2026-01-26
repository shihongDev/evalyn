# LLM Optimizer

Single-shot prompt optimization using an LLM to analyze disagreement patterns.

## Overview

The LLM optimizer is the default and simplest approach. It sends all disagreement examples to an LLM in a single prompt, asking it to analyze patterns and suggest rubric improvements.

**Paper**: N/A (standard prompt engineering approach)

## Algorithm

```
┌─────────────────────────────────────────────────────────────────┐
│                      LLM OPTIMIZER                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input:                                                         │
│    - Current rubric                                             │
│    - False positives (judge PASS, human FAIL)                   │
│    - False negatives (judge FAIL, human PASS)                   │
│    - Alignment statistics (accuracy, precision, recall, F1)    │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  OPTIMIZATION PROMPT                     │   │
│  │                                                          │   │
│  │  "You are an expert at improving LLM evaluation rubrics. │   │
│  │   Analyze the following calibration data..."             │   │
│  │                                                          │   │
│  │  Current Rubric:                                         │   │
│  │    - criterion 1                                         │   │
│  │    - criterion 2                                         │   │
│  │                                                          │   │
│  │  False Positives (judge too lenient):                    │   │
│  │    Example 1: input, output, judge_reason, human_notes   │   │
│  │    Example 2: ...                                        │   │
│  │                                                          │   │
│  │  False Negatives (judge too strict):                     │   │
│  │    Example 1: input, output, judge_reason, human_notes   │   │
│  │    Example 2: ...                                        │   │
│  │                                                          │   │
│  │  Return JSON with improved_rubric, reasoning, etc.       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│                    ┌─────────────┐                              │
│                    │   LLM       │                              │
│                    │  (1 call)   │                              │
│                    └──────┬──────┘                              │
│                           │                                     │
│                           ▼                                     │
│  Output:                                                        │
│    - improved_rubric: ["new criterion 1", "new criterion 2"]   │
│    - suggested_additions: ["criteria to add"]                  │
│    - suggested_removals: ["criteria to remove"]                │
│    - improvement_reasoning: "explanation"                      │
│    - estimated_improvement: "low|medium|high"                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## How It Uses the Metric

```
┌────────────────────────────────────────────────────────────────┐
│                    METRIC COMPONENTS                           │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  System Prompt ────────▶ Not directly used by optimizer       │
│                          (only the rubric is optimized)       │
│                                                                │
│  Rubric ───────────────▶ OPTIMIZED                            │
│    - "criterion 1"       The optimizer analyzes why the       │
│    - "criterion 2"       rubric leads to disagreements and    │
│    - "criterion 3"       suggests improvements                │
│                                                                │
│  Threshold ────────────▶ Separately suggested                 │
│                          Calibration engine suggests optimal  │
│                          threshold based on alignment metrics │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | gemini-2.5-flash-lite | LLM model for optimization |

## Cost & Performance

| Metric | Value |
|--------|-------|
| API Calls | 1 |
| Tokens | ~2-5k |
| Cost | ~$0.002 (with Gemini Flash) |

The LLM optimizer is the cheapest option since it makes a single API call regardless of dataset size.

## Important Notes

**Suggestions only**: The optimizer does NOT automatically apply changes to your rubric. It:
1. Analyzes disagreement patterns
2. Suggests improvements (saved to `prompts/*.txt` files)
3. Returns reasoning for why changes would help

You must manually review and apply the suggested changes to your metric configuration.

## When to Use

**Best for:**
- Quick iterations during development
- Small datasets (< 50 disagreements)
- When you want human-readable reasoning
- Cost-sensitive scenarios (single API call)

**Not ideal for:**
- Large-scale optimization
- When you need to validate improvements empirically

## Example

```bash
# Basic usage
evalyn calibrate --metric-id helpfulness --annotations annotations.jsonl

# With specific model
evalyn calibrate --metric-id helpfulness --annotations annotations.jsonl --model gemini-2.5-flash

# Metrics only (no optimization)
evalyn calibrate --metric-id helpfulness --annotations annotations.jsonl --no-optimize
```

## Output

```
--- PROMPT OPTIMIZATION ---
Analyzing 8 disagreement cases...

Suggested rubric improvements:
1. Be more lenient on partial answers that address the main question
2. Consider context relevance, not just factual accuracy
3. Penalize responses that are correct but off-topic

Improvement reasoning:
The false negatives suggest the judge is too strict on brevity, while
false positives indicate insufficient attention to relevance.

Estimated improvement: medium
```

## See Also

- [OPRO Optimizer](opro.md) - Iterative trajectory-based optimization
- [APE Optimizer](ape.md) - UCB-based search optimization
- [GEPA Optimizer](gepa.md) - Evolutionary optimization
