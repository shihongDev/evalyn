# Calibration Guide

## Overview

Calibration aligns LLM judges with human judgment by:
1. Comparing LLM predictions vs human annotations
2. Computing alignment metrics
3. Optimizing prompts to improve alignment
4. Validating that optimized prompts are actually better

## Workflow

```
1. Run evaluation       evalyn run-eval --latest
2. Annotate results     evalyn annotate --latest --per-metric
3. Calibrate            evalyn calibrate --metric-id X --annotations ann.jsonl
4. Re-evaluate          evalyn run-eval --latest --use-calibrated
```

## Annotation

### Overall Mode
```bash
evalyn annotate --latest
```
Rate each item as pass/fail with confidence (1-5).

### Per-Metric Mode (Recommended)
```bash
evalyn annotate --latest --per-metric
```
Agree/disagree with each LLM judge separately.

**Output:** `annotations.jsonl`
```json
{
  "item_id": "abc123",
  "human_label": "pass",
  "confidence": 4,
  "metric_labels": {
    "helpfulness_accuracy": {
      "agree_with_llm": true,
      "human_label": "pass"
    }
  }
}
```

## Calibrate Command

```bash
evalyn calibrate --metric-id helpfulness_accuracy \
  --annotations annotations.jsonl \
  --dataset data/myapp
```

**Options:**
| Flag | Default | Description |
|------|---------|-------------|
| `--optimizer` | llm | Optimization method: `llm` or `gepa` |
| `--no-optimize` | false | Only compute alignment metrics |
| `--show-examples` | false | Show disagreement examples |

## Alignment Metrics

| Metric | Description |
|--------|-------------|
| Accuracy | Overall agreement rate |
| Precision | Of LLM "pass", how many correct? |
| Recall | Of true "pass", how many caught? |
| F1 Score | Harmonic mean of precision/recall |
| Cohen's Kappa | Agreement adjusted for chance |

## Prompt Optimization

### LLM Optimizer (Default)
Analyzes disagreements and suggests prompt improvements.

```bash
evalyn calibrate --metric-id X --annotations ann.jsonl --optimizer llm
```

### GEPA Optimizer
Evolutionary search with LLM reflection. More thorough but slower.

```bash
evalyn calibrate --metric-id X --annotations ann.jsonl --optimizer gepa
```

## Validation

Calibration automatically validates optimized prompts:

1. Splits data 70% train / 30% validation
2. Runs both prompts on validation set
3. Compares F1 scores
4. Only recommends if >2% improvement

**Output:**
```
--- VALIDATION RESULTS ---
✅ SUCCESS - Optimized prompt is BETTER
  Original F1:     0.850
  Optimized F1:    0.920
  Improvement:     +7.0%

💡 RECOMMENDATION: USE OPTIMIZED PROMPT
```

If degraded:
```
❌ DEGRADED - Optimized prompt is WORSE
⚠️  RECOMMENDATION: KEEP ORIGINAL PROMPT
```

## Using Calibrated Prompts

```bash
evalyn run-eval --latest --use-calibrated
```

This loads optimized prompts from `calibrations/<metric_id>/prompts/`.

## Output Structure

```
calibrations/
  helpfulness_accuracy/
    calibration.json      # Full results
    prompts/
      <timestamp>_original.txt
      <timestamp>_optimized.txt
```

## Best Practices

1. **Annotate 20-50 items** - Enough for reliable calibration
2. **Use per-metric mode** - More granular feedback
3. **Check validation results** - Don't use degraded prompts
4. **Re-calibrate periodically** - As your agent evolves
