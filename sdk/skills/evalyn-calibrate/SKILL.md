---
name: evalyn-calibrate
description: Use when LLM judges need calibration, evaluation metrics seem misaligned with expectations, or annotation and judge tuning is needed
---

# evalyn-calibrate

## Overview

Guide annotation of evaluation results and calibrate LLM judges to align with human expectations. This is the feedback loop that makes evaluation reliable.

## Pre-flight

1. Verify evaluation runs exist:

```bash
evalyn list-runs --limit 1
```

If no runs: "You need evaluation results first. Invoke `evalyn-eval`."

2. Identify calibration targets:

```bash
evalyn analyze --run <latest-run-id>
```

Look for subjective metrics (type `[llm]`) with pass rates below 90%. Only subjective metrics can be calibrated - objective metrics are deterministic.

## Step 1: Annotate Results

Run interactive annotation with per-metric mode for targeted feedback:

```bash
evalyn annotate --run <latest-run-id> --per-metric
```

This is an interactive terminal session. The user will:
- See each item's input, output, and LLM judge results
- Agree or disagree with each metric's judgment
- Commands: `[y]es/pass`, `[n]o/fail`, `[s]kip`, `[v]iew full`, `[q]uit`

Guidelines for the user:
- Annotate at least 20-30 items for reliable calibration
- Focus on items where you disagree with the judge
- Be consistent - the calibrator learns from your patterns
- Each annotation saves immediately, so you can quit and resume later

## Step 2: Run Calibration

Start with the basic optimizer (fast, single-shot analysis):

```bash
evalyn calibrate --metric-id <target-metric>
```

The `calibrate` command auto-discovers annotations from the dataset directory.

The output shows alignment metrics:
- **Accuracy**: overall agreement rate
- **F1 Score**: balanced precision/recall
- **Cohen's Kappa**: agreement adjusted for chance

### Optimizer comparison

| Optimizer | Flag | Speed | Best For |
|-----------|------|-------|----------|
| `basic` | `--optimizer basic` | Fast (1 API call) | First pass, small annotation sets |
| `ape` | `--optimizer ape` | Medium | Exploring many prompt variants |
| `opro` | `--optimizer opro` | Medium | Iterative refinement |
| `gepa` | `--optimizer gepa` | Slow | Maximum quality, large annotation sets |

Optimizer-specific settings (iterations, candidates, etc.) are configured in `evalyn.yaml`, not via CLI flags.

If basic doesn't improve alignment enough:

```bash
evalyn calibrate --metric-id <target-metric> --optimizer gepa
```

## Step 3: Re-evaluate with Calibrated Judges

```bash
evalyn run-eval --dataset <dataset-path> --use-calibrated
```

The `--use-calibrated` flag loads optimized prompts from the `calibrations/` directory.

## Step 4: Compare Results

```bash
evalyn compare --run1 <original-run-id> --run2 <calibrated-run-id>
```

Check whether calibrated pass rates better match your expectations.

## Step 5: Assess and Iterate

View calibration history:

```bash
evalyn list-calibrations
```

If alignment improved: calibration successful. The judges now match your expectations.

If alignment is still poor:
- Try a different optimizer (`gepa` for maximum quality)
- Add more annotations (more data = better calibration)
- Review the metric's rubric in the metrics JSON file - it may need manual editing
- Run `evalyn cluster-misalignments --run <run-id>` to see patterns in disagreements

## No Automatic Next Step

This is the terminal skill. The user decides:
- **Iterate**: add more annotations and re-calibrate
- **Fix agent**: if judges are correct and agent has real issues
- **Expand testing**: `evalyn simulate --dataset <path> --modes similar,outlier` then re-evaluate
- **Ship**: export final report with `evalyn export --run <id> --format html`
