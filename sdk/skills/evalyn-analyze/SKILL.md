---
name: evalyn-analyze
description: Use when analyzing evalyn evaluation results, investigating failures, comparing runs, or understanding agent performance
---

# evalyn-analyze

## Overview

Analyze evaluation results progressively: summary, insights, failure clustering, and trend analysis. Interpret findings and recommend next actions based on pass rates.

## Pre-flight

Verify evaluation runs exist:

```bash
evalyn list-runs --limit 3
```

If no runs: "You need to run an evaluation first. Invoke `evalyn-eval`."

Identify the latest run ID from the output.

## Step 1: Metric Summary

```bash
evalyn analyze --run <run-id>
```

This shows:
- Per-metric pass rates and average scores
- Key findings (highest/lowest performing metrics)
- Overall health rating (GOOD/MODERATE/POOR)

You can also use short IDs (first 8 characters of run ID).

## Step 2: Deep Insights

```bash
evalyn insights --run <run-id>
```

Provides diagnostic and prescriptive analysis:
- Metric correlations (which metrics move together)
- Anomaly detection
- Actionable recommendations

## Step 3: Compare and Trend

If multiple runs exist (check `evalyn list-runs` output):

```bash
evalyn compare --run1 <previous-run-id> --run2 <latest-run-id>
```

For longer history across all runs in a dataset:

```bash
evalyn trend --dataset <dataset-path>
```

Note: use `--run1` and `--run2` flags for compare, not positional arguments.

## Step 4: Investigate Failures

If any metric has pass rate below 90%:

```bash
evalyn cluster-failures --run <run-id>
```

This clusters failed items by failure reason, revealing patterns (e.g., "all failures involve long inputs" or "failures cluster around a specific topic").

## Step 5: Interpret and Recommend

Based on the results, recommend next action:

| Overall Pass Rate | Interpretation | Recommendation |
|------------------|----------------|----------------|
| Above 95% | Agent performing well | Consider `evalyn simulate --dataset <path> --modes similar,outlier` for edge case testing. Export report: `evalyn export --run <id> --format html` |
| 80-95% | Moderate issues | Review failing items. Could be agent issues OR judge misalignment. Consider invoking `evalyn-calibrate` to verify judges are accurate. |
| Below 80% | Significant issues | Invoke `evalyn-calibrate` to annotate items and check if judges agree with human expectations. Fix agent if judges are correct. |

Key distinction to communicate: low pass rates can mean either (a) the agent is actually performing poorly, or (b) the LLM judges are too strict or misaligned. Calibration determines which.

## Export

Offer a shareable report:

```bash
evalyn export --run <run-id> --format html
```

Other formats: `json`, `csv`, `markdown`.

## Hand-off

If calibration recommended: "Invoke `evalyn-calibrate` to annotate results and align LLM judges with your expectations."

If agent is performing well: "Your agent looks good. To strengthen confidence, run it on more varied inputs and re-evaluate, or use `evalyn simulate --dataset <path> --modes similar,outlier` to generate edge cases."
