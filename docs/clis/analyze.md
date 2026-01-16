# analyze

Analyze evaluation results and generate actionable insights.

## Usage

```bash
evalyn analyze --run <run_id>
evalyn analyze --dataset <path>
evalyn analyze --latest
```

## Options

| Option | Description |
|--------|-------------|
| `--run ID` | Eval run ID to analyze |
| `--dataset PATH` | Dataset path (uses latest run from eval_runs/) |
| `--latest` | Use the most recently modified dataset |
| `--format` | Output format: table (default) or json |

## Description

The `analyze` command provides insights from your evaluation results:

- **Metric Summary** - Pass rates and average scores for each metric
- **Problem Detection** - Identifies metrics with high failure rates
- **Pattern Analysis** - Finds items failing multiple metrics
- **Health Assessment** - Overall evaluation health score
- **Recommendations** - Actionable next steps

## Output

```
======================================================================
  EVALUATION ANALYSIS
======================================================================

Run ID:      abc123
Dataset:     my-agent-v1
Items:       50
Started:     2024-01-15 10:30:00

======================================================================
  METRIC SUMMARY
======================================================================

  X helpfulness_accuracy          35/50 passed (70%)  avg=0.72
  X hallucination_risk            40/50 passed (80%)  avg=0.85
  V latency_ms                    50/50 passed (100%)  avg=245.3

======================================================================
  INSIGHTS
======================================================================

  - 'helpfulness_accuracy' has the highest failure rate (15/50 failed).
    Consider reviewing the rubric or calibrating.

  - Item 'item-23...' failed 3 metrics: helpfulness_accuracy,
    hallucination_risk, completeness

  - Overall health is MODERATE (83% pass rate)

======================================================================
  RECOMMENDATIONS
======================================================================

  1. Run 'evalyn annotate' to provide human labels for failed items
  2. Run 'evalyn calibrate' to improve metric alignment
```

## Examples

```bash
# Analyze a specific run by ID
evalyn analyze --run abc123-def456

# Analyze the latest run from a dataset
evalyn analyze --dataset data/my-agent-v1

# Analyze the most recently modified dataset
evalyn analyze --latest
```

## See Also

- [run-eval](run-eval.md) - Run evaluation to generate results
- [compare](compare.md) - Compare two evaluation runs
- [annotate](annotate.md) - Provide human labels for calibration
