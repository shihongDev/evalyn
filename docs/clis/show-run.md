# evalyn show-run

Display detailed results of an evaluation run.

## Usage

```bash
evalyn show-run --id <run_id>
```

## Options

| Option | Required | Description |
|--------|----------|-------------|
| `--id ID` | Yes | The run ID to display |
| `--format` | No | Output format: table (default) or json |

## Examples

### View run details
```bash
evalyn show-run --id abc123
```

### JSON output for scripting
```bash
evalyn show-run --id abc123 --format json
```

## Sample Output

```
============================================================
EVALUATION RUN: abc123
============================================================
Dataset: my-agent-v1-20250115
Created: 2025-01-15 14:30:22
Items:   100

SUMMARY
=======
Metric               | Type       | Result
---------------------|------------|------------------
latency_ms           | objective  | avg=1234.5ms, min=500, max=3000
output_nonempty      | objective  | 100.0% pass (100/100)
helpfulness_accuracy | subjective | 92.0% pass (92/100)
hallucination_risk   | subjective | 88.0% pass (88/100)
completeness         | subjective | 85.0% pass (85/100)

FAILURES (showing first 5)
==========================
Item: item_abc123
  Metric: helpfulness_accuracy
  Score: 0.35
  Reason: Response does not address the user's question

Item: item_def456
  Metric: hallucination_risk
  Score: 0.20
  Reason: Contains unsupported claim about statistics
```

## See Also

- [list-runs](list-runs.md) - List all runs
- [run-eval](run-eval.md) - Create evaluation run
