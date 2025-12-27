# evalyn run-eval

Run evaluation on a dataset using specified metrics.

## Usage

```bash
evalyn run-eval --dataset <path> [OPTIONS]
evalyn run-eval --latest [OPTIONS]
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--dataset PATH` | - | Dataset directory or file |
| `--latest` | false | Use most recent dataset |
| `--metrics FILE` | auto | Metrics JSON file(s), comma-separated |
| `--metrics-all` | false | Use all metrics from metrics/ folder |
| `--use-calibrated` | false | Use calibrated prompts if available |
| `--dataset-name NAME` | - | Custom name for the run |
| `--format FMT` | table | Output format: `table` or `json` |

## Metrics Resolution

1. If `--metrics` specified: uses those files
2. If `--metrics-all`: uses all JSON files in `metrics/` folder
3. Otherwise: uses `active_metric_set` from `meta.json`

## Examples

### Run on dataset with auto-detected metrics
```bash
evalyn run-eval --dataset data/my-agent-v1-20250115
```

### Run on latest dataset
```bash
evalyn run-eval --latest
```

### Specify metrics file
```bash
evalyn run-eval --dataset data/my-dataset --metrics metrics/llm-registry.json
```

### Use multiple metrics files
```bash
evalyn run-eval --dataset data/my-dataset --metrics "metrics/basic.json,metrics/custom.json"
```

### Use all available metrics
```bash
evalyn run-eval --dataset data/my-dataset --metrics-all
```

### Use calibrated prompts
```bash
evalyn run-eval --dataset data/my-dataset --use-calibrated
```

### JSON output for scripting
```bash
evalyn run-eval --latest --format json
```

## Sample Output

```
Loaded 5 metrics (2 objective, 3 subjective)
Dataset: 100 items

Running evaluation...
[████████████████████████████████████████] 100% latency_ms (objective)

RESULTS
=======
Metric              | Type       | Score/Pass Rate
--------------------|------------|----------------
latency_ms          | objective  | avg=1234.5ms
output_nonempty     | objective  | 100.0% pass
helpfulness_accuracy| subjective | 92.0% pass
hallucination_risk  | subjective | 88.0% pass
completeness        | subjective | 85.0% pass

Run saved to: data/my-dataset/eval_runs/20250115_143022_abc123.json
```

## Output Files

Eval runs are saved to `<dataset>/eval_runs/<timestamp>_<id>.json`:

```json
{
  "id": "abc123...",
  "dataset_name": "my-agent-v1",
  "created_at": "2025-01-15T14:30:22",
  "summary": {
    "latency_ms": {"avg": 1234.5, "min": 500, "max": 3000},
    "helpfulness_accuracy": {"pass_rate": 0.92, "total": 100, "passed": 92}
  },
  "metric_results": [...]
}
```

## See Also

- [build-dataset](build-dataset.md) - Build dataset first
- [suggest-metrics](suggest-metrics.md) - Generate metrics
- [list-runs](list-runs.md) - View past evaluation runs
- [show-run](show-run.md) - View run details
- [annotate](annotate.md) - Annotate results for calibration
