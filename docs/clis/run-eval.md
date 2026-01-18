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
| `--workers`, `-w` | 4 | Parallel workers for LLM evaluation (max: 16) |

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
evalyn run-eval --dataset data/my-dataset --metrics metrics/metrics.json
```

### Use multiple metrics files
```bash
evalyn run-eval --dataset data/my-dataset --metrics "metrics/metrics.json,metrics/metrics-custom.json"
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

Run folder: data/my-dataset/eval_runs/20250115-143022_abc12345
  results.json - evaluation data
  report.html  - analysis report
```

## Checkpoint and Resume

If an evaluation is interrupted (Ctrl+C), progress is automatically saved to `.eval_checkpoint.json` in the dataset directory.

Re-run the same command to resume from where it left off:

```bash
# Interrupted during evaluation
evalyn run-eval --dataset data/my-dataset
# ^C
# Evaluation interrupted.
# Progress saved to: data/my-dataset/.eval_checkpoint.json
# Resume with: evalyn run-eval --dataset data/my-dataset

# Resume from checkpoint
evalyn run-eval --dataset data/my-dataset
# Resuming from checkpoint...
```

The checkpoint is automatically removed when evaluation completes successfully.

## Output Files

Each eval run creates a dedicated folder in `<dataset>/eval_runs/<timestamp>_<run_id>/`:

```
eval_runs/
└── 20250115-143022_abc12345/
    ├── results.json   # Raw evaluation data
    └── report.html    # Interactive analysis report
```

### results.json
Raw evaluation data:
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

### report.html
Interactive visualization report with:
- Summary statistics cards
- Pass rate bar charts (color-coded by threshold)
- Score distribution with min/max ranges
- Pass/fail breakdown by metric
- Metric correlation heatmap
- Failed items list with failure reasons
- Run metadata

The HTML report uses Chart.js for interactive visualizations and is styled with Anthropic research paper aesthetics (light background, blue/coral colors).

## See Also

- [build-dataset](build-dataset.md) - Build dataset first
- [suggest-metrics](suggest-metrics.md) - Generate metrics
- [list-runs](list-runs.md) - View past evaluation runs
- [show-run](show-run.md) - View run details
- [annotate](annotate.md) - Annotate results for calibration
