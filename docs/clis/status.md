# status

Show comprehensive status of a dataset including items, metrics, runs, annotations, and calibrations.

## Usage

```bash
evalyn status --dataset <path>
evalyn status --latest
evalyn status  # Lists available datasets
```

## Options

| Option | Description |
|--------|-------------|
| `--dataset PATH` | Path to dataset directory |
| `--latest` | Use the most recently modified dataset |

## Description

The `status` command provides a complete overview of a dataset's current state in the evaluation pipeline. It shows:

- **Dataset info**: Item count, project name, version
- **Metrics**: Available metric sets and their counts
- **Eval runs**: Recent evaluation runs with result counts
- **Annotations**: Human annotation coverage percentage
- **Calibrations**: Calibrated metrics and optimized prompts
- **Suggested next step**: What to do next based on current state

This is useful for understanding where you are in the evaluation workflow and what steps remain.

## Output

```
============================================================
DATASET STATUS: myapp-v1-20250115-120000
============================================================
Path: data/myapp-v1-20250115-120000

--- DATASET ---
Items: 50
Project: myapp
Version: v1

--- METRICS ---
Metric sets: 2
  basic-20250115.json: 5 metrics
  llm-registry-20250115.json: 8 metrics

--- EVAL RUNS ---
Eval runs: 3
  20250115_120500_abc123: 250 results (2025-01-15 12:05:00)
  20250115_110000_def456: 250 results (2025-01-15 11:00:00)

--- ANNOTATIONS ---
Annotated: 20/50 (40%)

--- CALIBRATIONS ---
Calibrations: 2 across 2 metrics
  helpfulness_accuracy
  instruction_following

============================================================
SUGGESTED NEXT STEP:
  evalyn calibrate --metric-id <metric> --annotations ...
```

## Examples

```bash
# Check status of a specific dataset
evalyn status --dataset data/myapp-v1-20250115-120000

# Check status of the most recent dataset
evalyn status --latest

# List all available datasets (when no dataset specified)
evalyn status
```

## See Also

- [build-dataset](build-dataset.md) - Create a dataset from traces
- [run-eval](run-eval.md) - Run evaluation on a dataset
- [annotate](annotate.md) - Add human annotations
- [calibrate](calibrate.md) - Calibrate LLM judges
