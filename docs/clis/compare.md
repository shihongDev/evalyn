# compare

Compare two evaluation runs side-by-side to measure improvement or regression.

## Usage

```bash
evalyn compare --run1 <id_or_path> --run2 <id_or_path>
```

## Options

| Option | Description |
|--------|-------------|
| `--run1 ID` | First eval run ID or path to run JSON file (required) |
| `--run2 ID` | Second eval run ID or path to run JSON file (required) |

## Description

The `compare` command helps you:

- **Track Progress** - See if changes improved your agent
- **A/B Testing** - Compare different agent versions
- **Regression Detection** - Identify metrics that got worse
- **Delta Analysis** - Quantify improvement per metric

## Output

```
======================================================================
  EVALUATION COMPARISON
======================================================================

  Run 1: abc123... (my-agent-v1)
  Run 2: def456... (my-agent-v2)

======================================================================
  METRIC COMPARISON
======================================================================

  Metric                      Run 1       Run 2        Delta
  ------------------------- ------------ ------------ ------------
  helpfulness_accuracy             70%         85%     +15% up
  hallucination_risk               80%         90%     +10% up
  latency_ms                      100%        100%          =
  completeness                     60%         55%      -5% down

======================================================================
  SUMMARY
======================================================================

  Overall pass rate:
    Run 1: 77.5% (155/200)
    Run 2: 82.5% (165/200)
    Change: +5.0% up IMPROVED

  Metrics improved:  2
  Metrics regressed: 1
  Metrics unchanged: 1
```

## Examples

```bash
# Compare two runs by ID
evalyn compare --run1 abc123 --run2 def456

# Compare run files directly
evalyn compare --run1 data/v1/eval_runs/run1.json --run2 data/v2/eval_runs/run2.json

# Compare before and after calibration
evalyn compare \
  --run1 data/myproj/eval_runs/20240115_before.json \
  --run2 data/myproj/eval_runs/20240115_after_calibration.json
```

## Use Cases

### Version Comparison
Compare different versions of your agent:
```bash
# Build datasets for each version
evalyn build-dataset --project my-agent --version v1 --output data/v1
evalyn build-dataset --project my-agent --version v2 --output data/v2

# Run evaluations
evalyn run-eval --dataset data/v1
evalyn run-eval --dataset data/v2

# Compare results
evalyn compare \
  --run1 data/v1/eval_runs/*.json \
  --run2 data/v2/eval_runs/*.json
```

### Calibration Impact
Measure the effect of calibrating LLM judges:
```bash
evalyn run-eval --dataset data/myproj
evalyn calibrate --metric-id helpfulness_accuracy --annotations annotations.jsonl
evalyn run-eval --dataset data/myproj --use-calibrated
evalyn compare --run1 <before_id> --run2 <after_id>
```

## See Also

- [run-eval](run-eval.md) - Run evaluations
- [analyze](analyze.md) - Analyze a single run
- [calibrate](calibrate.md) - Calibrate LLM judges
