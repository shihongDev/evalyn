# evalyn calibrate

Calibrate LLM judges by comparing their results to human annotations.

## Usage

```bash
evalyn calibrate --metric-id <id> --annotations <file> [OPTIONS]
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--metric-id ID` | Required | Metric to calibrate |
| `--annotations FILE` | Required | Annotations file (JSONL) |
| `--dataset PATH` | - | Dataset path (for prompt optimization) |
| `--optimizer TYPE` | llm | Optimizer: `llm` or `gepa` |
| `--no-optimize` | false | Skip prompt optimization, only compute metrics |
| `--output FILE` | - | Save calibration record to file |
| `--show-examples` | false | Show disagreement examples |

## Alignment Metrics

| Metric | Description |
|--------|-------------|
| Accuracy | Overall agreement rate |
| Precision | True positives / (True positives + False positives) |
| Recall | True positives / (True positives + False negatives) |
| F1 Score | Harmonic mean of precision and recall |
| Specificity | True negatives / (True negatives + False positives) |
| Cohen's Kappa | Agreement adjusted for chance |

## Optimizers

| Optimizer | Description |
|-----------|-------------|
| `llm` | LLM analyzes disagreements and suggests rubric improvements |
| `gepa` | Genetic algorithm evolves prompts (requires `deap` package) |

## Examples

### Basic calibration (metrics only)
```bash
evalyn calibrate --metric-id helpfulness_accuracy --annotations data/my-dataset/annotations.jsonl --no-optimize
```

### Full calibration with LLM optimization
```bash
evalyn calibrate --metric-id helpfulness_accuracy --annotations data/my-dataset/annotations.jsonl --dataset data/my-dataset
```

### Show disagreement examples
```bash
evalyn calibrate --metric-id helpfulness_accuracy --annotations data/my-dataset/annotations.jsonl --show-examples
```

### Save calibration record
```bash
evalyn calibrate --metric-id helpfulness_accuracy --annotations data/my-dataset/annotations.jsonl --output calibration.json
```

### Use GEPA optimizer
```bash
evalyn calibrate --metric-id helpfulness_accuracy --annotations data/my-dataset/annotations.jsonl --optimizer gepa
```

## Sample Output

```
============================================================
CALIBRATION REPORT: helpfulness_accuracy
============================================================
Eval Run: 73bf3c5c-aa9c-4078-a668-8227bd2a1028
Samples:  50

--- ALIGNMENT METRICS ---
Accuracy:       84.0%
Precision:      88.2%
Recall:         90.0%
F1 Score:       89.1%
Specificity:    70.0%
Cohen's Kappa:  0.652

Confusion Matrix:
                   Human PASS  Human FAIL
  Judge PASS           36          3
  Judge FAIL           5           6

--- THRESHOLD ---
Current:   0.500
Suggested: 0.450

--- PROMPT OPTIMIZATION ---
Analyzing 8 disagreement cases...

Suggested rubric improvements:
1. Be more lenient on partial answers that address the main question
2. Consider context relevance, not just factual accuracy
3. Penalize responses that are correct but off-topic

--- SAVED FILES ---
Calibration: data/my-dataset/calibrations/helpfulness_accuracy/20250115_143022_llm.json
Optimized prompt: data/my-dataset/calibrations/helpfulness_accuracy/prompts/optimized.txt

============================================================
```

## Output Files

Calibration results are saved to `<dataset>/calibrations/<metric_id>/`:

```
calibrations/
  helpfulness_accuracy/
    20250115_143022_llm.json    # Calibration record
    prompts/
      original.txt              # Original prompt
      optimized.txt             # Optimized prompt
```

## See Also

- [annotate](annotate.md) - Create annotations first
- [run-eval](run-eval.md) - Re-run eval with `--use-calibrated`
- [list-calibrations](list-calibrations.md) - View calibration history
