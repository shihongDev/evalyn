# annotation-stats

Show annotation coverage and agreement statistics.

## Usage

```bash
evalyn annotation-stats --dataset <path>
```

## Options

| Option | Description |
|--------|-------------|
| `--dataset PATH` | (Required) Path to annotations.jsonl or dataset directory |

## Description

The `annotation-stats` command provides detailed statistics about your annotation progress and human-LLM agreement. It shows:

- **Coverage**: How many items have been annotated
- **Per-metric stats**: Pass/fail counts for each metric
- **Agreement analysis**: Where human annotators agree or disagree with LLM judges
- **False positive/negative rates**: Specific disagreement types

This helps you understand:
- Annotation progress and remaining work
- Which metrics need calibration (high disagreement)
- Overall quality of LLM judge predictions

## Output

```
Annotation Statistics
=====================

Total items: 50
With human labels: 25 (50%)
With eval results: 50 (100%)

Per-Metric Eval Results:
  helpfulness_accuracy: 50 total (42 passed, 8 failed)
  toxicity_safety: 50 total (48 passed, 2 failed)
  instruction_following: 50 total (35 passed, 15 failed)

Human-LLM Agreement (25 annotated items):
  helpfulness_accuracy:
    Agree: 20 (80%)
    Disagree: 5 (20%)
      - False positives: 3 (LLM passed, human failed)
      - False negatives: 2 (LLM failed, human passed)

  toxicity_safety:
    Agree: 24 (96%)
    Disagree: 1 (4%)
```

## Examples

```bash
# View stats for a dataset directory
evalyn annotation-stats --dataset data/myapp-v1

# View stats for an annotations file directly
evalyn annotation-stats --dataset data/myapp-v1/annotations.jsonl
```

## See Also

- [annotate](annotate.md) - Add human annotations
- [calibrate](calibrate.md) - Calibrate metrics based on annotations
- [export-for-annotation](export-for-annotation.md) - Export for external annotation
