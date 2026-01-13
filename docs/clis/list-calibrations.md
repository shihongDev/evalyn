# list-calibrations

List calibration records for a dataset.

## Usage

```bash
evalyn list-calibrations --dataset <path>
evalyn list-calibrations --latest
```

## Options

| Option | Description |
|--------|-------------|
| `--dataset PATH` | Path to dataset directory |
| `--latest` | Use the most recently modified dataset |
| `--format FORMAT` | Output format: `table` (default) or `json` |

## Description

The `list-calibrations` command shows all calibration records stored for a dataset. Each calibration includes:

- **Metric ID**: Which metric was calibrated
- **Timestamp**: When calibration was performed
- **Optimizer**: Method used (llm or gepa)
- **Alignment metrics**: Accuracy, F1 score, Cohen's Kappa
- **Sample count**: Number of annotations used

It also lists optimized prompt files that were generated during calibration.

## Output

```
Calibrations in myapp-v1-20250115-120000:
================================================================================
Metric                    Timestamp         Optimizer Acc     F1      Kappa   N
--------------------------------------------------------------------------------
helpfulness_accuracy      20250115_120000   gepa      85.0%   83.2%   0.712   25
instruction_following     20250115_110000   llm       78.0%   76.5%   0.623   20
toxicity_safety           20250114_150000   llm       92.0%   90.1%   0.845   30

================================================================================
Optimized prompts:
  helpfulness_accuracy: calibrations/helpfulness_accuracy/prompts/20250115_120000_full.txt
  instruction_following: calibrations/instruction_following/prompts/20250115_110000_full.txt
```

## Examples

```bash
# List calibrations for a specific dataset
evalyn list-calibrations --dataset data/myapp-v1

# List calibrations for the most recent dataset
evalyn list-calibrations --latest

# Get JSON output for programmatic use
evalyn list-calibrations --dataset data/myapp-v1 --format json
```

## See Also

- [calibrate](calibrate.md) - Calibrate LLM judges
- [annotate](annotate.md) - Create annotations for calibration
- [run-eval](run-eval.md) - Run evaluation with calibrated prompts (--use-calibrated)
