# validate

Validate dataset format and detect potential issues before running evaluations.

## Usage

```bash
evalyn validate --dataset <path>
evalyn validate --latest
```

## Options

| Option | Description |
|--------|-------------|
| `--dataset PATH` | Path to dataset directory or file |
| `--latest` | Use the most recently modified dataset |

## Description

The `validate` command checks your dataset for:

- **JSON format validity** - Ensures each line is valid JSON
- **Required fields** - Checks for `id`, `inputs`/`input`, `output`
- **Reference fields** - Warns if no `expected`/`reference` values (needed for BLEU/ROUGE)
- **Duplicate IDs** - Detects duplicate item IDs
- **Metadata consistency** - Validates `meta.json` if present
- **Metrics directory** - Checks if metrics files exist

## Output

```
Validating: data/my-dataset/dataset.jsonl

------------------------------------------------------------
VALIDATION RESULTS
------------------------------------------------------------

  Total items:        100
  With 'id':          100 (100%)
  With 'inputs':      100 (100%)
  With 'output':      100 (100%)
  With 'expected':    0 (0%)
  With 'metadata':    100 (100%)
  Unique IDs:         100

  meta.json:          Found
  Project:            my-agent
  Version:            v1

  Metrics files:      2
    - basic.json
    - llm-registry.json

WARNINGS (1):
  - No 'expected' or 'reference' values found. Reference-based metrics will not work.

Dataset is valid!
```

## Exit Codes

- `0` - Dataset is valid
- `1` - Validation errors found

## Examples

```bash
# Validate a specific dataset
evalyn validate --dataset data/my-agent-v1-20240101

# Validate the most recent dataset
evalyn validate --latest

# Use in CI/CD pipelines
evalyn validate --dataset data/my-dataset || exit 1
```

## See Also

- [build-dataset](build-dataset.md) - Build datasets from traces
- [run-eval](run-eval.md) - Run evaluation on validated dataset
