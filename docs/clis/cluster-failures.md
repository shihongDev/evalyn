# evalyn cluster-failures

Cluster and analyze failure cases from evaluation runs using LLM-powered semantic clustering.

## Usage

```bash
evalyn cluster-failures [OPTIONS]
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--metric-id ID` | all | Metric ID to analyze (default: all metrics with failures) |
| `--run-id ID` | latest | Eval run ID to analyze |
| `--dataset PATH` | - | Dataset folder for input/output context |
| `--latest` | false | Use most recently modified dataset |
| `--model MODEL` | gemini-2.5-flash-lite | LLM model for clustering |
| `--format FMT` | html | Output format: `html`, `table`, or `json` |
| `--output PATH` | auto | Output file path |
| `--quiet` | false | Suppress hints and extra output |

## Output Formats

| Format | Description |
|--------|-------------|
| `html` | Interactive scatter plot visualization |
| `table` | ASCII table for terminal |
| `json` | Machine-readable JSON |

## Examples

### Cluster failures from latest run
```bash
evalyn cluster-failures --latest
```

### Cluster failures for specific metric
```bash
evalyn cluster-failures --dataset data/my-dataset --metric-id helpfulness_accuracy
```

### Generate HTML visualization
```bash
evalyn cluster-failures --latest --format html --output failures.html
```

### JSON output for scripting
```bash
evalyn cluster-failures --latest --format json
```

### Use different model for clustering
```bash
evalyn cluster-failures --latest --model gpt-4o-mini
```

## Sample Output (table)

```
FAILURE CLUSTERS: helpfulness_accuracy
======================================
Analyzed: 15 failures from 100 items

Cluster 1: "Incomplete answers" (6 items)
  - Missing key details requested in query
  - Items: abc123, def456, ghi789, ...

Cluster 2: "Off-topic responses" (5 items)
  - Response doesn't address the actual question
  - Items: jkl012, mno345, ...

Cluster 3: "Formatting issues" (4 items)
  - Response lacks structure or readability
  - Items: pqr678, stu901, ...
```

## See Also

- [run-eval](run-eval.md) - Run evaluation first
- [cluster-misalignments](cluster-misalignments.md) - Cluster human/LLM disagreements
- [calibrate](calibrate.md) - Calibrate judges using analysis
