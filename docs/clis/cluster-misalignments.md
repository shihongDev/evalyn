# evalyn cluster-misalignments

Cluster and analyze cases where LLM judges disagree with human annotations.

## Usage

```bash
evalyn cluster-misalignments --metric-id <id> --annotations <file> [OPTIONS]
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--metric-id ID` | Required | Metric ID to analyze |
| `--annotations FILE` | Required | Annotations JSONL file |
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

### Cluster misalignments from latest run
```bash
evalyn cluster-misalignments --metric-id helpfulness_accuracy --annotations data/my-dataset/annotations.jsonl --latest
```

### Generate HTML visualization
```bash
evalyn cluster-misalignments --metric-id helpfulness_accuracy --annotations data/my-dataset/annotations.jsonl --dataset data/my-dataset --format html --output misalignments.html
```

### JSON output for scripting
```bash
evalyn cluster-misalignments --metric-id helpfulness_accuracy --annotations data/my-dataset/annotations.jsonl --latest --format json
```

### Use different model for clustering
```bash
evalyn cluster-misalignments --metric-id helpfulness_accuracy --annotations data/my-dataset/annotations.jsonl --latest --model gpt-4o-mini
```

## Sample Output (table)

```
MISALIGNMENT CLUSTERS: helpfulness_accuracy
============================================
Analyzed: 12 disagreements from 50 annotations

False Positives (Judge PASS, Human FAIL): 7 items
-------------------------------------------------
Cluster 1: "Technically correct but unhelpful" (4 items)
  - Judge passed on technically accurate but impractical answers
  - Items: abc123, def456, ...

Cluster 2: "Surface-level responses" (3 items)
  - Judge passed on shallow answers lacking depth
  - Items: ghi789, ...

False Negatives (Judge FAIL, Human PASS): 5 items
-------------------------------------------------
Cluster 3: "Overly strict on formatting" (3 items)
  - Judge failed valid answers due to formatting preferences
  - Items: jkl012, mno345, ...

Cluster 4: "Context-dependent correctness" (2 items)
  - Judge missed nuanced but valid responses
  - Items: pqr678, ...
```

## See Also

- [annotate](annotate.md) - Create annotations first
- [cluster-failures](cluster-failures.md) - Cluster failure cases
- [calibrate](calibrate.md) - Use insights to calibrate judges
