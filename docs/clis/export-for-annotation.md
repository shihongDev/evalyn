# export-for-annotation

Export dataset items with evaluation results for human annotation workflows.

## Usage

```bash
evalyn export-for-annotation --dataset <path> --output <file>
```

## Options

| Option | Description |
|--------|-------------|
| `--dataset PATH` | (Required) Path to dataset directory or dataset.jsonl file |
| `--output PATH` | (Required) Output path for annotation JSONL file |
| `--run-id ID` | Specific eval run ID to use (defaults to latest) |

## Description

The `export-for-annotation` command creates a JSONL file containing dataset items enriched with evaluation results. This is useful for:

- External annotation tools that need structured data
- Sharing annotation tasks with team members
- Creating annotation workflows outside the CLI

Each exported item includes:
- Original input and output
- Evaluation results from the specified (or latest) run
- Existing human labels (if any)
- Item metadata

## Output Format

Each line in the output JSONL contains:

```json
{
  "id": "item-123",
  "input": {"query": "What is the weather?"},
  "output": "The weather today is sunny...",
  "eval_results": {
    "helpfulness_accuracy": {
      "passed": true,
      "score": 0.85,
      "reason": "Response directly addresses the query..."
    }
  },
  "human_label": null,
  "metadata": {}
}
```

## Output

```
Using latest eval run: abc123-def456
Exported 50 items to annotations_export.jsonl
  - With eval results: 50
  - With human labels: 10
  - Awaiting annotation: 40
```

## Examples

```bash
# Export with latest eval results
evalyn export-for-annotation --dataset data/myapp-v1 --output to_annotate.jsonl

# Export with a specific eval run
evalyn export-for-annotation --dataset data/myapp-v1 --output to_annotate.jsonl --run-id abc123
```

## See Also

- [annotate](annotate.md) - Interactive CLI annotation
- [import-annotations](import-annotations.md) - Import completed annotations
- [annotation-stats](annotation-stats.md) - View annotation statistics
