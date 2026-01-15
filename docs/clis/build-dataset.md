# evalyn build-dataset

Build a dataset from stored function call traces.

## Usage

```bash
evalyn build-dataset [OPTIONS]
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--project NAME` | - | Filter by project name (recommended) |
| `--version V` | - | Filter by version |
| `--limit N` | 500 | Maximum number of items |
| `--output PATH` | auto | Custom output path |
| `--production` | false | Only production traces |
| `--simulation` | false | Only simulation traces |
| `--since DATE` | - | Filter traces after date (ISO format) |
| `--until DATE` | - | Filter traces before date (ISO format) |
| `--include-errors` | false | Include errored calls |

## Output Structure

```
data/<project>-<version>-<timestamp>/
  dataset.jsonl    # Dataset items (one JSON per line)
  meta.json        # Metadata (filters, counts, schema)
```

### dataset.jsonl format

```json
{"id": "abc123", "input": {"query": "..."}, "output": "...", "metadata": {...}}
{"id": "def456", "input": {"query": "..."}, "output": "...", "metadata": {...}}
```

### meta.json format

```json
{
  "project": "my-agent",
  "version": "v1",
  "created_at": "2025-01-15T10:30:00",
  "item_count": 100,
  "filters": {
    "production_only": false,
    "simulation_only": false
  }
}
```

## Examples

### Build dataset from project
```bash
evalyn build-dataset --project my-agent
```

### Build with version filter
```bash
evalyn build-dataset --project my-agent --version v2
```

### Build production-only dataset
```bash
evalyn build-dataset --project my-agent --production --limit 500
```

### Build from date range
```bash
evalyn build-dataset --project my-agent --since 2025-01-01 --until 2025-01-15
```

### Custom output path
```bash
evalyn build-dataset --project my-agent --output data/custom-dataset
```

## Sample Output

```
Building dataset...
  Project: my-agent
  Version: v1
  Limit: 200

Found 156 matching traces
Saved to: data/my-agent-v1-20250115_103045/
  - dataset.jsonl (156 items)
  - meta.json
```

## See Also

- [list-calls](list-calls.md) - View available traces
- [show-projects](show-projects.md) - View project summaries
- [suggest-metrics](suggest-metrics.md) - Suggest metrics for the dataset
- [run-eval](run-eval.md) - Run evaluation on the dataset
