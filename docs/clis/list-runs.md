# evalyn list-runs

List past evaluation runs.

## Usage

```bash
evalyn list-runs [OPTIONS]
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--limit N` | 20 | Maximum runs to display |
| `--dataset PATH` | - | Filter by dataset |

## Examples

### List recent runs
```bash
evalyn list-runs
```

### Limit results
```bash
evalyn list-runs --limit 10
```

## Sample Output

```
ID       | Dataset              | Created              | Metrics | Items
---------|----------------------|----------------------|---------|------
abc123   | my-agent-v1-20250115 | 2025-01-15 14:30:22 | 5       | 100
def456   | my-agent-v1-20250115 | 2025-01-15 10:15:00 | 5       | 100
ghi789   | my-agent-v2-20250114 | 2025-01-14 16:45:33 | 3       | 50
```

## See Also

- [show-run](show-run.md) - View run details
- [run-eval](run-eval.md) - Create new evaluation run
