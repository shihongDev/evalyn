# evalyn list-calls

List captured function call traces from the database.

## Usage

```bash
evalyn list-calls [OPTIONS]
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--db` | prod | Database to use: prod or test |
| `--limit N` | 20 | Maximum number of calls to display |
| `--offset N` | 0 | Skip first N results (pagination) |
| `--project NAME` | - | Filter by project name |
| `--function NAME` | - | Filter by function name (substring match) |
| `--error-only` | false | Show only calls with errors |
| `--sort FIELD` | -started_at | Sort by field: started_at, duration, function, status. Prefix with + for ascending, - for descending |
| `--format` | table | Output format: table or json |
| `--production` | false | Show only production traces |
| `--simulation` | false | Show only simulation traces |

## Output Columns

| Column | Description |
|--------|-------------|
| id | Unique call identifier |
| function | Function name |
| project | Project name from metadata |
| version | Version from metadata |
| sim | Simulation indicator (S = simulation) |
| status | OK or ERROR |
| file | Source file path |
| started_at | Start timestamp |
| ended_at | End timestamp |
| duration_ms | Execution time in milliseconds |

## Examples

### List recent calls
```bash
evalyn list-calls --limit 10
```

### Filter by project
```bash
evalyn list-calls --project my-agent --limit 50
```

### Show only production traces
```bash
evalyn list-calls --production
```

### Show only simulation traces
```bash
evalyn list-calls --simulation
```

### Filter by function name
```bash
evalyn list-calls --function research
```

### Show only failed calls
```bash
evalyn list-calls --error-only
```

### Sort by duration (descending)
```bash
evalyn list-calls --sort -duration
```

### Paginate results
```bash
evalyn list-calls --limit 20 --offset 40  # Skip first 40, show next 20
```

### JSON output for scripting
```bash
evalyn list-calls --format json
```

## Sample Output

```
id | function | project | version | sim | status | file | started_at | ended_at | duration_ms
------------------------------------------------------------------------------------------------------------
47fe2576... | research_agent | my-agent | v1 |   | OK | agent.py | 2025-01-15 08:13:37 | 2025-01-15 08:14:03 | 25296.37
1629e69e... | research_agent | my-agent | v1 | S | OK | agent.py | 2025-01-15 08:13:03 | 2025-01-15 08:13:29 | 26186.39

Showing 2 of 10 calls (8 more available)
```

## JSON Output Format

When using `--format json`, returns:
```json
{
  "calls": [...],
  "total": 10,
  "showing": 2,
  "offset": 0
}
```

## See Also

- [show-call](show-call.md) - View details of a specific call
- [show-projects](show-projects.md) - View project summaries
- [build-dataset](build-dataset.md) - Build dataset from traces
