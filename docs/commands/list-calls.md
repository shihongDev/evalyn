# evalyn list-calls

List captured function call traces from the database.

## Usage

```bash
evalyn list-calls [OPTIONS]
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--limit N` | 20 | Maximum number of calls to display |
| `--project NAME` | - | Filter by project name |
| `--version V` | - | Filter by version |
| `--production` | false | Show only production traces |
| `--simulation` | false | Show only simulation traces |

## Output Columns

| Column | Description |
|--------|-------------|
| id | Unique call identifier |
| function | Function name |
| project | Project name from metadata |
| version | Version from metadata |
| sim? | Simulation indicator |
| status | OK or ERROR |
| file | Source file path |
| start | Start timestamp |
| end | End timestamp |
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

### Filter by version
```bash
evalyn list-calls --project my-agent --version v2
```

### Show only production traces
```bash
evalyn list-calls --production
```

### Show only simulation traces
```bash
evalyn list-calls --simulation
```

## Sample Output

```
id | function | project | version | sim? | status | file | start | end | duration_ms
------------------------------------------------------------------------------------------------------------
47fe2576... | research_agent | my-agent | v1 |  | OK | agent.py | 2025-01-15 08:13:37 | 2025-01-15 08:14:03 | 25296.37
1629e69e... | research_agent | my-agent | v1 | S | OK | agent.py | 2025-01-15 08:13:03 | 2025-01-15 08:13:29 | 26186.39
```

## See Also

- [show-call](show-call.md) - View details of a specific call
- [show-projects](show-projects.md) - View project summaries
- [build-dataset](build-dataset.md) - Build dataset from traces
