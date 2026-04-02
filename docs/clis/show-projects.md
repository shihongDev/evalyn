# evalyn show-projects

Display a summary of all projects with traced calls.

## Usage

```bash
evalyn show-projects
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--db` | prod | Database to use: prod or test |
| `--limit N` | 1000 | Maximum number of calls to scan |

## Output

Displays a pipe-delimited table with columns:

| Column | Description |
|--------|-------------|
| project | Project name |
| version | Version from metadata |
| calls | Total number of traces |
| errors | Number of traces with errors |
| first | Earliest trace timestamp |
| last | Latest trace timestamp |

## Examples

### View all projects
```bash
evalyn show-projects
```

## Sample Output

```
project | version | calls | errors | first | last
------------------------------------------------------------------------------------------------------------------------
gemini-deep-research-agent | v1 | 156 | 2 | 2025-01-10 10:23:45 | 2025-01-15 08:14:03
my-chatbot | v1 | 42 | 0 | 2025-01-12 14:30:00 | 2025-01-14 16:45:22
```

## See Also

- [list-calls](list-calls.md) - List individual calls
- [build-dataset](build-dataset.md) - Build dataset from a project's traces
