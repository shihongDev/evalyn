# evalyn show-projects

Display a summary of all projects with traced calls.

## Usage

```bash
evalyn show-projects
```

## Options

None.

## Output

Shows for each project:
- Project name
- Number of traces
- Versions available
- Date range of traces

## Examples

### View all projects
```bash
evalyn show-projects
```

## Sample Output

```
PROJECT SUMMARY
===============

gemini-deep-research-agent
  Versions: v1, v2
  Traces: 156
  First: 2025-01-10 10:23:45
  Last:  2025-01-15 08:14:03

my-chatbot
  Versions: v1
  Traces: 42
  First: 2025-01-12 14:30:00
  Last:  2025-01-14 16:45:22
```

## See Also

- [list-calls](list-calls.md) - List individual calls
- [build-dataset](build-dataset.md) - Build dataset from a project's traces
