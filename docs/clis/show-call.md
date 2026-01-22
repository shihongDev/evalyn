# evalyn show-call

Display detailed information about a specific function call trace.

## Usage

```bash
evalyn show-call --id <call_id>
evalyn show-call --last
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--id ID` | - | The call ID to display (supports short IDs like `47fe25`) |
| `--last` | false | Show the most recent call |
| `--db` | prod | Database to use: prod or test |
| `--format` | table | Output format: table or json |

One of `--id` or `--last` is required.

## Output

Displays:
- Call metadata (function name, project, version, timestamps)
- Input parameters
- Output/return value
- Error message (if failed)
- LLM and tool call counts (from spans or trace events)
- Span tree (hierarchical view of LLM calls, tool calls, and nodes)
- Span timeline (chronological view when trace events are minimal)
- Events timeline (for legacy instrumentation)

### LLM/Tool Counting

The `llm_calls` and `tool_events` counts are derived from:
1. **Spans** (preferred): Counts spans with `span_type="llm_call"` or `span_type="tool_call"`
2. **Trace events** (legacy): Counts events matching LLM or tool patterns

This ensures accurate counts for both modern instrumentation (Claude Agent SDK, LangGraph) and legacy monkey-patch instrumentation (OpenAI, Anthropic direct).

## Examples

### View a specific call
```bash
evalyn show-call --id 47fe2576-03c3-4438-8708-b8ab38cf52e9
```

### View using short ID
```bash
evalyn show-call --id 47fe25
```

### View the most recent call
```bash
evalyn show-call --last
```

### JSON output for scripting
```bash
evalyn show-call --last --format json
```

### Get call ID from list-calls first
```bash
# First, list calls to find the ID
evalyn list-calls --limit 5

# Then view the specific call
evalyn show-call --id <id-from-list>
```

## Sample Output

```
================ Call Details ================
id       : 47fe2576-03c3-4438-8708-b8ab38cf52e9
function : research_agent
status   : OK
session  : abc123-session-id
started  : 2025-01-15 08:13:37.738881+00:00
ended    : 2025-01-15 08:14:03.035253+00:00
duration : 25296.37 ms
turns    : single (1)
llm_calls: 3 | tool_events: 2

Inputs:
  kwargs:
    - query: What are the latest developments in AI?

Output:
  type   : dict
  length : 1234 chars
  preview: {"answer": "Recent developments in AI include...", ...}

Span Tree:
`-- research_agent (25.3s)
    |-- llm: gemini:gemini-2.5-flash [150>200 tok] (1.2s)
    |-- tool: web_search (2.1s)
    `-- llm: gemini:gemini-2.5-flash [500>800 tok] (3.4s)

  5 spans | 2 LLM | 1 tools | 0 nodes
```

### Span Timeline (for hook-based instrumentation)

When trace events are minimal but spans exist (e.g., Claude Agent SDK), a chronological span timeline is shown:

```
Span Timeline (chronological):
----------------------------------------------------------------------------------------------------
  1 | +       0ms | session      | chat                 | 1461419ms |
  2 | +   13371ms | tool_call    | WebSearch            | 24898ms  | tool=WebSearch
  3 | +   14411ms | tool_call    | WebSearch            | 23520ms  | tool=WebSearch
  4 | +   19220ms | llm_call     | llm_turn_1           | 0ms      | model=claude-haiku-4-5-20251001
  5 | +   20548ms | llm_call     | llm_turn_2           | 0ms      | model=claude-haiku-4-5-20251001
...
```

## See Also

- [list-calls](list-calls.md) - List all captured calls
- [show-trace](show-trace.md) - View span tree only (with --verbose for details)
- [show-span](show-span.md) - View details of a single span
- [show-projects](show-projects.md) - View project summaries
