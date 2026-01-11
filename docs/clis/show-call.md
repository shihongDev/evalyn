# evalyn show-call

Display detailed information about a specific function call trace.

## Usage

```bash
evalyn show-call --id <call_id>
```

## Options

| Option | Required | Description |
|--------|----------|-------------|
| `--id ID` | Yes | The call ID to display |

## Output

Displays:
- Call metadata (function name, project, version, timestamps)
- Input parameters
- Output/return value
- Error message (if failed)
- Trace events timeline
- Span tree (hierarchical view of LLM calls, tool calls, and nodes)
- Duration and token counts

## Examples

### View a specific call
```bash
evalyn show-call --id 47fe2576-03c3-4438-8708-b8ab38cf52e9
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

Span Tree (Hierarchical):
`-- research_agent (25.3s)
    |-- llm: gemini:gemini-2.5-flash [150>200 tok] (1.2s)
    |-- tool: web_search (2.1s)
    `-- llm: gemini:gemini-2.5-flash [500>800 tok] (3.4s)

  5 spans | 2 LLM | 1 tools | 0 nodes

Events timeline:
idx | t+ms  | delta_ms | kind              | summary
------------------------------------------------------------------
1   |   0.0 |      0.0 | gemini.request    | model=gemini-2.5-flash
2   | 1234.5|   1234.5 | gemini.response   | tokens=350
3   | 1500.0|    265.5 | tool.call         | tool=web_search
...
```

## See Also

- [list-calls](list-calls.md) - List all captured calls
- [show-trace](show-trace.md) - View span tree only (simpler output)
- [show-projects](show-projects.md) - View project summaries
