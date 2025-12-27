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
- Trace events
- Duration

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
Call ID: 47fe2576-03c3-4438-8708-b8ab38cf52e9
Function: research_agent
Project: my-agent
Version: v1
Status: OK

Started: 2025-01-15 08:13:37.738881+00:00
Ended:   2025-01-15 08:14:03.035253+00:00
Duration: 25296.37ms

--- INPUTS ---
{
  "query": "What are the latest developments in AI?"
}

--- OUTPUT ---
{
  "answer": "Recent developments in AI include...",
  "sources": ["https://example.com/article1", ...]
}

--- TRACE EVENTS ---
[0.00s] gemini.request: {"model": "gemini-2.5-flash"}
[1.23s] web_search: {"query": "AI developments 2025"}
[3.45s] gemini.response: {"tokens": 1234}
```

## See Also

- [list-calls](list-calls.md) - List all captured calls
- [show-projects](show-projects.md) - View project summaries
