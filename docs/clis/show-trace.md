# show-trace

Display the hierarchical span tree for a traced call (Phoenix-style visualization).

## Usage

```bash
evalyn show-trace --id <call_id>
evalyn show-trace --last
```

## Options

| Option | Description |
|--------|-------------|
| `--id ID` | The call ID to display spans for |
| `--last` | Show the most recent call |
| `--max-depth N` | Maximum depth of span tree to display (default: unlimited) |

One of `--id` or `--last` is required.

## Description

Shows a hierarchical tree view of all spans captured during a function call, including:
- LLM calls (with token counts and model names)
- Tool calls
- Graph nodes (for LangGraph agents)
- Session spans
- Duration for each span

This is a simplified view compared to `show-call` - it focuses only on the span tree.

## Sample Output

```
Trace: run_agent (5.2s) [OK]
Call ID: 47fe2576-03c3-4438-8708-b8ab38cf52e9
Session: abc123-session

`-- session:main (5.2s)
    |-- graph:research_graph (4.8s)
    |   |-- node:search_node (1.2s)
    |   |   `-- llm_call gemini:gemini-2.5-flash [200>150 tokens] (1.1s)
    |   `-- node:summarize_node (2.3s)
    |       `-- llm_call openai:gpt-4 [500>300 tokens] (2.2s)
    `-- llm_call anthropic:claude-3 [100>50 tokens] (0.4s)

Summary: 7 spans | 3 LLM calls | 0 tool calls | 2 nodes
```

## Span Types

| Type | Description |
|------|-------------|
| `session` | Top-level session span |
| `graph` | LangGraph graph execution |
| `node` | LangGraph node execution |
| `llm_call` | LLM API call (OpenAI, Anthropic, Gemini) |
| `tool_call` | Tool/function execution |
| `scorer` | Metric evaluation span |

## Examples

```bash
# View span tree for a specific call
evalyn show-trace --id 47fe2576-03c3-4438-8708-b8ab38cf52e9

# View the most recent call
evalyn show-trace --last

# Limit tree depth to 3 levels
evalyn show-trace --last --max-depth 3

# Get call ID first, then view trace
evalyn list-calls --limit 1
evalyn show-trace --id <id-from-list>
```

## See Also

- [show-call](show-call.md) - Full call details including span tree
- [list-calls](list-calls.md) - List all captured calls
