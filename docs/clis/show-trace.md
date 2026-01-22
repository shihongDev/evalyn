# show-trace

Display the hierarchical span tree for a traced call (Phoenix-style visualization).

## Usage

```bash
evalyn show-trace --id <call_id>
evalyn show-trace --last
evalyn show-trace --id <call_id> --verbose
evalyn show-trace --id <call_id> --verbose --full
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--id ID` | - | The call ID to display spans for (supports short IDs) |
| `--last` | false | Show the most recent call |
| `--db` | prod | Database to use: prod or test |
| `--max-depth N` | unlimited | Maximum depth of span tree to display |
| `-v, --verbose` | false | Show detailed input/output for each span |
| `--full` | false | Show full content without truncation (use with --verbose) |

One of `--id` or `--last` is required.

## Description

Shows a hierarchical tree view of all spans captured during a function call, including:
- LLM calls (with token counts and model names)
- Tool calls (with input/output in verbose mode)
- Graph nodes (for LangGraph agents)
- User messages (captured via query() patching)
- Session spans
- Duration for each span

This is a simplified view compared to `show-call` - it focuses only on the span tree.

## Sample Output

### Basic Output

```
Trace: run_agent (5.2s) [OK]
Call ID: 47fe2576-03c3-4438-8708-b8ab38cf52e9
Session: abc123-session

└── session:main (5.2s)
    ├── graph:research_graph (4.8s)
    │   ├── node:search_node (1.2s)
    │   │   └── llm_call gemini:gemini-2.5-flash [200>150 tokens] (1.1s)
    │   └── node:summarize_node (2.3s)
    │       └── llm_call openai:gpt-4 [500>300 tokens] (2.2s)
    └── llm_call anthropic:claude-3 [100>50 tokens] (0.4s)

Summary: 7 spans | 3 LLM calls | 0 tool calls | 2 nodes
```

### Verbose Output (--verbose)

```
Trace: chat (1461.4s) [OK]
Call ID: 2567ea62-3a66-4f5c-a02b-55106e30d2f1

└── chat (1461.4s)
    ├── user_message user_input (0ms)
    │       content: Research current solutions for LLM agent evaluation infrastructure
    ├── tool_call Task (1793.6s)
    │       input: {'description': 'LLM agent benchmarks', 'prompt': 'Research...', 'subagent_type': 'researcher'}
    │       output: {'status': 'completed', 'content': [{'type': 'text', 'text': "I've completed..."}]}
    ├── tool_call WebSearch (595.7s)
    │       input: {'query': 'LLM agent evaluation frameworks'}
    │       output: {'query': '...', 'results': [{'title': '...', 'url': '...'}]}
    │       subagent: researcher
    ├── llm_call llm_turn_1 (0ms)
    │       model: claude-haiku-4-5-20251001
    │       output: Breaking into 4 research areas: current LLM agent evaluation...
```

### Full Output (--verbose --full)

With `--full`, content is not truncated - useful for debugging or extracting complete tool outputs.

## Span Types

| Type | Description |
|------|-------------|
| `session` | Top-level session span |
| `graph` | LangGraph graph execution |
| `node` | LangGraph node execution |
| `llm_call` | LLM API call (OpenAI, Anthropic, Gemini, Claude Agent SDK) |
| `tool_call` | Tool/function execution |
| `user_message` | User input captured from query() |
| `scorer` | Metric evaluation span |

## Examples

```bash
# View span tree for a specific call
evalyn show-trace --id 47fe2576-03c3-4438-8708-b8ab38cf52e9

# View using short ID
evalyn show-trace --id 47fe25

# View the most recent call
evalyn show-trace --last

# Limit tree depth to 3 levels
evalyn show-trace --last --max-depth 3

# Verbose mode - show input/output for each span
evalyn show-trace --last --verbose

# Full verbose mode - no truncation
evalyn show-trace --last --verbose --full

# Get call ID first, then view trace
evalyn list-calls --limit 1
evalyn show-trace --id <id-from-list>
```

## See Also

- [show-call](show-call.md) - Full call details including span tree
- [show-span](show-span.md) - View details of a single span
- [list-calls](list-calls.md) - List all captured calls
