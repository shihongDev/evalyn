# show-span

Display detailed information about a specific span within a traced call.

## Usage

```bash
evalyn show-span --call-id <call_id> --span <span_name_or_index>
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--call-id ID` | - | The call ID containing the span (required, supports short IDs) |
| `--span NAME` | - | Span name or index to display (required) |
| `--db` | prod | Database to use: prod or test |
| `--format` | table | Output format: table or json |

Both `--call-id` and `--span` are required.

## Description

Shows complete details for a single span, including:
- Span metadata (ID, type, status, parent)
- Timing information (start, end, duration)
- All attributes captured during execution

This is useful for:
- Debugging specific tool calls or LLM responses
- Extracting full input/output without truncation
- Inspecting span attributes for analysis

## Span Selection

The `--span` argument accepts:
- **Index**: Numeric index (0-based) from the span list
- **Name substring**: Partial match on span name (e.g., "llm_turn_1", "WebSearch", "Task")

If no exact match is found, available spans are listed.

## Sample Output

```
============================================================
Span: llm_turn_1
============================================================
ID        : fbe8df66-0c20-452d-94fc-6327796c182d
Type      : llm_call
Status    : ok
Parent    : ade63f1e-3bb2-438b-b70c-fe1ac52ac032
Started   : 2026-01-22 01:17:57.630773+00:00
Ended     : 2026-01-22 01:17:57.630786+00:00
Duration  : 0.01ms

Attributes:
  turn: 1
  model: claude-haiku-4-5-20251001
  provider: anthropic
  output_preview:
    Breaking into 4 research areas: current LLM agent evaluation
    frameworks, industry benchmark suites, enterprise evaluation
    infrastructure, and emerging evaluation methods.
  output:
    Breaking into 4 research areas: current LLM agent evaluation
    frameworks, industry benchmark suites, enterprise evaluation
    infrastructure, and emerging evaluation methods. Spawning researchers now.
============================================================
```

### Tool Call Span

```
============================================================
Span: WebSearch
============================================================
ID        : c389ae3c-7667-4b5a-81b8-373f77af81ae
Type      : tool_call
Status    : ok
Parent    : ade63f1e-3bb2-438b-b70c-fe1ac52ac032
Started   : 2026-01-22 01:11:06.498753+00:00
Ended     : 2026-01-22 01:24:57.144487+00:00
Duration  : 830645.73ms

Attributes:
  tool_name: WebSearch
  tool_use_id: toolu_01Cy7zwPP2A2k83kuBvW3nE1
  input:
    {'query': 'LLM agent evaluation frameworks tools 2025'}
  session_id: aadedbe3-d73c-4bdd-a3e7-eb7f8b25f67b
  input_size: 56
  input_truncated: False
  output_size: 3714
  output_truncated: False
  output:
    {'query': '...', 'results': [{'title': '...', 'url': '...'}]}
  executing_subagent: researcher
============================================================
```

## Examples

```bash
# View span by name
evalyn show-span --call-id 2567ea62 --span "llm_turn_1"

# View span by index
evalyn show-span --call-id 2567ea62 --span 0

# View a tool call span
evalyn show-span --call-id 2567ea62 --span "WebSearch"

# View a Task (subagent) span
evalyn show-span --call-id 2567ea62 --span "Task"

# JSON output for scripting
evalyn show-span --call-id 2567ea62 --span "llm_turn_1" --format json
```

### List Available Spans

If the span name doesn't match, available spans are listed:

```bash
$ evalyn show-span --call-id 2567ea62 --span "nonexistent"
Available spans:
  [0] session: chat
  [1] tool_call: Task
  [2] tool_call: Task
  [3] tool_call: WebSearch
  [4] llm_call: llm_turn_1
  ...
Error: Span 'nonexistent' not found. Use index or name substring.
```

## See Also

- [show-trace](show-trace.md) - View span tree with optional verbose details
- [show-call](show-call.md) - Full call details including span tree
- [list-calls](list-calls.md) - List all captured calls
