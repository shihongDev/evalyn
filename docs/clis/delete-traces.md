# evalyn delete-traces

Delete traces from SQLite storage.

## Usage

```bash
evalyn delete-traces [-n N] [--id ID ...] [--db test|prod] [--yes]
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `-n N` | 1 | Number of latest traces to delete (if no --id) |
| `--id ID ...` | - | Specific trace ID(s) to delete (supports short IDs) |
| `--db` | test | Database to use: test or prod |
| `--yes`, `-y` | false | Skip confirmation prompt |

## Description

Deletes traces from the local SQLite storage. By default, operates on the test database for safety.

When deleting:
- Shows a preview of traces to be deleted
- Requires confirmation (unless --yes is passed)
- Shows extra warning when targeting prod database
- Deletes associated OpenTelemetry spans automatically

## Examples

### Delete the most recent trace from test db
```bash
evalyn delete-traces
```

### Delete latest 5 traces from test db
```bash
evalyn delete-traces -n 5
```

### Delete specific traces by ID
```bash
# Full ID
evalyn delete-traces --id 47fe2576-03c3-4438-8708-b8ab38cf52e9

# Short ID (prefix)
evalyn delete-traces --id 47fe25

# Multiple IDs
evalyn delete-traces --id abc123 def456 ghi789
```

### Delete from production with auto-confirm
```bash
evalyn delete-traces -n 3 --db prod --yes
```

## Sample Output

```
Traces to delete from test database:

ID | Function | Started
------------------------------------------------------------
47fe2576 | research_agent | 2025-01-15 08:13:37
abc12345 | research_agent | 2025-01-15 08:10:22
def67890 | research_agent | 2025-01-15 08:05:11

Total: 3 trace(s)

Delete 3 trace(s) from test? [y/N]: y

Deleted 3 trace(s) from test database.
```

### Production warning

```
Traces to delete from prod database:
...
*** WARNING: You are about to delete from PRODUCTION database! ***

Delete 3 trace(s) from prod? [y/N]:
```

## See Also

- [list-calls](list-calls.md) - List captured traces
- [show-call](show-call.md) - View trace details
- [show-projects](show-projects.md) - View project summaries
