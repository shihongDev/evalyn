# quickstart

Guided first-run experience for new users. Detects your agent framework, generates instrumentation snippets, creates config, and suggests next steps.

## Usage

```bash
evalyn quickstart
evalyn quickstart --agent-file agent.py
evalyn quickstart --run "python agent.py"
evalyn quickstart --run "python agent.py" --timeout 60
```

## Options

| Option | Description |
|--------|-------------|
| `--agent-file PATH` | Path to your agent Python file (skips auto-detection scan) |
| `--run COMMAND` | Command to run your agent (e.g. "python agent.py") |
| `--timeout SECONDS` | Timeout in seconds for agent run (default: 120) |

## Description

The `quickstart` command walks you through:

1. **Framework detection**: Scans Python files for known imports (OpenAI, LangChain, CrewAI, Anthropic, Google ADK)
2. **Instrumentation snippet**: Generates the correct `import evalyn_sdk` snippet for your framework
3. **Config creation**: Creates `evalyn.yaml` with sensible defaults
4. **Optional agent run**: Runs your agent to capture initial traces
5. **Metric suggestion**: Suggests metrics based on captured traces
6. **Next steps**: Prints instructions for the full evaluation pipeline

If multiple frameworks are detected in your project, quickstart will ask you to pick one.

## Examples

```bash
# Auto-detect framework and walk through setup
evalyn quickstart

# Specify the agent file directly
evalyn quickstart --agent-file src/my_agent.py

# Run the agent as part of quickstart
evalyn quickstart --run "python agent.py" --timeout 60
```

## See Also

- [init](init.md) - Initialize config file only
- [one-click](one-click.md) - Run the full pipeline automatically
- [workflow](workflow.md) - Show evaluation workflow and next steps
