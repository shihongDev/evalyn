---
name: evalyn-setup
description: Use when setting up evalyn evaluation for an LLM agent project, instrumenting agent code, or adding the evalyn decorator
---

# evalyn-setup

## Overview

Guide a developer through instrumenting their LLM agent with evalyn so traces are captured for evaluation.

## Pre-flight

Check evalyn is installed:

```bash
uv pip show evalyn-sdk
```

If not installed, install from the evalyn repo:

```bash
uv pip install -e "./sdk[llm]"
```

Or if user has a separate project:

```bash
uv pip install evalyn-sdk
```

## Step 1: Detect Agent Framework

Scan the user's agent code for imports to determine the framework:

Supported frameworks (all auto-instrumented - decorator is sufficient):

`langchain`, `langgraph`, `anthropic`, `openai`, `google.generativeai`, `google.adk`, `claude_agent_sdk`

If no recognized framework: the decorator still works for any Python function, but LLM calls won't have token/cost details.

## Step 2: Add the Decorator

Add to the agent's main entry function. The decorator is named `eval` and is imported from `evalyn_sdk`:

```python
from evalyn_sdk import eval

@eval(project="<project-name>", version="v1")
def agent_function(query: str) -> str:
    # existing agent code
    ...
```

Rules:
- `project`: descriptive kebab-case name (e.g., "my-research-agent")
- `version`: tracks iterations, start with "v1"
- Wrap the outermost function that represents one agent invocation
- Do NOT wrap internal helper functions
- Optional `name` parameter overrides the display name (defaults to function name)

Real example from the codebase:

```python
from evalyn_sdk import eval

@eval(project="gemini-deep-research-agent", version="v1", name="research_agent")
def run_agent(question: str) -> str:
    ...
```

## Step 3: Run the Agent

Tell the user to run their agent with at least 3 different inputs to generate traces:

```bash
python path/to/agent.py "first test query"
python path/to/agent.py "second different query"
python path/to/agent.py "third varied query"
```

## Step 4: Verify Traces

```bash
evalyn list-calls --limit 5
```

Expected: table showing captured calls with project name, status, duration.

If no calls appear:
- Check the decorator is on the correct function
- Check the function is actually being called
- Check `EVALYN_AUTO_INSTRUMENT` is not set to "off"

## Step 5: Inspect a Trace

```bash
evalyn show-trace --last -v
```

This shows the hierarchical span tree: LLM calls, tool calls, token counts, costs. Walk the user through what was captured.

## Hand-off

"Your agent is instrumented and generating traces. Run it a few more times with varied inputs to build a representative sample, then invoke `evalyn-eval` to build a dataset and run evaluation."
