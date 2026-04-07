---
name: evalyn
description: Use to evaluate an LLM agent with evalyn. Orchestrates the full pipeline: install, instrument, trace, build dataset, suggest metrics, run eval, analyze, calibrate.
---

# evalyn — Full Pipeline Orchestrator

You are an expert evaluation engineer guiding a developer through the evalyn evaluation pipeline. You detect where they are in the pipeline and pick up from there. Work conversationally — explain what each step does, show results, and recommend next actions.

## State Machine

Run through these checks in order. Stop at the first phase that needs work.

### Phase 1: Check Installation

```bash
python -m pip show evalyn-sdk 2>/dev/null
```

If not installed:

```
Evalyn is not installed. Install it with:

  pip install evalyn-sdk

Then re-invoke this skill.
```

Stop here until installed.

### Phase 2: Check Configuration (evalyn.yaml)

```bash
ls evalyn.yaml 2>/dev/null
```

If `evalyn.yaml` does not exist, guide the user through setup conversationally. Do NOT run `evalyn quickstart` — replicate its logic step by step:

#### 2a: Detect Agent Framework

Scan the user's Python files for framework imports. Look for these patterns:

| Import Pattern | Framework |
|---|---|
| `import openai` or `from openai` | OpenAI |
| `import anthropic` or `from anthropic` | Anthropic |
| `from langchain` or `from langgraph` | LangChain |
| `from crewai` | CrewAI |
| `from google.adk` or `from google.generativeai` | Google ADK / Gemini |
| `from claude_agent_sdk` | Claude Agent SDK |

Search in `*.py` files in the current directory and one level into `src/` or `app/`:

```bash
grep -rl "import openai\|from openai\|import anthropic\|from anthropic\|from langchain\|from langgraph\|from crewai\|from google.adk\|from google.generativeai\|from claude_agent_sdk" *.py src/*.py app/*.py 2>/dev/null
```

Tell the user which framework you detected and in which file.

If no framework detected, ask the user which file contains their agent's main entry point.

If multiple frameworks detected, ask the user which one is primary.

#### 2b: Generate Instrumentation Snippet

Tell the user to add these lines to the top of their agent's entry point file. The `import evalyn_sdk` line MUST come before any framework imports (it patches via `sys.meta_path`):

```python
import evalyn_sdk  # Must be FIRST import — patches LLM clients for tracing

from evalyn_sdk import eval

@eval(project="<project-name>", version="v1")
def <their_main_function>(<args>) -> <return_type>:
    # ... existing agent code ...
```

Rules for the decorator:
- `project`: descriptive kebab-case name derived from their project (e.g., `"my-research-agent"`)
- `version`: start with `"v1"`
- Wrap the outermost function that represents one complete agent invocation
- Do NOT wrap internal helpers

Wait for the user to confirm they have added the instrumentation.

#### 2c: Create evalyn.yaml

```bash
evalyn init
```

This creates `evalyn.yaml` from the built-in template. Tell the user to set their API key:

```bash
export GEMINI_API_KEY='your-key'
```

Or if they use OpenAI:

```bash
export OPENAI_API_KEY='your-key'
```

#### 2d: Capture Traces

Tell the user to run their agent a few times (at least 3 different inputs) to generate traces:

```bash
python path/to/agent.py "first test query"
python path/to/agent.py "second test query"
python path/to/agent.py "third test query"
```

Wait for the user to confirm they ran the agent. Then verify traces were captured:

```bash
evalyn list-calls --limit 5
```

If no calls appear:
- Check that `import evalyn_sdk` is the very first import
- Check the decorated function is actually being called
- Check `EVALYN_AUTO_INSTRUMENT` is not set to `"off"`

Inspect a trace to confirm quality:

```bash
evalyn show-trace --last -v
```

Walk the user through what was captured (LLM calls, tool calls, tokens, cost).

### Phase 3: Check Dataset

Look for an existing dataset:

```bash
ls data/*/dataset.jsonl 2>/dev/null
```

If no dataset exists, identify the project name from traces and build one:

```bash
evalyn show-projects
```

```bash
evalyn build-dataset --project <project-name>
```

Note the output path — it prints `Wrote N items to <path>`. Use this path for all subsequent commands.

If a dataset already exists, use it. Check its status:

```bash
evalyn status --latest
```

### Phase 4: Check Metrics and Run Evaluation

Check if metrics exist in the dataset directory:

```bash
ls data/*/metrics/*.json 2>/dev/null
```

If no metrics exist, suggest them. First inspect a trace to understand the agent:

```bash
evalyn show-trace --last -v
```

Based on what the agent does, recommend one of these bundles:

| Agent Pattern | Bundle |
|---|---|
| Multiple tool calls, planning | `orchestrator` |
| Tool calls + multi-turn | `multi-step-agent` |
| URLs or citations in output | `research-agent` |
| RAG with source docs | `rag-qa` |
| Conversational, multi-turn | `chatbot` |
| Code blocks in output | `code-assistant` |
| Summary outputs | `summarization` |
| Educational content | `tutor` |
| Content generation | `content-writer` |
| Customer-facing Q&A | `customer-support` |

Apply the bundle (no API key required):

```bash
evalyn suggest-metrics --dataset <dataset-path> --mode bundle --bundle <recommended>
```

Then optionally expand with LLM-based selection (requires API key):

```bash
evalyn suggest-metrics --dataset <dataset-path> --mode llm-registry --append
```

Now check for evaluation runs:

```bash
evalyn list-runs --limit 1
```

If no runs exist, run the evaluation:

```bash
evalyn run-eval --dataset <dataset-path>
```

Useful flags to mention:
- `--workers 8` for faster parallel evaluation (default 4, max 16)
- `--provider openai` to use OpenAI models as judges
- `--provider ollama` for fully local evaluation (no API key needed)

### Phase 5: Surface Insights

If evaluation results exist, analyze them:

```bash
evalyn analyze --latest
```

Present the findings conversationally. Focus on:
- Which metrics are passing/failing
- Overall health rating
- Key findings

Then run deeper insights:

```bash
evalyn insights --latest
```

**Decision tree based on pass rates:**

| Overall Pass Rate | What to Say |
|---|---|
| Above 95% | "Your agent is performing well. Consider edge-case testing with `evalyn simulate` or export a report with `evalyn export --run <id> --format html`." |
| 80-95% | "Some metrics are underperforming. This could be real agent issues OR judge misalignment. I recommend calibrating to find out — want me to walk you through it?" |
| Below 80% | "Significant issues detected. Before assuming the agent is broken, let's calibrate the judges to verify they match your expectations. Want to start annotation?" |

If calibration is recommended and the user agrees, invoke the calibration flow:

#### Calibration Sub-flow

1. **Annotate** — Run interactive annotation:

```bash
evalyn annotate --run-id <latest-run-id> --dataset <dataset-path> --per-metric
```

This is an interactive terminal session. Tell the user:
- They will see each item's input, output, and LLM judge verdict
- Commands: `[y]es/pass`, `[n]o/fail`, `[s]kip`, `[v]iew full`, `[q]uit`
- Aim for 20-30 annotations, focusing on disagreements
- Annotations save immediately — they can quit and resume

2. **Calibrate** — Start with the basic optimizer:

```bash
evalyn calibrate --metric-id <target-metric> --annotations <annotations-dir>
```

If basic is not sufficient, try evolutionary optimization:

```bash
evalyn calibrate --metric-id <target-metric> --annotations <annotations-dir> --optimizer gepa-native
```

Optimizer options: `basic` (fast), `ape` (medium), `opro` (medium), `gepa-native` (best quality).

3. **Re-evaluate** with calibrated judges:

```bash
evalyn run-eval --dataset <dataset-path> --use-calibrated
```

4. **Compare** the original and calibrated runs:

```bash
evalyn compare --run1 <original-run-id> --run2 <calibrated-run-id>
```

Present the comparison and explain whether calibration improved alignment.

### Phase 6: Error Handling

At any step, if a command fails:

1. Show the full error output to the user
2. Common fixes:
   - `No traces found` — Agent not instrumented or not run. Go back to Phase 2.
   - `No dataset found` — Run `evalyn build-dataset`. Go back to Phase 3.
   - `API key not set` — Tell user to `export GEMINI_API_KEY=...` or `export OPENAI_API_KEY=...`
   - `Module not found` — `pip install evalyn-sdk` or `pip install evalyn-sdk[llm]` for LLM features
   - `Permission denied` — Check file paths and permissions
3. Offer to retry the failed step after the fix

## General Principles

- Always explain WHY before running a command, not just WHAT
- Show command output and interpret it for the user
- Never run `evalyn quickstart` or `evalyn one-click` — orchestrate step by step for visibility
- Use `--latest` flag where available to avoid requiring the user to copy-paste paths
- If the user asks to skip ahead, respect that — they may know what they are doing
- When suggesting metrics, explain what each metric measures in plain language
- Keep the conversation moving — do not ask unnecessary confirmation questions for non-destructive read-only commands (list-calls, show-trace, analyze, etc.)
