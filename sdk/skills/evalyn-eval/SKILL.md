---
name: evalyn-eval
description: Use when building evaluation datasets, selecting metrics, or running evaluations on an LLM agent project with evalyn
---

# evalyn-eval

## Overview

Build a dataset from traces, auto-recommend metrics based on trace analysis, and run evaluation. This skill reads actual trace data to make metric recommendations rather than asking abstract questions.

## Pre-flight

1. Verify traces exist:

```bash
evalyn list-calls --limit 5
```

If no traces: "You need to instrument your agent first. Invoke `evalyn-setup`."

2. Check if a dataset already exists:

```bash
ls data/*/dataset.jsonl 2>/dev/null
```

If dataset exists, skip to Step 2.

## Step 1: Build Dataset

Identify the project name from the `evalyn list-calls` output (project column).

```bash
evalyn build-dataset --project <project-name>
```

Capture the output path - it prints "Wrote N items to <path>". Use this path for all subsequent commands.

## Step 2: Auto-Recommend Metrics

Inspect a trace to understand the agent's behavior:

```bash
evalyn show-trace --last -v
```

Analyze the trace structure and recommend a bundle. Evalyn has 17 curated metric bundles:

| Trace Pattern | Recommended Bundle | Why |
|--------------|-------------------|-----|
| Multiple tool calls, planning steps | `orchestrator` | Agent orchestrates tools |
| Tool calls + multi-turn context | `multi-step-agent` | Complex agent workflow |
| URLs or citations in output | `research-agent` | Research/retrieval agent |
| RAG retrieval spans, source docs | `rag-qa` | Retrieval-augmented QA |
| Conversational, multi-turn | `chatbot` | Dialog agent |
| Code blocks in output | `code-assistant` | Code generation agent |
| Short summary outputs | `summarization` | Summarization agent |
| Educational/tutorial content | `tutor` | Teaching agent |
| Content generation, blog posts | `content-writer` | Content creation |
| Customer-facing Q&A | `customer-support` | Support agent |

To see all available bundles:

```bash
evalyn suggest-metrics --mode bundle --help
```

Apply the recommended bundle:

```bash
evalyn suggest-metrics --dataset <path> --mode bundle --bundle <recommended>
```

Then expand coverage with LLM-based selection from the full 130+ metric registry:

```bash
evalyn suggest-metrics --dataset <path> --mode llm-registry --append
```

This two-pass approach gives a solid base (curated bundle) plus tailored additions (LLM picks from full registry).

### Available metric modes

| Mode | What it does | Speed | API key needed |
|------|-------------|-------|----------------|
| `basic` | Heuristic-based suggestion | Instant | No |
| `bundle` | Preset metric bundles (17 available) | Instant | No |
| `llm-registry` | LLM picks from 130+ built-in metrics | ~10s | Yes |
| `llm-brainstorm` | LLM generates custom metrics | ~10s | Yes |

Do NOT use modes like `agent`, `rag`, or `classify` - those do not exist.

## Step 3: Run Evaluation

```bash
evalyn run-eval --dataset <path>
```

This will:
- Load metrics from the dataset's `metrics/` directory
- Run objective metrics (instant) and subjective metrics (LLM judge calls)
- Generate `results.json` and `report.html` in the `eval_runs/` directory
- Print a summary table with pass rates and scores

Note the run ID from the output for analysis.

Useful flags:
- `--workers 8`: increase parallel workers (default 4, max 16)
- `--provider openai`: use OpenAI instead of Gemini for LLM judges
- `--provider ollama`: use local Ollama models

## Hand-off

"Evaluation complete. Invoke `evalyn-analyze` to dig into the results, identify failures, and get recommendations."
