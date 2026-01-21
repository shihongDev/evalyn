# evalyn suggest-metrics

Suggest evaluation metrics for a target function based on its signature and traces.

## Usage

```bash
# Preferred: use project name (no module loading required)
evalyn suggest-metrics --project <name> [OPTIONS]

# Alternative: specify function directly
evalyn suggest-metrics --target <file.py:func> [OPTIONS]
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--project NAME` | - | Project name (use `evalyn show-projects` to see available) |
| `--version V` | - | Filter by version (optional, used with `--project`) |
| `--target SPEC` | - | Target function (`file.py:func` or `module:func`) |
| `--mode MODE` | auto | Selection mode (see below) |
| `--scope SCOPE` | all | Filter by scope: `overall`, `llm_call`, `tool_call`, `trace`, or `all` |
| `--llm-mode MODE` | api | LLM caller: `api` or `local` (ollama) |
| `--model NAME` | gemini-2.5-flash-lite | Model name |
| `--api-base URL` | - | Custom API base URL for `--llm-mode api` |
| `--api-key KEY` | - | API key override for `--llm-mode api` |
| `--llm-caller CALLABLE` | - | Optional callable path that accepts a prompt and returns metric dicts |
| `--num-traces N` | 5 | Sample traces to analyze |
| `-n`, `--num-metrics N` | 5 | Max metrics to return |
| `--bundle NAME` | - | Bundle name (when mode=bundle) |
| `--dataset PATH` | - | Save metrics to dataset folder |
| `--latest` | false | Use the most recently modified dataset |
| `--metrics-name NAME` | - | Custom metrics filename |
| `--format` | table | Output format: table or json |
| `--append` | false | Append to existing metrics.json instead of overwriting |

## Selection Modes

| Mode | Description | Output | LLM Required |
|------|-------------|--------|--------------|
| `basic` | Fast heuristic based on function signature | Objective + Subjective | No |
| `llm-registry` | LLM picks from 130+ built-in templates | Objective + Subjective | Yes |
| `llm-brainstorm` | LLM generates custom metrics with rubrics | **Subjective only** | Yes |
| `bundle` | Pre-configured metric set | Objective + Subjective | No |
| `auto` | Uses function's `@eval` hints or defaults to `llm-registry` | Varies | Maybe |

### Brainstorm Mode

Brainstorm mode generates **custom subjective metrics** tailored to your function. The LLM analyzes your function's traces and creates evaluation criteria with custom rubrics.

```bash
evalyn suggest-metrics --project myapp --mode llm-brainstorm --num-metrics 4
```

Example output:
```
- answer_completeness [subjective] :: Evaluates if the answer fully addresses the user's question
- clarity_and_structure [subjective] :: Assesses readability and logical flow
- relevance_and_focus [subjective] :: Determines if the answer stays on topic
```

Each metric includes a custom rubric used by the LLM judge at eval time.

> **Note:** Brainstorm only generates subjective metrics because custom objective metrics require code implementation. For objective metrics, use `--mode bundle` or `--mode llm-registry`.

## Bundles

17 curated bundles for common GenAI use cases:

| Bundle | Description |
|--------|-------------|
| **Conversational AI** | |
| `chatbot` | Safety, helpfulness, multi-turn memory |
| `customer-support` | Empathy, patience, escalation handling |
| **Content Generation** | |
| `content-writer` | Style, engagement, readability |
| `summarization` | Compression, reference overlap, grounding |
| `creative-writer` | Originality, engagement, vocabulary diversity |
| **Knowledge & Research** | |
| `rag-qa` | Grounding, citations, factual accuracy |
| `research-agent` | Citations, grounding, tool use |
| `tutor` | Pedagogical clarity, examples, patience |
| **Code & Technical** | |
| `code-assistant` | Syntax validity, complexity, technical accuracy |
| `data-extraction` | JSON validity, schema compliance |
| **Agents & Orchestration** | |
| `orchestrator` | Tool success, planning, error handling |
| `multi-step-agent` | Planning, context retention, memory |
| **High-Stakes Domains** | |
| `medical-advisor` | Medical accuracy, safety, ethics |
| `legal-assistant` | Legal compliance, citations, accuracy |
| `financial-advisor` | Financial prudence, safety, ethics |
| **Safety & Translation** | |
| `moderator` | Toxicity, bias, PII, manipulation |
| `translator` | BLEU, Levenshtein, cultural sensitivity |

## Examples

### Using project name (recommended)
```bash
# First, see available projects
evalyn show-projects

# Then suggest metrics by project
evalyn suggest-metrics --project myapp --mode basic
```

### Basic heuristic (fast, no API key)
```bash
evalyn suggest-metrics --project myapp --mode basic
```

### LLM-powered selection from registry
```bash
evalyn suggest-metrics --project myapp --mode llm-registry
```

### LLM brainstorm custom metrics
```bash
evalyn suggest-metrics --project myapp --mode llm-brainstorm --num-metrics 5
```

### Use a pre-defined bundle
```bash
evalyn suggest-metrics --project myapp --mode bundle --bundle research-agent
```

### Filter by scope
```bash
# Only metrics that evaluate the final output
evalyn suggest-metrics --project myapp --mode bundle --bundle orchestrator --scope overall

# Only trace-level aggregate metrics (counts, ratios)
evalyn suggest-metrics --project myapp --mode bundle --bundle orchestrator --scope trace
```

### Save to dataset folder
```bash
evalyn suggest-metrics --project myapp --dataset data/my-dataset --mode llm-brainstorm --metrics-name custom
```

### Use local Ollama
```bash
evalyn suggest-metrics --project myapp --mode llm-registry --llm-mode local --model llama3.1
```

### Using target function (alternative)
```bash
evalyn suggest-metrics --target agent.py:run_agent --mode basic
```

## Sample Output

```
Analyzing function: run_agent
  Signature: (query: str) -> dict
  Sample traces: 5

Suggested metrics (llm-registry):
- latency_ms [objective] :: Measures execution time
- output_nonempty [objective] :: Checks output is not empty
- helpfulness_accuracy [subjective] :: LLM judges if response is helpful and accurate
- hallucination_risk [subjective] :: LLM checks for unsupported claims

Saved to: data/my-dataset/metrics/metrics.json
```

## See Also

- [list-metrics](list-metrics.md) - View all available metric templates
- [build-dataset](build-dataset.md) - Build dataset first
- [run-eval](run-eval.md) - Run evaluation with metrics
