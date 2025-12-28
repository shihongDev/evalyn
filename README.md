# Evalyn

**Trace, evaluate, and calibrate your LLM agents — fully local, fully yours.**

## Why Evalyn?

| | |
|---|---|
| **Fully Local** | All data stays on your machine. SQLite storage, no cloud dependencies. |
| **Zero Config** | Just `import evalyn_sdk` — LLM calls auto-captured with tokens & cost. |
| **LLM Judges** | 50+ metrics including LLM-based judges for quality assessment. |
| **Human Calibration** | Align LLM judges with human feedback through annotation workflow. |
| **One Command** | Run the entire pipeline with `evalyn one-click`. |

## Install

```bash
pip install -e ".[dev,llm]"
```

## 30-Second Start

```python
from evalyn_sdk import eval

@eval(project="myapp", version="v1")
def my_agent(query: str) -> str:
    return call_llm(query)  # LLM calls auto-captured
```

```bash
export GEMINI_API_KEY="your-key"
python my_agent.py                           # Run agent, traces captured
evalyn one-click --project myapp             # Full evaluation pipeline
```

## Auto-Instrumentation

**Just import `evalyn_sdk`** — LLM calls are captured automatically:

```python
import evalyn_sdk  # Auto-patches OpenAI, Anthropic, Gemini, LangChain

# Your normal code - no changes needed
response = openai.chat.completions.create(model="gpt-4o", messages=[...])
# ^ Automatically logged with: tokens, cost, duration, request/response
```

**Captured automatically:**
- All LLM API calls (OpenAI, Anthropic, Google Gemini)
- Token usage (input/output)
- Cost in USD
- Duration
- Errors

**Disable if needed:**
```bash
export EVALYN_AUTO_INSTRUMENT=off
```

**For internal functions**, use `@trace`:
```python
from evalyn_sdk import trace

@trace
def process_results(data):
    # Logged as trace event within parent @eval call
    return transform(data)
```

## The Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│  1. COLLECT                                                      │
│                                                                  │
│     @eval decorator  →  TRACE (SQLite)  →  DATASET (JSONL)       │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  2. EVALUATE & CALIBRATE (iterate until aligned)                 │
│                                                                  │
│          ┌────────────────────────────────────┐                  │
│          │                                    │                  │
│          ▼                                    │                  │
│     EVALUATE  ───→  ANNOTATE  ───→  CALIBRATE │                  │
│    (LLM Judge)      (Human)       (Optimize)  │                  │
│          │                                    │                  │
│          └─────── RE-EVALUATE ◄───────────────┘                  │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  3. EXPAND                                                       │
│                                                                  │
│     SIMULATE  →  Generate synthetic queries  →  Back to Step 2   │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

**Or just run:**
```bash
evalyn one-click --project myapp
```

## Key Commands

| Command | What it does |
|---------|--------------|
| `evalyn one-click --project X` | Run full pipeline |
| `evalyn list-calls` | View captured traces |
| `evalyn build-dataset --project X` | Create dataset from traces |
| `evalyn suggest-metrics --project X` | Get metric recommendations |
| `evalyn run-eval --latest` | Run evaluation |
| `evalyn annotate --latest` | Human annotation (interactive) |
| `evalyn calibrate --metric-id X` | Calibrate LLM judge |

## Metrics

**Objective** (deterministic):
- `latency_ms`, `output_nonempty`, `json_valid`, `token_length`
- `bleu`, `rouge_l`, `jaccard_similarity` (need reference)

**Subjective** (LLM judge):
- `helpfulness_accuracy`, `hallucination_risk`, `toxicity_safety`
- `completeness`, `coherence`, `instruction_following`

```bash
evalyn list-metrics                              # See all 50+ metrics
evalyn suggest-metrics --project X --mode basic  # Fast heuristic
evalyn suggest-metrics --project X --mode llm-registry   # LLM picks from registry
evalyn suggest-metrics --project X --mode llm-brainstorm # Custom metrics with rubrics
```

**Brainstorm mode** generates custom subjective metrics tailored to your function's behavior.

## Calibration

LLM judges aren't perfect. Calibrate them with human feedback:

```bash
evalyn annotate --latest               # Label samples (pass/fail)
evalyn calibrate --metric-id helpfulness_accuracy --annotations ann.jsonl
evalyn run-eval --latest --use-calibrated  # Use improved prompts
```

Output: Precision, Recall, F1, Cohen's Kappa + optimized rubric.

## Configuration

```bash
evalyn init                            # Create evalyn.yaml
export GEMINI_API_KEY="your-key"       # Or OPENAI_API_KEY
```

## Documentation

| Guide | Description |
|-------|-------------|
| [docs/cli.md](docs/cli.md) | CLI reference |
| [docs/commands/](docs/commands/) | Detailed command docs |
| [docs/metrics.md](docs/metrics.md) | Metrics guide |
| [docs/calibration.md](docs/calibration.md) | Calibration workflow |

## Example

See [`example_agent/`](example_agent/) for a LangGraph integration.

## License

MIT
