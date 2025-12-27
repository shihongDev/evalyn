# evalyn suggest-metrics

Suggest evaluation metrics for a target function based on its signature and traces.

## Usage

```bash
evalyn suggest-metrics --target <file.py:func> [OPTIONS]
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--target SPEC` | Required | Target function (`file.py:func` or `module:func`) |
| `--mode MODE` | auto | Selection mode (see below) |
| `--llm-mode MODE` | api | LLM caller: `api` or `local` (ollama) |
| `--model NAME` | gemini-2.5-flash-lite | Model name |
| `--num-traces N` | 5 | Sample traces to analyze |
| `--num-metrics N` | - | Max metrics to return |
| `--bundle NAME` | - | Bundle name (when mode=bundle) |
| `--dataset PATH` | - | Save metrics to dataset folder |
| `--metrics-name NAME` | - | Custom metrics filename |

## Selection Modes

| Mode | Description | LLM Required |
|------|-------------|--------------|
| `basic` | Fast heuristic based on function signature | No |
| `llm-registry` | LLM picks from 50+ built-in templates | Yes |
| `llm-brainstorm` | LLM generates custom metric specs | Yes |
| `bundle` | Pre-configured metric set | No |
| `auto` | Uses function's `@eval` hints or defaults to `llm-registry` | Maybe |

## Bundles

| Bundle | Description |
|--------|-------------|
| `summarization` | ROUGE, BLEU, length, coherence |
| `orchestrator` | Tool usage, latency, error handling |
| `research-agent` | Accuracy, citations, completeness |

## Examples

### Basic heuristic (fast, no API key)
```bash
evalyn suggest-metrics --target agent.py:run_agent --mode basic
```

### LLM-powered selection from registry
```bash
evalyn suggest-metrics --target agent.py:run_agent --mode llm-registry
```

### LLM brainstorm custom metrics
```bash
evalyn suggest-metrics --target agent.py:run_agent --mode llm-brainstorm --model gpt-4
```

### Use a pre-defined bundle
```bash
evalyn suggest-metrics --target agent.py:run_agent --mode bundle --bundle research-agent
```

### Save to dataset folder
```bash
evalyn suggest-metrics --target agent.py:run_agent --dataset data/my-dataset --mode llm-registry
```

### Use local Ollama
```bash
evalyn suggest-metrics --target agent.py:run_agent --mode llm-registry --llm-mode local --model llama3.1
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

Saved to: data/my-dataset/metrics/llm-registry-20250115.json
```

## See Also

- [list-metrics](list-metrics.md) - View all available metric templates
- [build-dataset](build-dataset.md) - Build dataset first
- [run-eval](run-eval.md) - Run evaluation with metrics
