# Metrics Guide

## Overview

Evalyn provides 50+ metrics in two categories:

| Type | Description | Examples |
|------|-------------|----------|
| **Objective** | Deterministic, no LLM | latency, token count, JSON validity |
| **Subjective** | LLM-judge based | helpfulness, toxicity, hallucination |

## Objective Metrics

| Metric | Description |
|--------|-------------|
| `latency_ms` | Response time in milliseconds |
| `cost_usd` | Estimated API cost |
| `token_length` | Output token count |
| `exact_match` | Exact match with expected |
| `bleu` | BLEU score vs reference |
| `rouge_l` | ROUGE-L score vs reference |
| `json_valid` | Valid JSON output |
| `regex_match` | Matches regex pattern |
| `url_count` | Number of URLs in output |
| `tool_call_count` | Number of tool calls |
| `output_nonempty` | Non-empty output |

## Subjective Metrics

| Metric | Description |
|--------|-------------|
| `helpfulness_accuracy` | Response is helpful and accurate |
| `instruction_following` | Follows given instructions |
| `toxicity_safety` | Free of toxic content |
| `hallucination_risk` | Factually grounded |
| `clarity_coherence` | Clear and well-structured |
| `completeness` | Addresses all aspects |
| `tone_appropriateness` | Appropriate tone |

## Metric Selection

### Basic Mode (No LLM)
```bash
evalyn suggest-metrics --target agent.py:func --mode basic
```
Fast heuristic based on function signature and traces.

### LLM Registry Mode
```bash
evalyn suggest-metrics --target agent.py:func --mode llm-registry --llm-mode api
```
LLM analyzes your function and selects from 50+ templates.

### Bundle Mode
```bash
evalyn suggest-metrics --target agent.py:func --mode bundle --bundle research-agent
```

**Available bundles:**
- `summarization` - For text summarization agents
- `orchestrator` - For multi-step orchestration
- `research-agent` - For research/retrieval agents

## Using Metrics

### From File
```bash
evalyn run-eval --dataset data/myapp --metrics metrics/suggested.json
```

### Multiple Metric Files
```bash
evalyn run-eval --dataset data/myapp --metrics "metrics/basic.json,metrics/custom.json"
```

### All Metrics
```bash
evalyn run-eval --dataset data/myapp --metrics-all
```

## Custom Metrics

```python
from evalyn_sdk import Metric, MetricSpec, MetricType, MetricResult

def my_metric(output: str, **kwargs) -> MetricResult:
    score = len(output) / 1000
    return MetricResult(
        metric_id="my_custom",
        score=score,
        passed=score > 0.5,
        details={"length": len(output)}
    )

spec = MetricSpec(
    id="my_custom",
    name="My Custom Metric",
    type=MetricType.OBJECTIVE,
    description="Custom length check"
)

metric = Metric(spec=spec, handler=my_metric)
registry.add_metric(metric)
```

## Metric Results

After `run-eval`, results are saved to `eval_runs/<timestamp>_<id>.json`:

```json
{
  "id": "run_abc123",
  "results": [
    {
      "item_id": "item_1",
      "metric_id": "helpfulness_accuracy",
      "score": 0.92,
      "passed": true,
      "details": {
        "reason": "Response is accurate and helpful"
      }
    }
  ],
  "summary": {
    "helpfulness_accuracy": {
      "pass_rate": 0.92,
      "avg_score": 0.89
    }
  }
}
```

## Reference-Based Metrics

Some metrics require expected output:
- `exact_match`, `bleu`, `rouge_l`, `rouge_1`, `rouge_2`
- `token_overlap_f1`, `jaccard_similarity`
- `numeric_mae`, `numeric_rmse`

If your dataset has no expected values, these are automatically excluded.
