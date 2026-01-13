# select-metrics

LLM-guided selection of metrics from the built-in registry.

## Usage

```bash
evalyn select-metrics --target <module:func> --llm-caller <callable>
```

## Options

| Option | Description |
|--------|-------------|
| `--target TARGET` | (Required) Callable to analyze in the form `module:function` |
| `--llm-caller CALLABLE` | (Required) Callable that accepts a prompt and returns metric IDs or dicts |
| `--limit N` | Number of recent traces to include as examples (default: 5) |

## Description

The `select-metrics` command uses an LLM to intelligently select appropriate metrics from the built-in metric registry based on:

- The target function's signature and docstring
- Recent trace examples showing actual inputs/outputs
- The function's code metadata

This is an alternative to `suggest-metrics` that gives you more control over the LLM selection process by allowing you to provide a custom LLM caller.

## LLM Caller Interface

Your LLM caller should be a function that:
- Takes a prompt string as input
- Returns a list of metric IDs or metric dictionaries

```python
# Example LLM caller
def my_llm_caller(prompt: str) -> list:
    # Call your LLM API
    response = call_llm(prompt)
    # Parse and return metric IDs
    return ["helpfulness_accuracy", "toxicity_safety"]
```

## Output

```
Selected metrics:
- helpfulness_accuracy: [subjective] config={"rubric": [...], "threshold": 0.5}
- latency_ms: [objective] config={}
- output_nonempty: [objective] config={}
```

## Examples

```bash
# Select metrics using a custom LLM caller
evalyn select-metrics --target myapp.agent:run --llm-caller myapp.llm:select_metrics

# Include more trace examples
evalyn select-metrics --target myapp.agent:run --llm-caller myapp.llm:caller --limit 10
```

## See Also

- [suggest-metrics](suggest-metrics.md) - Suggest metrics with built-in LLM modes
- [list-metrics](list-metrics.md) - List all available metric templates
- [run-eval](run-eval.md) - Run evaluation with selected metrics
