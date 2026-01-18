# evalyn list-metrics

List all available built-in metric templates.

## Usage

```bash
evalyn list-metrics
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--format` | table | Output format: table or json |

## Output

Shows all available metric templates organized by type:
- **Objective**: Deterministic, computed metrics
- **Subjective**: LLM-judge based metrics

## Examples

### List all metrics
```bash
evalyn list-metrics
```

## Sample Output

```
OBJECTIVE METRICS (73 total)
============================
  Category: efficiency
    latency_ms          | Measure execution latency in milliseconds
    cost                | Estimated cost in USD
    token_length        | Count tokens in output
    compression_ratio   | Ratio of output length to input length

  Category: structure
    json_valid          | Checks whether output parses as JSON
    json_schema_keys    | Validates JSON has required keys
    regex_match         | Matches output against regex pattern
    xml_valid           | Checks XML validity
    syntax_valid        | Checks Python syntax validity
    output_nonempty     | PASS if output is not empty

  Category: correctness (reference-based)
    bleu                | BLEU score (requires reference)
    rouge_l             | ROUGE-L score (requires reference)
    rouge_1             | ROUGE-1 score (requires reference)
    token_overlap_f1    | Token overlap F1 (requires reference)
    levenshtein_similarity | Levenshtein edit distance similarity

  Category: robustness
    tool_call_count     | Counts tool/function calls
    llm_call_count      | Counts LLM API calls
    tool_success_ratio  | Ratio of successful tool calls

  ... (73 metrics total)

SUBJECTIVE METRICS (60 total)
=============================
  Category: safety
    toxicity_safety      | PASS if output is safe
    pii_safety           | PASS if no PII exposed
    manipulation_resistance | Resists jailbreak attempts

  Category: correctness
    helpfulness_accuracy | Helpful and accurate response
    factual_accuracy     | Factual claims are correct
    technical_accuracy   | Code/math/science is correct

  Category: agent
    reasoning_quality    | Clear logical reasoning
    tool_use_appropriateness | Tools used correctly
    planning_quality     | Coherent multi-step planning

  Category: domain
    medical_accuracy     | Medical info is accurate
    legal_compliance     | Legal info is qualified
    financial_prudence   | Financial advice is prudent

  Category: conversation
    context_retention    | Retains conversation context
    memory_consistency   | Consistent across turns

  ... (60 metrics total)
```

## Note on Reference-Based Metrics

Some metrics require a reference/expected value in the dataset:
- `bleu`, `rouge_*`, `token_overlap_f1`, `jaccard_similarity`
- `numeric_mae`, `numeric_rmse`, `numeric_rel_error`

These are automatically excluded when using `suggest-metrics` on datasets without reference values.

## See Also

- [suggest-metrics](suggest-metrics.md) - Get recommendations for your function
- [run-eval](run-eval.md) - Run evaluation with metrics
