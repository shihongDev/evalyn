# evalyn list-metrics

List all available built-in metric templates.

## Usage

```bash
evalyn list-metrics
```

## Options

None.

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
OBJECTIVE METRICS
=================
latency_ms          | Measures execution time in milliseconds
output_nonempty     | Checks if output is not empty
json_valid          | Validates JSON structure
token_length        | Counts tokens in output
tool_call_count     | Counts tool/function calls
llm_call_count      | Counts LLM API calls
url_count           | Counts URLs in output
bleu                | BLEU score (requires reference)
rouge_l             | ROUGE-L score (requires reference)
rouge_1             | ROUGE-1 score (requires reference)
rouge_2             | ROUGE-2 score (requires reference)
token_overlap_f1    | Token overlap F1 (requires reference)
jaccard_similarity  | Jaccard similarity (requires reference)
numeric_mae         | Mean absolute error (requires reference)
numeric_rmse        | Root mean square error (requires reference)

SUBJECTIVE METRICS (LLM Judge)
==============================
helpfulness_accuracy | Judges if response is helpful and accurate
toxicity_safety      | Checks for toxic or unsafe content
hallucination_risk   | Detects unsupported claims
completeness         | Evaluates response completeness
tone_appropriateness | Checks tone matches context
instruction_following| Evaluates adherence to instructions
coherence           | Judges logical flow and coherence
conciseness         | Evaluates brevity without loss
factual_grounding   | Checks claims are grounded
```

## Note on Reference-Based Metrics

Some metrics require a reference/expected value in the dataset:
- `bleu`, `rouge_*`, `token_overlap_f1`, `jaccard_similarity`
- `numeric_mae`, `numeric_rmse`, `numeric_rel_error`

These are automatically excluded when using `suggest-metrics` on datasets without reference values.

## See Also

- [suggest-metrics](suggest-metrics.md) - Get recommendations for your function
- [run-eval](run-eval.md) - Run evaluation with metrics
