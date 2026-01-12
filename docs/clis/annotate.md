# evalyn annotate

Interactively annotate dataset items with human labels for calibration.

## Usage

```bash
evalyn annotate --dataset <path> [OPTIONS]
evalyn annotate --latest [OPTIONS]
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--dataset PATH` | - | Dataset directory |
| `--latest` | false | Use most recent dataset |
| `--per-metric` | false | Annotate each metric separately |
| `--limit N` | - | Max items to annotate |
| `--restart` | false | Start from beginning (ignore previous progress) |

## Annotation Modes

### Standard Mode (default)
- Shows input and output
- Shows LLM judge results for reference
- You provide: pass/fail, confidence (1-5), notes

### Per-Metric Mode (`--per-metric`)
- Shows each metric result individually
- You indicate: agree/disagree with LLM judge
- More granular for calibration

## Examples

### Interactive annotation
```bash
evalyn annotate --dataset data/my-dataset
```

### Annotate latest dataset
```bash
evalyn annotate --latest
```

### Per-metric annotation mode
```bash
evalyn annotate --dataset data/my-dataset --per-metric
```

### Limit number of items
```bash
evalyn annotate --dataset data/my-dataset --limit 20
```

### Restart from beginning
```bash
evalyn annotate --dataset data/my-dataset --restart
```

## Interactive Session

```
================================================================================
ANNOTATION [1/50]
================================================================================

--- INPUT ---
{
  "query": "What are the latest AI developments?"
}

--- OUTPUT ---
Recent developments in artificial intelligence include...

--- LLM JUDGE RESULTS ---
  helpfulness_accuracy: PASS (confidence: 0.85)
    "Response is helpful and addresses the query accurately"

  hallucination_risk: PASS (confidence: 0.72)
    "No unsupported claims detected"

--------------------------------------------------------------------------------
Your assessment:

[P]ass / [F]ail / [S]kip / [Q]uit: p
Confidence (1-5): 4
Notes (optional): Good response, covers key points

Saved. Progress: 1/50
```

## Per-Metric Mode Session

```
================================================================================
ANNOTATION [1/50] - Metric: helpfulness_accuracy
================================================================================

--- INPUT ---
{ "query": "What are the latest AI developments?" }

--- OUTPUT ---
Recent developments in artificial intelligence include...

--- LLM JUDGE RESULT ---
Result: PASS (confidence: 0.85)
Reason: Response is helpful and addresses the query accurately

--------------------------------------------------------------------------------
Do you agree with the LLM judge?

[A]gree / [D]isagree / [S]kip / [Q]uit: a
Notes (optional):

Saved. Progress: 1/50
```

## Output

Annotations are saved to `<dataset>/annotations.jsonl`:

```json
{"id": "ann_001", "target_id": "item_abc", "label": "pass", "confidence": 4, "notes": "Good response"}
{"id": "ann_002", "target_id": "item_def", "label": "fail", "confidence": 5, "notes": "Incorrect facts"}
```

## See Also

- [run-eval](run-eval.md) - Run evaluation first to get LLM judge results
- [calibrate](calibrate.md) - Calibrate judges using annotations
