# import-annotations

Import human annotations from a JSONL file into storage.

## Usage

```bash
evalyn import-annotations --path <file>
```

## Options

| Option | Description |
|--------|-------------|
| `--path PATH` | (Required) Path to annotations JSONL file |

## Description

The `import-annotations` command loads annotations from a JSONL file and stores them in the Evalyn database. This is useful when:

- Annotations were created using external tools
- Team members annotated data offline
- Migrating annotations from another system

The annotations are stored in SQLite and can be used for calibration workflows.

## Input Format

Each line in the JSONL file should be a valid annotation object:

```json
{
  "id": "ann-123",
  "target_id": "item-456",
  "label": true,
  "confidence": 4,
  "notes": "Response was accurate and helpful",
  "annotator": "human",
  "metric_labels": {
    "helpfulness_accuracy": {
      "agree_with_llm": true,
      "human_label": true,
      "notes": ""
    }
  }
}
```

## Output

```
Imported 25 annotations into storage.
```

## Examples

```bash
# Import annotations from a file
evalyn import-annotations --path completed_annotations.jsonl

# Import after external annotation workflow
evalyn export-for-annotation --dataset data/myapp --output to_annotate.jsonl
# ... annotations completed externally ...
evalyn import-annotations --path to_annotate_completed.jsonl
```

## See Also

- [export-for-annotation](export-for-annotation.md) - Export items for annotation
- [annotate](annotate.md) - Interactive CLI annotation
- [calibrate](calibrate.md) - Use annotations for calibration
