# CLI Reference

## Setup

```bash
evalyn init [--output FILE] [--force]
```
Creates configuration file with API key and model defaults.

## Tracing

```bash
evalyn list-calls [--limit N] [--project X] [--version V] [--production] [--simulation]
evalyn show-call --id <call_id>
evalyn show-projects
```

## Datasets

```bash
evalyn build-dataset --project <name> [--version V] [--limit N] [--production] [--simulation] [--since DATE] [--until DATE]
```

**Output:** `data/<project>-<version>-<timestamp>/`
- `dataset.jsonl` - Dataset items
- `meta.json` - Metadata

## Metrics

```bash
# Suggest metrics
evalyn suggest-metrics --target <file.py:func> --mode <mode> [--llm-mode api|local] [--model NAME]

# Modes:
#   basic         - Fast heuristic, no LLM
#   llm-registry  - LLM picks from 50+ templates
#   llm-brainstorm - LLM generates custom metrics
#   bundle        - Pre-configured set (--bundle NAME)

# List available templates
evalyn list-metrics
```

## Evaluation

```bash
evalyn run-eval --dataset <path> [--metrics FILE] [--use-calibrated]
evalyn run-eval --latest [--use-calibrated]

evalyn list-runs
evalyn show-run --id <run_id>
```

## Annotation

```bash
evalyn annotate --dataset <path> [--per-metric] [--restart]
evalyn annotate --latest

evalyn annotation-stats --dataset <path>
evalyn export-for-annotation --dataset <path> --output <file>
evalyn import-annotations --path <file>
```

## Calibration

```bash
evalyn calibrate --metric-id <id> --annotations <file> [--dataset <path>] [--optimizer llm|gepa] [--no-optimize]

evalyn list-calibrations --dataset <path>
```

**Output:** `calibrations/<metric_id>/`
- `calibration.json` - Results with validation
- `prompts/` - Original and optimized prompts

## Simulation

```bash
evalyn simulate --dataset <path> [--target <file.py:func>] [--modes similar,outlier] [--num-similar N] [--num-outlier N]
```

## One-Click Pipeline

```bash
evalyn one-click --project <name> --target <file.py:func> [OPTIONS]
```

**Options:**
| Flag | Default | Description |
|------|---------|-------------|
| `--metric-mode` | basic | Metric selection mode |
| `--dataset-limit` | 100 | Max dataset items |
| `--annotation-limit` | 20 | Items to annotate |
| `--skip-annotation` | false | Skip annotation step |
| `--skip-calibration` | false | Skip calibration step |
| `--enable-simulation` | false | Generate synthetic data |
| `--optimizer` | llm | Calibration optimizer |
| `--production-only` | false | Only production traces |
| `--dry-run` | false | Preview without executing |

**Steps:**
1. Build dataset
2. Suggest metrics
3. Run evaluation
4. Human annotation
5. Calibrate judges
6. Re-evaluate with calibrated prompts
7. Generate simulations (if enabled)

## Status

```bash
evalyn status --dataset <path>
evalyn status --latest
```

## Common Patterns

```bash
# Quick evaluation (no annotation)
evalyn one-click --project myapp --target agent.py:func --skip-annotation

# Production-only dataset
evalyn build-dataset --project myapp --production

# Re-run with calibrated prompts
evalyn run-eval --latest --use-calibrated

# View latest dataset status
evalyn status --latest
```
