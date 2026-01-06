# CLI Reference

Quick reference for all Evalyn CLI commands. See [commands/](commands/) for detailed documentation with examples.

## Setup

```bash
evalyn init                    # Create config file
```
[Full documentation](commands/init.md)

## Tracing

```bash
evalyn list-calls              # List captured traces
evalyn show-call --id <id>     # View trace details
evalyn show-projects           # View project summaries
```
- [list-calls](commands/list-calls.md)
- [show-call](commands/show-call.md)
- [show-projects](commands/show-projects.md)

## Datasets

```bash
evalyn build-dataset --project <name>
```
[Full documentation](commands/build-dataset.md)

## Metrics

```bash
evalyn suggest-metrics --project <name> --mode <mode>
evalyn suggest-metrics --project <name> --mode llm-brainstorm  # Custom subjective metrics
evalyn list-metrics            # List available templates
```
- [suggest-metrics](commands/suggest-metrics.md)
- [list-metrics](commands/list-metrics.md)

## Evaluation

```bash
evalyn run-eval --dataset <path>    # Runs eval and generates HTML report
evalyn run-eval --latest            # Run on latest dataset
evalyn list-runs
evalyn show-run --id <id>
```

Each `run-eval` automatically generates an HTML analysis report alongside the JSON results with:
- Summary statistics and pass rates
- Interactive Chart.js visualizations
- Metric correlation analysis
- Failed items breakdown

- [run-eval](commands/run-eval.md)
- [list-runs](commands/list-runs.md)
- [show-run](commands/show-run.md)

## Annotation

```bash
evalyn annotate --dataset <path>
evalyn annotate --latest --per-metric
```
[Full documentation](commands/annotate.md)

## Calibration

```bash
evalyn calibrate --metric-id <id> --annotations <file>
```
[Full documentation](commands/calibrate.md)

## Simulation

```bash
evalyn simulate --dataset <path> --modes similar,outlier
```
[Full documentation](commands/simulate.md)

## One-Click Pipeline

```bash
evalyn one-click --project <name>
```
[Full documentation](commands/one-click.md)

Runs all steps: dataset → metrics → eval → annotate → calibrate → re-eval → simulate

## Common Patterns

```bash
# Quick evaluation
evalyn one-click --project myapp --skip-annotation

# Production traces only
evalyn build-dataset --project myapp --production

# Re-run with calibrated prompts
evalyn run-eval --latest --use-calibrated

# Full pipeline with simulation
evalyn one-click --project myapp --target agent.py:func --enable-simulation
```
