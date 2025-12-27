# CLI Command Reference

Complete documentation for all Evalyn CLI commands.

## Quick Reference

| Command | Description |
|---------|-------------|
| [init](init.md) | Initialize configuration file |
| [one-click](one-click.md) | Run complete evaluation pipeline |

### Tracing
| Command | Description |
|---------|-------------|
| [list-calls](list-calls.md) | List captured function traces |
| [show-call](show-call.md) | View trace details |
| [show-projects](show-projects.md) | View project summaries |

### Datasets
| Command | Description |
|---------|-------------|
| [build-dataset](build-dataset.md) | Build dataset from traces |

### Metrics
| Command | Description |
|---------|-------------|
| [suggest-metrics](suggest-metrics.md) | Suggest evaluation metrics |
| [list-metrics](list-metrics.md) | List available metric templates |

### Evaluation
| Command | Description |
|---------|-------------|
| [run-eval](run-eval.md) | Run evaluation on dataset |
| [list-runs](list-runs.md) | List evaluation runs |
| [show-run](show-run.md) | View run details |

### Annotation & Calibration
| Command | Description |
|---------|-------------|
| [annotate](annotate.md) | Interactive annotation |
| [calibrate](calibrate.md) | Calibrate LLM judges |

### Simulation
| Command | Description |
|---------|-------------|
| [simulate](simulate.md) | Generate synthetic test data |

## Common Workflows

### Quick Evaluation
```bash
evalyn build-dataset --project my-agent
evalyn suggest-metrics --target agent.py:func --dataset data/my-agent-...
evalyn run-eval --latest
```

### Full Pipeline
```bash
evalyn one-click --project my-agent
```

### Calibration Workflow
```bash
evalyn run-eval --latest
evalyn annotate --latest
evalyn calibrate --metric-id helpfulness_accuracy --annotations annotations.jsonl
evalyn run-eval --latest --use-calibrated
```
