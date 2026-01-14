# evalyn one-click

Run the complete evaluation pipeline in one command.

## Usage

```bash
evalyn one-click --project <name> [OPTIONS]
```

## Options

### Required
| Option | Description |
|--------|-------------|
| `--project NAME` | Project name to filter traces |

### Optional
| Option | Default | Description |
|--------|---------|-------------|
| `--target SPEC` | - | Target function (for simulation) |
| `--version V` | prompt | Version filter (prompts if multiple) |
| `--output-dir PATH` | auto | Custom output directory |
| `--dry-run` | false | Preview without executing |

### Dataset Options
| Option | Default | Description |
|--------|---------|-------------|
| `--dataset-limit N` | 100 | Max dataset items |
| `--production-only` | false | Only production traces |
| `--simulation-only` | false | Only simulation traces |
| `--since DATE` | - | Filter traces after date |
| `--until DATE` | - | Filter traces before date |

### Metrics Options
| Option | Default | Description |
|--------|---------|-------------|
| `--metric-mode MODE` | basic | Mode: basic, llm-registry, llm-brainstorm, bundle, all |
| `--llm-mode MODE` | api | LLM caller: api, local |
| `--model NAME` | gemini-2.5-flash-lite | Model name |
| `--bundle NAME` | - | Bundle name (for mode=bundle) |

### Annotation Options
| Option | Default | Description |
|--------|---------|-------------|
| `--skip-annotation` | false | Skip annotation step |
| `--annotation-limit N` | 20 | Max items to annotate |
| `--per-metric` | false | Per-metric annotation mode |

### Calibration Options
| Option | Default | Description |
|--------|---------|-------------|
| `--skip-calibration` | false | Skip calibration step |
| `--optimizer TYPE` | llm | Optimizer: llm, gepa |
| `--calibrate-all-metrics` | false | Calibrate all subjective metrics |

### Simulation Options
| Option | Default | Description |
|--------|---------|-------------|
| `--enable-simulation` | false | Enable simulation step |
| `--simulation-modes M` | similar | Modes to run |
| `--num-similar N` | 3 | Similar queries per seed |
| `--num-outlier N` | 2 | Outlier queries per seed |
| `--max-sim-seeds N` | 10 | Max seeds for simulation |

## Pipeline Steps

| Step | Description | Skip Flag |
|------|-------------|-----------|
| 1 | Build dataset from traces | - |
| 2 | Suggest metrics | - |
| 3 | Run initial evaluation | - |
| 4 | Human annotation | `--skip-annotation` |
| 5 | Calibrate judges | `--skip-calibration` |
| 6 | Re-evaluate with calibrated prompts | auto-skipped if no calibrations |
| 7 | Generate simulations | `--enable-simulation` required |

## Examples

### Basic run (prompts for version)
```bash
evalyn one-click --project my-agent
```

### With specific version
```bash
evalyn one-click --project my-agent --version v2
```

### Quick run (skip annotation/calibration)
```bash
evalyn one-click --project my-agent --skip-annotation --skip-calibration
```

### LLM-powered metric selection
```bash
evalyn one-click --project my-agent --metric-mode llm-registry
```

### Production traces only
```bash
evalyn one-click --project my-agent --production-only
```

### With simulation enabled
```bash
evalyn one-click --project my-agent --target agent.py:run_agent --enable-simulation
```

### Dry run (preview only)
```bash
evalyn one-click --project my-agent --dry-run
```

### Full pipeline with all options
```bash
evalyn one-click \
  --project my-agent \
  --version v2 \
  --target agent.py:run_agent \
  --metric-mode llm-registry \
  --dataset-limit 200 \
  --annotation-limit 30 \
  --enable-simulation
```

## Sample Output

```
======================================================================
               EVALYN ONE-CLICK EVALUATION PIPELINE
======================================================================

Project:  my-agent
Version:  v1
Mode:     basic
Output:   data/my-agent-v1-20250115_143022-oneclick

----------------------------------------------------------------------

[1/7] Building Dataset
  ✓ Found 156 items
  ✓ Saved to: data/my-agent-v1-20250115_143022-oneclick/1_dataset/dataset.jsonl

[2/7] Suggesting Metrics
  ✓ Selected 5 metrics (2 objective, 3 subjective)
    - latency_ms (objective)
    - output_nonempty (objective)
    - helpfulness_accuracy (subjective)
    - hallucination_risk (subjective)
    - completeness (subjective)
  ✓ Saved to: data/.../2_metrics/suggested.json

[3/7] Running Initial Evaluation
  ✓ Evaluated 156 items
  RESULTS:
    latency_ms: avg=1234.5ms
    helpfulness_accuracy: pass_rate=0.92
    hallucination_risk: pass_rate=0.88
  ✓ Saved to: data/.../3_initial_eval/run_20250115_abc123.json

[4/7] Human Annotation
  ... (interactive session) ...
  ✓ Annotated 20 items

[5/7] Calibrating Judges
  ✓ Calibrated helpfulness_accuracy
    Accuracy: 90.0%, F1: 0.89
  ✓ Saved to: data/.../5_calibration/

[6/7] Re-evaluating with Calibrated Prompts
  ✓ Used 1 calibrated prompts
  ✓ Evaluated 156 items
  RESULTS:
    helpfulness_accuracy: pass_rate=0.94 (was 0.92)
  ✓ Saved to: data/.../6_calibrated_eval/run_20250115_def456.json

[7/7] Generating Simulations
  ⏭️  SKIPPED (use --enable-simulation to enable)

======================================================================
                         PIPELINE COMPLETE
======================================================================

Output directory: data/my-agent-v1-20250115_143022-oneclick

Summary:
  ✓ 1_dataset: success
  ✓ 2_metrics: success
  ✓ 3_initial_eval: success
  ✓ 4_annotation: success
  ✓ 5_calibration: success
  ✓ 6_calibrated_eval: success
  ⏭️ 7_simulation: skipped
```

## Output Structure

```
data/my-agent-v1-20250115_143022-oneclick/
  pipeline_summary.json        # Full pipeline state
  1_dataset/
    dataset.jsonl
    meta.json
  2_metrics/
    suggested.json
  3_initial_eval/
    run_20250115_abc123.json
  4_annotation/
    annotations.jsonl
  5_calibration/
    helpfulness_accuracy/
      calibration.json
      prompts/
  6_calibrated_eval/
    run_20250115_def456.json
  7_simulations/               # (if enabled)
    sim-similar-20250115/
    sim-outlier-20250115/
```

## See Also

- [build-dataset](build-dataset.md) - Step 1: Build dataset
- [suggest-metrics](suggest-metrics.md) - Step 2: Suggest metrics
- [run-eval](run-eval.md) - Steps 3 & 6: Run evaluation
- [annotate](annotate.md) - Step 4: Annotation
- [calibrate](calibrate.md) - Step 5: Calibration
- [simulate](simulate.md) - Step 7: Simulation
