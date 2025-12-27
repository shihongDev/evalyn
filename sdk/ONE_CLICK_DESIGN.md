# One-Click Pipeline Design

## Overview

A single CLI command that runs the complete evaluation pipeline from dataset building to calibrated re-evaluation.

## Command Syntax

```bash
evalyn one-click --project <name> --target <file.py:func> [OPTIONS]
```

## Pipeline Steps

```
[1/7] Build Dataset
  ↓
[2/7] Suggest Metrics
  ↓
[3/7] Run Initial Evaluation
  ↓
[4/7] Human Annotation (optional, interactive)
  ↓
[5/7] Calibrate LLM Judges (optional, requires annotations)
  ↓
[6/7] Re-evaluate with Calibrated Prompts (optional, requires calibration)
  ↓
[7/7] Generate Simulations (optional)
```

## Parameters

### Required
- `--project <name>`: Project name to filter traces
- `--target <file.py:func>`: Target function for evaluation

### Optional - General
- `--version <v>`: Version filter (default: all versions)
- `--production-only`: Use only production traces
- `--simulation-only`: Use only simulation traces
- `--output-dir <path>`: Custom output directory (default: auto-generated)

### Optional - Dataset (Step 1)
- `--dataset-limit <N>`: Max dataset items (default: 100)
- `--since <date>`: Filter traces since date
- `--until <date>`: Filter traces until date

### Optional - Metrics (Step 2)
- `--metric-mode <mode>`: basic|llm-registry|llm-brainstorm|bundle (default: basic)
- `--llm-mode <mode>`: api|local (required if metric-mode is llm-*)
- `--model <name>`: LLM model name (default: gemini-2.5-flash-lite)
- `--bundle <name>`: Bundle name (if metric-mode=bundle)

### Optional - Annotation (Step 4)
- `--skip-annotation`: Skip annotation step (default: false)
- `--annotation-limit <N>`: Max items to annotate (default: 20)
- `--per-metric`: Use per-metric annotation mode

### Optional - Calibration (Step 5)
- `--skip-calibration`: Skip calibration step (default: false)
- `--optimizer <type>`: llm|gepa (default: llm)
- `--calibrate-all-metrics`: Calibrate all subjective metrics (default: only poorly-aligned ones)

### Optional - Simulation (Step 7)
- `--skip-simulation`: Skip simulation step (default: true, must opt-in)
- `--simulation-modes <modes>`: similar,outlier (default: similar)
- `--num-similar <N>`: Similar queries per seed (default: 3)
- `--num-outlier <N>`: Outlier queries per seed (default: 2)
- `--max-sim-seeds <N>`: Max seeds for simulation (default: 10)

### Behavior Flags
- `--auto-yes`: Skip all confirmation prompts (default: false)
- `--verbose`: Show detailed logs (default: false)
- `--dry-run`: Show what would be done without executing (default: false)

## Output Structure

```
data/<project>-<version>-<timestamp>-oneclick/
├── 1_dataset/
│   ├── dataset.jsonl
│   └── meta.json
│
├── 2_metrics/
│   └── suggested.json
│
├── 3_initial_eval/
│   └── run_<timestamp>_<id>.json
│
├── 4_annotations/
│   └── annotations.jsonl
│
├── 5_calibrations/
│   ├── <metric_id_1>/
│   │   ├── calibration.json
│   │   └── prompts/
│   │       ├── original.txt
│   │       └── optimized.txt
│   └── <metric_id_2>/
│       └── ...
│
├── 6_calibrated_eval/
│   └── run_<timestamp>_<id>.json
│
├── 7_simulations/
│   ├── sim-similar-<timestamp>/
│   │   ├── dataset.jsonl
│   │   └── meta.json
│   └── sim-outlier-<timestamp>/
│       └── ...
│
├── 8_simulation_eval/
│   └── run_<timestamp>_<id>.json
│
├── pipeline.log              # Detailed execution log
└── pipeline_summary.json     # Summary of all steps
```

## Progress Display

### Console Output

```
╔════════════════════════════════════════════════════════════════╗
║           EVALYN ONE-CLICK EVALUATION PIPELINE                 ║
╚════════════════════════════════════════════════════════════════╝

Project:  myproj
Target:   agent.py:my_agent
Version:  v1
Mode:     llm-registry (gemini-2.5-flash-lite)
Output:   data/myproj-v1-20250126_143022-oneclick/

────────────────────────────────────────────────────────────────

[1/7] Building Dataset
  → Filtering traces: project=myproj, version=v1, production_only=True
  ✓ Found 150 matching traces
  ✓ Built dataset: 100 items
  ✓ Saved to: 1_dataset/dataset.jsonl

[2/7] Suggesting Metrics
  → Mode: llm-registry
  → Model: gemini-2.5-flash-lite
  → Analyzing target function...
  ✓ Selected 5 metrics (2 objective, 3 subjective)
    - latency_ms (objective)
    - json_valid (objective)
    - helpfulness_accuracy (subjective)
    - toxicity_safety (subjective)
    - completeness (subjective)
  ✓ Saved to: 2_metrics/suggested.json

[3/7] Running Initial Evaluation
  → Evaluating 100 items with 5 metrics...
  [████████████████████████████████] 100/100
  ✓ Evaluation complete

  RESULTS:
    latency_ms:         avg=1234ms (min=500ms, max=3000ms)
    json_valid:         pass_rate=0.98 (98/100)
    helpfulness:        pass_rate=0.92 (92/100)
    toxicity:           pass_rate=1.00 (100/100)
    completeness:       pass_rate=0.88 (88/100)

  ✓ Saved to: 3_initial_eval/run_20250126_143045_abc123.json

[4/7] Human Annotation
  → Annotating 20 random items (20% sample)
  → Mode: per-metric (3 subjective metrics)

  Item 1/20
  ────────────────────────────────────────
  Input:  "What is machine learning?"
  Output: "Machine learning is..."

  Metric: helpfulness_accuracy
  LLM Judge: ✓ PASS (score: 0.92)
  Reason: Response is accurate and helpful

  Do you agree? [y/n]: y
  Confidence (1-5): 5

  [... interactive annotation continues ...]

  ✓ Completed 20 annotations
  ✓ Agreement with LLM judges: 85%
  ✓ Saved to: 4_annotations/annotations.jsonl

[5/7] Calibrating LLM Judges
  → Calibrating 3 subjective metrics...

  [5.1] helpfulness_accuracy
    ✓ Alignment metrics:
      - F1 Score: 0.85
      - Accuracy: 0.85
      - Cohen's Kappa: 0.70
    ✓ Prompt optimization (LLM)
    ✓ Validation: Original F1=0.850, Optimized F1=0.920
    ✅ SUCCESS: Optimized prompt is BETTER (+7.0% improvement)
    💡 Recommendation: USE OPTIMIZED
    ✓ Saved to: 5_calibrations/helpfulness_accuracy/

  [5.2] toxicity_safety
    ✓ Alignment metrics:
      - F1 Score: 1.00 (perfect agreement!)
      - Accuracy: 1.00
    ⏭️  SKIPPED optimization (perfect alignment)

  [5.3] completeness
    ✓ Alignment metrics:
      - F1 Score: 0.72
      - Accuracy: 0.75
    ✓ Prompt optimization (LLM)
    ✓ Validation: Original F1=0.720, Optimized F1=0.690
    ❌ DEGRADED: Optimized prompt is WORSE (-3.0% degradation)
    ⚠️  Recommendation: KEEP ORIGINAL
    ✓ Saved for reference: 5_calibrations/completeness/

  Summary: 1 improved, 1 perfect, 1 degraded

[6/7] Re-evaluating with Calibrated Prompts
  → Using 1 calibrated prompt (helpfulness_accuracy)
  → Evaluating 100 items with 5 metrics...
  [████████████████████████████████] 100/100
  ✓ Evaluation complete

  RESULTS (changes from initial):
    latency_ms:         avg=1234ms (no change)
    json_valid:         pass_rate=0.98 (no change)
    helpfulness:        pass_rate=0.96 (+4 items improved! ✓)
    toxicity:           pass_rate=1.00 (no change)
    completeness:       pass_rate=0.88 (no change)

  ✓ Saved to: 6_calibrated_eval/run_20250126_143145_def456.json

[7/7] Generating Simulations
  → Modes: similar, outlier
  → Seeds: 10 items (10% of dataset)
  → Generating similar queries (3 per seed)...
  ✓ Generated 30 similar queries
  → Generating outlier queries (2 per seed)...
  ✓ Generated 20 outlier queries
  → Running agent on 50 synthetic queries...
  [████████████████████████████████] 50/50
  ✓ Saved to: 7_simulations/

────────────────────────────────────────────────────────────────

╔════════════════════════════════════════════════════════════════╗
║                     PIPELINE COMPLETE                          ║
╚════════════════════════════════════════════════════════════════╝

Total time: 8m 34s

Output directory:
  data/myproj-v1-20250126_143022-oneclick/

Summary:
  Dataset:             100 items
  Initial eval:        5 metrics, 92.8% avg pass rate
  Annotations:         20 items (85% agreement)
  Calibrations:        1 improved, 1 perfect, 1 degraded
  Calibrated eval:     5 metrics, 94.0% avg pass rate (+1.2% ✓)
  Simulations:         50 queries generated

Next steps:
  1. Review calibrated results:
     evalyn show-run --id def456

  2. Evaluate simulations:
     evalyn run-eval --dataset data/myproj-v1-20250126_143022-oneclick/7_simulations/sim-similar-* --use-calibrated

  3. View full summary:
     cat data/myproj-v1-20250126_143022-oneclick/pipeline_summary.json
```

## Pipeline Summary JSON

```json
{
  "pipeline_id": "myproj-v1-20250126_143022",
  "started_at": "2025-01-26T14:30:22Z",
  "completed_at": "2025-01-26T14:38:56Z",
  "duration_seconds": 514,
  "config": {
    "project": "myproj",
    "version": "v1",
    "target": "agent.py:my_agent",
    "metric_mode": "llm-registry",
    "model": "gemini-2.5-flash-lite",
    "production_only": true,
    "dataset_limit": 100,
    "annotation_limit": 20
  },
  "steps": {
    "1_dataset": {
      "status": "success",
      "duration_seconds": 2,
      "output": "1_dataset/dataset.jsonl",
      "stats": {
        "total_traces": 150,
        "dataset_items": 100
      }
    },
    "2_metrics": {
      "status": "success",
      "duration_seconds": 15,
      "output": "2_metrics/suggested.json",
      "stats": {
        "total_metrics": 5,
        "objective_metrics": 2,
        "subjective_metrics": 3
      }
    },
    "3_initial_eval": {
      "status": "success",
      "duration_seconds": 120,
      "output": "3_initial_eval/run_20250126_143045_abc123.json",
      "stats": {
        "items_evaluated": 100,
        "avg_pass_rate": 0.928,
        "metric_results": {
          "latency_ms": {"avg": 1234, "min": 500, "max": 3000},
          "json_valid": {"pass_rate": 0.98},
          "helpfulness": {"pass_rate": 0.92},
          "toxicity": {"pass_rate": 1.00},
          "completeness": {"pass_rate": 0.88}
        }
      }
    },
    "4_annotations": {
      "status": "success",
      "duration_seconds": 180,
      "output": "4_annotations/annotations.jsonl",
      "stats": {
        "items_annotated": 20,
        "agreement_rate": 0.85
      }
    },
    "5_calibrations": {
      "status": "success",
      "duration_seconds": 90,
      "output": "5_calibrations/",
      "stats": {
        "metrics_calibrated": 3,
        "improved": 1,
        "perfect": 1,
        "degraded": 1,
        "calibration_results": {
          "helpfulness_accuracy": {
            "status": "improved",
            "original_f1": 0.850,
            "optimized_f1": 0.920,
            "improvement": 0.070
          },
          "toxicity_safety": {
            "status": "perfect",
            "f1": 1.000
          },
          "completeness": {
            "status": "degraded",
            "original_f1": 0.720,
            "optimized_f1": 0.690,
            "degradation": -0.030
          }
        }
      }
    },
    "6_calibrated_eval": {
      "status": "success",
      "duration_seconds": 95,
      "output": "6_calibrated_eval/run_20250126_143145_def456.json",
      "stats": {
        "items_evaluated": 100,
        "calibrated_metrics": 1,
        "avg_pass_rate": 0.940,
        "improvement": 0.012,
        "metric_results": {
          "latency_ms": {"avg": 1234},
          "json_valid": {"pass_rate": 0.98},
          "helpfulness": {"pass_rate": 0.96, "delta": 0.04},
          "toxicity": {"pass_rate": 1.00},
          "completeness": {"pass_rate": 0.88}
        }
      }
    },
    "7_simulations": {
      "status": "success",
      "duration_seconds": 12,
      "output": "7_simulations/",
      "stats": {
        "modes": ["similar", "outlier"],
        "seeds_used": 10,
        "similar_generated": 30,
        "outlier_generated": 20,
        "total_generated": 50
      }
    }
  },
  "summary": {
    "total_steps": 7,
    "successful_steps": 7,
    "failed_steps": 0,
    "dataset_size": 100,
    "initial_avg_pass_rate": 0.928,
    "calibrated_avg_pass_rate": 0.940,
    "improvement": 0.012,
    "calibrated_metrics": 1,
    "simulations_generated": 50
  }
}
```

## Error Handling

### Step Failures

If a step fails, the pipeline:
1. Logs the error to `pipeline.log`
2. Updates `pipeline_summary.json` with failure status
3. Asks user if they want to:
   - **Skip** this step and continue
   - **Retry** this step
   - **Abort** the entire pipeline

Example:
```
[3/7] Running Initial Evaluation
  → Evaluating 100 items with 5 metrics...
  ✗ ERROR: API rate limit exceeded for Gemini

Options:
  [s] Skip this step and continue
  [r] Retry with exponential backoff
  [a] Abort pipeline

Choice [s/r/a]: r

  → Retrying with backoff (waiting 10s)...
  [████████████████████████████████] 100/100
  ✓ Evaluation complete
```

### Resumable Pipeline

If pipeline is interrupted (Ctrl+C, crash):
```bash
# Resume from last completed step
evalyn one-click --resume data/myproj-v1-20250126_143022-oneclick/

# Output:
Detected incomplete pipeline at: data/myproj-v1-20250126_143022-oneclick/
Last completed step: [3/7] Running Initial Evaluation

Resume from [4/7] Human Annotation? [y/n]: y

[4/7] Human Annotation
  ...
```

## Skipping Steps

Users can skip steps via flags:
```bash
# Skip annotation and calibration
evalyn one-click --project myproj --target agent.py:my_agent \
  --skip-annotation --skip-calibration

# Output shows:
[1/7] Building Dataset ✓
[2/7] Suggesting Metrics ✓
[3/7] Running Initial Evaluation ✓
[4/7] Human Annotation ⏭️  SKIPPED
[5/7] Calibrating LLM Judges ⏭️  SKIPPED (requires annotations)
[6/7] Re-evaluating with Calibrated Prompts ⏭️  SKIPPED (no calibrations)
[7/7] Generating Simulations ⏭️  SKIPPED (default)
```

## Example Use Cases

### Minimal Pipeline (Fast)
```bash
# Just dataset + metrics + eval (no annotation, no calibration)
evalyn one-click --project myproj --target agent.py:my_agent \
  --metric-mode basic \
  --skip-annotation \
  --skip-simulation

# Completes in ~30 seconds
```

### Standard Pipeline (Recommended)
```bash
# Dataset + LLM metrics + eval + annotation + calibration
evalyn one-click --project myproj --target agent.py:my_agent \
  --metric-mode llm-registry \
  --llm-mode api \
  --annotation-limit 20

# Completes in ~5-10 minutes (includes human annotation)
```

### Full Pipeline (Comprehensive)
```bash
# All steps including simulation
evalyn one-click --project myproj --target agent.py:my_agent \
  --metric-mode llm-registry \
  --llm-mode api \
  --annotation-limit 30 \
  --optimizer gepa \
  --enable-simulation \
  --simulation-modes similar,outlier

# Completes in ~15-20 minutes
```

### Production Dataset Only
```bash
# Only use production traces, exclude simulations
evalyn one-click --project myproj --target agent.py:my_agent \
  --production-only \
  --dataset-limit 200 \
  --metric-mode llm-registry

# Useful for evaluating real user interactions
```

## Implementation Notes

### Code Structure

```python
# cli.py

def cmd_one_click(args):
    """
    Main orchestrator for one-click pipeline.
    """
    # Initialize pipeline
    pipeline = Pipeline(args)

    # Run steps sequentially
    try:
        pipeline.run_step_1_build_dataset()
        pipeline.run_step_2_suggest_metrics()
        pipeline.run_step_3_initial_eval()

        if not args.skip_annotation:
            pipeline.run_step_4_annotation()

            if not args.skip_calibration:
                pipeline.run_step_5_calibration()
                pipeline.run_step_6_calibrated_eval()

        if args.enable_simulation:
            pipeline.run_step_7_simulation()

    except KeyboardInterrupt:
        pipeline.save_state()
        print("Pipeline interrupted. Resume with: evalyn one-click --resume {}")
    except Exception as e:
        pipeline.handle_error(e)
    finally:
        pipeline.save_summary()


class Pipeline:
    def __init__(self, args):
        self.args = args
        self.output_dir = self._create_output_dir()
        self.state = PipelineState()
        self.logger = self._setup_logger()

    def run_step_1_build_dataset(self):
        self.logger.print_step_header(1, 7, "Building Dataset")
        # ... implementation ...
        self.state.mark_completed("1_dataset")

    # ... other steps ...
```

### Progress Tracking

- Use `rich` library for beautiful console output
- Progress bars for long-running tasks
- Color-coded status indicators (✓ green, ✗ red, ⏭️  yellow)
- Real-time updates during evaluation

### Logging

- Detailed logs to `pipeline.log`
- Console shows high-level progress
- Errors include full stack traces in log file

### State Management

- Save state after each step
- Resumable if interrupted
- Store intermediate results for debugging
