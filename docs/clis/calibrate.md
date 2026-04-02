# evalyn calibrate

Calibrate LLM judges by comparing their results to human annotations.

## Usage

```bash
evalyn calibrate --metric-id <id> --annotations <file> [OPTIONS]
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--metric-id ID` | Required | Metric to calibrate |
| `--annotations FILE` | Required | Annotations file (JSONL) |
| `--run-id ID` | latest | Eval run ID to calibrate |
| `--threshold N` | 0.5 | Current threshold for pass/fail |
| `--dataset PATH` | - | Dataset path (for prompt optimization) |
| `--latest` | false | Use the most recently modified dataset |
| `--no-optimize` | false | Skip prompt optimization, only compute metrics |
| `--optimizer TYPE` | basic | Optimizer: `basic`, `ape`, `opro`, `gepa`, `gepa-native`, `evoprompt`, `textgrad`, `miprov2`, `promptbreeder` |
| `--model MODEL` | gemini-2.5-flash-lite | LLM model for prompt optimization (llm mode) |
| `--gepa-task-lm MODEL` | gemini/gemini-2.5-flash | Task model for GEPA (model being optimized) |
| `--gepa-reflection-lm MODEL` | gemini/gemini-2.5-flash | Reflection model for GEPA (strong model for reflection) |
| `--gepa-max-calls N` | 150 | Max metric calls budget for GEPA optimization |
| `--gepa-native-task-model MODEL` | gemini-2.5-flash | Task model for GEPA-native optimizer |
| `--gepa-native-reflection-model MODEL` | gemini-2.5-flash | Reflection model for GEPA-native optimizer |
| `--gepa-native-max-calls N` | 150 | Max calls for GEPA-native optimization |
| `--gepa-native-initial-candidates N` | 5 | Initial candidates for GEPA-native |
| `--gepa-native-batch-size N` | 5 | Batch size for GEPA-native evaluation |
| `--opro-iterations N` | 10 | Max iterations for OPRO optimization |
| `--opro-candidates N` | 4 | Candidate prompts per OPRO iteration |
| `--opro-optimizer-model MODEL` | gemini-2.5-flash | Model for generating OPRO candidates |
| `--opro-scorer-model MODEL` | gemini-2.5-flash-lite | Model for scoring OPRO candidates |
| `--ape-candidates N` | 10 | Number of candidate prompts for APE |
| `--ape-rounds N` | 5 | UCB evaluation rounds for APE |
| `--ape-samples N` | 5 | Samples per candidate per APE round |
| `--evo-population N` | 8 | Population size for EvoPrompt |
| `--evo-generations N` | 5 | Number of generations for EvoPrompt |
| `--evo-mutation-rate F` | 0.3 | Mutation rate for EvoPrompt |
| `--textgrad-iterations N` | 8 | Max iterations for TextGrad |
| `--textgrad-threshold F` | 0.01 | Min F1 improvement threshold for TextGrad |
| `--mipro-instructions N` | 6 | Instruction candidates for MIPROv2 |
| `--mipro-demos N` | 3 | Few-shot demos for MIPROv2 |
| `--mipro-eval-samples N` | 10 | Eval samples per candidate for MIPROv2 |
| `--pb-population N` | 6 | Population size for PromptBreeder |
| `--pb-generations N` | 5 | Generations for PromptBreeder |
| `--show-examples` | false | Show disagreement examples |
| `--output FILE` | - | Save calibration record to file |
| `--format FMT` | table | Output format: `table` or `json` |
| `--verbose`, `-v` | false | Show detailed cost breakdown |

## Alignment Metrics

| Metric | Description |
|--------|-------------|
| Accuracy | Overall agreement rate |
| Precision | True positives / (True positives + False positives) |
| Recall | True positives / (True positives + False negatives) |
| F1 Score | Harmonic mean of precision and recall |
| Specificity | True negatives / (True negatives + False positives) |
| Cohen's Kappa | Agreement adjusted for chance |

## Optimizers

| Optimizer | Description | Best For |
|-----------|-------------|----------|
| `basic` | Single-shot LLM analysis of disagreements | Quick iterations, low cost |
| `ape` | UCB bandit search across candidates | Exploration vs exploitation |
| `opro` | Trajectory-based iterative refinement | Finding local optima |
| `gepa` | Evolutionary optimization with reflection (external) | Complex rubrics, diversity |
| `gepa-native` | Evolutionary with Pareto front + token tracking | Complex rubrics, cost-aware |
| `evoprompt` | Population-based mutation/crossover | Diverse candidate exploration |
| `textgrad` | Iterative critique-revise loop | Targeted incremental improvement |
| `miprov2` | Joint instruction + few-shot demo optimization | Rubrics that benefit from examples |
| `promptbreeder` | Self-referential prompt evolution | Novel rubric strategies |

See [Optimizer Documentation](../optimizers/README.md) for detailed algorithm descriptions and diagrams.

### LLM Optimizer (default)

Single API call to analyze disagreement patterns. Fast and cost-effective.

### GEPA Optimizer

GEPA (Generative Evolution of Prompts Algorithm) evolves prompts using LLM-based reflection. Configure with:

- `--gepa-task-lm`: The model being optimized (e.g., `gemini-2.5-flash-lite`)
- `--gepa-reflection-lm`: A strong model for reflection (e.g., `gemini-2.5-flash`)
- `--gepa-max-calls`: Budget for metric evaluations during optimization

### OPRO Optimizer

OPRO (Optimization by PROmpting) maintains a trajectory of past prompts and scores, using this history to generate improved candidates iteratively. Configure with:

- `--opro-iterations`: Max optimization iterations (default: 10)
- `--opro-candidates`: Candidates per iteration (default: 4)
- `--opro-optimizer-model`: Model for generating candidates
- `--opro-scorer-model`: Model for scoring candidates

### APE Optimizer

APE (Automatic Prompt Engineer) generates a pool of candidates and uses UCB (Upper Confidence Bound) to balance exploration and exploitation. Configure with:

- `--ape-candidates`: Number of candidate prompts (default: 10)
- `--ape-rounds`: UCB evaluation rounds (default: 5)
- `--ape-samples`: Samples per candidate per round (default: 5)

## Examples

### Basic calibration (metrics only)
```bash
evalyn calibrate --metric-id helpfulness_accuracy --annotations data/my-dataset/annotations.jsonl --no-optimize
```

### Full calibration with LLM optimization
```bash
evalyn calibrate --metric-id helpfulness_accuracy --annotations data/my-dataset/annotations.jsonl --dataset data/my-dataset
```

### Show disagreement examples
```bash
evalyn calibrate --metric-id helpfulness_accuracy --annotations data/my-dataset/annotations.jsonl --show-examples
```

### Save calibration record
```bash
evalyn calibrate --metric-id helpfulness_accuracy --annotations data/my-dataset/annotations.jsonl --output calibration.json
```

### Use GEPA optimizer
```bash
evalyn calibrate --metric-id helpfulness_accuracy --annotations data/my-dataset/annotations.jsonl --dataset data/my-dataset --optimizer gepa
```

### GEPA with custom models and budget
```bash
evalyn calibrate --metric-id helpfulness_accuracy --annotations data/my-dataset/annotations.jsonl --dataset data/my-dataset --optimizer gepa --gepa-task-lm gemini-2.5-flash-lite --gepa-reflection-lm gemini-2.5-flash --gepa-max-calls 100
```

### Use OPRO optimizer
```bash
evalyn calibrate --metric-id helpfulness_accuracy --annotations data/my-dataset/annotations.jsonl --dataset data/my-dataset --optimizer opro
```

### OPRO with custom iterations
```bash
evalyn calibrate --metric-id helpfulness_accuracy --annotations data/my-dataset/annotations.jsonl --dataset data/my-dataset --optimizer opro --opro-iterations 15 --opro-candidates 6
```

### Use APE optimizer
```bash
evalyn calibrate --metric-id helpfulness_accuracy --annotations data/my-dataset/annotations.jsonl --dataset data/my-dataset --optimizer ape
```

### APE with custom configuration
```bash
evalyn calibrate --metric-id helpfulness_accuracy --annotations data/my-dataset/annotations.jsonl --dataset data/my-dataset --optimizer ape --ape-candidates 15 --ape-rounds 8
```

### Use EvoPrompt optimizer
```bash
evalyn calibrate --metric-id helpfulness_accuracy --annotations data/my-dataset/annotations.jsonl --dataset data/my-dataset --optimizer evoprompt
```

### EvoPrompt with custom population
```bash
evalyn calibrate --metric-id helpfulness_accuracy --annotations data/my-dataset/annotations.jsonl --dataset data/my-dataset --optimizer evoprompt --evo-population 12 --evo-generations 8
```

### Use TextGrad optimizer
```bash
evalyn calibrate --metric-id helpfulness_accuracy --annotations data/my-dataset/annotations.jsonl --dataset data/my-dataset --optimizer textgrad
```

### Use MIPROv2 optimizer
```bash
evalyn calibrate --metric-id helpfulness_accuracy --annotations data/my-dataset/annotations.jsonl --dataset data/my-dataset --optimizer miprov2
```

### Use PromptBreeder optimizer
```bash
evalyn calibrate --metric-id helpfulness_accuracy --annotations data/my-dataset/annotations.jsonl --dataset data/my-dataset --optimizer promptbreeder
```

## Sample Output

```
============================================================
CALIBRATION REPORT: helpfulness_accuracy
============================================================
Eval Run: 73bf3c5c-aa9c-4078-a668-8227bd2a1028
Samples:  50

--- ALIGNMENT METRICS ---
Accuracy:       84.0%
Precision:      88.2%
Recall:         90.0%
F1 Score:       89.1%
Specificity:    70.0%
Cohen's Kappa:  0.652

Confusion Matrix:
                   Human PASS  Human FAIL
  Judge PASS           36          3
  Judge FAIL           5           6

--- THRESHOLD ---
Current:   0.500
Suggested: 0.450

--- PROMPT OPTIMIZATION ---
Analyzing 8 disagreement cases...

Suggested rubric improvements:
1. Be more lenient on partial answers that address the main question
2. Consider context relevance, not just factual accuracy
3. Penalize responses that are correct but off-topic

--- SAVED FILES ---
Calibration: data/my-dataset/calibrations/helpfulness_accuracy/20250115_143022_llm.json
Optimized prompt: data/my-dataset/calibrations/helpfulness_accuracy/prompts/optimized.txt

============================================================
```

## Output Files

Calibration results are saved to `<dataset>/calibrations/<metric_id>/`:

```
calibrations/
  helpfulness_accuracy/
    20250115_143022_llm.json    # Calibration record
    prompts/
      original.txt              # Original prompt
      optimized.txt             # Optimized prompt
```

## See Also

- [Optimizer Documentation](../optimizers/README.md) - Detailed optimizer algorithms and diagrams
- [annotate](annotate.md) - Create annotations first
- [run-eval](run-eval.md) - Re-run eval with `--use-calibrated`
