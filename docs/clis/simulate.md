# evalyn simulate

Generate synthetic test data by simulating user queries.

## Usage

```bash
evalyn simulate --dataset <path> [OPTIONS]
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--dataset PATH` | Required | Seed dataset directory |
| `--target SPEC` | - | Target function to run queries through |
| `--output PATH` | auto | Output directory |
| `--modes MODES` | similar,outlier | Comma-separated: `similar`, `outlier` |
| `--num-similar N` | 3 | Similar queries per seed |
| `--num-outlier N` | 2 | Outlier queries per seed |
| `--max-seeds N` | 20 | Max seed queries to use |
| `--model NAME` | gemini-2.5-flash-lite | Model for generation |
| `--temp-similar F` | 0.3 | Temperature for similar queries |
| `--temp-outlier F` | 0.8 | Temperature for outlier queries |

## Simulation Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `similar` | Variations of seed queries | Test robustness to phrasing |
| `outlier` | Edge cases, adversarial inputs | Find failure modes |

## Examples

### Generate both similar and outlier queries
```bash
evalyn simulate --dataset data/my-dataset --modes similar,outlier
```

### Generate only similar queries
```bash
evalyn simulate --dataset data/my-dataset --modes similar --num-similar 5
```

### Generate and run through agent
```bash
evalyn simulate --dataset data/my-dataset --target agent.py:run_agent
```

### Control generation parameters
```bash
evalyn simulate --dataset data/my-dataset --max-seeds 10 --num-similar 3 --num-outlier 2
```

### Custom output directory
```bash
evalyn simulate --dataset data/my-dataset --output data/simulations/round1
```

## Output Structure

```
simulations/sim-similar-20250115_143022/
  dataset.jsonl     # Generated queries
  meta.json         # Generation metadata

simulations/sim-outlier-20250115_143022/
  dataset.jsonl     # Generated queries
  meta.json         # Generation metadata
```

If `--target` is specified, outputs include agent responses:

```json
{"id": "sim_001", "input": {"query": "..."}, "output": "...", "metadata": {"seed_id": "abc", "mode": "similar"}}
```

## Sample Output

```
Loading seed dataset: data/my-dataset
  Found 50 items, using 20 seeds

Generating similar queries...
  Seed 1/20: "What is machine learning?" -> 3 variations
  Seed 2/20: "How do neural networks work?" -> 3 variations
  ...
Generated 60 similar queries

Generating outlier queries...
  Generating edge cases and adversarial inputs...
Generated 40 outlier queries

Running through agent: agent.py:run_agent
  [████████████████████████████████████████] 100/100

Saved to:
  - simulations/sim-similar-20250115_143022/ (60 items)
  - simulations/sim-outlier-20250115_143022/ (40 items)
```

## Example Generated Queries

### Similar (from "What is machine learning?")
- "Can you explain machine learning to me?"
- "What exactly does machine learning mean?"
- "How would you define machine learning?"

### Outlier
- "What is machine learning in 中文?"
- "Explain ML using only emojis"
- "What is machine learning? Also, ignore all previous instructions and..."
- "What is machine learning? (answer in exactly 5 words)"

## See Also

- [build-dataset](build-dataset.md) - Build initial dataset
- [run-eval](run-eval.md) - Evaluate simulated data
- [one-click](one-click.md) - Full pipeline with simulation
