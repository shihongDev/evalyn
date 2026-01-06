# evalyn analyze

Generate comprehensive analysis and visualization of evaluation results.

## Usage

```bash
evalyn analyze --dataset <path> [OPTIONS]
evalyn analyze --latest [OPTIONS]
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--dataset PATH` | - | Path to dataset directory |
| `--latest` | - | Use the most recently modified dataset |
| `--run-id ID` | - | Specific eval run ID to analyze (defaults to latest) |
| `--compare` | false | Compare multiple runs |
| `--num-runs N` | 5 | Number of runs to compare (with --compare) |
| `--format FORMAT` | text | Output format: `text` or `html` |
| `--output PATH` | - | Output file path for HTML report |
| `--verbose` | false | Show detailed information including failed items |

## Output Formats

### Text (Default)

ASCII-based report with:
- Summary statistics
- Pass rate bar charts
- Score statistics table
- Score distribution mini-charts
- Failed items (with --verbose)

### HTML

Self-contained interactive HTML report with embedded Chart.js visualizations:
- Summary statistics cards (items, metrics, pass rate, failures)
- Pass rate bar chart (color-coded by threshold)
- Score distribution chart (min-max range with average markers)
- Pass/fail stacked bar chart by metric
- Detailed metrics table with all statistics
- Metric correlation heatmap (when applicable)
- Failed items list with failure reasons
- Run metadata section

The HTML report uses Anthropic research paper styling:
- Light background (`#fafaf8`)
- Blue accent color (`#4a90a4`) for success
- Coral color (`#d65a4a`) for failures
- Clean, minimal design with subtle borders

## Examples

### Basic Analysis

```bash
# Analyze latest dataset's most recent run
evalyn analyze --latest

# Analyze specific dataset
evalyn analyze --dataset data/myapp-v1-20250101
```

### Verbose Mode

```bash
# Show failed items breakdown
evalyn analyze --latest --verbose
```

Output:
```
----------------------------------------------------------------------
  FAILED ITEMS (3)
----------------------------------------------------------------------
  abc123... failed: helpfulness_accuracy, coherence
  def456... failed: latency_ms
  ghi789... failed: hallucination_risk, toxicity_safety
```

### Compare Runs

```bash
# Compare last 3 runs
evalyn analyze --dataset data/myapp --compare --num-runs 3
```

Output:
```
======================================================================
  EVAL RUN COMPARISON
======================================================================

  Run          Date                    Items    Overall
  ------------ -------------------- -------- ----------
  e7161b71-40f 2025-12-25T00:26            5      80.0%
  653cd754-fb0 2025-12-25T00:25            5      60.0%
  c8caf251-4eb 2025-12-24T12:13            5      40.0%

----------------------------------------------------------------------
  PASS RATE BY METRIC
----------------------------------------------------------------------
  Metric                          Run1       Run2       Run3      Delta
  ------------------------- ---------- ---------- ---------- ----------
  helpfulness_accuracy           80.0%      60.0%      40.0%    +40.0%
  toxicity_safety               100.0%     100.0%     100.0% +     0.0%
```

### HTML Report

```bash
# Generate HTML report
evalyn analyze --dataset data/myapp --format html

# Custom output path
evalyn analyze --dataset data/myapp --format html --output reports/analysis.html
```

### Specific Run

```bash
# Analyze a specific run by ID
evalyn analyze --dataset data/myapp --run-id abc123
```

## Sample Output

```
======================================================================
  EVAL RUN ANALYSIS
======================================================================

  Run ID:     7d623233...
  Dataset:    myapp-v1
  Created:    2025-12-28T03:57:34
  Items:      50
  Metrics:    5

----------------------------------------------------------------------
  OVERALL SUMMARY
----------------------------------------------------------------------
  Items passing all metrics: 42/50 (84.0%)
  Items with failures:       8

----------------------------------------------------------------------
  METRIC PASS RATES
----------------------------------------------------------------------
  latency_ms                     ██████████████████░░░░░░░  72.0% (n=50)
  helpfulness_accuracy           ████████████████████░░░░░  80.0% (n=50)
  coherence                      ██████████████████████░░░  88.0% (n=50)
  toxicity_safety                █████████████████████████ 100.0% (n=50)
  output_nonempty                █████████████████████████ 100.0% (n=50)

----------------------------------------------------------------------
  SCORE STATISTICS
----------------------------------------------------------------------
  Metric                              Avg      Min      Max   StdDev
  ------------------------------ -------- -------- -------- --------
  latency_ms                     1234.567  456.789 3456.789  567.890
  helpfulness_accuracy              0.800    0.000    1.000    0.400
  coherence                         0.880    0.500    1.000    0.150
  toxicity_safety                   1.000    1.000    1.000    0.000
  output_nonempty                   1.000    1.000    1.000    0.000

----------------------------------------------------------------------
  SCORE DISTRIBUTIONS (0.0 → 1.0)
----------------------------------------------------------------------
  latency_ms                     [▃▂▃▄▆] avg=1234.57
  helpfulness_accuracy           [▂▁▁▁▆] avg=0.80
  coherence                      [▁▁▂▃▆] avg=0.88
  toxicity_safety                [▁▁▁▁▆] avg=1.00
  output_nonempty                [▁▁▁▁▆] avg=1.00

======================================================================
```

## See Also

- [run-eval](run-eval.md) - Run evaluation to generate results
- [list-runs](list-runs.md) - List available eval runs
- [show-run](show-run.md) - View raw eval run data
