# dashboard

Generate and open an interactive HTML insights dashboard in the default browser.

## Usage

```bash
evalyn dashboard
evalyn dashboard --latest
evalyn dashboard --run <run-id>
evalyn dashboard --dataset data/myapp/
evalyn dashboard --output report.html
```

## Options

| Option | Description |
|--------|-------------|
| `--run ID` | Eval run ID to analyze |
| `--dataset PATH` | Dataset path (uses latest run in that dataset) |
| `--latest` | Use the most recently modified dataset |
| `--output PATH` | Output file path (default: .evalyn/report.html) |

## Description

The `dashboard` command generates a comprehensive HTML insights dashboard and opens it in your default browser. It provides:

- Metric summaries and pass rates
- Score distributions
- Metric correlations
- Regression detection across runs
- Input feature analysis
- Actionable recommendations

This gives a visual overview of your evaluation results without needing an external service.

## Examples

```bash
# Open dashboard for the most recent dataset
evalyn dashboard --latest

# Open dashboard for a specific eval run
evalyn dashboard --run 220e8590

# Save dashboard to a custom path
evalyn dashboard --latest --output results/dashboard.html
```

## See Also

- [insights](insights.md) - CLI-based insights with optional HTML output
- [analyze](analyze.md) - Analyze eval run results
- [show-run](show-run.md) - View eval run details
