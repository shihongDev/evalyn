# report

Generate and open an interactive HTML insights report in the default browser.

(Renamed from `dashboard` in the 2026-05 rename: the static HTML output is now
`evalyn report`, while `evalyn dashboard` boots the localhost web IDE when
`evalyn-dashboard` is installed - and falls back to this same report when it
is not. See `dashboard.md` for the alias details.)

## Usage

```bash
evalyn report
evalyn report --latest
evalyn report --run <run-id>
evalyn report --dataset data/myapp/
evalyn report --output report.html
```

## Options

| Option | Description |
|--------|-------------|
| `--run ID` | Eval run ID to analyze |
| `--dataset PATH` | Dataset path (uses latest run in that dataset) |
| `--latest` | Use the most recently modified dataset |
| `--output PATH` | Output file path (default: `.evalyn/report.html`) |

## Description

The `report` command generates a comprehensive HTML insights report and opens
it in your default browser. It provides:

- Metric summaries and pass rates
- Score distributions
- Metric correlations
- Regression detection across runs
- Input feature analysis
- Actionable recommendations

This gives a visual overview of your evaluation results without needing an
external service.

## Examples

```bash
# Open the report for the most recent dataset
evalyn report --latest

# Open the report for a specific eval run
evalyn report --run 220e8590

# Save the report to a custom path
evalyn report --latest --output results/report.html
```

## See Also

- [dashboard](dashboard.md) - Boots the localhost web IDE when the
  `evalyn-dashboard` package is installed; otherwise an alias for this command.
- [analyze](analyze.md) - Terminal-based eval analysis.
- [insights](insights.md) - Programmatic insights output (JSON / table / HTML).
