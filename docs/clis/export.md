# export

Export evaluation results in various formats for sharing and reporting.

## Usage

```bash
evalyn export --run <run_id> --format <format> --output <file>
evalyn export --dataset <path> --format <format>
evalyn export --latest --format html --output report.html
```

## Options

| Option | Description |
|--------|-------------|
| `--run ID` | Eval run ID to export |
| `--dataset PATH` | Dataset path (uses latest run from eval_runs/) |
| `--latest` | Use the most recently modified dataset |
| `--format FORMAT` | Output format: json, csv, markdown, html (default: json) |
| `--output, -o PATH` | Output file path (prints to stdout if not specified) |

## Formats

### JSON (default)
Full evaluation data in JSON format.

```bash
evalyn export --latest --format json -o results.json
```

### CSV
Spreadsheet-compatible format with one row per metric result.

```bash
evalyn export --latest --format csv -o results.csv
```

Columns: `item_id`, `metric_id`, `score`, `passed`, `reason`

### Markdown
GitHub-friendly summary for documentation or PRs.

```bash
evalyn export --latest --format markdown -o EVALUATION.md
```

Output:
```markdown
# Evaluation Report

**Run ID:** abc123
**Dataset:** my-agent-v1
**Started:** 2024-01-15 10:30:00

## Summary

| Metric | Avg Score | Pass Rate |
|--------|-----------|-----------|
| helpfulness_accuracy | 0.72 | 70% |
| latency_ms | 245.3 | 100% |
```

### HTML
Standalone HTML report with styled tables.

```bash
evalyn export --latest --format html -o report.html
```

Features:
- Professional styling
- Color-coded pass rates (green/yellow/red)
- Self-contained (no external dependencies)
- Shareable with stakeholders

## Examples

```bash
# Export to JSON file
evalyn export --run abc123 --format json -o results.json

# Export to CSV for spreadsheet analysis
evalyn export --latest --format csv -o results.csv

# Generate HTML report for sharing
evalyn export --dataset data/my-agent --format html -o report.html

# Print markdown to stdout (for piping)
evalyn export --latest --format markdown

# Use in CI/CD to generate reports
evalyn export --latest --format html -o artifacts/eval-report.html
```

## Use Cases

### Generate Reports for Stakeholders
```bash
evalyn run-eval --dataset data/prod-v2
evalyn export --latest --format html -o reports/prod-v2-eval.html
```

### Export for Data Analysis
```bash
evalyn export --latest --format csv -o data/results.csv
python analyze_results.py data/results.csv
```

### Add to Pull Requests
```bash
evalyn export --latest --format markdown >> PR_DESCRIPTION.md
```

## See Also

- [run-eval](run-eval.md) - Run evaluations
- [analyze](analyze.md) - Generate insights from results
- [show-run](show-run.md) - View run details in CLI
