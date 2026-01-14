# trend

Show evaluation trends over time for a project across multiple runs.

## Usage

```bash
evalyn trend --project <dataset_name> [options]
```

## Options

| Option | Description |
|--------|-------------|
| `--project` | Project name (dataset_name) to analyze trends for (required) |
| `--limit` | Maximum number of runs to include (default: 20) |
| `--format` | Output format: `table` (default) or `json` |

## Description

The `trend` command helps you:

- **Track Progress Over Time** - See how your agent's performance evolves across multiple evaluation runs
- **Identify Trends** - Spot improving or regressing metrics at a glance
- **Handle Varying Data** - Works with runs that have different numbers of traces (pass rates are normalized)
- **CI/CD Integration** - JSON output for automated monitoring

This is an enhanced version of `compare` that works with more than two runs and automatically filters by project.

## Output

```
======================================================================
  EVALUATION TRENDS - my-agent
======================================================================

  Runs analyzed: 5 (oldest to newest)
  Time range: 2024-01-01 to 2024-01-15

----------------------------------------------------------------------
  RUN OVERVIEW
----------------------------------------------------------------------
  Run ID         Date               Items    Pass Rate      Delta
  -------------- ------------------ -------- ------------ ----------
  abc123..       2024-01-01 10:00      100        72.0%
  def456..       2024-01-05 14:30      100        75.0%     +3.0%
  ghi789..       2024-01-10 09:15      120        78.3%     +3.3%
  jkl012..       2024-01-12 16:45      120        80.0%     +1.7%
  mno345..       2024-01-15 11:00      125        82.4%     +2.4%

----------------------------------------------------------------------
  METRIC TRENDS (Pass Rate %)
----------------------------------------------------------------------
  Metric                      R1       R2       R3       R4       R5      Delta
  ---------------------- -------- -------- -------- -------- -------- ----------
  accuracy                  70.0%    72.0%    75.0%    78.0%    80.0%    +10.0%
  helpfulness               65.0%    68.0%    72.0%    75.0%    78.0%    +13.0%
  hallucination             85.0%    82.0%    80.0%    82.0%    84.0%     -1.0%
  safety                   100.0%   100.0%   100.0%   100.0%   100.0%         =

----------------------------------------------------------------------
  SUMMARY
----------------------------------------------------------------------
  Overall change: +10.4% (72.0% to 82.4%)

  Metrics improving (2):  accuracy, helpfulness
  Metrics regressing (1): hallucination
  Metrics stable (1):     safety

  Item count change: +25 (100 to 125)
======================================================================
```

## Examples

```bash
# Show trends for a project
evalyn trend --project my-agent

# Limit to last 5 runs
evalyn trend --project my-agent --limit 5

# JSON output for CI/CD pipelines
evalyn trend --project my-agent --format json
```

## JSON Output

When using `--format json`, the output is structured for programmatic consumption:

```json
{
  "project": "my-agent",
  "runs_analyzed": 5,
  "runs": [
    {
      "id": "abc123...",
      "created_at": "2024-01-01T10:00:00",
      "total_items": 100,
      "overall_pass_rate": 0.72,
      "metrics": {
        "accuracy": {"pass_rate": 0.70, "avg_score": 0.75, "count": 100}
      }
    }
  ],
  "trends": {
    "overall_delta": 0.104,
    "improving_metrics": ["accuracy", "helpfulness"],
    "regressing_metrics": ["hallucination"],
    "stable_metrics": ["safety"]
  }
}
```

## Use Cases

### Continuous Monitoring
Track agent quality over time as you iterate:
```bash
# After each deployment
evalyn run-eval --dataset data/my-agent
evalyn trend --project my-agent --limit 10
```

### CI/CD Integration
Add to your CI pipeline to detect regressions:
```bash
# In CI script
result=$(evalyn trend --project my-agent --format json)
delta=$(echo $result | jq '.trends.overall_delta')
if (( $(echo "$delta < -0.05" | bc -l) )); then
  echo "Regression detected!"
  exit 1
fi
```

### Comparing Different Item Counts
The trend command handles varying trace counts gracefully:
```bash
# Run 1: 100 items, Run 2: 150 items, Run 3: 120 items
# Pass rates are percentages, so comparison is fair
evalyn trend --project my-agent
```

## See Also

- [run-eval](run-eval.md) - Run evaluations
- [list-runs](list-runs.md) - List all evaluation runs
- [analyze](analyze.md) - Analyze a single run in detail
- [compare](compare.md) - Compare exactly two runs side-by-side
