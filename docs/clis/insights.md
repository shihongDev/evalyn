# insights

Comprehensive diagnostic, prescriptive, and proactive analysis of evaluation results.

## Usage

```bash
evalyn insights --latest
evalyn insights --run <id>
evalyn insights --dataset <path>
evalyn insights --format json
evalyn insights --format html
evalyn insights --latest --deep --provider gemini
evalyn insights --latest --deep --experts quality_analyst,strategist
```

## Options

| Option | Description |
|--------|-------------|
| `--run ID` | Eval run ID to analyze |
| `--dataset PATH` | Dataset path (uses latest run) |
| `--latest` | Use the most recently modified dataset |
| `--format` | Output format: table (default), json, or html |
| `--deep` | Enable LLM expert panel analysis |
| `--provider` | LLM provider for expert panel: gemini (default), openai, ollama |
| `--model` | Model override for expert panel |
| `--experts` | Comma-separated expert subset |

## Description

The `insights` command combines multiple analysis techniques into a single report:

- **Metric Correlations** - Find redundant or trading-off metrics via Pearson correlation
- **Regression Detection** - Flag pass rate drops compared to the previous run
- **Input Feature Analysis** - Correlate input characteristics (e.g. length) with outcomes
- **Score Distribution Analysis** - Detect unusual shapes (bimodal, cliff, skewed)
- **Prioritized Recommendations** - Actionable next steps ranked by priority

### Expert Panel (`--deep`)

When `--deep` is enabled, an LLM expert panel analyzes the deterministic insights. Four experts consult sequentially, each building on prior opinions:

| Expert | Focus |
|--------|-------|
| quality_analyst | Metric health, pass/fail patterns, rubric quality |
| metric_critic | Metric design, thresholds, measurement validity |
| data_scientist | Statistical patterns, distributions, correlations |
| strategist | Prioritized improvement roadmap |

A moderator then synthesizes all expert opinions into a unified analysis with an action plan and dissenting views.

Use `--experts` to run a subset: `--experts quality_analyst,strategist`

### HTML Dashboard (`--format html`)

Generates an interactive HTML report with:

- KPI summary bar (items, pass rate, metrics count, health status)
- Metric pass rate bar chart and radar chart
- Score distribution histograms per metric
- Item-metric heatmap (pass/fail grid)
- Input length vs score scatter plot
- Correlation matrix heatmap
- Regression waterfall chart (if previous run available)
- Recommendation cards with priority badges
- Collapsible expert panel discussion (if `--deep` used)

The dashboard is saved to `<dataset_dir>/insights_report.html`.

## Output

### Table format (default)

```
======================================================================
  EVALYN INSIGHTS
======================================================================

  Run: abc123def4... (my-agent-v1)
  Compared to: prev456789...

======================================================================
  DIAGNOSTICS
======================================================================

  Metric Correlations:
    helpfulness <-> completeness  r=0.92 (redundant)

  Score Distributions:
    toxicity_safety: cliff_at_one - 95% of scores at 1.0

  Input Feature Analysis:
    Longer inputs correlate with lower helpfulness scores

======================================================================
  REGRESSION ALERTS
======================================================================

  [CRITICAL]  helpfulness_accuracy dropped 25% (95% -> 70%)

======================================================================
  RECOMMENDATIONS (by priority)
======================================================================

  1. [regression] helpfulness_accuracy dropped significantly
     -> Review recent prompt changes and revert if needed
```

### With expert panel (`--deep`)

```
======================================================================
  EXPERT PANEL ANALYSIS
======================================================================

  [Quality Analyst] (confidence: 85%)
    The helpfulness metric shows concerning degradation...
      - helpfulness_accuracy dropped from 95% to 70%
      - 3 items fail both helpfulness and completeness
    Concerns: rubric may be too strict for edge cases

  [Strategist] (confidence: 80%)
    Focus on the helpfulness regression first...
      - Root cause analysis on the 15 newly failing items
      - Consider splitting helpfulness into sub-dimensions

  SYNTHESIS:
    All experts agree the helpfulness regression is the top priority...

  ACTION PLAN:
    1. Investigate the 15 newly failing items for common patterns
    2. Review recent prompt or model changes

  Tokens: 2400 in / 800 out (3200 total)
```

## Examples

```bash
# Quick insights on the latest dataset
evalyn insights --latest

# JSON output for programmatic use
evalyn insights --latest --format json

# Generate interactive HTML dashboard
evalyn insights --latest --format html

# Deep analysis with LLM expert panel
evalyn insights --latest --deep

# Use OpenAI for expert panel
evalyn insights --latest --deep --provider openai

# Run only specific experts
evalyn insights --latest --deep --experts quality_analyst,strategist

# Analyze a specific run
evalyn insights --run abc123

# Full dashboard with expert panel
evalyn insights --latest --deep --format html
```

## See Also

- [analyze](analyze.md) - Basic evaluation analysis
- [compare](compare.md) - Compare two evaluation runs
- [cluster-failures](cluster-failures.md) - Cluster failure cases by reason
- [trend](trend.md) - View metric trends over time
