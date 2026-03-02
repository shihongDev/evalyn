# Insights Engine Design

## Goal
Surface diagnostic (why), prescriptive (what to do), and proactive (alerting) insights across CLI.

## Architecture

### Layer 1: Insights Engine (`sdk/evalyn_sdk/analysis/insights.py`)

Shared analytical core:

| Module | Purpose |
|--------|---------|
| `compute_metric_correlations(run)` | Pearson/Spearman between metric scores - find redundant/trading-off pairs |
| `detect_regressions(run_current, run_previous)` | Flag metrics that dropped > threshold |
| `analyze_input_features(run)` | Correlate input length, token count with pass/fail rates |
| `analyze_score_distributions(run)` | Detect bimodal, skewed, cliff-edge distributions |
| `generate_recommendations(run, correlations, regressions, features)` | Prioritized actionable advice |
| `detect_anomalies(runs)` | Statistical outliers across runs |

All functions return typed dataclasses. All accept EvalRun + optional annotations.

### Layer 2: Enhance Existing Commands

- `analyze`: append "Key Findings" section (top 3-5 insights)
- `compare`: append "Regression Alerts" with severity levels
- `trend`: append "Anomaly Flags" on unusual movements

### Layer 3: New `evalyn insights` Command

```
evalyn insights --latest
evalyn insights --run <id>
evalyn insights --project <name>
evalyn insights --format table|json|html
```

Sections: Diagnostics, Recommendations (ranked), Alerts.

## Data Models

```python
@dataclass
class CorrelationResult:
    metric_a: str
    metric_b: str
    pearson: float
    spearman: float
    relationship: Literal["redundant", "tradeoff", "independent"]

@dataclass
class RegressionAlert:
    metric_id: str
    previous_pass_rate: float
    current_pass_rate: float
    delta: float
    severity: Literal["critical", "warning", "info"]

@dataclass
class FeatureInsight:
    feature_name: str          # e.g. "input_length"
    finding: str               # e.g. "Items > 500 tokens fail 3x more"
    affected_items: int
    pass_rate_low: float       # pass rate for low-feature group
    pass_rate_high: float      # pass rate for high-feature group

@dataclass
class DistributionInsight:
    metric_id: str
    shape: Literal["normal", "bimodal", "skewed_low", "skewed_high", "cliff"]
    finding: str

@dataclass
class Recommendation:
    priority: int              # 1=highest
    category: str              # "calibration", "metric_config", "data_quality"
    message: str
    action: str                # CLI command to run

@dataclass
class InsightsReport:
    correlations: List[CorrelationResult]
    regressions: List[RegressionAlert]
    feature_insights: List[FeatureInsight]
    distribution_insights: List[DistributionInsight]
    recommendations: List[Recommendation]
    anomalies: List[str]
```

## Implementation Chunks

| Agent | Files | Independent? |
|-------|-------|-------------|
| A: Metric Correlations | `analysis/insights.py` (correlation funcs), display in `analyze` | Yes |
| B: Regression Detection | `analysis/insights.py` (regression funcs), display in `compare` | Yes |
| C: Input Feature Analysis | `analysis/insights.py` (feature funcs), display in `analyze` | Yes |
| D: Recommendations Engine | `analysis/insights.py` (recommendation funcs) | Yes (uses interfaces from A/B/C) |
| E: `evalyn insights` CLI | `cli/commands/insights.py`, wire everything together | After A-D |

## Backwards Compatibility

- Existing command outputs unchanged; new sections appended
- New command `insights` is additive
- No model changes needed - reads from existing EvalRun/MetricResult
