# Insights Engine Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add diagnostic, prescriptive, and proactive insights to evalyn CLI via new `evalyn insights` command + enhanced existing commands.

**Architecture:** New `analysis/insights.py` module with pure functions operating on `RunAnalysis`/`EvalRun` data. New `cli/commands/insights.py` for CLI. Enhance `analyze`, `compare`, `trend` commands with appended insight sections.

**Tech Stack:** Python stdlib only (math, statistics, collections). No new dependencies.

---

### Task 1: Core Data Models + Metric Correlations (INDEPENDENT)

**Branch:** `feat/insights-correlations`

**Files:**
- Create: `sdk/evalyn_sdk/analysis/insights.py`
- Test: `tests/test_insights.py`

**Data models (top of insights.py):**

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Dict, Any

@dataclass
class CorrelationResult:
    metric_a: str
    metric_b: str
    pearson: float
    relationship: Literal["redundant", "tradeoff", "independent"]
    # redundant: pearson > 0.7, tradeoff: pearson < -0.5, else independent

@dataclass
class RegressionAlert:
    metric_id: str
    previous_pass_rate: float
    current_pass_rate: float
    delta: float
    severity: Literal["critical", "warning", "info"]
    # critical: drop > 15%, warning: drop > 5%, info: drop > 0%

@dataclass
class FeatureInsight:
    feature_name: str
    finding: str
    affected_items: int
    pass_rate_low: float
    pass_rate_high: float

@dataclass
class DistributionInsight:
    metric_id: str
    shape: Literal["normal", "bimodal", "skewed_low", "skewed_high", "cliff", "uniform"]
    finding: str

@dataclass
class Recommendation:
    priority: int
    category: str  # "calibration", "metric_config", "data_quality", "action"
    message: str
    action: str  # CLI command to run

@dataclass
class InsightsReport:
    correlations: List[CorrelationResult] = field(default_factory=list)
    regressions: List[RegressionAlert] = field(default_factory=list)
    feature_insights: List[FeatureInsight] = field(default_factory=list)
    distribution_insights: List[DistributionInsight] = field(default_factory=list)
    recommendations: List[Recommendation] = field(default_factory=list)
```

**Correlation function:**

```python
def compute_metric_correlations(run_analysis: RunAnalysis) -> List[CorrelationResult]:
    """Compute Pearson correlation between all metric score pairs.

    Uses item_stats to build per-item score vectors for each metric,
    then computes pairwise Pearson r. Requires >= 5 items with scores
    for both metrics.

    Returns only non-independent pairs (|r| > 0.5) sorted by |r| desc.
    """
```

Logic:
1. For each metric pair (A, B), collect items that have scores for both
2. Compute Pearson r using: r = cov(A,B) / (std(A) * std(B))
3. Classify: r > 0.7 = redundant, r < -0.5 = tradeoff, else independent
4. Return non-independent pairs sorted by abs(r) descending

**Test:**

```python
class TestMetricCorrelations:
    def test_perfectly_correlated_metrics(self):
        """Two metrics with same scores -> redundant."""
        # Build RunAnalysis where metric_a and metric_b have identical scores

    def test_anticorrelated_metrics(self):
        """Two metrics with inverse scores -> tradeoff."""

    def test_independent_metrics(self):
        """Two metrics with random scores -> empty result (filtered out)."""

    def test_insufficient_data(self):
        """Fewer than 5 items -> empty result."""
```

Run: `cd tests && uv run pytest test_insights.py::TestMetricCorrelations -v`

---

### Task 2: Regression Detection (INDEPENDENT)

**Branch:** `feat/insights-regressions`

**Files:**
- Modify: `sdk/evalyn_sdk/analysis/insights.py` (add function)
- Test: `tests/test_insights.py`

**Function:**

```python
def detect_regressions(
    current: RunAnalysis,
    previous: RunAnalysis,
) -> List[RegressionAlert]:
    """Compare two runs and detect metric pass rate drops.

    Severity thresholds:
    - critical: pass rate dropped > 15 percentage points
    - warning: dropped > 5 percentage points
    - info: any drop > 0

    Returns alerts sorted by severity (critical first), then by delta magnitude.
    Only returns metrics where pass rate decreased.
    """
```

Logic:
1. For each metric in current that also exists in previous
2. Compare pass_rate (from MetricStats)
3. If current < previous, create RegressionAlert with appropriate severity
4. Sort: critical first, then warning, then info. Within severity, by largest drop.

**Test:**

```python
class TestRegressionDetection:
    def test_critical_regression(self):
        """15%+ drop -> critical severity."""

    def test_warning_regression(self):
        """5-15% drop -> warning severity."""

    def test_no_regression(self):
        """Improvement or no change -> empty result."""

    def test_new_metric_ignored(self):
        """Metric only in current run -> no alert."""
```

---

### Task 3: Input Feature Analysis (INDEPENDENT)

**Branch:** `feat/insights-features`

**Files:**
- Modify: `sdk/evalyn_sdk/analysis/insights.py` (add function)
- Test: `tests/test_insights.py`

**Function:**

```python
def analyze_input_features(
    run_data: Dict[str, Any],
    run_analysis: RunAnalysis,
) -> List[FeatureInsight]:
    """Correlate input characteristics with pass/fail rates.

    Analyzes:
    - Input text length (chars) - split items at median, compare pass rates
    - Output text length (chars) - same approach

    Reports features where pass rate difference > 10 percentage points
    between low/high groups.

    Args:
        run_data: Raw run dict (contains metric_results with item_id)
        run_analysis: Analyzed run (contains item_stats with pass/fail per item)
    """
```

Why `run_data`? Need access to original dataset items for input/output text.
Actually, looking at the data model: `RunAnalysis.item_stats` has per-item metric results but NOT the original input/output text. The dataset.jsonl has that. So this function needs the dataset items loaded separately.

Revised signature:

```python
def analyze_input_features(
    dataset_items: List[Dict[str, Any]],
    run_analysis: RunAnalysis,
) -> List[FeatureInsight]:
```

Logic:
1. Build item_id -> {input_length, output_length} map from dataset items
2. For each feature (input_length, output_length):
   a. Compute median value
   b. Split items into low (< median) and high (>= median) groups
   c. Compute pass rate for each group (item passes = all metrics passed)
   d. If difference > 10pp, create FeatureInsight

**Test:**

```python
class TestInputFeatureAnalysis:
    def test_long_inputs_fail_more(self):
        """Items with long inputs fail more -> produces insight."""

    def test_no_significant_difference(self):
        """Similar pass rates across groups -> no insight."""

    def test_empty_dataset(self):
        """No items -> empty result."""
```

---

### Task 4: Score Distribution Analysis (INDEPENDENT)

**Branch:** `feat/insights-distributions`

**Files:**
- Modify: `sdk/evalyn_sdk/analysis/insights.py` (add function)
- Test: `tests/test_insights.py`

**Function:**

```python
def analyze_score_distributions(
    run_analysis: RunAnalysis,
) -> List[DistributionInsight]:
    """Detect unusual score distribution shapes per metric.

    Shapes detected:
    - cliff: >70% of scores at exactly 0.0 or 1.0
    - bimodal: >30% scores in [0,0.3] AND >30% in [0.7,1.0] with <20% in middle
    - skewed_low: mean < 0.3 and std > 0.1
    - skewed_high: mean > 0.8 and std < 0.15 (possibly too lenient)
    - uniform: std > 0.25 and no bucket >30%
    - normal: everything else (not reported)

    Only returns non-normal distributions.
    Requires >= 5 scores per metric.
    """
```

Logic:
1. For each metric in run_analysis.metric_stats:
2. Get scores list, skip if < 5
3. Compute buckets: [0-0.2], [0.2-0.4], [0.4-0.6], [0.6-0.8], [0.8-1.0]
4. Check shape heuristics in order: cliff, bimodal, skewed_low, skewed_high, uniform
5. First match wins. If none, skip (normal).

**Test:**

```python
class TestScoreDistributions:
    def test_cliff_distribution(self):
        """All scores at 1.0 -> cliff shape."""

    def test_bimodal_distribution(self):
        """Scores clustered at 0 and 1 -> bimodal."""

    def test_normal_not_reported(self):
        """Normal-looking distribution -> empty result."""
```

---

### Task 5: Recommendations Engine (depends on Tasks 1-4 interfaces, but code-independent)

**Branch:** `feat/insights-recommendations`

**Files:**
- Modify: `sdk/evalyn_sdk/analysis/insights.py` (add function)
- Test: `tests/test_insights.py`

**Function:**

```python
def generate_recommendations(
    run_analysis: RunAnalysis,
    correlations: List[CorrelationResult],
    regressions: List[RegressionAlert],
    feature_insights: List[FeatureInsight],
    distribution_insights: List[DistributionInsight],
    dataset_path: Optional[str] = None,
) -> List[Recommendation]:
    """Generate prioritized recommendations from all insight sources.

    Rules (in priority order):
    1. Critical regressions -> "Investigate immediately: metric X dropped Y%"
    2. Redundant metrics -> "Consider removing one of: X, Y (r=0.95)"
    3. Cliff distributions at 1.0 -> "Metric X may be too lenient (100% at max score)"
    4. Bimodal distributions -> "Metric X has split outcomes - review rubric"
    5. Feature insights -> "Items with long inputs fail more - consider input limits"
    6. Tradeoff metrics -> "Metrics X and Y trade off (r=-0.7) - decide which matters more"
    7. If problem metrics exist -> "Run: evalyn calibrate ..."
    8. If no annotations -> "Run: evalyn annotate ..."
    """
```

**Test:**

```python
class TestRecommendations:
    def test_critical_regression_is_priority_1(self):
        """Critical regressions get highest priority."""

    def test_empty_inputs_produce_annotation_hint(self):
        """No regressions/correlations but problem metrics -> suggest calibration."""

    def test_recommendations_sorted_by_priority(self):
        """Output is sorted by priority ascending."""
```

---

### Task 6: `evalyn insights` CLI Command (depends on Tasks 1-5)

**Branch:** `feat/insights-cli`

**Files:**
- Create: `sdk/evalyn_sdk/cli/commands/insights.py`
- Modify: `sdk/evalyn_sdk/cli/main.py` (add import + register_commands call)
- Modify: `sdk/evalyn_sdk/analysis/__init__.py` (export new symbols)
- Test: `tests/test_cli.py` (add TestInsights class)

**CLI interface:**

```
evalyn insights --latest
evalyn insights --run <id>
evalyn insights --dataset <path>
evalyn insights --format table|json
```

**Table output format:**

```
======================================================================
  EVALYN INSIGHTS
======================================================================

Run: abc12345 (my_dataset)

--- DIAGNOSTICS ---

  Metric Correlations:
    helpfulness <-> factual_accuracy  r=0.82 (redundant)
    safety <-> helpfulness           r=-0.61 (tradeoff)

  Score Distributions:
    toxicity_safety: cliff (100% at max score - may be too lenient)
    helpfulness: bimodal (split outcomes - review rubric)

  Input Feature Analysis:
    Items with input > 234 chars fail 3x more (22% vs 67% pass rate)

--- REGRESSION ALERTS ---

    [CRITICAL] helpfulness dropped 18% (92% -> 74%)
    [WARNING]  factual_accuracy dropped 8% (88% -> 80%)

--- RECOMMENDATIONS (by priority) ---

  1. [critical] Investigate helpfulness regression: 18% drop
     -> evalyn analyze --run <id>
  2. [metric_config] Consider removing factual_accuracy (redundant with helpfulness, r=0.82)
  3. [calibration] toxicity_safety scores always max - tighten rubric or remove
  4. [action] Long inputs correlate with failures - consider input preprocessing

======================================================================
```

**Implementation:** `cmd_insights` function that:
1. Loads run (reuse `_load_analysis_run` pattern from analysis.py)
2. Loads previous run if available (for regression detection)
3. Loads dataset items if available (for feature analysis)
4. Calls compute_metric_correlations, detect_regressions, analyze_input_features, analyze_score_distributions
5. Calls generate_recommendations
6. Formats and prints InsightsReport

**CLI test:**

```python
class TestInsights:
    def test_insights_help(self):
        result = run_cli("insights", "--help")
        result.assert_success()
        result.assert_output_contains("--format")

    def test_insights_no_data(self):
        result = run_cli("insights", "--latest")
        # Should fail gracefully with helpful message
```

---

### Task 7: Enhance Existing Commands (depends on Tasks 1-5)

**Branch:** `feat/insights-enhance-commands`

**Files:**
- Modify: `sdk/evalyn_sdk/cli/commands/analysis.py`

**Changes:**

1. **`cmd_analyze`**: After printing RECOMMENDATIONS section, add "KEY FINDINGS" section using `analyze_score_distributions` and `compute_metric_correlations` on the same RunAnalysis.

2. **`cmd_compare`**: After SUMMARY section, if regressions detected via `detect_regressions`, print "REGRESSION ALERTS" section with severity badges.

3. **`cmd_trend`**: After printing trend report, if latest run available, call `analyze_score_distributions` and print any non-normal distributions as warnings.

Keep additions minimal - 20-40 lines per command. Insights are optional - wrapped in try/except so failures don't break existing commands.

---

### Implementation Order

Parallel group 1 (fully independent): Tasks 1, 2, 3, 4
Sequential: Task 5 (needs models from 1-4)
Sequential: Tasks 6, 7 (needs everything)

### Unsolved Questions

None - all interfaces are defined, all data sources identified, no external dependencies needed.
