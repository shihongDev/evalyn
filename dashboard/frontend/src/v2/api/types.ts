/**
 * Typed contracts for /api/v2/* endpoints.
 *
 * These shapes are the source of truth: the FastAPI routers in
 * dashboard/evalyn_dashboard/api/v2/ MUST return these exact JSON shapes.
 * Both the frontend and backend tests assert against this contract.
 *
 * Empty states: every list field defaults to [] when no data exists on
 * disk. Numeric fields use null (not 0) when truly absent so empty-state
 * UI can distinguish "no data yet" from "real zero".
 */

export type RunStatus = 'completed' | 'running' | 'warn' | 'failed' | 'queued';

/** /api/v2/home - aggregated landing snapshot. */
export interface HomeSnapshot {
  /** Project context chip (from evalyn.yaml or .evalyn/ inferred name). */
  project: { name: string; version: string | null };
  /** Overall quality (weighted across sub-metrics) for the latest 30d window. */
  quality: {
    current: number | null; // 0..100
    delta_30d: number | null; // pts
    weighted_across_metrics: number;
    graded_items: number;
    timeline: { x: string; y: number }[]; // 5+ points
    ship_gate: number; // default 90
  };
  /** Sub-metrics today, sorted by importance. */
  sub_metrics: {
    label: string;
    value: number; // 0..100, lower-is-better still reported as percent
    delta: number; // pts (negative = better for inverse metrics)
    inverse: boolean; // true for hallucination-style metrics
  }[];
  /** Active experiments (top 3 most recent). */
  active_experiments: {
    id: string;
    name: string;
    status: RunStatus;
    pass: number | null;
    delta_pts: number | null;
    progress: { done: number; total: number } | null;
    spark: number[];
  }[];
  /** Recent activity feed. */
  recent_activity: {
    who: string;
    what: string; // verb: 'ran', 'flagged', 'reviewed', 'imported'
    target: string;
    when_iso: string;
    icon: string; // single char/glyph
    accent: boolean; // ember-tinted (e.g. co-pilot events)
  }[];
  /** Items needing user attention. */
  attention: {
    severity: 'fail' | 'warn' | 'info';
    title: string;
    subtitle: string;
    cta: string;
    cta_target: string; // route path
  }[];
  /** Co-pilot's morning brief (rendered text + suggested actions). */
  brief: {
    generated_at_iso: string;
    body_md: string;
    actions: { label: string; kind: 'primary' | 'secondary' | 'bare'; intent: string }[];
  } | null;
  /**
   * 30-day spend summary. Null when no runs exist at all (empty workspace).
   * total_30d/total_7d are USD sums where missing per-run cost is treated as 0.
   * runs_with_cost counts only runs whose recorded cost > 0 (free runs excluded).
   * daily_30d is one bucket per calendar day (UTC), oldest first.
   * projected_monthly is a linear extrapolation from the 7d window;
   * null when fewer than 3 distinct days had runs in the past 7d.
   */
  cost: {
    total_30d: number;
    total_7d: number;
    runs_with_cost: number;
    runs_total: number;
    daily_30d: number[];
    projected_monthly: number | null;
  } | null;
}

/** /api/v2/experiments - list view rows. */
export interface ExperimentRow {
  id: string;
  name: string;
  author: string;
  when_iso: string;
  status: RunStatus;
  pass: number | null; // 0..100
  delta: string; // formatted: '+3.0' or '-2.7' or 'baseline' or '-'
  items: string; // formatted: '500' or '312/500'
  duration: string; // '1m 24s'
  cost: string; // '$2.41'
  spark: number[] | null;
  tags: string[];
  err?: string;
}

export type ExperimentList = ExperimentRow[];

/** /api/v2/experiments/{id} - run detail. */
export interface ExperimentDetail {
  id: string;
  name: string;
  status: RunStatus;
  finished_at_iso: string;
  duration: string;
  cost: string;
  dataset: { name: string; n: number };
  model: { id: string; temp: number } | null;
  rubric: string | null;
  baseline_id: string | null;
  /** Headline stats - 4 cards. */
  headline: {
    label: string;
    value: string;
    delta: string;
    delta_kind: 'pass' | 'fail' | 'warn' | 'info';
    sub: string;
  }[];
  /** Cumulative pass-rate timeline (this run vs baseline). */
  pass_timeline: {
    y_min: number;
    y_max: number;
    ship_gate: number;
    x_labels: string[];
    series: { label: string; data: number[]; color_kind: 'ember' | 'steel' | 'fail' }[];
  };
  /** Failure clusters. */
  failure_clusters: {
    total_failures: number;
    total_items: number;
    clusters: {
      id: string;
      label: string;
      count: number;
      color_kind: 'fail' | 'warn' | 'steel' | 'violet' | 'text3';
      regression: boolean;
    }[];
  };
  /** Sub-metric breakdown vs baseline. */
  sub_metrics: {
    label: string;
    value: number;
    baseline: number | null;
    inverse: boolean;
  }[];
  /** Confusion matrix vs baseline. */
  confusion: {
    base_pass_v_pass: number; // kept passing
    base_pass_v_fail: number; // regressed
    base_fail_v_pass: number; // fixed
    base_fail_v_fail: number; // still failing
    net_delta: number;
  } | null;
  /** Top failed items preview. */
  failed_items_preview: {
    id: string;
    user: string;
    expected: string;
    got: string;
    cluster: string;
    score: number;
  }[];
}

/** /api/v2/experiments/{id}/items - per-item read-deep grid. */
export interface ExperimentItemPerMetric {
  metric_id: string;
  passed: boolean | null;
  score: number | null;
}
export interface ExperimentItemRow {
  id: string;
  input_preview: string;
  expected_preview: string | null;
  output_preview: string | null;
  per_metric: ExperimentItemPerMetric[];
  any_failed: boolean;
  passed_count: number;
  failed_count: number;
}
export interface ExperimentItemsResponse {
  total: number;
  offset: number;
  limit: number;
  items: ExperimentItemRow[];
  metric_ids: string[];
}
export type ExperimentItemsFilter = 'all' | 'passed' | 'failed';
export type ExperimentItemsSort = 'item_id' | 'score';

/** /api/v2/experiments/{id}/cluster/{cid} - cluster deep-dive. */
export interface ClusterDetail {
  cluster_id: string;
  label: string;
  pattern: string; // markdown-friendly prose
  total_in_cluster: number;
  total_failures_in_run: number;
  total_items_in_run: number;
  /** Frequency of trigger phrases. */
  triggers: { phrase: string; count: number }[];
  /** Cluster size across recent runs. */
  trend: {
    y_max: number;
    x_labels: string[];
    data: number[];
  };
  /** All items in this cluster. */
  items: {
    id: string;
    user: string;
    hallucinated: string;
    tier: string;
    score: number;
  }[];
  /** Co-pilot's suggested fix. */
  suggested_fix: {
    body_md: string;
    estimated_impact: string;
    cost: string;
    duration: string;
  } | null;
}

/** /api/v2/datasets */
export interface DatasetCard {
  name: string;
  n: number;
  source: string;
  tags: string[];
  coverage: { label: string; value: number }[];
  last_used_iso: string | null;
}
export type DatasetList = DatasetCard[];

/** /api/v2/datasets/{name} - per-dataset detail page payload. */
export interface DatasetDetail {
  name: string;
  n: number;
  source: string;
  tags: string[];
  coverage: { label: string; value: number }[];
  created_at_iso: string | null;
  /** First N items (cap 50). total_items may be larger. */
  items_preview: {
    id: string;
    input_preview: string;
    expected_preview: string | null;
  }[];
  total_items: number;
  /** Recent runs on this dataset, newest first, max 10. */
  recent_runs: {
    id: string;
    created_at_iso: string;
    pass: number | null;
    status: string;
    cost: string;
  }[];
  /** Metric definitions seen across this dataset's runs (deduped). */
  observed_metrics: {
    id: string;
    kind: 'LLM judge' | 'Programmatic';
    uses: number;
  }[];
}

/** /api/v2/rubrics */
export interface RubricRow {
  id: string;
  name: string;
  kind: 'LLM judge' | 'Programmatic' | 'Hybrid';
  dimensions: number;
  calibration_label: string; // 'k=0.81' or 'deterministic'
  calibration_kind: 'pass' | 'warn' | 'fail' | 'info';
  uses: number;
}
export type RubricList = RubricRow[];

export interface RubricDetail {
  id: string;
  name: string;
  calibration: {
    kappa: number | null;
    label: string;
    kind: 'pass' | 'warn' | 'fail' | 'info';
    false_positives_pct: number | null;
    false_negatives_pct: number | null;
    sample_size: number;
  };
  dimensions: {
    label: string;
    weight_pct: number;
    example: string;
    kind: 'judge' | 'prog';
  }[];
}

/** /api/v2/review/queue */
export interface ReviewItem {
  item_id: string;
  category: string;
  judge_confidence: number; // 0..1
  user_text: string;
  agent_response: string;
  expected: string;
  highlights: string[]; // substrings to mark in agent_response
  source_run_id: string;
  source_run_label: string;
  judge_breakdown: { label: string; score: number; kind: 'pass' | 'fail' | 'warn' }[];
  judge_reasoning: string;
}
/**
 * One per (dataset, metric_id) combo with enough verdicts to make
 * `evalyn calibrate` worthwhile. The frontend renders these as a
 * banner with a "Run calibrate" button that deep-links into the
 * CliRunner with `cli_args` pre-filled.
 */
export interface CalibrationSuggestion {
  metric_id: string;
  dataset: string;
  verdict_count: number;
  threshold: number; // typically 10
  cli_args: { metric_id: string; annotations: string };
}
export interface ReviewQueue {
  items: ReviewItem[];
  reviewers: { name: string; done: number; total: number; you: boolean }[];
  rationale: string; // why this batch
  calibration_suggestions: CalibrationSuggestion[];
}

/** /api/v2/review/verdict */
export interface ReviewVerdictPayload {
  item_id: string;
  source_run_id: string;
  verdict: 'pass' | 'fail' | 'skip';
  note: string | null;
}

/** /api/v2/reports/weekly */
export interface WeeklyReport {
  week_label: string;
  project_name: string;
  generated_at_iso: string;
  tldr_md: string;
  big_numbers: {
    label: string;
    value: string;
    delta: string;
    delta_kind: 'pass' | 'fail' | 'warn' | 'info';
    sub: string;
  }[];
  shipped: { text: string }[];
  blocking: {
    title: string;
    body_md: string;
    owner: string;
    eta: string;
  } | null;
  up_next: { text: string }[];
}
