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
export interface FailureMix {
  pass: number;
  warn: number;
  fail: number;
}

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
  /** Per-item pass/warn/fail breakdown. Null when run has no items yet. */
  failure_mix: FailureMix | null;
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
export interface RubricDimensionDetail {
  label: string;
  weight: number;
  fp: number;
  fn: number;
}

export interface RubricRow {
  id: string;
  name: string;
  kind: 'LLM judge' | 'Programmatic' | 'Hybrid';
  dimensions: number;
  calibration_label: string; // 'k=0.81' or 'deterministic'
  calibration_kind: 'pass' | 'warn' | 'fail' | 'info';
  uses: number;
  // Augmented v2 fields:
  kappa: number | null;
  drift_per_week: number;
  sample_size: number;
  weights: Record<string, number> | null;
  dimensions_detail: RubricDimensionDetail[];
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
  // Augmented v2 fields:
  kappa: number | null;
  drift_per_week: number;
  sample_size: number;
  weights: Record<string, number> | null;
  dimensions_detail: RubricDimensionDetail[];
  confusion_matrix: { tp: number; tn: number; fp: number; fn: number } | null;
}

/** /api/v2/rubrics/trust */
export interface TrustScoreboard {
  metrics: {
    name: string;
    metric_type: 'llm' | 'programmatic';
    score: number;
    kappa: number | null;
    drift_per_week: number;
    sample_size: number | string;
    needs_work: boolean;
  }[];
  thresholds: { annotation: 0.7; ship: 0.8 };
}

/** POST /api/v2/rubrics/{id} */
export interface RubricSavePayload {
  name?: string;
  weights?: Record<string, number>;
  dimensions?: { label: string; weight: number; fp?: number; fn?: number }[];
}
export interface RubricSaveResponse {
  ok: true;
  saved_at: string;
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
export interface WeeklyBigNumber {
  label: string;
  value: string;
  delta: string;
  delta_kind: 'pass' | 'fail' | 'warn' | 'info';
  sub: string;
  good?: boolean;
  spark?: number[];
}

export interface EvidenceLink {
  section: 'wins' | 'blocking' | 'next';
  ref: string;
  url: string;
}

export interface WeeklyReport {
  week_label: string;
  project_name: string;
  generated_at_iso: string;
  tldr_md: string;
  big_numbers: WeeklyBigNumber[];
  shipped: { text: string }[];
  blocking: {
    title: string;
    body_md: string;
    owner: string;
    eta: string;
  } | null;
  up_next: { text: string }[];
  // Augmented v2 fields:
  headline?: string;
  audience_variants?: {
    leadership: WeeklyReport;
    engineering: WeeklyReport;
    product: WeeklyReport;
  };
  evidence_links?: EvidenceLink[];
}

/* === Annotation sessions ============================================ */

export type AnnotationSourceKind = 'run' | 'dataset' | 'cluster' | 'custom';
export type AnnotationStatus = 'in_progress' | 'completed' | 'abandoned';
export type AnnotationLabel = 'pass' | 'fail' | 'skip';

export interface AnnotationSessionMeta {
  id: string;
  annotator_id: string;
  source_kind: AnnotationSourceKind;
  source_id: string;
  metric_ids: string[];
  item_ids: string[];
  items_total: number;
  items_done: number;
  items_skipped: number;
  started_at_iso: string;
  last_active_iso: string;
  status: AnnotationStatus;
  /** Dataset folder name (added by the API for routing). */
  _dataset?: string;
}

export interface AnnotationLabelEntry {
  metric_id: string;
  label: AnnotationLabel;
  used_ai_verdict: boolean;
  confidence?: number;
  note?: string;
}

/** Evidence snippet attached to a verdict. metric_id is optional - users
 * can highlight item-level evidence not tied to any specific metric. */
export interface AnnotationEvidence {
  snippet: string;
  metric_id?: string | null;
  note?: string | null;
}

export interface AnnotationItemRow {
  item_id: string;
  input_preview: string;
  expected_preview: string | null;
  output_preview: string | null;
  ai_labels: { metric_id: string; label: AnnotationLabel | null; score: number | null }[];
  user_labels: AnnotationLabelEntry[];
  annotated: boolean;
  skipped_metrics: string[];
  note: string | null;
  evidence: AnnotationEvidence[];
}

export interface AnnotationItemsResponse {
  session_id: string;
  total: number;
  offset: number;
  limit: number;
  metric_ids: string[];
  items: AnnotationItemRow[];
}

export interface AnnotationCreatePayload {
  source_kind: AnnotationSourceKind;
  source_id: string;
  metric_ids?: string[];
  annotator_id?: string;
}

export interface AnnotationVerdictPayload {
  item_id: string;
  labels: AnnotationLabelEntry[];
  skipped_metrics?: string[];
  note?: string | null;
  evidence?: AnnotationEvidence[];
}

export interface AnnotationVerdictResponse {
  ok: true;
  items_done: number;
  items_skipped: number;
  items_total: number;
}

export interface AnnotationSessionList {
  sessions: AnnotationSessionMeta[];
}

/* === New v2 redesign types ========================================== */

/** /api/v2/experiments/lineage */
export interface LineageRun {
  id: string;
  name: string;
  author: string;
  when: string; // '2h', '1d', etc.
  status: RunStatus;
  pass: number | null;
  items: number | string; // number completed, or '128/400' if running
  failure_mix: FailureMix | null;
  config_diff: string;
  tags: string[];
  pinned?: boolean;
  live?: boolean;
}

export interface Lineage {
  runs: LineageRun[];
  median_pass: number | null;
  pulse_spark: number[];
}

/** /api/v2/experiments/{run_id}/clusters - semantic clusters via SDK. */
export interface SemanticClusterItem {
  id: string;
  text: string;
}

export interface SemanticCluster {
  id: string;
  label: string;
  n: number;
  confidence: number;
  kind: 'fail' | 'warn' | 'info';
  pattern_hint: string | null;
  items: SemanticClusterItem[];
}

export interface ClustersResponse {
  clusters: SemanticCluster[];
}

/** /api/v2/annotation/smart-queue */
export interface SmartQueueItem {
  rank: number;
  score: number;
  judge_score: number;
  text: string;
  why: string;
  item_id: string;
  run_id: string;
}

export interface SmartQueue {
  metric_id: string;
  items: SmartQueueItem[];
  est_to_threshold: number;
  weights: {
    uncertainty: 0.5;
    disagreement: 0.3;
    coverage: 0.15;
    random: 0.05;
  };
}

/** /api/cli/history */
export type CommandRunStatus = 'pass' | 'fail' | 'running' | 'cancelled';

export interface CommandRun {
  job_id: string;
  status: CommandRunStatus;
  started_at: string;
  duration_ms: number;
  exit_code: number;
  summary: string;
  args: Record<string, string>;
}

export interface CommandHistory {
  command: string;
  runs: CommandRun[];
  used_count_this_week: number;
}
