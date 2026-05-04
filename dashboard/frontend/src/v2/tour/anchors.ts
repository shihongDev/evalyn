/**
 * Tour anchor registry.
 *
 * Compile-time selector contract for `data-coachmark="..."` attributes
 * on UI elements. Tour scripts reference CoachmarkId values; the useTour
 * hook resolves them to DOM selectors via `coachmarkSelector(id)`.
 *
 * Why a typed registry: prevents typos in tour scripts and lets a future
 * lint or audit step verify that every CoachmarkId has at least one
 * matching anchor in the codebase.
 *
 *   v1 scope: Home route only (5 anchors).
 *   v2: extend to Datasets, ExperimentsList, RunDetail, Review, Metrics
 *       and split this list into per-route arrays.
 */

export const COACHMARKS = [
  // Home (firstRun.home.v1 - shipped in iteration 10)
  'home-quality',
  'home-submetrics',
  'home-experiments',
  'home-activity',
  'home-copilot-brief',
  // Datasets (datasetUpload.v1)
  'datasets-list',
  'datasets-new-button',
  'datasets-search',
  // ExperimentsList (runEval.v1)
  'experiments-list',
  'experiments-new-button',
  'experiments-filters',
  // Review (reviewFailures.v1)
  'review-queue',
  'review-item',
  'review-verdict-buttons',
  // Metrics (readMetrics.v1)
  'metrics-list',
  'metrics-rubric',
  'metrics-chart',
] as const;

export type CoachmarkId = (typeof COACHMARKS)[number];

export function coachmarkSelector(id: CoachmarkId): string {
  return `[data-coachmark="${id}"]`;
}
