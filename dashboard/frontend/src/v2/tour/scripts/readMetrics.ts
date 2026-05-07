/**
 * readMetrics.v1 - "how do I read this metrics page?"
 *
 * Triggered when the user explicitly asks for a walk-through of the metrics
 * dashboard. Lives on the Metrics route. Anchors under the `metrics-*` prefix.
 */

import type { TourDef } from '../useTour';

export const READ_METRICS_TOUR_ID = 'readMetrics.v1';

export const readMetricsTour: TourDef = {
  id: READ_METRICS_TOUR_ID,
  steps: [
    {
      anchor: 'metrics-list',
      title: 'Per-rubric metrics',
      description:
        'Each rubric criterion gets its own pass rate. Use this when overall quality looks fine but one dimension is slipping.',
    },
    {
      anchor: 'metrics-rubric',
      title: 'Drill into a metric',
      description:
        'Click any metric to see per-item scores, judge confidence, and the items that pulled the average down.',
    },
    {
      anchor: 'metrics-chart',
      title: 'Calibration at a glance',
      description:
        'Judge-vs-human agreement, false positives, and false negatives over the most recent reviewed items. Watch this when you change the rubric or judge model.',
    },
  ],
};
