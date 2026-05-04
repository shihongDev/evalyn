/**
 * runEval.v1 - "how do I start an evaluation?"
 *
 * Auto-fires on first visit to /experiments. Anchors under `experiments-*`.
 */

import type { TourDef } from '../useTour';

export const RUN_EVAL_TOUR_ID = 'runEval.v1';

export const runEvalTour: TourDef = {
  id: RUN_EVAL_TOUR_ID,
  steps: [
    {
      anchor: 'experiments-list',
      title: 'Experiments',
      description:
        'Every evaluation run lands here. Status, pass rate, and judge cost are all visible at a glance.',
    },
    {
      anchor: 'experiments-filters',
      title: 'Find runs quickly',
      description:
        'Filter by status, dataset, judge model, or rubric to drill into a specific slice of work.',
    },
    {
      anchor: 'experiments-new-button',
      title: 'Start a new evaluation',
      description:
        'Click here to launch a run. Pick a dataset, a rubric, and a judge model; the rest is filled from your defaults.',
    },
  ],
};
