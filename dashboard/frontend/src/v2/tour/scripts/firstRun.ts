/**
 * firstRun.home.v1 - the orientation tour that fires on a new user's
 * first visit to Home.
 *
 * Auto-fires on first visit to /. Anchors under `home-*`.
 */

import type { TourDef } from '../useTour';

export const FIRST_RUN_TOUR_ID = 'firstRun.home.v1';

export const firstRunTour: TourDef = {
  id: FIRST_RUN_TOUR_ID,
  steps: [
    {
      anchor: 'home-quality',
      title: 'Overall quality',
      description:
        'Headline pass rate across the last 30 days. The single number you check in the morning.',
    },
    {
      anchor: 'home-submetrics',
      title: 'Sub-metrics',
      description:
        'Each rubric criterion broken out. Useful when overall is fine but one dimension is slipping.',
    },
    {
      anchor: 'home-experiments',
      title: 'Active experiments',
      description:
        'Runs in flight. Click any row to drill into the trace, judge calls, and per-item results.',
    },
    {
      anchor: 'home-activity',
      title: 'Recent activity',
      description:
        'What just happened: completed runs, deltas, regressions worth a look.',
    },
    {
      anchor: 'home-copilot-brief',
      title: 'Co-pilot brief',
      description:
        "Your assistant's read on what to investigate next. Click anything here to ask follow-ups.",
    },
  ],
};
