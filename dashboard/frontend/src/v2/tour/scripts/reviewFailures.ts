/**
 * reviewFailures.v1 - "how do I triage failures?"
 *
 * Auto-fires on first visit to /review. Anchors under `review-*`.
 */

import type { TourDef } from '../useTour';

export const REVIEW_FAILURES_TOUR_ID = 'reviewFailures.v1';

export const reviewFailuresTour: TourDef = {
  id: REVIEW_FAILURES_TOUR_ID,
  steps: [
    {
      anchor: 'review-queue',
      title: 'Review queue',
      description:
        'Items where the judge flagged a failure or where you want a second opinion. Sorted by severity and recency.',
    },
    {
      anchor: 'review-item',
      title: 'Inspect an item',
      description:
        "Each row shows the prompt, model output, and the judge's verdict. Click to expand and read the full trace.",
    },
    {
      anchor: 'review-verdict-buttons',
      title: 'Override or confirm',
      description:
        'Approve to ratify the judge, or override with a human verdict. Your overrides feed back into rubric calibration over time.',
    },
  ],
};
