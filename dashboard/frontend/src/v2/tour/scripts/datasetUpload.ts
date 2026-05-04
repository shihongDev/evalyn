/**
 * datasetUpload.v1 - "how do I add a dataset?"
 *
 * Triggered by the agent when the user explicitly asks to be shown how to
 * upload a dataset. Lives on the Datasets route. Anchors are declared in
 * tour/anchors.ts under the `datasets-*` prefix.
 */

import type { TourDef } from '../useTour';

export const DATASET_UPLOAD_TOUR_ID = 'datasetUpload.v1';

export const datasetUploadTour: TourDef = {
  id: DATASET_UPLOAD_TOUR_ID,
  steps: [
    {
      anchor: 'datasets-list',
      title: 'Your datasets',
      description:
        'Each row is a JSONL or CSV of items to evaluate against. Click any dataset to see its rows and stats.',
    },
    {
      anchor: 'datasets-search',
      title: 'Search and filter',
      description:
        'Once you have more than a handful, filter by name or tag to find the one you want.',
    },
    {
      anchor: 'datasets-new-button',
      title: 'Add a dataset',
      description:
        'Click here to upload a new dataset. Supports JSONL with `input` and `expected` fields, or CSV.',
    },
  ],
};
