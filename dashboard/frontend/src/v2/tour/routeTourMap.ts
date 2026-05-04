/**
 * Route -> tour-id mapping for the TourMenu's "Take a tour of this page"
 * action. Mirrors the per-route useRouteTour() wiring inside each route
 * component, so manually triggering a tour from the header always picks
 * the same tour the route would have auto-fired on first visit.
 *
 * Routes not listed here have no associated tour; the menu shows the
 * action disabled with a tooltip explaining there is no tour for the
 * current page.
 */

import { FIRST_RUN_TOUR_ID } from './scripts/firstRun';
import { DATASET_UPLOAD_TOUR_ID } from './scripts/datasetUpload';
import { RUN_EVAL_TOUR_ID } from './scripts/runEval';
import { REVIEW_FAILURES_TOUR_ID } from './scripts/reviewFailures';
import { READ_METRICS_TOUR_ID } from './scripts/readMetrics';

export function tourIdForPathname(pathname: string): string | null {
  // Order matters: more specific prefixes first (none currently overlap,
  // but keep this contract stable for future sub-routes).
  if (pathname === '/' || pathname === '') return FIRST_RUN_TOUR_ID;
  if (pathname.startsWith('/datasets')) return DATASET_UPLOAD_TOUR_ID;
  if (pathname.startsWith('/experiments')) return RUN_EVAL_TOUR_ID;
  if (pathname.startsWith('/review')) return REVIEW_FAILURES_TOUR_ID;
  if (pathname.startsWith('/metrics')) return READ_METRICS_TOUR_ID;
  return null;
}
