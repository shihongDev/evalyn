/**
 * First-visit gate(s) for route-level tours.
 *
 * Each tour has its own localStorage completion flag (see store.ts
 * `tourCompletedKey`). The gate consults two pieces of state:
 *   1. `evalyn.tour.enabled` - global on/off toggle (Settings).
 *   2. `evalyn.tour.completed.<tourId>` - per-tour completion flag.
 *
 *   returns true  when the user has the toggle on AND has not yet
 *                 completed this tour, OR localStorage is unavailable
 *                 (we default to enabled - private mode shouldn't
 *                 suppress onboarding once per session).
 *   returns false when the user explicitly disabled guidance or has
 *                 already completed this tour.
 */

import { TOUR_ENABLED_KEY, tourCompletedKey } from '../store/store';
import { FIRST_RUN_TOUR_ID } from './scripts/firstRun';

export function shouldFireRouteTour(tourId: string): boolean {
  try {
    if (window.localStorage.getItem(TOUR_ENABLED_KEY) === 'false') return false;
    if (window.localStorage.getItem(tourCompletedKey(tourId)) === '1') return false;
    return true;
  } catch {
    return true;
  }
}

/** Backward-compatible alias for the Home first-run tour. */
export function shouldFireFirstRunTour(): boolean {
  return shouldFireRouteTour(FIRST_RUN_TOUR_ID);
}
