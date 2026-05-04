/**
 * First-visit gate for the firstRun tour.
 *
 * Pure function consulting localStorage. Lives outside Home.tsx so the
 * route file can stick to component exports (React Fast Refresh requires
 * component-only modules) and so the gate can be unit tested without
 * dragging in jsdom layout.
 *
 *   returns true  when the user has the toggle on AND has not yet completed
 *                 the firstRun tour, OR localStorage is unavailable (we
 *                 default to enabled - private mode shouldn't suppress
 *                 onboarding once per session).
 *   returns false when the user explicitly disabled guidance or has already
 *                 completed the firstRun tour.
 */

import { TOUR_ENABLED_KEY, tourCompletedKey } from '../store/store';
import { FIRST_RUN_TOUR_ID } from './scripts/firstRun';

export function shouldFireFirstRunTour(): boolean {
  try {
    if (window.localStorage.getItem(TOUR_ENABLED_KEY) === 'false') return false;
    if (window.localStorage.getItem(tourCompletedKey(FIRST_RUN_TOUR_ID)) === '1') return false;
    return true;
  } catch {
    return true;
  }
}
