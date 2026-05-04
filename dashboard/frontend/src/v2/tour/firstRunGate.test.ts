/**
 * Gate tests for shouldFireFirstRunTour AND the generalized
 * shouldFireRouteTour. The route version is what each tabbed page now
 * uses to decide whether to auto-fire its first-visit walk-through.
 *
 * Behavior matrix (applies to both):
 *   1. clean localStorage          -> fire
 *   2. completion flag set         -> do not fire
 *   3. enabled flag explicitly off -> do not fire
 *   4. localStorage throws         -> fire (private mode fallback)
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { TOUR_ENABLED_KEY, tourCompletedKey } from '../store/store';
import { FIRST_RUN_TOUR_ID } from '../tour/scripts/firstRun';
import {
  shouldFireFirstRunTour,
  shouldFireRouteTour,
} from '../tour/firstRunGate';

const OTHER_TOUR_ID = 'datasetUpload.v1';

beforeEach(() => {
  window.localStorage.removeItem(TOUR_ENABLED_KEY);
  window.localStorage.removeItem(tourCompletedKey(FIRST_RUN_TOUR_ID));
  window.localStorage.removeItem(tourCompletedKey(OTHER_TOUR_ID));
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('shouldFireFirstRunTour', () => {
  it('fires when localStorage is clean (default-on for new users)', () => {
    expect(shouldFireFirstRunTour()).toBe(true);
  });

  it('does not fire after the user has completed the tour', () => {
    window.localStorage.setItem(tourCompletedKey(FIRST_RUN_TOUR_ID), '1');
    expect(shouldFireFirstRunTour()).toBe(false);
  });

  it('does not fire when the user has explicitly disabled guidance', () => {
    window.localStorage.setItem(TOUR_ENABLED_KEY, 'false');
    expect(shouldFireFirstRunTour()).toBe(false);
  });

  it('fires when localStorage is unavailable (private mode fallback)', () => {
    vi.spyOn(window.localStorage.__proto__, 'getItem').mockImplementation(() => {
      throw new Error('SecurityError');
    });
    expect(shouldFireFirstRunTour()).toBe(true);
  });
});

describe('shouldFireRouteTour (generalized to any tour id)', () => {
  it('fires for an arbitrary tour id when localStorage is clean', () => {
    expect(shouldFireRouteTour(OTHER_TOUR_ID)).toBe(true);
  });

  it('only suppresses the tour whose flag is set, not the others', () => {
    window.localStorage.setItem(tourCompletedKey(OTHER_TOUR_ID), '1');
    expect(shouldFireRouteTour(OTHER_TOUR_ID)).toBe(false);
    // The unrelated tour should still fire.
    expect(shouldFireRouteTour(FIRST_RUN_TOUR_ID)).toBe(true);
  });

  it('global enabled=false suppresses every tour', () => {
    window.localStorage.setItem(TOUR_ENABLED_KEY, 'false');
    expect(shouldFireRouteTour(OTHER_TOUR_ID)).toBe(false);
    expect(shouldFireRouteTour(FIRST_RUN_TOUR_ID)).toBe(false);
  });
});
