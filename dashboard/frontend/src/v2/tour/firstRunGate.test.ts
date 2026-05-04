/**
 * shouldFireFirstRunTour gate tests.
 *
 * The full Home component carries enough mock infrastructure (useV2Resource,
 * AppShell, react-router) that a pure unit test of the gate function gives
 * us the actual logic coverage we want without dragging in jsdom layout.
 *
 * Behavior matrix:
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

describe('shouldFireRouteTour (any tour id)', () => {
  it('fires for an arbitrary tour id when localStorage is clean', () => {
    expect(shouldFireRouteTour(OTHER_TOUR_ID)).toBe(true);
  });

  it('only suppresses the tour whose flag is set, not the others', () => {
    window.localStorage.setItem(tourCompletedKey(OTHER_TOUR_ID), '1');
    expect(shouldFireRouteTour(OTHER_TOUR_ID)).toBe(false);
    expect(shouldFireRouteTour(FIRST_RUN_TOUR_ID)).toBe(true);
  });

  it('global enabled=false suppresses every tour', () => {
    window.localStorage.setItem(TOUR_ENABLED_KEY, 'false');
    expect(shouldFireRouteTour(OTHER_TOUR_ID)).toBe(false);
    expect(shouldFireRouteTour(FIRST_RUN_TOUR_ID)).toBe(false);
  });
});
