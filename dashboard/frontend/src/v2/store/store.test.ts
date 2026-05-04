/**
 * Tour slice unit tests.
 *
 * Each test resets the store and (relevant) localStorage key so order does
 * not matter. We assert pure state transitions; useTour integration with
 * Driver.js is covered separately in tour/useTour.test.tsx.
 */

import { beforeEach, describe, expect, it } from 'vitest';
import { useV2Store, tourCompletedKey } from './store';

const TID = 'firstRun.home.v1';

beforeEach(() => {
  useV2Store.setState({ tourActiveId: null, tourStep: 0 });
  window.localStorage.removeItem(tourCompletedKey(TID));
});

describe('tour slice', () => {
  it('setTour starts a tour at step 0 by default', () => {
    useV2Store.getState().setTour(TID);
    const s = useV2Store.getState();
    expect(s.tourActiveId).toBe(TID);
    expect(s.tourStep).toBe(0);
  });

  it('advanceTour increments tourStep monotonically', () => {
    useV2Store.getState().setTour(TID);
    useV2Store.getState().advanceTour();
    useV2Store.getState().advanceTour();
    expect(useV2Store.getState().tourStep).toBe(2);
  });

  it('markTourComplete clears state AND writes the localStorage flag', () => {
    useV2Store.getState().setTour(TID);
    useV2Store.getState().advanceTour();
    useV2Store.getState().markTourComplete(TID);
    const s = useV2Store.getState();
    expect(s.tourActiveId).toBeNull();
    expect(s.tourStep).toBe(0);
    expect(window.localStorage.getItem(tourCompletedKey(TID))).toBe('1');
  });

  it('abandonTour clears state but does NOT write the completion flag', () => {
    useV2Store.getState().setTour(TID);
    useV2Store.getState().advanceTour();
    useV2Store.getState().abandonTour();
    const s = useV2Store.getState();
    expect(s.tourActiveId).toBeNull();
    expect(s.tourStep).toBe(0);
    expect(window.localStorage.getItem(tourCompletedKey(TID))).toBeNull();
  });
});
