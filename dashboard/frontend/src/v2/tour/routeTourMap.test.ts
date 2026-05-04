/**
 * routeTourMap tests.
 *
 * The TourMenu's "Take a tour of this page" action depends on this map
 * resolving the current pathname to the right tour id. If a route is not
 * in the map (e.g. /settings, /commands), the function returns null and
 * the menu disables that action.
 */

import { describe, expect, it } from 'vitest';
import { tourIdForPathname } from './routeTourMap';

describe('tourIdForPathname', () => {
  it('maps the root to the first-run home tour', () => {
    expect(tourIdForPathname('/')).toBe('firstRun.home.v1');
    expect(tourIdForPathname('')).toBe('firstRun.home.v1');
  });

  it('maps /datasets and its sub-routes to dataset upload', () => {
    expect(tourIdForPathname('/datasets')).toBe('datasetUpload.v1');
    expect(tourIdForPathname('/datasets/abc-123')).toBe('datasetUpload.v1');
  });

  it('maps /experiments and its sub-routes to run-eval', () => {
    expect(tourIdForPathname('/experiments')).toBe('runEval.v1');
    expect(tourIdForPathname('/experiments/run-99/cluster/c1')).toBe('runEval.v1');
  });

  it('maps /review to reviewFailures', () => {
    expect(tourIdForPathname('/review')).toBe('reviewFailures.v1');
  });

  it('maps /metrics to readMetrics', () => {
    expect(tourIdForPathname('/metrics')).toBe('readMetrics.v1');
  });

  it('returns null for routes with no associated tour', () => {
    expect(tourIdForPathname('/settings')).toBeNull();
    expect(tourIdForPathname('/commands')).toBeNull();
    expect(tourIdForPathname('/copilot/abc')).toBeNull();
  });
});
