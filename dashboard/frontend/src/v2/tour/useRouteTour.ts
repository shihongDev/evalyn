/**
 * useRouteTour - fire a route's first-visit tour after data has loaded.
 *
 * Used by each tabbed route (Home, Datasets, ExperimentsList, Review,
 * Metrics) to trigger the tour for that route exactly once per project,
 * gated by the global guidance toggle and the per-tour completion flag.
 *
 *   const ready = !!(data && !err);
 *   useRouteTour(DATASET_UPLOAD_TOUR_ID, ready);
 *
 * Behavior:
 *   - waits 500ms after `ready` flips true (matches the useTour
 *     anchor-not-found timeout, so any in-flight render finishes before
 *     the tour starts looking for selectors)
 *   - guarded by a ref so it cannot re-fire on re-render
 *   - collapses the co-pilot dock so the tour has a clean focal surface
 *   - cleanup cancels the pending timer if the route unmounts before
 *     the tour fires
 */

import { useEffect, useRef } from 'react';
import { useV2Store } from '../store/store';
import { shouldFireRouteTour } from './firstRunGate';

export function useRouteTour(tourId: string, ready: boolean): void {
  const setTour = useV2Store((s) => s.setTour);
  const setDockOpen = useV2Store((s) => s.setDockOpen);
  const triggered = useRef(false);

  useEffect(() => {
    if (triggered.current) return;
    if (!ready) return;
    if (!shouldFireRouteTour(tourId)) return;
    triggered.current = true;
    setDockOpen(false);
    const t = setTimeout(() => {
      setTour(tourId);
    }, 500);
    return () => clearTimeout(t);
  }, [tourId, ready, setTour, setDockOpen]);
}
