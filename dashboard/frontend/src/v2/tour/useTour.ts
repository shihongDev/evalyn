/**
 * useTour - drives the active tour via Driver.js.
 *
 *   tour state machine
 *   ──────────────────
 *
 *   idle ─── setTour(id) ──▶ resolving ─── all anchors waited 500ms ──▶ running
 *    ▲                          │                                         │
 *    │                          │ component unmount                       │
 *    │                          ▼                                         │
 *    │                       cancelled                                    │
 *    │                                                                    │
 *    │                                              user clicks Done on
 *    │                                              last step             │
 *    └──── markTourComplete(id) ◀─── completed ◀───────────────────────── │
 *    │                                                                    │
 *    │                                                                    │
 *    └──── abandonTour() ◀─── abandoned ◀─── user Esc / Skip / Close ─────┘
 *
 * Anchor-not-found policy: each step waits 500ms for its `data-coachmark`
 * element to be in the DOM. If still missing (empty workspace, conditional
 * render, route-level layout swap), the step is silently elided from the
 * tour. If every step elides, the tour abandons without firing rather
 * than showing an empty driver popover. This is preferable to the prior
 * "view changed - moving on" narration which fired too often on empty
 * tabs and made the tour feel broken.
 *
 * The tour is keyed off `tourActiveId` in the Zustand store. Mounting
 * this hook anywhere in the tree (typically AppShell) is enough.
 */

import { useEffect } from 'react';
import { driver, type DriveStep } from 'driver.js';
import 'driver.js/dist/driver.css';
import { useV2Store } from '../store/store';
import { coachmarkSelector, type CoachmarkId } from './anchors';
import { firstRunTour } from './scripts/firstRun';
import { datasetUploadTour } from './scripts/datasetUpload';
import { runEvalTour } from './scripts/runEval';
import { reviewFailuresTour } from './scripts/reviewFailures';
import { readMetricsTour } from './scripts/readMetrics';

export interface TourStepDef {
  anchor: CoachmarkId;
  title: string;
  description: string;
}

export interface TourDef {
  id: string;
  steps: TourStepDef[];
}

const TOUR_REGISTRY: Record<string, TourDef> = {
  [firstRunTour.id]: firstRunTour,
  [datasetUploadTour.id]: datasetUploadTour,
  [runEvalTour.id]: runEvalTour,
  [reviewFailuresTour.id]: reviewFailuresTour,
  [readMetricsTour.id]: readMetricsTour,
};

/**
 * Stable list of every tour id we know about. Used by Settings (reset
 * all flags) and TourMenu (manual trigger). Source of truth: TOUR_REGISTRY
 * keys, so adding a tour to the registry automatically picks it up.
 */
export const KNOWN_TOUR_IDS: readonly string[] = Object.keys(TOUR_REGISTRY);

const ANCHOR_WAIT_MS = 500;
const ANCHOR_POLL_MS = 50;

function waitForElement(selector: string, timeoutMs: number): Promise<Element | null> {
  return new Promise((resolve) => {
    const existing = document.querySelector(selector);
    if (existing) {
      resolve(existing);
      return;
    }
    const start = Date.now();
    const interval = setInterval(() => {
      const found = document.querySelector(selector);
      if (found || Date.now() - start >= timeoutMs) {
        clearInterval(interval);
        resolve(found);
      }
    }, ANCHOR_POLL_MS);
  });
}

export function useTour(): void {
  const tourActiveId = useV2Store((s) => s.tourActiveId);

  useEffect(() => {
    if (!tourActiveId) return;
    const tourDef = TOUR_REGISTRY[tourActiveId];
    if (!tourDef) {
      useV2Store.getState().abandonTour();
      return;
    }

    let cancelled = false;
    let driverObj: ReturnType<typeof driver> | null = null;

    const start = async () => {
      // Resolve each step's anchor in parallel. Steps whose anchor is not
      // in the DOM after ANCHOR_WAIT_MS are dropped from the tour - the
      // user sees a tighter sequence rather than a string of "view changed"
      // popovers when the route is in an empty / loading layout state.
      const resolved: (DriveStep | null)[] = await Promise.all(
        tourDef.steps.map(async (step): Promise<DriveStep | null> => {
          const selector = coachmarkSelector(step.anchor);
          const found = await waitForElement(selector, ANCHOR_WAIT_MS);
          if (!found) return null;
          return {
            element: selector,
            popover: { title: step.title, description: step.description },
          };
        }),
      );
      if (cancelled) return;

      const resolvedSteps: DriveStep[] = resolved.filter(
        (s): s is DriveStep => s !== null,
      );
      if (resolvedSteps.length === 0) {
        // Every step elided - the route is probably in an empty/skeleton
        // state. Abandon silently so the user is not left staring at an
        // empty driver popover. Manual re-trigger via TourMenu still works
        // once the page populates.
        useV2Store.getState().abandonTour();
        return;
      }

      driverObj = driver({
        showProgress: true,
        progressText: '{{current}} of {{total}}',
        nextBtnText: 'Next',
        prevBtnText: 'Back',
        doneBtnText: 'Done',
        steps: resolvedSteps,
        onDestroyStarted: () => {
          if (driverObj?.isLastStep()) {
            useV2Store.getState().markTourComplete(tourActiveId);
          } else {
            useV2Store.getState().abandonTour();
          }
          driverObj?.destroy();
        },
      });

      driverObj.drive();
    };

    void start();

    return () => {
      cancelled = true;
      driverObj?.destroy();
    };
  }, [tourActiveId]);
}
