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
 * Anchor-not-found policy (Codex #3 fix): each step waits 500ms for its
 * `data-coachmark` element to be in the DOM. If still missing, the step
 * renders centered with a fallback message and auto-advances after 1.5s.
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

const ANCHOR_WAIT_MS = 500;
const FALLBACK_ADVANCE_MS = 1500;
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
    let fallbackTimer: ReturnType<typeof setTimeout> | null = null;

    const start = async () => {
      const resolvedSteps: DriveStep[] = await Promise.all(
        tourDef.steps.map(async (step): Promise<DriveStep> => {
          const selector = coachmarkSelector(step.anchor);
          const found = await waitForElement(selector, ANCHOR_WAIT_MS);
          if (!found) {
            return {
              popover: {
                title: step.title,
                description: 'Looks like that view changed - moving on.',
                onPopoverRender: () => {
                  if (fallbackTimer) clearTimeout(fallbackTimer);
                  fallbackTimer = setTimeout(() => {
                    driverObj?.moveNext();
                  }, FALLBACK_ADVANCE_MS);
                },
              },
            };
          }
          return {
            element: selector,
            popover: {
              title: step.title,
              description: step.description,
            },
          };
        }),
      );
      if (cancelled) return;

      driverObj = driver({
        showProgress: true,
        progressText: '{{current}} of {{total}}',
        nextBtnText: 'Next',
        prevBtnText: 'Back',
        doneBtnText: 'Done',
        steps: resolvedSteps,
        onDestroyStarted: () => {
          if (fallbackTimer) clearTimeout(fallbackTimer);
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
      if (fallbackTimer) clearTimeout(fallbackTimer);
      driverObj?.destroy();
    };
  }, [tourActiveId]);
}
