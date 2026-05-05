/**
 * useTour hook tests with Driver.js mocked.
 *
 * Three behaviors covered:
 *   1. Idle: nothing happens when tourActiveId is null.
 *   2. Anchor present: Driver.js is configured with a step that points at
 *      the live element selector.
 *   3. Anchor missing (Codex #3 fix): the step falls back to fallback copy
 *      with no element binding, so Driver.js renders the popover centered
 *      and auto-advances.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, renderHook } from '@testing-library/react';

// vi.mock hoists above all top-level statements, so its factory cannot
// reference module-scope variables. vi.hoisted lifts the mock instances
// alongside it so the test can both wire the mock AND inspect calls.
const { driverMock, driveMock, destroyMock, isLastStepMock, moveNextMock } = vi.hoisted(() => {
  const drive = vi.fn();
  const destroy = vi.fn();
  const isLastStep = vi.fn(() => false);
  const moveNext = vi.fn();
  const driver = vi.fn(() => ({
    drive,
    destroy,
    isLastStep,
    moveNext,
  }));
  return {
    driverMock: driver,
    driveMock: drive,
    destroyMock: destroy,
    isLastStepMock: isLastStep,
    moveNextMock: moveNext,
  };
});

vi.mock('driver.js', () => ({ driver: driverMock }));

import { useV2Store } from '../store/store';
import { useTour } from './useTour';
import { FIRST_RUN_TOUR_ID, firstRunTour } from './scripts/firstRun';
import { coachmarkSelector } from './anchors';

function plantAnchors(ids: readonly string[]): HTMLDivElement[] {
  return ids.map((id) => {
    const el = document.createElement('div');
    el.setAttribute('data-coachmark', id);
    document.body.appendChild(el);
    return el;
  });
}

function clearAnchors(): void {
  document.querySelectorAll('[data-coachmark]').forEach((el) => el.remove());
}

beforeEach(() => {
  driverMock.mockClear();
  driveMock.mockClear();
  destroyMock.mockClear();
  isLastStepMock.mockClear();
  moveNextMock.mockClear();
  clearAnchors();
  useV2Store.setState({ tourActiveId: null, tourStep: 0 });
});

afterEach(() => {
  clearAnchors();
});

describe('useTour', () => {
  it('does nothing while tourActiveId is null', () => {
    renderHook(() => useTour());
    expect(driverMock).not.toHaveBeenCalled();
    expect(driveMock).not.toHaveBeenCalled();
  });

  it('configures Driver.js with selector-bound steps when all anchors are present', async () => {
    plantAnchors(firstRunTour.steps.map((s) => s.anchor));
    renderHook(() => useTour());
    act(() => {
      useV2Store.getState().setTour(FIRST_RUN_TOUR_ID);
    });
    // waitForElement uses 50ms polls with a 500ms ceiling. 700ms covers
    // the full async resolution loop deterministically.
    await new Promise((resolve) => setTimeout(resolve, 700));
    expect(driverMock).toHaveBeenCalledTimes(1);
    const config = driverMock.mock.calls[0][0] as { steps: unknown[] };
    expect(config.steps.length).toBe(firstRunTour.steps.length);
    const firstStep = config.steps[0] as { element?: string };
    expect(firstStep.element).toBe(coachmarkSelector(firstRunTour.steps[0].anchor));
    expect(driveMock).toHaveBeenCalledTimes(1);
  });

  it('elides steps whose anchor is missing instead of narrating "view changed"', async () => {
    // Plant only some anchors - the rest should be silently dropped from
    // the tour rather than rendered with a fallback popover.
    const presentIds = firstRunTour.steps.slice(0, 2).map((s) => s.anchor);
    plantAnchors(presentIds);
    renderHook(() => useTour());
    act(() => {
      useV2Store.getState().setTour(FIRST_RUN_TOUR_ID);
    });
    await new Promise((resolve) => setTimeout(resolve, 700));
    expect(driverMock).toHaveBeenCalledTimes(1);
    const config = driverMock.mock.calls[0][0] as {
      steps: Array<{ element?: string; popover?: { description?: string } }>;
    };
    // Tour ran with exactly the present anchors and nothing else.
    expect(config.steps.length).toBe(presentIds.length);
    expect(config.steps.every((s) => typeof s.element === 'string')).toBe(true);
    expect(
      config.steps.some((s) =>
        s.popover?.description?.includes('Looks like that view changed'),
      ),
    ).toBe(false);
  });

  it('abandons silently when no anchors are present (every step would elide)', async () => {
    // No anchors planted. Every step elides; tour should not invoke
    // Driver.js with an empty steps array.
    renderHook(() => useTour());
    act(() => {
      useV2Store.getState().setTour(FIRST_RUN_TOUR_ID);
    });
    await new Promise((resolve) => setTimeout(resolve, 700));
    expect(driverMock).not.toHaveBeenCalled();
    // Store must be cleared so the tour can be re-triggered later
    // (manually via TourMenu, or auto-fire on a populated revisit).
    expect(useV2Store.getState().tourActiveId).toBeNull();
  });
});
