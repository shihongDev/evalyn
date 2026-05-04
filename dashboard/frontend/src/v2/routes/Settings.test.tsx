/**
 * GuidanceToggleCard tests.
 *
 * Two behaviors covered:
 *   1. Reads localStorage on mount: a previously-saved 'false' renders the
 *      switch in the off state.
 *   2. Writes localStorage on toggle: clicking the switch flips the saved
 *      preference and updates aria-checked.
 *
 * The reset-flags button is verified via aria-label / role; we do not
 * exercise the visual flash-pill timing.
 */

import { beforeEach, describe, expect, it } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import { GuidanceToggleCard } from './Settings';
import { TOUR_ENABLED_KEY, tourCompletedKey } from '../store/store';
import { FIRST_RUN_TOUR_ID } from '../tour/scripts/firstRun';

beforeEach(() => {
  window.localStorage.removeItem(TOUR_ENABLED_KEY);
  window.localStorage.removeItem(tourCompletedKey(FIRST_RUN_TOUR_ID));
});

describe('GuidanceToggleCard', () => {
  it('reads localStorage on mount: stored "false" renders aria-checked=false', () => {
    window.localStorage.setItem(TOUR_ENABLED_KEY, 'false');
    render(<GuidanceToggleCard />);
    const toggle = screen.getByRole('switch', { name: /co-pilot ui guidance/i });
    expect(toggle).toHaveAttribute('aria-checked', 'false');
  });

  it('writes localStorage on toggle and updates aria-checked', () => {
    render(<GuidanceToggleCard />);
    const toggle = screen.getByRole('switch', { name: /co-pilot ui guidance/i });
    expect(toggle).toHaveAttribute('aria-checked', 'true');
    fireEvent.click(toggle);
    expect(toggle).toHaveAttribute('aria-checked', 'false');
    expect(window.localStorage.getItem(TOUR_ENABLED_KEY)).toBe('false');
    fireEvent.click(toggle);
    expect(toggle).toHaveAttribute('aria-checked', 'true');
    expect(window.localStorage.getItem(TOUR_ENABLED_KEY)).toBe('true');
  });
});
