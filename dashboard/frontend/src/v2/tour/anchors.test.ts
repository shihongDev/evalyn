/**
 * Anchor registry compile + runtime check.
 *
 * The CoachmarkId union is enforced by TypeScript at compile time. This
 * test additionally verifies the runtime selector helper produces the
 * documented `data-coachmark="..."` shape.
 */

import { describe, expect, it } from 'vitest';
import { COACHMARKS, coachmarkSelector, type CoachmarkId } from './anchors';

describe('anchor registry', () => {
  it('exports a non-empty list of coachmark ids', () => {
    expect(COACHMARKS.length).toBeGreaterThan(0);
  });

  it('coachmarkSelector produces the expected attribute selector', () => {
    const id: CoachmarkId = 'home-quality';
    expect(coachmarkSelector(id)).toBe('[data-coachmark="home-quality"]');
  });

  it('every coachmark id is unique', () => {
    expect(new Set(COACHMARKS).size).toBe(COACHMARKS.length);
  });
});
