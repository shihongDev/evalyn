/**
 * clipboard.ts tests.
 *
 * The two paths matter equally for caller correctness:
 * - Modern path: navigator.clipboard.writeText. Caller awaits and
 *   gets either fulfillment or a rejection.
 * - Fallback path: textarea + document.execCommand('copy'). The
 *   PREVIOUS implementation ignored execCommand's boolean return
 *   value, so a failed copy would resolve silently and leave
 *   callers showing "Copied" feedback when nothing was on the
 *   clipboard. This file pins the contract that the fallback
 *   throws when execCommand returns false.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { copyToClipboard } from './clipboard';

describe('copyToClipboard', () => {
  // The hook's behavior depends on whether navigator.clipboard
  // exists. Save and restore it across tests so we can simulate
  // both paths.
  let originalClipboard: Clipboard | undefined;

  beforeEach(() => {
    originalClipboard = navigator.clipboard;
  });

  afterEach(() => {
    Object.defineProperty(navigator, 'clipboard', {
      configurable: true,
      value: originalClipboard,
    });
  });

  it('uses navigator.clipboard.writeText when available', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, 'clipboard', {
      configurable: true,
      value: { writeText },
    });

    await copyToClipboard('hello');
    expect(writeText).toHaveBeenCalledWith('hello');
  });

  it('propagates writeText rejections to the caller', async () => {
    const writeText = vi.fn().mockRejectedValue(new Error('blocked'));
    Object.defineProperty(navigator, 'clipboard', {
      configurable: true,
      value: { writeText },
    });

    await expect(copyToClipboard('hello')).rejects.toThrow('blocked');
  });

  it('falls back to execCommand when navigator.clipboard is undefined', async () => {
    Object.defineProperty(navigator, 'clipboard', {
      configurable: true,
      value: undefined,
    });
    const exec = vi.fn().mockReturnValue(true);
    Object.defineProperty(document, 'execCommand', {
      configurable: true,
      writable: true,
      value: exec,
    });

    await copyToClipboard('fallback-text');
    expect(exec).toHaveBeenCalledWith('copy');
  });

  it('throws when execCommand returns false (regression)', async () => {
    // The fix this test was added to pin: the previous code
    // ignored execCommand's return value and resolved silently,
    // leaving callers showing "Copied ✓" when nothing was on
    // the clipboard.
    Object.defineProperty(navigator, 'clipboard', {
      configurable: true,
      value: undefined,
    });
    const exec = vi.fn().mockReturnValue(false);
    Object.defineProperty(document, 'execCommand', {
      configurable: true,
      writable: true,
      value: exec,
    });

    await expect(copyToClipboard('blocked-fallback')).rejects.toThrow(
      /clipboard copy failed/i,
    );
  });

  it('removes the temp textarea even when execCommand throws', async () => {
    // Defensive: if execCommand throws (some odd browser quirks
    // do this), the textarea must still be cleaned up. The
    // try/finally in the impl should handle this.
    Object.defineProperty(navigator, 'clipboard', {
      configurable: true,
      value: undefined,
    });
    const exec = vi.fn().mockImplementation(() => {
      throw new Error('exec quirk');
    });
    Object.defineProperty(document, 'execCommand', {
      configurable: true,
      writable: true,
      value: exec,
    });

    await expect(copyToClipboard('throwy')).rejects.toThrow(/exec quirk/);
    // No leftover textareas in the DOM.
    expect(document.querySelectorAll('textarea').length).toBe(0);
  });
});
