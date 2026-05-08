/**
 * Unit tests for notifications.ts: feature-detection + permission /
 * visibility gating.
 *
 * jsdom doesn't ship a real Notification API, so the default
 * isSupported() path is the "unsupported" branch. These tests
 * exercise both that path AND the supported path via vi.stubGlobal.
 *
 * The notifyJobTerminal path has three gates that all have to pass
 * before a Notification is constructed:
 *   1. API supported
 *   2. permission === 'granted'
 *   3. document.visibilityState === 'hidden'
 * Each test pins one of those gates in isolation.
 */

import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  notificationPermission,
  notifyJobTerminal,
  requestNotificationPermission,
} from './notifications';

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('feature-detection (no Notification API)', () => {
  it('notificationPermission returns "denied" when Notification is undefined', () => {
    // jsdom default: no Notification global.
    expect(notificationPermission()).toBe('denied');
  });

  it('requestNotificationPermission resolves to "denied" when unsupported', async () => {
    expect(await requestNotificationPermission()).toBe('denied');
  });

  it('notifyJobTerminal is a no-op when unsupported (no throw)', () => {
    // Just verifying it doesn't blow up. There's no observable side
    // effect to assert on the no-op path.
    expect(() =>
      notifyJobTerminal({
        jobId: 'j1',
        cliId: 'run-eval',
        status: 'complete',
      }),
    ).not.toThrow();
  });
});

/** Helper: install a mock Notification global with a given permission
 * state and a constructor we can inspect. Returns the constructor
 * spy and a `requestPermission` spy so tests can assert on calls. */
function installMockNotification(initialPermission: 'granted' | 'denied' | 'default' = 'granted') {
  const ctorSpy = vi.fn();
  const requestSpy = vi.fn(async () => initialPermission);
  // The Notification "type" is a constructor + static permission +
  // static requestPermission. Stub all three.
  class MockNotification {
    constructor(...args: unknown[]) {
      ctorSpy(...args);
    }
    onclick: (() => void) | null = null;
    close = vi.fn();
    static permission = initialPermission;
    static requestPermission = requestSpy;
  }
  vi.stubGlobal('Notification', MockNotification);
  return { ctorSpy, requestSpy, MockNotification };
}

describe('with mocked Notification API', () => {
  it('notificationPermission reads the static permission field', () => {
    installMockNotification('granted');
    expect(notificationPermission()).toBe('granted');
  });

  it('requestNotificationPermission delegates to the API', async () => {
    const { requestSpy } = installMockNotification('default');
    const result = await requestNotificationPermission();
    expect(requestSpy).toHaveBeenCalledTimes(1);
    expect(result).toBe('default');
  });

  it('requestNotificationPermission catches a thrown rejection and returns "denied"', async () => {
    class ThrowingNotification {
      static permission = 'default';
      static requestPermission = () => {
        throw new Error('legacy callback-only API');
      };
    }
    vi.stubGlobal('Notification', ThrowingNotification);
    expect(await requestNotificationPermission()).toBe('denied');
  });
});

describe('notifyJobTerminal gating', () => {
  it('no-ops when permission is not granted', () => {
    const { ctorSpy } = installMockNotification('default');
    Object.defineProperty(document, 'visibilityState', {
      value: 'hidden',
      configurable: true,
    });
    notifyJobTerminal({
      jobId: 'j1',
      cliId: 'run-eval',
      status: 'complete',
    });
    expect(ctorSpy).not.toHaveBeenCalled();
  });

  it('no-ops when tab is visible (foreground)', () => {
    const { ctorSpy } = installMockNotification('granted');
    Object.defineProperty(document, 'visibilityState', {
      value: 'visible',
      configurable: true,
    });
    notifyJobTerminal({
      jobId: 'j1',
      cliId: 'run-eval',
      status: 'complete',
    });
    expect(ctorSpy).not.toHaveBeenCalled();
  });

  it('fires when permission=granted AND tab is hidden', () => {
    const { ctorSpy } = installMockNotification('granted');
    Object.defineProperty(document, 'visibilityState', {
      value: 'hidden',
      configurable: true,
    });
    notifyJobTerminal({
      jobId: 'j1',
      cliId: 'run-eval',
      status: 'failed',
      exitCode: 2,
      durationS: 12.3,
    });
    expect(ctorSpy).toHaveBeenCalledTimes(1);
    const [title, options] = ctorSpy.mock.calls[0];
    expect(title).toBe('run-eval failed');
    expect(options).toMatchObject({
      tag: 'evalyn-job-j1',
      // failed jobs are NOT silent so the user notices them.
      silent: false,
      // Failed jobs persist until dismissed - the OS auto-dismiss
      // (~5s) is too short for a "your eval crashed" alert when
      // the user is in another window.
      requireInteraction: true,
    });
    // Body includes exit code + duration.
    expect(options.body).toContain('exit 2');
    expect(options.body).toContain('12.3s');
  });

  it('uses silent=true and requireInteraction=false on success (less intrusive)', () => {
    const { ctorSpy } = installMockNotification('granted');
    Object.defineProperty(document, 'visibilityState', {
      value: 'hidden',
      configurable: true,
    });
    notifyJobTerminal({
      jobId: 'j1',
      cliId: 'run-eval',
      status: 'complete',
      exitCode: 0,
    });
    const [, options] = ctorSpy.mock.calls[0];
    expect(options.silent).toBe(true);
    // Successful jobs auto-dismiss; only failures pin until acked.
    expect(options.requireInteraction).toBe(false);
  });

  it('per-job_id tag means a queued -> failed sequence replaces (not stacks)', () => {
    const { ctorSpy } = installMockNotification('granted');
    Object.defineProperty(document, 'visibilityState', {
      value: 'hidden',
      configurable: true,
    });
    notifyJobTerminal({
      jobId: 'j1',
      cliId: 'run-eval',
      status: 'complete',
    });
    notifyJobTerminal({
      jobId: 'j1',
      cliId: 'run-eval',
      status: 'failed',
    });
    // Both calls used the same tag - the OS dedupes by tag, so the
    // second notification replaces the first rather than stacking.
    expect(ctorSpy.mock.calls[0][1].tag).toBe('evalyn-job-j1');
    expect(ctorSpy.mock.calls[1][1].tag).toBe('evalyn-job-j1');
  });
});
