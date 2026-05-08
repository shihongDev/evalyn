/**
 * v2ws backoff tests.
 *
 * The previous fixed-2s reconnect delay was wasteful when the
 * server was down for real (deploy hiccup, network partition).
 * Exponential backoff with cap + jitter prevents thundering herd
 * on recovery while keeping fast reconnect for short blips.
 *
 * Pin the contract at the public boundary - the unexported
 * exponential math is hidden behind `_computeBackoffMs` which
 * we export for testability. The schedule itself involves
 * timers + WebSocket; integration coverage of those would be
 * brittle here, so we leave that to manual smoke tests.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  _computeBackoffMs,
  _resetV2EventStream,
  startV2EventStream,
} from './v2ws';

describe('_computeBackoffMs', () => {
  beforeEach(() => {
    // Pin Math.random so the jitter component is deterministic.
    // 0 -> the lower bound of [0, JITTER_MS).
    vi.spyOn(Math, 'random').mockReturnValue(0);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('first attempt uses the base delay', () => {
    expect(_computeBackoffMs(0)).toBe(2000);
  });

  it('doubles per attempt: 2s -> 4s -> 8s -> 16s', () => {
    expect(_computeBackoffMs(1)).toBe(4000);
    expect(_computeBackoffMs(2)).toBe(8000);
    expect(_computeBackoffMs(3)).toBe(16000);
  });

  it('caps at 30s past the doubling threshold', () => {
    // attempt=4 -> 32000 raw, capped to 30000
    expect(_computeBackoffMs(4)).toBe(30000);
    // Far past the cap stays capped (no overflow into negative
    // due to integer math; the cap is the load-bearing guard).
    expect(_computeBackoffMs(20)).toBe(30000);
  });

  it('adds jitter from Math.random', () => {
    // Math.random=0.5 -> 0.5 * 500 = 250ms jitter
    vi.spyOn(Math, 'random').mockReturnValue(0.5);
    expect(_computeBackoffMs(0)).toBe(2250);
    // At the cap: cap + jitter = 30250
    expect(_computeBackoffMs(10)).toBe(30250);
  });

  it('jitter is bounded below 500ms', () => {
    // Math.random returns [0, 1) so jitter * 500 is < 500.
    // Pin the upper bound by mocking random close to 1.
    vi.spyOn(Math, 'random').mockReturnValue(0.999);
    expect(_computeBackoffMs(0)).toBeLessThan(2500);
    expect(_computeBackoffMs(0)).toBeGreaterThanOrEqual(2000);
  });
});

describe('error log throttle', () => {
  // Stub the WebSocket constructor: capture the instance so the
  // test can fire 'error' / 'open' events at will. The real
  // WebSocket would try to connect and fail in jsdom; we want a
  // controllable stub.
  class StubWebSocket {
    readyState = 0;
    listeners: Record<string, Array<(e: Event) => void>> = {};
    constructor(public url: string) {
      // No-op; we don't actually connect.
    }
    addEventListener(type: string, handler: (e: Event) => void) {
      (this.listeners[type] = this.listeners[type] || []).push(handler);
    }
    fire(type: string, evt: Event = new Event(type)) {
      for (const h of this.listeners[type] ?? []) h(evt);
    }
    close() {
      this.fire('close', new CloseEvent('close'));
    }
  }

  let warnSpy: ReturnType<typeof vi.spyOn>;
  let lastWs: StubWebSocket | null = null;

  beforeEach(() => {
    _resetV2EventStream();
    lastWs = null;
    warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    // Override the WebSocket global with our stub. Each `new
    // WebSocket(...)` in startV2EventStream returns a fresh stub
    // we can drive from the test.
    vi.stubGlobal(
      'WebSocket',
      class extends StubWebSocket {
        constructor(url: string) {
          super(url);
          lastWs = this;
        }
      } as unknown as typeof WebSocket,
    );
  });

  afterEach(() => {
    warnSpy.mockRestore();
    vi.unstubAllGlobals();
    _resetV2EventStream();
  });

  it('logs once per outage and resets after a successful open', () => {
    startV2EventStream();
    expect(lastWs).not.toBeNull();

    // First error in the outage window: logs once.
    lastWs!.fire('error');
    expect(warnSpy).toHaveBeenCalledTimes(1);

    // Second + third errors in the SAME outage: throttled.
    lastWs!.fire('error');
    lastWs!.fire('error');
    expect(warnSpy).toHaveBeenCalledTimes(1);

    // Successful open resets the throttle so future errors log fresh.
    lastWs!.fire('open');
    lastWs!.fire('error');
    expect(warnSpy).toHaveBeenCalledTimes(2);
  });
});
