/**
 * Tests for store.subscribeJob — the WS dispatcher.
 *
 * We swap in a `MockWebSocket` factory so no real network call happens.
 * The mock exposes helpers to simulate incoming messages, server close
 * events, and to inspect the URL the store opened (for `?since=` resume).
 *
 * Covers:
 *   - Dispatches stdout/stderr line events to jobEvents
 *   - Tracks lastEventId per job
 *   - exit event flips job state and stores duration
 *   - Reconnect: when WS closes mid-run, store reopens with `?since=N`
 *   - Reconnect skipped after the job has finalized (exit received)
 *   - unsubscribeJob closes socket and prevents reconnect
 */

import { describe, expect, test, beforeEach, afterEach, vi } from 'vitest';
import { useStore, __resetStore, __jobConnsForTest } from '../store';

/* ------------------------- Mock WebSocket --------------------------- */
type Listener = (ev: unknown) => void;

class MockWebSocket {
  static instances: MockWebSocket[] = [];

  url: string;
  readyState = 0;
  listeners: Record<string, Listener[]> = {};

  constructor(url: string) {
    this.url = url;
    MockWebSocket.instances.push(this);
  }

  addEventListener(type: string, fn: Listener): void {
    (this.listeners[type] ??= []).push(fn);
  }

  removeEventListener(type: string, fn: Listener): void {
    this.listeners[type] = (this.listeners[type] ?? []).filter((f) => f !== fn);
  }

  close(): void {
    this.readyState = 3;
    for (const fn of this.listeners.close ?? []) fn({ code: 1000 });
  }

  /** Test helper: simulate the server pushing a JSON frame. */
  sendFrame(payload: unknown): void {
    for (const fn of this.listeners.message ?? []) fn({ data: JSON.stringify(payload) } as MessageEvent);
  }

  /** Test helper: simulate an unexpected drop. */
  triggerClose(): void {
    this.readyState = 3;
    for (const fn of this.listeners.close ?? []) fn({ code: 1006 });
  }

  /** Test helper: open. */
  triggerOpen(): void {
    this.readyState = 1;
    for (const fn of this.listeners.open ?? []) fn({});
  }
}

const factory = (url: string): WebSocket => new MockWebSocket(url) as unknown as WebSocket;

beforeEach(() => {
  vi.useFakeTimers();
  MockWebSocket.instances = [];
  __resetStore();
  // Seed the job so subscribe events have something to attach to.
  useStore.getState().upsertJob({
    id: 'j-1',
    cmd: 'evalyn list-runs',
    status: 'running',
    startedAt: Date.now(),
  });
});

afterEach(() => {
  vi.useRealTimers();
  __resetStore();
});

describe('store.subscribeJob', () => {
  test('opens a WebSocket and dispatches line events to jobEvents', () => {
    useStore.getState().subscribeJob('j-1', { factory });
    const ws = MockWebSocket.instances.at(-1)!;
    expect(ws.url).toContain('/ws/jobs/j-1');
    ws.triggerOpen();
    ws.sendFrame({ type: 'stdout', line: 'hello', event_id: 1 });
    ws.sendFrame({ type: 'stderr', line: 'warn', event_id: 2 });
    const lines = useStore.getState().jobEvents.get('j-1') ?? [];
    expect(lines).toHaveLength(2);
    expect(lines[0]).toMatchObject({ kind: 'stdout', text: 'hello' });
    expect(lines[1]).toMatchObject({ kind: 'stderr', text: 'warn' });
    expect(useStore.getState().jobLastEventId.get('j-1')).toBe(2);
  });

  test('exit event flips job state and records duration', () => {
    useStore.getState().subscribeJob('j-1', { factory });
    const ws = MockWebSocket.instances.at(-1)!;
    ws.sendFrame({ type: 'stdout', line: 'working', event_id: 1 });
    ws.sendFrame({ type: 'exit', code: 0, duration: '0:05', event_id: 2 });
    const job = useStore.getState().jobs.get('j-1');
    expect(job?.status).toBe('complete');
    expect(job?.exitCode).toBe(0);
    expect(job?.duration).toBe('0:05');
  });

  test('non-zero exit code marks job failed', () => {
    useStore.getState().subscribeJob('j-1', { factory });
    const ws = MockWebSocket.instances.at(-1)!;
    ws.sendFrame({ type: 'exit', code: 1, event_id: 1 });
    expect(useStore.getState().jobs.get('j-1')?.status).toBe('failed');
  });

  test('progress event updates job.progress + eta', () => {
    useStore.getState().subscribeJob('j-1', { factory });
    const ws = MockWebSocket.instances.at(-1)!;
    ws.sendFrame({ type: 'progress', progress: 0.42, eta: '0:30', event_id: 5 });
    const job = useStore.getState().jobs.get('j-1');
    expect(job?.progress).toBeCloseTo(0.42);
    expect(job?.eta).toBe('0:30');
    expect(useStore.getState().jobLastEventId.get('j-1')).toBe(5);
  });

  test('truncated event appends a warn marker line', () => {
    useStore.getState().subscribeJob('j-1', { factory });
    const ws = MockWebSocket.instances.at(-1)!;
    ws.sendFrame({ type: 'truncated', count: 1234, event_id: 7 });
    const lines = useStore.getState().jobEvents.get('j-1') ?? [];
    expect(lines[0].kind).toBe('warn');
    expect(lines[0].text).toContain('1234');
  });

  test('reconnect uses ?since=<lastEventId> when WS drops mid-run', () => {
    useStore.getState().subscribeJob('j-1', { factory });
    const first = MockWebSocket.instances.at(-1)!;
    first.sendFrame({ type: 'stdout', line: 'one', event_id: 11 });
    first.triggerClose();
    // First retry delay is 500ms — fast-forward.
    vi.advanceTimersByTime(600);
    const second = MockWebSocket.instances.at(-1)!;
    expect(second).not.toBe(first);
    expect(second.url).toContain('since=11');
  });

  test('does NOT reconnect after exit event flips job to complete', () => {
    useStore.getState().subscribeJob('j-1', { factory });
    const first = MockWebSocket.instances.at(-1)!;
    first.sendFrame({ type: 'exit', code: 0, event_id: 9 });
    first.triggerClose();
    vi.advanceTimersByTime(2000);
    // Only one socket should have been created.
    expect(MockWebSocket.instances.length).toBe(1);
  });

  test('unsubscribeJob closes the socket and prevents reconnect', () => {
    useStore.getState().subscribeJob('j-1', { factory });
    const ws = MockWebSocket.instances.at(-1)!;
    useStore.getState().unsubscribeJob('j-1');
    expect(ws.readyState).toBe(3);
    vi.advanceTimersByTime(5000);
    // No second socket should have been created.
    expect(MockWebSocket.instances.length).toBe(1);
    // Connection registry is cleaned up.
    expect(__jobConnsForTest.has('j-1')).toBe(false);
  });

  test('subscribing twice for the same job is a no-op', () => {
    useStore.getState().subscribeJob('j-1', { factory });
    useStore.getState().subscribeJob('j-1', { factory });
    expect(MockWebSocket.instances.length).toBe(1);
  });
});
