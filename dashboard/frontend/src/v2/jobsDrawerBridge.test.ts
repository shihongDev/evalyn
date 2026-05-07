/**
 * jobsDrawerBridge tests.
 *
 * The bridge is load-bearing for the SystemStatusCard's
 * "Failures (24h)" click-through. A regression in the emitter
 * (e.g. listeners not getting notified, or the nonce not
 * advancing) would silently break the path - the user clicks but
 * the drawer doesn't open. These tests pin the contract.
 *
 * The bridge is module-singleton state, so each test resets the
 * known-current event to a clean baseline by un-subscribing all
 * listeners and triggering a `closeJobsDrawer()` to drop the
 * "open=true" residue from any prior open call.
 */

import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  closeJobsDrawer,
  openJobsDrawer,
  subscribeJobsDrawer,
  type JobsDrawerOpenEvent,
} from './jobsDrawerBridge';

afterEach(() => {
  // Reset module state between tests. closeJobsDrawer() advances
  // the nonce too, but that's fine - tests only assert relative
  // monotonicity, not absolute values.
  closeJobsDrawer();
});

describe('jobsDrawerBridge', () => {
  it('subscribe replays the current event immediately on subscribe', () => {
    const events: JobsDrawerOpenEvent[] = [];
    const unsub = subscribeJobsDrawer((e) => events.push(e));
    // Synchronous replay: the listener has already fired with the
    // current closed state.
    expect(events).toHaveLength(1);
    expect(events[0].open).toBe(false);
    unsub();
  });

  it('openJobsDrawer notifies all listeners with open=true', () => {
    const a: JobsDrawerOpenEvent[] = [];
    const b: JobsDrawerOpenEvent[] = [];
    subscribeJobsDrawer((e) => a.push(e));
    subscribeJobsDrawer((e) => b.push(e));

    openJobsDrawer();

    // 1 replay + 1 open notification each.
    expect(a).toHaveLength(2);
    expect(b).toHaveLength(2);
    expect(a[1].open).toBe(true);
    expect(b[1].open).toBe(true);
  });

  it('forwards failureFilter option to subscribers', () => {
    const events: JobsDrawerOpenEvent[] = [];
    subscribeJobsDrawer((e) => events.push(e));

    openJobsDrawer({ failureFilter: true });

    const last = events[events.length - 1];
    expect(last.open).toBe(true);
    expect(last.failureFilter).toBe(true);
  });

  it('omits failureFilter when not provided', () => {
    const events: JobsDrawerOpenEvent[] = [];
    subscribeJobsDrawer((e) => events.push(e));

    openJobsDrawer();

    const last = events[events.length - 1];
    expect(last.open).toBe(true);
    expect(last.failureFilter).toBeUndefined();
  });

  it('nonce strictly advances on every emit', () => {
    const events: JobsDrawerOpenEvent[] = [];
    subscribeJobsDrawer((e) => events.push(e));

    const initialNonce = events[0].nonce;
    openJobsDrawer({ failureFilter: true });
    openJobsDrawer({ failureFilter: true }); // same options, must still bump
    closeJobsDrawer();

    // Each subsequent event has a strictly larger nonce than the
    // previous one. This is the property the AppShell subscriber
    // relies on to re-apply filters even when the value is
    // unchanged across two clicks.
    for (let i = 1; i < events.length; i++) {
      expect(events[i].nonce).toBeGreaterThan(events[i - 1].nonce);
    }
    // And the very first nonce post-replay must be < the second
    // open's nonce (sanity check on relative ordering).
    expect(events[events.length - 1].nonce).toBeGreaterThan(initialNonce);
  });

  it('closeJobsDrawer notifies with open=false', () => {
    const events: JobsDrawerOpenEvent[] = [];
    subscribeJobsDrawer((e) => events.push(e));

    openJobsDrawer({ failureFilter: true });
    closeJobsDrawer();

    const last = events[events.length - 1];
    expect(last.open).toBe(false);
  });

  it('unsubscribe stops further notifications', () => {
    const fn = vi.fn();
    const unsub = subscribeJobsDrawer(fn);
    expect(fn).toHaveBeenCalledTimes(1); // initial replay

    unsub();
    openJobsDrawer();
    expect(fn).toHaveBeenCalledTimes(1); // no new calls after unsub
  });

  it('late subscriber sees the latest known event, not just defaults', () => {
    // Open BEFORE subscribing. The bridge stores currentEvent so
    // a late-mounting listener can pick up the most recent state
    // without missing it. Important for AppShell which mounts
    // late on a route transition - if the drawer was opened
    // during the transition window, the new mount must still
    // see open=true.
    openJobsDrawer({ failureFilter: true });

    const events: JobsDrawerOpenEvent[] = [];
    subscribeJobsDrawer((e) => events.push(e));

    // Replay sees the prior open call.
    expect(events[0].open).toBe(true);
    expect(events[0].failureFilter).toBe(true);
  });
});
