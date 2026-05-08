/**
 * useCoPilotThread tests covering the four high-priority bug fixes:
 *
 *   F1 - tool_call_complete arriving after `final` patches the original
 *        bubble (not a freshly-spawned empty one). The tool entry must
 *        end in 'complete' status, not stuck on 'running'.
 *   F2 - confirm() catch path clears `pending` so the composer unlocks.
 *   F3 - WebSocket unclean close schedules an exponential-backoff
 *        reconnect (1s on the first attempt).
 *   F4 - send() in flight is dropped when resetTo() lands first, so the
 *        old thread_id never overwrites the new thread.
 *
 * The hook imports `api` and `subscribeAgent` from '../../api' (a single
 * module). vi.mock targets that module so we can swap implementations
 * per-test without touching the network.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, renderHook, waitFor } from '@testing-library/react';

const {
  startAgentThreadMock,
  sendAgentMessageMock,
  confirmAgentToolMock,
  subscribeAgentMock,
  capturedHandlers,
} = vi.hoisted(() => {
  const startAgentThread = vi.fn();
  const sendAgentMessage = vi.fn();
  const confirmAgentTool = vi.fn();
  // We capture the handlers passed into subscribeAgent so each test can
  // synthesise WS lifecycle events (open/message/close) on demand.
  const captured: { current: unknown } = { current: null };
  const subscribe = vi.fn((_threadId: string, handlers: unknown) => {
    captured.current = handlers;
    return { close: vi.fn() };
  });
  return {
    startAgentThreadMock: startAgentThread,
    sendAgentMessageMock: sendAgentMessage,
    confirmAgentToolMock: confirmAgentTool,
    subscribeAgentMock: subscribe,
    capturedHandlers: captured,
  };
});

vi.mock('../../api', () => ({
  api: {
    startAgentThread: startAgentThreadMock,
    sendAgentMessage: sendAgentMessageMock,
    confirmAgentTool: confirmAgentToolMock,
  },
  subscribeAgent: subscribeAgentMock,
}));

import { useCoPilotThread } from './useCoPilotThread';
import { useV2Store } from '../store/store';
import type { AgentWsEvent } from './types';

interface CapturedHandlers {
  onMessage: (msg: AgentWsEvent) => void;
  onOpen?: () => void;
  onError?: (ev: Event) => void;
  onClose?: (ev: CloseEvent) => void;
}

function getHandlers(): CapturedHandlers {
  const h = capturedHandlers.current as CapturedHandlers | null;
  if (!h) throw new Error('subscribeAgent has not been called yet');
  return h;
}

beforeEach(() => {
  startAgentThreadMock.mockReset();
  sendAgentMessageMock.mockReset();
  confirmAgentToolMock.mockReset();
  subscribeAgentMock.mockClear();
  capturedHandlers.current = null;
  vi.useRealTimers();
});

afterEach(() => {
  vi.useRealTimers();
});

describe('useCoPilotThread', () => {
  it('F1: tool_call_complete after final patches the original bubble', async () => {
    const { result } = renderHook(() =>
      useCoPilotThread({ initialThreadId: 'tid-1' }),
    );
    // Wait for the WS subscription to be wired up.
    await waitFor(() => {
      expect(subscribeAgentMock).toHaveBeenCalled();
    });
    const handlers = getHandlers();

    // 1. text_delta + tool_call_proposal land in the same agent bubble.
    act(() => {
      handlers.onMessage({
        type: 'text_delta',
        thread_id: 'tid-1',
        message_id: 'm-1',
        text: 'thinking...',
        ts: 1,
      });
      handlers.onMessage({
        type: 'tool_call_proposal',
        thread_id: 'tid-1',
        tool_call_id: 'tc-1',
        tool: 'list-runs',
        args: { dataset: 'demo' },
        preview_cmd: 'list-runs --dataset demo',
        ts: 2,
      });
      handlers.onMessage({
        type: 'tool_call_running',
        thread_id: 'tid-1',
        tool_call_id: 'tc-1',
        tool: 'list-runs',
        ts: 3,
      });
    });

    // 2. final closes the bubble (streaming = false).
    act(() => {
      handlers.onMessage({
        type: 'final',
        thread_id: 'tid-1',
        message_id: 'm-1',
        text: 'thinking... done.',
        ts: 4,
      });
    });

    // 3. tool_call_complete arrives AFTER final. Pre-fix this would land
    //    in a brand-new empty bubble and the original tool stays running.
    act(() => {
      handlers.onMessage({
        type: 'tool_call_complete',
        thread_id: 'tid-1',
        tool_call_id: 'tc-1',
        tool: 'list-runs',
        ok: true,
        output: 'two runs',
        exit_code: 0,
        ts: 5,
      });
    });

    const msgs = result.current.messages;
    // Find the tool entry across all messages.
    const tools = msgs.flatMap((m) => m.tools);
    const tc1 = tools.find((t) => t.tool_call_id === 'tc-1');
    expect(tc1).toBeDefined();
    expect(tc1?.status).toBe('complete');
    expect(tc1?.output_preview).toBe('two runs');
    // No second empty bubble was spawned by the complete event.
    const agentBubbles = msgs.filter((m) => m.role === 'agent');
    expect(agentBubbles.length).toBe(1);
  });

  it('F2: confirm() catch clears pending so composer unlocks', async () => {
    const { result } = renderHook(() =>
      useCoPilotThread({ initialThreadId: 'tid-2' }),
    );
    await waitFor(() => {
      expect(subscribeAgentMock).toHaveBeenCalled();
    });
    const handlers = getHandlers();

    // Drive a confirmation_required event so `pending` is set.
    act(() => {
      handlers.onMessage({
        type: 'confirmation_required',
        thread_id: 'tid-2',
        tool_call_id: 'tc-9',
        tool: 'run-eval',
        args: {},
        preview_cmd: 'run-eval --dataset x',
        ts: 1,
      });
    });
    expect(result.current.pending).not.toBeNull();

    confirmAgentToolMock.mockRejectedValueOnce(new Error('network down'));

    await act(async () => {
      await result.current.confirm(true);
    });

    expect(result.current.pending).toBeNull();
    expect(result.current.status).toBe('error');
  });

  it('F3: unclean WS close schedules a reconnect with backoff', async () => {
    vi.useFakeTimers();
    // Force the production jitter to 0 so the reconnect schedules at
    // exactly the base backoff (1000ms) rather than 1000 + [0, 500)ms.
    // Without this stub, vi.advanceTimersByTime(1000) below misses
    // the timer when the random jitter pushes it past 1000ms,
    // causing intermittent test failures. Restored at the end.
    const randomSpy = vi.spyOn(Math, 'random').mockReturnValue(0);

    try {
      renderHook(() => useCoPilotThread({ initialThreadId: 'tid-3' }));
      // initial subscribe call fires synchronously inside the effect.
      expect(subscribeAgentMock).toHaveBeenCalledTimes(1);
      const handlers = getHandlers();

      // Simulate an unclean close.
      act(() => {
        handlers.onClose?.({
          wasClean: false,
          code: 1006,
          reason: 'lost',
        } as CloseEvent);
      });

      // No immediate reconnect - it's scheduled via setTimeout.
      expect(subscribeAgentMock).toHaveBeenCalledTimes(1);

      // Advance past the 1s first-attempt backoff. With Math.random
      // stubbed to 0 the delay is exactly 1000ms, so this fires the
      // timer deterministically.
      await act(async () => {
        vi.advanceTimersByTime(1000);
      });

      expect(subscribeAgentMock).toHaveBeenCalledTimes(2);
    } finally {
      randomSpy.mockRestore();
      vi.useRealTimers();
    }
  });

  it('F4: send() resolving after resetTo() does not stomp the new thread', async () => {
    let resolveStart: ((v: { thread_id: string }) => void) | null = null;
    startAgentThreadMock.mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveStart = resolve;
        }),
    );

    const { result } = renderHook(() => useCoPilotThread());
    expect(result.current.threadId).toBeNull();

    // Kick off send (promise stays pending).
    let sendPromise: Promise<void>;
    act(() => {
      sendPromise = result.current.send('hello');
    });

    // User switches threads mid-flight.
    act(() => {
      result.current.resetTo('tid-NEW');
    });
    expect(result.current.threadId).toBe('tid-NEW');

    // Now the original send resolves with a stale id.
    await act(async () => {
      resolveStart?.({ thread_id: 'tid-OLD' });
      await sendPromise;
    });

    // The reset wins - threadId stays on the user's choice.
    expect(result.current.threadId).toBe('tid-NEW');
  });

  it('F6: same-frame double send creates one thread, not two', async () => {
    // Customer scenario: user types and Cmd+Enters fast at the
    // start of a fresh thread. Without the in-flight guard, two
    // parallel api.startAgentThread() calls fire and the backend
    // creates two threads. The FE only tracks one (whichever
    // setThreadId call lands last), so the user effectively
    // loses one of the conversations they started.
    //
    // Pin: the second send call inside the same paint frame
    // (before the first await yields back to setState commits)
    // must early-out and NOT trigger startAgentThreadMock.
    let resolveStart: ((v: { thread_id: string }) => void) | null = null;
    startAgentThreadMock.mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveStart = resolve;
        }),
    );

    const { result } = renderHook(() => useCoPilotThread());

    // First send: enters, marks in-flight, awaits.
    let firstPromise: Promise<void> | undefined;
    act(() => {
      firstPromise = result.current.send('hello');
    });
    expect(startAgentThreadMock).toHaveBeenCalledTimes(1);

    // Second send IN THE SAME act() context (before the first
    // resolves). The synchronous in-flight ref is true; this
    // should early-out without spawning a second mock call.
    let secondPromise: Promise<void> | undefined;
    act(() => {
      secondPromise = result.current.send('hello again');
    });
    expect(startAgentThreadMock).toHaveBeenCalledTimes(1);

    // Resolve the first send. The in-flight ref clears in the
    // finally block.
    await act(async () => {
      resolveStart?.({ thread_id: 'tid-1' });
      await firstPromise;
      await secondPromise;
    });

    // Still only ONE start call, even though two sends were
    // attempted. The user gets one thread.
    expect(startAgentThreadMock).toHaveBeenCalledTimes(1);
    expect(result.current.threadId).toBe('tid-1');
  });

  it('start_tour: tool_call_proposal with tool=start_tour calls setTour on the store', async () => {
    // Reset the tour state so we can detect the interception cleanly.
    useV2Store.setState({ tourActiveId: null, tourStep: 0 });

    renderHook(() => useCoPilotThread({ initialThreadId: 'tid-tour' }));
    await waitFor(() => {
      expect(subscribeAgentMock).toHaveBeenCalled();
    });
    const handlers = getHandlers();

    act(() => {
      handlers.onMessage({
        type: 'tool_call_proposal',
        thread_id: 'tid-tour',
        tool_call_id: 'tc-tour-1',
        tool: 'start_tour',
        args: { tour_id: 'runEval.v1' },
        preview_cmd: 'evalyn start_tour --tour-id runEval.v1',
        ts: 1,
      });
    });

    expect(useV2Store.getState().tourActiveId).toBe('runEval.v1');
    expect(useV2Store.getState().tourStep).toBe(0);

    // Cleanup: clear tour state for subsequent tests.
    useV2Store.setState({ tourActiveId: null, tourStep: 0 });
  });

  it('start_tour: malformed args (no tour_id) does not crash and leaves the store unchanged', async () => {
    useV2Store.setState({ tourActiveId: null, tourStep: 0 });

    renderHook(() => useCoPilotThread({ initialThreadId: 'tid-tour-bad' }));
    await waitFor(() => {
      expect(subscribeAgentMock).toHaveBeenCalled();
    });
    const handlers = getHandlers();

    act(() => {
      handlers.onMessage({
        type: 'tool_call_proposal',
        thread_id: 'tid-tour-bad',
        tool_call_id: 'tc-tour-2',
        tool: 'start_tour',
        args: {},
        preview_cmd: 'evalyn start_tour',
        ts: 1,
      });
    });

    expect(useV2Store.getState().tourActiveId).toBeNull();
  });

  it('F5: error / unclean-close logs are throttled to one per outage', async () => {
    // Pin the throttle introduced two ticks ago. Without it, a
    // 5-attempt reconnect ladder spammed 10+ identical
    // "agent ws error" / "agent ws closed unexpectedly" lines
    // per outage. Throttle reuses the existing `attempt` counter:
    // log only when attempt === 0.
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    try {
      renderHook(() => useCoPilotThread({ initialThreadId: 'tid-throttle' }));
      await waitFor(() => {
        expect(subscribeAgentMock).toHaveBeenCalled();
      });
      const handlers = getHandlers();

      // First error in the outage window: logs once.
      act(() => {
        handlers.onError?.(new Event('error'));
      });
      const firstCount = errorSpy.mock.calls.filter((c) =>
        typeof c[0] === 'string' && c[0].includes('agent ws error'),
      ).length;
      expect(firstCount).toBe(1);

      // First unclean close fires onClose with attempt still 0
      // (incremented inside the handler AFTER the log check).
      // After this onClose returns, attempt is 1, so subsequent
      // errors/closes are throttled.
      act(() => {
        handlers.onClose?.({
          wasClean: false,
          code: 1006,
          reason: 'lost',
        } as CloseEvent);
      });
      const closeCount = errorSpy.mock.calls.filter((c) =>
        typeof c[0] === 'string' && c[0].includes('agent ws closed'),
      ).length;
      expect(closeCount).toBe(1);

      // Subsequent errors within the same outage: throttled silent.
      // (Note: subscribeAgentMock would have re-fired the open path
      // via the reconnect setTimeout, but jsdom's fake timers aren't
      // engaged here so the outer reconnect doesn't fire. The inner
      // attempt counter stays at 1+ and throttles.)
      act(() => {
        handlers.onError?.(new Event('error'));
        handlers.onClose?.({
          wasClean: false,
          code: 1006,
          reason: 'lost',
        } as CloseEvent);
      });
      const errorCount = errorSpy.mock.calls.filter((c) =>
        typeof c[0] === 'string' && c[0].includes('agent ws error'),
      ).length;
      const closeCount2 = errorSpy.mock.calls.filter((c) =>
        typeof c[0] === 'string' && c[0].includes('agent ws closed'),
      ).length;
      expect(errorCount).toBe(1); // still 1 - throttled
      expect(closeCount2).toBe(1); // still 1 - throttled
    } finally {
      errorSpy.mockRestore();
    }
  });
});
