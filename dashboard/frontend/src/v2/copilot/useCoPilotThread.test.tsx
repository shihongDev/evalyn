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

    // Advance past the 1s first-attempt backoff.
    await act(async () => {
      vi.advanceTimersByTime(1000);
    });

    expect(subscribeAgentMock).toHaveBeenCalledTimes(2);

    vi.useRealTimers();
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
});
