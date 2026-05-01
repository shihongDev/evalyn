/**
 * Tests for the agent slice of the Zustand store.
 *
 * Covers each tagged-union branch of `dispatchAgentEvent`:
 *   - text_delta accumulates per message_id
 *   - tool_call_proposal pushes a card
 *   - tool_call_running flips status + records jobId
 *   - tool_call_complete records output + flips to complete (or error)
 *   - confirmation_required sets pendingConfirmation
 *   - error sets agent.error and flips status
 *   - final closes the streaming bubble + records suggestions
 *
 * Plus the action-level coverage:
 *   - sendChatMessage: appends user message, posts to /api/agent/chat,
 *     persists the returned threadId.
 *   - confirmAgent: clears pendingConfirmation + posts approve flag.
 *   - openSettings / closeSettings flip the modal flag.
 *   - saveProvider / testProvider / setActiveProvider / listProviderModels
 *     hit the right endpoints with the right body.
 */

import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest';
import { useStore, __resetStore, __attachAgentSocketForTest } from '../store';
import type { AgentWsEvent } from '../types/agent';

beforeEach(() => {
  __resetStore();
});

afterEach(() => {
  vi.restoreAllMocks();
});

/* ------------------------ dispatchAgentEvent ------------------------ */

describe('dispatchAgentEvent', () => {
  test('text_delta starts a new assistant bubble and appends', () => {
    const { dispatchAgentEvent } = useStore.getState();
    dispatchAgentEvent({ type: 'text_delta', message_id: 'm1', delta: 'Hello' });
    dispatchAgentEvent({ type: 'text_delta', message_id: 'm1', delta: ', world' });
    const { agent } = useStore.getState();
    expect(agent.messages).toHaveLength(1);
    expect(agent.messages[0].role).toBe('assistant');
    expect(agent.messages[0].text).toBe('Hello, world');
    expect(agent.messages[0].streaming).toBe(true);
    expect(agent.status).toBe('streaming');
  });

  test('tool_call_proposal appends a tool-call card', () => {
    const { dispatchAgentEvent } = useStore.getState();
    dispatchAgentEvent({
      type: 'tool_call_proposal',
      tool_call_id: 't1',
      tool: 'list-runs',
      args: { limit: 5 },
      preview_cmd: 'evalyn list-runs --limit 5',
    });
    const { agent } = useStore.getState();
    expect(agent.messages).toHaveLength(1);
    const card = agent.messages[0].toolCall;
    expect(card?.id).toBe('t1');
    expect(card?.tool).toBe('list-runs');
    expect(card?.status).toBe('proposed');
    expect(card?.previewCmd).toBe('evalyn list-runs --limit 5');
  });

  test('tool_call_running flips status and records jobId', () => {
    const { dispatchAgentEvent } = useStore.getState();
    dispatchAgentEvent({
      type: 'tool_call_proposal',
      tool_call_id: 't1',
      tool: 'list-runs',
      args: {},
      preview_cmd: 'evalyn list-runs',
    });
    dispatchAgentEvent({ type: 'tool_call_running', tool_call_id: 't1', job_id: 'j-9' });
    const card = useStore.getState().agent.messages[0].toolCall;
    expect(card?.status).toBe('running');
    expect(card?.jobId).toBe('j-9');
  });

  test('tool_call_complete records output and flips to complete on exit 0', () => {
    const { dispatchAgentEvent } = useStore.getState();
    dispatchAgentEvent({
      type: 'tool_call_proposal',
      tool_call_id: 't1',
      tool: 'list-runs',
      args: {},
      preview_cmd: 'evalyn list-runs',
    });
    dispatchAgentEvent({
      type: 'tool_call_complete',
      tool_call_id: 't1',
      output: '3 runs found',
      exit_code: 0,
    });
    const card = useStore.getState().agent.messages[0].toolCall;
    expect(card?.status).toBe('complete');
    expect(card?.output).toBe('3 runs found');
  });

  test('tool_call_complete with non-zero exit_code flips to error', () => {
    const { dispatchAgentEvent } = useStore.getState();
    dispatchAgentEvent({
      type: 'tool_call_proposal',
      tool_call_id: 't1',
      tool: 'broken',
      args: {},
      preview_cmd: 'evalyn broken',
    });
    dispatchAgentEvent({
      type: 'tool_call_complete',
      tool_call_id: 't1',
      output: 'oops',
      exit_code: 1,
    });
    const card = useStore.getState().agent.messages[0].toolCall;
    expect(card?.status).toBe('error');
    expect(card?.error).toBe('exit 1');
  });

  test('confirmation_required sets pendingConfirmation + flips card', () => {
    const { dispatchAgentEvent } = useStore.getState();
    dispatchAgentEvent({
      type: 'confirmation_required',
      tool_call_id: 't1',
      tool: 'delete-run',
      args: { id: 'abc' },
      preview_cmd: 'evalyn delete-run abc',
    });
    const { agent } = useStore.getState();
    expect(agent.status).toBe('awaiting_confirmation');
    expect(agent.pendingConfirmation).toEqual({
      toolCallId: 't1',
      tool: 'delete-run',
      args: { id: 'abc' },
      previewCmd: 'evalyn delete-run abc',
    });
    expect(agent.messages[0].toolCall?.status).toBe('awaiting_confirmation');
  });

  test('error event sets agent.error and flips status', () => {
    const { dispatchAgentEvent } = useStore.getState();
    dispatchAgentEvent({
      type: 'error',
      kind: 'auth',
      message: 'Invalid API key',
      provider: 'openai',
    });
    const { agent } = useStore.getState();
    expect(agent.status).toBe('error');
    expect(agent.error).toEqual({
      kind: 'auth',
      message: 'Invalid API key',
      provider: 'openai',
    });
  });

  test('final event closes streaming and records suggestions', () => {
    const { dispatchAgentEvent } = useStore.getState();
    dispatchAgentEvent({ type: 'text_delta', message_id: 'm1', delta: 'Done.' });
    dispatchAgentEvent({
      type: 'final',
      message_id: 'm1',
      text: 'Done.',
      suggestions: [
        { label: 'Annotate failures', cli_id: 'annotate', args: { limit: 20 } },
      ],
    });
    const { agent } = useStore.getState();
    expect(agent.status).toBe('idle');
    expect(agent.messages[0].streaming).toBe(false);
    expect(agent.messages[0].finalSuggestions).toHaveLength(1);
    expect(agent.messages[0].finalSuggestions?.[0]).toEqual({
      label: 'Annotate failures',
      cliId: 'annotate',
      args: { limit: 20 },
    });
  });
});

/* --------------------------- sendChatMessage ------------------------ */

describe('sendChatMessage', () => {
  test('appends user message and POSTs /api/agent/chat for new threads', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      statusText: 'OK',
      json: async () => ({ thread_id: 'th-1' }),
      text: async () => '',
    });
    vi.stubGlobal('fetch', fetchMock);
    // Mock WebSocket so the post-thread-create attach doesn't fail.
    class FakeWs {
      addEventListener(): void {}
      removeEventListener(): void {}
      close(): void {}
    }
    vi.stubGlobal('WebSocket', FakeWs);

    await useStore.getState().sendChatMessage('hello agent');
    const { agent } = useStore.getState();
    expect(agent.threadId).toBe('th-1');
    expect(agent.messages).toHaveLength(1);
    expect(agent.messages[0].role).toBe('user');
    expect(agent.messages[0].text).toBe('hello agent');
    expect(fetchMock).toHaveBeenCalledOnce();
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe('/api/agent/chat');
    expect((init as RequestInit).method).toBe('POST');
    const body = JSON.parse((init as RequestInit).body as string);
    expect(body.message).toBe('hello agent');
  });

  test('reuses thread id on subsequent sends', async () => {
    useStore.setState((s) => ({ agent: { ...s.agent, threadId: 'th-existing' } }));
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      statusText: 'OK',
      json: async () => ({ ok: true }),
      text: async () => '',
    });
    vi.stubGlobal('fetch', fetchMock);

    await useStore.getState().sendChatMessage('follow-up');
    expect(fetchMock).toHaveBeenCalledOnce();
    expect(fetchMock.mock.calls[0][0]).toBe('/api/agent/chat/th-existing');
  });

  test('blank message is a no-op', async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);
    await useStore.getState().sendChatMessage('   ');
    expect(fetchMock).not.toHaveBeenCalled();
    expect(useStore.getState().agent.messages).toHaveLength(0);
  });
});

/* ----------------------------- confirmAgent ------------------------- */

describe('confirmAgent', () => {
  test('approve POSTs the confirm endpoint and clears pending', async () => {
    useStore.setState((s) => ({
      agent: {
        ...s.agent,
        threadId: 'th-1',
        pendingConfirmation: {
          toolCallId: 't1',
          tool: 'delete-run',
          args: {},
          previewCmd: 'evalyn delete-run',
        },
        messages: [
          {
            id: 'tc-t1',
            role: 'assistant',
            toolCall: {
              id: 't1',
              tool: 'delete-run',
              args: {},
              previewCmd: 'evalyn delete-run',
              status: 'awaiting_confirmation',
            },
          },
        ],
        status: 'awaiting_confirmation',
      },
    }));
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      statusText: 'OK',
      json: async () => ({ ok: true }),
      text: async () => '',
    });
    vi.stubGlobal('fetch', fetchMock);

    await useStore.getState().confirmAgent(true);
    const { agent } = useStore.getState();
    expect(agent.pendingConfirmation).toBeNull();
    expect(agent.messages[0].toolCall?.status).toBe('running');
    expect(fetchMock).toHaveBeenCalledOnce();
    expect(fetchMock.mock.calls[0][0]).toBe('/api/agent/chat/th-1/confirm');
    const body = JSON.parse(fetchMock.mock.calls[0][1].body);
    expect(body.approve).toBe(true);
    expect(body.tool_call_id).toBe('t1');
  });

  test('reject flips card status to rejected', async () => {
    useStore.setState((s) => ({
      agent: {
        ...s.agent,
        threadId: 'th-1',
        pendingConfirmation: {
          toolCallId: 't1',
          tool: 'delete-run',
          args: {},
          previewCmd: 'evalyn delete-run',
        },
        messages: [
          {
            id: 'tc-t1',
            role: 'assistant',
            toolCall: {
              id: 't1',
              tool: 'delete-run',
              args: {},
              previewCmd: 'evalyn delete-run',
              status: 'awaiting_confirmation',
            },
          },
        ],
        status: 'awaiting_confirmation',
      },
    }));
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      statusText: 'OK',
      json: async () => ({ ok: true }),
      text: async () => '',
    });
    vi.stubGlobal('fetch', fetchMock);

    await useStore.getState().confirmAgent(false);
    const { agent } = useStore.getState();
    expect(agent.messages[0].toolCall?.status).toBe('rejected');
    expect(agent.status).toBe('idle');
    expect(JSON.parse(fetchMock.mock.calls[0][1].body).approve).toBe(false);
  });
});

/* --------------------------- settings actions ----------------------- */

describe('settings actions', () => {
  test('openSettings / closeSettings flip the modal flag', () => {
    expect(useStore.getState().settingsOpen).toBe(false);
    useStore.getState().openSettings();
    expect(useStore.getState().settingsOpen).toBe(true);
    useStore.getState().closeSettings();
    expect(useStore.getState().settingsOpen).toBe(false);
  });

  test('saveProvider POSTs payload + merges response into provider state', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      statusText: 'OK',
      json: async () => ({
        id: 'openai',
        name: 'OpenAI',
        hasKey: true,
        requiresKey: true,
        model: 'gpt-5.1',
        testStatus: 'untested',
      }),
      text: async () => '',
    });
    vi.stubGlobal('fetch', fetchMock);

    await useStore.getState().saveProvider('openai', { api_key: 'sk-test' });
    const { settings } = useStore.getState();
    expect(settings.providers.openai.hasKey).toBe(true);
    expect(settings.providers.openai.model).toBe('gpt-5.1');
    expect(fetchMock.mock.calls[0][0]).toBe('/api/settings/providers/openai');
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({ api_key: 'sk-test' });
  });

  test('testProvider records ok status on success', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      statusText: 'OK',
      json: async () => ({ ok: true }),
      text: async () => '',
    });
    vi.stubGlobal('fetch', fetchMock);

    await useStore.getState().testProvider('openai');
    expect(useStore.getState().settings.providers.openai.testStatus).toBe('ok');
  });

  test('testProvider records error status on failure', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      statusText: 'OK',
      json: async () => ({ ok: false, error: 'invalid key' }),
      text: async () => '',
    });
    vi.stubGlobal('fetch', fetchMock);

    await useStore.getState().testProvider('openai');
    const p = useStore.getState().settings.providers.openai;
    expect(p.testStatus).toBe('error');
    expect(p.testError).toBe('invalid key');
  });

  test('setActiveProvider POSTs and updates settings.active', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      statusText: 'OK',
      json: async () => ({ ok: true }),
      text: async () => '',
    });
    vi.stubGlobal('fetch', fetchMock);

    await useStore.getState().setActiveProvider('anthropic');
    expect(useStore.getState().settings.active).toBe('anthropic');
    expect(fetchMock.mock.calls[0][0]).toBe('/api/settings/active');
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({ provider: 'anthropic' });
  });

  test('listProviderModels caches the result on the provider state', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      statusText: 'OK',
      json: async () => ({ models: ['gpt-5.1', 'gpt-5'] }),
      text: async () => '',
    });
    vi.stubGlobal('fetch', fetchMock);

    const result = await useStore.getState().listProviderModels('openai');
    expect(result).toEqual(['gpt-5.1', 'gpt-5']);
    expect(useStore.getState().settings.providers.openai.models).toEqual(['gpt-5.1', 'gpt-5']);
    expect(useStore.getState().settings.providers.openai.loadingModels).toBe(false);
  });
});

/* ------------------- WS attach + dispatch end-to-end ----------------- */

describe('agent WebSocket attach', () => {
  type Listener = (ev: unknown) => void;

  class MockWs {
    static instances: MockWs[] = [];
    url: string;
    listeners: Record<string, Listener[]> = {};
    constructor(url: string) {
      this.url = url;
      MockWs.instances.push(this);
    }
    addEventListener(type: string, fn: Listener): void {
      (this.listeners[type] ??= []).push(fn);
    }
    removeEventListener(type: string, fn: Listener): void {
      this.listeners[type] = (this.listeners[type] ?? []).filter((f) => f !== fn);
    }
    close(): void {
      for (const fn of this.listeners.close ?? []) fn({});
    }
    sendFrame(payload: AgentWsEvent): void {
      for (const fn of this.listeners.message ?? []) fn({ data: JSON.stringify(payload) });
    }
  }

  test('attached socket dispatches incoming frames to the store', () => {
    const factory = (url: string): WebSocket => new MockWs(url) as unknown as WebSocket;
    __attachAgentSocketForTest('th-1', factory);
    expect(useStore.getState().agent.threadId).toBe('th-1');
    const ws = MockWs.instances.at(-1)!;
    expect(ws.url).toContain('/ws/agent/th-1');
    ws.sendFrame({ type: 'text_delta', message_id: 'm1', delta: 'streaming…' });
    expect(useStore.getState().agent.messages[0].text).toBe('streaming…');
  });
});
