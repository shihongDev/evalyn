/**
 * Tests for ChatPanel.
 *
 * Covers:
 *   - Empty state hint renders.
 *   - Each message shape renders the right card:
 *       text bubble · tool-call card · confirmation card · suggestion card
 *   - Confirmation buttons call confirmAgent(true|false).
 *   - Suggestion click opens a CliForm tab via openCli.
 *   - Provider error banner appears with link to SettingsModal.
 *   - Sending a message via the composer triggers sendChatMessage.
 *   - Settings gear in the header triggers openSettings.
 */

import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import ChatPanel from '../../components/ChatPanel';
import { __resetStore, useStore } from '../../store';
import type { AgentState, ChatMessage } from '../../types/agent';

const setAgent = (patch: Partial<AgentState>) => {
  useStore.setState((s) => ({ agent: { ...s.agent, ...patch } }));
};

beforeEach(() => {
  __resetStore();
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('ChatPanel - empty state', () => {
  test('renders the dock-right header', () => {
    render(<ChatPanel />);
    // Header has both "Ask" text and an emphasised "agent" inline; the
    // testid keeps this assertion future-proof against copy changes.
    expect(screen.getByTestId('chat-panel')).toBeInTheDocument();
    expect(screen.getByLabelText('Ask the agent')).toBeInTheDocument();
    expect(screen.getByLabelText('Agent settings')).toBeInTheDocument();
  });

  test('shows hint when message list is empty', () => {
    render(<ChatPanel />);
    expect(screen.getByText(/Hi\. I can read your traces/)).toBeInTheDocument();
  });
});

describe('ChatPanel - rendering messages', () => {
  test('renders user + assistant text bubbles', () => {
    const messages: ChatMessage[] = [
      { id: 'u1', role: 'user', text: 'Why did gemini regress?' },
      { id: 'a1', role: 'assistant', text: 'Looking at runs **82dddcc3**.' },
    ];
    setAgent({ messages });
    render(<ChatPanel />);
    expect(screen.getByText(/Why did gemini regress/)).toBeInTheDocument();
    expect(screen.getByText(/Looking at runs/)).toBeInTheDocument();
    // Markdownish bold renders the run id as <b>.
    expect(screen.getByText('82dddcc3').tagName).toBe('B');
  });

  test('renders an inline tool-call card with status pill', () => {
    const messages: ChatMessage[] = [
      {
        id: 'tc-1',
        role: 'assistant',
        toolCall: {
          id: 't1',
          tool: 'list-runs',
          args: {},
          previewCmd: 'evalyn list-runs',
          status: 'running',
        },
      },
    ];
    setAgent({ messages });
    render(<ChatPanel />);
    expect(screen.getByTestId('tool-call-t1')).toBeInTheDocument();
    expect(screen.getByText('running...')).toBeInTheDocument();
  });

  test('renders captured stdout for a complete tool call', () => {
    const messages: ChatMessage[] = [
      {
        id: 'tc-1',
        role: 'assistant',
        toolCall: {
          id: 't1',
          tool: 'list-runs',
          args: {},
          previewCmd: 'evalyn list-runs',
          status: 'complete',
          output: '3 runs found',
        },
      },
    ];
    setAgent({ messages });
    render(<ChatPanel />);
    expect(screen.getByText('3 runs found')).toBeInTheDocument();
  });
});

describe('ChatPanel - confirmation card', () => {
  test('shows approve / reject buttons when status is awaiting_confirmation', () => {
    const messages: ChatMessage[] = [
      {
        id: 'tc-1',
        role: 'assistant',
        toolCall: {
          id: 't1',
          tool: 'delete-run',
          args: { id: 'abc' },
          previewCmd: 'evalyn delete-run abc',
          status: 'awaiting_confirmation',
        },
      },
    ];
    setAgent({
      messages,
      pendingConfirmation: {
        toolCallId: 't1',
        tool: 'delete-run',
        args: { id: 'abc' },
        previewCmd: 'evalyn delete-run abc',
        sideEffects: ['Permanently delete the run', 'Cannot be undone'],
      },
      status: 'awaiting_confirmation',
    });
    render(<ChatPanel />);
    expect(screen.getByTestId('approve-t1')).toBeInTheDocument();
    expect(screen.getByTestId('reject-t1')).toBeInTheDocument();
    expect(screen.getByTestId('approve-session-t1')).toBeInTheDocument();
    // The redesigned card surfaces the COMMAND + THIS WILL bullets.
    expect(screen.getByText('COMMAND')).toBeInTheDocument();
    expect(screen.getByText('THIS WILL')).toBeInTheDocument();
    expect(screen.getByText('Permanently delete the run')).toBeInTheDocument();
  });

  test('falls back to default side-effect copy when none supplied', () => {
    const messages: ChatMessage[] = [
      {
        id: 'tc-1',
        role: 'assistant',
        toolCall: {
          id: 't1',
          tool: 'unknown-tool',
          args: {},
          previewCmd: 'evalyn unknown-tool',
          status: 'awaiting_confirmation',
        },
      },
    ];
    setAgent({
      messages,
      pendingConfirmation: {
        toolCallId: 't1',
        tool: 'unknown-tool',
        args: {},
        previewCmd: 'evalyn unknown-tool',
      },
      status: 'awaiting_confirmation',
    });
    render(<ChatPanel />);
    expect(screen.getByTestId('side-effects-t1')).toHaveTextContent(
      /This is a write command/,
    );
  });

  test('approve button posts to /api/agent/chat/{id}/confirm', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      statusText: 'OK',
      json: async () => ({ ok: true }),
      text: async () => '',
    });
    vi.stubGlobal('fetch', fetchMock);

    setAgent({
      threadId: 'th-1',
      messages: [
        {
          id: 'tc-1',
          role: 'assistant',
          toolCall: {
            id: 't1',
            tool: 'delete-run',
            args: { id: 'abc' },
            previewCmd: 'evalyn delete-run abc',
            status: 'awaiting_confirmation',
          },
        },
      ],
      pendingConfirmation: {
        toolCallId: 't1',
        tool: 'delete-run',
        args: { id: 'abc' },
        previewCmd: 'evalyn delete-run abc',
      },
      status: 'awaiting_confirmation',
    });

    render(<ChatPanel />);
    fireEvent.click(screen.getByTestId('approve-t1'));
    // Wait a microtask for the fetch to land.
    await Promise.resolve();
    await Promise.resolve();
    expect(fetchMock).toHaveBeenCalledOnce();
    expect(fetchMock.mock.calls[0][0]).toBe('/api/agent/chat/th-1/confirm');
    const body = JSON.parse(fetchMock.mock.calls[0][1].body);
    expect(body.approve).toBe(true);
    expect(body.tool_call_id).toBe('t1');
    // Vanilla "Approve once" must NOT send the override fields.
    expect(body.args_override).toBeUndefined();
    expect(body.auto_approve_session).toBeUndefined();
  });

  test('edit args + Save + Approve sends args_override to the server', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      statusText: 'OK',
      json: async () => ({ ok: true }),
      text: async () => '',
    });
    vi.stubGlobal('fetch', fetchMock);

    setAgent({
      threadId: 'th-1',
      messages: [
        {
          id: 'tc-1',
          role: 'assistant',
          toolCall: {
            id: 't1',
            tool: 'run-eval',
            args: { dataset: 'wrong.jsonl', workers: 4 },
            previewCmd: 'evalyn run-eval --dataset wrong.jsonl --workers 4',
            status: 'awaiting_confirmation',
          },
        },
      ],
      pendingConfirmation: {
        toolCallId: 't1',
        tool: 'run-eval',
        args: { dataset: 'wrong.jsonl', workers: 4 },
        previewCmd: 'evalyn run-eval --dataset wrong.jsonl --workers 4',
      },
      status: 'awaiting_confirmation',
    });

    render(<ChatPanel />);
    fireEvent.click(screen.getByTestId('edit-args-t1'));
    const input = screen.getByTestId('args-editor-input-t1-dataset') as HTMLInputElement;
    fireEvent.change(input, { target: { value: 'evals/correct.jsonl' } });
    fireEvent.click(screen.getByTestId('args-editor-save-t1'));
    // After save, the chip shows we have edits.
    expect(screen.getByTestId('args-edited-chip-t1')).toBeInTheDocument();
    fireEvent.click(screen.getByTestId('approve-t1'));
    await Promise.resolve();
    await Promise.resolve();

    expect(fetchMock).toHaveBeenCalledOnce();
    const body = JSON.parse(fetchMock.mock.calls[0][1].body);
    expect(body.approve).toBe(true);
    expect(body.tool_call_id).toBe('t1');
    expect(body.args_override).toEqual({ dataset: 'evals/correct.jsonl', workers: 4 });
  });

  test('Approve · session sends auto_approve_session=true', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      statusText: 'OK',
      json: async () => ({ ok: true }),
      text: async () => '',
    });
    vi.stubGlobal('fetch', fetchMock);

    setAgent({
      threadId: 'th-1',
      messages: [
        {
          id: 'tc-1',
          role: 'assistant',
          toolCall: {
            id: 't1',
            tool: 'run-eval',
            args: { dataset: 'd.jsonl' },
            previewCmd: 'evalyn run-eval --dataset d.jsonl',
            status: 'awaiting_confirmation',
          },
        },
      ],
      pendingConfirmation: {
        toolCallId: 't1',
        tool: 'run-eval',
        args: { dataset: 'd.jsonl' },
        previewCmd: 'evalyn run-eval --dataset d.jsonl',
      },
      status: 'awaiting_confirmation',
    });

    render(<ChatPanel />);
    fireEvent.click(screen.getByTestId('approve-session-t1'));
    await Promise.resolve();
    await Promise.resolve();

    expect(fetchMock).toHaveBeenCalledOnce();
    const body = JSON.parse(fetchMock.mock.calls[0][1].body);
    expect(body.approve).toBe(true);
    expect(body.tool_call_id).toBe('t1');
    expect(body.auto_approve_session).toBe(true);
    expect(body.args_override).toBeUndefined();
  });
});

describe('ChatPanel - suggestion cards', () => {
  test('clicking a suggestion opens a cli: tab', () => {
    const messages: ChatMessage[] = [
      {
        id: 'a1',
        role: 'assistant',
        text: 'Want me to:',
        finalSuggestions: [
          { label: 'Annotate failures', cliId: 'annotate', args: { limit: 20 } },
        ],
      },
    ];
    setAgent({ messages });
    render(<ChatPanel />);
    expect(screen.getByText('Annotate failures')).toBeInTheDocument();
    fireEvent.click(screen.getByTestId('suggestion-annotate'));
    const tabs = useStore.getState().tabs;
    expect(tabs.find((t) => t.id === 'cli:annotate')).toBeTruthy();
  });

  test('suggestion click seeds the form via activeFormSeed (openCli consumes the prefill)', () => {
    const messages: ChatMessage[] = [
      {
        id: 'a1',
        role: 'assistant',
        finalSuggestions: [
          { label: 'Calibrate', cliId: 'calibrate', args: { iterations: 8 } },
        ],
      },
    ];
    setAgent({ messages });
    render(<ChatPanel />);
    fireEvent.click(screen.getByTestId('suggestion-calibrate'));
    // ChatPanel writes args to sessionStorage; openCli reads + clears them
    // and routes them through activeFormSeed so the form picks them up.
    const seed = useStore.getState().activeFormSeed;
    expect(seed).toEqual({ cliId: 'calibrate', args: { iterations: 8 } });
    expect(useStore.getState().activeCliId).toBe('calibrate');
    // sessionStorage was consumed by openCli — should now be empty.
    expect(sessionStorage.getItem('cli:prefill:calibrate')).toBeNull();
  });
});

describe('ChatPanel - error banner', () => {
  test('hidden when no error', () => {
    render(<ChatPanel />);
    expect(screen.queryByTestId('agent-error-banner')).not.toBeInTheDocument();
  });

  test('shows error banner with message + provider', () => {
    setAgent({
      error: { kind: 'auth', message: 'Invalid API key', provider: 'openai' },
      status: 'error',
    });
    render(<ChatPanel />);
    const banner = screen.getByTestId('agent-error-banner');
    expect(banner).toHaveTextContent('Invalid API key');
    expect(banner).toHaveTextContent('openai');
  });

  test('error banner Open settings button calls openSettings', () => {
    setAgent({
      error: { kind: 'auth', message: 'Invalid API key', provider: 'openai' },
      status: 'error',
    });
    render(<ChatPanel />);
    expect(useStore.getState().settingsOpen).toBe(false);
    fireEvent.click(screen.getByText(/Open settings/));
    expect(useStore.getState().settingsOpen).toBe(true);
  });
});

describe('ChatPanel - composer', () => {
  test('Enter sends a message via sendChatMessage', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      statusText: 'OK',
      json: async () => ({ thread_id: 'th-1' }),
      text: async () => '',
    });
    vi.stubGlobal('fetch', fetchMock);
    class FakeWs {
      addEventListener(): void {}
      removeEventListener(): void {}
      close(): void {}
    }
    vi.stubGlobal('WebSocket', FakeWs);

    render(<ChatPanel />);
    const ta = screen.getByLabelText('Ask the agent') as HTMLTextAreaElement;
    fireEvent.change(ta, { target: { value: 'list my recent runs' } });
    fireEvent.keyDown(ta, { key: 'Enter' });
    await Promise.resolve();
    await Promise.resolve();
    expect(fetchMock).toHaveBeenCalled();
    expect(fetchMock.mock.calls[0][0]).toBe('/api/agent/chat');
  });

  test('settings gear in header opens the modal', () => {
    render(<ChatPanel />);
    fireEvent.click(screen.getByLabelText('Agent settings'));
    expect(useStore.getState().settingsOpen).toBe(true);
  });

  test('close button hides the chat', () => {
    render(<ChatPanel />);
    fireEvent.click(screen.getByLabelText('Hide chat'));
    expect(useStore.getState().chatVisible).toBe(false);
  });
});

describe('ChatPanel - P1 chat surface upgrades', () => {
  test('completed tool call is collapsed by default and shows a peek', () => {
    const messages: ChatMessage[] = [
      {
        id: 'tc-1',
        role: 'assistant',
        toolCall: {
          id: 't1',
          tool: 'list-runs',
          args: {},
          previewCmd: 'evalyn list-runs',
          status: 'complete',
          output: 'line1\nline2\nline3',
        },
      },
    ];
    setAgent({ messages });
    render(<ChatPanel />);
    // Peek is rendered (collapsed default for `complete`).
    expect(screen.getByTestId('tool-call-peek-t1')).toBeInTheDocument();
    // Clicking the header expands.
    fireEvent.click(screen.getByTestId('tool-call-header-t1'));
    expect(screen.queryByTestId('tool-call-peek-t1')).not.toBeInTheDocument();
  });

  test('errored tool call auto-expands and cannot be collapsed', () => {
    const messages: ChatMessage[] = [
      {
        id: 'tc-1',
        role: 'assistant',
        toolCall: {
          id: 't1',
          tool: 'run-eval',
          args: {},
          previewCmd: 'evalyn run-eval',
          status: 'error',
          output: 'first\nsecond',
          error: 'subprocess exited 1',
        },
      },
    ];
    setAgent({ messages });
    render(<ChatPanel />);
    // No peek means already expanded.
    expect(screen.queryByTestId('tool-call-peek-t1')).not.toBeInTheDocument();
    // Click header — still no peek (collapse is locked).
    fireEvent.click(screen.getByTestId('tool-call-header-t1'));
    expect(screen.queryByTestId('tool-call-peek-t1')).not.toBeInTheDocument();
  });

  test('streaming caret shows when message.streaming is true', () => {
    const messages: ChatMessage[] = [
      { id: 'a1', role: 'assistant', text: 'thinking', streaming: true },
    ];
    setAgent({ messages });
    render(<ChatPanel />);
    expect(screen.getByTestId('streaming-caret')).toBeInTheDocument();
  });

  test('cold-start empty state renders three suggestion chips', () => {
    render(<ChatPanel />);
    expect(screen.getByTestId('chip-Walk me through evalyn')).toBeInTheDocument();
    expect(screen.getByTestId('chip-Show me what evalyn can do')).toBeInTheDocument();
    expect(screen.getByTestId('chip-Help me instrument my agent')).toBeInTheDocument();
  });

  test('API key banner appears when no provider is active and dismisses on click', () => {
    render(<ChatPanel />);
    const banner = screen.getByTestId('api-key-banner');
    expect(banner).toBeInTheDocument();
    fireEvent.click(screen.getByTestId('api-key-banner-dismiss'));
    expect(screen.queryByTestId('api-key-banner')).not.toBeInTheDocument();
    expect(useStore.getState().chatBannerDismissed).toBe(true);
  });

  test('API key banner [Add key] opens settings', () => {
    render(<ChatPanel />);
    fireEvent.click(screen.getByTestId('api-key-banner-add'));
    expect(useStore.getState().settingsOpen).toBe(true);
  });

  test('API key banner is hidden once a provider is active', () => {
    useStore.setState((s) => ({
      settings: { ...s.settings, active: 'openai' },
    }));
    render(<ChatPanel />);
    expect(screen.queryByTestId('api-key-banner')).not.toBeInTheDocument();
  });
});
