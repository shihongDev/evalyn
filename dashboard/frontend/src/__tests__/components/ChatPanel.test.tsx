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
    expect(screen.getByText(/Ask anything about your evals/)).toBeInTheDocument();
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
    expect(screen.getByText('running…')).toBeInTheDocument();
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
      },
      status: 'awaiting_confirmation',
    });
    render(<ChatPanel />);
    expect(screen.getByTestId('approve-t1')).toBeInTheDocument();
    expect(screen.getByTestId('reject-t1')).toBeInTheDocument();
    expect(screen.getByText(/Approve to run/i)).toBeInTheDocument();
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

  test('suggestion click stashes pre-fill args in sessionStorage', () => {
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
    const stored = sessionStorage.getItem('cli:prefill:calibrate');
    expect(stored).toBe(JSON.stringify({ iterations: 8 }));
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
