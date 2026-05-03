/**
 * Smoke tests for InlineChat (Cmd+I popover).
 *
 * Covers:
 *   - Closed by default (renders nothing).
 *   - Opens via the openInlineChat action.
 *   - Esc closes the popover.
 *   - Send pipes the message into sendChatMessage and shows the dock-right
 *     chat panel.
 */

import { beforeEach, describe, expect, test, vi } from 'vitest';
import { act, fireEvent, render, screen } from '@testing-library/react';
import InlineChat from '../../components/InlineChat';
import { __resetStore, useStore } from '../../store';

beforeEach(() => {
  __resetStore();
});

describe('InlineChat', () => {
  test('closed by default — renders nothing', () => {
    render(<InlineChat />);
    expect(screen.queryByTestId('inline-chat')).toBeNull();
  });

  test('opens via openInlineChat action', () => {
    render(<InlineChat />);
    act(() => useStore.getState().openInlineChat({ x: 200, y: 200 }));
    expect(screen.getByTestId('inline-chat')).toBeInTheDocument();
    // Textarea autofocus is deferred via setTimeout(0); we don't assert on
    // document.activeElement to avoid flakiness in jsdom — presence is enough.
    expect(screen.getByTestId('inline-chat-input')).toBeInTheDocument();
  });

  test('Esc key closes the popover', () => {
    render(<InlineChat />);
    act(() => useStore.getState().openInlineChat({ x: 100, y: 100 }));
    expect(useStore.getState().inlineChatOpen).toBe(true);
    fireEvent.keyDown(screen.getByTestId('inline-chat'), { key: 'Escape' });
    expect(useStore.getState().inlineChatOpen).toBe(false);
    expect(screen.queryByTestId('inline-chat')).toBeNull();
  });

  test('Send calls sendChatMessage and shows the chat panel', () => {
    const sendChatMessage = vi.fn().mockResolvedValue(undefined);
    const setChatVisible = vi.fn();
    useStore.setState({ sendChatMessage, setChatVisible, chatVisible: false });
    render(<InlineChat />);
    act(() => useStore.getState().openInlineChat({ x: 100, y: 100 }));

    const textarea = screen.getByTestId('inline-chat-input');
    fireEvent.change(textarea, { target: { value: 'why did this regress' } });
    fireEvent.click(screen.getByTestId('inline-chat-send'));

    expect(sendChatMessage).toHaveBeenCalledWith('why did this regress');
    expect(setChatVisible).toHaveBeenCalledWith(true);
    expect(useStore.getState().inlineChatOpen).toBe(false);
  });

  test('Send is disabled for empty input', () => {
    const sendChatMessage = vi.fn();
    useStore.setState({ sendChatMessage });
    render(<InlineChat />);
    act(() => useStore.getState().openInlineChat(null));
    const sendBtn = screen.getByTestId('inline-chat-send') as HTMLButtonElement;
    expect(sendBtn.disabled).toBe(true);
    fireEvent.click(sendBtn);
    expect(sendChatMessage).not.toHaveBeenCalled();
  });
});
