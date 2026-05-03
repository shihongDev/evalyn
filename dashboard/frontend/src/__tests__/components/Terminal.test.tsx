/**
 * Tests for the Terminal component.
 *
 * Covers:
 *   - empty-state placeholder when no lines
 *   - renders a single job's lines
 *   - applies kind-fallback colors (e.g. fail = var(--fail))
 *   - parses ANSI SGR within a line
 *   - auto-scrolls to bottom on new lines when pinned
 *   - does NOT auto-scroll when user scrolled up (unpinned)
 */

import { describe, expect, test, beforeEach } from 'vitest';
import { act, render, screen } from '@testing-library/react';
import Terminal from '../../components/Terminal';
import { useStore, __resetStore } from '../../store';
import type { JobLine } from '../../types/jobs';

beforeEach(() => {
  __resetStore();
});

const setJobLines = (jobId: string, lines: JobLine[]) => {
  for (const line of lines) {
    useStore.getState().appendJobLine(jobId, line);
  }
};

describe('Terminal', () => {
  test('shows empty-state hint when no jobs have lines', () => {
    render(<Terminal jobId="j-empty" />);
    expect(screen.getByText(/Output will appear here/i)).toBeInTheDocument();
  });

  test('renders lines from a single job', () => {
    setJobLines('j-1', [
      { kind: 'info', text: 'starting' },
      { kind: 'stdout', text: 'first line' },
      { kind: 'stdout', text: 'second line' },
    ]);
    render(<Terminal jobId="j-1" />);
    expect(screen.getByText('starting')).toBeInTheDocument();
    expect(screen.getByText('first line')).toBeInTheDocument();
    expect(screen.getByText('second line')).toBeInTheDocument();
  });

  test('parses inline ANSI color codes', () => {
    setJobLines('j-2', [{ kind: 'stdout', text: '\x1b[31mERROR\x1b[0m boom' }]);
    render(<Terminal jobId="j-2" />);
    const errorSpan = screen.getByText('ERROR');
    expect(errorSpan).toHaveStyle({ color: 'var(--fail)' });
    expect(screen.getByText((_, el) => el?.textContent === ' boom')).toBeInTheDocument();
  });

  test('uses kind fallback color for stderr lines', () => {
    setJobLines('j-3', [{ kind: 'stderr', text: 'fatal' }]);
    const { container } = render(<Terminal jobId="j-3" />);
    const row = container.querySelector('div[style*="--fail"]');
    expect(row).toBeTruthy();
  });

  test('default mode (no jobId) merges all jobs', () => {
    setJobLines('j-a', [{ kind: 'stdout', text: 'aaa' }]);
    setJobLines('j-b', [{ kind: 'stdout', text: 'bbb' }]);
    render(<Terminal />);
    expect(screen.getByText('aaa')).toBeInTheDocument();
    expect(screen.getByText('bbb')).toBeInTheDocument();
  });

  test('auto-scrolls to bottom when pinned and a new line arrives', () => {
    setJobLines('j-scroll', [{ kind: 'stdout', text: 'line 1' }]);
    render(<Terminal jobId="j-scroll" />);
    const el = screen.getByTestId('terminal-scroll');
    // jsdom doesn't lay out so scrollHeight is 0 by default — we patch
    // the relevant getters to simulate a real scrollable region.
    Object.defineProperty(el, 'scrollHeight', { configurable: true, value: 1000 });
    Object.defineProperty(el, 'clientHeight', { configurable: true, value: 200 });
    el.scrollTop = 0;
    act(() => {
      useStore.getState().appendJobLine('j-scroll', { kind: 'stdout', text: 'line 2' });
    });
    // After the effect fires, the panel should be pinned to the bottom.
    expect(el.scrollTop).toBe(1000);
  });

  test('does NOT auto-scroll when user has scrolled up', async () => {
    setJobLines('j-up', [{ kind: 'stdout', text: 'line 1' }]);
    const { rerender } = render(<Terminal jobId="j-up" />);
    const el = screen.getByTestId('terminal-scroll');
    // Set up a tall scrollable region with the user scrolled to the top.
    Object.defineProperty(el, 'scrollHeight', { configurable: true, value: 1000 });
    Object.defineProperty(el, 'clientHeight', { configurable: true, value: 200 });
    el.scrollTop = 0;
    // Simulate the user scrolling up via the onScroll handler. With
    // distance = 1000 - (0 + 200) = 800 (>>8), the component should
    // mark itself as "not pinned".
    el.dispatchEvent(new Event('scroll', { bubbles: true }));
    rerender(<Terminal jobId="j-up" />);
    expect(el.dataset.pinned).toBe('0');
    el.scrollTop = 50;
    act(() => {
      useStore.getState().appendJobLine('j-up', { kind: 'stdout', text: 'line 2' });
    });
    // Still 50 — the component must not have scrolled to bottom.
    expect(el.scrollTop).toBe(50);
  });
});
