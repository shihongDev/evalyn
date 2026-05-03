/**
 * Smoke tests for CommandPalette.
 *
 * Covers:
 *   - Renders when paletteOpen=true; absent otherwise.
 *   - Cmd+K (registered in App.tsx) toggles paletteOpen.
 *   - Typing filters CLI rows.
 *   - Enter activates the highlighted row (openCli for CLI rows).
 *   - Ask-agent row sends the query through sendChatMessage.
 */

import { beforeEach, describe, expect, test, vi } from 'vitest';
import { act, fireEvent, render, screen } from '@testing-library/react';
import { useEffect } from 'react';
import CommandPalette from '../../components/CommandPalette';
import { __resetStore, useStore } from '../../store';
import type { CliSchema } from '../../types/catalog';

const seedCatalog: CliSchema[] = [
  {
    id: 'run-eval',
    name: 'run-eval',
    group: 'Eval',
    blurb: 'Execute an evaluation against a dataset.',
    params: [],
  },
  {
    id: 'analyze',
    name: 'analyze',
    group: 'Iterate',
    blurb: 'Analyze a completed run.',
    params: [],
  },
  {
    id: 'build-dataset',
    name: 'build-dataset',
    group: 'Data',
    blurb: 'Build an eval dataset from traces.',
    params: [],
  },
];

beforeEach(() => {
  __resetStore();
  useStore.setState({ catalog: seedCatalog });
});

/**
 * Test harness that mounts both CommandPalette AND the Cmd+K listener (in
 * isolation). App.tsx owns the real listener; we replicate its behaviour so
 * tests don't have to mount the entire app shell.
 */
const Harness = () => {
  const setPaletteOpen = useStore((s) => s.setPaletteOpen);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && (e.key === 'k' || e.key === 'K')) {
        e.preventDefault();
        const { paletteOpen } = useStore.getState();
        setPaletteOpen(!paletteOpen);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [setPaletteOpen]);
  return <CommandPalette />;
};

describe('CommandPalette', () => {
  test('does not render when paletteOpen=false', () => {
    render(<Harness />);
    expect(screen.queryByTestId('command-palette')).toBeNull();
  });

  test('renders when paletteOpen=true', () => {
    render(<Harness />);
    act(() => useStore.getState().setPaletteOpen(true));
    expect(screen.getByTestId('command-palette')).toBeInTheDocument();
    // Section headers from the catalog.
    expect(screen.getByText('Run command')).toBeInTheDocument();
    expect(screen.getByText('Ask agent')).toBeInTheDocument();
  });

  test('Cmd+K toggles palette open and closed', () => {
    render(<Harness />);
    expect(useStore.getState().paletteOpen).toBe(false);
    fireEvent.keyDown(window, { key: 'k', metaKey: true });
    expect(useStore.getState().paletteOpen).toBe(true);
    fireEvent.keyDown(window, { key: 'k', metaKey: true });
    expect(useStore.getState().paletteOpen).toBe(false);
  });

  test('typing filters rows by id', () => {
    render(<Harness />);
    act(() => useStore.getState().setPaletteOpen(true));
    const input = screen.getByTestId('command-palette-input');
    fireEvent.change(input, { target: { value: 'analyze' } });
    expect(screen.getByTestId('palette-row-cli:analyze')).toBeInTheDocument();
    expect(screen.queryByTestId('palette-row-cli:run-eval')).toBeNull();
    expect(screen.queryByTestId('palette-row-cli:build-dataset')).toBeNull();
  });

  test('Enter on highlighted CLI row calls openCli', () => {
    const openCli = vi.fn();
    useStore.setState({ openCli });
    render(<Harness />);
    act(() => useStore.getState().setPaletteOpen(true));
    const input = screen.getByTestId('command-palette-input');
    fireEvent.change(input, { target: { value: 'analyze' } });
    fireEvent.keyDown(screen.getByTestId('command-palette'), { key: 'Enter' });
    expect(openCli).toHaveBeenCalledWith('analyze');
    expect(useStore.getState().paletteOpen).toBe(false);
  });

  test('Ask agent row triggers sendChatMessage with the query', () => {
    const sendChatMessage = vi.fn().mockResolvedValue(undefined);
    useStore.setState({ sendChatMessage });
    render(<Harness />);
    act(() => useStore.getState().setPaletteOpen(true));
    const input = screen.getByTestId('command-palette-input');
    fireEvent.change(input, { target: { value: 'why did this regress' } });
    // Click the ask row directly (filtering may still leave CLI matches first).
    fireEvent.click(screen.getByTestId('palette-row-ask'));
    expect(sendChatMessage).toHaveBeenCalledWith('why did this regress');
    expect(useStore.getState().paletteOpen).toBe(false);
  });
});
