/**
 * Smoke test for TitleBar.
 */

import { describe, expect, test, beforeEach } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import TitleBar from '../../components/TitleBar';
import { __resetStore, useStore } from '../../store';

beforeEach(() => {
  __resetStore();
});

describe('TitleBar', () => {
  test('renders local-server chip', () => {
    render(<TitleBar />);
    expect(screen.getByText(/Local · 7401/)).toBeInTheDocument();
  });

  test('renders evalyn brand mark', () => {
    render(<TitleBar />);
    expect(screen.getByText('evalyn')).toBeInTheDocument();
  });

  test('shows active tab title in title bar', () => {
    useStore.getState().addTab({ id: 'cli:run-eval', title: 'run-eval', kind: 'cli' });
    render(<TitleBar />);
    expect(screen.getByText('run-eval')).toBeInTheDocument();
  });

  test('clicking Settings opens the settings modal', () => {
    render(<TitleBar />);
    expect(useStore.getState().settingsOpen).toBe(false);
    fireEvent.click(screen.getByText('Settings'));
    expect(useStore.getState().settingsOpen).toBe(true);
  });
});
