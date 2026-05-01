/**
 * Smoke test for TitleBar.
 *
 * Verifies the component mounts without throwing and exposes the
 * localhost chip + ⌘K palette button.
 */

import { describe, expect, test, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import TitleBar from '../../components/TitleBar';
import { __resetStore, useStore } from '../../store';

beforeEach(() => {
  __resetStore();
});

describe('TitleBar', () => {
  test('renders localhost chip', () => {
    render(<TitleBar />);
    expect(screen.getByText('localhost:7401')).toBeInTheDocument();
  });

  test('renders ⌘K kbd', () => {
    render(<TitleBar />);
    expect(screen.getByText('⌘K')).toBeInTheDocument();
  });

  test('renders evalyn workbench wordmark', () => {
    render(<TitleBar />);
    expect(screen.getByText(/evalyn/i)).toBeInTheDocument();
    expect(screen.getByText(/workbench/i)).toBeInTheDocument();
  });

  test('breadcrumbs reflect active tab', () => {
    useStore.getState().addTab({ id: 'cli:run-eval', title: 'run-eval', kind: 'cli' });
    render(<TitleBar />);
    expect(screen.getByText(/run-eval/)).toBeInTheDocument();
  });
});
