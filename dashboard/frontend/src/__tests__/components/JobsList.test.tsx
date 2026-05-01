/**
 * Tests for the JobsList component.
 *
 * Covers:
 *   - empty-state hint
 *   - renders one row per job
 *   - status pill text + class for each state
 *   - command preview + truncation
 *   - elapsed time uses `duration` for finished, formatted ms for running
 *   - Cancel button only on running rows; calls api.cancelJob and
 *     optimistically marks the job cancelled
 *   - Open button calls store.addTab with `job:<id>` id
 */

import { describe, expect, test, beforeEach, vi, afterEach } from 'vitest';
import { fireEvent, render, screen, act } from '@testing-library/react';
import JobsList from '../../components/JobsList';
import { useStore, __resetStore } from '../../store';
import * as apiModule from '../../api';
import type { Job } from '../../types/jobs';

beforeEach(() => {
  __resetStore();
});

afterEach(() => {
  vi.restoreAllMocks();
});

const addJob = (job: Job) => useStore.getState().upsertJob(job);

describe('JobsList', () => {
  test('renders empty-state hint when no jobs', () => {
    render(<JobsList />);
    expect(screen.getByText(/no jobs yet/i)).toBeInTheDocument();
  });

  test('renders one row per job in the store', () => {
    addJob({ id: 'j-1', cmd: 'evalyn list-runs', status: 'running', startedAt: Date.now() });
    addJob({ id: 'j-2', cmd: 'evalyn analyze', status: 'complete', duration: '0:08' });
    render(<JobsList />);
    expect(screen.getByTestId('job-row-j-1')).toBeInTheDocument();
    expect(screen.getByTestId('job-row-j-2')).toBeInTheDocument();
  });

  test('shows correct status label for each state', () => {
    addJob({ id: 'r1', cmd: 'evalyn run-eval', status: 'running', startedAt: Date.now() });
    addJob({ id: 'c1', cmd: 'evalyn analyze', status: 'complete' });
    addJob({ id: 'f1', cmd: 'evalyn calibrate', status: 'failed' });
    addJob({ id: 'x1', cmd: 'evalyn report', status: 'cancelled' });
    render(<JobsList />);
    expect(screen.getByText('▸ running')).toBeInTheDocument();
    expect(screen.getByText('✓ complete')).toBeInTheDocument();
    expect(screen.getByText('✗ failed')).toBeInTheDocument();
    expect(screen.getByText('⊘ cancelled')).toBeInTheDocument();
  });

  test('command preview shows accent head + tail', () => {
    addJob({ id: 'j-pre', cmd: 'evalyn run-eval --dataset ./data.jsonl --workers 8', status: 'running', startedAt: Date.now() });
    render(<JobsList />);
    expect(screen.getByText('evalyn run-eval')).toBeInTheDocument();
    expect(screen.getByText(/--dataset/)).toBeInTheDocument();
  });

  test('elapsed time formats from startedAt for running jobs', () => {
    const now = Date.now();
    vi.spyOn(Date, 'now').mockReturnValue(now);
    addJob({ id: 'j-run', cmd: 'evalyn run-eval', status: 'running', startedAt: now - 65_000 });
    render(<JobsList />);
    // 65 seconds -> 1:05
    expect(screen.getByText('1:05')).toBeInTheDocument();
  });

  test('elapsed shows the stored duration once complete', () => {
    addJob({ id: 'j-done', cmd: 'evalyn analyze', status: 'complete', duration: '0:08' });
    render(<JobsList />);
    expect(screen.getByText('0:08')).toBeInTheDocument();
  });

  test('Cancel button only renders for running rows', () => {
    addJob({ id: 'r1', cmd: 'evalyn run-eval', status: 'running', startedAt: Date.now() });
    addJob({ id: 'c1', cmd: 'evalyn analyze', status: 'complete' });
    render(<JobsList />);
    expect(screen.getByLabelText('Cancel job r1')).toBeInTheDocument();
    expect(screen.queryByLabelText('Cancel job c1')).toBeNull();
  });

  test('Cancel button calls api.cancelJob and marks job cancelled', async () => {
    addJob({ id: 'r1', cmd: 'evalyn run-eval', status: 'running', startedAt: Date.now() });
    const spy = vi.spyOn(apiModule.api, 'cancelJob').mockResolvedValue({ ok: true });
    render(<JobsList />);
    await act(async () => {
      fireEvent.click(screen.getByLabelText('Cancel job r1'));
    });
    expect(spy).toHaveBeenCalledWith('r1');
    expect(useStore.getState().jobs.get('r1')?.status).toBe('cancelled');
  });

  test('Open button calls store.addTab with job:<id>', () => {
    addJob({ id: 'r1', cmd: 'evalyn run-eval', status: 'complete' });
    render(<JobsList />);
    fireEvent.click(screen.getByLabelText('Open job r1'));
    const tabs = useStore.getState().tabs;
    expect(tabs).toHaveLength(1);
    expect(tabs[0].id).toBe('job:r1');
    expect(tabs[0].kind).toBe('job');
  });

  test('rows are sorted most-recent-first by startedAt', () => {
    const t = Date.now();
    addJob({ id: 'old', cmd: 'evalyn x', status: 'complete', startedAt: t - 5000 });
    addJob({ id: 'new', cmd: 'evalyn y', status: 'complete', startedAt: t });
    const { container } = render(<JobsList />);
    const rows = container.querySelectorAll('[data-testid^="job-row-"]');
    expect(rows[0].getAttribute('data-testid')).toBe('job-row-new');
    expect(rows[1].getAttribute('data-testid')).toBe('job-row-old');
  });
});
