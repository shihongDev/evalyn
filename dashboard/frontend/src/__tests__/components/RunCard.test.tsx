/**
 * RunCard - promote-rows-to-dataset UI (P2).
 *
 * Verifies:
 * - Failed-rows table renders on expand once `/api/runs/{id}` returns
 *   `metric_results`.
 * - Per-row checkboxes select / deselect; the floating action bar
 *   shows the count and disappears when zero rows are selected.
 * - "Add to dataset" opens the modal pre-filled with a default name.
 * - Submitting the modal POSTs to `/api/promote/run-failures` with the
 *   selected row hashes.
 * - 409 conflict surfaces an inline modal error.
 * - "Cancel" on the modal closes it without firing the API.
 */

import { describe, expect, test, beforeEach, vi, afterEach } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import RunCard from '../../views/RunCard';
import { __resetStore, useStore, type RunRecord } from '../../store';
import type { CliSchema } from '../../types/catalog';

const cliFixture: CliSchema = {
  id: 'run-eval',
  name: 'run-eval',
  group: 'Eval',
  blurb: 'Run eval.',
  params: [{ name: 'dataset', kind: 'path', required: true }],
};

const baseRun: RunRecord = {
  id: 'run-001',
  cliId: 'run-eval',
  args: { dataset: './a.jsonl' },
  jobId: 'j-1',
  startedAt: 1_700_000_000_000,
  pinned: false,
};

const _runDetailFixture = {
  id: 'run-001',
  dataset_name: 'src',
  metric_results: [
    {
      metric_id: 'helpfulness',
      item_id: 'itm-A',
      passed: false,
      details: { reason: 'bad answer A' },
    },
    {
      metric_id: 'helpfulness',
      item_id: 'itm-B',
      passed: false,
      details: { reason: 'bad answer B' },
    },
    {
      metric_id: 'helpfulness',
      item_id: 'itm-C',
      passed: true,
      details: { reason: 'ok' },
    },
  ],
};

const _ok = (body: unknown) => ({
  ok: true,
  status: 200,
  statusText: 'OK',
  json: async () => body,
  text: async () => JSON.stringify(body),
});

const _err = (status: number, statusText: string, body = '') => ({
  ok: false,
  status,
  statusText,
  json: async () => ({}),
  text: async () => body,
});

beforeEach(() => {
  __resetStore();
  useStore.getState().setCatalog([cliFixture]);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('RunCard - promote rows to dataset', () => {
  test('renders failed rows table after expand and lets user select rows', async () => {
    const fetchMock = vi.fn(async (url: string) => {
      if (url === '/api/runs/run-001') return _ok(_runDetailFixture);
      return _err(404, 'Not Found');
    });
    vi.stubGlobal('fetch', fetchMock);

    render(<RunCard run={baseRun} prev={null} defaultExpanded />);

    // The table appears once the fetch resolves.
    await waitFor(() => {
      expect(screen.getByTestId('failed-rows-table')).toBeInTheDocument();
    });

    // Two failed rows visible (passing row excluded).
    expect(screen.getByTestId('failed-row-itm-A')).toBeInTheDocument();
    expect(screen.getByTestId('failed-row-itm-B')).toBeInTheDocument();
    expect(screen.queryByTestId('failed-row-itm-C')).toBeNull();

    // No selection → no action bar.
    expect(screen.queryByTestId('promote-action-bar')).toBeNull();

    // Select one row → action bar appears.
    fireEvent.click(screen.getByLabelText(/Select row itm-A/));
    const bar = await screen.findByTestId('promote-action-bar');
    expect(bar).toHaveTextContent('1 row selected');

    // Select a second; count updates.
    fireEvent.click(screen.getByLabelText(/Select row itm-B/));
    expect(screen.getByTestId('promote-action-bar')).toHaveTextContent(
      '2 rows selected',
    );

    // Cancel clears the selection and hides the bar.
    fireEvent.click(screen.getByTestId('promote-cancel'));
    expect(screen.queryByTestId('promote-action-bar')).toBeNull();
  });

  test('Add to dataset opens modal, submits, and clears selection on success', async () => {
    const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
      if (url === '/api/runs/run-001') return _ok(_runDetailFixture);
      if (url === '/api/promote/run-failures' && init?.method === 'POST') {
        return _ok({
          dataset_path: '/tmp/.evalyn/data/datasets/regressions-x',
          dataset_name: 'regressions-x',
          item_count: 1,
          skipped: [],
        });
      }
      return _err(404, 'Not Found');
    });
    vi.stubGlobal('fetch', fetchMock);

    render(<RunCard run={baseRun} prev={null} defaultExpanded />);
    await waitFor(() => screen.getByTestId('failed-rows-table'));

    fireEvent.click(screen.getByLabelText(/Select row itm-A/));
    fireEvent.click(screen.getByTestId('promote-add'));

    // Modal opened with a default dataset name.
    const nameInput = screen.getByTestId('promote-dataset-name') as HTMLInputElement;
    expect(nameInput.value).toMatch(/^regressions-\d{8}-\d{4}$/);

    // Override the name and submit.
    fireEvent.change(nameInput, { target: { value: 'regressions-x' } });
    fireEvent.click(screen.getByTestId('promote-submit'));

    await waitFor(() => {
      expect(screen.getByTestId('promote-success')).toBeInTheDocument();
    });

    // Verify the POST payload.
    const promoteCall = fetchMock.mock.calls.find(
      (c) => c[0] === '/api/promote/run-failures',
    );
    expect(promoteCall).toBeTruthy();
    const body = JSON.parse(promoteCall![1]!.body as string);
    expect(body.run_id).toBe('run-001');
    expect(body.row_hashes).toEqual(['itm-A']);
    expect(body.dataset_name).toBe('regressions-x');

    // Selection cleared after success.
    expect(screen.queryByTestId('promote-action-bar')).toBeNull();
  });

  test('409 conflict surfaces an inline error and keeps modal open', async () => {
    const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
      if (url === '/api/runs/run-001') return _ok(_runDetailFixture);
      if (url === '/api/promote/run-failures' && init?.method === 'POST') {
        return _err(409, 'Conflict', "dataset 'regressions-x' already exists");
      }
      return _err(404, 'Not Found');
    });
    vi.stubGlobal('fetch', fetchMock);

    render(<RunCard run={baseRun} prev={null} defaultExpanded />);
    await waitFor(() => screen.getByTestId('failed-rows-table'));

    fireEvent.click(screen.getByLabelText(/Select row itm-A/));
    fireEvent.click(screen.getByTestId('promote-add'));
    fireEvent.click(screen.getByTestId('promote-submit'));

    const alert = await screen.findByRole('alert');
    expect(alert).toHaveTextContent(/already exists/i);

    // Modal still open; selection unchanged.
    expect(screen.getByTestId('promote-dataset-name')).toBeInTheDocument();
    expect(screen.getByTestId('promote-action-bar')).toHaveTextContent('1 row');
  });

  test('Cancel button on modal closes it without firing the API', async () => {
    const fetchMock = vi.fn(async (url: string) => {
      if (url === '/api/runs/run-001') return _ok(_runDetailFixture);
      return _err(404, 'Not Found');
    });
    vi.stubGlobal('fetch', fetchMock);

    render(<RunCard run={baseRun} prev={null} defaultExpanded />);
    await waitFor(() => screen.getByTestId('failed-rows-table'));

    fireEvent.click(screen.getByLabelText(/Select row itm-A/));
    fireEvent.click(screen.getByTestId('promote-add'));
    expect(screen.getByTestId('promote-dataset-name')).toBeInTheDocument();

    // Click the modal's Cancel button (not the floating bar's).
    const cancelBtn = screen
      .getAllByRole('button', { name: /Cancel/i })
      .find((b) => !b.dataset.testid || b.dataset.testid !== 'promote-cancel');
    expect(cancelBtn).toBeTruthy();
    fireEvent.click(cancelBtn!);

    // Modal closed; no promote API call recorded.
    expect(screen.queryByTestId('promote-dataset-name')).toBeNull();
    const promoteCall = fetchMock.mock.calls.find(
      (c) => c[0] === '/api/promote/run-failures',
    );
    expect(promoteCall).toBeUndefined();
  });

  test('hides the failed-rows table when /api/runs/{id} 404s', async () => {
    const fetchMock = vi.fn(async () => _err(404, 'Not Found'));
    vi.stubGlobal('fetch', fetchMock);

    render(<RunCard run={baseRun} prev={null} defaultExpanded />);

    // Give the rejected fetch a tick to settle.
    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalled();
    });
    expect(screen.queryByTestId('failed-rows-table')).toBeNull();
    expect(screen.queryByTestId('promote-action-bar')).toBeNull();
  });

  test('blocks submit when dataset name is empty/whitespace', async () => {
    const fetchMock = vi.fn(async (url: string) => {
      if (url === '/api/runs/run-001') return _ok(_runDetailFixture);
      return _err(404, 'Not Found');
    });
    vi.stubGlobal('fetch', fetchMock);

    render(<RunCard run={baseRun} prev={null} defaultExpanded />);
    await waitFor(() => screen.getByTestId('failed-rows-table'));

    fireEvent.click(screen.getByLabelText(/Select row itm-A/));
    fireEvent.click(screen.getByTestId('promote-add'));
    const nameInput = screen.getByTestId('promote-dataset-name') as HTMLInputElement;
    fireEvent.change(nameInput, { target: { value: '   ' } });
    const submit = screen.getByTestId('promote-submit') as HTMLButtonElement;
    expect(submit.disabled).toBe(true);
  });
});
