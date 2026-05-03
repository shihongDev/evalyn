/**
 * Tests for CliForm.
 *
 * Verifies:
 *   - All three modes render (form / preview / raw).
 *   - Default values populate the assembled command preview.
 *   - Submit posts to /api/cli/run via the store action.
 *   - On success, a job tab opens (`job:<id>`); the cli form tab is kept
 *     open so the post-run summary can replace the form body in place.
 *   - Required-field validation paints field-scoped errors and surfaces
 *     a "N fields need attention" anchor (no top banner for required).
 */

import { describe, expect, test, beforeEach, vi, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import CliForm from '../../views/CliForm';
import { __resetStore, useStore } from '../../store';
import type { CliSchema } from '../../types/catalog';

const cliFixture: CliSchema = {
  id: 'run-eval',
  name: 'run-eval',
  group: 'Eval',
  blurb: 'Run an evaluation.',
  params: [
    { name: 'dataset', kind: 'path', required: true },
    { name: 'workers', kind: 'number', default: 4 },
    { name: 'verbose', kind: 'bool', default: false },
  ],
};

beforeEach(() => {
  __resetStore();
  // Pre-open the tab so closeTab(`cli:<id>`) has something to close.
  useStore.getState().openCli('run-eval');
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('CliForm - render', () => {
  test('renders header with cli name + group', () => {
    render(<CliForm cli={cliFixture} />);
    expect(screen.getByRole('heading', { name: 'run-eval' })).toBeInTheDocument();
    expect(screen.getByText(cliFixture.group)).toBeInTheDocument();
  });

  test('renders blurb', () => {
    render(<CliForm cli={cliFixture} />);
    expect(screen.getByText(cliFixture.blurb)).toBeInTheDocument();
  });

  test('default mode is preview (per TWEAK_DEFAULTS)', () => {
    render(<CliForm cli={cliFixture} />);
    // preview mode shows the "live command" label
    expect(screen.getByText('live command')).toBeInTheDocument();
  });
});

describe('CliForm - mode switching', () => {
  test('form mode renders fields without preview pane', () => {
    useStore.getState().setTweak('cliFormMode', 'form');
    render(<CliForm cli={cliFixture} />);
    expect(screen.queryByText('live command')).not.toBeInTheDocument();
    expect(screen.getByLabelText(/dataset/i)).toBeInTheDocument();
  });

  test('raw mode renders the assembled command (read-only) with a copy button', () => {
    useStore.getState().setTweak('cliFormMode', 'raw');
    render(<CliForm cli={cliFixture} />);
    const pre = screen.getByLabelText('raw command');
    expect(pre.tagName).toBe('PRE');
    expect(pre.textContent).toBe('evalyn run-eval');
    expect(screen.getByLabelText('copy raw command')).toBeInTheDocument();
  });

  test('preview command updates as values change', () => {
    render(<CliForm cli={cliFixture} />);
    const ds = screen.getByLabelText(/dataset/i) as HTMLInputElement;
    fireEvent.change(ds, { target: { value: './data.jsonl' } });
    // The bottom-bar code element shows the assembled command.
    expect(
      screen.getAllByText(/evalyn run-eval --dataset \.\/data\.jsonl/)[0],
    ).toBeInTheDocument();
  });
});

describe('CliForm - submit', () => {
  test('Run posts to /api/cli/run and opens a job tab (cli tab stays open)', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      statusText: 'OK',
      json: async () => ({ job_id: 'j-42' }),
      text: async () => '',
    });
    vi.stubGlobal('fetch', fetchMock);

    render(<CliForm cli={cliFixture} />);
    // Fill the required field.
    fireEvent.change(screen.getByLabelText(/dataset/i), {
      target: { value: './data.jsonl' },
    });
    fireEvent.click(screen.getByRole('button', { name: /Run/i }));

    await waitFor(() => {
      const tabs = useStore.getState().tabs;
      expect(tabs.find((t) => t.id === 'job:j-42')).toBeTruthy();
    });

    expect(fetchMock).toHaveBeenCalledOnce();
    const call = fetchMock.mock.calls[0];
    expect(call[0]).toBe('/api/cli/run');
    const body = JSON.parse(call[1].body);
    expect(body.cli_id).toBe('run-eval');
    expect(body.args.dataset).toBe('./data.jsonl');

    // The cli form tab is kept open so the in-place post-run summary can
    // appear once the job completes — the user is no longer hijacked away
    // from the form context.
    const tabs = useStore.getState().tabs;
    expect(tabs.find((t) => t.id === 'cli:run-eval')).toBeTruthy();
  });

  test('blocks submit when required field empty (field-scoped errors)', async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);

    render(<CliForm cli={cliFixture} />);
    fireEvent.click(screen.getByRole('button', { name: /Run/i }));

    expect(fetchMock).not.toHaveBeenCalled();
    // The "N fields need attention" anchor surfaces the count; the
    // detailed errors live on individual ParamFields.
    expect(
      await screen.findByRole('button', { name: /field needs attention/i }),
    ).toBeInTheDocument();
    // Field-scoped error renders on the dataset row.
    expect(screen.getByText('required')).toBeInTheDocument();
  });

  test('shows server error on non-2xx', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: false,
      status: 422,
      statusText: 'Unprocessable Entity',
      json: async () => ({}),
      text: async () => 'invalid args',
    });
    vi.stubGlobal('fetch', fetchMock);

    render(<CliForm cli={cliFixture} />);
    fireEvent.change(screen.getByLabelText(/dataset/i), {
      target: { value: './data.jsonl' },
    });
    fireEvent.click(screen.getByRole('button', { name: /Run/i }));

    expect(await screen.findByRole('alert')).toHaveTextContent(/invalid args/);
    // The cli form tab should still be open since the run failed.
    const tabs = useStore.getState().tabs;
    expect(tabs.find((t) => t.id === 'cli:run-eval')).toBeTruthy();
  });
});

describe('CliForm - cancel', () => {
  test('Cancel closes the cli tab', () => {
    render(<CliForm cli={cliFixture} />);
    fireEvent.click(screen.getByRole('button', { name: /Cancel/i }));
    const tabs = useStore.getState().tabs;
    expect(tabs.find((t) => t.id === 'cli:run-eval')).toBeFalsy();
  });
});

describe('CliForm - recent runs strip', () => {
  test('does not render when no runs exist for this CLI', () => {
    render(<CliForm cli={cliFixture} />);
    // Heading copy is "Recent {cliId} runs" — absent when runHistory empty.
    expect(screen.queryByText(/recent run-eval runs/i)).not.toBeInTheDocument();
  });

  test('renders one card per run for this CLI; click seeds the form via editRunArgs', () => {
    // Inject a run record by calling the store directly.
    useStore.setState((s) => ({
      runHistory: [
        ...s.runHistory,
        {
          id: 'r-prev',
          cliId: 'run-eval',
          args: { dataset: './prev.jsonl', workers: 7 },
          jobId: 'j-prev',
          startedAt: Date.now() - 60_000,
          pinned: false,
        },
      ],
    }));
    const { container } = render(<CliForm cli={cliFixture} />);
    expect(screen.getByText(/recent run-eval runs/i)).toBeInTheDocument();
    const card = container.querySelector('[data-recent-run-id="r-prev"]');
    expect(card).toBeTruthy();
    fireEvent.click(card!);
    // editRunArgs sets the activeFormSeed; the Workspace path consumes it.
    expect(useStore.getState().activeFormSeed).toEqual({
      cliId: 'run-eval',
      args: { dataset: './prev.jsonl', workers: 7 },
    });
  });
});

describe('CliForm - post-run summary', () => {
  test('shows "Run completed" card when the underlying job reaches a terminal state', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      statusText: 'OK',
      json: async () => ({ job_id: 'j-99' }),
      text: async () => '',
    });
    vi.stubGlobal('fetch', fetchMock);

    render(<CliForm cli={cliFixture} />);
    fireEvent.change(screen.getByLabelText(/dataset/i), {
      target: { value: './data.jsonl' },
    });
    fireEvent.click(screen.getByRole('button', { name: /Run/i }));

    // Wait until the run has been registered in the store.
    await waitFor(() => {
      expect(useStore.getState().jobs.get('j-99')).toBeTruthy();
    });

    // Simulate the WS exit event by upserting the job with a terminal state.
    useStore.getState().upsertJob({
      id: 'j-99',
      cmd: 'evalyn run-eval --dataset ./data.jsonl',
      cliId: 'run-eval',
      status: 'complete',
      exitCode: 0,
    });

    expect(await screen.findByText(/run completed/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /open in run viewer/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /re-run with same args/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /new run/i })).toBeInTheDocument();
    // The cli tab is still open — the form context is preserved.
    expect(useStore.getState().tabs.find((t) => t.id === 'cli:run-eval')).toBeTruthy();
  });

  test('"New run" returns the form to defaults', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      statusText: 'OK',
      json: async () => ({ job_id: 'j-100' }),
      text: async () => '',
    });
    vi.stubGlobal('fetch', fetchMock);

    render(<CliForm cli={cliFixture} />);
    fireEvent.change(screen.getByLabelText(/dataset/i), {
      target: { value: './data.jsonl' },
    });
    fireEvent.click(screen.getByRole('button', { name: /Run/i }));
    await waitFor(() => {
      expect(useStore.getState().jobs.get('j-100')).toBeTruthy();
    });
    useStore.getState().upsertJob({
      id: 'j-100',
      cmd: 'evalyn run-eval --dataset ./data.jsonl',
      cliId: 'run-eval',
      status: 'complete',
      exitCode: 0,
    });
    await screen.findByRole('button', { name: /new run/i });
    fireEvent.click(screen.getByRole('button', { name: /new run/i }));
    // Form re-renders with defaults; "Run completed" copy is gone.
    expect(screen.queryByText(/run completed/i)).not.toBeInTheDocument();
    const ds = screen.getByLabelText(/dataset/i) as HTMLInputElement;
    expect(ds.value).toBe('');
  });
});
