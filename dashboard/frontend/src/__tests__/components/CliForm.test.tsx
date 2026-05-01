/**
 * Tests for CliForm.
 *
 * Verifies:
 *   - All three modes render (form / preview / raw).
 *   - Default values populate the assembled command preview.
 *   - Submit posts to /api/cli/run via the store action.
 *   - On success, a job tab opens (`job:<id>`) and the cli form tab closes.
 *   - Required-field validation blocks submission.
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

  test('raw mode renders a textarea with the command', () => {
    useStore.getState().setTweak('cliFormMode', 'raw');
    render(<CliForm cli={cliFixture} />);
    const ta = screen.getByLabelText('raw command') as HTMLTextAreaElement;
    expect(ta.value).toBe('evalyn run-eval');
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
  test('Run posts to /api/cli/run and opens a job tab', async () => {
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

    // The cli form tab should be closed; the job tab is now active.
    const tabs = useStore.getState().tabs;
    expect(tabs.find((t) => t.id === 'cli:run-eval')).toBeFalsy();
  });

  test('blocks submit when required field empty', async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);

    render(<CliForm cli={cliFixture} />);
    fireEvent.click(screen.getByRole('button', { name: /Run/i }));

    expect(fetchMock).not.toHaveBeenCalled();
    expect(await screen.findByRole('alert')).toHaveTextContent(/dataset/);
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
