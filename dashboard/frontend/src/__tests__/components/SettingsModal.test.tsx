/**
 * Tests for SettingsModal.
 *
 * Covers:
 *   - Modal hidden by default; opens via store.openSettings.
 *   - Lists the three default providers (OpenAI / Anthropic / Ollama).
 *   - API key input is `<input type="password">`.
 *   - Save key POSTs the right endpoint.
 *   - Test button POSTs the test endpoint and reflects ok / error.
 *   - Active-provider radio fires setActiveProvider.
 *   - Model dropdown loads on demand from /api/settings/models/{provider}.
 *   - Escape key closes the modal.
 *   - Clicking the dim overlay closes the modal.
 */

import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import SettingsModal from '../../components/SettingsModal';
import { __resetStore, useStore } from '../../store';

const okJson = (body: unknown) => ({
  ok: true,
  status: 200,
  statusText: 'OK',
  json: async () => body,
  text: async () => '',
});

beforeEach(() => {
  __resetStore();
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('SettingsModal - visibility', () => {
  test('hidden by default', () => {
    render(<SettingsModal />);
    expect(screen.queryByTestId('settings-modal')).not.toBeInTheDocument();
  });

  test('opens when settingsOpen is true', () => {
    // Block /api/settings probe so the test stays deterministic.
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('no network')));
    useStore.getState().openSettings();
    render(<SettingsModal />);
    expect(screen.getByTestId('settings-modal')).toBeInTheDocument();
    expect(screen.getByText('Settings')).toBeInTheDocument();
  });
});

describe('SettingsModal - provider list', () => {
  test('renders the three default providers', () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('no network')));
    useStore.getState().openSettings();
    render(<SettingsModal />);
    expect(screen.getByTestId('provider-row-openai')).toBeInTheDocument();
    expect(screen.getByTestId('provider-row-anthropic')).toBeInTheDocument();
    expect(screen.getByTestId('provider-row-ollama')).toBeInTheDocument();
  });

  test('expanding a provider reveals the API key input as type=password', () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('no network')));
    useStore.getState().openSettings();
    render(<SettingsModal />);
    fireEvent.click(screen.getByTestId('provider-row-openai').querySelector('button')!);
    const keyInput = screen.getByLabelText('OpenAI API key') as HTMLInputElement;
    expect(keyInput.type).toBe('password');
  });

  test('Ollama row hides the API key input (requiresKey=false)', () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('no network')));
    useStore.getState().openSettings();
    render(<SettingsModal />);
    fireEvent.click(screen.getByTestId('provider-row-ollama').querySelector('button')!);
    expect(screen.queryByLabelText(/Ollama \(local\) API key/)).not.toBeInTheDocument();
  });
});

describe('SettingsModal - save api key', () => {
  test('Save POSTs /api/settings/{provider}', async () => {
    const fetchMock = vi
      .fn()
      // First call: loadSettings on open. Reject to skip the merge.
      .mockRejectedValueOnce(new Error('no network'))
      // Second call: saveProvider.
      .mockResolvedValueOnce(
        okJson({
          id: 'openai',
          name: 'OpenAI',
          hasKey: true,
          requiresKey: true,
          model: null,
          testStatus: 'untested',
        }),
      );
    vi.stubGlobal('fetch', fetchMock);

    useStore.getState().openSettings();
    render(<SettingsModal />);
    fireEvent.click(screen.getByTestId('provider-row-openai').querySelector('button')!);
    const keyInput = screen.getByLabelText('OpenAI API key');
    fireEvent.change(keyInput, { target: { value: 'sk-test' } });
    fireEvent.click(screen.getByRole('button', { name: 'Save' }));

    await waitFor(() => {
      expect(useStore.getState().settings.providers.openai.hasKey).toBe(true);
    });

    const saveCall = fetchMock.mock.calls.find(
      (c) => c[0] === '/api/settings/openai',
    );
    expect(saveCall).toBeTruthy();
    expect(JSON.parse(saveCall![1].body)).toEqual({ api_key: 'sk-test' });
  });
});

describe('SettingsModal - test connection', () => {
  test('Test button records ok status', async () => {
    const fetchMock = vi
      .fn()
      .mockRejectedValueOnce(new Error('no network')) // loadSettings on open
      .mockResolvedValueOnce(okJson({ ok: true })); // testProvider
    vi.stubGlobal('fetch', fetchMock);

    useStore.getState().openSettings();
    render(<SettingsModal />);
    fireEvent.click(screen.getByTestId('provider-row-openai').querySelector('button')!);
    fireEvent.click(screen.getByRole('button', { name: 'Test' }));

    await waitFor(() => {
      expect(useStore.getState().settings.providers.openai.testStatus).toBe('ok');
    });

    const testCall = fetchMock.mock.calls.find(
      (c) => c[0] === '/api/settings/test/openai',
    );
    expect(testCall).toBeTruthy();
  });

  test('Test failure renders the error message', async () => {
    const fetchMock = vi
      .fn()
      .mockRejectedValueOnce(new Error('no network'))
      .mockResolvedValueOnce(okJson({ ok: false, error: 'Invalid API key' }));
    vi.stubGlobal('fetch', fetchMock);

    useStore.getState().openSettings();
    render(<SettingsModal />);
    fireEvent.click(screen.getByTestId('provider-row-openai').querySelector('button')!);
    fireEvent.click(screen.getByRole('button', { name: 'Test' }));

    await waitFor(() => {
      expect(screen.getByRole('alert')).toHaveTextContent('Invalid API key');
    });
  });
});

describe('SettingsModal - active provider radio', () => {
  test('selecting a radio fires setActiveProvider', async () => {
    const fetchMock = vi
      .fn()
      .mockRejectedValueOnce(new Error('no network')) // loadSettings
      .mockResolvedValueOnce(okJson({ ok: true })); // setActiveProvider
    vi.stubGlobal('fetch', fetchMock);

    useStore.getState().openSettings();
    render(<SettingsModal />);
    const radio = screen.getByLabelText('Set Anthropic active');
    fireEvent.click(radio);

    await waitFor(() => {
      expect(useStore.getState().settings.active).toBe('anthropic');
    });
    const activeCall = fetchMock.mock.calls.find((c) => c[0] === '/api/settings/active');
    expect(activeCall).toBeTruthy();
  });
});

describe('SettingsModal - model list', () => {
  test('expanding loads models for providers with a key set', async () => {
    // Seed openai as having a key so the lazy fetch fires.
    useStore.setState((s) => ({
      settings: {
        ...s.settings,
        providers: {
          ...s.settings.providers,
          openai: { ...s.settings.providers.openai, hasKey: true },
        },
      },
    }));
    const fetchMock = vi
      .fn()
      .mockRejectedValueOnce(new Error('no network')) // loadSettings
      .mockResolvedValueOnce(okJson({ models: ['gpt-5.1', 'gpt-5'] })); // listModels
    vi.stubGlobal('fetch', fetchMock);

    useStore.getState().openSettings();
    render(<SettingsModal />);
    fireEvent.click(screen.getByTestId('provider-row-openai').querySelector('button')!);

    await waitFor(() => {
      const select = screen.getByLabelText('OpenAI model') as HTMLSelectElement;
      const opts = Array.from(select.options).map((o) => o.value);
      expect(opts).toContain('gpt-5.1');
    });
  });
});

describe('SettingsModal - dismiss', () => {
  test('Escape key closes the modal', () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('no network')));
    useStore.getState().openSettings();
    render(<SettingsModal />);
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(useStore.getState().settingsOpen).toBe(false);
  });

  test('clicking the dim overlay closes the modal', () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('no network')));
    useStore.getState().openSettings();
    render(<SettingsModal />);
    fireEvent.click(screen.getByTestId('settings-modal'));
    expect(useStore.getState().settingsOpen).toBe(false);
  });

  test('close button (×) closes the modal', () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('no network')));
    useStore.getState().openSettings();
    render(<SettingsModal />);
    fireEvent.click(screen.getByLabelText('Close settings'));
    expect(useStore.getState().settingsOpen).toBe(false);
  });
});
