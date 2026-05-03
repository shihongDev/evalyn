/**
 * SettingsModal — provider configuration.
 *
 * Spec §8: provider list (OpenAI/Anthropic/Ollama), API key input
 * (`<input type="password">`), test button, model dropdown populated from
 * `/api/settings/models/{provider}`, and an active-provider radio.
 *
 * Modal pattern follows the same dim-overlay + centered card we use for
 * other workbench modals; visual tokens come from `styles/index.css`.
 *
 * Behavior contract (matches Lane C1 backend):
 *   - API keys are never returned to the client. The server only reports
 *     `hasKey: true | false` per provider.
 *   - Model lists load on demand when a provider section is expanded;
 *     the result is cached on the provider state until the modal closes.
 *   - Test button POSTs `/api/settings/providers/{provider}/test`. Result
 *     becomes a small status pill next to the provider name.
 *   - The active radio enforces single-active; the server is the source
 *     of truth, the frontend reflects.
 */

import { useEffect, useState } from 'react';
import { useStore } from '../store';
import type { ProviderState } from '../types/agent';

const PROVIDER_ORDER = ['openai', 'anthropic', 'ollama'];

const TEST_PILL_TONE: Record<string, string> = {
  ok: 'pass',
  error: 'fail',
  testing: 'warn',
  untested: 'text-3',
};

/** Single provider section. Self-contained because each provider has its
 *  own API key buffer, expansion state, and test status. */
const ProviderRow = ({ provider }: { provider: ProviderState }) => {
  const saveProvider = useStore((s) => s.saveProvider);
  const testProvider = useStore((s) => s.testProvider);
  const setActiveProvider = useStore((s) => s.setActiveProvider);
  const listProviderModels = useStore((s) => s.listProviderModels);
  const active = useStore((s) => s.settings.active);

  const [expanded, setExpanded] = useState(false);
  const [apiKey, setApiKey] = useState('');
  const [savingKey, setSavingKey] = useState(false);

  const isActive = active === provider.id;
  const tone = TEST_PILL_TONE[provider.testStatus] ?? 'text-3';

  // Lazily fetch models the first time the user expands the section.
  useEffect(() => {
    if (!expanded) return;
    if (provider.models != null || provider.loadingModels) return;
    if (provider.requiresKey && !provider.hasKey) return;
    void listProviderModels(provider.id).catch(() => {
      // Silently swallow — the dropdown will show "—" until retried.
    });
  }, [expanded, provider, listProviderModels]);

  const onSaveKey = async () => {
    if (!apiKey.trim()) return;
    setSavingKey(true);
    try {
      await saveProvider(provider.id, { api_key: apiKey });
      setApiKey('');
    } finally {
      setSavingKey(false);
    }
  };

  const onModelChange = async (model: string) => {
    await saveProvider(provider.id, { model });
  };

  const onTest = async () => {
    await testProvider(provider.id);
  };

  const onSetActive = async () => {
    await setActiveProvider(provider.id);
  };

  return (
    <div
      data-testid={`provider-row-${provider.id}`}
      style={{
        border: '1px solid var(--line)',
        borderRadius: 6,
        background: 'var(--bg-1)',
        marginBottom: 10,
        overflow: 'hidden',
      }}
    >
      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 10,
          width: '100%',
          padding: '12px 14px',
          background: 'transparent',
          border: 0,
          color: 'var(--text-1)',
          textAlign: 'left',
        }}
      >
        <input
          type="radio"
          name="active-provider"
          aria-label={`Set ${provider.name} active`}
          checked={isActive}
          onChange={(e) => {
            e.stopPropagation();
            void onSetActive();
          }}
          onClick={(e) => e.stopPropagation()}
        />
        <span style={{ fontFamily: 'var(--serif)', fontSize: 15, color: 'var(--text-0)' }}>
          {provider.name}
        </span>
        {provider.hasKey && (
          <span className="chip pass dot" style={{ fontSize: 10 }}>
            key set
          </span>
        )}
        {!provider.hasKey && provider.requiresKey && (
          <span className="chip" style={{ fontSize: 10, color: 'var(--text-3)' }}>
            no key
          </span>
        )}
        <span className={`chip mono ${tone}`} style={{ fontSize: 10 }}>
          {provider.testStatus}
        </span>
        <span className="grow" />
        <span className="text-3 mono" style={{ fontSize: 11 }}>
          {expanded ? '▾' : '▸'}
        </span>
      </button>

      {expanded && (
        <div
          style={{
            padding: '12px 14px 14px',
            borderTop: '1px solid var(--line)',
            background: 'var(--bg-2)',
            display: 'flex',
            flexDirection: 'column',
            gap: 10,
          }}
        >
          {provider.requiresKey && (
            <label style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              <span className="text-2" style={{ fontSize: 11 }}>
                API key
              </span>
              <div style={{ display: 'flex', gap: 8 }}>
                <input
                  type="password"
                  aria-label={`${provider.name} API key`}
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  placeholder={provider.hasKey ? '•••••••• (set)' : 'paste key'}
                  className="textarea"
                  style={{
                    flex: 1,
                    background: 'var(--bg-1)',
                    border: '1px solid var(--line)',
                    borderRadius: 6,
                    padding: '6px 10px',
                    fontFamily: 'var(--mono)',
                    fontSize: 12,
                  }}
                />
                <button
                  type="button"
                  className="btn primary sm"
                  disabled={!apiKey.trim() || savingKey}
                  onClick={onSaveKey}
                >
                  {savingKey ? 'Saving…' : 'Save'}
                </button>
              </div>
            </label>
          )}

          <div style={{ display: 'flex', gap: 10, alignItems: 'flex-end' }}>
            <label style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 6 }}>
              <span className="text-2" style={{ fontSize: 11 }}>
                Model
              </span>
              <select
                aria-label={`${provider.name} model`}
                value={provider.model ?? ''}
                onChange={(e) => void onModelChange(e.target.value)}
                disabled={provider.loadingModels || (provider.models?.length ?? 0) === 0}
                style={{
                  background: 'var(--bg-1)',
                  border: '1px solid var(--line)',
                  borderRadius: 6,
                  padding: '6px 10px',
                  fontFamily: 'var(--mono)',
                  fontSize: 12,
                  color: 'var(--text-1)',
                }}
              >
                <option value="">
                  {provider.loadingModels
                    ? 'loading…'
                    : provider.models == null
                      ? '—'
                      : provider.models.length === 0
                        ? 'no models available'
                        : 'select a model'}
                </option>
                {provider.models?.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
            </label>

            <button
              type="button"
              className="btn ghost sm"
              onClick={onTest}
              disabled={provider.testStatus === 'testing'}
            >
              {provider.testStatus === 'testing' ? 'Testing…' : 'Test'}
            </button>
          </div>

          {provider.testStatus === 'error' && provider.testError && (
            <div
              role="alert"
              className="mono"
              style={{
                fontSize: 11,
                color: 'var(--fail)',
                background: 'var(--fail-soft)',
                padding: '6px 10px',
                borderRadius: 4,
              }}
            >
              {provider.testError}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

const SettingsModal = () => {
  const open = useStore((s) => s.settingsOpen);
  const closeSettings = useStore((s) => s.closeSettings);
  const providers = useStore((s) => s.settings.providers);
  const loadSettings = useStore((s) => s.loadSettings);

  // Refresh settings each time the modal opens — covers the case where
  // another tab updated keys.
  useEffect(() => {
    if (!open) return;
    void loadSettings();
  }, [open, loadSettings]);

  // Close on Escape.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') closeSettings();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, closeSettings]);

  if (!open) return null;

  const ordered = PROVIDER_ORDER.map((id) => providers[id]).filter(
    (p): p is ProviderState => Boolean(p),
  );
  // Append any unknown providers below the canonical three.
  const known = new Set(PROVIDER_ORDER);
  for (const p of Object.values(providers)) {
    if (!known.has(p.id)) ordered.push(p);
  }

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="settings-modal-title"
      data-testid="settings-modal"
      onClick={closeSettings}
      style={{
        position: 'fixed',
        inset: 0,
        background: 'rgba(0,0,0,0.5)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 50,
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          width: 540,
          maxHeight: '80vh',
          background: 'var(--bg-1)',
          border: '1px solid var(--line-2)',
          borderRadius: 12,
          boxShadow: 'var(--shadow-soft)',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
        }}
      >
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 10,
            padding: '14px 18px',
            borderBottom: '1px solid var(--line)',
            background: 'var(--bg-2)',
          }}
        >
          <span
            id="settings-modal-title"
            style={{ fontFamily: 'var(--serif)', fontSize: 18, color: 'var(--text-0)' }}
          >
            Settings
          </span>
          <span className="grow" />
          <button
            type="button"
            className="btn ghost icon"
            aria-label="Close settings"
            onClick={closeSettings}
          >
            ×
          </button>
        </div>

        <div style={{ padding: '16px 18px', overflowY: 'auto' }}>
          <div className="text-2" style={{ fontSize: 11, marginBottom: 10 }}>
            Choose a model provider for the agent. API keys are stored on your
            machine in <code className="mono">~/.evalyn/credentials.json</code>{' '}
            (mode 600) and never sent to the browser.
          </div>
          {ordered.map((p) => (
            <ProviderRow key={p.id} provider={p} />
          ))}
        </div>
      </div>
    </div>
  );
};

export default SettingsModal;
