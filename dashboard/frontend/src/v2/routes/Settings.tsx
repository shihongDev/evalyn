/**
 * Settings - LLM provider + API key configuration.
 *
 * Three sections (top to bottom):
 *   1. Active provider card - dropdown to switch the provider used by the
 *      co-pilot and any LLM-judge metric in a run.
 *   2. Per-provider config cards - one per known provider (anthropic, openai,
 *      ollama). Shows a status pill, an API key input (password + reveal),
 *      a model dropdown lazy-loaded from /api/settings/models/{provider},
 *      a "Test connection" button that calls /api/settings/test/{provider},
 *      and a "Save" button that posts the diff to /api/settings/{provider}.
 *   3. Inline error and success surfaces - errors render in red beneath the
 *      relevant input/button; successful saves flash a 2-second chip.
 *
 * Initial fetch goes through useV2Resource('settings', settingsApi.list)
 * for the standard skeleton/cache treatment used by the rest of v2.
 */

import { useEffect, useMemo, useState } from 'react';
import { AppShell } from '../AppShell';
import { Btn, Card, Eyebrow, Pill, Skeleton, Spinner, UpdatingChip } from '../ui';
import { useV2Resource } from '../hooks/useV2Resource';
import { E } from '../tokens';
import { settingsApi, type ProviderState, type SettingsState } from '../api/settings';

// Providers we always render a card for, even if the backend has not seen
// them yet. Order is intentional: anthropic first (the default for new
// projects), then openai, then ollama.
const KNOWN_PROVIDERS: { id: string; label: string }[] = [
  { id: 'anthropic', label: 'Anthropic' },
  { id: 'openai', label: 'OpenAI' },
  { id: 'ollama', label: 'Ollama (local)' },
];

const SELECT_STYLE = {
  background: E.panel,
  color: E.text1,
  border: `1px solid ${E.hair2}`,
  borderRadius: 6,
  padding: '6px 10px',
  fontSize: 12.5,
  fontFamily: E.fSans,
  cursor: 'pointer',
  outline: 'none',
  minWidth: 200,
} as const;

const INPUT_STYLE = {
  background: E.panel,
  color: E.text1,
  border: `1px solid ${E.hair2}`,
  borderRadius: 6,
  padding: '6px 10px',
  fontSize: 12.5,
  fontFamily: E.fMono,
  outline: 'none',
  flex: 1,
  minWidth: 0,
} as const;

function providerLabel(id: string): string {
  const known = KNOWN_PROVIDERS.find((p) => p.id === id);
  return known?.label ?? id;
}

function relativeTime(iso: string | null | undefined): string {
  if (!iso) return '';
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return iso;
  const diff = Date.now() - t;
  const mins = Math.floor(diff / 60_000);
  if (mins < 1) return 'just now';
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  if (days < 30) return `${days}d ago`;
  const months = Math.floor(days / 30);
  return `${months}mo ago`;
}

export default function Settings() {
  const { data, err, refetch, reloading, isInitialLoad } = useV2Resource<SettingsState>(
    'settings',
    settingsApi.list,
  );

  return (
    <AppShell breadcrumb={['Settings']}>
      <div style={{ padding: '32px 36px', maxWidth: 920 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <Eyebrow>Workspace configuration</Eyebrow>
          <UpdatingChip visible={reloading && !isInitialLoad} />
        </div>
        <h1
          style={{
            fontFamily: E.fSerif,
            fontSize: 34,
            fontWeight: 400,
            margin: '4px 0 6px',
            color: E.text0,
            letterSpacing: '-0.015em',
          }}
        >
          Settings
        </h1>
        <p style={{ fontSize: 13, color: E.text2, margin: 0 }}>
          Configure LLM providers and API keys for the co-pilot and judge runs.
        </p>

        {err && (
          <Card style={{ padding: 16, marginTop: 22, borderColor: E.fail }}>
            <Eyebrow style={{ color: E.fail }}>Error loading settings</Eyebrow>
            <div style={{ fontFamily: E.fMono, fontSize: 12, color: E.text2, marginTop: 6 }}>
              {err}
            </div>
            <div style={{ marginTop: 10 }}>
              <Btn kind="secondary" size="sm" onClick={() => void refetch()} disabled={reloading}>
                {reloading ? <><Spinner size={11} /> Retrying</> : 'Retry'}
              </Btn>
            </div>
          </Card>
        )}

        {!data && !err && (
          <div style={{ marginTop: 22, display: 'flex', flexDirection: 'column', gap: 14 }}>
            <Card style={{ padding: 20 }}>
              <Skeleton w={120} h={11} />
              <div style={{ marginTop: 12 }}>
                <Skeleton w={260} h={28} />
              </div>
              <div style={{ marginTop: 8 }}>
                <Skeleton w={400} h={12} />
              </div>
            </Card>
            {[0, 1, 2].map((i) => (
              <Card key={i} style={{ padding: 20 }}>
                <Skeleton w={100} h={11} />
                <div style={{ marginTop: 10 }}>
                  <Skeleton w={180} h={20} />
                </div>
                <div style={{ marginTop: 14 }}>
                  <Skeleton w="100%" h={32} />
                </div>
                <div style={{ marginTop: 10 }}>
                  <Skeleton w="60%" h={32} />
                </div>
              </Card>
            ))}
          </div>
        )}

        {data && (
          <>
            <ActiveProviderCard data={data} onChanged={() => void refetch()} />

            <div style={{ display: 'flex', flexDirection: 'column', gap: 14, marginTop: 14 }}>
              {KNOWN_PROVIDERS.map((p) => (
                <ProviderCard
                  key={p.id}
                  id={p.id}
                  label={p.label}
                  state={data.providers[p.id] ?? { is_set: false }}
                  onSaved={() => void refetch()}
                />
              ))}
            </div>
          </>
        )}

        <div style={{ height: 30 }} />
      </div>
    </AppShell>
  );
}

interface ActiveProviderCardProps {
  data: SettingsState;
  onChanged: () => void;
}

function ActiveProviderCard({ data, onChanged }: ActiveProviderCardProps) {
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  // Only providers that are actually configured can be activated. The
  // backend rejects activating an unconfigured provider with a 400, so we
  // pre-filter the dropdown to avoid that footgun.
  const options = useMemo(() => {
    return Object.entries(data.providers)
      .filter(([, st]) => st.is_set)
      .map(([id, st]) => ({
        id,
        label: st.model ? `${providerLabel(id)} - ${st.model}` : providerLabel(id),
      }));
  }, [data.providers]);

  async function handleChange(next: string) {
    if (next === data.active) return;
    setPending(true);
    setError(null);
    setSuccess(false);
    try {
      await settingsApi.setActive(next);
      setSuccess(true);
      setTimeout(() => setSuccess(false), 2000);
      onChanged();
    } catch (e) {
      setError(String(e));
    } finally {
      setPending(false);
    }
  }

  return (
    <Card style={{ padding: 20, marginTop: 22 }}>
      <Eyebrow>Active provider</Eyebrow>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginTop: 10 }}>
        <select
          value={data.active ?? ''}
          onChange={(e) => void handleChange(e.target.value)}
          disabled={pending || options.length === 0}
          style={{ ...SELECT_STYLE, minWidth: 280 }}
          aria-label="Active provider"
        >
          {options.length === 0 && (
            <option value="" style={{ background: E.panel }}>
              No providers configured yet
            </option>
          )}
          {options.map((o) => (
            <option key={o.id} value={o.id} style={{ background: E.panel }}>
              {o.label}
            </option>
          ))}
        </select>
        {pending && <Spinner size={12} />}
        {success && (
          <Pill mono color={E.pass} bg={E.passDim}>
            Saved
          </Pill>
        )}
      </div>
      <p style={{ fontSize: 12, color: E.text2, margin: '8px 0 0' }}>
        This provider is used by the co-pilot and any LLM-judge metric in your runs.
      </p>
      {error && (
        <div style={{ marginTop: 8, fontSize: 12, color: E.fail, fontFamily: E.fMono }}>
          {error}
        </div>
      )}
    </Card>
  );
}

interface ProviderCardProps {
  id: string;
  label: string;
  state: ProviderState;
  onSaved: () => void;
}

function ProviderCard({ id, label, state, onSaved }: ProviderCardProps) {
  // Edit-state lives locally per card. The api_key field is undefined until
  // the user types something - that lets us send only the fields they
  // actually edited (POST {} with no fields would 400).
  const [apiKey, setApiKey] = useState<string>('');
  const [model, setModel] = useState<string>(state.model ?? '');
  const [revealKey, setRevealKey] = useState(false);
  const [models, setModels] = useState<string[] | null>(null);
  const [modelsErr, setModelsErr] = useState<string | null>(null);
  const [modelsLoading, setModelsLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState<{ ok: boolean; message: string } | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [saveSuccess, setSaveSuccess] = useState(false);

  // Reset local edit state if the upstream snapshot changes (e.g. after a
  // successful save kicks a refetch). We compare just the bits the form
  // mirrors so we don't fight the user mid-edit.
  useEffect(() => {
    setModel(state.model ?? '');
    setApiKey('');
    setRevealKey(false);
  }, [state.model, state.is_set]);

  // Lazy-load the model catalogue on mount. /api/settings/models/openai and
  // /anthropic return a hard-coded list (cheap); /ollama hits the local
  // server and may fail (502) - we surface that inline rather than blowing
  // up the card.
  useEffect(() => {
    let cancelled = false;
    setModelsLoading(true);
    settingsApi
      .models(id)
      .then((r) => {
        if (cancelled) return;
        setModels(r.models);
        setModelsErr(null);
      })
      .catch((e) => {
        if (cancelled) return;
        setModelsErr(String(e));
        setModels([]);
      })
      .finally(() => {
        if (!cancelled) setModelsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [id]);

  const dirty = apiKey.length > 0 || model !== (state.model ?? '');

  async function handleSave() {
    if (!dirty) return;
    setSaving(true);
    setSaveError(null);
    setSaveSuccess(false);
    const body: { api_key?: string; model?: string } = {};
    if (apiKey.length > 0) body.api_key = apiKey;
    if (model !== (state.model ?? '')) body.model = model;
    try {
      await settingsApi.save(id, body);
      setSaveSuccess(true);
      setApiKey('');
      setRevealKey(false);
      setTimeout(() => setSaveSuccess(false), 2000);
      onSaved();
    } catch (e) {
      setSaveError(String(e));
    } finally {
      setSaving(false);
    }
  }

  async function handleTest() {
    setTesting(true);
    setTestResult(null);
    try {
      const r = await settingsApi.test(id);
      setTestResult({ ok: r.ok, message: r.ok ? 'Connection ok' : (r.error ?? 'Test failed') });
    } catch (e) {
      setTestResult({ ok: false, message: String(e) });
    } finally {
      setTesting(false);
    }
  }

  // Display value for the API key input. When the user has edited (apiKey
  // is non-empty) we honor that. Otherwise: if the key is set on the
  // backend, show a redacted placeholder; if not set, show empty so the
  // input visually invites typing.
  const displayValue = apiKey.length > 0
    ? apiKey
    : state.is_set
      ? '••••••••••••'
      : '';

  // Ollama uses a base_url instead of an API key; we still expose a key
  // field for symmetry but the placeholder hints that it's optional.
  const isOllama = id === 'ollama';

  return (
    <Card style={{ padding: 20 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        <Eyebrow>{label}</Eyebrow>
        {state.is_set ? (
          <Pill mono color={E.pass} bg={E.passDim}>
            configured
          </Pill>
        ) : (
          <Pill mono color={E.text3} bg={E.panel3}>
            not set
          </Pill>
        )}
      </div>

      {/* API key row */}
      <div style={{ marginTop: 14 }}>
        <label
          htmlFor={`apikey-${id}`}
          style={{ fontSize: 11.5, color: E.text2, display: 'block', marginBottom: 6 }}
        >
          {isOllama ? 'API key (optional)' : 'API key'}
        </label>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <input
            id={`apikey-${id}`}
            type={revealKey ? 'text' : 'password'}
            value={displayValue}
            placeholder={isOllama ? 'Leave blank for local Ollama' : 'sk-...'}
            onChange={(e) => setApiKey(e.target.value)}
            onFocus={() => {
              // First focus on a redacted key clears the placeholder dots
              // so the user can type a fresh key without manually deleting.
              if (apiKey.length === 0 && state.is_set) setApiKey('');
            }}
            style={INPUT_STYLE}
          />
          <Btn
            kind="ghost"
            size="sm"
            onClick={() => setRevealKey((v) => !v)}
            title={revealKey ? 'Hide key' : 'Show key'}
          >
            {revealKey ? 'Hide' : 'Show'}
          </Btn>
        </div>
      </div>

      {/* Model row */}
      <div style={{ marginTop: 12 }}>
        <label
          htmlFor={`model-${id}`}
          style={{ fontSize: 11.5, color: E.text2, display: 'block', marginBottom: 6 }}
        >
          Model
        </label>
        <select
          id={`model-${id}`}
          value={model}
          onChange={(e) => setModel(e.target.value)}
          disabled={modelsLoading || (models?.length ?? 0) === 0}
          style={SELECT_STYLE}
        >
          {model === '' && (
            <option value="" style={{ background: E.panel }}>
              {modelsLoading ? 'Loading models...' : 'Select a model'}
            </option>
          )}
          {/* Always render the currently-saved model even if the catalogue
              omits it - prevents a stale value from silently disappearing. */}
          {model !== '' && !(models ?? []).includes(model) && (
            <option value={model} style={{ background: E.panel }}>
              {model}
            </option>
          )}
          {(models ?? []).map((m) => (
            <option key={m} value={m} style={{ background: E.panel }}>
              {m}
            </option>
          ))}
        </select>
        {modelsErr && (
          <div style={{ marginTop: 6, fontSize: 11.5, color: E.fail, fontFamily: E.fMono }}>
            {modelsErr}
          </div>
        )}
      </div>

      {/* Action row */}
      <div
        style={{
          marginTop: 16,
          display: 'flex',
          alignItems: 'center',
          gap: 10,
          flexWrap: 'wrap',
        }}
      >
        <Btn kind="primary" size="md" onClick={() => void handleSave()} disabled={!dirty || saving}>
          {saving ? <><Spinner size={11} /> Saving</> : 'Save'}
        </Btn>
        <Btn kind="secondary" size="md" onClick={() => void handleTest()} disabled={testing || !state.is_set}>
          {testing ? <><Spinner size={11} /> Testing</> : 'Test connection'}
        </Btn>
        {testResult && (
          <Pill
            mono
            color={testResult.ok ? E.pass : E.fail}
            bg={testResult.ok ? E.passDim : E.failDim}
          >
            {testResult.ok ? 'Pass' : 'Fail'}
          </Pill>
        )}
        {saveSuccess && (
          <Pill mono color={E.pass} bg={E.passDim}>
            Saved
          </Pill>
        )}
      </div>

      {testResult && !testResult.ok && (
        <div
          style={{
            marginTop: 8,
            fontSize: 12,
            color: E.fail,
            fontFamily: E.fMono,
            wordBreak: 'break-word',
          }}
        >
          {testResult.message}
        </div>
      )}

      {saveError && (
        <div style={{ marginTop: 8, fontSize: 12, color: E.fail, fontFamily: E.fMono }}>
          {saveError}
        </div>
      )}

      {state.added_at && (
        <div style={{ marginTop: 12, fontSize: 11, color: E.text3, fontFamily: E.fMono }}>
          Last set: {relativeTime(state.added_at)}
        </div>
      )}

      {!state.is_set && (
        <div style={{ marginTop: 12, fontSize: 12, color: E.text2 }}>
          Add API key {String.fromCharCode(8594)}
        </div>
      )}
    </Card>
  );
}
