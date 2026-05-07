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
import { Btn, Card, Eyebrow, Pill, Skeleton, Spinner, StatusDot, UpdatingChip } from '../ui';
import { useV2Resource } from '../hooks/useV2Resource';
import { E } from '../tokens';
import { errorMessage } from '../api/errors';
import { fetchSystemHealth, type SystemHealth } from '../api/health';
import { settingsApi, type ProviderState, type SettingsState } from '../api/settings';
import { TOUR_ENABLED_KEY, tourCompletedKey } from '../store/store';
import { KNOWN_TOUR_IDS } from '../tour/useTour';

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

/** Relative-time formatter for the Test chip's "tested Ns ago" label.
 * Takes a millisecond timestamp + an explicit "now" so the parent can
 * trigger re-renders via state (the regular Date.now() call inside
 * would freeze the label until something else re-renders). Includes
 * second-resolution for the first minute - useful right after a click
 * because "tested just now" -> "tested 30s ago" is a quick natural
 * progression that confirms the chip is live. */
function relativeTestTime(testedAtMs: number, nowMs: number): string {
  const diff = Math.max(0, nowMs - testedAtMs);
  const sec = Math.floor(diff / 1000);
  if (sec < 5) return 'tested just now';
  if (sec < 60) return `tested ${sec}s ago`;
  const mins = Math.floor(sec / 60);
  if (mins < 60) return `tested ${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `tested ${hours}h ago`;
  return `tested ${Math.floor(hours / 24)}d ago`;
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
          <UpdatingChip
            visible={reloading && !isInitialLoad}
            error={data ? err : null}
            onRetry={refetch}
          />
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

            <BulkTestCard data={data} />

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

            <GuidanceToggleCard />

            <SystemStatusCard />
          </>
        )}

        <div style={{ height: 30 }} />
      </div>
    </AppShell>
  );
}

/**
 * Lives at the bottom of Settings as a passive status surface
 * for operators running the dashboard long-term. Pairs with the
 * `jobs_persisted` + `jobs_db_bytes` fields recently added to
 * `/api/health`.
 *
 * Customer scenario this answers: "I've been running the
 * dashboard for two months. Is the sqlite mirror getting big? Did
 * the server hot-restart since I last tabbed away? How many slots
 * are currently busy?" Settings is the right surface (not the top
 * bar) because none of these are actionable enough to deserve
 * always-visible chrome - operators look here when they want
 * answers.
 *
 * Polled every 15s (matches the perceived "still fresh" window
 * for status surfaces). Hidden entirely on fetch failure -
 * health is a nice-to-have indicator, never load-bearing.
 */
function SystemStatusCard() {
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [loaded, setLoaded] = useState(false);

  // Visibility-aware polling. When the tab is hidden the user
  // can't see the card so polling is pure waste; we pause on
  // hidden, resume on visible. On resume, fire an immediate
  // refetch so the card lands fresh - a 15s wait after tabbing
  // back would feel stale (uptime climbing from a wall-clock
  // value the user can guess is wrong).
  //
  // Same pattern as the drawer's start/stop helper. Keeping the
  // interval id in a ref-like local scope (closed over by start()
  // and stop()) avoids a stale-closure bug where stop() could
  // miss the latest interval handle.
  useEffect(() => {
    let cancelled = false;
    let intervalId: number | null = null;

    async function poll() {
      const h = await fetchSystemHealth();
      if (cancelled) return;
      setHealth(h);
      setLoaded(true);
    }

    function start() {
      if (intervalId !== null) return;
      void poll();
      intervalId = window.setInterval(() => void poll(), 15000);
    }

    function stop() {
      if (intervalId !== null) {
        window.clearInterval(intervalId);
        intervalId = null;
      }
    }

    if (typeof document === 'undefined' || document.visibilityState === 'visible') {
      start();
    }
    const onVisibility = () => {
      if (document.visibilityState === 'visible') {
        start();
      } else {
        stop();
      }
    };
    document.addEventListener('visibilitychange', onVisibility);
    return () => {
      cancelled = true;
      stop();
      document.removeEventListener('visibilitychange', onVisibility);
    };
  }, []);

  if (!loaded) return null;
  if (health === null) return null;

  return (
    <Card style={{ padding: 20, marginTop: 14 }}>
      <Eyebrow>System status</Eyebrow>
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(2, 1fr)',
          gap: '10px 24px',
          marginTop: 12,
          fontFamily: E.fMono,
          fontSize: 12,
        }}
      >
        <StatusRow
          label="Version"
          value={health.version}
          tooltip="Build identifier reported by the running server"
        />
        <StatusRow
          label="Uptime"
          value={formatUptime(health.uptime_seconds)}
          tooltip={`Started at ${new Date(health.started_at * 1000).toLocaleString()}`}
        />
        <StatusRow
          label="Running jobs"
          value={
            health.max_concurrent > 0
              ? `${health.running} / ${health.max_concurrent}`
              : `${health.running}`
          }
          tooltip={
            health.max_concurrent > 0
              ? `Concurrent slot cap is ${health.max_concurrent}`
              : 'Concurrency cap is disabled'
          }
        />
        <StatusRow
          label="Agent threads"
          value={`${health.agent_open_threads} open / ${health.agent_threads} total`}
          tooltip="Co-pilot threads still active vs all in memory"
        />
        <StatusRow
          label="Persisted jobs"
          value={health.jobs_persisted.toLocaleString()}
          tooltip="Rows in the sqlite mirror"
        />
        <StatusRow
          label="DB size"
          value={formatBytes(health.jobs_db_bytes)}
          tooltip="Main + WAL + SHM file size; vacuumed on graceful shutdown"
        />
        <StatusRow
          label="Last vacuum"
          value={
            health.last_vacuum_at === null
              ? 'never (this session)'
              : `${formatRelativeAgo(health.last_vacuum_at * 1000)} ago`
          }
          tooltip={
            health.last_vacuum_at === null
              ? 'No VACUUM has run since the server started. Vacuum runs on graceful shutdown; restart the server to compact.'
              : `Last VACUUM at ${new Date(health.last_vacuum_at * 1000).toLocaleString()}`
          }
        />
      </div>
      <p style={{ fontSize: 11, color: E.text3, margin: '14px 0 0' }}>
        Polled every 15s. Numbers are best-effort: a degraded
        persistence layer reports 0 rather than failing.
      </p>
    </Card>
  );
}

function StatusRow({
  label,
  value,
  tooltip,
}: {
  label: string;
  value: string;
  tooltip?: string;
}) {
  return (
    <div
      title={tooltip}
      style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}
    >
      <span style={{ color: E.text3, minWidth: 110 }}>{label}</span>
      <span style={{ color: E.text0 }}>{value}</span>
    </div>
  );
}

/**
 * Format a byte count as human-readable. KiB-base (1024) since
 * the values come from `os.stat().st_size` which the user will
 * compare against `du`, `ls -lh`, etc., all of which are KiB-base
 * by default on the platforms operators care about.
 */
export function formatBytes(n: number): string {
  if (!Number.isFinite(n) || n < 0) return '-';
  if (n < 1024) return `${n} B`;
  const units = ['KB', 'MB', 'GB', 'TB'];
  let value = n / 1024;
  let i = 0;
  while (value >= 1024 && i < units.length - 1) {
    value /= 1024;
    i += 1;
  }
  // 1 decimal under 100, none above - reads cleanly across the
  // common range (a few KB up to multi-GB).
  return value < 100
    ? `${value.toFixed(1)} ${units[i]}`
    : `${Math.round(value)} ${units[i]}`;
}

/**
 * Format uptime as the largest meaningful unit, e.g. "3d 4h",
 * "1h 12m", "47s". Operators want a glance-able scale, not
 * second-precision for a 9-day-old process.
 */
/**
 * "5m" / "2h" / "3d" relative-time formatter, expressed as a
 * single largest unit. Used by the "Last vacuum" row where the
 * caller appends " ago". For events more than 30 days old we
 * fall back to "30d+" rather than computing months - in this
 * app's normal lifecycle, a vacuum that hasn't run in 30 days
 * means the process has been up for >30 days, which is itself
 * worth flagging as "you should probably restart".
 */
export function formatRelativeAgo(eventAtMs: number): string {
  if (!Number.isFinite(eventAtMs)) return '-';
  const diffMs = Math.max(0, Date.now() - eventAtMs);
  const sec = Math.floor(diffMs / 1000);
  if (sec < 60) return `${sec}s`;
  const min = Math.floor(sec / 60);
  if (min < 60) return `${min}m`;
  const hr = Math.floor(min / 60);
  if (hr < 24) return `${hr}h`;
  const days = Math.floor(hr / 24);
  if (days <= 30) return `${days}d`;
  return '30d+';
}

export function formatUptime(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return '-';
  const s = Math.floor(seconds);
  const days = Math.floor(s / 86400);
  const hours = Math.floor((s % 86400) / 3600);
  const mins = Math.floor((s % 3600) / 60);
  if (days > 0) return `${days}d ${hours}h`;
  if (hours > 0) return `${hours}h ${mins}m`;
  if (mins > 0) return `${mins}m`;
  return `${s}s`;
}

/**
 * Co-pilot UI guidance toggle. Pure client-side preference - persisted to
 * localStorage, not the provider-keyed /api/settings endpoint. The toggle
 * gates the first-visit autotour fire path in routes/Home.tsx.
 *
 *   storage:
 *     evalyn.tour.enabled         'false' = off, anything else (or missing) = on
 *     evalyn.tour.completed.<id>  '1' once user has seen the tour for <id>
 */
export function GuidanceToggleCard() {
  const [enabled, setEnabled] = useState<boolean>(() => {
    try {
      return window.localStorage.getItem(TOUR_ENABLED_KEY) !== 'false';
    } catch {
      return true;
    }
  });
  const [resetFlash, setResetFlash] = useState(false);

  const handleToggle = () => {
    const next = !enabled;
    setEnabled(next);
    try {
      window.localStorage.setItem(TOUR_ENABLED_KEY, String(next));
    } catch {
      // localStorage unavailable (private mode, quota); the in-memory state
      // still flips so the user gets feedback for this session.
    }
  };

  const handleReset = () => {
    // Clear EVERY tour's completion flag so each tab will re-fire its
    // first-visit guidance the next time the user opens that route.
    try {
      for (const id of KNOWN_TOUR_IDS) {
        window.localStorage.removeItem(tourCompletedKey(id));
      }
    } catch {
      // localStorage unavailable - silent best-effort.
    }
    setResetFlash(true);
    window.setTimeout(() => setResetFlash(false), 2000);
  };

  return (
    <Card style={{ padding: 20, marginTop: 14 }}>
      <Eyebrow>Co-pilot UI guidance</Eyebrow>
      <div style={{ marginTop: 12, display: 'flex', alignItems: 'center', gap: 14 }}>
        <button
          type="button"
          role="switch"
          aria-checked={enabled}
          aria-label="Co-pilot UI guidance"
          onClick={handleToggle}
          style={{
            width: 36,
            height: 20,
            borderRadius: 10,
            border: 'none',
            background: enabled ? E.ember : E.panel3,
            position: 'relative',
            cursor: 'pointer',
            transition: 'background 120ms',
            padding: 0,
            flexShrink: 0,
          }}
        >
          <span
            style={{
              position: 'absolute',
              top: 2,
              left: enabled ? 18 : 2,
              width: 16,
              height: 16,
              borderRadius: '50%',
              background: '#fff',
              transition: 'left 120ms',
            }}
          />
        </button>
        <div style={{ fontSize: 13, color: E.text1 }}>
          {enabled ? 'On' : 'Off'} - first-visit walk-throughs on dashboard sections
        </div>
      </div>
      <div style={{ marginTop: 10, fontSize: 11.5, color: E.text3, lineHeight: 1.5 }}>
        When on, the co-pilot offers a short tour the first time you open a section. Already-seen tours stay dismissed unless you reset them below.
      </div>
      <div style={{ marginTop: 12, display: 'flex', alignItems: 'center', gap: 10 }}>
        <Btn kind="bare" size="sm" onClick={handleReset}>
          Reset first-visit flags
        </Btn>
        {resetFlash && (
          <Pill mono color={E.pass} bg={E.passDim}>
            Reset
          </Pill>
        )}
      </div>
    </Card>
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
      setError(errorMessage(e));
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

interface BulkTestCardProps {
  data: SettingsState;
}

/** Bulk "Test all configured" affordance.
 *
 * Customer-cared admin scenario: user has multiple providers
 * configured (OpenAI + Anthropic + Ollama) and wants a single click
 * to verify which ones still work after a key rotation, base-URL
 * change, or model deprecation. Without this, they'd have to scroll
 * to each card and click Test individually.
 *
 * Sequential rather than parallel so:
 *   - rate-limit / capacity errors don't cascade across providers
 *   - the result list grows top-down in run order, which reads
 *     naturally for the user
 *   - one slow probe (cold ollama) doesn't bottleneck the others
 *     visually - the in-flight pill makes progress obvious
 *
 * Hidden when no providers are configured (nothing to test).
 */
function BulkTestCard({ data }: BulkTestCardProps) {
  const configured = useMemo(
    () =>
      Object.entries(data.providers)
        .filter(([, st]) => st.is_set)
        .map(([id]) => id)
        .sort(),
    [data.providers],
  );
  const [running, setRunning] = useState<string | null>(null);
  const [results, setResults] = useState<
    Record<string, { ok: boolean; message: string }>
  >({});

  if (configured.length === 0) return null;

  const onRunAll = async () => {
    setResults({});
    for (const id of configured) {
      setRunning(id);
      try {
        const r = await settingsApi.test(id);
        setResults((prev) => ({
          ...prev,
          [id]: {
            ok: r.ok,
            message: r.ok ? 'Connection ok' : r.error ?? 'Test failed',
          },
        }));
      } catch (e) {
        setResults((prev) => ({
          ...prev,
          [id]: { ok: false, message: String(e) },
        }));
      }
    }
    setRunning(null);
  };

  return (
    <Card style={{ padding: 16, marginTop: 14 }}>
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: 12,
        }}
      >
        <div>
          <Eyebrow>Health check</Eyebrow>
          <div
            style={{
              fontSize: 12,
              color: E.text2,
              marginTop: 4,
              fontFamily: E.fMono,
            }}
          >
            Test {configured.length} configured provider
            {configured.length === 1 ? '' : 's'}
          </div>
        </div>
        <Btn
          kind="secondary"
          size="md"
          onClick={() => void onRunAll()}
          disabled={running !== null}
        >
          {running ? (
            <>
              <Spinner size={11} /> Testing {running}
            </>
          ) : (
            'Test all configured'
          )}
        </Btn>
      </div>
      {Object.keys(results).length > 0 && (
        <div
          style={{
            marginTop: 12,
            display: 'flex',
            flexDirection: 'column',
            gap: 6,
          }}
        >
          {configured
            .filter((id) => id in results)
            .map((id) => {
              const r = results[id];
              return (
                <div
                  key={id}
                  style={{
                    display: 'flex',
                    alignItems: 'baseline',
                    gap: 8,
                    fontSize: 12,
                    fontFamily: E.fMono,
                  }}
                >
                  <span style={{ color: r.ok ? E.pass : E.fail, width: 14 }}>
                    {r.ok ? '✓' : '✗'}
                  </span>
                  <span style={{ color: E.text1, minWidth: 80 }}>{id}</span>
                  <span style={{ color: r.ok ? E.text2 : E.fail, flex: 1 }}>
                    {r.message}
                  </span>
                </div>
              );
            })}
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
  const [testResult, setTestResult] = useState<{
    ok: boolean;
    message: string;
    /** Wall-clock time the test landed, used to render "tested 30s ago"
     * so the user can tell at a glance whether the chip reflects a
     * fresh probe or a stale earlier result. */
    testedAtMs: number;
  } | null>(null);
  // Re-render once a minute while a result is visible so the
  // "tested Ns/Nm ago" label stays current. Cheap: 1Hz/min and only
  // when a result exists. Skipping setInterval entirely when no
  // result is showing keeps idle pages quiet.
  const [nowMs, setNowMs] = useState(Date.now());
  useEffect(() => {
    if (!testResult) return;
    const id = window.setInterval(() => setNowMs(Date.now()), 60_000);
    return () => window.clearInterval(id);
  }, [testResult]);
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
        setModelsErr(errorMessage(e));
        setModels([]);
      })
      .finally(() => {
        if (!cancelled) setModelsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [id]);

  // Compute dirty against the *trimmed* key so a stray space bar does
  // not enable Save (and worse, submit a whitespace-only key that the
  // backend would happily store and then fail every API call with).
  const trimmedKey = apiKey.trim();
  const dirty = trimmedKey.length > 0 || model !== (state.model ?? '');

  async function handleSave() {
    // Guard same-frame double-clicks: the Save button is
    // disabled={!dirty || saving}, but React's batched state
    // updates leave a same-frame second click able to slip
    // through with the OLD closure (saving=false). Bail
    // explicitly so a rapid second click is a no-op.
    if (saving || !dirty) return;
    setSaving(true);
    setSaveError(null);
    setSaveSuccess(false);
    const body: { api_key?: string; model?: string } = {};
    // Always submit the trimmed value - secret managers and password
    // managers commonly paste with a trailing newline or stray space
    // that would silently break auth on every subsequent request.
    if (trimmedKey.length > 0) body.api_key = trimmedKey;
    if (model !== (state.model ?? '')) body.model = model;
    try {
      await settingsApi.save(id, body);
      setSaveSuccess(true);
      setApiKey('');
      setRevealKey(false);
      setTimeout(() => setSaveSuccess(false), 2000);
      onSaved();
    } catch (e) {
      setSaveError(errorMessage(e));
    } finally {
      setSaving(false);
    }
  }

  async function handleTest() {
    setTesting(true);
    setTestResult(null);
    try {
      const r = await settingsApi.test(id);
      const now = Date.now();
      setTestResult({
        ok: r.ok,
        message: r.ok ? 'Connection ok' : (r.error ?? 'Test failed'),
        testedAtMs: now,
      });
      // Sync the relative-time clock so the first render of the
      // result chip shows "tested just now" rather than a diff
      // against a stale mount-time nowMs.
      setNowMs(now);
    } catch (e) {
      const now = Date.now();
      setTestResult({
        ok: false,
        message: String(e),
        testedAtMs: now,
      });
      setNowMs(now);
    } finally {
      setTesting(false);
    }
  }

  // The input's controlled value mirrors apiKey directly. When the backend
  // already has a key, we surface that via placeholder + adjacent helper
  // text rather than seeding the value with bullet characters - those used
  // to get appended to the user's first keystroke (sk-abc became
  // ••••••••sk-abc) because controlled input + decorative value don't mix.
  const isOllama = id === 'ollama';
  const inputPlaceholder = isOllama
    ? 'Leave blank for local Ollama'
    : state.is_set
      ? 'Type to replace, or leave blank to keep current key'
      : 'sk-...';

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
            value={apiKey}
            placeholder={inputPlaceholder}
            onChange={(e) => setApiKey(e.target.value)}
            // Enter-to-save: every form-style surface in the
            // dashboard accepts Enter as "submit", but this card
            // had no <form> and no onKeyDown, so the canonical
            // submit gesture did nothing. The handleSave() guard
            // re-checks dirty + saving so a held-down Enter
            // doesn't fire the second save mid-network.
            onKeyDown={(e) => {
              if (e.key === 'Enter' && dirty && !saving) {
                e.preventDefault();
                void handleSave();
              }
            }}
            style={INPUT_STYLE}
            spellCheck={false}
            autoComplete="off"
          />
          <Btn
            kind="ghost"
            size="sm"
            onClick={() => setRevealKey((v) => !v)}
            disabled={apiKey.length === 0}
            title={
              apiKey.length === 0
                ? 'Type a key to reveal it'
                : revealKey
                  ? 'Hide key'
                  : 'Show key'
            }
          >
            {revealKey ? 'Hide' : 'Show'}
          </Btn>
        </div>
        {state.is_set && apiKey.length === 0 && !isOllama && (
          <div
            style={{
              marginTop: 6,
              fontSize: 11,
              color: E.text3,
              fontFamily: E.fMono,
              display: 'flex',
              alignItems: 'center',
              gap: 6,
            }}
          >
            <StatusDot status="pass" size={5} />
            Key is set. The current value is hidden for safety.
          </div>
        )}
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
          <>
            <Pill
              mono
              color={testResult.ok ? E.pass : E.fail}
              bg={testResult.ok ? E.passDim : E.failDim}
            >
              {testResult.ok ? 'Pass' : 'Fail'}
            </Pill>
            <span
              title={new Date(testResult.testedAtMs).toLocaleString()}
              style={{
                fontFamily: E.fMono,
                fontSize: 10.5,
                color: E.text3,
              }}
            >
              {relativeTestTime(testResult.testedAtMs, nowMs)}
            </span>
          </>
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
          {state.updated_at && state.updated_at !== state.added_at && (
            <> {String.fromCharCode(183)} Rotated: {relativeTime(state.updated_at)}</>
          )}
        </div>
      )}

      {!state.is_set && PROVIDER_KEY_HELP[id] && (
        <div style={{ marginTop: 12, fontSize: 12, color: E.text2 }}>
          {isOllama ? (
            <>
              Run Ollama locally and point it at the default port. See{' '}
              <a
                href={PROVIDER_KEY_HELP[id]}
                target="_blank"
                rel="noopener noreferrer"
                style={{ color: E.ember, textDecoration: 'none' }}
              >
                ollama.com {String.fromCharCode(8599)}
              </a>
              {' '}for setup.
            </>
          ) : (
            <>
              Need a key? Get one at{' '}
              <a
                href={PROVIDER_KEY_HELP[id]}
                target="_blank"
                rel="noopener noreferrer"
                style={{ color: E.ember, textDecoration: 'none' }}
              >
                {new URL(PROVIDER_KEY_HELP[id]).hostname.replace(/^www\./, '')}{' '}
                {String.fromCharCode(8599)}
              </a>
              .
            </>
          )}
        </div>
      )}
    </Card>
  );
}

const PROVIDER_KEY_HELP: Record<string, string> = {
  anthropic: 'https://console.anthropic.com/settings/keys',
  openai: 'https://platform.openai.com/api-keys',
  ollama: 'https://ollama.com/download',
};
