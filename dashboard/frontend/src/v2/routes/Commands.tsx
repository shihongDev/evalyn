/**
 * Commands - tri-pane CLI workbench.
 *
 * Layout (220px / 1fr / 320px):
 *   - Left rail: search, recents, group list.
 *   - Center: focused command form, live shell preview, "what it does" +
 *     "common after this" footers.
 *   - Right: ask co-pilot slot + per-command run history.
 *
 * Data:
 *   - `useV2Resource('commands', listCli)` - the CLI catalog.
 *   - `loadJobsHistory()` filtered by `cli_id` for run history (substitutes
 *     for an as-yet-unimplemented `v2.commandHistory(id)` endpoint -
 *     keeps everything client-side and consistent with RecentJobsDrawer).
 *   - `localStorage` key `evalyn:commands:recent` for the recent list
 *     (last 5 cli ids the user invoked from this page).
 *
 * Deep-link form (preserved from the previous catalog):
 *   /commands?prefill=<cmd>&<param>=<value>...
 * selects the command on mount AND opens the CliRunner pre-filled with
 * those values, then strips the params so a refresh doesn't re-fire.
 */

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
} from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { AppShell } from '../AppShell';
import { Btn, Card, Eyebrow, Pill, Skeleton, StatusDot, UpdatingChip } from '../ui';
import {
  commandGroup,
  commandSummary,
  listCli,
  previewCommand,
  type CliParam,
  type CliSchema,
} from '../api/cli';
import { useV2Resource } from '../hooks/useV2Resource';
import { openCliRunner } from '../cliRunnerBridge';
import {
  loadJobsHistory,
  subscribeJobsHistory,
  type JobHistoryEntry,
} from '../jobsHistory';
import { copyToClipboard } from '../clipboard';
import { MOD_KEY } from '../platform';
import { E } from '../tokens';

// --- constants ---------------------------------------------------------

const GROUP_ORDER = [
  'Eval',
  'Dataset',
  'Tracing',
  'Analysis',
  'Annotation',
  'Insights',
  'Infrastructure',
  'Simulation',
  'Export',
  'Quickstart',
  'Other',
  'Misc',
];

const RECENT_KEY = 'evalyn:commands:recent';
const RECENT_MAX = 5;

/** Hardcoded chain hints. id -> follow-up command ids. Hidden if absent. */
const COMMON_AFTER: Record<string, string[]> = {
  eval: ['cluster-failures', 'review', 'eval-judge'],
  'eval-judge': ['eval-judge --calibrate', 'review', 'analyze'],
  'cluster-failures': ['review', 'analyze'],
  analyze: ['report', 'compare'],
  compare: ['report'],
  'dataset add': ['dataset publish', 'eval'],
  'dataset publish': ['eval', 'compare'],
  'trace import': ['eval', 'analyze'],
  'trace tail': ['analyze'],
  insights: ['report', 'compare'],
  report: ['weekly'],
};

// --- localStorage helpers ----------------------------------------------

function loadRecentIds(): string[] {
  try {
    if (typeof window === 'undefined') return [];
    const raw = window.localStorage.getItem(RECENT_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as unknown;
    if (!Array.isArray(parsed)) return [];
    return parsed.filter((s): s is string => typeof s === 'string').slice(0, RECENT_MAX);
  } catch {
    return [];
  }
}

function pushRecentId(id: string): string[] {
  const cur = loadRecentIds().filter((x) => x !== id);
  cur.unshift(id);
  const next = cur.slice(0, RECENT_MAX);
  try {
    if (typeof window !== 'undefined') {
      window.localStorage.setItem(RECENT_KEY, JSON.stringify(next));
    }
  } catch {
    // ignore quota / private mode
  }
  return next;
}

// --- param coercion (deep-link) ----------------------------------------

function coerceParam(p: CliParam, raw: string): unknown {
  if (p.kind === 'bool') {
    const v = raw.trim().toLowerCase();
    return v === 'true' || v === '1' || v === 'yes' || v === 'on';
  }
  if (p.kind === 'number') {
    const n = Number(raw);
    return Number.isFinite(n) ? n : raw;
  }
  if (p.kind === 'multiselect') {
    return raw
      .split(',')
      .map((s) => s.trim())
      .filter(Boolean);
  }
  return raw;
}

// --- group helpers -----------------------------------------------------

interface GroupBucket {
  group: string;
  items: CliSchema[];
}

function bucketByGroup(cmds: CliSchema[]): GroupBucket[] {
  const buckets = new Map<string, CliSchema[]>();
  for (const cmd of cmds) {
    const g = commandGroup(cmd);
    const arr = buckets.get(g) ?? [];
    arr.push(cmd);
    buckets.set(g, arr);
  }
  for (const arr of buckets.values()) {
    arr.sort((a, b) => a.id.localeCompare(b.id));
  }
  return Array.from(buckets.entries())
    .sort((a, b) => {
      const ai = GROUP_ORDER.indexOf(a[0]);
      const bi = GROUP_ORDER.indexOf(b[0]);
      const aRank = ai === -1 ? GROUP_ORDER.length : ai;
      const bRank = bi === -1 ? GROUP_ORDER.length : bi;
      if (aRank !== bRank) return aRank - bRank;
      return a[0].localeCompare(b[0]);
    })
    .map(([group, items]) => ({ group, items }));
}

function matchesQuery(cmd: CliSchema, q: string): boolean {
  if (!q) return true;
  const needle = q.toLowerCase();
  if (cmd.id.toLowerCase().includes(needle)) return true;
  if (commandGroup(cmd).toLowerCase().includes(needle)) return true;
  if (commandSummary(cmd).toLowerCase().includes(needle)) return true;
  return false;
}

// --- form-value defaults ------------------------------------------------

function buildDefaults(cmd: CliSchema): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const p of cmd.params) {
    if (p.default !== undefined) {
      out[p.name] = p.default;
    } else if (p.kind === 'bool') {
      out[p.name] = false;
    } else if (p.kind === 'multiselect') {
      out[p.name] = [];
    } else {
      out[p.name] = '';
    }
  }
  return out;
}

/** Strip empty/false/[] values so previewCommand renders a tidy line. */
function nonEmpty(values: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(values)) {
    if (v === undefined || v === null || v === '') continue;
    if (Array.isArray(v) && v.length === 0) continue;
    if (typeof v === 'boolean' && !v) continue;
    out[k] = v;
  }
  return out;
}

// --- relative-time formatter -------------------------------------------

function relativeTime(iso: string): string {
  const t = Date.parse(iso);
  if (!Number.isFinite(t)) return '';
  const sec = Math.max(0, (Date.now() - t) / 1000);
  if (sec < 60) return `${Math.floor(sec)}s ago`;
  const min = sec / 60;
  if (min < 60) return `${Math.floor(min)}m ago`;
  const hr = min / 60;
  if (hr < 24) return `${Math.floor(hr)}h ago`;
  const day = hr / 24;
  if (day < 7) return `${Math.floor(day)}d ago`;
  return `${Math.floor(day / 7)}w ago`;
}

function jobStatusDotKind(status: JobHistoryEntry['status']): string {
  if (status === 'complete') return 'pass';
  if (status === 'failed' || status === 'cancelled') return 'fail';
  if (status === 'running' || status === 'queued') return 'running';
  return 'idle';
}

const WEEK_MS = 7 * 24 * 3600 * 1000;

/** Count entries whose started_at_iso falls within the past `WEEK_MS`. */
function countRunsThisWeek(entries: JobHistoryEntry[], now: number): number {
  const cutoff = now - WEEK_MS;
  let n = 0;
  for (const e of entries) {
    const t = Date.parse(e.started_at_iso);
    if (Number.isFinite(t) && t >= cutoff) n++;
  }
  return n;
}

/** Distinct cli_ids invoked in the past week, across the global job history. */
function distinctCliIdsThisWeek(entries: JobHistoryEntry[], now: number): number {
  const cutoff = now - WEEK_MS;
  const ids = new Set<string>();
  for (const e of entries) {
    const t = Date.parse(e.started_at_iso);
    if (Number.isFinite(t) && t >= cutoff) ids.add(e.cli_id);
  }
  return ids.size;
}

// --- subscribe to localStorage-backed jobs history ---------------------

function useJobHistoryFor(cliId: string | null): JobHistoryEntry[] {
  const [all, setAll] = useState<JobHistoryEntry[]>(() => loadJobsHistory());
  useEffect(() => {
    // Subscribe to history mutations (this tab + cross-tab via storage).
    // The lazy state initializer above already seeded `all` from
    // localStorage on mount, so we don't need a redundant setAll here.
    const off = subscribeJobsHistory(() => setAll(loadJobsHistory()));
    return off;
  }, []);
  return useMemo(() => {
    if (!cliId) return [];
    return all.filter((e) => e.cli_id === cliId).slice(0, 10);
  }, [all, cliId]);
}

// =======================================================================
// Sub-component: LeftRail
// =======================================================================

interface LeftRailProps {
  cmds: CliSchema[];
  groups: GroupBucket[];
  query: string;
  setQuery: (s: string) => void;
  recentIds: string[];
  activeId: string | null;
  activeGroup: string | null;
  onSelect: (id: string) => void;
  onSelectGroup: (group: string) => void;
}

function LeftRail({
  cmds,
  groups,
  query,
  setQuery,
  recentIds,
  activeId,
  activeGroup,
  onSelect,
  onSelectGroup,
}: LeftRailProps) {
  const byId = useMemo(() => {
    const m = new Map<string, CliSchema>();
    for (const c of cmds) m.set(c.id, c);
    return m;
  }, [cmds]);

  const recents = useMemo(
    () => recentIds.map((id) => byId.get(id)).filter((c): c is CliSchema => Boolean(c)),
    [recentIds, byId],
  );

  return (
    <div
      style={{
        borderRight: `1px solid ${E.hair}`,
        padding: 14,
        overflow: 'auto',
        background: E.panel,
      }}
    >
      {/* Search */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 6,
          padding: '6px 8px',
          background: E.panel2,
          border: `1px solid ${E.hair2}`,
          borderRadius: 6,
          marginBottom: 14,
        }}
      >
        <input
          placeholder="Filter commands"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          style={{
            flex: 1,
            background: 'transparent',
            border: 'none',
            outline: 'none',
            color: E.text1,
            fontSize: 12,
            padding: 0,
            minWidth: 0,
          }}
        />
      </div>

      {recents.length > 0 && (
        <>
          <Eyebrow>Recent</Eyebrow>
          <div
            style={{
              marginTop: 6,
              display: 'flex',
              flexDirection: 'column',
              gap: 4,
              marginBottom: 18,
            }}
          >
            {recents.map((cmd) => {
              const isActive = activeId === cmd.id;
              return (
                <button
                  key={cmd.id}
                  type="button"
                  onClick={() => onSelect(cmd.id)}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 6,
                    padding: '5px 8px',
                    borderRadius: 4,
                    cursor: 'pointer',
                    background: isActive ? E.emberDim : 'transparent',
                    border: `1px solid ${isActive ? E.emberRim : 'transparent'}`,
                    textAlign: 'left',
                    color: 'inherit',
                    width: '100%',
                  }}
                >
                  <StatusDot status="idle" />
                  <span
                    style={{
                      fontFamily: E.fMono,
                      fontSize: 11.5,
                      color: isActive ? E.text0 : E.text2,
                      whiteSpace: 'nowrap',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                    }}
                  >
                    {cmd.id}
                  </span>
                </button>
              );
            })}
          </div>
        </>
      )}

      <Eyebrow>Groups</Eyebrow>
      <div
        style={{
          marginTop: 6,
          display: 'flex',
          flexDirection: 'column',
          gap: 2,
        }}
      >
        {groups.map(({ group, items }) => {
          const isActive = activeGroup === group;
          return (
            <button
              key={group}
              type="button"
              onClick={() => onSelectGroup(group)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 6,
                padding: '5px 8px',
                borderRadius: 4,
                cursor: 'pointer',
                background: isActive ? E.panel2 : 'transparent',
                border: 'none',
                textAlign: 'left',
                width: '100%',
              }}
            >
              <span
                style={{
                  fontSize: 12,
                  color: isActive ? E.text0 : E.text2,
                  flex: 1,
                }}
              >
                {group}
              </span>
              <span style={{ fontFamily: E.fMono, fontSize: 10, color: E.text3 }}>
                {items.length}
              </span>
            </button>
          );
        })}
      </div>

      {groups.length === 0 && (
        <div style={{ fontSize: 11, color: E.text3, marginTop: 8 }}>
          No commands match "{query}".
        </div>
      )}
    </div>
  );
}

// =======================================================================
// Sub-component: ParamInput
// =======================================================================

interface ParamInputProps {
  param: CliParam;
  value: unknown;
  onChange: (v: unknown) => void;
}

const INPUT_BASE: CSSProperties = {
  padding: '6px 10px',
  background: E.panel2,
  border: `1px solid ${E.hair2}`,
  borderRadius: 5,
  color: E.text1,
  fontSize: 12,
  fontFamily: E.fSans,
  outline: 'none',
};

function ParamInput({ param, value, onChange }: ParamInputProps) {
  const { kind, options } = param;

  if (kind === 'bool') {
    const checked = Boolean(value);
    return (
      <label
        style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: 6,
          fontSize: 12,
          color: E.text2,
          cursor: 'pointer',
        }}
      >
        <input
          type="checkbox"
          checked={checked}
          onChange={(e) => onChange(e.target.checked)}
        />
        {checked ? 'enabled' : 'disabled'}
      </label>
    );
  }

  if (kind === 'select') {
    return (
      <select
        value={typeof value === 'string' ? value : ''}
        onChange={(e) => onChange(e.target.value)}
        style={{ ...INPUT_BASE, fontFamily: E.fSans }}
      >
        <option value="">-</option>
        {(options ?? []).map((o) => (
          <option key={o} value={o}>
            {o}
          </option>
        ))}
      </select>
    );
  }

  if (kind === 'multiselect') {
    const arr = Array.isArray(value) ? (value as unknown[]).map(String) : [];
    const remaining = (options ?? []).filter((o) => !arr.includes(o));
    return (
      <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', alignItems: 'center' }}>
        {arr.map((m) => (
          <button
            key={m}
            type="button"
            onClick={() => onChange(arr.filter((x) => x !== m))}
            title="Remove"
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: 4,
              padding: '2px 8px',
              borderRadius: 999,
              fontSize: 11,
              fontFamily: E.fMono,
              color: E.ember,
              background: E.emberDim,
              border: 'none',
              cursor: 'pointer',
            }}
          >
            {m}
            <span style={{ opacity: 0.7 }}>x</span>
          </button>
        ))}
        {remaining.length > 0 ? (
          <select
            value=""
            onChange={(e) => {
              const v = e.target.value;
              if (v) onChange([...arr, v]);
            }}
            style={{
              padding: '2px 8px',
              borderRadius: 999,
              fontSize: 11,
              fontFamily: E.fMono,
              color: E.text3,
              background: E.panel2,
              border: `1px solid ${E.hair2}`,
              cursor: 'pointer',
            }}
          >
            <option value="">+ add</option>
            {remaining.map((o) => (
              <option key={o} value={o}>
                {o}
              </option>
            ))}
          </select>
        ) : null}
        {(!options || options.length === 0) && (
          <FreeAddPill
            onAdd={(v) => {
              if (!arr.includes(v)) onChange([...arr, v]);
            }}
          />
        )}
      </div>
    );
  }

  if (kind === 'long-text') {
    return (
      <textarea
        value={typeof value === 'string' ? value : ''}
        onChange={(e) => onChange(e.target.value)}
        rows={3}
        style={{ ...INPUT_BASE, width: '100%', resize: 'vertical', fontFamily: E.fMono }}
      />
    );
  }

  if (kind === 'number') {
    return (
      <input
        type="text"
        inputMode="numeric"
        value={value == null ? '' : String(value)}
        onChange={(e) => onChange(e.target.value)}
        style={{ ...INPUT_BASE, fontFamily: E.fMono, width: 120 }}
      />
    );
  }

  // string / path / fallback
  return (
    <input
      type="text"
      value={typeof value === 'string' ? value : value == null ? '' : String(value)}
      onChange={(e) => onChange(e.target.value)}
      placeholder={param.help ?? undefined}
      style={{ ...INPUT_BASE, width: '100%', fontFamily: kind === 'path' ? E.fMono : E.fSans }}
    />
  );
}

// "+ add" pill input for multiselect with no options: type and Enter.
function FreeAddPill({ onAdd }: { onAdd: (v: string) => void }) {
  const [draft, setDraft] = useState('');
  const [editing, setEditing] = useState(false);
  if (!editing) {
    return (
      <button
        type="button"
        onClick={() => setEditing(true)}
        style={{
          padding: '2px 8px',
          borderRadius: 999,
          fontSize: 11,
          fontFamily: E.fMono,
          color: E.text3,
          background: E.panel2,
          border: `1px solid ${E.hair2}`,
          cursor: 'pointer',
        }}
      >
        + add
      </button>
    );
  }
  return (
    <input
      autoFocus
      type="text"
      value={draft}
      onChange={(e) => setDraft(e.target.value)}
      onKeyDown={(e) => {
        if (e.key === 'Enter' && draft.trim()) {
          onAdd(draft.trim());
          setDraft('');
          setEditing(false);
        } else if (e.key === 'Escape') {
          setDraft('');
          setEditing(false);
        }
      }}
      onBlur={() => {
        if (draft.trim()) onAdd(draft.trim());
        setDraft('');
        setEditing(false);
      }}
      placeholder="value"
      style={{
        padding: '2px 8px',
        borderRadius: 999,
        fontSize: 11,
        fontFamily: E.fMono,
        color: E.text1,
        background: E.panel2,
        border: `1px solid ${E.emberRim}`,
        outline: 'none',
        width: 120,
      }}
    />
  );
}

// =======================================================================
// Sub-component: ShellPreview
// =======================================================================

const SHELL_BG = '#2c281f';
const SHELL_TEXT = '#f6e4d2';
const SHELL_DIM = '#7a6d5a';
const SHELL_FLAG = '#a8caa8';

interface ShellPreviewProps {
  cliId: string;
  values: Record<string, unknown>;
}

function ShellPreview({ cliId, values }: ShellPreviewProps) {
  const trimmed = useMemo(() => nonEmpty(values), [values]);
  const fullLine = useMemo(() => previewCommand(cliId, trimmed), [cliId, trimmed]);

  const [copied, setCopied] = useState(false);
  const onCopy = useCallback(() => {
    void copyToClipboard(fullLine).then(() => {
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1100);
    });
  }, [fullLine]);

  // Build display tokens: prompt, "evalyn <cmd>", then per-flag chunks.
  // Each chunk = "--flag value(s)" rendered on its own continuation line
  // when the command has any flags.
  const flagChunks = useMemo(() => {
    const chunks: { flag: string; values: string[] }[] = [];
    for (const [name, value] of Object.entries(trimmed)) {
      const flag = `--${name.replace(/_/g, '-')}`;
      if (typeof value === 'boolean') {
        if (value) chunks.push({ flag, values: [] });
        continue;
      }
      if (Array.isArray(value)) {
        chunks.push({ flag, values: value.map(String) });
        continue;
      }
      chunks.push({ flag, values: [String(value)] });
    }
    return chunks;
  }, [trimmed]);

  return (
    <div
      style={{
        marginTop: 14,
        padding: 12,
        background: SHELL_BG,
        borderRadius: 6,
        fontFamily: E.fMono,
        fontSize: 12,
        color: SHELL_TEXT,
        lineHeight: 1.7,
        position: 'relative',
        overflowX: 'auto',
      }}
    >
      <div>
        <span style={{ color: SHELL_DIM }}>$</span>{' '}
        <span style={{ color: E.ember }}>evalyn {cliId}</span>
        {flagChunks.length > 0 && (
          <span style={{ color: SHELL_DIM }}> \</span>
        )}
      </div>
      {flagChunks.map((chunk, i) => {
        const last = i === flagChunks.length - 1;
        return (
          <div key={`${chunk.flag}-${i}`}>
            &nbsp;&nbsp;
            <span style={{ color: SHELL_FLAG }}>{chunk.flag}</span>
            {chunk.values.length > 0 && ' '}
            {chunk.values.join(' ')}
            {!last && <span style={{ color: SHELL_DIM }}> \</span>}
          </div>
        );
      })}
      <button
        type="button"
        onClick={onCopy}
        title="Copy command"
        style={{
          position: 'absolute',
          top: 8,
          right: 8,
          fontFamily: E.fMono,
          fontSize: 10,
          color: copied ? SHELL_TEXT : SHELL_DIM,
          padding: '2px 6px',
          border: `1px solid ${SHELL_BG}`,
          background: 'rgba(255,255,255,0.04)',
          borderRadius: 4,
          cursor: 'pointer',
        }}
      >
        {copied ? 'copied' : 'copy'}
      </button>
    </div>
  );
}

// =======================================================================
// Sub-component: CommandForm (center pane body)
// =======================================================================

interface CommandFormProps {
  cli: CliSchema;
  values: Record<string, unknown>;
  setValues: (v: Record<string, unknown>) => void;
  recentRunCount: number;
  onRun: () => void;
  onSelect: (id: string) => void;
  cmdsById: Map<string, CliSchema>;
}

function CommandForm({
  cli,
  values,
  setValues,
  recentRunCount,
  onRun,
  onSelect,
  cmdsById,
}: CommandFormProps) {
  const summary = commandSummary(cli);
  const group = commandGroup(cli);
  const params = cli.params;

  function setField(name: string, v: unknown) {
    setValues({ ...values, [name]: v });
  }

  const followUps = (COMMON_AFTER[cli.id] ?? []).filter((id) => {
    const baseId = id.split(' ')[0];
    return cmdsById.has(baseId);
  });

  return (
    <div style={{ padding: 22, overflow: 'auto' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
        <Pill bg={E.steelDim} color={E.steel} mono>
          {group}
        </Pill>
        {recentRunCount > 0 && (
          <Pill bg={E.passDim} color={E.pass} mono>
            used {recentRunCount}x recently
          </Pill>
        )}
      </div>
      <h2
        style={{
          fontFamily: E.fMono,
          fontSize: 22,
          fontWeight: 500,
          color: E.text0,
          margin: '4px 0 6px',
        }}
      >
        evalyn {cli.id}
      </h2>
      {summary && (
        <p style={{ fontSize: 13, color: E.text2, lineHeight: 1.55, marginBottom: 16 }}>
          {summary}
        </p>
      )}

      {/* Form */}
      <Card style={{ padding: 16 }}>
        <Eyebrow>Parameters</Eyebrow>
        {params.length === 0 ? (
          <div style={{ marginTop: 10, fontSize: 12, color: E.text3 }}>
            No parameters - this command runs as-is.
          </div>
        ) : (
          <div
            style={{
              marginTop: 10,
              display: 'grid',
              gridTemplateColumns: '140px 1fr',
              rowGap: 10,
              columnGap: 14,
              fontSize: 12.5,
              alignItems: 'center',
            }}
          >
            {params.map((p) => (
              <ParamRow
                key={p.name}
                param={p}
                value={values[p.name]}
                onChange={(v) => setField(p.name, v)}
              />
            ))}
          </div>
        )}

        <ShellPreview cliId={cli.id} values={values} />

        <div style={{ marginTop: 14, display: 'flex', gap: 8, alignItems: 'center' }}>
          <Btn kind="primary" size="md" onClick={onRun}>
            Run
          </Btn>
          <Btn
            kind="secondary"
            size="md"
            disabled
            title="Cluster runs are not wired up yet"
          >
            Run on cluster
          </Btn>
          <Btn kind="ghost" size="md" disabled title="Scheduled runs are not wired up yet">
            Schedule
          </Btn>
          <span style={{ flex: 1 }} />
          <Btn
            kind="ghost"
            size="md"
            onClick={() => {
              void copyToClipboard(previewCommand(cli.id, nonEmpty(values)));
            }}
            title={`${MOD_KEY}+C copies the rendered command line`}
          >
            {MOD_KEY}+C copy
          </Btn>
        </div>
      </Card>

      <div style={{ marginTop: 14, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
        <Card style={{ padding: 12 }}>
          <Eyebrow>What it does</Eyebrow>
          <p
            style={{
              fontSize: 11.5,
              color: E.text2,
              lineHeight: 1.55,
              marginTop: 6,
              whiteSpace: 'pre-wrap',
            }}
          >
            {summary || 'No description available for this command.'}
          </p>
        </Card>
        {followUps.length > 0 && (
          <Card style={{ padding: 12 }}>
            <Eyebrow>Common after this</Eyebrow>
            <div
              style={{
                marginTop: 6,
                display: 'flex',
                flexDirection: 'column',
                gap: 4,
              }}
            >
              {followUps.map((id) => {
                const baseId = id.split(' ')[0];
                const exists = cmdsById.has(baseId);
                return (
                  <button
                    key={id}
                    type="button"
                    onClick={() => exists && onSelect(baseId)}
                    style={{
                      fontFamily: E.fMono,
                      fontSize: 11,
                      color: E.ember,
                      cursor: exists ? 'pointer' : 'default',
                      background: 'transparent',
                      border: 'none',
                      textAlign: 'left',
                      padding: 0,
                    }}
                  >
                    -&gt; {id}
                  </button>
                );
              })}
            </div>
          </Card>
        )}
      </div>
    </div>
  );
}

interface ParamRowProps {
  param: CliParam;
  value: unknown;
  onChange: (v: unknown) => void;
}

function ParamRow({ param, value, onChange }: ParamRowProps) {
  const flag = `--${param.name.replace(/_/g, '-')}`;
  return (
    <>
      <span
        style={{
          color: E.text2,
          fontFamily: E.fMono,
          fontSize: 11,
          alignSelf: 'flex-start',
          paddingTop: 6,
        }}
        title={param.help ?? undefined}
      >
        {flag}
        {param.required ? <span style={{ color: E.fail }}> *</span> : null}
      </span>
      <div style={{ minWidth: 0 }}>
        <ParamInput param={param} value={value} onChange={onChange} />
        {param.help && (
          <div style={{ fontSize: 10.5, color: E.text3, marginTop: 4 }}>{param.help}</div>
        )}
      </div>
    </>
  );
}

// =======================================================================
// Sub-component: CoPilotSlot
// =======================================================================

interface CoPilotSlotProps {
  activeCmd: CliSchema | null;
}

function CoPilotSlot({ activeCmd }: CoPilotSlotProps) {
  const navigate = useNavigate();
  const [prompt, setPrompt] = useState('');

  const submit = useCallback(() => {
    const text = prompt.trim();
    if (!text) return;
    const seed = activeCmd
      ? `For \`evalyn ${activeCmd.id}\`: ${text}`
      : text;
    navigate(`/copilot?prefill=${encodeURIComponent(seed)}`);
  }, [prompt, activeCmd, navigate]);

  return (
    <div style={{ padding: 14, borderBottom: `1px solid ${E.hair}` }}>
      <Eyebrow>Ask co-pilot</Eyebrow>
      <div
        style={{
          marginTop: 8,
          padding: 10,
          background: E.panel2,
          borderRadius: 6,
          border: `1px solid ${E.hair2}`,
          fontSize: 12,
          color: E.text3,
          lineHeight: 1.5,
        }}
      >
        Try{' '}
        <span style={{ color: E.text2 }}>
          "explain what this command does"
        </span>{' '}
        or{' '}
        <span style={{ color: E.text2 }}>"suggest flags for my dataset"</span>
        {activeCmd && (
          <>
            {' '}
            -&gt;{' '}
            <span style={{ color: E.ember, fontFamily: E.fMono, fontSize: 11 }}>
              {activeCmd.id}
            </span>
          </>
        )}
      </div>
      <input
        placeholder="What do you want to do?"
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter') submit();
        }}
        style={{
          marginTop: 6,
          width: '100%',
          padding: '8px 10px',
          background: E.panel2,
          border: `1px solid ${E.hair2}`,
          borderRadius: 6,
          fontSize: 12,
          color: E.text1,
          outline: 'none',
          boxSizing: 'border-box',
        }}
      />
    </div>
  );
}

// =======================================================================
// Sub-component: RunHistoryList
// =======================================================================

interface RunHistoryListProps {
  cli: CliSchema | null;
  history: JobHistoryEntry[];
}

function RunHistoryList({ cli, history }: RunHistoryListProps) {
  if (!cli) {
    return (
      <div style={{ padding: 14, flex: 1, overflow: 'auto' }}>
        <Eyebrow>Run history</Eyebrow>
        <div style={{ marginTop: 8, fontSize: 11.5, color: E.text3 }}>
          Pick a command to see its run history.
        </div>
      </div>
    );
  }

  return (
    <div style={{ padding: 14, flex: 1, overflow: 'auto', minHeight: 0 }}>
      <Eyebrow>Run history - evalyn {cli.id}</Eyebrow>
      {history.length === 0 ? (
        <div
          style={{
            marginTop: 10,
            padding: 10,
            border: `1px dashed ${E.hair2}`,
            borderRadius: 6,
            fontSize: 11.5,
            color: E.text3,
            lineHeight: 1.5,
          }}
        >
          No runs from this browser yet. Hit Run to start one.
        </div>
      ) : (
        <div style={{ marginTop: 8, display: 'flex', flexDirection: 'column', gap: 8 }}>
          {history.map((h) => {
            const isPass = h.status === 'complete';
            const isFail = h.status === 'failed' || h.status === 'cancelled';
            const accent = isPass ? E.pass : isFail ? E.fail : E.text3;
            const exit = h.exit_code;
            return (
              <div
                key={h.job_id}
                style={{
                  padding: 10,
                  background: E.panel2,
                  border: `1px solid ${E.hair2}`,
                  borderLeft: `3px solid ${accent}`,
                  borderRadius: 6,
                }}
              >
                <div
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 6,
                    marginBottom: 3,
                  }}
                >
                  <StatusDot status={jobStatusDotKind(h.status)} />
                  <span style={{ fontSize: 11.5, color: E.text1 }}>
                    {relativeTime(h.started_at_iso)}
                  </span>
                  <span style={{ flex: 1 }} />
                  <span
                    style={{
                      fontFamily: E.fMono,
                      fontSize: 10,
                      color:
                        exit === 0
                          ? E.text3
                          : exit == null
                            ? E.text3
                            : E.fail,
                    }}
                  >
                    {exit == null ? h.status : `exit ${exit}`}
                  </span>
                </div>
                <div
                  style={{
                    fontFamily: E.fMono,
                    fontSize: 10.5,
                    color: E.text3,
                    lineHeight: 1.5,
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                  }}
                  title={previewCommand(h.cli_id, h.cli_args)}
                >
                  {previewCommand(h.cli_id, h.cli_args)}
                </div>
                <div style={{ marginTop: 6, display: 'flex', gap: 6 }}>
                  <button
                    type="button"
                    onClick={() => openCliRunner(cli, { initialValues: h.cli_args })}
                    style={{
                      fontSize: 10.5,
                      color: E.ember,
                      cursor: 'pointer',
                      background: 'transparent',
                      border: 'none',
                      padding: 0,
                    }}
                  >
                    Re-run
                  </button>
                  <span style={{ color: E.text4 }}>·</span>
                  <button
                    type="button"
                    onClick={() => openCliRunner(cli, { resumeJobId: h.job_id })}
                    style={{
                      fontSize: 10.5,
                      color: E.text2,
                      cursor: 'pointer',
                      background: 'transparent',
                      border: 'none',
                      padding: 0,
                    }}
                  >
                    View output
                  </button>
                  <span style={{ color: E.text4 }}>·</span>
                  <button
                    type="button"
                    onClick={() =>
                      openCliRunner(cli, { initialValues: h.cli_args })
                    }
                    style={{
                      fontSize: 10.5,
                      color: E.text2,
                      cursor: 'pointer',
                      background: 'transparent',
                      border: 'none',
                      padding: 0,
                    }}
                  >
                    Edit & re-run
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

// =======================================================================
// Loading skeletons
// =======================================================================

function LeftRailSkeleton() {
  return (
    <div
      style={{
        borderRight: `1px solid ${E.hair}`,
        padding: 14,
        background: E.panel,
      }}
    >
      <Skeleton w="100%" h={28} style={{ borderRadius: 6, marginBottom: 14 }} />
      <Skeleton w={50} h={9} style={{ marginBottom: 8 }} />
      {[0, 1, 2, 3].map((i) => (
        <div key={i} style={{ marginBottom: 6 }}>
          <Skeleton w="80%" h={11} />
        </div>
      ))}
      <div style={{ height: 12 }} />
      <Skeleton w={50} h={9} style={{ marginBottom: 8 }} />
      {[0, 1, 2, 3, 4, 5, 6, 7].map((i) => (
        <div key={i} style={{ marginBottom: 6 }}>
          <Skeleton w="70%" h={11} />
        </div>
      ))}
    </div>
  );
}

function CenterPaneSkeleton() {
  return (
    <div style={{ padding: 22 }}>
      <Skeleton w={60} h={16} style={{ borderRadius: 999, marginBottom: 8 }} />
      <Skeleton w={220} h={28} style={{ marginBottom: 8 }} />
      <Skeleton w="80%" h={12} style={{ marginBottom: 4 }} />
      <Skeleton w="60%" h={12} style={{ marginBottom: 16 }} />
      <Card style={{ padding: 16 }}>
        <Skeleton w={80} h={9} style={{ marginBottom: 10 }} />
        {[0, 1, 2, 3].map((i) => (
          <div
            key={i}
            style={{
              display: 'grid',
              gridTemplateColumns: '140px 1fr',
              columnGap: 14,
              rowGap: 8,
              marginBottom: 8,
              alignItems: 'center',
            }}
          >
            <Skeleton w={100} h={10} />
            <Skeleton w="100%" h={26} style={{ borderRadius: 5 }} />
          </div>
        ))}
        <Skeleton w="100%" h={68} style={{ borderRadius: 6, marginTop: 10 }} />
      </Card>
    </div>
  );
}

function RightPaneSkeleton() {
  return (
    <div
      style={{
        borderLeft: `1px solid ${E.hair}`,
        background: E.panel,
        padding: 14,
      }}
    >
      <Skeleton w={80} h={9} style={{ marginBottom: 8 }} />
      <Skeleton w="100%" h={56} style={{ borderRadius: 6, marginBottom: 16 }} />
      <Skeleton w="100%" h={36} style={{ borderRadius: 6, marginBottom: 16 }} />
      <Skeleton w={120} h={9} style={{ marginBottom: 8 }} />
      {[0, 1, 2].map((i) => (
        <div key={i} style={{ marginBottom: 8 }}>
          <Skeleton w="100%" h={64} style={{ borderRadius: 6 }} />
        </div>
      ))}
    </div>
  );
}

// =======================================================================
// Top-level page
// =======================================================================

export default function Commands() {
  const {
    data: cmds,
    err,
    refetch,
    reloading,
    isInitialLoad,
  } = useV2Resource<CliSchema[]>('commands', listCli);
  const [query, setQuery] = useState('');
  const [searchParams, setSearchParams] = useSearchParams();
  const deepLinkFiredRef = useRef(false);

  // --- recents (localStorage-backed) -----------------------------------
  const [recentIds, setRecentIds] = useState<string[]>(() => loadRecentIds());

  // --- groups + lookup -------------------------------------------------
  const allGroups = useMemo(() => (cmds ? bucketByGroup(cmds) : []), [cmds]);

  const filteredGroups = useMemo(() => {
    if (!cmds) return [] as GroupBucket[];
    const filtered = cmds.filter((c) => matchesQuery(c, query));
    return bucketByGroup(filtered);
  }, [cmds, query]);

  const cmdsById = useMemo(() => {
    const m = new Map<string, CliSchema>();
    if (cmds) {
      for (const c of cmds) m.set(c.id, c);
    }
    return m;
  }, [cmds]);

  // --- active command --------------------------------------------------
  const [activeId, setActiveId] = useState<string | null>(null);
  const activeCmd = activeId ? cmdsById.get(activeId) ?? null : null;

  // First mount: pick a default command. Prefer ?prefill=, then first
  // recent in catalog, then first command in first group.
  useEffect(() => {
    if (!cmds || activeId) return;
    const prefill = searchParams.get('prefill');
    if (prefill && cmdsById.has(prefill)) {
      setActiveId(prefill);
      return;
    }
    const fromRecent = recentIds.find((id) => cmdsById.has(id));
    if (fromRecent) {
      setActiveId(fromRecent);
      return;
    }
    const first = allGroups[0]?.items[0];
    if (first) setActiveId(first.id);
  }, [cmds, activeId, allGroups, cmdsById, recentIds, searchParams]);

  // --- form-values cache (per-command) ---------------------------------
  // Persist in-memory across switches so the user doesn't lose edits when
  // they tab between commands. New commands seed from `buildDefaults`.
  const [formByCmd, setFormByCmd] = useState<Record<string, Record<string, unknown>>>({});

  function getValuesFor(cmd: CliSchema): Record<string, unknown> {
    return formByCmd[cmd.id] ?? buildDefaults(cmd);
  }

  function setValuesFor(cmd: CliSchema, values: Record<string, unknown>) {
    setFormByCmd((prev) => ({ ...prev, [cmd.id]: values }));
  }

  // Seed form values when a command becomes active for the first time.
  useEffect(() => {
    if (!activeCmd) return;
    if (formByCmd[activeCmd.id]) return;
    setFormByCmd((prev) => ({ ...prev, [activeCmd.id]: buildDefaults(activeCmd) }));
    // We intentionally do not include formByCmd in deps - we only want
    // to seed once per command.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeCmd]);

  // --- deep-link handling (preserved) ----------------------------------
  // Same contract as the old Commands page: `?prefill=<id>` selects a
  // command AND opens the runner pre-filled with any matching params.
  // We strip the params after firing so a refresh doesn't re-launch.
  useEffect(() => {
    if (deepLinkFiredRef.current) return;
    if (!cmds) return;
    const prefill = searchParams.get('prefill');
    if (!prefill) return;
    const cmd = cmds.find((c) => c.id === prefill);
    if (!cmd) {
      deepLinkFiredRef.current = true;
      const next = new URLSearchParams(searchParams);
      next.delete('prefill');
      setSearchParams(next, { replace: true });
      return;
    }
    const initialValues: Record<string, unknown> = {};
    for (const p of cmd.params) {
      const v = searchParams.get(p.name);
      if (v != null) initialValues[p.name] = coerceParam(p, v);
    }
    deepLinkFiredRef.current = true;
    if (Object.keys(initialValues).length > 0) {
      setFormByCmd((prev) => ({
        ...prev,
        [cmd.id]: { ...buildDefaults(cmd), ...initialValues },
      }));
    }
    setActiveId(cmd.id);
    openCliRunner(cmd, { initialValues });
    const next = new URLSearchParams(searchParams);
    next.delete('prefill');
    for (const p of cmd.params) next.delete(p.name);
    setSearchParams(next, { replace: true });
  }, [cmds, searchParams, setSearchParams]);

  // --- jobs history (per-cmd) ------------------------------------------
  const history = useJobHistoryFor(activeCmd?.id ?? null);

  // "Now" reference snapshotted at mount + refreshed when the page
  // remounts - the small staleness (< page lifetime) is fine for the
  // "used X times this week" pill and avoids calling Date.now() during
  // render (react-hooks/purity).
  const [nowMs] = useState<number>(() => Date.now());

  // Count "recent runs" surfaced as the usage pill - matches the active
  // command id within the last week.
  const recentRunCount = useMemo(
    () => (activeCmd ? countRunsThisWeek(history, nowMs) : 0),
    [activeCmd, history, nowMs],
  );

  // --- selection handlers ----------------------------------------------
  const onSelect = useCallback((id: string) => {
    setActiveId(id);
  }, []);

  const onSelectGroup = useCallback(
    (group: string) => {
      const bucket =
        filteredGroups.find((b) => b.group === group) ??
        allGroups.find((b) => b.group === group);
      const first = bucket?.items[0];
      if (first) setActiveId(first.id);
    },
    [filteredGroups, allGroups],
  );

  const onRun = useCallback(() => {
    if (!activeCmd) return;
    const values = formByCmd[activeCmd.id] ?? buildDefaults(activeCmd);
    openCliRunner(activeCmd, { initialValues: values });
    setRecentIds(pushRecentId(activeCmd.id));
  }, [activeCmd, formByCmd]);

  // --- header counts ---------------------------------------------------
  const totalCount = cmds?.length ?? 0;
  // The "used this week" pill reads from the global jobs history (any
  // command kicked off from any page counts). We depend on `history`
  // so the count refreshes when this page launches a job for the
  // active command - close enough; cross-page launches catch up on the
  // next visit. loadJobsHistory() is a pure synchronous localStorage
  // read so this is cheap.
  const usedThisWeek = useMemo(() => {
    void history;
    return cmds ? distinctCliIdsThisWeek(loadJobsHistory(), nowMs) : 0;
  }, [cmds, nowMs, history]);

  // current group from active command
  const activeGroup = activeCmd ? commandGroup(activeCmd) : null;

  // --- render ----------------------------------------------------------

  const showSkeleton = !cmds && !err;

  return (
    <AppShell contextChip={{ name: 'Commands', version: '' }}>
      <div
        style={{
          padding: 0,
          height: 'calc(100vh - 56px)',
          display: 'flex',
          flexDirection: 'column',
        }}
      >
        {/* Header */}
        <div
          style={{
            padding: '20px 24px 14px',
            borderBottom: `1px solid ${E.hair}`,
            background: E.panel,
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <Eyebrow>Commands - CLI workbench</Eyebrow>
            <UpdatingChip
              visible={reloading && !isInitialLoad}
              error={cmds ? err : null}
              onRetry={refetch}
            />
          </div>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 14, marginTop: 4 }}>
            <h1
              style={{
                fontFamily: E.fSerif,
                fontSize: 26,
                fontWeight: 400,
                margin: 0,
                color: E.text0,
                letterSpacing: '-0.015em',
                lineHeight: 1.1,
              }}
            >
              Build, run, replay
            </h1>
            <span style={{ flex: 1 }} />
            {cmds && (
              <span style={{ fontFamily: E.fMono, fontSize: 11, color: E.text3 }}>
                {totalCount} commands - {usedThisWeek} used this week
              </span>
            )}
          </div>
        </div>

        {err && !cmds && (
          <Card style={{ margin: 24, padding: 16, borderColor: E.fail }}>
            <Eyebrow style={{ color: E.fail }}>Error loading commands</Eyebrow>
            <div style={{ fontFamily: E.fMono, fontSize: 12, color: E.text2, marginTop: 6 }}>
              {err}
            </div>
            <div style={{ marginTop: 10 }}>
              <Btn kind="secondary" size="sm" onClick={refetch}>
                Retry
              </Btn>
            </div>
          </Card>
        )}

        {/* Tri-pane body */}
        <div
          style={{
            flex: 1,
            display: 'grid',
            gridTemplateColumns: '220px 1fr 320px',
            minHeight: 0,
          }}
        >
          {showSkeleton ? (
            <>
              <LeftRailSkeleton />
              <CenterPaneSkeleton />
              <RightPaneSkeleton />
            </>
          ) : (
            <>
              {cmds ? (
                <LeftRail
                  cmds={cmds}
                  groups={filteredGroups}
                  query={query}
                  setQuery={setQuery}
                  recentIds={recentIds}
                  activeId={activeId}
                  activeGroup={activeGroup}
                  onSelect={onSelect}
                  onSelectGroup={onSelectGroup}
                />
              ) : (
                <LeftRailSkeleton />
              )}

              <div style={{ overflow: 'auto', minHeight: 0 }}>
                {activeCmd ? (
                  <CommandForm
                    cli={activeCmd}
                    values={getValuesFor(activeCmd)}
                    setValues={(v) => setValuesFor(activeCmd, v)}
                    recentRunCount={recentRunCount}
                    onRun={onRun}
                    onSelect={onSelect}
                    cmdsById={cmdsById}
                  />
                ) : cmds ? (
                  <div
                    style={{
                      padding: 32,
                      color: E.text3,
                      fontSize: 13,
                    }}
                  >
                    Pick a command from the left rail.
                  </div>
                ) : (
                  <CenterPaneSkeleton />
                )}
              </div>

              <div
                style={{
                  borderLeft: `1px solid ${E.hair}`,
                  background: E.panel,
                  display: 'flex',
                  flexDirection: 'column',
                  minHeight: 0,
                }}
              >
                <CoPilotSlot activeCmd={activeCmd} />
                <RunHistoryList cli={activeCmd} history={history} />
              </div>
            </>
          )}
        </div>

        {/* Why this changed footer */}
        <div
          style={{
            padding: '12px 24px',
            background: E.panel,
            border: `1px dashed ${E.hair2}`,
            borderTop: `1px solid ${E.hair}`,
            fontSize: 11.5,
            color: E.text2,
            lineHeight: 1.5,
          }}
        >
          <span style={{ color: E.text3, fontFamily: E.fMono, fontSize: 10.5 }}>
            why this changed -&gt;
          </span>{' '}
          The CLI is a notebook, not a man page. Live shell-string preview
          teaches the syntax as you fill the form. Per-command run history
          turns this page into a lab journal, and co-pilot moves from a
          buried button to a permanent slot.
        </div>
      </div>
    </AppShell>
  );
}
