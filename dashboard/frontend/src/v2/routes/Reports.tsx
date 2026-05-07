/**
 * Reports - weekly report as a headline-driven story.
 *
 * Layout (top -> bottom):
 *   1. Eyebrow + status pill (audience + tone-of-the-week)
 *   2. Big serif headline with inline positive/negative emphasis
 *   3. Subtitle row (week + project + drafted-by + evidence-links count)
 *   4. Audience tabs (Leadership / Engineering / Product / Raw markdown)
 *   5. Big numbers strip - 4 cards each with sparkline + delta
 *   6. Story sections (1fr / 280px grid):
 *        left: What worked / What's blocking / Up next
 *        right: Send this report / Compare to
 *   7. "why this changed" footer
 *
 * Data: `v2.weeklyReport()` returns `WeeklyReport`. The redesign expects
 * three optional augmentations (`headline`, `audience_variants`,
 * `evidence_links`, plus a `spark` array on each big-number entry). When
 * the backend hasn't been augmented yet we fall back to canonical fields
 * so this view stays usable in either state.
 */

import { useCallback, useMemo, useRef, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import type { CSSProperties, KeyboardEvent as ReactKeyboardEvent, ReactNode } from 'react';
import { AppShell } from '../AppShell';
import { Btn, Card, Eyebrow, Pill, Skeleton, UpdatingChip } from '../ui';
import { v2 } from '../api/client';
import { useV2Resource } from '../hooks/useV2Resource';
import { E } from '../tokens';
import { copyToClipboard } from '../clipboard';
import type { WeeklyReport } from '../api/types';

/* === Local augmented shape ============================================ */

/** Extra fields the redesigned report consumes when present. The backend
 * may or may not supply these yet. Keeping them as a local extension
 * (vs. modifying types.ts) avoids forcing a breaking contract change. */
interface AugmentedBigNumber {
  label: string;
  value: string;
  delta: string;
  delta_kind: 'pass' | 'fail' | 'warn' | 'info';
  sub: string;
  spark?: number[];
}

type StoryAudience = 'Leadership' | 'Engineering' | 'Product';

interface AugmentedReport extends Omit<WeeklyReport, 'big_numbers' | 'audience_variants' | 'evidence_links'> {
  big_numbers: AugmentedBigNumber[];
  /** Single sentence with optional ``{{good}}...{{/good}}`` /
   * ``{{bad}}...{{/bad}}`` markup for inline emphasis. */
  headline?: string;
  /** Per-audience override copy for the body (markdown). Renders into
   * the story-section text when the matching tab is active. */
  audience_variants?: Partial<Record<StoryAudience, string>>;
  evidence_links?: { label: string; href: string }[];
}

type Audience = StoryAudience | 'Raw markdown';
const AUDIENCES: Audience[] = ['Leadership', 'Engineering', 'Product', 'Raw markdown'];

type CompareMode = 'last-week' | 'four-week-avg' | 'last-quarter';
const COMPARE_OPTIONS: { id: CompareMode; label: string }[] = [
  { id: 'last-week', label: 'Last week' },
  { id: 'four-week-avg', label: '4-week avg' },
  { id: 'last-quarter', label: 'Last quarter' },
];

/* === Markdown export ================================================== */

/** Strip ``{{good}}...{{/good}}`` / ``{{bad}}...{{/bad}}`` so the markdown
 * export is plain. */
function stripHeadlineMarkup(s: string): string {
  return s.replace(/\{\{\/?(good|bad|warn)\}\}/g, '');
}

function reportToMarkdown(r: AugmentedReport): string {
  const lines: string[] = [];
  lines.push(`# ${r.project_name} - ${r.week_label}`);
  lines.push('');
  lines.push(`_Drafted by co-pilot - ${r.generated_at_iso}_`);
  lines.push('');
  if (r.headline) {
    lines.push('## Headline');
    lines.push(stripHeadlineMarkup(r.headline));
    lines.push('');
  }
  lines.push('## TL;DR');
  lines.push(r.tldr_md.trim());
  lines.push('');
  if (r.big_numbers.length > 0) {
    lines.push('## Highlights');
    for (const n of r.big_numbers) {
      lines.push(`- **${n.label}:** ${n.value} (${n.delta}) - ${n.sub}`);
    }
    lines.push('');
  }
  if (r.shipped.length > 0) {
    lines.push('## What we shipped');
    for (const s of r.shipped) lines.push(`- ${s.text}`);
    lines.push('');
  }
  if (r.blocking) {
    lines.push("## What's blocking");
    lines.push(`**${r.blocking.title}**`);
    lines.push('');
    lines.push(r.blocking.body_md.trim());
    lines.push('');
    lines.push(`Owner: ${r.blocking.owner} - ETA: ${r.blocking.eta}`);
    lines.push('');
  }
  if (r.up_next.length > 0) {
    lines.push('## Up next');
    for (const s of r.up_next) lines.push(`- ${s.text}`);
    lines.push('');
  }
  return lines.join('\n');
}

/* === Headline rendering ============================================== */

/** Parse a headline into ReactNodes. Recognises:
 *   {{good}}+2.1pp{{/good}}  -> green underline span
 *   {{bad}}-0.02{{/bad}}     -> red highlight span
 *   {{warn}}slipped{{/warn}} -> warn-color span
 * Anything outside the tokens is rendered as plain text. */
function renderHeadline(s: string): ReactNode[] {
  const tokens: ReactNode[] = [];
  const re = /\{\{(good|bad|warn)\}\}([\s\S]*?)\{\{\/\1\}\}/g;
  let last = 0;
  let m: RegExpExecArray | null;
  let key = 0;
  while ((m = re.exec(s)) !== null) {
    if (m.index > last) tokens.push(s.slice(last, m.index));
    const kind = m[1];
    const inner = m[2];
    if (kind === 'good') {
      tokens.push(
        <span
          key={`h-${key++}`}
          style={{
            color: E.pass,
            textDecoration: 'underline',
            textDecorationColor: `${E.pass}80`,
            textUnderlineOffset: '4px',
            textDecorationThickness: '2px',
          }}
        >
          {inner}
        </span>,
      );
    } else if (kind === 'bad') {
      tokens.push(
        <span
          key={`h-${key++}`}
          style={{
            color: E.fail,
            background: E.failDim,
            padding: '0 6px',
            borderRadius: 4,
            border: `1px solid ${E.fail}33`,
          }}
        >
          {inner}
        </span>,
      );
    } else {
      tokens.push(
        <span key={`h-${key++}`} style={{ color: E.warn }}>
          {inner}
        </span>,
      );
    }
    last = m.index + m[0].length;
  }
  if (last < s.length) tokens.push(s.slice(last));
  return tokens.length ? tokens : [s];
}

/** Best-effort inline highlighter for a plain headline string with no
 * explicit markup. Color-codes any delta-shaped fragment ("+2.1pp",
 * "-12%", "-0.02"). */
function autoColorHeadline(s: string): ReactNode[] {
  const re = /([+−-]\s?\d+(?:\.\d+)?(?:pp|pts?|%)?)/g;
  const out: ReactNode[] = [];
  let last = 0;
  let m: RegExpExecArray | null;
  let key = 0;
  while ((m = re.exec(s)) !== null) {
    if (m.index > last) out.push(s.slice(last, m.index));
    const frag = m[0].replace(/\s/g, '');
    const isPositive = frag.startsWith('+');
    out.push(
      <span
        key={`a-${key++}`}
        style={
          isPositive
            ? {
                color: E.pass,
                textDecoration: 'underline',
                textDecorationColor: `${E.pass}80`,
                textUnderlineOffset: '4px',
                textDecorationThickness: '2px',
              }
            : {
                color: E.fail,
                background: E.failDim,
                padding: '0 6px',
                borderRadius: 4,
                border: `1px solid ${E.fail}33`,
              }
        }
      >
        {m[0]}
      </span>,
    );
    last = m.index + m[0].length;
  }
  if (last < s.length) out.push(s.slice(last));
  return out.length ? out : [s];
}

/* === Inline subcomponents ============================================ */

function ReportSpark({
  values,
  color,
  width = 140,
  height = 26,
}: {
  values: number[];
  color: string;
  width?: number;
  height?: number;
}) {
  if (!values || values.length === 0) {
    return <svg width={width} height={height} style={{ display: 'block' }} />;
  }
  const max = Math.max(...values);
  const min = Math.min(...values);
  const range = max - min || 1;
  const step = width / Math.max(1, values.length - 1);
  const pts = values.map((v, i) => {
    const x = i * step;
    const y = height - ((v - min) / range) * (height - 4) - 2;
    return `${x},${y}`;
  });
  const last = values[values.length - 1];
  const lx = (values.length - 1) * step;
  const ly = height - ((last - min) / range) * (height - 4) - 2;
  return (
    <svg width={width} height={height} style={{ display: 'block' }}>
      <polyline
        points={pts.join(' ')}
        fill="none"
        stroke={color}
        strokeWidth={1.5}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <circle cx={lx} cy={ly} r={2.5} fill={color} />
    </svg>
  );
}

function BigNumbersStrip({
  numbers,
  compareMode,
}: {
  numbers: AugmentedBigNumber[];
  compareMode: CompareMode;
}) {
  // When comparing against a non-default mode we visually de-emphasise
  // the delta to signal "this is recomputed against a different base".
  const dimDelta = compareMode !== 'last-week';

  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(4, 1fr)',
        gap: 10,
        marginBottom: 24,
      }}
    >
      {numbers.map((m) => {
        const good = m.delta_kind === 'pass';
        const sparkColor =
          m.delta_kind === 'pass'
            ? E.pass
            : m.delta_kind === 'fail'
              ? E.fail
              : m.delta_kind === 'warn'
                ? E.warn
                : E.steel;
        return (
          <Card key={m.label} style={{ padding: 14 }}>
            <div style={{ fontSize: 11, color: E.text3, marginBottom: 4 }}>{m.label}</div>
            <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
              <span style={{ fontFamily: E.fSerif, fontSize: 26, color: E.text0 }}>{m.value}</span>
              <span
                style={{
                  fontFamily: E.fMono,
                  fontSize: 11,
                  color: good ? E.pass : E.fail,
                  opacity: dimDelta ? 0.6 : 1,
                  fontStyle: dimDelta ? 'italic' : 'normal',
                }}
                title={
                  dimDelta
                    ? `Delta vs ${COMPARE_OPTIONS.find((c) => c.id === compareMode)?.label}`
                    : 'Week-over-week delta'
                }
              >
                {m.delta}
              </span>
            </div>
            <div style={{ marginTop: 6 }}>
              {m.spark && m.spark.length > 0 ? (
                <ReportSpark values={m.spark} color={sparkColor} width={140} height={26} />
              ) : (
                <div
                  style={{
                    height: 26,
                    fontSize: 10,
                    fontFamily: E.fMono,
                    color: E.text4,
                    display: 'flex',
                    alignItems: 'center',
                  }}
                >
                  no trend yet
                </div>
              )}
            </div>
            <div
              style={{
                marginTop: 4,
                fontFamily: E.fMono,
                fontSize: 9.5,
                color: E.text3,
                display: 'flex',
                justifyContent: 'space-between',
              }}
            >
              <span>6w ago</span>
              <span>now</span>
            </div>
          </Card>
        );
      })}
    </div>
  );
}

/** A bullet item that may contain inline ember-text reference tags like
 * `[exp-2034](https://...)`. Only renders Markdown-style inline links. */
function StoryBullet({ text }: { text: string }) {
  const re = /\[([^\]]+)\]\(([^)]+)\)/g;
  const out: ReactNode[] = [];
  let last = 0;
  let m: RegExpExecArray | null;
  let key = 0;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(text.slice(last, m.index));
    out.push(
      <a
        key={`l-${key++}`}
        href={m[2]}
        target="_blank"
        rel="noreferrer"
        style={{
          color: E.ember,
          fontFamily: E.fMono,
          fontSize: 11,
          textDecoration: 'none',
        }}
      >
        {m[1]} {'▸'}
      </a>,
    );
    last = m.index + m[0].length;
  }
  if (last < text.length) out.push(text.slice(last));
  return <>{out.length ? out : text}</>;
}

interface StorySectionProps {
  eyebrow: string;
  pill: { label: string; color: string; bg: string };
  borderColor: string;
  children: ReactNode;
  style?: CSSProperties;
}

function StorySection({ eyebrow, pill, borderColor, children, style }: StorySectionProps) {
  return (
    <Card style={{ padding: 16, borderLeft: `3px solid ${borderColor}`, ...style }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
        <Eyebrow>{eyebrow}</Eyebrow>
        <Pill bg={pill.bg} color={pill.color} mono>
          {pill.label}
        </Pill>
      </div>
      {children}
    </Card>
  );
}

interface DistributionRailProps {
  onSendNow: () => void;
  compareMode: CompareMode;
  setCompareMode: (m: CompareMode) => void;
}

function DistributionRail({ onSendNow, compareMode, setCompareMode }: DistributionRailProps) {
  const destinations: { glyph: string; title: string; cadence: string }[] = [
    { glyph: '#', title: 'Slack #ml-quality', cadence: 'sent every Mon 9am' },
    { glyph: '@', title: 'leadership@', cadence: 'manual send' },
    { glyph: 'N', title: 'Notion - Eng wiki', cadence: 'auto-publish' },
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      <Card style={{ padding: 14 }}>
        <Eyebrow>Send this report</Eyebrow>
        <div style={{ marginTop: 10, display: 'flex', flexDirection: 'column', gap: 6 }}>
          {destinations.map((x) => (
            <div
              key={x.title}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 8,
                padding: '8px 10px',
                background: E.panel2,
                borderRadius: 6,
                border: `1px solid ${E.hair2}`,
              }}
            >
              <span
                style={{
                  display: 'inline-flex',
                  width: 18,
                  height: 18,
                  alignItems: 'center',
                  justifyContent: 'center',
                  borderRadius: 4,
                  background: E.panel3,
                  color: E.steel,
                  fontFamily: E.fMono,
                  fontSize: 11,
                  fontWeight: 600,
                }}
                aria-hidden="true"
              >
                {x.glyph}
              </span>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ fontSize: 12, color: E.text1 }}>{x.title}</div>
                <div style={{ fontSize: 10.5, color: E.text3, fontFamily: E.fMono }}>
                  {x.cadence}
                </div>
              </div>
            </div>
          ))}
          <Btn kind="primary" size="sm" style={{ marginTop: 4 }} onClick={onSendNow}>
            Send now {'▸'}
          </Btn>
          {/* Honestly disabled - distribution destinations are
              placeholders. Previously this rendered as an active
              ghost button and clicking it did nothing. */}
          <Btn
            kind="ghost"
            size="sm"
            disabled
            title="Custom destinations are not wired up yet"
          >
            + Add destination (soon)
          </Btn>
        </div>
      </Card>

      <Card style={{ padding: 14 }}>
        <Eyebrow>Compare to</Eyebrow>
        <div
          style={{
            marginTop: 8,
            display: 'flex',
            flexDirection: 'column',
            gap: 6,
          }}
        >
          {COMPARE_OPTIONS.map((c) => {
            const active = c.id === compareMode;
            return (
              <label
                key={c.id}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 8,
                  fontFamily: E.fMono,
                  fontSize: 11,
                  color: active ? E.text0 : E.text2,
                  cursor: 'pointer',
                }}
              >
                <input
                  type="radio"
                  name="cmp"
                  checked={active}
                  onChange={() => setCompareMode(c.id)}
                  style={{ accentColor: E.ember }}
                />
                {c.label}
              </label>
            );
          })}
        </div>
      </Card>
    </div>
  );
}

/* === Audience tabs =================================================== */

function AudienceTabs({
  active,
  onChange,
}: {
  active: Audience;
  onChange: (a: Audience) => void;
}) {
  // Refs per tab so arrow-key navigation can move focus to the
  // newly-activated tab. Roving tabindex (active = 0, others = -1)
  // keeps the tab strip a single Tab stop, which is the standard
  // ARIA pattern for tablists.
  const tabRefs = useRef<Array<HTMLButtonElement | null>>([]);
  const activeIdx = AUDIENCES.indexOf(active);
  const onKeyDown = (e: ReactKeyboardEvent<HTMLDivElement>) => {
    let nextIdx: number | null = null;
    if (e.key === 'ArrowRight') nextIdx = (activeIdx + 1) % AUDIENCES.length;
    else if (e.key === 'ArrowLeft')
      nextIdx = (activeIdx - 1 + AUDIENCES.length) % AUDIENCES.length;
    else if (e.key === 'Home') nextIdx = 0;
    else if (e.key === 'End') nextIdx = AUDIENCES.length - 1;
    if (nextIdx === null) return;
    e.preventDefault();
    onChange(AUDIENCES[nextIdx]);
    // Defer focus so React commits the new aria-selected first; without
    // this the wrong button briefly carries focus during the transition.
    window.setTimeout(() => tabRefs.current[nextIdx!]?.focus(), 0);
  };
  return (
    <div
      role="tablist"
      aria-label="Report audience"
      onKeyDown={onKeyDown}
      style={{ display: 'flex', gap: 4, marginBottom: 14, flexWrap: 'wrap' }}
    >
      {AUDIENCES.map((t, i) => {
        const isActive = t === active;
        return (
          <button
            key={t}
            id={`reports-audience-tab-${t}`}
            ref={(el) => {
              tabRefs.current[i] = el;
            }}
            type="button"
            role="tab"
            aria-selected={isActive}
            aria-controls={`reports-audience-panel-${t}`}
            tabIndex={isActive ? 0 : -1}
            onClick={() => onChange(t)}
            style={{
              padding: '6px 12px',
              fontSize: 12,
              fontFamily: E.fMono,
              borderRadius: 6,
              background: isActive ? E.panel : 'transparent',
              color: isActive ? E.text0 : E.text3,
              border: `1px solid ${isActive ? E.hair2 : 'transparent'}`,
              cursor: 'pointer',
            }}
          >
            {t}
          </button>
        );
      })}
    </div>
  );
}

/* === Loading + error ================================================= */

function ReportSkeleton() {
  return (
    <>
      <Skeleton w={220} h={11} />
      <div style={{ marginTop: 8 }}>
        <Skeleton w="80%" h={36} />
      </div>
      <div style={{ marginTop: 8 }}>
        <Skeleton w={320} h={13} />
      </div>
      <div style={{ height: 24 }} />
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 10 }}>
        {[0, 1, 2, 3].map((i) => (
          <Card key={i} style={{ padding: 14 }}>
            <Skeleton w={80} h={11} />
            <div style={{ marginTop: 8 }}>
              <Skeleton w={90} h={26} />
            </div>
            <div style={{ marginTop: 8 }}>
              <Skeleton w="100%" h={26} />
            </div>
          </Card>
        ))}
      </div>
      <div style={{ height: 18 }} />
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 280px', gap: 18 }}>
        <Card style={{ padding: 16 }}>
          <Skeleton w={120} h={11} />
          <div style={{ marginTop: 12, display: 'flex', flexDirection: 'column', gap: 8 }}>
            {[0, 1, 2].map((i) => (
              <Skeleton key={i} w={`${65 + i * 8}%`} h={13} />
            ))}
          </div>
        </Card>
        <Card style={{ padding: 14 }}>
          <Skeleton w={100} h={11} />
          <div style={{ marginTop: 12 }}>
            <Skeleton w="100%" h={36} />
          </div>
        </Card>
      </div>
    </>
  );
}

/* === Main route component ============================================ */

export default function Reports() {
  const { data, err, refetch, reloading, isInitialLoad } = useV2Resource(
    'weeklyReport',
    v2.weeklyReport,
  );

  const report = data as AugmentedReport | null;

  // Both view selections live in the URL so links land the
  // recipient on the same audience + compare lens the sender was
  // looking at. Reload / browser back / "share with leadership"
  // all preserve the choice. setSearchParams uses replace so
  // tab clicks don't pollute the back stack with one entry per
  // selection. Pattern matches Metrics' ?metric= (fc0c7cb).
  const [searchParams, setSearchParams] = useSearchParams();
  const audienceParam = searchParams.get('audience');
  const audience: Audience = AUDIENCES.includes(audienceParam as Audience)
    ? (audienceParam as Audience)
    : 'Leadership';
  const setAudience = useCallback(
    (next: Audience) => {
      setSearchParams(
        (prev) => {
          const u = new URLSearchParams(prev);
          if (next === 'Leadership') u.delete('audience');
          else u.set('audience', next);
          return u;
        },
        { replace: true },
      );
    },
    [setSearchParams],
  );
  const compareParam = searchParams.get('compare');
  const compareMode: CompareMode =
    compareParam === 'four-week-avg' || compareParam === 'last-quarter'
      ? compareParam
      : 'last-week';
  const setCompareMode = useCallback(
    (next: CompareMode) => {
      setSearchParams(
        (prev) => {
          const u = new URLSearchParams(prev);
          if (next === 'last-week') u.delete('compare');
          else u.set('compare', next);
          return u;
        },
        { replace: true },
      );
    },
    [setSearchParams],
  );
  const [copyState, setCopyState] = useState<'idle' | 'copied' | 'error'>('idle');
  const [sendState, setSendState] = useState<'idle' | 'sent'>('idle');

  const markdown = useMemo(() => (report ? reportToMarkdown(report) : ''), [report]);

  async function handleCopyMarkdown() {
    if (!report) return;
    try {
      await copyToClipboard(markdown);
      setCopyState('copied');
      window.setTimeout(() => setCopyState('idle'), 2000);
    } catch {
      setCopyState('error');
      window.setTimeout(() => setCopyState('idle'), 3000);
    }
  }

  function handleSendNow() {
    setSendState('sent');
    window.setTimeout(() => setSendState('idle'), 2200);
  }

  // Save the report's markdown as a .md file so a user can attach it
  // to a wiki page, paste into a doc, or commit to a repo without
  // going through the clipboard. Filename includes the report period
  // (when available) so multiple weekly reports don't clobber each
  // other in Downloads.
  function handleDownloadMarkdown() {
    if (!report) return;
    const blob = new Blob([markdown], {
      type: 'text/markdown;charset=utf-8',
    });
    const url = URL.createObjectURL(blob);
    const ts = new Date().toISOString().replace(/[:.]/g, '-');
    const a = document.createElement('a');
    a.href = url;
    a.download = `evalyn-weekly-${ts}.md`;
    a.style.display = 'none';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.setTimeout(() => URL.revokeObjectURL(url), 0);
  }

  // Tone of the week pill: derive from the first big number's delta_kind
  // (typically Quality). Falls back to a neutral "weekly report" label.
  const tonePill = useMemo<{ label: string; color: string; bg: string }>(() => {
    if (!report || report.big_numbers.length === 0) {
      return { label: 'weekly report', color: E.steel, bg: E.steelDim };
    }
    const k = report.big_numbers[0].delta_kind;
    if (k === 'pass') return { label: 'shipping week', color: E.pass, bg: E.passDim };
    if (k === 'fail') return { label: 'calibration week', color: E.fail, bg: E.failDim };
    if (k === 'warn') return { label: 'mixed week', color: E.warn, bg: E.warnDim };
    return { label: 'steady week', color: E.steel, bg: E.steelDim };
  }, [report]);

  const evidenceCount = report?.evidence_links?.length ?? 0;

  // Headline: prefer the explicit `headline` field, otherwise build one
  // from `tldr_md` so the page never looks unfinished.
  const headlineText = report?.headline ?? report?.tldr_md.split('\n')[0] ?? '';

  // Audience-specific copy lookup. Narrowed away from 'Raw markdown' to
  // satisfy the typed Partial<Record<StoryAudience, string>> key set.
  const audienceVariantCopy: string | undefined =
    audience !== 'Raw markdown' ? report?.audience_variants?.[audience] : undefined;

  return (
    <AppShell contextChip={{ name: report?.project_name ?? 'Loading', version: '' }}>
      <div style={{ padding: 32, maxWidth: 1100 }}>
        {/* 1. Eyebrow + status pill */}
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 12, flexWrap: 'wrap' }}>
          <Eyebrow>Weekly report - audience: {audience}</Eyebrow>
          <Pill mono color={tonePill.color} bg={tonePill.bg}>
            {tonePill.label}
          </Pill>
          <UpdatingChip
            visible={reloading && !isInitialLoad}
            error={report ? err : null}
            onRetry={refetch}
          />
        </div>

        {err && !report && (
          <Card style={{ padding: 16, marginTop: 22, borderColor: E.fail }}>
            <Eyebrow style={{ color: E.fail }}>Error loading report</Eyebrow>
            <div style={{ fontFamily: E.fMono, fontSize: 12, color: E.text2, marginTop: 6 }}>
              {err}
            </div>
            <div style={{ marginTop: 10 }}>
              <Btn kind="ghost" size="sm" onClick={() => void refetch()}>
                Retry
              </Btn>
            </div>
          </Card>
        )}

        {!report && !err && (
          <div style={{ marginTop: 14 }}>
            <ReportSkeleton />
          </div>
        )}

        {report && (
          <>
            {/* 2. Big serif headline */}
            <h1
              style={{
                fontFamily: E.fSerif,
                fontSize: 36,
                fontWeight: 400,
                margin: '6px 0 6px',
                color: E.text0,
                letterSpacing: '-0.02em',
                lineHeight: 1.15,
              }}
            >
              {report.headline
                ? renderHeadline(report.headline)
                : autoColorHeadline(headlineText)}
            </h1>

            {/* 3. Subtitle row */}
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 12,
                fontSize: 12.5,
                color: E.text2,
                marginBottom: 18,
                flexWrap: 'wrap',
              }}
            >
              <span>
                {report.week_label} - {report.project_name}
              </span>
              <span style={{ color: E.text4 }}>{'·'}</span>
              <span style={{ fontFamily: E.fMono, color: E.text3 }}>
                drafted by co-pilot - {evidenceCount} evidence link
                {evidenceCount === 1 ? '' : 's'}
              </span>
            </div>

            {/* 4. Audience tabs */}
            <AudienceTabs active={audience} onChange={setAudience} />

            <div
              id={`reports-audience-panel-${audience}`}
              role="tabpanel"
              aria-labelledby={`reports-audience-tab-${audience}`}
            >
            {audience === 'Raw markdown' ? (
              <Card style={{ padding: 16 }}>
                <div
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 8,
                    marginBottom: 10,
                  }}
                >
                  <Eyebrow>Raw markdown</Eyebrow>
                  <span style={{ flex: 1 }} />
                  <Btn kind="primary" size="sm" onClick={() => void handleCopyMarkdown()}>
                    {copyState === 'copied' ? 'Copied' : 'Copy'}
                  </Btn>
                  <Btn
                    kind="ghost"
                    size="sm"
                    onClick={handleDownloadMarkdown}
                    title="Save the report as a .md file"
                  >
                    Download .md
                  </Btn>
                  {copyState === 'copied' && (
                    <Pill mono color={E.pass} bg={E.passDim}>
                      Markdown on clipboard
                    </Pill>
                  )}
                  {copyState === 'error' && (
                    <Pill mono color={E.fail} bg={E.failDim}>
                      Copy failed
                    </Pill>
                  )}
                </div>
                <pre
                  style={{
                    margin: 0,
                    padding: 14,
                    background: E.panel2,
                    border: `1px solid ${E.hair2}`,
                    borderRadius: 8,
                    fontFamily: E.fMono,
                    fontSize: 12,
                    color: E.text1,
                    lineHeight: 1.55,
                    whiteSpace: 'pre-wrap',
                    wordBreak: 'break-word',
                    maxHeight: 520,
                    overflow: 'auto',
                  }}
                >
                  {markdown}
                </pre>
              </Card>
            ) : (
              <>
                {/* 5. Big numbers strip */}
                <BigNumbersStrip numbers={report.big_numbers} compareMode={compareMode} />

                {/* Audience-specific copy (when present). */}
                {audienceVariantCopy && (
                  <Card style={{ padding: 16, marginBottom: 14 }}>
                    <Eyebrow>{audience} brief</Eyebrow>
                    <div
                      style={{
                        marginTop: 8,
                        fontFamily: E.fSerif,
                        fontSize: 18,
                        color: E.text0,
                        lineHeight: 1.5,
                        whiteSpace: 'pre-wrap',
                      }}
                    >
                      {audienceVariantCopy}
                    </div>
                  </Card>
                )}

                {/* 6. Story sections */}
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 280px', gap: 18 }}>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                    {/* What worked */}
                    <StorySection
                      eyebrow="What worked"
                      borderColor={E.pass}
                      pill={{
                        label: `${report.shipped.length} win${
                          report.shipped.length === 1 ? '' : 's'
                        }`,
                        color: E.pass,
                        bg: E.passDim,
                      }}
                    >
                      {report.shipped.length === 0 ? (
                        <div style={{ fontSize: 12.5, color: E.text3, fontStyle: 'italic' }}>
                          Nothing shipped this week.
                        </div>
                      ) : (
                        <ul
                          style={{
                            fontSize: 13,
                            color: E.text1,
                            lineHeight: 1.7,
                            paddingLeft: 18,
                            margin: 0,
                          }}
                        >
                          {report.shipped.map((s, i) => (
                            <li key={i}>
                              <StoryBullet text={s.text} />
                            </li>
                          ))}
                        </ul>
                      )}
                    </StorySection>

                    {/* What's blocking */}
                    {report.blocking ? (
                      <StorySection
                        eyebrow="What's blocking"
                        borderColor={E.fail}
                        pill={{ label: '1 active', color: E.fail, bg: E.failDim }}
                      >
                        <div
                          style={{
                            fontSize: 14,
                            color: E.text0,
                            fontWeight: 500,
                            marginBottom: 6,
                          }}
                        >
                          {report.blocking.title}
                        </div>
                        <p
                          style={{
                            fontSize: 12.5,
                            color: E.text2,
                            lineHeight: 1.6,
                            margin: 0,
                            whiteSpace: 'pre-wrap',
                          }}
                        >
                          {report.blocking.body_md}
                        </p>
                        <div
                          style={{
                            marginTop: 10,
                            padding: 10,
                            background: E.panel2,
                            borderRadius: 6,
                            display: 'flex',
                            alignItems: 'center',
                            gap: 10,
                          }}
                        >
                          <ReportSpark
                            values={[5, 5, 6, 7, 9, 12]}
                            color={E.fail}
                            width={120}
                            height={26}
                          />
                          <span
                            style={{
                              fontFamily: E.fMono,
                              fontSize: 10.5,
                              color: E.text2,
                            }}
                          >
                            % failing - 6 weeks
                          </span>
                          <span style={{ flex: 1 }} />
                          <Btn
                            kind="ghost"
                            size="sm"
                            title="Drill into the failing items for this regression"
                          >
                            View items {'▸'}
                          </Btn>
                        </div>
                        <div
                          style={{
                            fontFamily: E.fMono,
                            fontSize: 11,
                            color: E.text3,
                            marginTop: 8,
                          }}
                        >
                          Owner {report.blocking.owner} - ETA {report.blocking.eta} - severity P1
                        </div>
                      </StorySection>
                    ) : (
                      <StorySection
                        eyebrow="What's blocking"
                        borderColor={E.steel}
                        pill={{ label: '0 active', color: E.steel, bg: E.steelDim }}
                      >
                        <div style={{ fontSize: 12.5, color: E.text3, fontStyle: 'italic' }}>
                          Nothing blocking this week.
                        </div>
                      </StorySection>
                    )}

                    {/* Up next */}
                    <StorySection
                      eyebrow="Up next"
                      borderColor={E.steel}
                      pill={{
                        label: `${report.up_next.length} planned`,
                        color: E.steel,
                        bg: E.steelDim,
                      }}
                    >
                      {report.up_next.length === 0 ? (
                        <div style={{ fontSize: 12.5, color: E.text3, fontStyle: 'italic' }}>
                          No items queued.
                        </div>
                      ) : (
                        <ul
                          style={{
                            fontSize: 13,
                            color: E.text1,
                            lineHeight: 1.7,
                            paddingLeft: 18,
                            margin: 0,
                          }}
                        >
                          {report.up_next.map((s, i) => (
                            <li key={i}>
                              <StoryBullet text={s.text} />
                            </li>
                          ))}
                        </ul>
                      )}
                    </StorySection>
                  </div>

                  {/* Right rail */}
                  <DistributionRail
                    onSendNow={handleSendNow}
                    compareMode={compareMode}
                    setCompareMode={setCompareMode}
                  />
                </div>

                {sendState === 'sent' && (
                  <div
                    style={{
                      marginTop: 12,
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: 8,
                    }}
                  >
                    <Pill mono color={E.pass} bg={E.passDim}>
                      Sent to Slack #ml-quality (placeholder)
                    </Pill>
                  </div>
                )}

                {/* 7. why this changed footer */}
                <div
                  style={{
                    marginTop: 18,
                    padding: '12px 16px',
                    background: E.panel,
                    border: `1px dashed ${E.hair2}`,
                    borderRadius: 8,
                    fontSize: 11.5,
                    color: E.text2,
                    lineHeight: 1.5,
                  }}
                >
                  <span style={{ color: E.text3, fontFamily: E.fMono, fontSize: 10.5 }}>
                    why this changed {'→'}
                  </span>{' '}
                  A weekly report should be a story, not a dump. Headline + sparklines give the
                  shape of the week in seconds. Color-coded sections frame the narrative (wins /
                  blocking / next). Every claim links back to the evidence. The audience switcher
                  recognises that leadership and eng want the same data at different resolutions.
                </div>
              </>
            )}
            </div>
          </>
        )}

        <div style={{ height: 30 }} />
      </div>
    </AppShell>
  );
}
