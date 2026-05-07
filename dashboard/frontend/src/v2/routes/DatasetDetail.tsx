/**
 * DatasetDetail - per-dataset deep view.
 *
 * URL: ``/datasets/:name``. Renders what's IN the dataset (items preview,
 * coverage, tags) plus how it's been used (recent runs, observed metrics).
 *
 * Primary CTA: "Run new evaluation" deep-links to /commands with the
 * dataset prefilled. Lane I owns the runner UX downstream; we just hand
 * the URL contract over.
 *
 * Cache key: ``dataset:<name>`` so navigating Datasets list -> card ->
 * back -> same card is instant (useV2Resource module-level cache).
 */

import { useCallback, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { AppShell } from '../AppShell';
import { Btn, Card, Eyebrow, Pill, Skeleton, StackBar, StatusDot, UpdatingChip } from '../ui';
import { v2 } from '../api/client';
import type { DatasetDetail as DatasetDetailT } from '../api/types';
import { useV2Resource } from '../hooks/useV2Resource';
import { useProject } from '../hooks/useProject';
import { copyToClipboard } from '../clipboard';
import { E } from '../tokens';

const COVERAGE_COLORS = [E.ember, E.steel, '#a78bfa', E.warn, E.text3];

/** Render the dataset's summary metadata as portable markdown. The
 * actual item rows aren't included - this is the "tell a coworker about
 * this dataset" view, not the full export. For full data dump, the CLI
 * has `evalyn export` and `evalyn export-for-annotation`. */
function datasetToMarkdown(d: DatasetDetailT): string {
  const lines: string[] = [];
  lines.push(`# ${d.name}`);
  lines.push('');
  const meta: string[] = [`${d.n} item${d.n === 1 ? '' : 's'}`];
  if (d.source) meta.push(`source: ${d.source}`);
  if (d.created_at_iso) meta.push(`created ${d.created_at_iso}`);
  lines.push(meta.join(' - '));
  lines.push('');
  if (d.tags.length > 0) {
    lines.push(`**Tags:** ${d.tags.join(', ')}`);
    lines.push('');
  }
  if (d.coverage.length > 0) {
    lines.push('## Coverage');
    for (const c of d.coverage) lines.push(`- ${c.label}: ${c.value}`);
    lines.push('');
  }
  if (d.observed_metrics.length > 0) {
    lines.push('## Metrics observed across runs');
    for (const m of d.observed_metrics) {
      lines.push(`- \`${m.id}\` (${m.kind}) - ${m.uses}x`);
    }
    lines.push('');
  }
  if (d.recent_runs.length > 0) {
    lines.push(`## Recent runs (${d.recent_runs.length})`);
    lines.push('');
    lines.push('| Run | Status | Pass | Cost | When |');
    lines.push('|---|---|---|---|---|');
    for (const r of d.recent_runs) {
      const pass = r.pass != null ? `${(r.pass * 100).toFixed(1)}%` : '-';
      lines.push(`| \`${r.id}\` | ${r.status} | ${pass} | ${r.cost} | ${r.created_at_iso} |`);
    }
    lines.push('');
  }
  return lines.join('\n');
}


function formatWhen(iso: string | null): string {
  if (!iso) return '-';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}

function truncate(s: string | null | undefined, max = 140): string {
  if (!s) return '';
  return s.length > max ? s.slice(0, max) + '…' : s;
}

export default function DatasetDetail() {
  const { name: rawName } = useParams<{ name: string }>();
  const name = rawName ?? '';
  const navigate = useNavigate();
  const project = useProject();

  const fetcher = useCallback(() => v2.dataset(name), [name]);
  const { data, err, refetch, reloading, isInitialLoad } = useV2Resource<DatasetDetailT>(
    `dataset:${name}`,
    fetcher,
  );
  const [exportState, setExportState] = useState<'idle' | 'copied' | 'error'>('idle');

  async function handleExport() {
    if (!data) return;
    try {
      await copyToClipboard(datasetToMarkdown(data));
      setExportState('copied');
      window.setTimeout(() => setExportState('idle'), 2000);
    } catch {
      setExportState('error');
      window.setTimeout(() => setExportState('idle'), 3000);
    }
  }

  const goRunNewEval = () => {
    const target = `/commands?prefill=run-eval&dataset=${encodeURIComponent(name)}`;
    navigate(target);
  };
  const goRun = (runId: string) => {
    navigate(`/experiments/${encodeURIComponent(runId)}`);
  };

  // Error state - server returned 404 or any other non-2xx.
  if (err && !data) {
    return (
      <AppShell contextChip={project ?? undefined} breadcrumb={['Datasets', name]}>
        <div style={{ padding: '32px 36px', maxWidth: 720 }}>
          <Card style={{ padding: 22, borderColor: E.fail }}>
            <Eyebrow style={{ color: E.fail }}>Dataset not found</Eyebrow>
            <div
              style={{
                fontFamily: E.fMono,
                fontSize: 12,
                color: E.text2,
                marginTop: 8,
                wordBreak: 'break-word',
              }}
            >
              {err}
            </div>
            <div style={{ marginTop: 14 }}>
              <Btn kind="secondary" size="md" onClick={() => navigate('/datasets')}>
                ← Back to datasets
              </Btn>
            </div>
          </Card>
        </div>
      </AppShell>
    );
  }

  // Loading skeleton.
  if (!data) {
    return (
      <AppShell contextChip={project ?? undefined} breadcrumb={['Datasets', name]}>
        <div style={{ padding: '32px 36px' }}>
          <Skeleton w={140} h={11} />
          <div style={{ marginTop: 8 }}>
            <Skeleton w={360} h={32} />
          </div>
          <div style={{ marginTop: 12 }}>
            <Skeleton w={280} h={13} />
          </div>
          <div style={{ marginTop: 22, display: 'grid', gridTemplateColumns: '1.4fr 1fr', gap: 14 }}>
            <Card style={{ padding: 18 }}>
              <Skeleton w={120} h={11} />
              <div style={{ marginTop: 14 }}>
                <Skeleton w="100%" h={180} style={{ borderRadius: 6 }} />
              </div>
            </Card>
            <Card style={{ padding: 18 }}>
              <Skeleton w={120} h={11} />
              <div style={{ marginTop: 14 }}>
                <Skeleton w="100%" h={180} style={{ borderRadius: 6 }} />
              </div>
            </Card>
          </div>
        </div>
      </AppShell>
    );
  }

  const segments = data.coverage.map((c, i) => ({
    value: c.value,
    color: COVERAGE_COLORS[i % COVERAGE_COLORS.length],
    label: `${c.label}: ${c.value}`,
  }));
  const totalRuns = data.recent_runs.length;
  const lastUsed = data.recent_runs[0]?.created_at_iso ?? null;
  const subtitleBits: string[] = [
    `${data.n} item${data.n === 1 ? '' : 's'}`,
    `${totalRuns} eval run${totalRuns === 1 ? '' : 's'}`,
  ];
  if (lastUsed) {
    subtitleBits.push(`last used ${formatWhen(lastUsed)}`);
  }

  return (
    <AppShell contextChip={project ?? undefined} breadcrumb={['Datasets', name]}>
      <div style={{ padding: '32px 36px', maxWidth: 1180 }}>
        {/* HEADER */}
        <div style={{ display: 'flex', alignItems: 'flex-end', gap: 16 }}>
          <div style={{ minWidth: 0, flex: 1 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <Eyebrow>Dataset</Eyebrow>
              <UpdatingChip
                visible={reloading && !isInitialLoad}
                error={data ? err : null}
                onRetry={refetch}
              />
            </div>
            <h1
              style={{
                fontFamily: E.fMono,
                fontSize: 28,
                fontWeight: 500,
                margin: '4px 0 0 0',
                color: E.text0,
                letterSpacing: '-0.01em',
                wordBreak: 'break-word',
              }}
            >
              {data.name}
            </h1>
            <p style={{ fontSize: 13, color: E.text2, marginTop: 6 }}>
              {subtitleBits.join(' · ')}
              {data.source ? ` · ${data.source}` : ''}
            </p>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexShrink: 0 }}>
            <Btn
              kind="secondary"
              size="md"
              onClick={() => void handleExport()}
              title={
                exportState === 'copied'
                  ? 'Dataset summary on clipboard'
                  : exportState === 'error'
                    ? 'Browser blocked clipboard access'
                    : 'Copy this dataset (tags, coverage, observed metrics, recent runs) as markdown'
              }
            >
              {exportState === 'copied'
                ? '✓ Copied'
                : exportState === 'error'
                  ? '✗ Failed'
                  : 'Copy summary'}
            </Btn>
            <Btn kind="primary" size="md" onClick={goRunNewEval}>
              Run new evaluation →
            </Btn>
          </div>
        </div>

        {/* TAGS + COVERAGE */}
        <Card style={{ padding: 18, marginTop: 22 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
            <Eyebrow>Tags</Eyebrow>
            {data.tags.length === 0 && (
              <span style={{ fontSize: 12, color: E.text3, fontFamily: E.fMono }}>none</span>
            )}
            {data.tags.map((t) => (
              <Pill
                key={t}
                mono
                style={{ fontSize: 10, background: E.panel2, color: E.text2 }}
              >
                {t}
              </Pill>
            ))}
          </div>
          {data.coverage.length > 0 ? (
            <>
              <Eyebrow style={{ marginTop: 16, marginBottom: 8 }}>Coverage</Eyebrow>
              <StackBar segments={segments} w="100%" h={8} />
              <div
                style={{
                  display: 'flex',
                  flexWrap: 'wrap',
                  gap: 12,
                  marginTop: 10,
                  fontSize: 11,
                  color: E.text2,
                  fontFamily: E.fMono,
                }}
              >
                {data.coverage.map((c, i) => (
                  <span
                    key={c.label}
                    style={{ display: 'inline-flex', alignItems: 'center', gap: 5 }}
                  >
                    <span
                      style={{
                        width: 7,
                        height: 7,
                        borderRadius: 1.5,
                        background: COVERAGE_COLORS[i % COVERAGE_COLORS.length],
                      }}
                    />
                    {c.label} {c.value}
                  </span>
                ))}
              </div>
            </>
          ) : (
            <div style={{ marginTop: 14, fontSize: 12, color: E.text3 }}>
              No category coverage on this dataset.
            </div>
          )}
        </Card>

        {/* TWO-COL: RECENT RUNS + OBSERVED METRICS */}
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: '1.5fr 1fr',
            gap: 14,
            marginTop: 14,
          }}
        >
          {/* RECENT RUNS */}
          <Card style={{ padding: 0, overflow: 'hidden' }}>
            <div
              style={{
                padding: '14px 18px',
                borderBottom: `1px solid ${E.hair}`,
                display: 'flex',
                alignItems: 'center',
              }}
            >
              <Eyebrow>Recent runs</Eyebrow>
              <span style={{ flex: 1 }} />
              <span style={{ fontFamily: E.fMono, fontSize: 10, color: E.text3 }}>
                {totalRuns}
              </span>
            </div>
            {totalRuns === 0 ? (
              <div style={{ padding: 22, textAlign: 'center', fontSize: 13, color: E.text3 }}>
                No evals on this dataset yet. Click <strong>Run new evaluation</strong> above to
                create one.
              </div>
            ) : (
              <div>
                {data.recent_runs.map((r, i) => (
                  <div
                    key={r.id}
                    onClick={() => goRun(r.id)}
                    role="button"
                    tabIndex={0}
                    aria-label={`Open run ${r.id}, ${r.status}${r.pass != null ? `, ${r.pass.toFixed(1)}% pass` : ''}`}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault();
                        goRun(r.id);
                      }
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.background = E.panel2;
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.background = 'transparent';
                    }}
                    style={{
                      display: 'grid',
                      gridTemplateColumns: '1fr 90px 70px 80px 80px',
                      alignItems: 'center',
                      gap: 10,
                      padding: '11px 18px',
                      borderTop: i ? `1px solid ${E.hair}` : 'none',
                      cursor: 'pointer',
                      fontSize: 12,
                      transition: 'background 140ms',
                    }}
                  >
                    <div
                      title={r.id}
                      style={{
                        fontFamily: E.fMono,
                        fontSize: 11.5,
                        color: E.text1,
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                      }}
                    >
                      {r.id}
                    </div>
                    <div style={{ color: E.text2 }}>{formatWhen(r.created_at_iso)}</div>
                    <div
                      style={{
                        fontFamily: E.fMono,
                        color: r.pass != null ? E.text0 : E.text3,
                      }}
                    >
                      {r.pass != null ? `${r.pass.toFixed(1)}%` : '-'}
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: E.text2 }}>
                      <StatusDot status={r.status} />
                      <span>{r.status}</span>
                    </div>
                    <div style={{ fontFamily: E.fMono, color: E.text2, textAlign: 'right' }}>
                      {r.cost}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </Card>

          {/* OBSERVED METRICS */}
          <Card style={{ padding: 0, overflow: 'hidden' }}>
            <div
              style={{
                padding: '14px 18px',
                borderBottom: `1px solid ${E.hair}`,
                display: 'flex',
                alignItems: 'center',
              }}
            >
              <Eyebrow>Observed metrics</Eyebrow>
              <span style={{ flex: 1 }} />
              <span style={{ fontFamily: E.fMono, fontSize: 10, color: E.text3 }}>
                {data.observed_metrics.length}
              </span>
            </div>
            {data.observed_metrics.length === 0 ? (
              <div style={{ padding: 22, textAlign: 'center', fontSize: 13, color: E.text3 }}>
                No metrics graded on this dataset yet.
              </div>
            ) : (
              <div>
                {data.observed_metrics.map((m, i) => (
                  <div
                    key={m.id}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 10,
                      padding: '11px 18px',
                      borderTop: i ? `1px solid ${E.hair}` : 'none',
                      fontSize: 12,
                    }}
                  >
                    <div
                      title={m.id}
                      style={{
                        fontFamily: E.fMono,
                        fontSize: 11.5,
                        color: E.text1,
                        flex: 1,
                        minWidth: 0,
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                      }}
                    >
                      {m.id}
                    </div>
                    <Pill
                      mono
                      color={m.kind === 'LLM judge' ? E.steel : E.text2}
                      bg={m.kind === 'LLM judge' ? E.steelDim : E.panel2}
                      style={{ fontSize: 10 }}
                    >
                      {m.kind}
                    </Pill>
                    <span
                      style={{
                        fontFamily: E.fMono,
                        fontSize: 11,
                        color: E.text3,
                        minWidth: 36,
                        textAlign: 'right',
                      }}
                    >
                      {m.uses}×
                    </span>
                  </div>
                ))}
              </div>
            )}
          </Card>
        </div>

        {/* ITEMS PREVIEW */}
        <Card style={{ padding: 0, marginTop: 14, overflow: 'hidden' }}>
          <div
            style={{
              padding: '14px 18px',
              borderBottom: `1px solid ${E.hair}`,
              display: 'flex',
              alignItems: 'center',
            }}
          >
            <Eyebrow>Items preview</Eyebrow>
            <span style={{ flex: 1 }} />
            <span style={{ fontFamily: E.fMono, fontSize: 10, color: E.text3 }}>
              {data.items_preview.length} of {data.total_items}
            </span>
          </div>
          {data.items_preview.length === 0 ? (
            <div style={{ padding: 22, textAlign: 'center', fontSize: 13, color: E.text3 }}>
              This dataset is empty.
            </div>
          ) : (
            <div style={{ maxHeight: 520, overflow: 'auto' }}>
              {data.items_preview.map((it, i) => (
                <div
                  key={it.id}
                  style={{
                    padding: '12px 18px',
                    borderTop: i ? `1px solid ${E.hair}` : 'none',
                  }}
                >
                  <div
                    style={{
                      fontFamily: E.fMono,
                      fontSize: 10.5,
                      color: E.text3,
                      marginBottom: 4,
                    }}
                  >
                    {it.id}
                  </div>
                  <div
                    style={{
                      fontSize: 12,
                      color: E.text1,
                      lineHeight: 1.5,
                      wordBreak: 'break-word',
                    }}
                  >
                    {truncate(it.input_preview)}
                  </div>
                  {it.expected_preview && (
                    <div
                      style={{
                        fontSize: 11.5,
                        color: E.text2,
                        marginTop: 4,
                        lineHeight: 1.45,
                        wordBreak: 'break-word',
                      }}
                    >
                      <span style={{ color: E.text3, fontFamily: E.fMono, fontSize: 10 }}>
                        expected:{' '}
                      </span>
                      {truncate(it.expected_preview, 200)}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
          <div
            style={{
              padding: '10px 18px',
              borderTop: `1px solid ${E.hair}`,
              fontSize: 11,
              color: E.text3,
              fontFamily: E.fMono,
              background: E.panel2,
            }}
          >
            Showing {data.items_preview.length} of {data.total_items}. Open full dataset in CLI:{' '}
            <span style={{ color: E.text1 }}>evalyn dataset show {data.name}</span>
          </div>
        </Card>

        <div style={{ height: 30 }} />
      </div>
    </AppShell>
  );
}
