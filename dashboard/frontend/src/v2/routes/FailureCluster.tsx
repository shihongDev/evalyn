/**
 * FailureCluster - deep-dive on a single cluster of failed items.
 * Wires v2.cluster(runId, clusterId) into the design from screens-4.jsx.
 */

import { useCallback, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { AppShell } from '../AppShell';
import { Bar, Btn, Card, Eyebrow, Glossary, LineChart, Pill, Skeleton, Spinner, UpdatingChip } from '../ui';
import { v2 } from '../api/client';
import type { ClusterDetail } from '../api/types';
import { useV2Resource } from '../hooks/useV2Resource';
import { useProject } from '../hooks/useProject';
import { preloadCoPilotThread } from '../routePreloads';
import { linkifyText, makeUrlCounter } from '../textRender';
import { copyToClipboard } from '../clipboard';
import { E } from '../tokens';

/** Render the cluster as portable markdown - label, prose pattern,
 * trigger phrases, item table, and suggested fix if present. The output
 * pastes cleanly into Slack, Notion, GitHub issues, or email. */
function clusterToMarkdown(d: ClusterDetail, runId: string): string {
  const lines: string[] = [];
  lines.push(`# ${d.label}`);
  lines.push('');
  lines.push(`Run \`${runId}\` - ${d.total_in_cluster} of ${d.total_failures_in_run} failures`);
  if (d.total_items_in_run > 0) {
    const pct = ((d.total_in_cluster / d.total_items_in_run) * 100).toFixed(1);
    lines.push(`(${pct}% of all items in this run)`);
  }
  lines.push('');
  lines.push('## Pattern');
  lines.push(d.pattern.trim());
  lines.push('');
  if (d.triggers.length > 0) {
    lines.push('## Trigger phrases');
    for (const t of d.triggers) lines.push(`- "${t.phrase}" (${t.count}x)`);
    lines.push('');
  }
  if (d.items.length > 0) {
    lines.push(`## Failed items (${d.items.length})`);
    lines.push('');
    lines.push('| ID | Input | Output | Metric | Score |');
    lines.push('|---|---|---|---|---|');
    for (const it of d.items) {
      const u = it.user.replaceAll('|', '\\|').replaceAll('\n', ' ');
      const o = it.hallucinated.replaceAll('|', '\\|').replaceAll('\n', ' ');
      lines.push(`| ${it.id} | ${u} | ${o} | ${it.tier} | ${it.score.toFixed(2)} |`);
    }
    lines.push('');
  }
  if (d.suggested_fix) {
    lines.push('## Co-pilot\'s suggested fix');
    lines.push(d.suggested_fix.body_md.trim());
    lines.push('');
    lines.push(
      `_Estimated impact: ${d.suggested_fix.estimated_impact} - cost ${d.suggested_fix.cost} - duration ${d.suggested_fix.duration}_`,
    );
  }
  return lines.join('\n');
}


export default function FailureCluster() {
  const params = useParams<{ runId: string; clusterId: string }>();
  const navigate = useNavigate();
  const project = useProject();
  const runId = params.runId ?? '';
  const clusterId = params.clusterId ?? '';

  const fetcher = useCallback(
    () => v2.cluster(runId, clusterId),
    [runId, clusterId],
  );
  const { data, err, refetch, reloading, isInitialLoad } = useV2Resource<ClusterDetail>(
    `cluster:${runId}:${clusterId}`,
    fetcher,
  );

  // Per-button copy state. Two buttons share the pattern: bundle-as-md
  // for the whole cluster, and copy the suggested fix prose alone. Each
  // flips its own label briefly so the user gets affordance feedback.
  const [bundleCopy, setBundleCopy] = useState<'idle' | 'copied' | 'error'>('idle');
  const [fixCopy, setFixCopy] = useState<'idle' | 'copied' | 'error'>('idle');

  async function handleCopyBundle() {
    if (!data) return;
    try {
      await copyToClipboard(clusterToMarkdown(data, runId));
      setBundleCopy('copied');
      window.setTimeout(() => setBundleCopy('idle'), 2000);
    } catch {
      setBundleCopy('error');
      window.setTimeout(() => setBundleCopy('idle'), 3000);
    }
  }

  async function handleCopyFix() {
    if (!data?.suggested_fix) return;
    try {
      await copyToClipboard(data.suggested_fix.body_md.trim());
      setFixCopy('copied');
      window.setTimeout(() => setFixCopy('idle'), 2000);
    } catch {
      setFixCopy('error');
      window.setTimeout(() => setFixCopy('idle'), 3000);
    }
  }

  const triggerMax = data ? Math.max(...data.triggers.map((t) => t.count), 1) : 1;
  const trendMax = data?.trend.y_max ?? 20;

  return (
    <AppShell
      contextChip={project ?? undefined}
      breadcrumb={['Experiments', runId, 'Failures', data?.label ?? clusterId]}
    >
      <div style={{ padding: '28px 36px' }}>
        {err && (
          <Card style={{ padding: 16, marginBottom: 16, borderColor: E.fail }}>
            <div role="alert">
              <Eyebrow style={{ color: E.fail }}>Error loading cluster</Eyebrow>
              <div style={{ fontFamily: E.fMono, fontSize: 12, color: E.text2, marginTop: 6 }}>
                {err}
              </div>
            </div>
            <div style={{ marginTop: 10, display: 'flex', gap: 8 }}>
              <Btn
                kind="secondary"
                size="sm"
                onClick={() => void refetch()}
                disabled={reloading}
                aria-busy={reloading}
              >
                {reloading ? (
                  <>
                    <Spinner size={11} /> Retrying
                  </>
                ) : (
                  'Retry'
                )}
              </Btn>
              <Btn
                kind="ghost"
                size="sm"
                onClick={() => navigate(`/experiments/${encodeURIComponent(runId)}`)}
              >
                Back to run
              </Btn>
            </div>
          </Card>
        )}
        {!data && !err && (
          <>
            <Skeleton w={220} h={11} />
            <div style={{ marginTop: 8 }}>
              <Skeleton w={420} h={34} />
            </div>
            <div style={{ marginTop: 12 }}>
              <Skeleton w="80%" h={13} />
            </div>
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: '1fr 1fr',
                gap: 14,
                marginTop: 22,
              }}
            >
              <Card style={{ padding: 18 }}>
                <Skeleton w={100} h={11} />
                <div style={{ marginTop: 14 }}>
                  <Skeleton w="100%" h={14} />
                  <div style={{ marginTop: 8 }}>
                    <Skeleton w="90%" h={14} />
                  </div>
                </div>
              </Card>
              <Card style={{ padding: 18 }}>
                <Skeleton w={180} h={11} />
                <div style={{ marginTop: 14 }}>
                  <Skeleton w="100%" h={150} style={{ borderRadius: 6 }} />
                </div>
              </Card>
            </div>
          </>
        )}
        {data && (
          <>
            <div style={{ display: 'flex', alignItems: 'flex-start', gap: 18 }}>
              <div style={{ flex: 1 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 6 }}>
                  <span style={{ fontSize: 11, color: E.text3, fontFamily: E.fMono }}>
                    {data.total_in_cluster} of {data.total_failures_in_run} failures
                    {data.total_items_in_run > 0
                      ? ` - ${((data.total_in_cluster / data.total_items_in_run) * 100).toFixed(1)}% of all items`
                      : ''}
                  </span>
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
                    margin: 0,
                    color: E.text0,
                    letterSpacing: '-0.015em',
                  }}
                >
                  {data.label}
                </h1>
                <p
                  style={{
                    fontSize: 13.5,
                    color: E.text2,
                    marginTop: 8,
                    lineHeight: 1.55,
                    maxWidth: 720,
                  }}
                >
                  {linkifyText(data.pattern, makeUrlCounter())}
                </p>
              </div>
              <Btn
                kind="primary"
                size="md"
                onClick={() => navigate('/copilot')}
                onMouseEnter={() => void preloadCoPilotThread()}
                onFocus={() => void preloadCoPilotThread()}
                title="Open the co-pilot - paste this cluster's pattern to get started"
              >
                Fix with co-pilot -&gt;
              </Btn>
            </div>

            <div
              style={{
                display: 'grid',
                gridTemplateColumns: '1fr 1fr',
                gap: 14,
                marginTop: 22,
              }}
            >
              <Card style={{ padding: 18 }}>
                <Eyebrow>Pattern</Eyebrow>
                <div style={{ marginTop: 10, fontSize: 13, color: E.text1, lineHeight: 1.6 }}>
                  {linkifyText(data.pattern, makeUrlCounter())}
                </div>
                <div
                  style={{
                    marginTop: 14,
                    padding: 12,
                    background: E.panel2,
                    borderRadius: 7,
                    border: `1px solid ${E.hair}`,
                  }}
                >
                  <Eyebrow style={{ marginBottom: 6 }}>Common 3-word fragments</Eyebrow>
                  <div role="list" aria-label="Common 3-word fragments by frequency">
                  {data.triggers.map((r) => (
                    <div
                      key={r.phrase}
                      role="listitem"
                      aria-label={`"${r.phrase}" appears ${r.count} time${r.count === 1 ? '' : 's'}`}
                      style={{
                        display: 'grid',
                        gridTemplateColumns: '160px 1fr 30px',
                        gap: 8,
                        alignItems: 'center',
                        marginTop: 6,
                      }}
                    >
                      <span style={{ fontFamily: E.fMono, fontSize: 11, color: E.text2 }}>
                        {r.phrase}
                      </span>
                      <Bar value={r.count} max={triggerMax} w={'100%'} h={4} color={E.fail} />
                      <span
                        style={{
                          fontFamily: E.fMono,
                          fontSize: 11,
                          color: E.text3,
                          textAlign: 'right',
                        }}
                      >
                        {r.count}
                      </span>
                    </div>
                  ))}
                  </div>
                </div>
              </Card>

              <Card style={{ padding: 18 }}>
                <Eyebrow>
                  <Glossary term="A failure cluster groups items that failed for the same reason. v2 buckets by metric; LLM clustering coming.">
                    Cluster
                  </Glossary>{' '}
                  trend across recent runs
                </Eyebrow>
                <div style={{ marginTop: 14 }}>
                  {data.trend.data.length >= 2 ? (
                    <LineChart
                      w={420}
                      h={150}
                      yMin={0}
                      yMax={trendMax}
                      xLabels={data.trend.x_labels}
                      series={[{ color: E.fail, fill: true, width: 2, data: data.trend.data }]}
                      title={`Cluster size across the last ${data.trend.data.length} runs, peaking at ${Math.max(...data.trend.data)} failures.`}
                    />
                  ) : (
                    <div
                      style={{
                        fontSize: 12.5,
                        color: E.text2,
                        lineHeight: 1.55,
                        padding: '24px 4px',
                      }}
                    >
                      Cluster size: {data.total_in_cluster} items in this run.
                      Not enough run history yet to chart a trend (need &gt;=3
                      runs in this dataset).
                    </div>
                  )}
                </div>
              </Card>
            </div>

            <Card style={{ marginTop: 14, padding: 0, overflow: 'hidden' }}>
              <div
                style={{
                  padding: '12px 18px',
                  borderBottom: `1px solid ${E.hair}`,
                  display: 'flex',
                  alignItems: 'center',
                }}
              >
                <span style={{ fontSize: 13, color: E.text0, fontWeight: 500 }}>
                  All {data.items.length} items in this cluster
                </span>
                <span style={{ flex: 1 }} />
                <Btn
                  kind="ghost"
                  size="sm"
                  onClick={() => void handleCopyBundle()}
                  title={
                    bundleCopy === 'copied'
                      ? 'Cluster bundle on clipboard'
                      : bundleCopy === 'error'
                        ? 'Browser blocked clipboard access'
                        : 'Copy this cluster (label, pattern, items, suggested fix) as markdown - paste into Slack, Notion, or a GitHub issue'
                  }
                >
                  {bundleCopy === 'copied'
                    ? '✓ Copied'
                    : bundleCopy === 'error'
                      ? '✗ Failed'
                      : 'Copy as markdown'}
                </Btn>
              </div>
              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: '50px 1fr 1fr 90px 70px',
                  padding: '10px 18px',
                  borderBottom: `1px solid ${E.hair}`,
                  fontFamily: E.fMono,
                  fontSize: 10,
                  color: E.text3,
                  letterSpacing: '0.06em',
                }}
              >
                <span>ID</span>
                <span>USER</span>
                <span>OUTPUT</span>
                <span>METRIC</span>
                <span style={{ textAlign: 'right' }}>SCORE</span>
              </div>
              {data.items.map((s, i) => (
                <div
                  key={s.id}
                  style={{
                    display: 'grid',
                    gridTemplateColumns: '50px 1fr 1fr 90px 70px',
                    padding: '10px 18px',
                    borderTop: i ? `1px solid ${E.hair}` : 'none',
                    alignItems: 'center',
                    gap: 8,
                    fontSize: 12,
                  }}
                >
                  <span style={{ fontFamily: E.fMono, fontSize: 11, color: E.text3 }}>{s.id}</span>
                  <span
                    title={s.user}
                    style={{
                      color: E.text2,
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                    }}
                  >
                    {s.user}
                  </span>
                  <span
                    title={s.hallucinated}
                    style={{
                      color: E.text1,
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                    }}
                  >
                    {s.hallucinated}
                  </span>
                  <Pill mono color={E.fail} bg={E.failDim} style={{ fontSize: 10 }}>
                    {s.tier}
                  </Pill>
                  <span
                    style={{
                      textAlign: 'right',
                      fontFamily: E.fSerif,
                      fontSize: 14,
                      color: E.fail,
                    }}
                  >
                    {s.score.toFixed(2)}
                  </span>
                </div>
              ))}
            </Card>

            {data.suggested_fix && (
              <Card
                accent
                style={{
                  marginTop: 14,
                  padding: 22,
                  background: E.emberDim,
                  border: `1px solid ${E.emberRim}`,
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 10 }}>
                  <span
                    style={{
                      width: 26,
                      height: 26,
                      borderRadius: 6,
                      background: `linear-gradient(135deg, ${E.ember}, #b8501f)`,
                      color: E.emberInk,
                      fontWeight: 700,
                      fontFamily: E.fSerif,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontSize: 14,
                    }}
                  >
                    e
                  </span>
                  <Eyebrow style={{ color: E.ember }}>Co-pilot's suggested fix</Eyebrow>
                </div>
                <div style={{ fontSize: 14, color: E.text1, lineHeight: 1.6 }}>
                  {linkifyText(data.suggested_fix.body_md, makeUrlCounter())}
                </div>
                <div
                  style={{
                    marginTop: 10,
                    fontSize: 12,
                    color: E.text2,
                    fontFamily: E.fMono,
                  }}
                >
                  Estimated impact: {data.suggested_fix.estimated_impact}
                </div>
                <div
                  style={{
                    marginTop: 14,
                    display: 'flex',
                    alignItems: 'center',
                    gap: 8,
                    flexWrap: 'wrap',
                  }}
                >
                  <Btn
                    kind="primary"
                    size="md"
                    onClick={() => void handleCopyFix()}
                    title={
                      fixCopy === 'copied'
                        ? 'Fix on clipboard'
                        : fixCopy === 'error'
                          ? 'Browser blocked clipboard access'
                          : 'Copy this suggested fix to your clipboard - paste into a PR description, ticket, or Slack thread'
                    }
                  >
                    {fixCopy === 'copied'
                      ? '✓ Copied'
                      : fixCopy === 'error'
                        ? '✗ Failed'
                        : 'Copy fix'}
                  </Btn>
                  <span
                    style={{
                      fontSize: 11.5,
                      color: E.text3,
                      fontFamily: E.fMono,
                    }}
                  >
                    Apply manually then re-run via the experiment header
                  </span>
                </div>
              </Card>
            )}

            <div style={{ height: 30 }} />
          </>
        )}
      </div>
    </AppShell>
  );
}
