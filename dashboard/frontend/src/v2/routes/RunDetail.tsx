/**
 * RunDetail - the deepest read view: headline stats, pass-rate vs baseline,
 * failure clusters, sub-metric breakdown, confusion matrix, failed item preview.
 */

import { useEffect, useState } from 'react';
import { useNavigate, useParams, useSearchParams } from 'react-router-dom';
import { AppShell } from '../AppShell';
import { Card, Eyebrow, Pill, Btn, StatusDot, Donut, LineChart } from '../ui';
import type { LineSeries } from '../ui';
import { v2 } from '../api/client';
import type { ExperimentDetail } from '../api/types';
import { E } from '../tokens';

const CLUSTER_COLOR: Record<string, string> = {
  fail: E.fail,
  warn: E.warn,
  steel: E.steel,
  violet: '#a78bfa',
  text3: E.text3,
};

const SERIES_COLOR: Record<string, string> = {
  ember: E.ember,
  steel: E.steel,
  fail: E.fail,
};

function deltaColor(kind: 'pass' | 'fail' | 'warn' | 'info'): string {
  if (kind === 'pass') return E.pass;
  if (kind === 'fail') return E.fail;
  if (kind === 'warn') return E.warn;
  return E.steel;
}

export default function RunDetail() {
  const { runId } = useParams<{ runId: string }>();
  const [searchParams] = useSearchParams();
  const compareWith = searchParams.get('compare');
  const navigate = useNavigate();
  const [detail, setDetail] = useState<ExperimentDetail | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState(0);

  useEffect(() => {
    if (!runId) return;
    setDetail(null);
    setErr(null);
    v2.experiment(runId)
      .then(setDetail)
      .catch((e) => setErr(String(e)));
  }, [runId]);

  if (err) {
    return (
      <AppShell breadcrumb={['Experiments', runId ?? '']}>
        <div style={{ padding: '32px 36px' }}>
          <Card style={{ padding: 16, borderColor: E.fail }}>
            <Eyebrow style={{ color: E.fail }}>Run not found</Eyebrow>
            <div style={{ fontFamily: E.fMono, fontSize: 12, color: E.text2, marginTop: 6 }}>{err}</div>
            <div style={{ marginTop: 12 }}>
              <Btn kind="secondary" size="sm" onClick={() => navigate('/experiments')}>
                ← Back to experiments
              </Btn>
            </div>
          </Card>
        </div>
      </AppShell>
    );
  }

  if (!detail) {
    return (
      <AppShell breadcrumb={['Experiments', runId ?? '']}>
        <div style={{ padding: '32px 36px', color: E.text3, fontSize: 13 }}>Loading...</div>
      </AppShell>
    );
  }

  const tabs = [
    'Summary',
    `Items - ${detail.dataset.n}`,
    `Failures - ${detail.failure_clusters.total_failures}`,
    'Compare',
    'Trace',
  ];

  const headerExtra = (
    <div style={{ display: 'flex', gap: 6 }}>
      <Btn kind="ghost" size="sm" disabled title="Coming soon">
        ↗ Share
      </Btn>
      <Btn kind="ghost" size="sm" disabled title="Coming soon">
        ⎘ Duplicate
      </Btn>
      <Btn kind="primary" size="sm" disabled title="Coming soon">
        ↻ Re-run
      </Btn>
    </div>
  );

  const passSeries: LineSeries[] = detail.pass_timeline.series.map((s) => ({
    color: SERIES_COLOR[s.color_kind] ?? E.text2,
    width: s.color_kind === 'ember' ? 2 : 1.5,
    fill: s.color_kind === 'ember',
    data: s.data,
  }));

  const donutSegments = detail.failure_clusters.clusters.map((c) => ({
    value: c.count,
    color: CLUSTER_COLOR[c.color_kind] ?? E.text3,
    label: c.label,
  }));

  return (
    <AppShell
      contextChip={{ name: 'Customer Support Agent', version: 'v0.3' }}
      breadcrumb={['Experiments', detail.name]}
      headerExtra={headerExtra}
    >
      <div style={{ padding: '28px 36px 0' }}>
        {/* HEADER */}
        <div style={{ display: 'flex', alignItems: 'flex-start', gap: 18 }}>
          <div style={{ flex: 1 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 4 }}>
              <StatusDot status={detail.status} animated={detail.status === 'running'} />
              <span style={{ fontFamily: E.fMono, fontSize: 11, color: E.text3 }}>
                {detail.id} - {detail.status} {detail.finished_at_iso} - {detail.duration} - {detail.cost}
              </span>
            </div>
            <h1
              style={{
                fontFamily: E.fSerif,
                fontSize: 32,
                fontWeight: 400,
                margin: 0,
                color: E.text0,
                letterSpacing: '-0.015em',
              }}
            >
              {detail.name}
            </h1>
            <div style={{ display: 'flex', gap: 8, marginTop: 10, flexWrap: 'wrap', alignItems: 'center' }}>
              <Pill mono style={{ background: E.panel2, color: E.text2 }}>
                {detail.dataset.name} - {detail.dataset.n}
              </Pill>
              {detail.model && (
                <Pill mono style={{ background: E.panel2, color: E.text2 }}>
                  {detail.model.id} - temp {detail.model.temp}
                </Pill>
              )}
              {detail.rubric && (
                <Pill mono style={{ background: E.panel2, color: E.text2 }}>
                  {detail.rubric}
                </Pill>
              )}
              {detail.baseline_id && (
                <>
                  <span style={{ color: E.text4, fontSize: 11 }}>-</span>
                  <span style={{ fontSize: 11, color: E.text3 }}>vs.</span>
                  <Pill mono style={{ background: E.panel2, color: E.steel }}>
                    {detail.baseline_id}
                  </Pill>
                </>
              )}
              {compareWith && (
                <>
                  <span style={{ color: E.text4, fontSize: 11 }}>-</span>
                  <span style={{ fontSize: 11, color: E.text3 }}>compare overlay:</span>
                  <Pill mono style={{ background: E.panel2, color: E.ember }}>
                    {compareWith}
                  </Pill>
                </>
              )}
            </div>
          </div>
        </div>

        {/* TABS */}
        <div style={{ display: 'flex', gap: 2, marginTop: 22, borderBottom: `1px solid ${E.hair}` }}>
          {tabs.map((t, i) => {
            const isActive = i === activeTab;
            return (
              <button
                key={t}
                type="button"
                onClick={() => setActiveTab(i)}
                style={{
                  padding: '9px 14px',
                  fontSize: 12.5,
                  cursor: 'pointer',
                  background: 'transparent',
                  border: 'none',
                  color: isActive ? E.text0 : E.text2,
                  fontWeight: isActive ? 500 : 400,
                  borderBottom: `2px solid ${isActive ? E.ember : 'transparent'}`,
                  marginBottom: -1,
                }}
              >
                {t}
              </button>
            );
          })}
        </div>
      </div>

      <div style={{ padding: '20px 36px' }}>
        {/* HEADLINE STAT ROW */}
        {detail.headline.length > 0 && (
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: `repeat(${detail.headline.length}, 1fr)`,
              gap: 14,
            }}
          >
            {detail.headline.map((s) => (
              <Card key={s.label} style={{ padding: 16 }}>
                <Eyebrow>{s.label}</Eyebrow>
                <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, marginTop: 6 }}>
                  <span style={{ fontFamily: E.fSerif, fontSize: 28, color: E.text0 }}>{s.value}</span>
                  <span style={{ fontFamily: E.fMono, fontSize: 11, color: deltaColor(s.delta_kind) }}>{s.delta}</span>
                </div>
                <div style={{ fontSize: 11, color: E.text3, marginTop: 4 }}>{s.sub}</div>
              </Card>
            ))}
          </div>
        )}

        {/* MAIN GRID: timeline + clusters */}
        <div style={{ display: 'grid', gridTemplateColumns: '1.6fr 1fr', gap: 14, marginTop: 14 }}>
          <Card style={{ padding: 18 }}>
            <div style={{ display: 'flex', alignItems: 'center', marginBottom: 12 }}>
              <Eyebrow>Pass rate - this run vs. baseline</Eyebrow>
              <span style={{ flex: 1 }} />
              <div style={{ display: 'flex', gap: 12, fontSize: 10, fontFamily: E.fMono }}>
                {detail.pass_timeline.series.map((s) => {
                  const c = SERIES_COLOR[s.color_kind] ?? E.text2;
                  return (
                    <span key={s.label} style={{ color: c, display: 'inline-flex', alignItems: 'center', gap: 5 }}>
                      <span style={{ width: 8, height: 2, background: c }} />
                      {s.label}
                    </span>
                  );
                })}
                <span style={{ color: E.text4, display: 'inline-flex', alignItems: 'center', gap: 5 }}>
                  <span style={{ width: 8, height: 0, borderTop: `1px dashed ${E.text4}` }} />
                  ship gate
                </span>
              </div>
            </div>
            {passSeries.length > 0 && (
              <LineChart
                w={620}
                h={200}
                yMin={detail.pass_timeline.y_min}
                yMax={detail.pass_timeline.y_max}
                baseline={detail.pass_timeline.ship_gate}
                xLabels={detail.pass_timeline.x_labels}
                series={passSeries}
              />
            )}
            <div style={{ fontSize: 11, color: E.text3, marginTop: 8 }}>
              Cumulative pass rate as items were graded.
            </div>
          </Card>

          <Card style={{ padding: 18 }}>
            <div style={{ display: 'flex', alignItems: 'center', marginBottom: 12 }}>
              <Eyebrow>
                Failure clusters - {detail.failure_clusters.total_failures} of {detail.failure_clusters.total_items}
              </Eyebrow>
              <span style={{ flex: 1 }} />
              <Btn
                kind="bare"
                size="sm"
                onClick={() => {
                  const first = detail.failure_clusters.clusters[0];
                  if (first) navigate(`/experiments/${encodeURIComponent(detail.id)}/cluster/${encodeURIComponent(first.id)}`);
                }}
                disabled={detail.failure_clusters.clusters.length === 0}
              >
                Open all →
              </Btn>
            </div>
            {detail.failure_clusters.clusters.length === 0 ? (
              <div style={{ fontSize: 12.5, color: E.text3, padding: '12px 0' }}>No failure clusters.</div>
            ) : (
              <div style={{ display: 'flex', alignItems: 'center', gap: 18 }}>
                <Donut
                  size={120}
                  thick={18}
                  segments={donutSegments}
                  center={
                    <div style={{ textAlign: 'center' }}>
                      <div style={{ fontFamily: E.fSerif, fontSize: 22, color: E.text0 }}>
                        {detail.failure_clusters.total_failures}
                      </div>
                      <div style={{ fontSize: 9, color: E.text3, fontFamily: E.fMono, letterSpacing: '0.08em' }}>
                        FAILS
                      </div>
                    </div>
                  }
                />
                <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 7 }}>
                  {detail.failure_clusters.clusters.map((c) => (
                    <button
                      key={c.id}
                      type="button"
                      onClick={() =>
                        navigate(`/experiments/${encodeURIComponent(detail.id)}/cluster/${encodeURIComponent(c.id)}`)
                      }
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: 8,
                        fontSize: 12,
                        background: 'transparent',
                        border: 'none',
                        cursor: 'pointer',
                        padding: 0,
                        textAlign: 'left',
                      }}
                    >
                      <span
                        style={{
                          width: 7,
                          height: 7,
                          borderRadius: 2,
                          background: CLUSTER_COLOR[c.color_kind] ?? E.text3,
                          flexShrink: 0,
                        }}
                      />
                      <span
                        style={{
                          flex: 1,
                          color: E.text1,
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap',
                        }}
                      >
                        {c.label}
                      </span>
                      {c.regression && (
                        <Pill mono color={E.fail} bg={E.failDim} style={{ fontSize: 9 }}>
                          regress
                        </Pill>
                      )}
                      <span
                        style={{ fontFamily: E.fMono, fontSize: 11, color: E.text2, width: 26, textAlign: 'right' }}
                      >
                        {c.count}
                      </span>
                    </button>
                  ))}
                </div>
              </div>
            )}
          </Card>
        </div>

        {/* SUB-METRICS + CONFUSION */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14, marginTop: 14 }}>
          <Card style={{ padding: 18 }}>
            <Eyebrow>Sub-metric breakdown</Eyebrow>
            {detail.sub_metrics.length === 0 ? (
              <div style={{ marginTop: 14, fontSize: 12.5, color: E.text3 }}>No sub-metrics.</div>
            ) : (
              <div style={{ marginTop: 14, display: 'flex', flexDirection: 'column', gap: 14 }}>
                {detail.sub_metrics.map((m) => {
                  const baseline = m.baseline ?? 0;
                  const better = m.baseline == null
                    ? E.steel
                    : m.inverse
                      ? m.value < m.baseline ? E.pass : E.fail
                      : m.value > m.baseline ? E.pass : E.steel;
                  return (
                    <div key={m.label}>
                      <div
                        style={{
                          display: 'flex',
                          justifyContent: 'space-between',
                          alignItems: 'baseline',
                          marginBottom: 5,
                          fontSize: 12,
                        }}
                      >
                        <span style={{ color: E.text1 }}>
                          {m.label}
                          {m.inverse ? ' (lower=better)' : ''}
                        </span>
                        <div style={{ display: 'flex', gap: 12, alignItems: 'baseline', fontFamily: E.fMono }}>
                          {m.baseline != null && (
                            <span style={{ color: E.steel, fontSize: 11 }}>base {m.baseline}%</span>
                          )}
                          <span style={{ color: E.text0, fontSize: 13 }}>{m.value}%</span>
                        </div>
                      </div>
                      <div style={{ position: 'relative', height: 6, background: E.panel2, borderRadius: 3 }}>
                        {m.baseline != null && (
                          <div
                            style={{
                              position: 'absolute',
                              left: 0,
                              top: 0,
                              bottom: 0,
                              width: `${baseline}%`,
                              background: E.steel,
                              opacity: 0.4,
                              borderRadius: 3,
                            }}
                          />
                        )}
                        <div
                          style={{
                            position: 'absolute',
                            left: 0,
                            top: 0,
                            bottom: 0,
                            width: `${m.value}%`,
                            background: better,
                            borderRadius: 3,
                          }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </Card>

          <Card style={{ padding: 18 }}>
            <div style={{ display: 'flex', alignItems: 'center' }}>
              <Eyebrow>Item-level comparison vs. baseline</Eyebrow>
              <span style={{ flex: 1 }} />
              <Pill mono style={{ background: E.panel2, color: E.text2, fontSize: 10 }}>
                {detail.dataset.n} items
              </Pill>
            </div>
            {detail.confusion ? (
              <>
                <div
                  style={{
                    marginTop: 14,
                    display: 'grid',
                    gridTemplateColumns: '70px 1fr 1fr',
                    gridTemplateRows: '24px 1fr 1fr',
                    gap: 4,
                    fontSize: 11,
                    fontFamily: E.fMono,
                  }}
                >
                  <div></div>
                  <div
                    style={{
                      color: E.text3,
                      textAlign: 'center',
                      display: 'flex',
                      alignItems: 'flex-end',
                      justifyContent: 'center',
                    }}
                  >
                    new PASS
                  </div>
                  <div
                    style={{
                      color: E.text3,
                      textAlign: 'center',
                      display: 'flex',
                      alignItems: 'flex-end',
                      justifyContent: 'center',
                    }}
                  >
                    new FAIL
                  </div>

                  <div
                    style={{
                      color: E.text3,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'flex-end',
                      paddingRight: 8,
                    }}
                  >
                    base PASS
                  </div>
                  <div
                    style={{
                      background: E.passDim,
                      border: `1px solid ${E.pass}33`,
                      padding: 14,
                      borderRadius: 6,
                    }}
                  >
                    <div style={{ fontFamily: E.fSerif, fontSize: 22, color: E.text0 }}>
                      {detail.confusion.base_pass_v_pass}
                    </div>
                    <div style={{ fontSize: 10, color: E.text3, marginTop: 2 }}>kept passing</div>
                  </div>
                  <div
                    style={{
                      background: E.failDim,
                      border: `1px solid ${E.fail}33`,
                      padding: 14,
                      borderRadius: 6,
                    }}
                  >
                    <div style={{ fontFamily: E.fSerif, fontSize: 22, color: E.fail }}>
                      {detail.confusion.base_pass_v_fail}
                    </div>
                    <div style={{ fontSize: 10, color: E.text3, marginTop: 2 }}>regressed</div>
                  </div>

                  <div
                    style={{
                      color: E.text3,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'flex-end',
                      paddingRight: 8,
                    }}
                  >
                    base FAIL
                  </div>
                  <div
                    style={{
                      background: E.emberDim,
                      border: `1px solid ${E.emberRim}`,
                      padding: 14,
                      borderRadius: 6,
                    }}
                  >
                    <div style={{ fontFamily: E.fSerif, fontSize: 22, color: E.ember }}>
                      {detail.confusion.base_fail_v_pass}
                    </div>
                    <div style={{ fontSize: 10, color: E.text3, marginTop: 2 }}>fixed</div>
                  </div>
                  <div
                    style={{
                      background: E.panel2,
                      border: `1px solid ${E.hair}`,
                      padding: 14,
                      borderRadius: 6,
                    }}
                  >
                    <div style={{ fontFamily: E.fSerif, fontSize: 22, color: E.text2 }}>
                      {detail.confusion.base_fail_v_fail}
                    </div>
                    <div style={{ fontSize: 10, color: E.text3, marginTop: 2 }}>still failing</div>
                  </div>
                </div>
                <div
                  style={{
                    marginTop: 14,
                    padding: 10,
                    background: E.panel2,
                    borderRadius: 6,
                    fontSize: 11.5,
                    color: E.text2,
                    lineHeight: 1.5,
                  }}
                >
                  {/* net_delta is signed; show explicit + prefix when positive */}
                  <b style={{ color: E.text0 }}>
                    Net {detail.confusion.net_delta >= 0 ? '+' : ''}
                    {detail.confusion.net_delta} items
                  </b>{' '}
                  moved to passing.
                  {detail.confusion.base_pass_v_fail > 0
                    ? ` The ${detail.confusion.base_pass_v_fail} regressions are the ones to investigate before shipping.`
                    : ''}
                </div>
              </>
            ) : (
              <div style={{ marginTop: 14, fontSize: 12.5, color: E.text3 }}>
                No baseline to compare against.
              </div>
            )}
          </Card>
        </div>

        {/* FAILED ITEMS PREVIEW */}
        {detail.failed_items_preview.length > 0 && (
          <Card style={{ marginTop: 14, padding: 0, overflow: 'hidden' }}>
            <div
              style={{
                padding: '12px 18px',
                borderBottom: `1px solid ${E.hair}`,
                display: 'flex',
                alignItems: 'center',
                gap: 10,
              }}
            >
              <span style={{ fontSize: 13, color: E.text0, fontWeight: 500 }}>Failed items</span>
              <Pill mono color={E.fail} bg={E.failDim} style={{ fontSize: 10 }}>
                {detail.failure_clusters.total_failures}
              </Pill>
              <span style={{ flex: 1 }} />
              <Btn kind="ghost" size="sm" disabled title="Coming soon">
                Filter ▾
              </Btn>
              <Btn kind="ghost" size="sm" disabled title="Coming soon">
                Group: cluster ▾
              </Btn>
            </div>
            {detail.failed_items_preview.map((s, i) => (
              <div
                key={s.id}
                style={{
                  padding: '14px 18px',
                  borderTop: i ? `1px solid ${E.hair}` : 'none',
                  display: 'grid',
                  gridTemplateColumns: '70px 1fr 90px 60px',
                  gap: 14,
                  alignItems: 'flex-start',
                }}
              >
                <div style={{ fontFamily: E.fMono, fontSize: 11, color: E.text3, paddingTop: 2 }}>{s.id}</div>
                <div style={{ fontSize: 12.5, lineHeight: 1.55 }}>
                  <div style={{ color: E.text1, marginBottom: 4 }}>
                    <span style={{ color: E.text3, fontFamily: E.fMono, fontSize: 10, marginRight: 6 }}>USER</span>
                    {s.user}
                  </div>
                  <div style={{ color: E.text3, fontSize: 11.5, marginBottom: 4 }}>
                    <span style={{ fontFamily: E.fMono, fontSize: 10, marginRight: 6 }}>EXPECT</span>
                    {s.expected}
                  </div>
                  <div style={{ color: E.text1, fontSize: 12 }}>
                    <span style={{ color: E.fail, fontFamily: E.fMono, fontSize: 10, marginRight: 6 }}>GOT</span>
                    <span style={{ background: E.failDim, padding: '1px 5px', borderRadius: 3 }}>{s.got}</span>
                  </div>
                </div>
                <div>
                  <Pill mono color={E.fail} bg={E.failDim}>
                    {s.cluster}
                  </Pill>
                </div>
                <div style={{ textAlign: 'right', fontFamily: E.fSerif, fontSize: 16, color: E.fail }}>{s.score}</div>
              </div>
            ))}
            <div
              style={{
                padding: '11px 18px',
                borderTop: `1px solid ${E.hair}`,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <Btn kind="bare" size="sm" disabled title="Coming soon">
                View all {detail.failure_clusters.total_failures} failures →
              </Btn>
            </div>
          </Card>
        )}

        <div style={{ height: 30 }} />
      </div>
    </AppShell>
  );
}
