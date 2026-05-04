/**
 * Home - landing snapshot. Hero quality, sub-metrics, active experiments,
 * recent activity, attention queue, co-pilot brief. Sources from v2.home().
 */

import { useEffect, useMemo, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { AppShell } from '../AppShell';
import { useV2Store } from '../store/store';
import { FIRST_RUN_TOUR_ID } from '../tour/scripts/firstRun';
import { shouldFireFirstRunTour } from '../tour/firstRunGate';
import {
  Card,
  Eyebrow,
  Glossary,
  Pill,
  Btn,
  StatusDot,
  Spark,
  Bar,
  LineChart,
  Skeleton,
  Spinner,
  UpdatingChip,
} from '../ui';
import { Welcome } from '../ui/Welcome';
import { v2 } from '../api/client';
import { useV2Resource } from '../hooks/useV2Resource';
import { E } from '../tokens';

function formatDelta(d: number, inverse: boolean): string {
  if (d === 0) return 'flat';
  // Inverse metrics: lower is better, so a positive delta is bad.
  if (inverse) return d > 0 ? `+${d}` : `${d}`;
  return d > 0 ? `+${d}` : `${d}`;
}

function deltaColor(d: number, inverse: boolean): string {
  if (d === 0) return E.text3;
  const goodDirection = inverse ? d < 0 : d > 0;
  return goodDirection ? E.pass : E.fail;
}

function severityColor(sev: 'fail' | 'warn' | 'info'): string {
  if (sev === 'fail') return E.fail;
  if (sev === 'warn') return E.warn;
  return E.steel;
}

function fmtUsd(n: number): string {
  return `$${n.toFixed(2)}`;
}

function shortTime(iso: string): string {
  const then = Date.parse(iso);
  if (Number.isNaN(then)) return '';
  const diffMs = Date.now() - then;
  const m = Math.floor(diffMs / 60_000);
  if (m < 1) return 'now';
  if (m < 60) return `${m}m`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h`;
  const d = Math.floor(h / 24);
  return `${d}d`;
}

export default function Home() {
  const { data: snap, err, refetch, reloading, isInitialLoad } = useV2Resource(
    'home',
    v2.home,
  );
  const navigate = useNavigate();
  const setTour = useV2Store((s) => s.setTour);
  const setDockOpen = useV2Store((s) => s.setDockOpen);
  const tourTriggered = useRef(false);

  // useTour() is mounted globally in AppShell; Home only triggers the
  // firstRun tour via setTour below. Earlier versions mounted useTour here.
  //
  // First-visit detection. Fires the firstRun tour when:
  //   1. user has the global setting enabled (default on - missing key counts as on)
  //   2. user has not previously completed the tour
  //   3. Home has loaded its data (no skeleton, no error)
  // Guarded by a ref so it cannot re-fire on re-render.
  useEffect(() => {
    if (tourTriggered.current) return;
    if (!snap || err) return;
    if (!shouldFireFirstRunTour()) return;
    tourTriggered.current = true;
    // Collapse the copilot sheet (mobile) / dock (desktop) so the tour has
    // a single focal surface.
    setDockOpen(false);
    // 500ms post-mount delay matches the useTour anchor-not-found timeout, so
    // any anchor that is still resolving when we trigger will degrade to the
    // skip+narrate fallback rather than a hard miss.
    const t = setTimeout(() => {
      setTour(FIRST_RUN_TOUR_ID);
    }, 500);
    return () => clearTimeout(t);
  }, [snap, err, setTour, setDockOpen]);

  const briefTime = useMemo(() => {
    if (!snap?.brief) return '';
    return shortTime(snap.brief.generated_at_iso);
  }, [snap]);

  if (err && !snap) {
    return (
      <AppShell>
        <div style={{ padding: '32px 36px', maxWidth: 1100 }}>
          <Card style={{ padding: 16, borderColor: E.fail }}>
            <Eyebrow style={{ color: E.fail }}>Error loading home</Eyebrow>
            <div style={{ fontFamily: E.fMono, fontSize: 12, color: E.text2, marginTop: 6 }}>{err}</div>
          </Card>
        </div>
      </AppShell>
    );
  }

  if (!snap) {
    return (
      <AppShell>
        <div style={{ padding: '32px 36px', maxWidth: 1100 }}>
          <Eyebrow style={{ marginBottom: 8 }}>Project home</Eyebrow>
          <Skeleton w={300} h={36} style={{ marginTop: 4 }} />
          <div style={{ marginTop: 24, display: 'grid', gridTemplateColumns: '1.6fr 1fr 1fr', gap: 18 }}>
            <Card style={{ padding: 22 }}>
              <Eyebrow>Overall quality - 30d</Eyebrow>
              <div style={{ display: 'flex', alignItems: 'baseline', gap: 10, marginTop: 10 }}>
                <Skeleton w={120} h={56} />
              </div>
              <div style={{ marginTop: 14 }}>
                <Skeleton w="100%" h={150} style={{ borderRadius: 8 }} />
              </div>
            </Card>
            <Card style={{ padding: 22 }}>
              <Eyebrow>Sub-metrics today</Eyebrow>
              <div style={{ marginTop: 14, display: 'flex', flexDirection: 'column', gap: 14 }}>
                {[0, 1, 2, 3].map((i) => (
                  <div key={i}>
                    <Skeleton w={140} h={11} />
                    <div style={{ marginTop: 6 }}>
                      <Skeleton w="100%" h={3} />
                    </div>
                  </div>
                ))}
              </div>
            </Card>
            <Card style={{ padding: 22 }}>
              <Eyebrow>Spend this month</Eyebrow>
              <Skeleton w={100} h={36} style={{ marginTop: 10 }} />
              <Skeleton w="80%" h={11} style={{ marginTop: 10 }} />
              <Skeleton w="100%" h={28} style={{ marginTop: 12, borderRadius: 4 }} />
            </Card>
          </div>
          <div style={{ marginTop: 18, display: 'grid', gridTemplateColumns: '1.2fr 1fr 1fr', gap: 18 }}>
            {[0, 1, 2].map((i) => (
              <Card key={i} style={{ padding: 18 }}>
                <Skeleton w={120} h={11} />
                <div style={{ marginTop: 14, display: 'flex', flexDirection: 'column', gap: 10 }}>
                  <Skeleton w="90%" h={12} />
                  <Skeleton w="80%" h={12} />
                  <Skeleton w="85%" h={12} />
                </div>
              </Card>
            ))}
          </div>
        </div>
      </AppShell>
    );
  }

  const q = snap.quality;
  const hasQuality = q.current != null;
  const timelineData = q.timeline.map((p) => p.y);
  const timelineLabels = q.timeline.map((p) => p.x);

  if (!hasQuality) {
    return (
      <AppShell contextChip={snap.project}>
        <div style={{ padding: '32px 36px', maxWidth: 1100 }}>
          <Eyebrow style={{ marginBottom: 8 }}>Project home</Eyebrow>
          <h1
            style={{
              fontFamily: E.fSerif,
              fontSize: 38,
              fontWeight: 400,
              margin: 0,
              color: E.text0,
              letterSpacing: '-0.015em',
              lineHeight: 1.05,
            }}
          >
            {snap.project.name}
          </h1>
          <p style={{ fontSize: 14, color: E.text2, marginTop: 8, lineHeight: 1.55, maxWidth: 640 }}>
            No evaluation runs yet. Load the demo to explore a populated workspace, or run your first eval from the CLI.
          </p>

          <div style={{ marginTop: 24, display: 'flex', flexDirection: 'column', gap: 18 }}>
            <Welcome />

            <Card style={{ padding: 0, overflow: 'hidden' }}>
              <div style={{ padding: '14px 18px', borderBottom: `1px solid ${E.hair}` }}>
                <span style={{ fontSize: 13, color: E.text0, fontWeight: 500 }}>Recent activity</span>
              </div>
              <div style={{ padding: 18, fontSize: 12.5, color: E.text3 }}>No runs yet.</div>
            </Card>
          </div>

          <div style={{ height: 30 }} />
        </div>
      </AppShell>
    );
  }

  return (
    <AppShell contextChip={snap.project}>
      <div style={{ padding: '32px 36px', maxWidth: 1100 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
          <Eyebrow>Project home</Eyebrow>
          <UpdatingChip visible={reloading && !isInitialLoad} />
        </div>
        <h1
          style={{
            fontFamily: E.fSerif,
            fontSize: 38,
            fontWeight: 400,
            margin: 0,
            color: E.text0,
            letterSpacing: '-0.015em',
            lineHeight: 1.05,
          }}
        >
          {snap.project.name}
        </h1>
        <p style={{ fontSize: 14, color: E.text2, marginTop: 8, lineHeight: 1.55, maxWidth: 640 }}>
          A jobs-to-be-done view of the assistant's quality. Last 30 days - {q.graded_items.toLocaleString()} graded items - {snap.active_experiments.length} active experiments.
        </p>

        {/* HERO QUALITY + SUB-METRICS + COST */}
        <div style={{ marginTop: 24, display: 'grid', gridTemplateColumns: snap.cost ? '1.6fr 1fr 1fr' : '1.6fr 1fr', gap: 18 }}>
          <Card style={{ padding: 22 }} data-coachmark="home-quality">
            <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between' }}>
              <div>
                <Eyebrow>Overall quality - 30d</Eyebrow>
                {hasQuality ? (
                  <>
                    <div style={{ display: 'flex', alignItems: 'baseline', gap: 10, marginTop: 8 }}>
                      <span style={{ fontFamily: E.fSerif, fontSize: 52, color: E.text0, lineHeight: 1, fontWeight: 400 }}>
                        {q.current!.toFixed(1)}
                      </span>
                      <span style={{ fontSize: 18, color: E.text2 }}>%</span>
                      {q.delta_30d != null && (
                        <Pill
                          mono
                          color={q.delta_30d >= 0 ? E.pass : E.fail}
                          bg={q.delta_30d >= 0 ? E.passDim : E.failDim}
                        >
                          {q.delta_30d >= 0 ? '↑' : '↓'} {Math.abs(q.delta_30d).toFixed(1)} pts vs. 30d ago
                        </Pill>
                      )}
                    </div>
                    <div style={{ fontSize: 12, color: E.text3, marginTop: 8 }}>
                      Weighted across {q.weighted_across_metrics} sub-metrics - {q.graded_items.toLocaleString()} graded items
                    </div>
                  </>
                ) : (
                  <div style={{ marginTop: 14, fontSize: 13, color: E.text2, lineHeight: 1.55 }}>
                    No data yet - load demo or run an evaluation.
                  </div>
                )}
              </div>
              {hasQuality && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 4, alignItems: 'flex-end', fontSize: 10, fontFamily: E.fMono }}>
                  <span style={{ color: E.ember, display: 'inline-flex', alignItems: 'center', gap: 5 }}>
                    <span style={{ width: 8, height: 2, background: E.ember }} />
                    This project
                  </span>
                  <span style={{ color: E.steel, display: 'inline-flex', alignItems: 'center', gap: 5 }}>
                    <span style={{ width: 8, height: 0, borderTop: `2px dashed ${E.steel}` }} />
                    <Glossary term="Ship gate is the minimum quality threshold for shipping. Runs above it are deemed releasable.">
                      Ship gate
                    </Glossary>{' '}
                    ({q.ship_gate}%)
                  </span>
                </div>
              )}
            </div>
            {hasQuality && timelineData.length > 1 && (
              <div style={{ marginTop: 14 }}>
                <LineChart
                  w={580}
                  h={150}
                  yMin={Math.min(...timelineData) - 2}
                  yMax={Math.max(...timelineData) + 2}
                  baseline={q.ship_gate}
                  xLabels={timelineLabels}
                  series={[{ color: E.ember, fill: true, width: 2, data: timelineData }]}
                />
              </div>
            )}
          </Card>

          <Card style={{ padding: 22 }} data-coachmark="home-submetrics">
            <Eyebrow>Sub-metrics today</Eyebrow>
            {snap.sub_metrics.length === 0 ? (
              <div style={{ marginTop: 14, fontSize: 13, color: E.text3 }}>No sub-metrics yet.</div>
            ) : (
              <div style={{ marginTop: 14, display: 'flex', flexDirection: 'column', gap: 12 }}>
                {snap.sub_metrics.map((m) => {
                  const c = deltaColor(m.delta, m.inverse);
                  return (
                    <div key={m.label}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 4 }}>
                        <span style={{ fontSize: 12, color: E.text2 }}>{m.label}</span>
                        <div style={{ display: 'flex', gap: 8, alignItems: 'baseline' }}>
                          <span style={{ fontFamily: E.fSerif, fontSize: 16, color: E.text0 }}>{m.value}%</span>
                          <span style={{ fontFamily: E.fMono, fontSize: 10, color: c }}>
                            {m.delta === 0 ? '-' : formatDelta(m.delta, m.inverse)}
                          </span>
                        </div>
                      </div>
                      <Bar
                        value={m.inverse ? Math.max(0, 100 - m.value * 20) : m.value}
                        max={100}
                        w={'100%'}
                        h={3}
                        color={m.inverse ? E.fail : c === E.pass ? E.pass : E.steel}
                      />
                    </div>
                  );
                })}
              </div>
            )}
          </Card>

          {snap.cost && (
            <Card style={{ padding: 22 }}>
              <Eyebrow>Spend this month</Eyebrow>
              <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, marginTop: 10 }}>
                <span style={{ fontFamily: E.fSerif, fontSize: 36, color: E.text0, lineHeight: 1, fontWeight: 400 }}>
                  {fmtUsd(snap.cost.total_30d)}
                </span>
              </div>
              <div style={{ marginTop: 8, fontSize: 11.5, color: E.text2, fontFamily: E.fMono, lineHeight: 1.5 }}>
                {fmtUsd(snap.cost.total_7d)} last 7 days
                {snap.cost.projected_monthly != null && (
                  <>
                    {' - '}
                    <span style={{ color: E.text3 }}>Projected monthly: {fmtUsd(snap.cost.projected_monthly)}</span>
                  </>
                )}
              </div>
              {snap.cost.daily_30d.length > 0 && (
                <div style={{ marginTop: 12 }}>
                  <Spark data={snap.cost.daily_30d} color={E.steel} w={260} h={28} />
                </div>
              )}
              <div style={{ marginTop: 10, fontSize: 11, color: E.text3, lineHeight: 1.45 }}>
                {snap.cost.runs_with_cost} of {snap.cost.runs_total} runs incurred LLM-judge cost. Free runs use programmatic-only metrics.
              </div>
            </Card>
          )}
        </div>

        {/* THREE-COL: Active experiments | Recent activity | Attention */}
        <div style={{ marginTop: 18, display: 'grid', gridTemplateColumns: '1.2fr 1fr 1fr', gap: 18 }}>
          <Card style={{ padding: 0, overflow: 'hidden' }} data-coachmark="home-experiments">
            <div style={{ padding: '14px 18px', borderBottom: `1px solid ${E.hair}`, display: 'flex', alignItems: 'center' }}>
              <span style={{ fontSize: 13, color: E.text0, fontWeight: 500 }}>Active experiments</span>
              <span style={{ flex: 1 }} />
              <Btn kind="bare" size="sm" onClick={() => navigate('/experiments')}>
                View all →
              </Btn>
            </div>
            {snap.active_experiments.length === 0 ? (
              <div style={{ padding: 18, fontSize: 12.5, color: E.text3 }}>No active experiments.</div>
            ) : (
              snap.active_experiments.map((e, i) => {
                const sparkColor = e.status === 'warn' ? E.warn : e.status === 'running' ? E.ember : E.pass;
                const subline = e.status === 'running' && e.progress
                  ? `${e.progress.done} / ${e.progress.total} items`
                  : e.pass != null
                    ? `pass ${e.pass}%${e.delta_pts != null ? ` - ${e.delta_pts >= 0 ? '+' : ''}${e.delta_pts} pts` : ''}`
                    : '';
                return (
                  <button
                    key={e.id}
                    type="button"
                    onClick={() => navigate(`/experiments/${e.id}`)}
                    style={{
                      width: '100%',
                      padding: '14px 18px',
                      borderTop: i ? `1px solid ${E.hair}` : 'none',
                      display: 'flex',
                      alignItems: 'center',
                      gap: 14,
                      background: 'transparent',
                      border: 'none',
                      borderTopLeftRadius: 0,
                      cursor: 'pointer',
                      textAlign: 'left',
                    }}
                  >
                    <StatusDot status={e.status} animated={e.status === 'running'} />
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{ fontSize: 13, color: E.text0, fontWeight: 500 }}>{e.name}</div>
                      <div style={{ fontSize: 11, color: E.text3, marginTop: 2, fontFamily: E.fMono }}>{subline}</div>
                    </div>
                    {e.spark.length > 0 && <Spark data={e.spark} color={sparkColor} dot />}
                  </button>
                );
              })
            )}
          </Card>

          <Card style={{ padding: 0, overflow: 'hidden' }} data-coachmark="home-activity">
            <div style={{ padding: '14px 18px', borderBottom: `1px solid ${E.hair}` }}>
              <span style={{ fontSize: 13, color: E.text0, fontWeight: 500 }}>Recent activity</span>
            </div>
            {snap.recent_activity.length === 0 ? (
              <div style={{ padding: 18, fontSize: 12.5, color: E.text3 }}>No recent activity.</div>
            ) : (
              snap.recent_activity.map((a, i) => (
                <div
                  key={`${a.who}-${a.when_iso}-${i}`}
                  style={{
                    padding: '11px 18px',
                    borderTop: i ? `1px solid ${E.hair}` : 'none',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 10,
                    fontSize: 12,
                  }}
                >
                  <span
                    style={{
                      width: 22,
                      height: 22,
                      borderRadius: 5,
                      background: a.accent ? E.ember : E.panel3,
                      color: a.accent ? E.emberInk : E.text2,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontSize: 11,
                      flexShrink: 0,
                      fontWeight: a.accent ? 600 : 400,
                    }}
                  >
                    {a.icon}
                  </span>
                  <span style={{ color: E.text2 }}>
                    <b style={{ color: E.text0 }}>{a.who}</b> {a.what} <span style={{ color: E.text1 }}>{a.target}</span>
                  </span>
                  <span style={{ flex: 1 }} />
                  <span style={{ color: E.text3, fontFamily: E.fMono, fontSize: 10 }}>{shortTime(a.when_iso)}</span>
                </div>
              ))
            )}
          </Card>

          <Card style={{ padding: 0, overflow: 'hidden' }}>
            <div style={{ padding: '14px 18px', borderBottom: `1px solid ${E.hair}`, display: 'flex', alignItems: 'center' }}>
              <span style={{ fontSize: 13, color: E.text0, fontWeight: 500 }}>Needs your attention</span>
              <span style={{ flex: 1 }} />
              <Pill mono color={E.ember} bg={E.emberDim}>
                {snap.attention.length}
              </Pill>
            </div>
            {snap.attention.length === 0 ? (
              <div style={{ padding: 18, fontSize: 12.5, color: E.text3 }}>Nothing waiting.</div>
            ) : (
              snap.attention.map((a, i) => (
                <div key={`${a.title}-${i}`} style={{ padding: '12px 18px', borderTop: i ? `1px solid ${E.hair}` : 'none' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                    <span style={{ width: 4, height: 14, borderRadius: 2, background: severityColor(a.severity) }} />
                    <span style={{ fontSize: 12.5, color: E.text0, fontWeight: 500 }}>{a.title}</span>
                  </div>
                  <div style={{ fontSize: 11.5, color: E.text3, marginLeft: 12, marginBottom: 6 }}>{a.subtitle}</div>
                  <Btn kind="secondary" size="sm" style={{ marginLeft: 12 }} onClick={() => navigate(a.cta_target)}>
                    {a.cta} →
                  </Btn>
                </div>
              ))
            )}
          </Card>
        </div>

        {/* CO-PILOT BRIEF */}
        {snap.brief && (
          <Card style={{ marginTop: 18, padding: 0, overflow: 'hidden' }} data-coachmark="home-copilot-brief">
            <div style={{ padding: '14px 22px', borderBottom: `1px solid ${E.hair}`, display: 'flex', alignItems: 'center', gap: 8 }}>
              <span
                style={{
                  width: 22,
                  height: 22,
                  borderRadius: 5,
                  background: `linear-gradient(135deg, ${E.ember}, #b8501f)`,
                  color: E.emberInk,
                  fontWeight: 700,
                  fontFamily: E.fSerif,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: 13,
                }}
              >
                e
              </span>
              <span style={{ fontSize: 13, color: E.text0, fontWeight: 500 }}>Co-pilot's morning brief</span>
              <span style={{ fontSize: 11, color: E.text3 }}>- generated {briefTime}</span>
              <span style={{ flex: 1 }} />
              <Btn
                kind="bare"
                size="sm"
                onClick={() => void refetch()}
                disabled={reloading}
                title={reloading ? 'Refreshing...' : 'Refetch the morning brief'}
              >
                {reloading ? <Spinner size={11} /> : '↻'}
              </Btn>
            </div>
            <div style={{ padding: '18px 22px', fontSize: 13.5, color: E.text1, lineHeight: 1.65 }}>
              <div style={{ whiteSpace: 'pre-wrap' }}>{snap.brief.body_md}</div>
              {snap.brief.actions.length > 0 && (
                <div style={{ marginTop: 12, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                  {snap.brief.actions.map((act) => (
                    <Btn
                      key={act.label}
                      kind={act.kind}
                      size="sm"
                      onClick={() => navigate(act.intent)}
                    >
                      {act.label}
                    </Btn>
                  ))}
                </div>
              )}
            </div>
          </Card>
        )}

        <div style={{ height: 30 }} />
      </div>
    </AppShell>
  );
}
