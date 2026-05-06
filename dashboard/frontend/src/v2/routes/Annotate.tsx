/**
 * Annotate landing - source picker + resumable sessions.
 *
 * Three jobs:
 *  1. List in-progress + recently-completed sessions for resume.
 *  2. Let the user start a new session against a run or dataset.
 *  3. Surface the calibration-ready signal (annotation count >=10) so
 *     the user knows when their work has reached threshold.
 */

import { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { AppShell } from '../AppShell';
import { Btn, Card, Eyebrow, Pill, Skeleton, StatusDot, UpdatingChip } from '../ui';
import { useV2Resource } from '../hooks/useV2Resource';
import { v2 } from '../api/client';
import { annotationApi } from '../api/annotation';
import type {
  AnnotationSessionList,
  AnnotationSessionMeta,
  AnnotationSourceKind,
  ExperimentList,
  DatasetList,
} from '../api/types';
import { E } from '../tokens';

function shortRel(iso: string): string {
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return '';
  const m = Math.floor((Date.now() - t) / 60_000);
  if (m < 1) return 'just now';
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  return `${Math.floor(h / 24)}d ago`;
}

function statusColor(s: AnnotationSessionMeta['status']): string {
  if (s === 'in_progress') return E.ember;
  if (s === 'completed') return E.pass;
  return E.text3;
}

function statusLabel(s: AnnotationSessionMeta['status']): string {
  if (s === 'in_progress') return 'in progress';
  return s;
}

function SourcePicker({
  experiments,
  datasets,
  onCreated,
}: {
  experiments: ExperimentList | null;
  datasets: DatasetList | null;
  onCreated: (session: AnnotationSessionMeta) => void;
}) {
  const [sourceKind, setSourceKind] = useState<AnnotationSourceKind>('run');
  const [sourceId, setSourceId] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const options = sourceKind === 'run' ? experiments : datasets;
  const optionItems = useMemo(() => {
    if (!options) return [] as { id: string; label: string }[];
    return options.map((o) => {
      // Cards from /datasets are { name }, rows from /experiments are { id, name }.
      const id = (o as { id?: string }).id ?? (o as { name: string }).name;
      const label =
        sourceKind === 'run'
          ? `${(o as { id: string }).id} - ${(o as { name: string }).name ?? ''}`
          : (o as { name: string; n?: number }).name +
            ((o as { n?: number }).n != null
              ? ` (${(o as { n?: number }).n} items)`
              : '');
      return { id, label };
    });
  }, [options, sourceKind]);

  // Reset selection when switching kind so we don't submit a stale id.
  useEffect(() => {
    setSourceId('');
    setErr(null);
  }, [sourceKind]);

  async function submit() {
    if (!sourceId) {
      setErr('Pick a source first.');
      return;
    }
    setSubmitting(true);
    setErr(null);
    try {
      const session = await annotationApi.createSession({
        source_kind: sourceKind,
        source_id: sourceId,
      });
      onCreated(session);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <Card style={{ padding: 22 }}>
      <Eyebrow>Start a new session</Eyebrow>
      <div style={{ marginTop: 14, display: 'flex', flexDirection: 'column', gap: 10 }}>
        <div style={{ display: 'flex', gap: 8 }}>
          <Btn
            kind={sourceKind === 'run' ? 'primary' : 'secondary'}
            size="sm"
            onClick={() => setSourceKind('run')}
          >
            Run
          </Btn>
          <Btn
            kind={sourceKind === 'dataset' ? 'primary' : 'secondary'}
            size="sm"
            onClick={() => setSourceKind('dataset')}
          >
            Dataset
          </Btn>
        </div>
        <select
          value={sourceId}
          onChange={(e) => setSourceId(e.target.value)}
          disabled={submitting || optionItems.length === 0}
          style={{
            background: E.panel2,
            color: E.text1,
            border: `1px solid ${E.hair2}`,
            borderRadius: 6,
            padding: '6px 10px',
            fontSize: 13,
            fontFamily: E.fSans,
            outline: 'none',
          }}
        >
          <option value="">
            {options == null
              ? 'Loading...'
              : optionItems.length === 0
                ? `No ${sourceKind}s available`
                : `Pick a ${sourceKind}`}
          </option>
          {optionItems.map((o) => (
            <option key={o.id} value={o.id}>
              {o.label}
            </option>
          ))}
        </select>
        <Btn kind="primary" size="md" onClick={submit} disabled={submitting || !sourceId}>
          {submitting ? 'Creating...' : 'Start annotating →'}
        </Btn>
        {err && (
          <div
            style={{
              padding: 10,
              background: E.failDim,
              border: `1px solid ${E.fail}33`,
              borderRadius: 6,
              fontSize: 12,
              color: E.fail,
              fontFamily: E.fMono,
            }}
          >
            {err}
          </div>
        )}
        <div style={{ fontSize: 11, color: E.text3, lineHeight: 1.55 }}>
          A session annotates every metric on each item. Pre-labels from the AI judge are
          shown so you can confirm or override - typical pace is around three seconds per
          item.
        </div>
      </div>
    </Card>
  );
}

export default function Annotate() {
  const navigate = useNavigate();
  const {
    data: sessionList,
    err,
    refetch,
    reloading,
    isInitialLoad,
  } = useV2Resource<AnnotationSessionList>('annotation/sessions', () =>
    annotationApi.listSessions(),
  );
  const { data: experiments } = useV2Resource<ExperimentList>('experiments', v2.experiments);
  const { data: datasets } = useV2Resource<DatasetList>('datasets', v2.datasets);

  const sessions = sessionList?.sessions ?? [];
  const inProgress = sessions.filter((s) => s.status === 'in_progress');
  const recent = sessions.filter((s) => s.status !== 'in_progress').slice(0, 5);

  return (
    <AppShell>
      <div style={{ padding: '32px 36px', maxWidth: 1100 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
          <Eyebrow>Human annotation</Eyebrow>
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
          Annotate
        </h1>
        <p style={{ fontSize: 14, color: E.text2, marginTop: 8, lineHeight: 1.55, maxWidth: 640 }}>
          Label items by hand to calibrate your judges. Pre-labels show what the AI
          decided; you confirm or override. Once a metric has 10 verdicts the dashboard
          will surface a one-click "Run calibrate" on Home.
        </p>

        {err && (
          <Card style={{ marginTop: 22, padding: 16, borderColor: E.fail }}>
            <Eyebrow style={{ color: E.fail }}>Error loading sessions</Eyebrow>
            <div style={{ fontFamily: E.fMono, fontSize: 12, color: E.text2, marginTop: 6 }}>{err}</div>
            <div style={{ marginTop: 10 }}>
              <Btn kind="secondary" size="sm" onClick={refetch}>
                Retry
              </Btn>
            </div>
          </Card>
        )}

        <div style={{ marginTop: 24, display: 'grid', gridTemplateColumns: '1.4fr 1fr', gap: 18 }}>
          {/* SESSIONS COLUMN */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            <Card style={{ padding: 0, overflow: 'hidden' }}>
              <div
                style={{
                  padding: '14px 18px',
                  borderBottom: `1px solid ${E.hair}`,
                  display: 'flex',
                  alignItems: 'center',
                  gap: 8,
                }}
              >
                <span style={{ fontSize: 13, color: E.text0, fontWeight: 500 }}>In progress</span>
                <Pill mono color={E.text3} bg={E.panel3}>
                  {inProgress.length}
                </Pill>
              </div>
              {!sessionList && isInitialLoad && (
                <div style={{ padding: 18 }}>
                  <Skeleton w="60%" h={12} />
                  <div style={{ marginTop: 10 }}>
                    <Skeleton w="80%" h={12} />
                  </div>
                </div>
              )}
              {sessionList && inProgress.length === 0 && (
                <div style={{ padding: 18, fontSize: 12.5, color: E.text3 }}>
                  No sessions in progress. Start one on the right.
                </div>
              )}
              {inProgress.map((s, i) => (
                <button
                  key={s.id}
                  type="button"
                  onClick={() => navigate(`/annotate/${encodeURIComponent(s.id)}`)}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.background = E.panel2;
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.background = 'transparent';
                  }}
                  style={{
                    display: 'block',
                    width: '100%',
                    textAlign: 'left',
                    padding: '14px 18px',
                    borderTop: i ? `1px solid ${E.hair}` : 'none',
                    background: 'transparent',
                    border: 'none',
                    cursor: 'pointer',
                    transition: 'background 140ms',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <StatusDot status="running" size={6} />
                    <span style={{ fontSize: 13, color: E.text0, fontWeight: 500 }}>
                      {s.source_kind}: {s.source_id}
                    </span>
                    <Pill mono style={{ fontSize: 9.5, padding: '1px 7px' }}>
                      {s.metric_ids.length} metric{s.metric_ids.length !== 1 ? 's' : ''}
                    </Pill>
                  </div>
                  <div style={{ fontSize: 11, color: E.text3, marginTop: 4, fontFamily: E.fMono }}>
                    {s.items_done}/{s.items_total} done · {shortRel(s.last_active_iso)}
                  </div>
                </button>
              ))}
            </Card>

            {recent.length > 0 && (
              <Card style={{ padding: 0, overflow: 'hidden' }}>
                <div style={{ padding: '12px 18px', borderBottom: `1px solid ${E.hair}` }}>
                  <span style={{ fontSize: 13, color: E.text0, fontWeight: 500 }}>Recent</span>
                </div>
                {recent.map((s, i) => (
                  <button
                    key={s.id}
                    type="button"
                    onClick={() => navigate(`/annotate/${encodeURIComponent(s.id)}`)}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.background = E.panel2;
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.background = 'transparent';
                    }}
                    style={{
                      display: 'block',
                      width: '100%',
                      textAlign: 'left',
                      padding: '11px 18px',
                      borderTop: i ? `1px solid ${E.hair}` : 'none',
                      background: 'transparent',
                      border: 'none',
                      cursor: 'pointer',
                      transition: 'background 140ms',
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <StatusDot status={s.status === 'completed' ? 'pass' : 'idle'} size={5} />
                      <span style={{ fontSize: 12.5, color: E.text1 }}>
                        {s.source_kind}: {s.source_id}
                      </span>
                      <Pill
                        mono
                        color={statusColor(s.status)}
                        style={{ fontSize: 9.5, padding: '1px 7px' }}
                      >
                        {statusLabel(s.status)}
                      </Pill>
                    </div>
                    <div style={{ fontSize: 11, color: E.text3, marginTop: 3, fontFamily: E.fMono }}>
                      {s.items_done}/{s.items_total} · {shortRel(s.last_active_iso)}
                    </div>
                  </button>
                ))}
              </Card>
            )}
          </div>

          {/* PICKER COLUMN */}
          <SourcePicker
            experiments={experiments}
            datasets={datasets}
            onCreated={(session) => {
              // Force-refresh the session list so the new one appears
              // immediately on return-nav, then route to it.
              void refetch();
              navigate(`/annotate/${encodeURIComponent(session.id)}`);
            }}
          />
        </div>

        <div style={{ height: 30 }} />
      </div>
    </AppShell>
  );
}
