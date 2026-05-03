/**
 * ExperimentsList - browseable table of all evaluation runs.
 * Selection state is client-side; "Compare 2 selected" routes to RunDetail
 * with ?compare=<otherId>.
 */

import { useEffect, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { AppShell } from '../AppShell';
import { Card, Eyebrow, Pill, Btn, StatusDot, Spark } from '../ui';
import { v2 } from '../api/client';
import type { ExperimentRow } from '../api/types';
import { E } from '../tokens';

const COLS = '24px 2.4fr 90px 90px 80px 70px 90px 110px';
const FILTERS = ['Status: Any', 'Tag: Any', 'Author: Any', 'Sort: Recent'];

function deltaColor(d: string): string {
  if (d === 'baseline') return E.steel;
  if (d === '-' || d === '—' || d === '') return E.text3;
  if (d.startsWith('-') || d.startsWith('−')) return E.fail;
  return E.pass;
}

function sparkColor(status: string): string {
  if (status === 'warn') return E.warn;
  if (status === 'running') return E.ember;
  if (status === 'failed') return E.fail;
  return E.pass;
}

export default function ExperimentsList() {
  const [rows, setRows] = useState<ExperimentRow[] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [query, setQuery] = useState('');
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const isNew = searchParams.get('new') === '1';

  useEffect(() => {
    v2.experiments()
      .then(setRows)
      .catch((e) => setErr(String(e)));
  }, []);

  function toggle(id: string) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  function compareSelected() {
    if (selected.size !== 2) return;
    const [a, b] = Array.from(selected);
    navigate(`/experiments/${encodeURIComponent(a)}?compare=${encodeURIComponent(b)}`);
  }

  const filtered = rows
    ? rows.filter((r) => {
        if (!query.trim()) return true;
        const q = query.toLowerCase();
        return (
          r.name.toLowerCase().includes(q) ||
          r.author.toLowerCase().includes(q) ||
          r.tags.some((t) => t.toLowerCase().includes(q))
        );
      })
    : null;

  return (
    <AppShell contextChip={{ name: 'Experiments', version: '' }}>
      <div style={{ padding: '32px 36px' }}>
        <div style={{ display: 'flex', alignItems: 'flex-end', gap: 16, marginBottom: 6 }}>
          <div>
            <Eyebrow>All evaluations</Eyebrow>
            <h1
              style={{
                fontFamily: E.fSerif,
                fontSize: 32,
                fontWeight: 400,
                margin: 0,
                color: E.text0,
                letterSpacing: '-0.015em',
                lineHeight: 1.1,
              }}
            >
              Experiments
            </h1>
          </div>
          <span style={{ flex: 1 }} />
          <Btn kind="secondary" size="md" disabled title="Coming soon">
            ↗ Export CSV
          </Btn>
          <Btn kind="primary" size="md" onClick={() => navigate('/experiments?new=1')}>
            ＋ New evaluation
          </Btn>
        </div>

        {isNew && (
          <Card style={{ marginTop: 14, padding: 14, borderColor: E.emberRim }}>
            <Eyebrow style={{ color: E.ember }}>New evaluation</Eyebrow>
            <div style={{ marginTop: 6, fontSize: 13, color: E.text2 }}>
              Wizard coming soon. For now run evaluations from the CLI: <span style={{ fontFamily: E.fMono }}>evalyn run</span>.
            </div>
          </Card>
        )}

        {err && (
          <Card style={{ marginTop: 14, padding: 16, borderColor: E.fail }}>
            <Eyebrow style={{ color: E.fail }}>Error loading experiments</Eyebrow>
            <div style={{ fontFamily: E.fMono, fontSize: 12, color: E.text2, marginTop: 6 }}>{err}</div>
          </Card>
        )}

        {/* FILTER + COMPARE BAR */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            marginTop: 16,
            padding: 8,
            background: E.panel,
            border: `1px solid ${E.hair}`,
            borderRadius: 10,
          }}
        >
          <input
            placeholder="Search by name, author, tag..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            style={{
              flex: 1,
              background: 'transparent',
              border: 'none',
              outline: 'none',
              color: E.text1,
              fontSize: 13,
              padding: '4px 10px',
            }}
          />
          {FILTERS.map((f) => (
            <Btn key={f} kind="ghost" size="sm" style={{ fontSize: 11 }} disabled title="Coming soon">
              {f} ▾
            </Btn>
          ))}
          <span style={{ width: 1, height: 20, background: E.hair2 }} />
          <Btn kind="secondary" size="sm" disabled={selected.size !== 2} onClick={compareSelected}>
            Compare {selected.size} selected →
          </Btn>
        </div>

        {/* TABLE */}
        {!rows && !err && <div style={{ marginTop: 14, color: E.text3, fontSize: 13 }}>Loading...</div>}

        {rows && rows.length === 0 && (
          <Card style={{ marginTop: 14, padding: 28, textAlign: 'center' }}>
            <Eyebrow>No experiments yet</Eyebrow>
            <div style={{ marginTop: 8, fontSize: 14, color: E.text2 }}>
              Run your first evaluation to see results here.
            </div>
            <div style={{ marginTop: 14 }}>
              <Btn kind="primary" size="md" onClick={() => navigate('/experiments?new=1')}>
                Run your first eval
              </Btn>
            </div>
          </Card>
        )}

        {filtered && filtered.length > 0 && (
          <Card style={{ marginTop: 14, padding: 0, overflow: 'hidden' }}>
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: COLS,
                padding: '11px 18px',
                borderBottom: `1px solid ${E.hair}`,
                fontFamily: E.fMono,
                fontSize: 10,
                color: E.text3,
                textTransform: 'uppercase',
                letterSpacing: '0.08em',
              }}
            >
              <span></span>
              <span>Experiment</span>
              <span>Pass</span>
              <span>Δ</span>
              <span>Items</span>
              <span>Dur</span>
              <span>Cost</span>
              <span style={{ textAlign: 'right' }}>Trend</span>
            </div>
            {filtered.map((r, i) => {
              const isSelected = selected.has(r.id);
              return (
                <div
                  key={r.id}
                  style={{
                    display: 'grid',
                    gridTemplateColumns: COLS,
                    padding: '14px 18px',
                    borderTop: i ? `1px solid ${E.hair}` : 'none',
                    alignItems: 'center',
                    gap: 8,
                    background: isSelected ? E.emberDim : 'transparent',
                    cursor: 'pointer',
                  }}
                  onClick={(ev) => {
                    if ((ev.target as HTMLElement).tagName === 'INPUT') return;
                    navigate(`/experiments/${encodeURIComponent(r.id)}`);
                  }}
                >
                  <input
                    type="checkbox"
                    checked={isSelected}
                    onChange={() => toggle(r.id)}
                    onClick={(ev) => ev.stopPropagation()}
                    style={{ accentColor: E.ember }}
                  />
                  <div style={{ minWidth: 0 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                      <StatusDot status={r.status} animated={r.status === 'running'} />
                      <span style={{ fontSize: 13.5, color: E.text0, fontWeight: 500 }}>{r.name}</span>
                      {r.tags.map((t) => (
                        <Pill
                          key={t}
                          mono
                          style={{ fontSize: 9.5, padding: '1px 7px', background: E.panel3, color: E.text2 }}
                        >
                          {t}
                        </Pill>
                      ))}
                      {r.err && (
                        <Pill mono color={E.fail} bg={E.failDim} style={{ fontSize: 9.5 }}>
                          {r.err}
                        </Pill>
                      )}
                    </div>
                    <div style={{ fontSize: 11, color: E.text3, marginTop: 3, fontFamily: E.fMono }}>
                      {r.id} - {r.author} - {r.when_iso}
                    </div>
                  </div>
                  <div style={{ fontFamily: E.fSerif, fontSize: 17, color: r.pass != null ? E.text0 : E.text3 }}>
                    {r.pass != null ? `${r.pass}%` : '-'}
                  </div>
                  <div style={{ fontFamily: E.fMono, fontSize: 12, color: deltaColor(r.delta) }}>{r.delta}</div>
                  <div style={{ fontFamily: E.fMono, fontSize: 11, color: E.text2 }}>{r.items}</div>
                  <div style={{ fontFamily: E.fMono, fontSize: 11, color: E.text2 }}>{r.duration}</div>
                  <div style={{ fontFamily: E.fMono, fontSize: 11, color: E.text2 }}>{r.cost}</div>
                  <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
                    {r.spark && r.spark.length > 0 ? (
                      <Spark data={r.spark} color={sparkColor(r.status)} dot w={90} h={24} />
                    ) : (
                      <span style={{ fontSize: 10, color: E.text3, fontFamily: E.fMono }}>n/a</span>
                    )}
                  </div>
                </div>
              );
            })}
          </Card>
        )}

        {filtered && filtered.length === 0 && rows && rows.length > 0 && (
          <div style={{ marginTop: 14, padding: 18, fontSize: 13, color: E.text3 }}>
            No experiments match "{query}".
          </div>
        )}

        <div style={{ height: 30 }} />
      </div>
    </AppShell>
  );
}
