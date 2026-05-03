/**
 * Datasets - list of evaluation input sets, one card per dataset.
 * Wires v2.datasets() into the design from screens-3.jsx.
 */

import { useState } from 'react';
import { AppShell } from '../AppShell';
import { Btn, Card, Eyebrow, Pill, Skeleton, StackBar, UpdatingChip } from '../ui';
import { v2 } from '../api/client';
import { loadDemo } from '../api/demo';
import { useV2Resource } from '../hooks/useV2Resource';
import { useProject } from '../hooks/useProject';
import { E } from '../tokens';

const COVERAGE_COLORS = [E.ember, E.steel, '#a78bfa', E.warn, E.text3];
const NEW_DATASET_HINT = 'Use `evalyn build-dataset` from the CLI';
const IMPORT_CSV_HINT = 'No CSV importer yet - use `evalyn build-dataset` from the CLI';
const COMING_SOON = 'Coming soon';

export default function Datasets() {
  const project = useProject();
  const { data, err, reloading, isInitialLoad } = useV2Resource(
    'datasets',
    v2.datasets,
  );
  const [demoLoading, setDemoLoading] = useState(false);
  const [demoErr, setDemoErr] = useState<string | null>(null);

  const handleLoadDemo = async () => {
    setDemoErr(null);
    setDemoLoading(true);
    try {
      await loadDemo();
      window.location.reload();
    } catch (e) {
      setDemoErr(e instanceof Error ? e.message : String(e));
      setDemoLoading(false);
    }
  };

  const totalItems = data ? data.reduce((s, d) => s + d.n, 0) : 0;

  return (
    <AppShell contextChip={project ?? undefined}>
      <div style={{ padding: '32px 36px' }}>
        <div style={{ display: 'flex', alignItems: 'flex-end', gap: 16 }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <Eyebrow>Evaluation inputs</Eyebrow>
              <UpdatingChip visible={reloading && !isInitialLoad} />
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
              Datasets
            </h1>
            <p style={{ fontSize: 13, color: E.text2, marginTop: 4 }}>
              The questions you grade your agent against
              {data ? ` - ${data.length} active sets - ${totalItems} items total` : ''}
            </p>
          </div>
          <span style={{ flex: 1 }} />
          <Btn kind="secondary" size="md" disabled title={IMPORT_CSV_HINT}>
            Import CSV
          </Btn>
          <Btn kind="primary" size="md" disabled title={NEW_DATASET_HINT}>
            + New dataset
          </Btn>
        </div>

        {err && (
          <Card style={{ padding: 16, marginTop: 22, borderColor: E.fail }}>
            <Eyebrow style={{ color: E.fail }}>Error loading datasets</Eyebrow>
            <div style={{ fontFamily: E.fMono, fontSize: 12, color: E.text2, marginTop: 6 }}>
              {err}
            </div>
          </Card>
        )}

        {!data && !err && (
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(2, 1fr)',
              gap: 14,
              marginTop: 22,
            }}
          >
            {[0, 1, 2, 3].map((i) => (
              <Card key={i} style={{ padding: 18 }}>
                <div style={{ display: 'flex', alignItems: 'flex-start', gap: 10 }}>
                  <Skeleton w={32} h={32} style={{ borderRadius: 7 }} />
                  <div style={{ flex: 1 }}>
                    <Skeleton w="60%" h={13} />
                    <div style={{ marginTop: 6 }}>
                      <Skeleton w="40%" h={10} />
                    </div>
                  </div>
                  <Skeleton w={36} h={22} />
                </div>
                <div style={{ marginTop: 14 }}>
                  <Skeleton w="100%" h={7} />
                </div>
              </Card>
            ))}
          </div>
        )}

        {data && data.length === 0 && (
          <Card style={{ padding: 32, marginTop: 22, textAlign: 'center' }}>
            <div style={{ fontSize: 14, color: E.text1, marginBottom: 14 }}>
              No datasets yet. Load the demo to explore, or build one from the CLI.
            </div>
            <div style={{ display: 'flex', gap: 8, justifyContent: 'center' }}>
              <Btn kind="primary" size="md" onClick={handleLoadDemo} disabled={demoLoading}>
                {demoLoading ? 'Loading...' : 'Load demo'}
              </Btn>
              <Btn kind="secondary" size="md" disabled title={IMPORT_CSV_HINT}>
                Import CSV
              </Btn>
            </div>
            {demoErr && (
              <div
                style={{
                  marginTop: 14,
                  padding: '8px 12px',
                  background: E.failDim,
                  border: `1px solid ${E.fail}33`,
                  borderRadius: 6,
                  color: E.fail,
                  fontSize: 12,
                  fontFamily: E.fMono,
                  textAlign: 'left',
                  wordBreak: 'break-word',
                }}
              >
                {demoErr}
              </div>
            )}
          </Card>
        )}

        {data && data.length > 0 && (
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(2, 1fr)',
              gap: 14,
              marginTop: 22,
            }}
          >
            {data.map((s) => {
              const segments = s.coverage.map((c, i) => ({
                value: c.value,
                color: COVERAGE_COLORS[i % COVERAGE_COLORS.length],
                label: `${c.label}: ${c.value}`,
              }));
              return (
                <Card key={s.name} hover style={{ padding: 18 }}>
                  <div style={{ display: 'flex', alignItems: 'flex-start', gap: 10 }}>
                    <div
                      style={{
                        width: 32,
                        height: 32,
                        borderRadius: 7,
                        background: E.panel2,
                        border: `1px solid ${E.hair2}`,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        color: E.text2,
                        fontSize: 14,
                      }}
                    >
                      ◫
                    </div>
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div
                        style={{
                          fontFamily: E.fMono,
                          fontSize: 13,
                          color: E.text0,
                          fontWeight: 500,
                        }}
                      >
                        {s.name}
                      </div>
                      <div style={{ fontSize: 11, color: E.text3, marginTop: 2 }}>
                        {s.source}
                        {s.last_used_iso ? ` - last used ${s.last_used_iso}` : ''}
                      </div>
                    </div>
                    <div style={{ fontFamily: E.fSerif, fontSize: 22, color: E.text0 }}>{s.n}</div>
                  </div>
                  <div style={{ display: 'flex', gap: 5, marginTop: 10 }}>
                    {s.tags.map((t) => (
                      <Pill
                        key={t}
                        mono
                        style={{ fontSize: 10, background: E.panel2, color: E.text2 }}
                      >
                        {t}
                      </Pill>
                    ))}
                  </div>
                  {s.coverage.length > 0 && (
                    <>
                      <Eyebrow style={{ marginTop: 14, marginBottom: 8 }}>Coverage</Eyebrow>
                      <StackBar segments={segments} w={'100%'} h={7} />
                      <div
                        style={{
                          display: 'flex',
                          flexWrap: 'wrap',
                          gap: 10,
                          marginTop: 8,
                          fontSize: 10.5,
                          color: E.text2,
                          fontFamily: E.fMono,
                        }}
                      >
                        {s.coverage.map((c, i) => (
                          <span
                            key={c.label}
                            style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}
                          >
                            <span
                              style={{
                                width: 6,
                                height: 6,
                                borderRadius: 1.5,
                                background: COVERAGE_COLORS[i % COVERAGE_COLORS.length],
                              }}
                            />
                            {c.label} {c.value}
                          </span>
                        ))}
                      </div>
                    </>
                  )}
                  <div style={{ display: 'flex', gap: 6, marginTop: 14 }}>
                    <Btn kind="secondary" size="sm" disabled title={COMING_SOON}>
                      Open
                    </Btn>
                    <Btn kind="ghost" size="sm" disabled title={COMING_SOON}>
                      Use in eval -&gt;
                    </Btn>
                  </div>
                </Card>
              );
            })}
          </div>
        )}

        <div style={{ height: 30 }} />
      </div>
    </AppShell>
  );
}
