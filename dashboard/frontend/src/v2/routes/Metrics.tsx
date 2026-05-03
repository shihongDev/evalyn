/**
 * Metrics & rubrics - left list of rubrics + right detail panel.
 * Wires v2.rubrics() and v2.rubric(id) into the design from screens-3.jsx.
 */

import { useEffect, useState } from 'react';
import { AppShell } from '../AppShell';
import { Btn, Card, Eyebrow, Pill } from '../ui';
import { v2 } from '../api/client';
import type { RubricDetail, RubricRow } from '../api/types';
import { E } from '../tokens';

const KIND_COLOR: Record<RubricRow['kind'], string> = {
  'LLM judge': E.ember,
  Programmatic: E.steel,
  Hybrid: E.warn,
};

function pillBg(kind: 'pass' | 'warn' | 'fail' | 'info'): string {
  if (kind === 'pass') return E.passDim;
  if (kind === 'warn') return E.warnDim;
  if (kind === 'fail') return E.failDim;
  return E.infoDim;
}

function pillColor(kind: 'pass' | 'warn' | 'fail' | 'info'): string {
  if (kind === 'pass') return E.pass;
  if (kind === 'warn') return E.warn;
  if (kind === 'fail') return E.fail;
  return E.info;
}

export default function Metrics() {
  const [list, setList] = useState<RubricRow[] | null>(null);
  const [listErr, setListErr] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<RubricDetail | null>(null);
  const [detailErr, setDetailErr] = useState<string | null>(null);

  useEffect(() => {
    v2.rubrics()
      .then((rows) => {
        setList(rows);
        if (rows.length > 0) setSelectedId(rows[0].id);
      })
      .catch((e: unknown) => setListErr(String(e)));
  }, []);

  useEffect(() => {
    if (!selectedId) return;
    setDetail(null);
    setDetailErr(null);
    v2.rubric(selectedId)
      .then(setDetail)
      .catch((e: unknown) => setDetailErr(String(e)));
  }, [selectedId]);

  return (
    <AppShell contextChip={{ name: 'Customer Support Agent', version: 'v0.3' }}>
      <div style={{ padding: '32px 36px' }}>
        <div style={{ display: 'flex', alignItems: 'flex-end' }}>
          <div>
            <Eyebrow>How quality is graded</Eyebrow>
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
              Metrics &amp; rubrics
            </h1>
          </div>
          <span style={{ flex: 1 }} />
          <Btn kind="primary" size="md">+ New rubric</Btn>
        </div>

        {listErr && (
          <Card style={{ padding: 16, marginTop: 22, borderColor: E.fail }}>
            <Eyebrow style={{ color: E.fail }}>Error loading rubrics</Eyebrow>
            <div style={{ fontFamily: E.fMono, fontSize: 12, color: E.text2, marginTop: 6 }}>
              {listErr}
            </div>
          </Card>
        )}

        <div
          style={{
            display: 'grid',
            gridTemplateColumns: '1fr 1.4fr',
            gap: 14,
            marginTop: 22,
          }}
        >
          {/* List */}
          <Card style={{ padding: 0, overflow: 'hidden' }}>
            <div
              style={{
                padding: '12px 16px',
                borderBottom: `1px solid ${E.hair}`,
                fontSize: 12,
                color: E.text3,
                fontFamily: E.fMono,
                letterSpacing: '0.06em',
              }}
            >
              YOUR RUBRICS
            </div>
            {!list && !listErr && (
              <div style={{ padding: 16, color: E.text3, fontSize: 13 }}>Loading...</div>
            )}
            {list && list.length === 0 && (
              <div style={{ padding: 24, color: E.text3, fontSize: 13, textAlign: 'center' }}>
                No rubrics defined yet.
              </div>
            )}
            {list &&
              list.map((r, i) => {
                const isActive = r.id === selectedId;
                return (
                  <button
                    key={r.id}
                    type="button"
                    onClick={() => setSelectedId(r.id)}
                    style={{
                      width: '100%',
                      display: 'flex',
                      alignItems: 'center',
                      gap: 12,
                      padding: '14px 16px',
                      borderTop: i ? `1px solid ${E.hair}` : 'none',
                      cursor: 'pointer',
                      background: isActive ? E.emberDim : 'transparent',
                      border: 'none',
                      textAlign: 'left',
                    }}
                  >
                    <span
                      style={{
                        width: 4,
                        height: 32,
                        borderRadius: 2,
                        background: KIND_COLOR[r.kind],
                      }}
                    />
                    <div style={{ flex: 1 }}>
                      <div style={{ fontFamily: E.fMono, fontSize: 13, color: E.text0 }}>
                        {r.name}
                      </div>
                      <div style={{ fontSize: 11, color: E.text3, marginTop: 2 }}>
                        {r.kind} - {r.dimensions} dimensions - {r.calibration_label}
                      </div>
                    </div>
                    <div style={{ fontFamily: E.fMono, fontSize: 11, color: E.text2 }}>
                      {r.uses}x
                    </div>
                  </button>
                );
              })}
          </Card>

          {/* Detail */}
          <Card style={{ padding: 0, overflow: 'hidden' }}>
            {!selectedId && (
              <div style={{ padding: 24, color: E.text3, fontSize: 13, textAlign: 'center' }}>
                No rubrics defined yet.
              </div>
            )}
            {selectedId && detailErr && (
              <div style={{ padding: 16 }}>
                <Eyebrow style={{ color: E.fail }}>Error loading rubric</Eyebrow>
                <div
                  style={{ fontFamily: E.fMono, fontSize: 12, color: E.text2, marginTop: 6 }}
                >
                  {detailErr}
                </div>
              </div>
            )}
            {selectedId && !detail && !detailErr && (
              <div style={{ padding: 16, color: E.text3, fontSize: 13 }}>Loading...</div>
            )}
            {detail && (
              <>
                <div
                  style={{
                    padding: '14px 18px',
                    borderBottom: `1px solid ${E.hair}`,
                    display: 'flex',
                    alignItems: 'center',
                  }}
                >
                  <span style={{ fontFamily: E.fMono, fontSize: 13, color: E.text0 }}>
                    {detail.name}
                  </span>
                  <Pill
                    mono
                    color={pillColor(detail.calibration.kind)}
                    bg={pillBg(detail.calibration.kind)}
                    style={{ fontSize: 10, marginLeft: 8 }}
                  >
                    {detail.calibration.label}
                  </Pill>
                  <span style={{ flex: 1 }} />
                  <Btn kind="ghost" size="sm">Duplicate</Btn>
                  <Btn kind="ghost" size="sm">Re-calibrate</Btn>
                </div>
                <div style={{ padding: 18 }}>
                  <Eyebrow>Dimensions - weighted</Eyebrow>
                  <div
                    style={{
                      marginTop: 12,
                      display: 'flex',
                      flexDirection: 'column',
                      gap: 12,
                    }}
                  >
                    {detail.dimensions.map((d) => (
                      <div
                        key={d.label}
                        style={{
                          padding: 12,
                          background: E.panel2,
                          borderRadius: 8,
                          border: `1px solid ${E.hair}`,
                        }}
                      >
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                          <span
                            style={{
                              fontSize: 13,
                              color: E.text0,
                              fontWeight: 500,
                              flex: 1,
                            }}
                          >
                            {d.label}
                          </span>
                          <Pill
                            mono
                            style={{
                              fontSize: 9.5,
                              background: d.kind === 'judge' ? E.emberDim : E.panel3,
                              color: d.kind === 'judge' ? E.ember : E.text2,
                            }}
                          >
                            {d.kind === 'judge' ? 'LLM judge' : 'programmatic'}
                          </Pill>
                          <span
                            style={{
                              fontFamily: E.fMono,
                              fontSize: 12,
                              color: E.text0,
                              width: 36,
                              textAlign: 'right',
                            }}
                          >
                            {d.weight_pct}%
                          </span>
                        </div>
                        <div
                          style={{
                            fontSize: 11.5,
                            color: E.text3,
                            marginTop: 4,
                            fontStyle: 'italic',
                          }}
                        >
                          "{d.example}"
                        </div>
                      </div>
                    ))}
                  </div>

                  <Eyebrow style={{ marginTop: 18, marginBottom: 8 }}>
                    Calibration - last {detail.calibration.sample_size} human-reviewed items
                  </Eyebrow>
                  <div
                    style={{
                      display: 'grid',
                      gridTemplateColumns: 'repeat(3, 1fr)',
                      gap: 8,
                    }}
                  >
                    <CalibCell
                      label="COHEN'S KAPPA"
                      value={
                        detail.calibration.kappa != null
                          ? detail.calibration.kappa.toFixed(2)
                          : '-'
                      }
                      sub={detail.calibration.label}
                      color={pillColor(detail.calibration.kind)}
                    />
                    <CalibCell
                      label="FALSE POSITIVES"
                      value={
                        detail.calibration.false_positives_pct != null
                          ? `${detail.calibration.false_positives_pct.toFixed(1)}%`
                          : '-'
                      }
                      sub="judge says pass, human fails"
                      color={E.warn}
                    />
                    <CalibCell
                      label="FALSE NEGATIVES"
                      value={
                        detail.calibration.false_negatives_pct != null
                          ? `${detail.calibration.false_negatives_pct.toFixed(1)}%`
                          : '-'
                      }
                      sub="judge says fail, human passes"
                      color={E.pass}
                    />
                  </div>
                </div>
              </>
            )}
          </Card>
        </div>

        <div style={{ height: 30 }} />
      </div>
    </AppShell>
  );
}

function CalibCell({
  label,
  value,
  sub,
  color,
}: {
  label: string;
  value: string;
  sub: string;
  color: string;
}) {
  return (
    <div
      style={{
        padding: 10,
        background: E.panel2,
        borderRadius: 6,
        border: `1px solid ${E.hair}`,
      }}
    >
      <div
        style={{
          fontSize: 10,
          color: E.text3,
          fontFamily: E.fMono,
          letterSpacing: '0.06em',
        }}
      >
        {label}
      </div>
      <div style={{ fontFamily: E.fSerif, fontSize: 20, color, marginTop: 2 }}>{value}</div>
      <div style={{ fontSize: 10.5, color: E.text3, marginTop: 2 }}>{sub}</div>
    </div>
  );
}
