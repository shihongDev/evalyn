/**
 * v2 chart primitives - SVG, no chart library.
 * Ported 1:1 from /tmp/evalyn-v2/design-system.jsx.
 */

import type { CSSProperties, ReactNode } from 'react';
import { E } from '../tokens';

export function Spark({
  data,
  w = 80,
  h = 24,
  color = E.pass,
  dot,
}: {
  data: number[];
  w?: number;
  h?: number;
  color?: string;
  dot?: boolean;
}) {
  if (!data || data.length === 0) {
    return <svg width={w} height={h} style={{ display: 'block' }} />;
  }
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const pts = data
    .map((v, i) => {
      const x = (i / Math.max(1, data.length - 1)) * (w - 4) + 2;
      const y = h - ((v - min) / range) * (h - 6) - 3;
      return `${x},${y}`;
    })
    .join(' ');
  const last = data[data.length - 1];
  const lx = w - 2;
  const ly = h - ((last - min) / range) * (h - 6) - 3;
  return (
    <svg width={w} height={h} style={{ display: 'block' }}>
      <polyline
        points={pts}
        fill="none"
        stroke={color}
        strokeWidth={1.5}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      {dot && <circle cx={lx} cy={ly} r={2} fill={color} />}
    </svg>
  );
}

export function Bars({
  data,
  w = 80,
  h = 24,
  color = E.pass,
  track = E.panel3,
}: {
  data: number[];
  w?: number;
  h?: number;
  color?: string;
  track?: string;
}) {
  if (!data || data.length === 0) {
    return <svg width={w} height={h} style={{ display: 'block' }} />;
  }
  const max = Math.max(...data) || 1;
  const bw = (w - (data.length - 1) * 2) / data.length;
  return (
    <svg width={w} height={h} style={{ display: 'block' }}>
      {data.map((v, i) => {
        const bh = Math.max(1, (v / max) * h);
        const x = i * (bw + 2);
        return (
          <rect
            key={i}
            x={x}
            y={h - bh}
            width={bw}
            height={bh}
            fill={i === data.length - 1 ? color : track}
            rx={1}
          />
        );
      })}
    </svg>
  );
}

export function Bar({
  value,
  max = 100,
  w = 100,
  h = 4,
  color = E.pass,
  track = E.panel3,
  style,
}: {
  value: number;
  max?: number;
  w?: number | string;
  h?: number;
  color?: string;
  track?: string;
  style?: CSSProperties;
}) {
  const pct = Math.max(2, (value / Math.max(1, max)) * 100);
  return (
    <div
      style={{
        width: w,
        height: h,
        background: track,
        borderRadius: 999,
        overflow: 'hidden',
        ...style,
      }}
    >
      <div style={{ width: `${pct}%`, height: '100%', background: color }} />
    </div>
  );
}

export function StackBar({
  segments,
  w = 200,
  h = 6,
  gap = 1,
}: {
  segments: { value: number; color: string; label?: string }[];
  w?: number | string;
  h?: number;
  gap?: number;
}) {
  const total = segments.reduce((s, x) => s + x.value, 0) || 1;
  return (
    <div
      style={{
        width: w,
        height: h,
        display: 'flex',
        gap,
        borderRadius: 999,
        overflow: 'hidden',
        background: E.panel3,
      }}
    >
      {segments.map((s, i) => (
        <div
          key={i}
          title={s.label}
          style={{ width: `${(s.value / total) * 100}%`, height: '100%', background: s.color }}
        />
      ))}
    </div>
  );
}

interface DonutSegment {
  value: number;
  color: string;
  label?: string;
}

interface DonutProps {
  value?: number;
  max?: number;
  size?: number;
  color?: string;
  thickness?: number;
  thick?: number;
  segments?: DonutSegment[];
  center?: ReactNode;
}

export function Donut({
  value,
  max = 100,
  size = 56,
  color = E.pass,
  thickness = 4,
  thick,
  segments,
  center,
}: DonutProps) {
  const t = thick != null ? thick : thickness;
  const r = size / 2 - t;
  const c = 2 * Math.PI * r;

  if (Array.isArray(segments) && segments.length) {
    const total = segments.reduce((a, s) => a + (s.value || 0), 0) || 1;
    let acc = 0;
    return (
      <div style={{ position: 'relative', width: size, height: size, display: 'inline-block' }}>
        <svg width={size} height={size} style={{ display: 'block' }}>
          <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke={E.panel3} strokeWidth={t} />
          {segments.map((s, i) => {
            const len = (s.value / total) * c;
            const rotation = (acc / total) * 360 - 90;
            acc += s.value;
            return (
              <circle
                key={i}
                cx={size / 2}
                cy={size / 2}
                r={r}
                fill="none"
                stroke={s.color}
                strokeWidth={t}
                strokeDasharray={`${len} ${c}`}
                strokeDashoffset={0}
                transform={`rotate(${rotation} ${size / 2} ${size / 2})`}
              />
            );
          })}
        </svg>
        {center && (
          <div
            style={{
              position: 'absolute',
              inset: 0,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              pointerEvents: 'none',
            }}
          >
            {center}
          </div>
        )}
      </div>
    );
  }

  const off = c * (1 - (value || 0) / max);
  return (
    <svg width={size} height={size} style={{ display: 'block' }}>
      <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke={E.panel3} strokeWidth={t} />
      <circle
        cx={size / 2}
        cy={size / 2}
        r={r}
        fill="none"
        stroke={color}
        strokeWidth={t}
        strokeDasharray={c}
        strokeDashoffset={off}
        strokeLinecap="round"
        transform={`rotate(-90 ${size / 2} ${size / 2})`}
      />
    </svg>
  );
}

export interface LineSeries {
  color: string;
  data: number[];
  width?: number;
  fill?: boolean;
  dashed?: boolean;
}

interface LineChartProps {
  series: LineSeries[];
  w?: number | string;
  h?: number;
  yMin?: number;
  yMax?: number;
  baseline?: number;
  padX?: number;
  padY?: number;
  xLabels?: string[];
}

export function LineChart({
  series,
  w = 540,
  h = 180,
  yMin = 0,
  yMax = 100,
  baseline,
  padX = 36,
  padY = 14,
  xLabels,
}: LineChartProps) {
  const numW = typeof w === 'number' ? w : 540;
  const innerW = numW - padX - 12;
  const innerH = h - padY * 2;
  const xAt = (i: number, len: number) =>
    padX + (i / Math.max(1, len - 1)) * innerW;
  const yAt = (v: number) =>
    padY + innerH - ((v - yMin) / Math.max(1e-9, yMax - yMin)) * innerH;
  const ticks = [yMin, yMin + (yMax - yMin) / 2, yMax];
  return (
    <svg width={w} height={h} style={{ display: 'block', overflow: 'visible' }}>
      {ticks.map((t, i) => (
        <g key={i}>
          <line
            x1={padX}
            x2={numW - 12}
            y1={yAt(t)}
            y2={yAt(t)}
            stroke={E.hair}
            strokeWidth={1}
            strokeDasharray={i === ticks.length - 1 ? '0' : '2 4'}
          />
          <text
            x={padX - 8}
            y={yAt(t) + 3}
            textAnchor="end"
            fontSize={9}
            fill={E.text3}
            fontFamily={E.fMono}
          >
            {Math.round(t)}
          </text>
        </g>
      ))}
      {baseline != null && (
        <g>
          <line
            x1={padX}
            x2={numW - 12}
            y1={yAt(baseline)}
            y2={yAt(baseline)}
            stroke={E.steel}
            strokeWidth={1}
            strokeDasharray="3 3"
            opacity={0.6}
          />
          <text
            x={numW - 14}
            y={yAt(baseline) - 4}
            textAnchor="end"
            fontSize={9}
            fill={E.steel}
            fontFamily={E.fMono}
          >
            baseline {baseline}
          </text>
        </g>
      )}
      {series.map((s, si) => {
        const len = s.data.length;
        if (len === 0) return null;
        const pts = s.data.map((v, i) => `${xAt(i, len)},${yAt(v)}`).join(' ');
        const area = `M ${xAt(0, len)},${yAt(yMin)} L ${s.data
          .map((v, i) => `${xAt(i, len)},${yAt(v)}`)
          .join(' L ')} L ${xAt(len - 1, len)},${yAt(yMin)} Z`;
        return (
          <g key={si}>
            {s.fill && <path d={area} fill={s.color} opacity={0.08} />}
            <polyline
              points={pts}
              fill="none"
              stroke={s.color}
              strokeWidth={s.width ?? 2}
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeDasharray={s.dashed ? '4 3' : '0'}
            />
            <circle
              cx={xAt(len - 1, len)}
              cy={yAt(s.data[len - 1])}
              r={3.5}
              fill={E.ink}
              stroke={s.color}
              strokeWidth={2}
            />
          </g>
        );
      })}
      {xLabels &&
        xLabels.map((l, i) => (
          <text
            key={i}
            x={xAt(i, xLabels.length)}
            y={h - 2}
            textAnchor="middle"
            fontSize={9}
            fill={E.text3}
            fontFamily={E.fMono}
          >
            {l}
          </text>
        ))}
    </svg>
  );
}

export function Heatmap({
  data,
  rowLabels,
  colLabels,
  cellSize = 28,
  color = E.ember,
}: {
  data: number[][];
  rowLabels: string[];
  colLabels: string[];
  cellSize?: number;
  color?: string;
}) {
  const max = Math.max(...data.flat()) || 1;
  return (
    <svg
      width={cellSize * colLabels.length + 90}
      height={cellSize * rowLabels.length + 24}
      style={{ display: 'block' }}
    >
      {colLabels.map((l, i) => (
        <text
          key={l}
          x={88 + i * cellSize + cellSize / 2}
          y={14}
          textAnchor="middle"
          fontSize={9}
          fill={E.text3}
          fontFamily={E.fMono}
        >
          {l}
        </text>
      ))}
      {rowLabels.map((rl, ri) => (
        <text
          key={rl}
          x={84}
          y={24 + ri * cellSize + cellSize / 2 + 3}
          textAnchor="end"
          fontSize={10}
          fill={E.text2}
        >
          {rl}
        </text>
      ))}
      {data.map((row, ri) =>
        row.map((v, ci) => {
          const op = 0.1 + (v / max) * 0.85;
          return (
            <g key={`${ri}-${ci}`}>
              <rect
                x={88 + ci * cellSize}
                y={20 + ri * cellSize}
                width={cellSize - 2}
                height={cellSize - 2}
                fill={ri === ci ? color : E.steel}
                opacity={op}
                rx={3}
              />
              <text
                x={88 + ci * cellSize + cellSize / 2}
                y={20 + ri * cellSize + cellSize / 2 + 3}
                textAnchor="middle"
                fontSize={10}
                fontFamily={E.fMono}
                fill={op > 0.4 ? E.text0 : E.text3}
              >
                {v}
              </text>
            </g>
          );
        }),
      )}
    </svg>
  );
}
