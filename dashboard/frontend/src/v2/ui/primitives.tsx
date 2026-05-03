/**
 * v2 UI primitives - Card, Eyebrow, Pill, Btn, StatusDot.
 * Ported 1:1 from /tmp/evalyn-v2/design-system.jsx.
 */

import type { CSSProperties, MouseEventHandler, ReactNode } from 'react';
import { E } from '../tokens';

interface CardProps {
  children?: ReactNode;
  style?: CSSProperties;
  pad?: number;
  hover?: boolean;
  accent?: boolean;
  onClick?: MouseEventHandler<HTMLDivElement>;
}

export function Card({ children, style, pad = 0, hover, accent, onClick }: CardProps) {
  return (
    <div
      onClick={onClick}
      style={{
        background: E.panel,
        border: `1px solid ${accent ? E.emberRim : E.hair}`,
        borderRadius: 12,
        padding: pad,
        transition: hover ? 'border-color 120ms' : undefined,
        cursor: onClick ? 'pointer' : undefined,
        ...style,
      }}
    >
      {children}
    </div>
  );
}

export function Eyebrow({ children, style }: { children?: ReactNode; style?: CSSProperties }) {
  return (
    <div
      style={{
        fontSize: 10,
        color: E.text3,
        fontFamily: E.fMono,
        textTransform: 'uppercase',
        letterSpacing: '0.1em',
        ...style,
      }}
    >
      {children}
    </div>
  );
}

interface PillProps {
  children?: ReactNode;
  color?: string;
  bg?: string;
  mono?: boolean;
  style?: CSSProperties;
  title?: string;
}

export function Pill({ children, color = E.text2, bg = E.panel3, mono, style, title }: PillProps) {
  return (
    <span
      title={title}
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 5,
        padding: '2px 8px',
        borderRadius: 999,
        fontSize: 11,
        fontFamily: mono ? E.fMono : E.fSans,
        color,
        background: bg,
        cursor: title ? 'help' : undefined,
        ...style,
      }}
    >
      {children}
    </span>
  );
}

type BtnKind = 'primary' | 'secondary' | 'ghost' | 'bare' | 'danger';
type BtnSize = 'sm' | 'md' | 'lg';

interface BtnProps {
  children?: ReactNode;
  kind?: BtnKind;
  size?: BtnSize;
  style?: CSSProperties;
  onClick?: MouseEventHandler<HTMLButtonElement>;
  disabled?: boolean;
  type?: 'button' | 'submit' | 'reset';
  title?: string;
}

const BTN_SIZES: Record<BtnSize, CSSProperties> = {
  sm: { padding: '4px 10px', fontSize: 11 },
  md: { padding: '6px 12px', fontSize: 12 },
  lg: { padding: '8px 16px', fontSize: 13 },
};

const BTN_KINDS: Record<BtnKind, CSSProperties> = {
  primary: { background: E.ember, color: E.emberInk, border: 'none', fontWeight: 500 },
  secondary: { background: E.panel2, color: E.text1, border: `1px solid ${E.hair2}` },
  ghost: { background: 'transparent', color: E.text2, border: `1px solid ${E.hair2}` },
  bare: { background: 'transparent', color: E.text2, border: 'none' },
  danger: { background: E.failDim, color: E.fail, border: `1px solid ${E.fail}33` },
};

export function Btn({
  children,
  kind = 'ghost',
  size = 'md',
  style,
  onClick,
  disabled,
  type = 'button',
  title,
}: BtnProps) {
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      title={title}
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 6,
        borderRadius: 6,
        cursor: disabled ? 'not-allowed' : 'pointer',
        opacity: disabled ? 0.55 : 1,
        ...BTN_SIZES[size],
        ...BTN_KINDS[kind],
        ...style,
      }}
    >
      {children}
    </button>
  );
}

type DotStatus = 'pass' | 'completed' | 'fail' | 'failed' | 'running' | 'warn' | 'info' | 'idle';

export function StatusDot({
  status,
  size = 6,
  animated,
}: {
  status: DotStatus | string;
  size?: number;
  animated?: boolean;
}) {
  const c =
    status === 'pass' || status === 'completed'
      ? E.pass
      : status === 'fail' || status === 'failed'
        ? E.fail
        : status === 'running'
          ? E.ember
          : status === 'warn'
            ? E.warn
            : status === 'info'
              ? E.steel
              : E.text3;
  return (
    <span
      style={{
        display: 'inline-block',
        width: size,
        height: size,
        borderRadius: '50%',
        background: c,
        flexShrink: 0,
        boxShadow: animated ? `0 0 0 ${size * 0.6}px ${c}22` : 'none',
        animation: animated ? 'eDotPulse 1.6s ease-in-out infinite' : 'none',
      }}
    />
  );
}
