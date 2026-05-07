/**
 * v2 UI primitives - Card, Eyebrow, Pill, Btn, StatusDot.
 * Ported 1:1 from /tmp/evalyn-v2/design-system.jsx.
 */

import type {
  CSSProperties,
  FocusEventHandler,
  KeyboardEventHandler,
  MouseEvent as ReactMouseEvent,
  MouseEventHandler,
  ReactNode,
} from 'react';
import { E } from '../tokens';

interface CardProps {
  children?: ReactNode;
  style?: CSSProperties;
  pad?: number;
  hover?: boolean;
  accent?: boolean;
  onClick?: MouseEventHandler<HTMLDivElement>;
  /** Accessible name when the card itself is the action (i.e. has
   * `onClick`). Required for SR users to know what activating the
   * card does - especially when the visible card content is a chart
   * or icon rather than a sentence. Ignored when `onClick` is unset
   * (a non-interactive card has no need for an aria-label). */
  'aria-label'?: string;
  /** Optional coachmark id for the co-pilot UI guidance tour to anchor on. */
  'data-coachmark'?: string;
}

export function Card({
  children,
  style,
  pad = 0,
  hover,
  accent,
  onClick,
  'aria-label': ariaLabel,
  'data-coachmark': dataCoachmark,
}: CardProps) {
  // When the card itself is the click target, expose it as a button
  // to AT and add Enter/Space activation so keyboard users can reach
  // and trigger it. Without this the card was effectively
  // mouse-only - tab skipped right past it.
  const interactive = Boolean(onClick);
  const onKeyDown: KeyboardEventHandler<HTMLDivElement> | undefined = interactive
    ? (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          // Stop space from scrolling the page; mirror the click path.
          e.preventDefault();
          onClick?.(e as unknown as ReactMouseEvent<HTMLDivElement>);
        }
      }
    : undefined;
  return (
    <div
      onClick={onClick}
      onKeyDown={onKeyDown}
      role={interactive ? 'button' : undefined}
      tabIndex={interactive ? 0 : undefined}
      aria-label={interactive ? ariaLabel : undefined}
      data-coachmark={dataCoachmark}
      // .eCardHover lifts the card on hover via box-shadow and a 1 px
      // upward translate. Box-shadow is intentionally chosen over a
      // border-color change so the caller's inline `style.borderColor`
      // (e.g. Datasets's selected-state ember rim) keeps winning -
      // shadow adds depth without touching the border layer.
      className={hover ? 'eCardHover' : undefined}
      style={{
        background: E.panel,
        border: `1px solid ${accent ? E.emberRim : E.hair}`,
        borderRadius: 12,
        padding: pad,
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
  /** Hover hook - useful for prefetching cached resources before a click. */
  onMouseEnter?: MouseEventHandler<HTMLButtonElement>;
  /** Focus hook - mirrors `onMouseEnter` for keyboard users. */
  onFocus?: FocusEventHandler<HTMLButtonElement>;
  disabled?: boolean;
  type?: 'button' | 'submit' | 'reset';
  title?: string;
  /** Optional coachmark id for the co-pilot UI guidance tour to anchor on. */
  'data-coachmark'?: string;
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
  onMouseEnter,
  onFocus,
  disabled,
  type = 'button',
  title,
  'data-coachmark': dataCoachmark,
}: BtnProps) {
  return (
    <button
      type={type}
      onClick={onClick}
      onMouseEnter={onMouseEnter}
      onFocus={onFocus}
      disabled={disabled}
      title={title}
      data-coachmark={dataCoachmark}
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
  label,
}: {
  status: DotStatus | string;
  size?: number;
  animated?: boolean;
  /**
   * Opt-in accessible label. When set, the dot is exposed to screen
   * readers as an image with this name AND gets a sighted hover
   * tooltip with the same text. When omitted, the dot stays
   * decorative - which is correct for the majority of usages where
   * adjacent text already conveys the status (e.g. "<dot> Quality").
   * Use a label only when the dot's color is the SOLE indicator of
   * status and there's no nearby text the user can read.
   */
  label?: string;
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
      role={label ? 'img' : undefined}
      aria-label={label}
      title={label}
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
