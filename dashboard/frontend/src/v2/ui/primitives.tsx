/**
 * v2 UI primitives - Card, Eyebrow, Pill, Btn, StatusDot.
 * Ported 1:1 from /tmp/evalyn-v2/design-system.jsx.
 *
 * ## Conventions for callers (enforce in PRs - none of this is
 *    enforced by tooling yet)
 *
 * 1. `isActive`-style highlighting must ship a matching ARIA
 *    attribute. The visual highlight is half the signal; SR users
 *    need the other half. Pick by intent:
 *      - tab strip            -> role=tablist + role=tab + aria-selected
 *      - list of options      -> role=listbox + role=option + aria-selected
 *      - "current page" link  -> aria-current="page"
 *      - toggle (one of N)    -> aria-pressed
 *    Whenever you write `isActive ?` in a `style` prop, add the
 *    ARIA attr in the same JSX block, with the same source of
 *    truth (not a parallel computation).
 *
 * 2. Clickable parent + inner CTA Btn = single tab stop. If a row
 *    is `role="button"` AND visually contains an "Open ->"-style
 *    Btn that fires the same nav, mark the Btn `tabIndex={-1}`
 *    and `aria-hidden`. Otherwise keyboard users tab through two
 *    stops per row going to one place. The Btn primitive accepts
 *    both props for exactly this use.
 *
 * 3. Truncated dynamic text needs a `title=` mirror. Any
 *    `textOverflow: 'ellipsis'` over `{interpolation}` content
 *    must have a `title=` on the same tag (or a parent button)
 *    so sighted users can read the full string on hover. SR users
 *    are usually covered by the parent's aria-label, but title=
 *    serves a different audience.
 *
 * 4. Clipboard writes must surface error states. A bare
 *    `navigator.clipboard.writeText().then(ok)` swallows
 *    permission/insecure-context rejections silently. Every copy
 *    button in v2 uses a `'idle' | 'copied' | 'error'` state and
 *    flashes "Copy failed" in red on rejection (see Pane Copy in
 *    AnnotateSession, Reports markdown copy, etc.).
 *
 * 5. `aria-modal="true"` requires explicit focus management.
 *    aria-modal is a *claim*; the on-open focus move and focus
 *    restoration on close are still the caller's responsibility.
 *    All four current modals (CommandPalette, ShortcutHelpOverlay,
 *    RecentJobsDrawer, CliRunner) follow the same pattern: capture
 *    prevFocus on open, focus a meaningful element inside, restore
 *    prevFocus on unmount.
 *
 * 6. Prefetch hooks must fire for keyboard users too. Any
 *    `onMouseEnter` that warms a route chunk or in-flight fetch
 *    needs a paired `onFocus` on the same element with the same
 *    body. Mouse hover triggers warm; keyboard Tab triggers
 *    focus; both should be sub-perception by click time. The
 *    AppShell nav rail also adds `onTouchStart` for mobile - copy
 *    that triple when the link is on the primary nav surface.
 *    Concrete pattern: hoist the warm into a `warm` helper and
 *    attach it to all three event props rather than duplicating
 *    the body inline (see ExperimentsList lineage row).
 *
 * 7. Form inputs need a programmatic label, not just adjacent
 *    text. A styled `<div>` next to an `<input>` looks labeled
 *    but isn't - SR users tabbing in hear the type ("spin
 *    button", "edit, blank") with no field name. Pick one:
 *      - wrap the field in a real `<label htmlFor={id}>` and
 *        give the input a matching `id` (use useId() so two
 *        instances on the same page can't collide). This is
 *        what the CliRunner ParamField does.
 *      - if the input is decorative-row-style with no separate
 *        visible label (e.g. CommandPalette search, CoPilotDock
 *        chat box, Datasets / RecentJobsDrawer search), put an
 *        explicit `aria-label` on the input itself so the
 *        placeholder isn't the only signal of purpose.
 *    The `placeholder` attribute is not a substitute for either -
 *    it disappears on focus.
 *
 * 8. Async handlers need an internal busy guard, not just
 *    `disabled` on the trigger. React's batched state updates
 *    let a same-frame second activation slip through with the
 *    OLD closure, so two POSTs go out for one click pair. The
 *    pattern is belts-and-suspenders:
 *      external: `disabled={busy || ...}` on every trigger
 *      internal: `if (busy) return;` as the first line of the
 *               handler body
 *    The internal guard is mandatory for any handler that:
 *      - is bound to a keyboard auto-repeat key (Enter, Space,
 *        N/P/F type single-letter keys) - holding the key
 *        bypasses the visual disabled cue entirely;
 *      - is bound to two or more triggers (e.g. a header button +
 *        an empty-state button + a hotkey) where the disabled
 *        prop has to be passed to all of them and one might
 *        drift over time.
 *    Sites following the pattern: RunDetail handleRerun, both
 *    copilot submit() functions, Settings handleSave, Datasets
 *    handleLoadDemo, AnnotateSession submitVerdict +
 *    finalizeSession + saveAllUnsaved.
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
  /** Override the button's tab order. Pass `-1` to remove the button
   * from the keyboard tab sequence - useful when a parent element has
   * already taken on the interactive role (e.g. a clickable card whose
   * inner CTA button is decorative for keyboard users). The mouse
   * click path stays intact. */
  tabIndex?: number;
  /** Hide the button from accessibility tree. Use together with
   * tabIndex={-1} when an enclosing interactive ancestor already
   * exposes the same action - prevents AT from announcing the
   * button as a duplicate. */
  'aria-hidden'?: boolean;
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
  tabIndex,
  'aria-hidden': ariaHidden,
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
      tabIndex={tabIndex}
      aria-hidden={ariaHidden}
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
