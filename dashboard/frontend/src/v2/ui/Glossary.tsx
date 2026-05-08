/**
 * Glossary - inline ``?`` superscript with a native tooltip.
 *
 * Wraps a short label so a hover-help icon appears immediately after it.
 * The tooltip uses the browser's native ``title`` attribute, which keeps
 * the primitive zero-dependency and accessible without trapping focus.
 *
 * Keep ``term`` short (under 120 chars) - longer text gets truncated by
 * the OS chrome on most platforms.
 */

import type { ReactNode } from 'react';
import { E } from '../tokens';

interface GlossaryProps {
  /** The definition that surfaces on hover. */
  term: string;
  /** The visible label being annotated. */
  children: ReactNode;
}

export function Glossary({ term, children }: GlossaryProps) {
  return (
    <span style={{ display: 'inline-flex', alignItems: 'baseline', gap: 3 }}>
      {children}
      <span
        title={term}
        // Sighted users see "?" and hover for the definition; SR
        // users never see the title attribute reliably across
        // engines, so they previously got "Definition, image" with
        // no payload. Putting the actual term in the accessible
        // name means an SR pass over a sentence reads naturally:
        // "Pass rate. Definition: The percent of items that..."
        aria-label={`Definition: ${term}`}
        role="img"
        style={{
          fontSize: '0.7em',
          color: E.text3,
          cursor: 'help',
          fontFamily: E.fMono,
          userSelect: 'none',
          lineHeight: 1,
          marginLeft: 1,
        }}
      >
        ?
      </span>
    </span>
  );
}
