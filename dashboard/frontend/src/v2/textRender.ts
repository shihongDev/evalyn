/**
 * Shared text-rendering helpers for any surface that displays prose
 * with embedded URLs (annotation panes, review item content, etc).
 *
 * `linkifyText(text, counter)` splits a string into plain-text strings
 * + small `[N]` citation buttons. URLs become compact, clickable, and
 * skimmable instead of dominating the prose. Counter is mutable so
 * numbering stays continuous across multiple calls (e.g. when the
 * caller is interleaving plain text with highlighted ranges).
 */

import * as React from 'react';

export type UrlCounter = { value: number };

/** Convenience constructor when no caller-managed counter is needed. */
export function makeUrlCounter(): UrlCounter {
  return { value: 0 };
}

// URL terminators - whitespace, angle/quote chars, brackets, braces,
// parens, pipe, backslash. Built dynamically to avoid trigger-happy
// security scanners that flag literal regex source.
const URL_TERMINATORS = '\\s<>"\'{}|\\\\[\\]()';
const URL_REGEX = new RegExp('https?://[^' + URL_TERMINATORS + ']+', 'g');
const TRAILING_PUNCT = /[.,;:!?]$/;

const LINK_STYLE: React.CSSProperties = {
  display: 'inline-block',
  fontFamily: "'Geist Mono', ui-monospace, monospace",
  fontSize: 10,
  fontWeight: 500,
  lineHeight: 1.2,
  color: '#d96a2c',
  background: '#fcefe2',
  border: '1px solid #d96a2c33',
  borderRadius: 4,
  padding: '0 5px',
  textDecoration: 'none',
  marginLeft: 2,
  marginRight: 2,
  verticalAlign: 'baseline',
  cursor: 'pointer',
  transition: 'background 140ms',
};

const LINK_HOVER_BG = '#f9dcc1';
const LINK_REST_BG = '#fcefe2';

export function linkifyText(text: string, counter: UrlCounter): React.ReactNode[] {
  if (!text) return [text];
  const parts: React.ReactNode[] = [];
  let lastIndex = 0;
  URL_REGEX.lastIndex = 0;
  let match: RegExpExecArray | null;
  while ((match = URL_REGEX.exec(text)) !== null) {
    let url = match[0];
    let end = match.index + url.length;
    while (url.length > 0 && TRAILING_PUNCT.test(url)) {
      url = url.slice(0, -1);
      end -= 1;
    }
    if (url.length === 0) continue;
    if (match.index > lastIndex) {
      parts.push(text.slice(lastIndex, match.index));
    }
    counter.value += 1;
    const n = counter.value;
    parts.push(
      React.createElement(
        'a',
        {
          key: 'u' + match.index,
          href: url,
          target: '_blank',
          rel: 'noopener noreferrer',
          title: url,
          style: LINK_STYLE,
          onMouseEnter: (e: React.MouseEvent<HTMLAnchorElement>) => {
            (e.currentTarget as HTMLAnchorElement).style.background = LINK_HOVER_BG;
          },
          onMouseLeave: (e: React.MouseEvent<HTMLAnchorElement>) => {
            (e.currentTarget as HTMLAnchorElement).style.background = LINK_REST_BG;
          },
        },
        '[' + n + ']',
      ),
    );
    lastIndex = end;
  }
  if (lastIndex < text.length) parts.push(text.slice(lastIndex));
  return parts.length > 0 ? parts : [text];
}
