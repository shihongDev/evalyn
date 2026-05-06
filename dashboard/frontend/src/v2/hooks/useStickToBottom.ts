/**
 * useStickToBottom - chat-style auto-scroll behavior for streaming
 * conversations.
 *
 * Returns a ref + onScroll handler to attach to a scrollable container,
 * plus a `scrolledUp` boolean and `jumpToBottom` callback the surface
 * can use to render a "Jump to latest" pill when the user has scrolled
 * up while messages continue to arrive.
 *
 * Whenever `dep` changes (typically the messages array, or a length /
 * length-of-last-message tuple), the container scrolls to its bottom
 * IF AND ONLY IF the user was already near the bottom before the
 * update. If the user has scrolled up to read older messages, the hook
 * does not yank them back - they stay in place until they scroll
 * within `slack` pixels of the bottom again or click `jumpToBottom`.
 *
 * Standard chat UX pattern. Used by both the in-page Co-pilot thread
 * (/copilot) and the right-side dock variant.
 */

import { useCallback, useEffect, useRef, useState } from 'react';

export function useStickToBottom<T>(
  dep: T,
  options: { slack?: number } = {},
): {
  scrollRef: React.RefObject<HTMLDivElement | null>;
  onScroll: () => void;
  scrolledUp: boolean;
  jumpToBottom: () => void;
} {
  const slack = options.slack ?? 80;
  const scrollRef = useRef<HTMLDivElement | null>(null);
  // Default true so the very first message lands at the bottom rather
  // than at the empty state's scroll position. After the first user
  // scroll the ref becomes ground truth.
  const stickRef = useRef(true);
  // Mirror the inverse of stickRef into render-state so the surface
  // can show a "Jump to latest" affordance only when the user is
  // scrolled up. Two separate sources of truth (ref for the effect's
  // sync read, state for renders) is the cleanest way to avoid stale
  // closures inside the scroll listener.
  const [scrolledUp, setScrolledUp] = useState(false);

  const onScroll = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    const distance = el.scrollHeight - (el.scrollTop + el.clientHeight);
    const atBottom = distance < slack;
    stickRef.current = atBottom;
    setScrolledUp((prev) => (prev === !atBottom ? prev : !atBottom));
  }, [slack]);

  const jumpToBottom = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
    stickRef.current = true;
    setScrolledUp(false);
  }, []);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    if (!stickRef.current) return;
    // Use scrollTop = scrollHeight rather than scrollIntoView - the
    // latter forces the parent's overflow ancestors to scroll too,
    // which can yank the page chrome on tall layouts.
    el.scrollTop = el.scrollHeight;
  }, [dep]);

  return { scrollRef, onScroll, scrolledUp, jumpToBottom };
}
