/**
 * Clipboard helper - one writeText call with a textarea fallback for
 * non-secure contexts (older browsers, file:// pages, http: dev servers).
 *
 * Routes use this for "Copy report" / "Copy as markdown" / "Share URL"
 * actions. Throws on any failure so callers can flip a UI state to
 * "error" and tell the user that the browser blocked clipboard access.
 */

export async function copyToClipboard(text: string): Promise<void> {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }
  // Fallback: textarea + execCommand. Works in non-secure contexts
  // where navigator.clipboard is undefined. execCommand returns
  // boolean false on failure (e.g. browser blocked, deprecated path
  // disabled, no document.body); the previous version of this code
  // ignored the return value and reported success regardless,
  // leaving callers in a "Copied" state when nothing was actually
  // on the clipboard. Throw so the caller's catch path flips to
  // "Copy failed" feedback for the user.
  const ta = document.createElement('textarea');
  ta.value = text;
  ta.style.position = 'fixed';
  ta.style.left = '-9999px';
  document.body.appendChild(ta);
  ta.select();
  let ok = false;
  try {
    ok = document.execCommand('copy');
  } finally {
    document.body.removeChild(ta);
  }
  if (!ok) {
    throw new Error('clipboard copy failed (execCommand returned false)');
  }
}
