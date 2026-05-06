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
  // where navigator.clipboard is undefined.
  const ta = document.createElement('textarea');
  ta.value = text;
  ta.style.position = 'fixed';
  ta.style.left = '-9999px';
  document.body.appendChild(ta);
  ta.select();
  try {
    document.execCommand('copy');
  } finally {
    document.body.removeChild(ta);
  }
}
