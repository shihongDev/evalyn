/**
 * Platform-aware UI helpers.
 *
 * Detected once at module load. Shared across surfaces so cheat sheets,
 * footers, tooltips, and any other "modifier shortcut" display read
 * consistently — `⌘` on Mac, `Ctrl` elsewhere.
 *
 * Behavior is platform-agnostic in code (handlers always check
 * `e.metaKey || e.ctrlKey`); only the displayed glyph changes.
 *
 * navigator.platform is technically deprecated in favor of
 * navigator.userAgentData.platform, but the latter isn't supported
 * in Safari or Firefox as of late 2025, so the older API is the
 * cross-browser-correct choice for now.
 */

export const IS_MAC: boolean =
  typeof navigator !== 'undefined' &&
  /Mac|iPod|iPhone|iPad/.test(navigator.platform);

/** Glyph for the primary modifier shortcut. ⌘ on Mac, Ctrl elsewhere. */
export const MOD_KEY: string = IS_MAC ? '⌘' : 'Ctrl';
