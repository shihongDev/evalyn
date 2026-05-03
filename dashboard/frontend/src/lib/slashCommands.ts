/**
 * Slash-command parser for the chat composer.
 *
 * Client-side only: matches `/run`, `/compare`, `/explain`, `/clear`,
 * `/settings`, `/help` at the start of the input and either expands to a
 * structured prompt (kind 'send') or signals a side-effect action (kind
 * 'action'). Returns null when the leading token doesn't match any known
 * slash command - caller should fall through to "send as normal text".
 */

export type SlashAction = 'clear' | 'settings' | 'help';

export type SlashParseResult =
  | { kind: 'send'; text: string }
  | { kind: 'action'; action: SlashAction }
  | null;

/**
 * Inline help text rendered when the user types `/help`. The composer (or
 * caller) appends this as a synthetic assistant turn - never sent over the
 * wire to the agent.
 */
export const SLASH_HELP_TEXT =
  'Slash commands:\n' +
  '  /run <cli> - ask the agent to run a CLI\n' +
  '  /compare <a> <b> - compare two runs\n' +
  '  /explain <id> - walk through a run\n' +
  '  /clear - start a new thread\n' +
  '  /settings - open settings\n' +
  '  /help - show this list';

/**
 * Parse a composer input string. Trims whitespace, splits on the first
 * whitespace run to isolate the slash token, and dispatches by token name.
 *
 * Behavior notes:
 * - `/run` with no args returns `null` so the user sees a no-op rather than
 *   a malformed prompt; the caller falls through to plain-text send.
 * - `/compare` requires two args; with fewer, returns `null`.
 * - `/explain` requires one arg; with none, returns `null`.
 * - `/clear`, `/settings`, `/help` ignore any extra arguments - typing
 *   `/help help` still yields the help action.
 * - Anything that doesn't start with `/` returns `null`.
 */
export const parseSlashCommand = (input: string): SlashParseResult => {
  if (typeof input !== 'string') return null;
  const trimmed = input.trim();
  if (!trimmed.startsWith('/')) return null;

  const match = /^\/(\S+)\s*([\s\S]*)$/.exec(trimmed);
  if (!match) return null;
  const cmd = match[1].toLowerCase();
  const tail = match[2].trim();
  const args = tail.length > 0 ? tail.split(/\s+/) : [];

  switch (cmd) {
    case 'run':
      if (args.length < 1) return null;
      return { kind: 'send', text: `please run \`${args[0]}\` and show me the output.` };
    case 'compare':
      if (args.length < 2) return null;
      return {
        kind: 'send',
        text: `compare runs ${args[0]} and ${args[1]} and tell me what changed.`,
      };
    case 'explain':
      if (args.length < 1) return null;
      return { kind: 'send', text: `walk me through what happened in run ${args[0]}.` };
    case 'clear':
      return { kind: 'action', action: 'clear' };
    case 'settings':
      return { kind: 'action', action: 'settings' };
    case 'help':
      return { kind: 'action', action: 'help' };
    default:
      return null;
  }
};
