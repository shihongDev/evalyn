/**
 * Unit tests for the chat composer slash-command parser.
 *
 * Covers each known command, missing-arg behaviour, extra-arg tolerance,
 * case-insensitivity, leading whitespace, and non-slash inputs.
 */

import { describe, expect, test } from 'vitest';
import { parseSlashCommand, SLASH_HELP_TEXT } from '../../lib/slashCommands';

describe('parseSlashCommand', () => {
  test('returns null for plain text', () => {
    expect(parseSlashCommand('hello there')).toBeNull();
    expect(parseSlashCommand('')).toBeNull();
    expect(parseSlashCommand('   ')).toBeNull();
  });

  test('non-string input returns null', () => {
    // @ts-expect-error - guarding against runtime misuse
    expect(parseSlashCommand(undefined)).toBeNull();
    // @ts-expect-error - guarding against runtime misuse
    expect(parseSlashCommand(null)).toBeNull();
  });

  test('/run expands to send prompt', () => {
    const r = parseSlashCommand('/run analyze');
    expect(r).toEqual({
      kind: 'send',
      text: 'please run `analyze` and show me the output.',
    });
  });

  test('/run without args returns null (caller falls through)', () => {
    expect(parseSlashCommand('/run')).toBeNull();
    expect(parseSlashCommand('/run   ')).toBeNull();
  });

  test('/run with extra args picks first arg only', () => {
    const r = parseSlashCommand('/run analyze whatever extra');
    expect(r).toEqual({
      kind: 'send',
      text: 'please run `analyze` and show me the output.',
    });
  });

  test('/compare requires two args', () => {
    expect(parseSlashCommand('/compare')).toBeNull();
    expect(parseSlashCommand('/compare a')).toBeNull();
    const r = parseSlashCommand('/compare runA runB');
    expect(r).toEqual({
      kind: 'send',
      text: 'compare runs runA and runB and tell me what changed.',
    });
  });

  test('/explain requires one arg', () => {
    expect(parseSlashCommand('/explain')).toBeNull();
    const r = parseSlashCommand('/explain run-123');
    expect(r).toEqual({
      kind: 'send',
      text: 'walk me through what happened in run run-123.',
    });
  });

  test('/clear, /settings, /help return action results', () => {
    expect(parseSlashCommand('/clear')).toEqual({ kind: 'action', action: 'clear' });
    expect(parseSlashCommand('/settings')).toEqual({
      kind: 'action',
      action: 'settings',
    });
    expect(parseSlashCommand('/help')).toEqual({ kind: 'action', action: 'help' });
  });

  test('action commands tolerate extra args', () => {
    expect(parseSlashCommand('/help help')).toEqual({
      kind: 'action',
      action: 'help',
    });
    expect(parseSlashCommand('/clear  whatever')).toEqual({
      kind: 'action',
      action: 'clear',
    });
  });

  test('case-insensitive command tokens', () => {
    expect(parseSlashCommand('/RUN x')).toEqual({
      kind: 'send',
      text: 'please run `x` and show me the output.',
    });
    expect(parseSlashCommand('/HELP')).toEqual({ kind: 'action', action: 'help' });
  });

  test('leading whitespace is tolerated', () => {
    expect(parseSlashCommand('   /clear')).toEqual({
      kind: 'action',
      action: 'clear',
    });
  });

  test('unknown slash command returns null (caller falls through)', () => {
    expect(parseSlashCommand('/unknown thing')).toBeNull();
    expect(parseSlashCommand('/why does x happen?')).toBeNull();
  });

  test('SLASH_HELP_TEXT lists all six commands', () => {
    expect(SLASH_HELP_TEXT).toContain('/run');
    expect(SLASH_HELP_TEXT).toContain('/compare');
    expect(SLASH_HELP_TEXT).toContain('/explain');
    expect(SLASH_HELP_TEXT).toContain('/clear');
    expect(SLASH_HELP_TEXT).toContain('/settings');
    expect(SLASH_HELP_TEXT).toContain('/help');
  });
});
