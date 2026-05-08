import { describe, expect, it } from 'vitest';
import { formatCapacityRetryHint, errorMessage, CapacityError } from './errors';

describe('formatCapacityRetryHint', () => {
  it('says "now" when the server hint is 0 or 1 second', () => {
    expect(formatCapacityRetryHint(0)).toBe('Try again now');
    expect(formatCapacityRetryHint(1)).toBe('Try again now');
  });

  it('renders precise seconds for values in [2, 59]', () => {
    expect(formatCapacityRetryHint(2)).toBe('Try again in 2s');
    expect(formatCapacityRetryHint(5)).toBe('Try again in 5s');
    expect(formatCapacityRetryHint(30)).toBe('Try again in 30s');
    expect(formatCapacityRetryHint(59)).toBe('Try again in 59s');
  });

  it('rounds down to whole minutes for >= 60 seconds', () => {
    expect(formatCapacityRetryHint(60)).toBe('Try again in 1m');
    expect(formatCapacityRetryHint(89)).toBe('Try again in 1m');
    expect(formatCapacityRetryHint(120)).toBe('Try again in 2m');
    expect(formatCapacityRetryHint(305)).toBe('Try again in 5m');
  });
});

describe('errorMessage', () => {
  it('extracts message from Error instances', () => {
    expect(errorMessage(new Error('boom'))).toBe('boom');
  });

  it('handles CapacityError (subclass of Error)', () => {
    const e = new CapacityError({
      message: 'queue full',
      running: 8,
      maxConcurrent: 8,
      retryAfterSeconds: 10,
    });
    expect(errorMessage(e)).toBe('queue full');
  });

  it('falls back to String() for non-Error values', () => {
    expect(errorMessage('plain string')).toBe('plain string');
    expect(errorMessage(42)).toBe('42');
  });
});
