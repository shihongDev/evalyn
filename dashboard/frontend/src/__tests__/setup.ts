/**
 * Vitest setup: extend `expect` with jest-dom matchers and ensure cleanup
 * between tests.
 */

import '@testing-library/jest-dom/vitest';
import { afterEach } from 'vitest';
import { cleanup } from '@testing-library/react';

afterEach(() => {
  cleanup();
});
