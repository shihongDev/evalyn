/**
 * useProject - shared project meta (name + version) for the AppShell context chip.
 *
 * Implementation note: this used to maintain its own module-level cache. It now
 * piggybacks on `useV2Resource('home', v2.home)` so the request is shared with
 * the Home route and the nav prefetch path. The chip lights up the moment any
 * caller has hydrated the home cache.
 */

import { v2 } from '../api/client';
import { useV2Resource } from './useV2Resource';

type Project = { name: string; version: string | null };

export function useProject(): Project | null {
  const { data } = useV2Resource('home', v2.home);
  return data ? data.project : null;
}
