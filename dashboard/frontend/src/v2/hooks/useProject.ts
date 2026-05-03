/**
 * useProject - shared project meta (name + version) fetched once and cached
 * across every route that renders an AppShell context chip.
 *
 * Backed by /api/v2/home (the snapshot already exposes the resolved project
 * identity from evalyn.yaml or .evalyn/ inference). The hook keeps a module-
 * level cache so navigating between routes never re-fires the request.
 */

import { useEffect, useState } from 'react';
import { v2 } from '../api/client';

type Project = { name: string; version: string | null };

let _cache: Project | null = null;
const _subs: ((p: Project) => void)[] = [];

export function useProject(): Project | null {
  const [proj, setProj] = useState<Project | null>(_cache);
  useEffect(() => {
    if (_cache) {
      setProj(_cache);
      return;
    }
    v2
      .home()
      .then((s) => {
        _cache = s.project;
        setProj(s.project);
        _subs.forEach((f) => f(s.project));
      })
      .catch(() => {
        /* swallow - chip just stays unset */
      });
    _subs.push(setProj);
    return () => {
      const i = _subs.indexOf(setProj);
      if (i >= 0) _subs.splice(i, 1);
    };
  }, []);
  return proj;
}
