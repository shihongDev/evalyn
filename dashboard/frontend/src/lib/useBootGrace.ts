/**
 * useBootGrace - returns true while the component has been mounted for less
 * than `windowMs`.
 *
 * Used by list views (CliCatalog, JobsList, RunsList) to render a skeleton
 * during the boot-fetch grace window so the empty state never flashes for
 * users on a fast connection.
 */

import { useEffect, useState } from 'react';

export const useBootGrace = (windowMs = 1500): boolean => {
  const [grace, setGrace] = useState(true);
  useEffect(() => {
    const t = setTimeout(() => setGrace(false), windowMs);
    return () => clearTimeout(t);
  }, [windowMs]);
  return grace;
};
