import { useCallback, useEffect, useRef, useState } from "react";

/** Poll an async fetcher on an interval. Failures keep the last good value. */
export function usePolling<T>(fetcher: () => Promise<T>, intervalMs: number, deps: any[] = []) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  const refresh = useCallback(async () => {
    try {
      const value = await fetcherRef.current();
      setData(value);
      setError(null);
    } catch (e: any) {
      setError(e?.message || "unreachable");
    }
  }, []);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, intervalMs);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [refresh, intervalMs, ...deps]);

  return { data, error, refresh };
}
