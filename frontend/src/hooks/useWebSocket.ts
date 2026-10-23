import { useEffect, useState, useCallback } from "react";
import { OccupancyWebSocket, type WsMessage } from "../services/websocket";

export function useWebSocket() {
  const [lastMessage, setLastMessage] = useState<WsMessage | null>(null);

  useEffect(() => {
    const ws = new OccupancyWebSocket();
    ws.connect();

    const unsubscribe = ws.onMessage((msg) => {
      setLastMessage(msg);
    });

    return () => {
      unsubscribe();
      ws.disconnect();
    };
  }, []);

  return { lastMessage };
}

export function usePolling<T>(
  fetcher: () => Promise<T>,
  intervalMs = 5000
) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<Error | null>(null);

  const refetch = useCallback(async () => {
    try {
      const result = await fetcher();
      setData(result);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)));
    }
  }, [fetcher]);

  useEffect(() => {
    refetch();
    const timer = setInterval(refetch, intervalMs);
    return () => clearInterval(timer);
  }, [refetch, intervalMs]);

  return { data, error, refetch };
}
