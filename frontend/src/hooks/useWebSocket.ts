import { useEffect } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { OccupancyWebSocket } from "../services/websocket";

/**
 * Keep the dashboard live.
 *
 * A gate event invalidates the queries that depend on it instead of trying to
 * patch them by hand: an unmatched exit, a duplicate or a reconnection all end
 * in the same correct state, which is whatever the database says.
 */
export function useWebSocket() {
  const queryClient = useQueryClient();

  useEffect(() => {
    const ws = new OccupancyWebSocket();
    ws.connect();

    const unsubscribe = ws.onMessage((msg) => {
      if (msg.type === "ping") return;
      void queryClient.invalidateQueries({ queryKey: ["occupancy"] });
      void queryClient.invalidateQueries({ queryKey: ["active-sessions"] });
      void queryClient.invalidateQueries({ queryKey: ["daily-stats"] });
    });

    return () => {
      unsubscribe();
      ws.disconnect();
    };
  }, [queryClient]);
}
