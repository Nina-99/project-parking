/** Keep the cache honest when a gate event arrives. */
import { useEffect } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useEventStream } from "./context";

export function useLiveEvents(): void {
  const stream = useEventStream();
  const queryClient = useQueryClient();

  useEffect(() => {
    stream.connect();

    const unsubscribe = stream.onMessage((message) => {
      if (message.type === "ping") return;
      void queryClient.invalidateQueries({ queryKey: ["occupancy"] });
      void queryClient.invalidateQueries({ queryKey: ["active-sessions"] });
      void queryClient.invalidateQueries({ queryKey: ["daily-stats"] });
      void queryClient.invalidateQueries({ queryKey: ["history"] });
      void queryClient.invalidateQueries({ queryKey: ["vehicle-search"] });
      void queryClient.invalidateQueries({ queryKey: ["vehicle"] });
      void queryClient.invalidateQueries({ queryKey: ["model-stats"] });
    });

    return () => {
      unsubscribe();
      stream.disconnect();
    };
  }, [stream, queryClient]);
}
