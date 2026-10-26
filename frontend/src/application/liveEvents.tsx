// SPDX-FileCopyrightText: 2026 Nina
// SPDX-License-Identifier: LicenseRef-Proprietary

/** One socket for the whole app: query invalidation plus a short event feed. */
import {
  createContext,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from "react";
import { useQueryClient } from "@tanstack/react-query";
import type { WsMessage } from "../domain/types";
import { useEventStream } from "./context";

const MAX_EVENTS = 30;
const EventsContext = createContext<WsMessage[] | null>(null);

export function LiveEventsProvider({ children }: { children: ReactNode }) {
  const stream = useEventStream();
  const queryClient = useQueryClient();
  const [events, setEvents] = useState<WsMessage[]>([]);

  useEffect(() => {
    stream.connect();

    const unsubscribe = stream.onMessage((message) => {
      if (message.type === "ping") return;

      setEvents((previous) => [message, ...previous].slice(0, MAX_EVENTS));

      void queryClient.invalidateQueries({ queryKey: ["occupancy"] });
      void queryClient.invalidateQueries({ queryKey: ["active-sessions"] });
      void queryClient.invalidateQueries({ queryKey: ["daily-stats"] });
      void queryClient.invalidateQueries({ queryKey: ["history"] });
      void queryClient.invalidateQueries({ queryKey: ["vehicle-search"] });
      void queryClient.invalidateQueries({ queryKey: ["vehicle"] });
      void queryClient.invalidateQueries({ queryKey: ["model-stats"] });
      void queryClient.invalidateQueries({ queryKey: ["cameras"] });
    });

    return () => {
      unsubscribe();
      stream.disconnect();
    };
  }, [stream, queryClient]);

  return (
    <EventsContext.Provider value={events}>{children}</EventsContext.Provider>
  );
}

export function useLiveEvents(): WsMessage[] {
  const events = useContext(EventsContext);
  if (events === null) {
    throw new Error("useLiveEvents was used outside LiveEventsProvider");
  }
  return events;
}
