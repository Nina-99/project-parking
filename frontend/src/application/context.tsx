/** Composition root for the ports. Presentation never imports an adapter. */
import { createContext, useContext, type ReactNode } from "react";
import type { ApiPort, EventStream } from "./ports";

const ApiContext = createContext<ApiPort | null>(null);
const EventStreamContext = createContext<EventStream | null>(null);

export function ApiProvider({ api, children }: { api: ApiPort; children: ReactNode }) {
  return <ApiContext.Provider value={api}>{children}</ApiContext.Provider>;
}

export function EventStreamProvider({
  stream,
  children,
}: {
  stream: EventStream;
  children: ReactNode;
}) {
  return (
    <EventStreamContext.Provider value={stream}>{children}</EventStreamContext.Provider>
  );
}

function required<T>(value: T | null, name: string): T {
  if (value === null) {
    throw new Error(`${name} was used outside its provider`);
  }
  return value;
}

export function useApi(): ApiPort {
  return required(useContext(ApiContext), "ApiPort");
}

export function useEventStream(): EventStream {
  return required(useContext(EventStreamContext), "EventStream");
}
