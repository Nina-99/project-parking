/** Wire the adapters once. Everything below depends on the ports. */
import type { ReactNode } from "react";
import { ApiProvider, EventStreamProvider } from "./application/context";
import { HttpApi } from "./infrastructure/http-api";
import { WebSocketEventStream } from "./infrastructure/ws-stream";

const api = new HttpApi();
const events = new WebSocketEventStream();

export function AppProviders({ children }: { children: ReactNode }) {
  return (
    <ApiProvider api={api}>
      <EventStreamProvider stream={events}>{children}</EventStreamProvider>
    </ApiProvider>
  );
}
