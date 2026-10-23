/** Wire the adapters once. Everything below depends on the ports. */
import type { ReactNode } from "react";
import { ApiProvider, EventStreamProvider } from "./application/context";
import { SessionProvider } from "./application/session";
import { HttpApi } from "./infrastructure/http-api";
import { HttpAuth } from "./infrastructure/http-auth";
import { WebSocketEventStream } from "./infrastructure/ws-stream";

const api = new HttpApi();
const auth = new HttpAuth();
const events = new WebSocketEventStream();

export function AppProviders({ children }: { children: ReactNode }) {
  return (
    <SessionProvider auth={auth}>
      <ApiProvider api={api}>
        <EventStreamProvider stream={events}>{children}</EventStreamProvider>
      </ApiProvider>
    </SessionProvider>
  );
}
