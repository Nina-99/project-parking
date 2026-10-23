/** WebSocket adapter. Reconnection lives here, not in a component. */
import type { EventStream, MessageHandler } from "../application/ports";
import type { WsMessage } from "../domain/types";

const RECONNECT_MS = 3000;

export class WebSocketEventStream implements EventStream {
  private socket: WebSocket | null = null;
  private handlers = new Set<MessageHandler>();
  private reconnectTimer: number | null = null;
  private stopped = true;

  connect(): void {
    if (
      this.socket?.readyState === WebSocket.OPEN ||
      this.socket?.readyState === WebSocket.CONNECTING
    ) {
      return;
    }
    this.stopped = false;
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const url = `${protocol}//${window.location.host}/api/v1/ws/occupancy`;
    this.socket = new WebSocket(url);

    this.socket.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data) as WsMessage;
        this.handlers.forEach((handler) => handler(message));
      } catch {
        console.warn("Ignoring a non-JSON websocket message");
      }
    };

    this.socket.onclose = () => {
      this.socket = null;
      if (this.stopped) return;
      this.reconnectTimer = window.setTimeout(() => this.connect(), RECONNECT_MS);
    };

    this.socket.onerror = () => this.socket?.close();
  }

  disconnect(): void {
    this.stopped = true;
    if (this.reconnectTimer !== null) {
      window.clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this.socket?.close();
    this.socket = null;
  }

  onMessage(handler: MessageHandler): () => void {
    this.handlers.add(handler);
    return () => {
      this.handlers.delete(handler);
    };
  }
}
