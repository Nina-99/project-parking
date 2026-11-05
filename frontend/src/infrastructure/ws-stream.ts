// SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
// SPDX-License-Identifier: LicenseRef-Proprietary

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
    const socket = new WebSocket(url);
    this.socket = socket;

    socket.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data) as WsMessage;
        this.handlers.forEach((handler) => handler(message));
      } catch {
        console.warn("Ignoring a non-JSON websocket message");
      }
    };

    socket.onclose = (event) => {
      // A socket closed by StrictMode's first effect must not null the current
      // one or schedule a reconnect that races the second effect.
      if (this.socket !== socket) return;
      this.socket = null;
      if (this.stopped || event.code === 1008) return;
      this.reconnectTimer = window.setTimeout(() => this.connect(), RECONNECT_MS);
    };

    socket.onerror = () => socket.close();
  }

  disconnect(): void {
    this.stopped = true;
    if (this.reconnectTimer !== null) {
      window.clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    const socket = this.socket;
    this.socket = null;
    socket?.close();
  }

  onMessage(handler: MessageHandler): () => void {
    this.handlers.add(handler);
    return () => {
      this.handlers.delete(handler);
    };
  }
}
