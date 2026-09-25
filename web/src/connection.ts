import type { ClientMessage, ServerMessage } from "./protocol";

// Retry after 1 s, then 2 s, 4 s, ... up to 5 s while the backend is down
// (each failed attempt also logs an error in the Vite terminal).
const RECONNECT_MIN_MS = 1000;
const RECONNECT_MAX_MS = 5000;

export interface ConnectionHandlers {
  onMessage(message: ServerMessage): void;
  onConnectedChange(connected: boolean): void;
}

/** The page's WebSocket URL: same host the page came from (Vite proxies /ws in dev). */
export function defaultSocketUrl(): string {
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  return `${scheme}://${location.host}/ws`;
}

/**
 * WebSocket to the backend. Reconnects automatically, e.g. after you restart
 * the Python server; the server re-sends the scene on every new connection.
 */
export class Connection {
  private readonly url: string;
  private readonly handlers: ConnectionHandlers;
  private socket: WebSocket | null = null;
  private reconnectDelay = RECONNECT_MIN_MS;

  constructor(url: string, handlers: ConnectionHandlers) {
    this.url = url;
    this.handlers = handlers;
    this.open();
  }

  /** Send a command; silently dropped while disconnected. */
  send(message: ClientMessage): void {
    if (this.socket?.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify(message));
    }
  }

  private open(): void {
    const socket = new WebSocket(this.url);
    this.socket = socket;
    socket.onopen = () => {
      this.reconnectDelay = RECONNECT_MIN_MS;
      this.handlers.onConnectedChange(true);
    };
    socket.onmessage = (event: MessageEvent<string>) => {
      // Trusted cast: our own server sends these, and tests/test_protocol.py
      // checks that its Pydantic models match protocol.ts.
      this.handlers.onMessage(JSON.parse(event.data) as ServerMessage);
    };
    socket.onclose = () => {
      this.handlers.onConnectedChange(false);
      setTimeout(() => this.open(), this.reconnectDelay);
      this.reconnectDelay = Math.min(2 * this.reconnectDelay, RECONNECT_MAX_MS);
    };
  }
}
