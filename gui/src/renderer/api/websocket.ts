/**
 * WebSocket manager for real-time simulation streaming.
 *
 * Connects to ws://localhost:<port>/ws/<sim_id>, receives ScalarUpdate
 * messages, and manages reconnection.
 */

import { WS_BASE_URL, WS_RECONNECT_DELAY_MS } from "@shared/constants";

import type { ScalarUpdate } from "./types";

export type ScalarUpdateHandler = (update: ScalarUpdate) => void;
export type ConnectionHandler = (connected: boolean) => void;

export class SimulationWebSocket {
  private ws: WebSocket | null = null;
  private simId: string;
  private onScalarUpdate: ScalarUpdateHandler;
  private onConnectionChange: ConnectionHandler;
  private shouldReconnect = true;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(
    simId: string,
    onScalarUpdate: ScalarUpdateHandler,
    onConnectionChange: ConnectionHandler
  ) {
    this.simId = simId;
    this.onScalarUpdate = onScalarUpdate;
    this.onConnectionChange = onConnectionChange;
  }

  /**
   * Open WebSocket connection to the simulation.
   */
  connect(): void {
    this.shouldReconnect = true;
    this.doConnect();
  }

  /**
   * Close connection and stop reconnecting.
   */
  disconnect(): void {
    this.shouldReconnect = false;
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.onConnectionChange(false);
  }

  /**
   * Request binary field data from the server.
   */
  requestFields(fields: string[] = ["rho", "Te"], downsample = 1): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(
        JSON.stringify({
          type: "request_fields",
          fields,
          downsample,
        })
      );
    }
  }

  /**
   * Check if currently connected.
   */
  get connected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  // ── Internal ────────────────────────────────────────────────

  private doConnect(): void {
    const url = `${WS_BASE_URL}/ws/${this.simId}`;
    console.log(`[ws] Connecting to ${url}...`);

    try {
      this.ws = new WebSocket(url);
    } catch (err) {
      console.error("[ws] Failed to create WebSocket:", err);
      this.scheduleReconnect();
      return;
    }

    this.ws.onopen = () => {
      console.log(`[ws] Connected to sim ${this.simId}`);
      this.onConnectionChange(true);
    };

    this.ws.onmessage = (event: MessageEvent) => {
      this.handleMessage(event);
    };

    this.ws.onclose = (event: CloseEvent) => {
      console.log(
        `[ws] Disconnected (code=${event.code}, reason=${event.reason})`
      );
      this.onConnectionChange(false);
      this.scheduleReconnect();
    };

    this.ws.onerror = (event: Event) => {
      console.error("[ws] Error:", event);
    };
  }

  private handleMessage(event: MessageEvent): void {
    // Binary frame = field data (handle later)
    if (event.data instanceof Blob || event.data instanceof ArrayBuffer) {
      return;
    }

    // Text frame = JSON message
    try {
      const msg = JSON.parse(event.data as string);

      if (msg.type === "scalar") {
        this.onScalarUpdate(msg as ScalarUpdate);

        // If simulation finished, stop reconnecting
        if ((msg as ScalarUpdate).finished) {
          this.shouldReconnect = false;
        }
      }
      // field_header messages handled by a future field data subscriber
    } catch (err) {
      console.warn("[ws] Failed to parse message:", err);
    }
  }

  private scheduleReconnect(): void {
    if (!this.shouldReconnect) return;

    this.reconnectTimer = setTimeout(() => {
      console.log("[ws] Attempting reconnect...");
      this.doConnect();
    }, WS_RECONNECT_DELAY_MS);
  }
}
