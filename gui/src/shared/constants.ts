/**
 * Shared constants between Electron main process and renderer.
 */

/** Default port for the DPF Python backend server. */
export const DEFAULT_SERVER_PORT = 8765;

/** Default port for the AI/ML inference server. */
export const DEFAULT_AI_PORT = 8766;

/** Base URL for REST API calls. */
export const API_BASE_URL = `http://localhost:${DEFAULT_SERVER_PORT}`;

/** WebSocket base URL. */
export const WS_BASE_URL = `ws://localhost:${DEFAULT_SERVER_PORT}`;

/** Health check polling interval (ms). */
export const HEALTH_POLL_INTERVAL_MS = 500;

/** Maximum health check attempts before giving up. */
export const MAX_HEALTH_RETRIES = 60; // 30 seconds at 500ms

/** WebSocket reconnect delay (ms). */
export const WS_RECONNECT_DELAY_MS = 1000;

/** Maximum scalar history ring buffer size. */
export const MAX_SCALAR_HISTORY = 10_000;

/** Chart update throttle — max frame rate for ECharts updates. */
export const CHART_UPDATE_FPS = 30;

/** IPC channel names for Electron main ↔ renderer communication. */
export const IPC = {
  SERVER_STATUS: "server:status",
  SERVER_PORT: "server:port",
  SERVER_READY: "server:ready",
  SERVER_ERROR: "server:error",
  SERVER_LOG: "server:log",
} as const;
