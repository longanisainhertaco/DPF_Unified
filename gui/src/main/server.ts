/**
 * Python backend subprocess manager.
 *
 * Spawns `dpf serve --port <port>` as a child process, polls /api/health
 * until the server is ready, and kills it cleanly on app quit.
 */

import { ChildProcess, spawn } from "child_process";
import http from "http";

import {
  DEFAULT_SERVER_PORT,
  HEALTH_POLL_INTERVAL_MS,
  MAX_HEALTH_RETRIES,
} from "../shared/constants";

let serverProcess: ChildProcess | null = null;
let serverPort: number = DEFAULT_SERVER_PORT;

export interface ServerStatus {
  ready: boolean;
  port: number;
  backends: {
    python: boolean;
    athena: boolean;
    athenak: boolean;
  };
  error?: string;
}

/**
 * Start the DPF Python backend server as a child process.
 *
 * @param port - Port number (default 8765)
 * @param onLog - Callback for server stdout/stderr lines
 * @returns Promise that resolves when the server health check passes
 */
export function startServer(
  port: number = DEFAULT_SERVER_PORT,
  onLog?: (line: string) => void
): Promise<ServerStatus> {
  serverPort = port;

  return new Promise((resolve, reject) => {
    // Spawn `dpf serve --port <port>`
    const proc = spawn("dpf", ["serve", "--port", String(port)], {
      stdio: ["ignore", "pipe", "pipe"],
      env: { ...process.env },
    });

    serverProcess = proc;

    // Forward stdout/stderr
    proc.stdout?.on("data", (data: Buffer) => {
      const lines = data.toString().trim().split("\n");
      lines.forEach((line) => onLog?.(`[server] ${line}`));
    });

    proc.stderr?.on("data", (data: Buffer) => {
      const lines = data.toString().trim().split("\n");
      lines.forEach((line) => onLog?.(`[server:err] ${line}`));
    });

    proc.on("error", (err) => {
      onLog?.(`[server] Failed to start: ${err.message}`);
      reject(
        new Error(
          `Failed to start DPF server: ${err.message}. ` +
            "Ensure 'dpf' CLI is installed (pip install dpf-unified)."
        )
      );
    });

    proc.on("exit", (code, signal) => {
      onLog?.(
        `[server] Process exited (code=${code}, signal=${signal})`
      );
      serverProcess = null;
    });

    // Poll health endpoint until ready
    pollHealth(port, 0, onLog)
      .then((status) => resolve(status))
      .catch((err) => reject(err));
  });
}

/**
 * Poll the /api/health endpoint until the server responds.
 */
function pollHealth(
  port: number,
  attempt: number,
  onLog?: (line: string) => void
): Promise<ServerStatus> {
  return new Promise((resolve, reject) => {
    if (attempt >= MAX_HEALTH_RETRIES) {
      reject(
        new Error(
          `Server did not become ready after ${MAX_HEALTH_RETRIES} attempts`
        )
      );
      return;
    }

    setTimeout(() => {
      const url = `http://localhost:${port}/api/health`;

      http
        .get(url, (res) => {
          let body = "";
          res.on("data", (chunk: Buffer) => {
            body += chunk.toString();
          });
          res.on("end", () => {
            try {
              const data = JSON.parse(body);
              onLog?.(`[server] Health check OK: ${JSON.stringify(data)}`);
              resolve({
                ready: true,
                port,
                backends: data.backends || {
                  python: true,
                  athena: false,
                  athenak: false,
                },
              });
            } catch {
              // Retry on parse error
              pollHealth(port, attempt + 1, onLog).then(resolve).catch(reject);
            }
          });
        })
        .on("error", () => {
          // Server not ready yet — retry
          if (attempt % 10 === 0) {
            onLog?.(
              `[server] Waiting for backend (attempt ${attempt + 1}/${MAX_HEALTH_RETRIES})...`
            );
          }
          pollHealth(port, attempt + 1, onLog).then(resolve).catch(reject);
        });
    }, HEALTH_POLL_INTERVAL_MS);
  });
}

/**
 * Stop the Python backend server if running.
 */
export function stopServer(): void {
  if (serverProcess) {
    console.log("[server] Stopping Python backend...");
    serverProcess.kill("SIGTERM");

    // Force kill after 5 seconds if still alive
    const forceKillTimeout = setTimeout(() => {
      if (serverProcess) {
        console.log("[server] Force-killing Python backend...");
        serverProcess.kill("SIGKILL");
      }
    }, 5000);

    serverProcess.on("exit", () => {
      clearTimeout(forceKillTimeout);
      serverProcess = null;
    });
  }
}

/**
 * Check if the server process is currently running.
 */
export function isServerRunning(): boolean {
  return serverProcess !== null && !serverProcess.killed;
}

/**
 * Get the current server port.
 */
export function getServerPort(): number {
  return serverPort;
}
