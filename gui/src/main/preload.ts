/**
 * Electron preload script — exposes a secure bridge between
 * the main process and the renderer (React app).
 *
 * Uses contextBridge to selectively expose IPC channels.
 */

import { contextBridge, ipcRenderer } from "electron";

/**
 * API exposed to the renderer process as `window.dpf`.
 */
contextBridge.exposeInMainWorld("dpf", {
  /**
   * Get the server port from the main process.
   */
  getServerPort: (): Promise<number> =>
    ipcRenderer.invoke("server:port"),

  /**
   * Listen for server status updates.
   */
  onServerStatus: (
    callback: (status: {
      ready: boolean;
      port: number;
      backends: { python: boolean; athena: boolean; athenak: boolean };
      error?: string;
    }) => void
  ) => {
    const handler = (
      _event: Electron.IpcRendererEvent,
      status: {
        ready: boolean;
        port: number;
        backends: { python: boolean; athena: boolean; athenak: boolean };
        error?: string;
      }
    ) => callback(status);
    ipcRenderer.on("server:status", handler);
    return () => {
      ipcRenderer.removeListener("server:status", handler);
    };
  },

  /**
   * Listen for server log messages.
   */
  onServerLog: (callback: (line: string) => void) => {
    const handler = (_event: Electron.IpcRendererEvent, line: string) =>
      callback(line);
    ipcRenderer.on("server:log", handler);
    return () => {
      ipcRenderer.removeListener("server:log", handler);
    };
  },
});

/**
 * TypeScript declaration for the exposed API.
 * Import this type in renderer code for type safety.
 */
export interface DPFBridge {
  getServerPort: () => Promise<number>;
  onServerStatus: (
    callback: (status: {
      ready: boolean;
      port: number;
      backends: { python: boolean; athena: boolean; athenak: boolean };
      error?: string;
    }) => void
  ) => () => void;
  onServerLog: (callback: (line: string) => void) => () => void;
}
