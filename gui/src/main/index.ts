/**
 * Electron main process entry point.
 *
 * Creates the BrowserWindow, spawns the Python DPF backend,
 * and manages the application lifecycle.
 */

import path from "path";

import { BrowserWindow, app, ipcMain } from "electron";

import { DEFAULT_SERVER_PORT } from "../shared/constants";

import { ServerStatus, startServer, stopServer } from "./server";

// Guard against EPIPE when parent shell disconnects (stdout pipe closed)
process.on("uncaughtException", (err: NodeJS.ErrnoException) => {
  if (err.code === "EPIPE" || err.message?.includes("EPIPE")) {
    // Silently ignore — stdout pipe is gone, app continues running
    return;
  }
  // Re-throw non-EPIPE errors
  throw err;
});

/** Safe log that swallows EPIPE errors from broken stdout pipes. */
function safeLog(...args: unknown[]): void {
  try {
    console.log(...args);
  } catch {
    // stdout pipe broken — ignore
  }
}

let mainWindow: BrowserWindow | null = null;
let serverStatus: ServerStatus = {
  ready: false,
  port: DEFAULT_SERVER_PORT,
  backends: { python: false, athena: false, athenak: false, metal: false },
};

function createWindow(): void {
  mainWindow = new BrowserWindow({
    width: 1600,
    height: 1000,
    minWidth: 1200,
    minHeight: 700,
    backgroundColor: "#121212",
    title: "DPF Simulator",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
    },
  });

  // In development, load from Vite dev server; in production, load built files
  const isDev = process.env.NODE_ENV === "development" || !app.isPackaged;

  if (isDev) {
    mainWindow.loadURL("http://localhost:5173");
    // Open DevTools in dev mode
    mainWindow.webContents.openDevTools({ mode: "detach" });
  } else {
    mainWindow.loadFile(path.join(__dirname, "../renderer/index.html"));
  }

  mainWindow.on("closed", () => {
    mainWindow = null;
  });
}

/**
 * Start the Python backend and notify the renderer.
 */
async function bootBackend(): Promise<void> {
  const port = DEFAULT_SERVER_PORT;

  try {
    serverStatus = await startServer(port, (line: string) => {
      safeLog(line);
      // Forward logs to renderer
      mainWindow?.webContents.send("server:log", line);
    });

    // Notify renderer that server is ready
    mainWindow?.webContents.send("server:status", serverStatus);
  } catch (err) {
    const errorMsg = err instanceof Error ? err.message : String(err);
    safeLog(`[main] Server boot failed: ${errorMsg}`);
    serverStatus = {
      ready: false,
      port,
      backends: { python: false, athena: false, athenak: false, metal: false },
      error: errorMsg,
    };
    mainWindow?.webContents.send("server:status", serverStatus);
  }
}

// ── IPC Handlers ───────────────────────────────────────────────

ipcMain.handle("server:port", () => serverStatus.port);

// ── App Lifecycle ──────────────────────────────────────────────

app.whenReady().then(async () => {
  createWindow();

  // Boot the Python backend
  await bootBackend();

  app.on("activate", () => {
    // macOS: re-create window when dock icon is clicked
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on("window-all-closed", () => {
  stopServer();
  if (process.platform !== "darwin") {
    app.quit();
  }
});

app.on("before-quit", () => {
  stopServer();
});
