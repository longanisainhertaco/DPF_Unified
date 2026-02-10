/**
 * Root application component.
 *
 * Phase K.1: Server connection management
 * Phase K.2: DashboardShell with parameter forms, scope, and co-pilot
 */

import { useEffect, useState } from "react";

import { fetchHealth, fetchPresets } from "./api/client";
import { DashboardShell } from "./components/layout/DashboardShell";
import { TopBar } from "./components/layout/TopBar";
import { useConfigStore } from "./stores/config";
import { useSimulationStore } from "./stores/simulation";
import type { ServerStatus } from "./types";

type ConnectionState = "connecting" | "connected" | "error";

export default function App() {
  const [connectionState, setConnectionState] =
    useState<ConnectionState>("connecting");
  const [serverStatus, setServerStatus] = useState<ServerStatus | null>(null);
  const [errorMessage, setErrorMessage] = useState<string>("");
  const [coPilotOpen, setCoPilotOpen] = useState(false);

  const simStatus = useSimulationStore((s) => s.status);
  const currentStep = useSimulationStore((s) => s.currentStep);
  const currentTime = useSimulationStore((s) => s.currentTime);
  const setPresets = useConfigStore((s) => s.setPresets);

  // Listen for server status from Electron main process (IPC bridge)
  useEffect(() => {
    if (window.dpf) {
      const unsubStatus = window.dpf.onServerStatus((status) => {
        setServerStatus(status);
        if (status.ready) {
          setConnectionState("connected");
        } else if (status.error) {
          setConnectionState("error");
          setErrorMessage(status.error);
        }
      });

      return () => {
        unsubStatus();
      };
    }

    // Fallback: direct HTTP polling (for development without Electron)
    const pollInterval = setInterval(async () => {
      try {
        const health = await fetchHealth();
        setServerStatus({
          ready: true,
          port: 8765,
          backends: health.backends,
        });
        setConnectionState("connected");
        clearInterval(pollInterval);
      } catch {
        // Still connecting...
      }
    }, 1000);

    return () => clearInterval(pollInterval);
  }, []);

  // Load presets once connected
  useEffect(() => {
    if (connectionState === "connected") {
      fetchPresets()
        .then((presets) => setPresets(presets))
        .catch((err) => console.warn("Failed to load presets:", err));
    }
  }, [connectionState, setPresets]);

  return (
    <div className="flex h-screen flex-col bg-dpf-bg text-gray-200">
      {/* Top Bar */}
      <TopBar
        backends={
          serverStatus?.backends ?? {
            python: false,
            athena: false,
            athenak: false,
            metal: false,
          }
        }
        simStatus={simStatus}
        step={currentStep}
        time={currentTime}
        coPilotOpen={coPilotOpen}
        onToggleCoPilot={() => setCoPilotOpen((prev) => !prev)}
      />

      {/* Main Content Area */}
      <main className="flex flex-1 overflow-hidden">
        {connectionState === "connecting" && (
          <div className="flex flex-1 items-center justify-center">
            <ConnectingScreen />
          </div>
        )}
        {connectionState === "connected" && serverStatus && (
          <DashboardShell coPilotOpen={coPilotOpen} />
        )}
        {connectionState === "error" && (
          <div className="flex flex-1 items-center justify-center">
            <ErrorScreen message={errorMessage} />
          </div>
        )}
      </main>
    </div>
  );
}

// ── Sub-components ──────────────────────────────────────────────

function ConnectingScreen() {
  return (
    <div className="flex flex-col items-center gap-4 animate-fade-in">
      <div className="h-12 w-12 animate-spin rounded-full border-2 border-dpf-border border-t-accent-cyan" />
      <div className="text-center">
        <p className="font-mono text-sm text-accent-cyan">
          Connecting to DPF Engine...
        </p>
        <p className="mt-1 text-xs text-gray-600">
          Starting Python backend server
        </p>
      </div>
    </div>
  );
}

function ErrorScreen({ message }: { message: string }) {
  return (
    <div className="flex max-w-lg flex-col items-center gap-4 p-8 animate-fade-in">
      <div className="flex h-12 w-12 items-center justify-center rounded-full bg-accent-crimson-dim">
        <span className="text-xl text-accent-crimson">!</span>
      </div>
      <div className="text-center">
        <p className="font-mono text-sm text-accent-crimson">
          Failed to connect to DPF Engine
        </p>
        <p className="mt-2 max-w-md text-xs text-gray-500">{message}</p>
        <p className="mt-4 text-xs text-gray-600">
          Ensure the DPF CLI is installed:{" "}
          <code className="text-accent-cyan">pip install dpf-unified</code>
        </p>
      </div>
    </div>
  );
}
