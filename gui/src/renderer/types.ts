/**
 * TypeScript type declarations for the Electron preload bridge.
 */

export interface ServerStatus {
  ready: boolean;
  port: number;
  backends: {
    python: boolean;
    athena: boolean;
    athenak: boolean;
    metal: boolean;
  };
  error?: string;
}

export interface DPFBridge {
  getServerPort: () => Promise<number>;
  onServerStatus: (callback: (status: ServerStatus) => void) => () => void;
  onServerLog: (callback: (line: string) => void) => () => void;
}

declare global {
  interface Window {
    dpf?: DPFBridge;
  }
}
